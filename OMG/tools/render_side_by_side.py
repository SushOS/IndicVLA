#!/usr/bin/env python3
"""Side-by-side English vs Hindi motion, 25 prompts, captions in their native scripts.

  LEFT  = frozen t5-base + English caption   (the path OMG was trained with)
  RIGHT = frozen MuRIL + trained adapter + the SAME clip's Hindi caption

Controlled: identical history seed and identical generation noise on both sides, so the
only difference is which text drove the frozen DiT.

WHY THE CAPTIONS ARE COMPOSITED WITH PIL RATHER THAN MuJoCo's OVERLAY
---------------------------------------------------------------------
MuJoCo's built-in overlay font is ASCII-only. Passing Devanagari through
`render_qpos_video(overlay_lines=...)` renders boxes, not text. So both halves are
rendered WITHOUT overlay and the captions are drawn afterwards with PIL using
NotoSansDevanagari for Hindi and NotoSans for English.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/workspace")
from maila_train import WindowDataset, build  # noqa: E402

BAND = 96            # header height in px
BG = (18, 18, 20)
FG_EN = (215, 225, 240)
FG_HI = (255, 214, 140)


def gen_qpos(model, ds, i, seed, num_frames, cfg):
    """Batch for MotionGenerator.generate().

    generate() needs canon_root_pos / canon_root_quat to de-canonicalise its output back
    into world frame. Our windows are ALREADY canonicalised, so those cannot be recomputed
    from features -- they were persisted per window at conversion time as
    anchor_root_pos / anchor_root_quat in manifest.parquet. Omitting them raises
    KeyError: 'canon_root_pos'.
    """
    dev = next(model.parameters()).device
    w = torch.from_numpy(ds.x[i:i + 1]).to(dev)
    ap_ = np.asarray(ds.man.iloc[i]["anchor_root_pos"], dtype=np.float32).reshape(1, 3)
    aq_ = np.asarray(ds.man.iloc[i]["anchor_root_quat"], dtype=np.float32).reshape(1, 4)
    b = {
        "history_features": w[:, :ds.L], "prev_state_features": w[:, :ds.L],
        "motion_features": w[:, ds.L:],
        "canon_root_pos": torch.from_numpy(ap_).to(dev),
        "canon_root_quat": torch.from_numpy(aq_).to(dev),
        "mask": {"valid": torch.ones(1, ds.H, dtype=torch.bool, device=dev)},
        "has_text": torch.ones(1, dtype=torch.bool, device=dev),
        "fps": torch.tensor([30.0], device=dev),
    }
    return b


def draw_header(w, texts_fonts):
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (w, BAND), BG)
    d = ImageDraw.Draw(img)
    half = w // 2
    for k, (txt, font, colour) in enumerate(texts_fonts):
        x0 = k * half
        # wrap to two lines by character budget, measured against the actual font
        line, lines = "", []
        for word in str(txt).split():
            trial = (line + " " + word).strip()
            if d.textlength(trial, font=font) > half - 32 and line:
                lines.append(line); line = word
            else:
                line = trial
            if len(lines) == 2:
                break
        if line and len(lines) < 2:
            lines.append(line)
        for j, ln in enumerate(lines[:2]):
            d.text((x0 + 16, 14 + j * 34), ln, font=font, fill=colour)
    return np.array(img)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="/workspace/runs/maila_v1")
    ap.add_argument("--ckpt", default=None, help="defaults to selection.json's pick")
    ap.add_argument("--data", default="/workspace/data/omg125")
    ap.add_argument("--split", default="val")
    ap.add_argument("--omg", default="/workspace/models/omg/checkpoints/updated/100m/sstep=170000.ckpt")
    ap.add_argument("--t5", default="/workspace/models/t5-base-local")
    ap.add_argument("--muril", default="/workspace/models/muril-base-cased")
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--num-frames", type=int, default=60)
    ap.add_argument("--cfg-scale", type=float, default=2.5)
    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/workspace/runs/maila_v1/side_by_side")
    a = ap.parse_args()

    import imageio.v2 as imageio
    from PIL import ImageFont
    from omg.render.mujoco import render_qpos_video

    dev = torch.device("cuda")
    run = Path(a.run)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tmp = out / "_tmp"; tmp.mkdir(exist_ok=True)

    ck = a.ckpt
    if ck is None:
        sel = json.loads((run / "selection.json").read_text())
        ck = str(run / sel["selected"])
    print(f"checkpoint: {Path(ck).name}")

    ds = WindowDataset(a.data, a.split, "hi")
    rng = np.random.default_rng(a.seed)
    order = rng.permutation(len(ds))
    seen, picks = set(), []
    for i in order:
        c = ds.hi[int(i)]
        if c in seen:
            continue
        seen.add(c); picks.append(int(i))
        if len(picks) >= a.n:
            break
    print(f"{len(picks)} clips with distinct Hindi captions")

    f_hi = ImageFont.truetype("/workspace/fonts/dev.ttf", 26)
    f_en = ImageFont.truetype("/workspace/fonts/lat.ttf", 26)

    # ---- Hindi arm: frozen MuRIL + trained adapter
    m_hi = build("100m", a.omg, a.t5, a.muril, dev, True, True, 0.0)
    st = torch.load(ck, map_location="cpu", weights_only=False)
    m_hi.text_encoder.adapter.load_state_dict(st["adapter"])
    m_hi.text_encoder.adapter.eval()

    # ---- English arm: the frozen t5 path the DiT was trained with
    m_en = build("100m", a.omg, a.t5, a.muril, dev, True, True, 0.0)
    from omg.generation.conditions.t5 import FrozenT5TextEncoder
    m_en.text_encoder._t5 = FrozenT5TextEncoder(
        model_name=a.t5, max_length=50, output_dim=768).to(dev).eval()
    m_en.text_encoder.passthrough_t5 = True

    manifest = []
    for k, i in enumerate(picks):
        sides = {}
        for tag, model, cap in (("en", m_en, ds.en[i]), ("hi", m_hi, ds.hi[i])):
            b = gen_qpos(model, ds, i, a.seed, a.num_frames, a.cfg_scale)
            b["caption"] = [cap]
            with torch.no_grad():
                torch.manual_seed(a.seed)            # identical noise on both sides
                torch.cuda.manual_seed_all(a.seed)
                g = model.generate(b, num_frames=a.num_frames, cfg_scale=a.cfg_scale)
            q = g["qpos_36"][0].float().cpu().numpy()
            p = tmp / f"{k:02d}_{tag}.mp4"
            render_qpos_video(q, str(p), fps=30, width=a.width, height=a.height,
                              title="", overlay_lines=None)   # no MuJoCo text: ASCII-only
            sides[tag] = (imageio.mimread(str(p), memtest=False), q)

        fr_en, q_en = sides["en"]
        fr_hi, q_hi = sides["hi"]
        n = min(len(fr_en), len(fr_hi))
        hdr = draw_header(a.width * 2, [(ds.en[i], f_en, FG_EN), (ds.hi[i], f_hi, FG_HI)])
        name = f"cmp_{k:02d}.mp4"
        with imageio.get_writer(str(out / name), fps=30, quality=8) as wtr:
            for t in range(n):
                row = np.concatenate([fr_en[t][:, :, :3], fr_hi[t][:, :, :3]], axis=1)
                wtr.append_data(np.concatenate([hdr, row], axis=0))
        d_en = float(np.linalg.norm(q_en[-1, :2] - q_en[0, :2]))
        d_hi = float(np.linalg.norm(q_hi[-1, :2] - q_hi[0, :2]))
        manifest.append({"file": name, "en": ds.en[i], "hi": ds.hi[i],
                         "xy_disp_en": d_en, "xy_disp_hi": d_hi})
        print(f"  {name}  en_disp={d_en:.2f} hi_disp={d_hi:.2f}  {ds.hi[i][:38]}", flush=True)

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                                       encoding="utf-8")
    de = np.array([m["xy_disp_en"] for m in manifest])
    dh = np.array([m["xy_disp_hi"] for m in manifest])
    print(f"\n  mean xy_disp  english={de.mean():.2f} m  hindi={dh.mean():.2f} m")
    print(f"  spearman(en,hi) = {np.corrcoef(de.argsort().argsort(), dh.argsort().argsort())[0,1]:+.2f}")
    print(f"  wrote {len(manifest)} side-by-side videos -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
