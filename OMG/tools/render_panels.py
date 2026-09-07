#!/usr/bin/env python3
"""N-panel side-by-side motion comparison for Run 2.

Generalises tools/render_side_by_side.py (which was hard-wired to 2 panels, English left /
Hindi right) to an arbitrary column set, because after the English ablation a 2-panel
English-vs-Hindi figure is no longer the right comparison.

ARMS
  gt        the clip's OWN future window, decoded -- ground truth, no generation
  omg_en    stock OMG: frozen t5-base + the English caption (what OMG ships with)
  omg_hi    stock OMG: frozen t5-base + the HINDI caption. THE REAL BASELINE, and the
            one nobody measured until the end of Run 2: it scores rel +1.80% against a
            text-blind floor of 0.00%, i.e. it is very nearly deaf to the caption.
  maila_hi  frozen MuRIL + trained adapter (hi calibration) + Hindi caption
  maila_en  frozen MuRIL + trained adapter (en calibration) + English caption

WHY GROUND TRUTH BELONGS IN EVERY FIGURE
  A 2-panel en|hi comparison cannot be read: if the panels differ you cannot tell which one
  is wrong, and if they agree you have learned nothing. The gt column makes both decidable.

CONTROLS
  Every arm generates from the SAME clip, the SAME history window, the SAME anchor root, and
  the SAME replayed noise seed. The only variable is which text drove the frozen DiT.

  Captions are composited with PIL, never MuJoCo's overlay: MuJoCo's built-in font is
  ASCII-only and renders Devanagari as boxes (render_side_by_side.py:13).

Each clip also gets root-XY displacement per arm in the manifest, so a visual claim like
"the Hindi arm did not move" is backed by a number -- Run 1 found exactly that on
"individual takes steps to their right": en 1.66 m vs hi 0.06 m.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/workspace")

BAND = 104
BG = (18, 18, 20)
COLOURS = {
    "gt": (170, 200, 175),
    "omg_en": (200, 200, 205),
    "omg_hi": (225, 150, 140),
    "maila_hi": (255, 214, 140),
    "maila_en": (150, 195, 240),
}
TITLES = {
    "gt": "ground truth",
    "omg_en": "stock OMG + English",
    "omg_hi": "stock OMG + Hindi (baseline)",
    "maila_hi": "MAILA + Hindi",
    "maila_en": "MAILA + English",
}
LANG_OF = {"omg_en": "en", "omg_hi": "hi", "maila_hi": "hi", "maila_en": "en", "gt": "en"}


def draw_header(total_w, cols, fonts):
    """One caption block per column: arm title above, the caption itself below."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (total_w, BAND), BG)
    d = ImageDraw.Draw(img)
    cw = total_w // len(cols)
    for k, (arm, caption) in enumerate(cols):
        x0 = k * cw
        colour = COLOURS.get(arm, (220, 220, 220))
        d.text((x0 + 14, 8), TITLES.get(arm, arm), font=fonts["label"], fill=colour)
        font = fonts["hi"] if LANG_OF.get(arm) == "hi" else fonts["en"]
        line, lines = "", []
        for word in str(caption).split():
            trial = (line + " " + word).strip()
            if d.textlength(trial, font=font) > cw - 28 and line:
                lines.append(line)
                line = word
            else:
                line = trial
            if len(lines) == 2:
                break
        if line and len(lines) < 2:
            lines.append(line)
        for j, ln in enumerate(lines[:2]):
            d.text((x0 + 14, 40 + j * 30), ln, font=font, fill=colour)
    return np.array(img)


def batch_for(ds, i, dev):
    """generate() needs canon_root_* to de-canonicalise back to world frame.

    The windows are already canonicalised, so these cannot be recovered from the features --
    convert_g1_npz_to_omg125.py persists them per window as anchor_root_pos/quat.
    """
    w = torch.from_numpy(ds.x[i:i + 1]).to(dev)
    ap = np.asarray(ds.man.iloc[i]["anchor_root_pos"], dtype=np.float32).reshape(1, 3)
    aq = np.asarray(ds.man.iloc[i]["anchor_root_quat"], dtype=np.float32).reshape(1, 4)
    return {
        "history_features": w[:, :ds.L], "prev_state_features": w[:, :ds.L],
        "motion_features": w[:, ds.L:],
        "canon_root_pos": torch.from_numpy(ap).to(dev),
        "canon_root_quat": torch.from_numpy(aq).to(dev),
        "mask": {"valid": torch.ones(1, ds.H, dtype=torch.bool, device=dev)},
        "has_text": torch.ones(1, dtype=torch.bool, device=dev),
        "fps": torch.tensor([30.0], device=dev),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/workspace/data/amass_g1_omg125")
    ap.add_argument("--split", default="test")
    ap.add_argument("--arms", default="gt,omg_hi,maila_hi",
                    help="comma-separated columns, left to right")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--indices", default=None, help="explicit comma-separated clip indices")
    ap.add_argument("--num-frames", type=int, default=60)
    ap.add_argument("--cfg-scale", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--width", type=int, default=560)
    ap.add_argument("--height", type=int, default=460)
    ap.add_argument("--models", default="/workspace/models")
    ap.add_argument("--ckpt-hi", default=None)
    ap.add_argument("--ckpt-en", default=None)
    ap.add_argument("--calib-hi", default="/workspace/adapter_calibration.pt")
    ap.add_argument("--calib-en", default="/workspace/adapter_calibration_en.pt")
    ap.add_argument("--out", default="/workspace/panels")
    a = ap.parse_args()

    import imageio.v2 as imageio
    from PIL import ImageFont
    from maila_train import build
    from maila_train_motionpivot import MultiCaptionWindowDataset
    from omg.generation.conditions.t5 import FrozenT5TextEncoder
    from omg.render.mujoco import render_qpos_video

    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    bad = [x for x in arms if x not in TITLES]
    if bad:
        raise SystemExit(f"unknown arms {bad}; choose from {sorted(TITLES)}")
    if "gt" in arms and a.num_frames > 60:
        raise SystemExit("ground truth is only H=60 frames; drop gt or set --num-frames 60")

    dev = torch.device("cuda")
    ds = MultiCaptionWindowDataset(a.data, a.split, ["hi"])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    model = build("100m", f"{a.models}/omg/checkpoints/updated/100m/sstep=170000.ckpt",
                  f"{a.models}/t5-base-local", f"{a.models}/muril-base-cased",
                  dev, True, True, 0.0)
    rep = model.representation
    base_encoder = model.text_encoder

    encoders = {}
    if "omg_en" in arms or "omg_hi" in arms:
        encoders["t5"] = FrozenT5TextEncoder(model_name=f"{a.models}/t5-base-local",
                                             max_length=50, output_dim=768).to(dev).eval()
    from maila_encoder import MurilAdapterEncoder

    def adapter_encoder(calib, ckpt):
        enc = MurilAdapterEncoder(muril_name=f"{a.models}/muril-base-cased",
                                  t5_model_name=f"{a.models}/t5-base-local",
                                  t5_calibration=calib, norm_mode="corpus").to(dev).eval()
        st = torch.load(ckpt, map_location="cpu", weights_only=False)["adapter"]
        enc.adapter.load_state_dict(st)
        enc.adapter.eval()
        return enc

    if "maila_hi" in arms:
        if not a.ckpt_hi:
            raise SystemExit("--ckpt-hi is required for the maila_hi arm")
        encoders["maila_hi"] = adapter_encoder(a.calib_hi, a.ckpt_hi)
    if "maila_en" in arms:
        if not a.ckpt_en:
            raise SystemExit("--ckpt-en is required for the maila_en arm")
        encoders["maila_en"] = adapter_encoder(a.calib_en, a.ckpt_en)

    # clip choice: distinct Hindi captions, so the gallery is not the same prompt repeated
    if a.indices:
        picks = [int(x) for x in a.indices.split(",")]
    else:
        rng = np.random.default_rng(a.seed)
        seen, picks = set(), []
        for i in rng.permutation(len(ds)):
            i = int(i)
            hi = ds.caps["hi"][i][0] if ds.caps["hi"][i] else ""
            en = ds.caps["en"][i][0] if ds.caps["en"][i] else ""
            if not hi or not en or hi in seen:
                continue
            seen.add(hi)
            picks.append(i)
            if len(picks) >= a.n:
                break
    print(f"  {len(picks)} clips | arms {arms} | {a.num_frames} frames | cfg {a.cfg_scale}")

    fonts = {"hi": ImageFont.truetype("/workspace/fonts/dev.ttf", 24),
             "en": ImageFont.truetype("/workspace/fonts/lat.ttf", 24),
             "label": ImageFont.truetype("/workspace/fonts/lat.ttf", 19)}

    @torch.no_grad()
    def qpos_for(arm, i):
        if arm == "gt":
            w = torch.from_numpy(ds.x[i:i + 1, ds.L:]).to(dev)
        # Shards store RAW features: convert_g1_npz_to_omg125.encode_windows() ends with
        # `rep.codec.assemble_features(comps)`, which does NOT normalise. But rep.decode()
        # begins with denormalize_features(), so decoding a shard directly applies a spurious
        # SECOND denormalisation and compresses the motion to a near-frozen pose. That is what
        # made the ground-truth panel look stuck. normalize -> decode cancels exactly, and is
        # equivalent to the codec.split_features() inverse the converter's own
        # verify_roundtrip() uses. Training is unaffected: motion_generator._target_sequence
        # calls representation.encode(), which normalises internally.
            d = rep.decode(rep.normalize_features(w))
            ap_ = torch.from_numpy(
                np.asarray(ds.man.iloc[i]["anchor_root_pos"], np.float32).reshape(1, 3)).to(dev)
            aq_ = torch.from_numpy(
                np.asarray(ds.man.iloc[i]["anchor_root_quat"], np.float32).reshape(1, 4)).to(dev)
            return rep.compose_qpos_36(d, ap_, aq_)[0].float().cpu().numpy()
        lang = LANG_OF[arm]
        caps = ds.caps[lang][i]
        model.text_encoder = encoders["t5"] if arm.startswith("omg_") else encoders[arm]
        b = batch_for(ds, i, dev)
        b["caption"] = [caps[0] if caps else ""]
        torch.manual_seed(a.seed)                      # identical noise for every arm
        torch.cuda.manual_seed_all(a.seed)
        g = model.generate(b, num_frames=a.num_frames, cfg_scale=a.cfg_scale)
        model.text_encoder = base_encoder
        return g["qpos_36"][0].float().cpu().numpy()

    manifest = []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for k, i in enumerate(picks):
            frames, disp = {}, {}
            for arm in arms:
                q = qpos_for(arm, i)
                p = tmp / f"{k:02d}_{arm}.mp4"
                render_qpos_video(q, str(p), fps=30, width=a.width, height=a.height,
                                  title="", overlay_lines=None)
                frames[arm] = imageio.mimread(str(p), memtest=False)
                disp[arm] = float(np.linalg.norm(q[-1, :2] - q[0, :2]))
            n = min(len(frames[x]) for x in arms)
            cols = [(arm, ds.caps[LANG_OF[arm]][i][0] if ds.caps[LANG_OF[arm]][i] else "")
                    for arm in arms]
            hdr = draw_header(a.width * len(arms), cols, fonts)
            name = f"panel_{k:02d}_clip{i}.mp4"
            with imageio.get_writer(str(out / name), fps=30, quality=8) as wtr:
                for t in range(n):
                    row = np.concatenate([frames[arm][t][:, :, :3] for arm in arms], axis=1)
                    wtr.append_data(np.concatenate([hdr, row], axis=0))
            row = {"file": name, "clip": i, "arms": arms,
                   "caption_en": ds.caps["en"][i][0] if ds.caps["en"][i] else "",
                   "caption_hi": ds.caps["hi"][i][0] if ds.caps["hi"][i] else "",
                   "xy_disp_m": {k2: round(v, 4) for k2, v in disp.items()}}
            manifest.append(row)
            print(f"  [{k + 1}/{len(picks)}] {name}  disp " +
                  "  ".join(f"{x}={disp[x]:.2f}m" for x in arms), flush=True)

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                                       encoding="utf-8")
    print(f"\nwrote {len(manifest)} panels to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
