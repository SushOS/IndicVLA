#!/usr/bin/env python3
"""Direction minimal-pair test: does swapping right<->left in the caption flip the motion?

WHY THIS EXISTS
---------------
The worst divergence in the 25-clip side-by-side was a direction word:
"individual takes steps to their right" -> en_disp=1.66m, hi_disp=0.06m. This test turns
that one anecdote into a measured rate.

DESIGN
------
Word forms below were taken from an actual scan of the test split's Hindi captions (not
guessed): 7 right-forms, 6 left-forms, filtered to exclude false positives that share the
"बा" prefix without meaning "left" (बाहर=outside, बार=time/turn, बाद=after, बारी=turn,
बाधा=obstacle, बाहों=arms, बात=matter, बातचीत=conversation).

For every caption containing exactly one direction word, build the counterfactual by
substituting the matching opposite-side form, then generate BOTH under identical seed pose
and identical noise (only the direction word differs).

CALIBRATION ARM (why it's here)
--------------------------------
Left/right confusion is a documented weakness of text-to-motion models generally, not
necessarily something specific to this adapter. An ENGLISH-swap arm (right<->left,
frozen t5) runs alongside the Hindi-swap arm so a Hindi-only deficit isn't wrongly blamed
on the adapter if the base OMG-100M checkpoint itself doesn't reliably encode direction.

METRIC
------
Motion is canonicalised on the anchor frame with the anchor's HEADING removed (yaw-only),
so root_pos_local's lateral axis (index 1, the "y" of root_pos_local) is the signed
left/right offset in the robot's own reference frame at generation time. A caption that
correctly encodes "right" should push this axis in a consistent sign versus its "left"
counterfactual.

    flip_correct = sign(lateral_swapped) != sign(lateral_original)   (direction acted on)
    also reported: raw magnitude of the lateral OFFSET DIFFERENCE, for clips where both
    happen to be small (near-zero motion is not evidence of "correctly encoded direction")
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/workspace")
from maila_train import WindowDataset, build  # noqa: E402

RIGHT = ["दाहिनी", "दाहिने", "दाईं", "दाएं", "दाहिना", "दायीं", "दाएँ"]
LEFT = ["बायीं", "बाएं", "बाईं", "बाएँ", "बायां", "बायाँ"]
# right/left forms are grammatically paired by ending pattern (gender/case), pair by
# matching suffix so the swap stays as grammatical as machine-translated text allows:
PAIR = {
    "दाहिनी": "बायीं", "दाहिने": "बाएं", "दाईं": "बाईं", "दाएं": "बाएं",
    "दाहिना": "बायां", "दायीं": "बायीं", "दाएँ": "बाएँ",
}
PAIR.update({v: k for k, v in PAIR.items()})

EN_RIGHT = [" right"]
EN_LEFT = [" left"]


def find_swaps_hi(cap: str) -> str | None:
    hits = [w for w in PAIR if w in cap]
    if len(hits) != 1:
        return None
    w = hits[0]
    return cap.replace(w, PAIR[w], 1)


def find_swaps_en(cap: str) -> str | None:
    lc = f" {cap.lower()} "
    has_r, has_l = " right " in lc, " left " in lc
    if has_r and not has_l:
        return re.sub(r"\bright\b", "left", cap, count=1, flags=re.IGNORECASE)
    if has_l and not has_r:
        return re.sub(r"\bleft\b", "right", cap, count=1, flags=re.IGNORECASE)
    return None


@torch.no_grad()
def lateral_offset(model, ds, i, cap, seed, dev):
    w = torch.from_numpy(ds.x[i:i + 1]).to(dev)
    ap_ = np.asarray(ds.man.iloc[i]["anchor_root_pos"], np.float32).reshape(1, 3)
    aq_ = np.asarray(ds.man.iloc[i]["anchor_root_quat"], np.float32).reshape(1, 4)
    b = {
        "history_features": w[:, :ds.L], "prev_state_features": w[:, :ds.L],
        "motion_features": w[:, ds.L:],
        "canon_root_pos": torch.from_numpy(ap_).to(dev),
        "canon_root_quat": torch.from_numpy(aq_).to(dev),
        "mask": {"valid": torch.ones(1, ds.H, dtype=torch.bool, device=dev)},
        "caption": [cap], "has_text": torch.ones(1, dtype=torch.bool, device=dev),
        "fps": torch.tensor([30.0], device=dev),
    }
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    g = model.generate(b, num_frames=ds.H, cfg_scale=2.5)
    q = g["qpos_36"][0].float().cpu().numpy()

    rep = model.representation
    qpos_t = torch.from_numpy(q).unsqueeze(0).to(dev)
    fk = rep.kinematics.forward_kinematics(qpos_t)
    comps = rep.codec.canonicalize(
        qpos_36=qpos_t, body_pos_w=fk["body_pos_w"], body_quat_w=fk["body_quat_w"],
        anchor_root_pos=qpos_t[:, 0, 0:3], anchor_root_quat=qpos_t[:, 0, 3:7],
        fps=torch.tensor([30.0], device=dev))
    lateral = float(comps.root_pos_local[0, -1, 1])   # end-of-window lateral offset
    disp = float(np.linalg.norm(q[-1, :2] - q[0, :2]))
    return lateral, disp


def run_arm(label, model, ds, pairs, seed, dev, out):
    print(f"\n=== {label} ===")
    rows = []
    for i, cap, cap_swap in pairs:
        lat0, d0 = lateral_offset(model, ds, i, cap, seed, dev)
        lat1, d1 = lateral_offset(model, ds, i, cap_swap, seed, dev)
        flipped = (lat0 * lat1 < 0) and (abs(lat0) > 0.01 or abs(lat1) > 0.01)
        rows.append({"i": int(i), "cap": cap, "cap_swap": cap_swap,
                     "lateral_orig": lat0, "lateral_swap": lat1,
                     "disp_orig": d0, "disp_swap": d1, "flipped": bool(flipped)})
        print(f"  lat={lat0:+.3f}->{lat1:+.3f}  {'FLIP' if flipped else '    '}  {cap[:44]}")
    rate = sum(r["flipped"] for r in rows) / max(len(rows), 1)
    print(f"  {label}: {sum(r['flipped'] for r in rows)}/{len(rows)} flipped ({rate*100:.1f}%)")
    (out / f"{label}.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    return rate


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/workspace/data/omg125")
    ap.add_argument("--split", default="test")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--omg", default="/workspace/models/omg/checkpoints/updated/100m/sstep=170000.ckpt")
    ap.add_argument("--t5", default="/workspace/models/t5-base-local")
    ap.add_argument("--muril", default="/workspace/models/muril-base-cased")
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/workspace/out/direction_pairs")
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    ds = WindowDataset(a.data, a.split, "hi")
    rng = np.random.default_rng(a.seed)
    order = rng.permutation(len(ds))

    hi_pairs, en_pairs = [], []
    for i in order:
        i = int(i)
        if len(hi_pairs) < a.n:
            sw = find_swaps_hi(ds.hi[i])
            if sw:
                hi_pairs.append((i, ds.hi[i], sw))
        if len(en_pairs) < a.n:
            sw = find_swaps_en(ds.en[i])
            if sw:
                en_pairs.append((i, ds.en[i], sw))
        if len(hi_pairs) >= a.n and len(en_pairs) >= a.n:
            break
    print(f"built {len(hi_pairs)} Hindi pairs, {len(en_pairs)} English pairs")

    m_hi = build("100m", a.omg, a.t5, a.muril, dev, True, True, 0.0)
    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    m_hi.text_encoder.adapter.load_state_dict(st["adapter"])
    m_hi.text_encoder.adapter.eval()
    rate_hi = run_arm("hindi_swap", m_hi, ds, hi_pairs, a.seed, dev, out)
    del m_hi; torch.cuda.empty_cache()

    m_en = build("100m", a.omg, a.t5, a.muril, dev, True, True, 0.0)
    from omg.generation.conditions.t5 import FrozenT5TextEncoder
    m_en.text_encoder._t5 = FrozenT5TextEncoder(model_name=a.t5, max_length=50,
                                                output_dim=768).to(dev).eval()
    m_en.text_encoder.passthrough_t5 = True
    rate_en = run_arm("english_swap_calibration", m_en, ds, en_pairs, a.seed, dev, out)

    print(f"\nSUMMARY: English calibration flip rate = {rate_en*100:.1f}%   "
          f"Hindi flip rate = {rate_hi*100:.1f}%")
    verdict = ("HINDI-SPECIFIC DEFICIT" if rate_hi < rate_en - 0.15
               else "DIRECTION SENSITIVITY IS COMPARABLE (or a base-model limitation)")
    print(f"VERDICT: {verdict}")
    (out / "summary.json").write_text(json.dumps(
        {"rate_english_calibration": rate_en, "rate_hindi": rate_hi, "verdict": verdict},
        indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
