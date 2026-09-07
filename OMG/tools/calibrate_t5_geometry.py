#!/usr/bin/env python3
"""Measure t5-base's per-channel output geometry -- the distribution the frozen DiT reads.

WHY THIS STAGE EXISTS (measured 2026-09-06)
-------------------------------------------
OMG's text projection is `nn.Identity()`. There is NO learned layer between the text
encoder and cross-attention, so whatever the adapter emits is fed to the frozen DiT raw.
The DiT was fitted on t5-base contexts, whose valid tokens have L2 norm ~6.76.

MAILA's ResidualAdapter ends in `nn.LayerNorm(768)`. With the default affine (w=1, b=0)
LayerNorm pins every token to unit variance, i.e. norm exactly sqrt(768) = 27.71 -- 4.1x
too long, unreachable by any value of alpha, and true of every MAILA run to date.

Cost of that mismatch, on an untrained adapter over 1,024 val windows:
    out_norm w=1 / b=0    L_correct 0.07454   rel -1.56%   tok_norm 27.713
    out_norm recoloured   L_correct 0.03883   rel +0.04%   tok_norm  7.396
Half the initial loss is scale error, not semantics.

This script measures mean_t5[d] and std_t5[d] over valid (unmasked) tokens so the adapter's
out_norm can be initialised as weight=std, bias=mean -- recolouring LayerNorm's z into t5's
marginal distribution at zero parameter cost.

LEAKAGE: statistics come from TRAIN-split English captions only. They are 1,536 scalars of
first/second-moment summary, never val/test text.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="root holding train/ shards")
    ap.add_argument("--t5", default="/workspace/models/t5-base-local")
    ap.add_argument("--out", default="/workspace/t5_channel_calibration.pt")
    ap.add_argument("--n-captions", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-length", type=int, default=50)
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, "/workspace")
    from maila_train_motionpivot import MultiCaptionWindowDataset
    from omg.generation.conditions.t5 import FrozenT5TextEncoder

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t5 = FrozenT5TextEncoder(model_name=a.t5, max_length=a.max_length,
                             output_dim=a.width).to(dev).eval()

    ds = MultiCaptionWindowDataset(a.data, "train", ["hi"])
    n = min(a.n_captions, len(ds))
    sel = np.random.default_rng(a.seed).choice(len(ds), size=n, replace=False)

    # float64 accumulators: 34k tokens x 768 dims in float32 loses ~3 decimal digits on S2
    S = torch.zeros(a.width, dtype=torch.float64, device=dev)
    S2 = torch.zeros(a.width, dtype=torch.float64, device=dev)
    ntok = 0
    with torch.no_grad():
        for s in range(0, n, a.batch_size):
            caps = [ds.caps["en"][int(i)][0] if ds.caps["en"][int(i)] else ""
                    for i in sel[s:s + a.batch_size]]
            o = t5(caps, device=dev)
            v = o["context"].double()[o["mask"]]        # (valid_tokens, width)
            S += v.sum(0)
            S2 += (v * v).sum(0)
            ntok += int(v.shape[0])

    if ntok < 10 * a.width:
        raise SystemExit(f"only {ntok} valid tokens for {a.width} channels -- raise "
                         f"--n-captions; second moments would be unreliable")

    mu = (S / ntok).float().cpu()
    sd = ((S2 / ntok).float().cpu() - mu ** 2).clamp_min(1e-12).sqrt()
    # a channel with near-zero variance would make the recoloured adapter blind on that
    # dimension; t5-base has none, but assert rather than assume
    if float(sd.min()) < 1e-4:
        raise SystemExit(f"degenerate channel: min std {float(sd.min()):.2e}")

    expected_norm = float((mu ** 2 + sd ** 2).sum().sqrt())
    payload = {
        "mu": mu, "sd": sd, "n_tokens": ntok, "n_captions": n,
        "source": "train-split English captions", "t5": a.t5, "seed": a.seed,
        "expected_token_norm": expected_norm,
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    digest = hashlib.sha256(out.read_bytes()).hexdigest()

    meta = {k: v for k, v in payload.items() if k not in ("mu", "sd")}
    meta.update({"sha256": digest, "mean_abs_mu": float(mu.abs().mean()),
                 "mean_sd": float(sd.mean()), "layernorm_default_norm": a.width ** 0.5})
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"  t5 geometry from {ntok:,} valid tokens / {n:,} train English captions")
    print(f"    mean |mu| {float(mu.abs().mean()):.4f}   mean sd {float(sd.mean()):.4f}")
    print(f"    implied token norm {expected_norm:.3f}  vs  LayerNorm default "
          f"{a.width ** 0.5:.3f}  ({a.width ** 0.5 / expected_norm:.2f}x too long)")
    print(f"  wrote {out}  sha256 {digest[:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
