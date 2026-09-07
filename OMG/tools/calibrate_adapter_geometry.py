#!/usr/bin/env python3
"""Measure BOTH ends of the adapter's frozen interface.

target (t5)  mu, sd        per-channel statistics of t5-base's context output. This is the
                           distribution the frozen DiT was fitted on; OMG's `proj` is
                           nn.Identity(), so the adapter's output must land here.
source (MuRIL) src_mu, src_sd
                           per-channel statistics of in_norm(MuRIL(caption)) -- the adapter's
                           own input -- measured ACROSS the corpus.

The source half is what nn.LayerNorm cannot see. LayerNorm standardises within a token, so a
direction shared by every token survives it. MuRIL's is enormous: ||mu_corpus|| = 26.91
against a token norm of 27.71, leaving ~0.42 units of caption-specific content under ~7.3
units of constant, and two different captions arriving at cosine 0.9966.

Subtracting src_mu and dividing by src_sd removes it; recolouring by (mu, sd) then lands the
result in t5's distribution. Measured effect at init: btwn/mu 0.0581 -> 0.8042 (t5: 0.7201),
cos(i,j) 0.9966 -> 0.6080 (t5: 0.6563).

Supersedes calibrate_t5_geometry.py, whose output has no source half.
LEAKAGE: every statistic comes from the TRAIN split.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


class Moments:
    """float64 streaming per-channel mean/std. float32 loses ~3 digits on the 2nd moment."""

    def __init__(self, width: int, dev) -> None:
        self.s = torch.zeros(width, dtype=torch.float64, device=dev)
        self.s2 = torch.zeros(width, dtype=torch.float64, device=dev)
        self.n = 0

    def add(self, v: torch.Tensor) -> None:      # v: (tokens, width)
        v = v.double()
        self.s += v.sum(0)
        self.s2 += (v * v).sum(0)
        self.n += int(v.shape[0])

    def finish(self) -> tuple[torch.Tensor, torch.Tensor]:
        mu = (self.s / self.n).float().cpu()
        sd = ((self.s2 / self.n).float().cpu() - mu ** 2).clamp_min(1e-12).sqrt()
        return mu, sd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--t5", default="/workspace/models/t5-base-local")
    ap.add_argument("--muril", default="/workspace/models/muril-base-cased")
    ap.add_argument("--langs", default="hi", help="comma-separated source languages")
    ap.add_argument("--out", default="/workspace/adapter_calibration.pt")
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
    from transformers import AutoModel, AutoTokenizer

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    langs = [x.strip() for x in a.langs.split(",") if x.strip()]
    ds = MultiCaptionWindowDataset(a.data, "train", langs)
    n = min(a.n_captions, len(ds))
    sel = np.random.default_rng(a.seed).choice(len(ds), size=n, replace=False)

    def caps_for(lg):
        return [ds.caps[lg][int(i)][0] if ds.caps[lg][int(i)] else "" for i in sel]

    # ---- target: t5-base on ENGLISH, the DiT's own conditioning distribution ----
    t5 = FrozenT5TextEncoder(model_name=a.t5, max_length=a.max_length,
                             output_dim=a.width).to(dev).eval()
    tgt = Moments(a.width, dev)
    en = caps_for("en")
    with torch.no_grad():
        for s in range(0, n, a.batch_size):
            o = t5(en[s:s + a.batch_size], device=dev)
            tgt.add(o["context"][o["mask"]])
    mu, sd = tgt.finish()
    del t5
    torch.cuda.empty_cache()

    # ---- source: in_norm(MuRIL(caption)) over the TRAINING languages ----
    tok = AutoTokenizer.from_pretrained(a.muril)
    muril = AutoModel.from_pretrained(a.muril).to(dev).eval()
    src = Moments(a.width, dev)
    with torch.no_grad():
        for lg in langs:
            caps = caps_for(lg)
            for s in range(0, n, a.batch_size):
                t = tok(caps[s:s + a.batch_size], max_length=a.max_length,
                        padding="max_length", truncation=True, return_tensors="pt")
                t = {k: v.to(dev) for k, v in t.items()}
                h = muril(input_ids=t["input_ids"],
                          attention_mask=t["attention_mask"]).last_hidden_state
                # in_norm at init == plain LayerNorm (weight 1, bias 0)
                src.add(F.layer_norm(h, (a.width,))[t["attention_mask"].bool()])
    src_mu, src_sd = src.finish()

    for nm, v in (("sd", sd), ("src_sd", src_sd)):
        if float(v.min()) < 1e-4:
            raise SystemExit(f"degenerate channel in {nm}: min {float(v.min()):.2e}")

    payload = {"mu": mu, "sd": sd, "src_mu": src_mu, "src_sd": src_sd,
               "n_target_tokens": tgt.n, "n_source_tokens": src.n,
               "n_captions": n, "langs": langs, "seed": a.seed,
               "t5": a.t5, "muril": a.muril, "source": "train split only"}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    meta = {k: v for k, v in payload.items() if not torch.is_tensor(v)}
    meta.update({"sha256": digest,
                 "target_token_norm": float((mu ** 2 + sd ** 2).sum().sqrt()),
                 "source_common_norm": float(src_mu.norm()),
                 "layernorm_default_norm": a.width ** 0.5})
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"  target t5   : {tgt.n:,} tokens | implied norm "
          f"{meta['target_token_norm']:.3f} (LayerNorm default {a.width ** 0.5:.3f})")
    print(f"  source MuRIL: {src.n:,} tokens over {langs} | ||common component|| "
          f"{meta['source_common_norm']:.3f}  <- what LayerNorm cannot remove")
    print(f"  wrote {out}  sha256 {digest[:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
