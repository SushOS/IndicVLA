#!/usr/bin/env python3
"""Is the trained adapter READING the Hindi, or has it collapsed to a constant context?

Falling loss cannot answer this (plan section 12: "training loss falls but all prompts
produce similar motion"), and research-log 99 showed a bare sensitivity number is easy to
over-read. So this reports Δ against two anchors on identical windows/noise/timesteps:

  RANDOM adapter   an untrained adapter -- what "not reading the text" looks like here
  ENGLISH t5       the frozen path the DiT was actually trained with -- the ceiling
  TRAINED adapter  the checkpoint under test

    L_correct = diffusion loss with the window's own caption
    L_wrong   = same window, another window's caption
    delta     = L_wrong - L_correct        (higher = the caption matters more)
    rel       = delta / L_correct          (scale-free, comparable across arms)

A trained adapter near RANDOM's rel has collapsed. Near ENGLISH's rel means it is using
the text about as much as the original conditioning did.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "/workspace")
from maila_train import WindowDataset, build, collate, native_loss  # noqa: E402


@torch.no_grad()
def arm(model, ds, idx, caps, batch, seeds, dev):
    tot, n = 0.0, 0
    for bi, s in enumerate(range(0, len(idx), batch)):
        sel = idx[s:s + batch]
        b = collate([ds[int(i)] for i in sel], dev)
        b["caption"] = [caps[int(i)] for i in sel]
        torch.manual_seed(seeds[bi]); torch.cuda.manual_seed_all(seeds[bi])
        l, *_ = native_loss(model, b)
        tot += float(l) * len(sel); n += len(sel)
    return tot / n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--data", default="/workspace/data/omg125")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--omg", default="/workspace/models/omg/checkpoints/updated/100m/sstep=170000.ckpt")
    ap.add_argument("--t5", default="/workspace/models/t5-base-local")
    ap.add_argument("--muril", default="/workspace/models/muril-base-cased")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/workspace/out/text_sensitivity.json")
    a = ap.parse_args()

    dev = torch.device("cuda")
    ds = WindowDataset(a.data, a.split, "hi")
    rng = np.random.default_rng(a.seed)
    idx = rng.choice(len(ds), size=min(a.n, len(ds)), replace=False)
    hi = ds.hi
    en = ds.en
    # swap: each window gets a DIFFERENT window's caption (shift by one over the sample)
    shift = np.roll(idx, 1)
    hi_wrong = {int(i): hi[int(j)] for i, j in zip(idx, shift)}
    en_wrong = {int(i): en[int(j)] for i, j in zip(idx, shift)}
    hi_w = [hi_wrong.get(i, hi[i]) for i in range(len(ds))]
    en_w = [en_wrong.get(i, en[i]) for i in range(len(ds))]

    nb = (len(idx) + a.batch - 1) // a.batch
    seeds = np.random.default_rng(a.seed + 1).integers(0, 2**31, size=nb).tolist()

    rows = []

    def report(label, m, correct, wrong):
        lc = arm(m, ds, idx, correct, a.batch, seeds, dev)
        lw = arm(m, ds, idx, wrong, a.batch, seeds, dev)
        d = lw - lc
        rows.append({"arm": label, "L_correct": lc, "L_wrong": lw,
                     "delta": d, "rel": d / max(lc, 1e-9)})
        print(f"  {label:34s} L_correct={lc:.5f} L_wrong={lw:.5f} "
              f"delta={d:+.5f} rel={d/max(lc,1e-9)*100:+6.2f}%", flush=True)

    # ---- lower anchor: untrained adapter
    m = build("100m", a.omg, a.t5, a.muril, dev, True, True, 0.0)
    report("RANDOM adapter (untrained)", m, hi, hi_w)
    del m; torch.cuda.empty_cache()

    # ---- upper anchor: the frozen English path the DiT was trained with
    m = build("100m", a.omg, a.t5, a.muril, dev, True, True, 0.0)
    m.text_encoder.passthrough_t5 = True
    m.text_encoder._t5 = m.text_encoder._t5 or __import__(
        "omg.generation.conditions.t5", fromlist=["FrozenT5TextEncoder"]
    ).FrozenT5TextEncoder(model_name=a.t5, max_length=50, output_dim=768).to(dev).eval()
    report("ENGLISH t5 (ceiling)", m, en, en_w)
    del m; torch.cuda.empty_cache()

    # ---- the checkpoints under test
    for c in a.ckpts:
        st = torch.load(c, map_location="cpu", weights_only=False)
        m = build("100m", a.omg, a.t5, a.muril, dev, True, True, 0.0)
        m.text_encoder.adapter.load_state_dict(st["adapter"])
        m.text_encoder.adapter.eval()
        report(f"TRAINED {Path(c).stem} (alpha={float(m.text_encoder.adapter.alpha):+.4f})",
               m, hi, hi_w)
        del m; torch.cuda.empty_cache()

    rand = next(r for r in rows if r["arm"].startswith("RANDOM"))
    eng = next(r for r in rows if r["arm"].startswith("ENGLISH"))
    best = max((r for r in rows if r["arm"].startswith("TRAINED")), key=lambda r: r["rel"])
    span = eng["rel"] - rand["rel"]
    frac = (best["rel"] - rand["rel"]) / span if abs(span) > 1e-9 else float("nan")
    print(f"\n  random rel={rand['rel']*100:+.2f}%  english rel={eng['rel']*100:+.2f}%  "
          f"best trained rel={best['rel']*100:+.2f}%")
    print(f"  recovered {frac*100:.1f}% of the random->english sensitivity gap")
    print(f"\n  VERDICT: {'READING THE TEXT' if frac > 0.25 else 'LIKELY COLLAPSED - investigate before spending more steps'}")
    Path(a.out).write_text(json.dumps({"rows": rows, "recovered_frac": frac}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
