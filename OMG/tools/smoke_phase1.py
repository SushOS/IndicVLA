#!/usr/bin/env python3
"""MAILA plan section 8, Phase 1 - adapter smoke tests. All six, with pass/fail.

  1 output contract is exactly [B,50,768] + [B,50] bool
  2 a backward pass reaches the adapter and NOTHING else
  3 a tiny 16-32 window subset can be overfit  (the load-bearing one -- see below)
  4 different Hindi prompts, identical history and noise, produce different motion
  5 conditional vs null-context differ, i.e. CFG is actually using the Hindi context
  6 no translator/transliterator package is reachable in the inference path

Test 3 is the one that matters most architecturally. The adapter is
`out_norm(LN(H) + alpha*MLP(LN(H)))` with alpha init 0.1, which biases the output toward
normalised MuRIL. MuRIL space is NOT t5 space, so unlike a same-space residual there is no
safe identity init and alpha MUST grow. If the tiny set cannot be overfit, the residual
framing is underpowered and the identity path should become a full linear map -- do not
fix that by tuning the learning rate.
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

OK, BAD = "PASS", "FAIL"
results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(f"  [{OK if cond else BAD}] {name}" + (f"  {detail}" if detail else ""), flush=True)
    return bool(cond)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/workspace/data/omg125")
    ap.add_argument("--ckpt", default="/workspace/models/omg/checkpoints/updated/100m/sstep=170000.ckpt")
    ap.add_argument("--t5", default="/workspace/models/t5-base-local")
    ap.add_argument("--muril", default="/workspace/models/muril-base-cased")
    ap.add_argument("--self-qk", default="true")
    ap.add_argument("--cross-qk", default="true")
    ap.add_argument("--n-tiny", type=int, default=32)
    ap.add_argument("--overfit-steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="/workspace/out/smoke_phase1.json")
    a = ap.parse_args()

    dev = torch.device("cuda")
    torch.manual_seed(0)
    print(f"contract under test: self_qk={a.self_qk} cross_qk={a.cross_qk}\n")

    ds = WindowDataset(a.data, "train", "hi", limit=4096)
    model = build("100m", a.ckpt, a.t5, a.muril, dev,
                  a.self_qk == "true", a.cross_qk == "true", 0.0)
    enc = model.text_encoder

    # ---------------------------------------------------------------- 1 contract
    caps = [ds.hi[i] for i in range(4)]
    out = enc(caps, device=dev)
    check("1 context shape [B,50,768]", tuple(out["context"].shape) == (4, 50, 768),
          str(tuple(out["context"].shape)))
    check("1 mask shape [B,50] bool",
          tuple(out["mask"].shape) == (4, 50) and out["mask"].dtype == torch.bool,
          f"{tuple(out['mask'].shape)} {out['mask'].dtype}")
    check("1 mask is a contiguous prefix",
          all(bool(m.tolist() == sorted(m.tolist(), reverse=True)) for m in out["mask"]))
    check("1 context finite", bool(torch.isfinite(out["context"]).all()))

    # ---------------------------------------------------------------- 2 gradient routing
    idx = list(range(a.n_tiny))
    batch = collate([ds[i] for i in idx[:8]], dev)
    model.zero_grad(set_to_none=True)
    loss, *_ = native_loss(model, batch)
    loss.backward()
    got = {n for n, p in model.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0}
    non_adapter = [n for n in got if not n.startswith("text_encoder.adapter")]
    check("2 gradient reaches the adapter",
          any(n.startswith("text_encoder.adapter") for n in got), f"{len(got)} tensors with grad")
    check("2 no gradient anywhere else", not non_adapter, str(non_adapter[:3]))
    model.zero_grad(set_to_none=True)

    # ---------------------------------------------------------------- 3 tiny overfit
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr)
    tiny = collate([ds[i] for i in idx], dev)
    first = last = None
    a0 = float(enc.adapter.alpha)
    for s in range(a.overfit_steps):
        opt.zero_grad(set_to_none=True)
        torch.manual_seed(1234)                    # fix noise/timestep: isolate fitting
        l, *_ = native_loss(model, tiny)
        l.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if s == 0:
            first = float(l)
        last = float(l)
        if s % 100 == 0:
            print(f"      overfit step {s:4d} loss={float(l):.5f} alpha={float(enc.adapter.alpha):+.4f}",
                  flush=True)
    drop = (first - last) / max(first, 1e-9)
    check("3 tiny set overfits (loss drops >40%)", drop > 0.40,
          f"{first:.5f} -> {last:.5f}  ({drop*100:.1f}% drop)")
    check("3 alpha moved off its init", abs(float(enc.adapter.alpha) - a0) > 1e-3,
          f"{a0:.4f} -> {float(enc.adapter.alpha):+.4f}")

    # ---------------------------------------------------------------- 4 prompt sensitivity
    enc.adapter.eval()
    with torch.no_grad():
        seed_w = np.stack([ds[i]["history"] for i in range(2)])
        fut = np.stack([ds[i]["future"] for i in range(2)])
        distinct = [c for c in dict.fromkeys(ds.hi)][:2]
        ctxs = []
        for cap in distinct:
            b = {
                "motion_features": torch.from_numpy(fut).to(dev),
                "history_features": torch.from_numpy(seed_w).to(dev),
                "mask": {"valid": torch.ones(2, fut.shape[1], dtype=torch.bool, device=dev)},
                "caption": [cap, cap],
                "has_text": torch.ones(2, dtype=torch.bool, device=dev),
                "fps": torch.full((2,), 30.0, device=dev),
            }
            torch.manual_seed(7)
            _, diff, *_ = native_loss(model, b)
            ctxs.append(diff["pred_x0"].float().cpu().numpy())
    d_prompt = float(np.abs(ctxs[0] - ctxs[1]).mean())
    check("4 different prompts -> different motion", d_prompt > 1e-4,
          f"mean|dx0| = {d_prompt:.3e}")

    # ---------------------------------------------------------------- 5 CFG uses the context
    with torch.no_grad():
        c_cond = enc([distinct[0]], device=dev)
        c_null = enc([distinct[0]], force_null_text=True, device=dev)
    d_null = float((c_cond["context"] - c_null["context"]).abs().mean())
    check("5 conditional context differs from null", d_null > 1e-4, f"mean|d| = {d_null:.3e}")
    check("5 null context matches cached t5 null exactly",
          bool(torch.equal(c_null["context"][0], enc.null_context[0])))

    # ---------------------------------------------------------------- 6 no translator
    banned = ("indictrans", "IndicTransToolkit", "googletrans", "translate",
              "transformers.models.marian", "ai4bharat", "nllb", "sacremoses")
    loaded = [m for m in sys.modules if any(b.lower() in m.lower() for b in banned)]
    check("6 no translator package imported", not loaded, str(loaded[:4]))
    check("6 encoder holds no t5 at inference", enc._t5 is None,
          "passthrough_t5 is off")

    n_pass = sum(1 for _, ok, _ in results if ok)
    print(f"\n  {n_pass}/{len(results)} checks passed")
    Path(a.out).write_text(json.dumps(
        [{"check": n, "pass": ok, "detail": d} for n, ok, d in results], indent=2))
    print(f"  EXIT: {'PHASE1_PASS' if n_pass == len(results) else 'PHASE1_FAIL'}")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
