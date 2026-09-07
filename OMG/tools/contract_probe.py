#!/usr/bin/env python3
"""EXP-F-lite: settle the qk-norm contract with the diffusion loss, on data already on disk.

The official gate (benchmark text, N=1024, GEN R@1) needs all 62 GB of OMG-Data because
the LeRobot loader demands globally contiguous indices. That download is running; this
probe answers the same question in minutes using our converted bones-seed val windows and
t5-base, both already present.

WHAT IT MEASURES
----------------
For each contract, on identical windows / noise / timesteps:

    L_correct = OMG diffusion loss with the clip's OWN English caption
    L_wrong   = same, with another clip's caption
    delta     = L_wrong - L_correct

`delta` is how much the frozen model's motion prediction degrades when the text stops
matching -- i.e. how much it is actually USING the caption. This is the swap test of
research-log 99, but scored with the real training objective instead of xy-displacement,
which 99 concluded was too blunt to select a contract.

The winning contract is the one with the larger delta (reads text) AND the lower
L_correct (predicts the right motion). If those two disagree, report it rather than
picking -- that would mean the probe is not decisive and the official gate must rule.

Noise and timestep are drawn once per batch and REPLAYED for every arm via RNG state, so
arms differ only by qk-norm and caption.
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


def load_windows(root, split, n, seed):
    d = Path(root) / split
    x = np.concatenate([np.load(s)["features"] for s in sorted(glob.glob(str(d / "shard_*.npz")))])
    man = pd.read_parquet(d / "manifest.parquet")
    meta = json.loads((d / "_meta.json").read_text())
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x), size=min(n, len(x)), replace=False)
    return x[idx], man.iloc[idx].reset_index(drop=True), int(meta["L"]), int(meta["H"])


def build(exp, ckpt, t5, self_qk, cross_qk, device):
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from hydra.utils import instantiate
    from omg.core.paths import resolve_repo_path
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(resolve_repo_path("configs/generation")),
                               version_base="1.3"):
        cfg = compose(config_name="train", overrides=[
            f"exp={exp}", "data=omg_data_lerobot", "logger=none", "trainer=1gpu",
            f"denoiser.self_attention_qk_norm={str(self_qk).lower()}",
            f"denoiser.cross_attention_qk_norm={str(cross_qk).lower()}",
            f"model.text_encoder.model_name={t5}",
            "model.text_mask_prob=0.0",
        ])
    m = instantiate(cfg.model)
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    m.load_state_dict(payload.get("state_dict", payload), strict=True)
    m = m.to(device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


@torch.no_grad()
def arm_loss(model, x, caps, L, H, device, batch, seeds):
    """Mean diffusion loss over batches, with per-batch noise/timestep fixed by `seeds`."""
    tot, n = 0.0, 0
    for bi, s in enumerate(range(0, len(x), batch)):
        w = torch.from_numpy(x[s:s + batch]).to(device)
        b = w.shape[0]
        bt = {
            "motion_features": w[:, L:],
            "history_features": w[:, :L],
            "mask": {"valid": torch.ones(b, H, dtype=torch.bool, device=device)},
            "caption": caps[s:s + batch],
            "has_text": torch.ones(b, dtype=torch.bool, device=device),
            "fps": torch.full((b,), 30.0, device=device),
        }
        torch.manual_seed(seeds[bi]); torch.cuda.manual_seed_all(seeds[bi])
        target, valid, hl = model._target_sequence(bt)
        conds = model._conditions(bt)
        d = model.diffusion.training_losses(model.denoiser, target, conds, valid, history_len=hl)
        tot += float(d["diffusion_loss"]) * b
        n += b
    return tot / n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/workspace/data/omg125")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--exp", default="100m")
    ap.add_argument("--ckpt", default="/workspace/models/omg/checkpoints/updated/100m/sstep=170000.ckpt")
    ap.add_argument("--t5", default="/workspace/models/t5-base-local")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/workspace/out/contract_probe.json")
    a = ap.parse_args()

    dev = torch.device("cuda")
    x, man, L, H = load_windows(a.data, a.split, a.n, a.seed)
    correct = man["en"].astype(str).tolist()
    wrong = correct[-1:] + correct[:-1]              # each window gets a neighbour's caption
    nb = (len(x) + a.batch - 1) // a.batch
    seeds = np.random.default_rng(a.seed).integers(0, 2**31, size=nb).tolist()
    print(f"{len(x)} {a.split} windows, L={L} H={H}, {len(set(correct))} distinct captions")

    rows = []
    for sq, cq in [(False, False), (True, True), (False, True), (True, False)]:
        m = build(a.exp, a.ckpt, a.t5, sq, cq, dev)
        lc = arm_loss(m, x, correct, L, H, dev, a.batch, seeds)
        lw = arm_loss(m, x, wrong, L, H, dev, a.batch, seeds)
        rows.append({"self_qk": sq, "cross_qk": cq, "L_correct": lc,
                     "L_wrong": lw, "delta": lw - lc})
        print(f"  self={str(sq):5s} cross={str(cq):5s}  L_correct={lc:.5f}  "
              f"L_wrong={lw:.5f}  delta={lw-lc:+.5f}", flush=True)
        del m
        torch.cuda.empty_cache()

    best_delta = max(rows, key=lambda r: r["delta"])
    best_loss = min(rows, key=lambda r: r["L_correct"])
    print(f"\n  largest delta (uses text most): self={best_delta['self_qk']} "
          f"cross={best_delta['cross_qk']}  delta={best_delta['delta']:+.5f}")
    print(f"  lowest  L_correct              : self={best_loss['self_qk']} "
          f"cross={best_loss['cross_qk']}  L={best_loss['L_correct']:.5f}")
    agree = (best_delta["self_qk"], best_delta["cross_qk"]) == \
            (best_loss["self_qk"], best_loss["cross_qk"])
    print(f"\n  VERDICT: {'CONTRACT = self_qk=%s cross_qk=%s' % (best_delta['self_qk'], best_delta['cross_qk']) if agree else 'NOT DECISIVE - the two criteria disagree; defer to the official GEN R@1 gate'}")
    Path(a.out).write_text(json.dumps({"rows": rows, "agree": agree}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
