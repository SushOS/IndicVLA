#!/usr/bin/env python3
"""Finalize the MAILA run: re-score every checkpoint, select, render, push everything.

WHY RE-SCORE
------------
`maila_train.py`'s in-loop validation samples FRESH diffusion timesteps each time, so its
`native` number oscillated in a 0.029-0.046 band with no trend across 25k steps. That is a
measurement defect, not a model verdict -- and plan section 8 Phase 2 says to select a
checkpoint on a validation score, so selecting on that number would be selecting on noise.

Here every checkpoint sees IDENTICAL windows, noise and timesteps (fixed per-batch seeds),
and selection uses text-sensitivity rather than `native`:

    rel = (L_wrong - L_correct) / L_correct

The RANDOM anchor showed `native` is dominated by "produce usable conditioning at all"
(L_correct 0.104 -> 0.037), which saturated by ~step 6000. `rel` is the part still moving,
and it is the capability the project actually needs.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/workspace")
from maila_train import WindowDataset, build, collate, native_loss  # noqa: E402


@torch.no_grad()
def score(model, ds, idx, caps, batch, seeds, dev):
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
    ap.add_argument("--run", default="/workspace/runs/maila_v1")
    ap.add_argument("--data", default="/workspace/data/omg125")
    ap.add_argument("--omg", default="/workspace/models/omg/checkpoints/updated/100m/sstep=170000.ckpt")
    ap.add_argument("--t5", default="/workspace/models/t5-base-local")
    ap.add_argument("--muril", default="/workspace/models/muril-base-cased")
    ap.add_argument("--n", type=int, default=768)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--n-render", type=int, default=6)
    ap.add_argument("--repo", default="CodeSushh/PragyaVLA-omgdit-runs")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    dev = torch.device("cuda")
    run = Path(a.run)
    ds = WindowDataset(a.data, "val", "hi")
    rng = np.random.default_rng(a.seed)
    idx = rng.choice(len(ds), size=min(a.n, len(ds)), replace=False)
    shift = np.roll(idx, 1)
    wrong_map = {int(i): ds.hi[int(j)] for i, j in zip(idx, shift)}
    hi_w = [wrong_map.get(i, ds.hi[i]) for i in range(len(ds))]
    nb = (len(idx) + a.batch - 1) // a.batch
    seeds = np.random.default_rng(a.seed + 1).integers(0, 2**31, size=nb).tolist()

    print(f"re-scoring on {len(idx)} val windows, fixed seeds\n")
    rows = []

    m = build("100m", a.omg, a.t5, a.muril, dev, True, True, 0.0)
    rows.append({"ckpt": "RANDOM", "step": 0, "alpha": float(m.text_encoder.adapter.alpha),
                 "L_correct": score(m, ds, idx, ds.hi, a.batch, seeds, dev),
                 "L_wrong": score(m, ds, idx, hi_w, a.batch, seeds, dev)})
    del m; torch.cuda.empty_cache()

    m = build("100m", a.omg, a.t5, a.muril, dev, True, True, 0.0)
    from omg.generation.conditions.t5 import FrozenT5TextEncoder
    m.text_encoder._t5 = FrozenT5TextEncoder(model_name=a.t5, max_length=50,
                                             output_dim=768).to(dev).eval()
    m.text_encoder.passthrough_t5 = True
    en_w = [ds.en[int(j)] if int(i) in wrong_map else ds.en[i]
            for i, j in zip(range(len(ds)), np.arange(len(ds)))]
    for i, j in zip(idx, shift):
        en_w[int(i)] = ds.en[int(j)]
    rows.append({"ckpt": "ENGLISH_t5", "step": -1, "alpha": None,
                 "L_correct": score(m, ds, idx, ds.en, a.batch, seeds, dev),
                 "L_wrong": score(m, ds, idx, en_w, a.batch, seeds, dev)})
    del m; torch.cuda.empty_cache()

    for c in sorted(glob.glob(str(run / "adapter_step*.pt"))):
        st = torch.load(c, map_location="cpu", weights_only=False)
        m = build("100m", a.omg, a.t5, a.muril, dev, True, True, 0.0)
        m.text_encoder.adapter.load_state_dict(st["adapter"])
        m.text_encoder.adapter.eval()
        rows.append({"ckpt": Path(c).name, "step": int(st.get("step", 0)),
                     "alpha": float(m.text_encoder.adapter.alpha),
                     "L_correct": score(m, ds, idx, ds.hi, a.batch, seeds, dev),
                     "L_wrong": score(m, ds, idx, hi_w, a.batch, seeds, dev)})
        del m; torch.cuda.empty_cache()
        print(f"  scored {Path(c).name}", flush=True)

    for r in rows:
        r["delta"] = r["L_wrong"] - r["L_correct"]
        r["rel"] = r["delta"] / max(r["L_correct"], 1e-9)
    rand = next(r for r in rows if r["ckpt"] == "RANDOM")
    eng = next(r for r in rows if r["ckpt"] == "ENGLISH_t5")
    span = eng["rel"] - rand["rel"]
    for r in rows:
        r["pct_of_gap"] = (r["rel"] - rand["rel"]) / span * 100 if abs(span) > 1e-9 else None

    print(f"\n{'ckpt':28s} {'alpha':>7s} {'L_corr':>8s} {'L_wrong':>8s} {'rel':>8s} {'%gap':>7s}")
    for r in rows:
        al = f"{r['alpha']:+.4f}" if r["alpha"] is not None else "   -  "
        pg = f"{r['pct_of_gap']:6.1f}%" if r["pct_of_gap"] is not None else "     -"
        print(f"{r['ckpt'][:28]:28s} {al:>7s} {r['L_correct']:8.5f} {r['L_wrong']:8.5f} "
              f"{r['rel']*100:+7.2f}% {pg:>7s}")

    trained = [r for r in rows if r["ckpt"].startswith("adapter_step")]
    best = max(trained, key=lambda r: r["rel"])
    print(f"\nSELECTED: {best['ckpt']}  rel={best['rel']*100:+.2f}%  "
          f"({best['pct_of_gap']:.1f}% of the random->english gap)")
    (run / "selection.json").write_text(json.dumps(
        {"rows": rows, "selected": best["ckpt"], "n_windows": int(len(idx)),
         "note": "fixed-seed re-score; selection on text-sensitivity rel, not native loss"},
        indent=2))

    # ------------------------------------------------------------------ renders
    print("\nrendering Hindi-driven motion with the selected checkpoint")
    st = torch.load(run / best["ckpt"], map_location="cpu", weights_only=False)
    m = build("100m", a.omg, a.t5, a.muril, dev, True, True, 0.0)
    m.text_encoder.adapter.load_state_dict(st["adapter"])
    m.text_encoder.adapter.eval()
    from omg.render.mujoco import render_qpos_video
    outdir = run / "renders"; outdir.mkdir(exist_ok=True)
    seen, picks = set(), []
    for i in idx:
        c = ds.hi[int(i)]
        if c not in seen:
            seen.add(c); picks.append(int(i))
        if len(picks) >= a.n_render:
            break
    manifest = []
    for k, i in enumerate(picks):
        w = torch.from_numpy(ds.x[i:i + 1]).to(dev)
        b = {
            "history_features": w[:, :ds.L], "prev_state_features": w[:, :ds.L],
            "motion_features": w[:, ds.L:],
            "mask": {"valid": torch.ones(1, ds.H, dtype=torch.bool, device=dev)},
            "caption": [ds.hi[i]], "has_text": torch.ones(1, dtype=torch.bool, device=dev),
            "fps": torch.tensor([30.0], device=dev),
        }
        with torch.no_grad():
            torch.manual_seed(0)
            gen = m.generate(b, num_frames=60, cfg_scale=2.5)
        q = gen["qpos_36"][0].float().cpu().numpy()
        name = f"hi_{k:02d}.mp4"
        render_qpos_video(q, str(outdir / name), fps=30, width=960, height=540,
                          title=ds.en[i][:70], overlay_lines=[ds.hi[i][:70], ds.en[i][:70]])
        disp = float(np.linalg.norm(q[-1, :2] - q[0, :2]))
        manifest.append({"file": name, "hi": ds.hi[i], "en": ds.en[i], "xy_disp": disp})
        print(f"  {name}  disp={disp:.2f}m  {ds.hi[i][:44]}", flush=True)
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    # ------------------------------------------------------------------ push
    print("\npushing to HF")
    from huggingface_hub import HfApi
    api = HfApi(token=open("/root/.creds/hf_token").read().strip())
    api.create_repo(a.repo, exist_ok=True, private=True)
    readme = run / "README.md"
    readme.write_text(f"""# MAILA-OMG Hindi adapter - {a.run_name}

Frozen MuRIL -> trainable residual adapter -> frozen OMG-100M -> G1 motion.
No translation at inference.

## Contract (EXP-F, settled by GEN R@1 on 1024 samples, retrieval batch 32)

| arm | REFERENCE R@1 | GENERATED R@1 |
|---|---|---|
| none (false/false) | 0.6680 | 0.4971 |
| **self_and_cross (true/true)** | 0.6680 | **0.6582** |

REFERENCE 0.6680 in both arms == Gate A exact (research-log Part XV s90), so the
measurement stack is validated. `true/true` wins by 16.1 points -> used for training.

## Training
- OMG-100M `sstep=170000`, fully frozen; only the 4,725,505-param adapter trains
- 56,868 train windows / 5,688 val, 4,015 distinct Hindi captions
- L_native (OMG's own diffusion loss) + lambda_resp * L_response (English-teacher motion
  distillation, same noise/timestep in both branches)
- text_mask_prob=0.0: the DiT is frozen and the CFG null is a cached t5-base constant, so
  dropped rows give the adapter zero gradient
- 25,000 steps, effective batch 256, bf16, 0.39 s/step

## Selection
In-loop `native` validation was uninformative (flat 0.029-0.046, no trend) because it
samples fresh timesteps each eval. Checkpoints were re-scored with FIXED seeds and
selected on text-sensitivity `rel = (L_wrong - L_correct)/L_correct`.

Selected: **{best['ckpt']}**, rel {best['rel']*100:+.2f}%
({best['pct_of_gap']:.1f}% of the random->English-t5 sensitivity gap).

## Caveat
Hindi here is MACHINE-TRANSLATED from English. A strong result partly measures MT
invertibility. The `reserve` split (5,659 windows, 408 distinct captions, disjoint
super-groups) is held out for native-speaker authoring.
""", encoding="utf-8")
    for f in ["selection.json", "history.json", "README.md"]:
        p = run / f
        if p.exists():
            api.upload_file(path_or_fileobj=str(p), path_in_repo=f"{a.run_name}/{f}",
                            repo_id=a.repo)
    api.upload_folder(folder_path=str(outdir), path_in_repo=f"{a.run_name}/renders",
                      repo_id=a.repo)
    for f in ("maila_encoder.py", "maila_train.py", "text_sensitivity.py",
              "contract_probe.py", "smoke_phase1.py", "finalize_run.py"):
        s = Path("/workspace") / f
        if s.exists():
            api.upload_file(path_or_fileobj=str(s), path_in_repo=f"{a.run_name}/code/{f}",
                            repo_id=a.repo)
    for f in glob.glob("/workspace/out/expf_*/metrics.json"):
        arm = Path(f).parent.name
        api.upload_file(path_or_fileobj=f, path_in_repo=f"{a.run_name}/expf/{arm}_metrics.json",
                        repo_id=a.repo)
    print(f"pushed -> {a.repo}/{a.run_name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
