#!/usr/bin/env python3
"""MAILA-OMG: train the Hindi adapter against a fully frozen OMG-100M.

L_total = L_native + lambda_resp(step) * L_response          (MAILA plan section 7)

  L_native   OMG's OWN clean-motion diffusion loss, conditioned on Hindi. Obtained by
             calling `model.diffusion.training_losses(...)` -- the same call OMG's
             `_shared_step` makes -- so the reduction matches the pipeline that was
             reproduced, rather than a re-implementation of it (plan 7.1).

  L_response || sg[x0_en] - x0_hi ||^2 with the SAME motion, history, noise and timestep
             in both branches. The RNG state is saved and restored around the two calls,
             because if the branches drew different noise the loss would be dominated by
             that difference instead of by the conditioning (plan 7.2 step 3).

The document renders the total with a leading '-' on the response term; that is a markdown
list-marker artifact of a mangled '+' (a decaying POSITIVE weight is scheduled in 7.3).
It is added here, not subtracted.

Everything except the ~4.7M-parameter adapter is frozen and asserted so at startup.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, "/workspace")  # so hydra can import maila_encoder


# ---------------------------------------------------------------- data
class WindowDataset(torch.utils.data.Dataset):
    """Converted bones-seed windows: (70,125) canonicalised on frame 9 -> history + future."""

    def __init__(self, root: str, split: str, lang: str = "hi", limit: int | None = None):
        d = Path(root) / split
        shards = sorted(glob.glob(str(d / "shard_*.npz")))
        if not shards:
            raise SystemExit(f"no shards under {d}")
        self.x = np.concatenate([np.load(s)["features"] for s in shards])
        self.man = pd.read_parquet(d / "manifest.parquet")
        meta = json.loads((d / "_meta.json").read_text())
        self.L, self.H = int(meta["L"]), int(meta["H"])
        if len(self.man) != len(self.x):
            raise SystemExit(f"manifest {len(self.man)} != features {len(self.x)}")
        if limit:
            self.x, self.man = self.x[:limit], self.man.iloc[:limit].reset_index(drop=True)
        self.lang = lang
        self.hi = self.man[lang].astype(str).tolist()
        self.en = self.man["en"].astype(str).tolist()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        w = self.x[i]
        return {"history": w[: self.L], "future": w[self.L:], "hi": self.hi[i], "en": self.en[i]}


def collate(items, device):
    hist = torch.from_numpy(np.stack([b["history"] for b in items])).to(device)
    fut = torch.from_numpy(np.stack([b["future"] for b in items])).to(device)
    b = len(items)
    return {
        "motion_features": fut,                                  # (B,H,125) un-normalised
        "history_features": hist,                                # (B,L,125)
        "mask": {"valid": torch.ones(b, fut.shape[1], dtype=torch.bool, device=device)},
        "caption": [x["hi"] for x in items],
        "caption_en": [x["en"] for x in items],
        "has_text": torch.ones(b, dtype=torch.bool, device=device),
        "fps": torch.full((b,), 30.0, device=device),
    }


# ---------------------------------------------------------------- model
def build(exp, ckpt, t5, muril, device, self_qk, cross_qk, text_mask_prob,
          t5_calibration=None, norm_mode="layernorm"):
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
            f"model.text_mask_prob={text_mask_prob}",
            # hydra runs in struct mode: NEW keys need '+', and the t5 class's
            # `model_name` must be deleted or instantiate() passes an unexpected kwarg.
            "model.text_encoder._target_=maila_encoder.MurilAdapterEncoder",
            "~model.text_encoder.model_name",
            f"+model.text_encoder.muril_name={muril}",
            f"+model.text_encoder.t5_model_name={t5}",
            *([f"+model.text_encoder.t5_calibration={t5_calibration}"]
              if t5_calibration else []),
            f"+model.text_encoder.norm_mode={norm_mode}",
        ])
    model = instantiate(cfg.model)
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = payload.get("state_dict", payload)
    # the swapped encoder has different keys than the checkpoint's t5 -- load the rest strictly
    missing, unexpected = model.load_state_dict(sd, strict=False)
    bad = [k for k in unexpected if not k.startswith("text_encoder.")]
    if bad:
        raise SystemExit(f"unexpected non-text_encoder keys in checkpoint: {bad[:6]}")
    denoiser_missing = [k for k in missing if k.startswith("denoiser.")]
    if denoiser_missing:
        raise SystemExit(f"DENOISER WEIGHTS MISSING: {denoiser_missing[:6]}")
    print(f"  loaded {exp} step={payload.get('global_step')} "
          f"self_qk={self_qk} cross_qk={cross_qk} text_mask_prob={text_mask_prob}")
    print(f"  encoder keys not in ckpt (expected, new module): {len(missing) - 0}")

    model = model.to(device)
    # freeze everything, then re-enable ONLY the adapter
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.text_encoder.adapter.parameters():
        p.requires_grad_(True)
    model.eval()
    model.text_encoder.adapter.train()

    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if not all(t.startswith("text_encoder.adapter") for t in trainable):
        raise SystemExit(f"non-adapter params are trainable: "
                         f"{[t for t in trainable if not t.startswith('text_encoder.adapter')][:5]}")
    print(f"  trainable: {len(trainable)} tensors, {n_tr:,} params (adapter only)")
    return model


def make_teacher(t5_path, device):
    from omg.generation.conditions.t5 import FrozenT5TextEncoder
    t = FrozenT5TextEncoder(model_name=t5_path, max_length=50, output_dim=768)
    t = t.to(device).eval()
    for p in t.parameters():
        p.requires_grad_(False)
    return t


# ---------------------------------------------------------------- losses
def native_loss(model, batch):
    """OMG's own diffusion loss, conditioned on whatever text_encoder returns."""
    target, valid, hist_len = model._target_sequence(batch)
    conds = model._conditions(batch)
    diff = model.diffusion.training_losses(model.denoiser, target, conds, valid,
                                           history_len=hist_len)
    return diff["diffusion_loss"], diff, target, valid, hist_len


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/workspace/data/omg125")
    ap.add_argument("--exp", default="100m")
    ap.add_argument("--ckpt", default="/workspace/models/omg/checkpoints/updated/100m/sstep=170000.ckpt")
    ap.add_argument("--t5", default="/workspace/models/t5-base-local")
    ap.add_argument("--muril", default="/workspace/models/muril-base-cased")
    ap.add_argument("--self-qk", default="true")
    ap.add_argument("--cross-qk", default="true")
    ap.add_argument("--lang", default="hi")
    ap.add_argument("--steps", type=int, default=25000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--accum", type=int, default=4, help="effective batch = bs * accum")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--text-mask-prob", type=float, default=0.0,
                    help="0.0 (user decision 2026-08-22), deviating from plan 7.3's 0.3. "
                         "The DiT is FROZEN and the CFG null context is a cached t5-base "
                         "constant, so the unconditional branch contains no trainable weight: "
                         "a dropped row produces exactly zero adapter gradient. Keeping 0.3 "
                         "would waste ~30%% of steps. CFG at inference is unaffected -- verify "
                         "with the Phase-1 null-vs-conditional smoke test.")
    ap.add_argument("--lambda-resp-hi", type=float, default=0.5)
    ap.add_argument("--lambda-resp-lo", type=float, default=0.1)
    ap.add_argument("--resp-hold", type=int, default=5000)
    ap.add_argument("--teacher-prob", type=float, default=0.5)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--limit-train", type=int, default=None)
    ap.add_argument("--out", default="/workspace/runs/maila_v1")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--hf-repo", default="PragyaVLA/PragyaVLA-omgdit-runs")
    ap.add_argument("--hf-fallback-repo", default="CodeSushh/PragyaVLA-omgdit-runs")
    ap.add_argument("--wandb-project", default="maila-omg-hindi-adapter")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-wandb", action="store_true")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = torch.device("cuda")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    run_name = a.run_name or f"maila-{a.lang}-{time.strftime('%Y%m%d-%H%M%S')}"

    tr = WindowDataset(a.data, "train", a.lang, a.limit_train)
    va = WindowDataset(a.data, "val", a.lang)
    print(f"train {len(tr):,} windows | val {len(va):,} | L={tr.L} H={tr.H}")
    print(f"distinct {a.lang} captions in train: {len(set(tr.hi)):,}")

    model = build(a.exp, a.ckpt, a.t5, a.muril, dev,
                  a.self_qk == "true", a.cross_qk == "true", a.text_mask_prob)
    teacher = make_teacher(a.t5, dev)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.weight_decay)

    def lr_at(s):
        if s < a.warmup:
            return a.lr * (s + 1) / a.warmup
        t = (s - a.warmup) / max(1, a.steps - a.warmup)
        return 1e-6 + 0.5 * (a.lr - 1e-6) * (1 + math.cos(math.pi * t))

    def lam(s):
        if s < a.resp_hold:
            return a.lambda_resp_hi
        t = (s - a.resp_hold) / max(1, a.steps - a.resp_hold)
        return a.lambda_resp_hi + (a.lambda_resp_lo - a.lambda_resp_hi) * min(1.0, t)

    wb = None
    if not a.no_wandb:
        import wandb
        wb = wandb.init(project=a.wandb_project, name=run_name,
                        config={**vars(a), "train_windows": len(tr), "val_windows": len(va),
                                "trainable_params": sum(p.numel() for p in params)})

    rng = np.random.default_rng(a.seed)
    hist = []
    t0 = time.time()
    for step in range(1, a.steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        opt.zero_grad(set_to_none=True)
        agg = {"loss": 0.0, "native": 0.0, "resp": 0.0}

        for _ in range(a.accum):
            idx = rng.integers(0, len(tr), size=a.batch_size)
            batch = collate([tr[int(i)] for i in idx], dev)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                state = torch.cuda.get_rng_state(dev), torch.get_rng_state()
                l_native, diff, target, valid, hlen = native_loss(model, batch)

                l_resp = torch.zeros((), device=dev)
                if lam(step) > 0 and rng.random() < a.teacher_prob:
                    # identical motion / history / noise / timestep; only the text differs
                    torch.cuda.set_rng_state(state[0], dev)
                    torch.set_rng_state(state[1])
                    with torch.no_grad():
                        enc = teacher(batch["caption_en"], has_text=batch["has_text"], device=dev)
                        tb = dict(batch)
                        tb["_text_override"] = enc
                        conds_en = model._conditions(batch)
                        conds_en["text_context"], conds_en["text_mask"] = enc["context"], enc["mask"]
                        d_en = model.diffusion.training_losses(
                            model.denoiser, target, conds_en, valid, history_len=hlen)
                    l_resp = F.mse_loss(diff["pred_x0"], d_en["pred_x0"].detach())

                loss = (l_native + lam(step) * l_resp) / a.accum
            loss.backward()
            agg["loss"] += float(loss) * a.accum / a.accum
            agg["native"] += float(l_native) / a.accum
            agg["resp"] += float(l_resp) / a.accum

        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

        if step % a.log_every == 0 or step == 1:
            alpha = float(model.text_encoder.adapter.alpha)
            row = {"step": step, "loss": agg["loss"], "native": agg["native"],
                   "resp": agg["resp"], "lambda_resp": lam(step), "lr": lr_at(step),
                   "alpha": alpha, "grad_norm": float(gn),
                   "sec_per_step": (time.time() - t0) / step}
            hist.append(row)
            print(f"  {step:6d} loss={row['loss']:.5f} native={row['native']:.5f} "
                  f"resp={row['resp']:.5f} alpha={alpha:+.4f} gn={row['grad_norm']:.2f} "
                  f"{row['sec_per_step']:.2f}s/step", flush=True)
            if wb:
                wb.log(row, step=step)

        if step % a.val_every == 0 or step == a.steps:
            model.text_encoder.adapter.eval()
            vs = []
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for k in range(0, min(len(va), 512), a.batch_size):
                    vb = collate([va[j] for j in range(k, min(k + a.batch_size, len(va)))], dev)
                    vl, *_ = native_loss(model, vb)
                    vs.append(float(vl))
            model.text_encoder.adapter.train()
            vmean = float(np.mean(vs))
            print(f"  [val] step {step}: native={vmean:.5f}", flush=True)
            if wb:
                wb.log({"val/native": vmean}, step=step)

        if step % a.ckpt_every == 0 or step == a.steps:
            p = out / f"adapter_step{step:06d}.pt"
            model.text_encoder.save_adapter(
                str(p), step=step, args=vars(a), history=hist[-50:],
                omg_ckpt=os.path.basename(a.ckpt),
                contract={"self_qk": a.self_qk, "cross_qk": a.cross_qk})
            (out / "history.json").write_text(json.dumps(hist, indent=1))
            # push immediately: workspace_is_volume=false, nothing here survives a recycle
            try:
                from huggingface_hub import HfApi
                api = HfApi(token=open("/root/.creds/hf_token").read().strip())
                repo = a.hf_repo
                try:
                    api.create_repo(repo, exist_ok=True, private=True)
                    api.upload_file(path_or_fileobj=str(p),
                                    path_in_repo=f"{run_name}/{p.name}", repo_id=repo)
                except Exception:
                    repo = a.hf_fallback_repo       # org is out of private storage
                    api.create_repo(repo, exist_ok=True, private=True)
                    api.upload_file(path_or_fileobj=str(p),
                                    path_in_repo=f"{run_name}/{p.name}", repo_id=repo)
                api.upload_file(path_or_fileobj=str(out / "history.json"),
                                path_in_repo=f"{run_name}/history.json", repo_id=repo)
                print(f"  pushed {p.name} -> {repo}/{run_name}", flush=True)
            except Exception as e:
                print(f"  !! HF push failed: {str(e)[:160]}", flush=True)

    print(f"\ndone in {(time.time()-t0)/60:.1f} min -> {out}")
    if wb:
        wb.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
