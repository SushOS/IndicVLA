#!/usr/bin/env python3
"""MAILA Run 2 -- motion-pivot cross-lingual training against a frozen OMG-100M.

    L = 1/2 [ L_native(A) + L_native(B) ]
      + lambda_x(step)  * || x0_hat(A) - x0_hat(B) ||^2      <- motion is the pivot
      + lambda_en       * || sg[x0_hat(en)] - x0_hat(.) ||^2  <- geometry anchor (optional)

WHAT CHANGED FROM `maila_train.py` (Run 1), AND WHY
---------------------------------------------------
Run 1's second branch was frozen T5 on English under `no_grad`. Its output is a
CONSTANT for a given (motion, tau, eps), so that term is distillation toward a fixed
target -- and, critically, toward *English's prediction of the motion* rather than the
motion itself. Ground truth entered only through L_native.

Here the second branch is another view of the SAME clip through the SAME adapter, and
it carries gradient. Both branches regress toward the real motion x0 via their own
native loss; the consistency term additionally requires that their residual errors
MATCH. That is the difference between "be like English" and "agree with each other
about the physics".

THE SIGN FLIP THIS INTRODUCES (read before touching lambda_x)
--------------------------------------------------------------
Run 1's English target was caption-dependent, so matching it FORCED caption-dependence:
the teacher was an anti-collapse force. Consistency has the opposite sign -- the cheapest
way for two branches to agree is for neither to read its text. Measured economics
(research log s5.5 + SSOT s7.2): collapsing costs ~0.011 of native loss and saves
~lambda_x * 0.004, i.e. a ~5x losing trade at lambda_x=0.5, break-even near 2.6.
KEEP lambda_x <= 0.5. Keep --lambda-en > 0 by default; it restores exactly the pressure
being removed. And watch `tiv` (text-induced variance) in the logs -- it collapses long
before `rel` or R@1 do.

RNG REPLAY
----------
Every branch replays one saved RNG state, so tau, eps, x_t and even the adapter dropout
mask are identical and the ONLY difference between branches is which caption produced
the context. Order of consumption is identical across branches (conditions -> dropout,
then training_losses -> tau, eps) and all shapes are caption-independent, so the replay
is exact.

HINDI-ONLY (Run 2a):   --langs hi --pair-mode paraphrase
FOUR LANGUAGES:        --langs hi,bn,ta,te --pair-mode both
"""
from __future__ import annotations

import argparse
import collections
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

sys.path.insert(0, "/workspace")

from maila_train import build, make_teacher, native_loss  # noqa: E402  (reuse Run 1 wiring)

SLOTS = [1, 2, 3, 4]


# ---------------------------------------------------------------- data
class MultiCaptionWindowDataset(torch.utils.data.Dataset):
    """Windows from convert_g1_npz_to_omg125.py, exposing every caption slot per language."""

    def __init__(self, root: str, split: str, langs: list[str], limit: int | None = None):
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
        self.langs = list(langs)

        # caps[lang][i]  -> non-empty caption strings for window i
        # slots[lang][i] -> the caption-slot index each of those came from, kept in parallel so
        #                   a cross-lingual pair can be classified parallel vs non-parallel
        self.caps: dict[str, list[list[str]]] = {}
        self.slots: dict[str, list[list[int]]] = {}
        for lg in self.langs + ["en"]:
            present = [s for s in SLOTS if f"{lg}_{s}" in self.man.columns]
            if not present:
                raise SystemExit(f"manifest has no caption columns for language {lg!r}")
            arr = self.man[[f"{lg}_{s}" for s in present]].fillna("").astype(str).values
            self.caps[lg] = [[c for c in row if c.strip()] for row in arr]
            self.slots[lg] = [[s for s, c in zip(present, row) if c.strip()] for row in arr]

        n_multi = sum(1 for lg in self.langs for r in self.caps[lg] if len(r) >= 2)
        print(f"  {split}: {len(self.x):,} windows | langs={self.langs} | "
              f"L={self.L} H={self.H} | window-language pairs with >=2 captions: {n_multi:,}")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        w = self.x[i]
        return {
            "history": w[: self.L],
            "future": w[self.L:],
            "caps": {lg: self.caps[lg][i] for lg in self.langs},
            "slots": {lg: self.slots[lg][i] for lg in self.langs},
            "en": self.caps["en"][i][0] if self.caps["en"][i] else "",
        }


def sample_views(items, langs, pair_mode, rng, paraphrase_frac=0.2, nonparallel_only=False):
    """Two (language, caption) views of each window, plus the KIND of each pair.

    The pair TYPE is decided FIRST, by an explicit coin flip, and only then is a pair drawn
    within that type. An earlier version tried cross-lingual first and fell back to
    paraphrase, which -- with 4 languages -- succeeded on the cross-lingual branch nearly
    every time and made `both` mean "cross-lingual with a rare fallback". The mix has to be
    a stated ratio you can ablate and report, not an accident of control flow.

    Default 0.2 matches the natural edge ratio: with 4 languages x 3 caption slots there are
    C(12,2)=66 unordered pairs, of which 4*C(3,2)=12 are paraphrase -> 12/66 = 0.18.

    `nonparallel_only` forces cross-lingual pairs to use DIFFERENT caption slots, which is
    the strong form of the claim (two independently-written sentences in two scripts, related
    only through the motion). Left off, ~1/3 of cross-lingual pairs land on the same slot and
    are therefore translations of each other.

    Returns (a_texts, b_texts, kinds) where kind is one of:
        para | cross_nonparallel | cross_parallel | degenerate
    """
    a_txt, b_txt, kinds = [], [], []
    for it in items:
        para_langs = [lg for lg in langs if len(it["caps"][lg]) >= 2]
        cross_langs = [lg for lg in langs if len(it["caps"][lg]) >= 1]
        can_para, can_cross = bool(para_langs), len(cross_langs) >= 2

        if pair_mode == "paraphrase":
            want_para = True
        elif pair_mode == "crosslingual":
            want_para = False
        else:
            want_para = bool(rng.random() < paraphrase_frac)
        if want_para and not can_para:
            want_para = False
        if not want_para and not can_cross:
            want_para = True

        if want_para and can_para:
            lg = para_langs[rng.integers(len(para_langs))]
            cs = it["caps"][lg]
            i1, i2 = rng.choice(len(cs), size=2, replace=False)
            a, b, kind = cs[i1], cs[i2], "para"
        elif can_cross:
            i, j = rng.choice(len(cross_langs), size=2, replace=False)
            la, lb = cross_langs[i], cross_langs[j]
            ia = int(rng.integers(len(it["caps"][la])))
            cand_b = range(len(it["caps"][lb]))
            if nonparallel_only:
                other = [k for k in cand_b if it["slots"][lb][k] != it["slots"][la][ia]]
                cand_b = other or list(cand_b)      # fall back if the language has one slot
            cand_b = list(cand_b)
            ib = int(cand_b[rng.integers(len(cand_b))])
            a, b = it["caps"][la][ia], it["caps"][lb][ib]
            kind = ("cross_parallel" if it["slots"][la][ia] == it["slots"][lb][ib]
                    else "cross_nonparallel")
        else:
            pool = [c for lg in langs for c in it["caps"][lg]]
            c = pool[rng.integers(len(pool))] if pool else it["en"]
            a, b, kind = c, c, "degenerate"       # consistency term is exactly 0 here

        a_txt.append(a); b_txt.append(b); kinds.append(kind)
    return a_txt, b_txt, kinds


def collate(items, device):
    hist = torch.from_numpy(np.stack([b["history"] for b in items])).to(device)
    fut = torch.from_numpy(np.stack([b["future"] for b in items])).to(device)
    b = len(items)
    return {
        "motion_features": fut,
        "history_features": hist,
        "mask": {"valid": torch.ones(b, fut.shape[1], dtype=torch.bool, device=device)},
        "caption": [""] * b,                       # filled per branch
        "caption_en": [x["en"] for x in items],
        "has_text": torch.ones(b, dtype=torch.bool, device=device),
        "fps": torch.full((b,), 30.0, device=device),
    }


# ---------------------------------------------------------------- rng replay
def save_rng(dev):
    return torch.cuda.get_rng_state(dev), torch.get_rng_state()


def restore_rng(state, dev):
    torch.cuda.set_rng_state(state[0], dev)
    torch.set_rng_state(state[1])


# ---------------------------------------------------------------- diagnostics
@torch.no_grad()
def text_induced_variance(model, ds, idx, lang, batch_size, dev, seed=0):
    """mean ||x0_hat(c_i) - x0_hat(c_{i+1})||^2 with MATCHED history and noise.

    Never trained on -- this stays a valid selection signal. Collapse drives it to 0
    while native loss, `rel` and R@1 can all still look healthy.
    """
    tot, n = 0.0, 0
    for s in range(0, len(idx), batch_size):
        sel = idx[s:s + batch_size]
        if len(sel) < 2:
            continue
        items = [ds[int(i)] for i in sel]
        b = collate(items, dev)
        own = [ds.caps[lang][int(i)][0] if ds.caps[lang][int(i)] else "" for i in sel]
        rolled = own[1:] + own[:1]
        st = save_rng(dev)
        b["caption"] = own
        _, d1, *_ = native_loss(model, b)
        restore_rng(st, dev)
        b["caption"] = rolled
        _, d2, *_ = native_loss(model, b)
        tot += float(F.mse_loss(d1["pred_x0"], d2["pred_x0"])) * len(sel)
        n += len(sel)
    return tot / max(n, 1)


@torch.no_grad()
def geometry_gate(model, ds, lang, dev, strict=True, n=128):
    """Refuse to train unless the adapter's OUTPUT DISTRIBUTION is one the frozen DiT can read.

    Two Run-2 failures were invisible to every shape check and to state_dict loading, and each
    cost GPU-days before anyone measured the tensor rather than its shape:

      #14  out_norm pinned the token norm to sqrt(768) = 27.71 against t5-base's 6.758, so the
           frozen cross-attention read context vectors 4.1x too long and roughly HALF the
           denoising loss was scale error rather than semantics.
      #15  nn.LayerNorm standardises WITHIN a token and so cannot remove MuRIL's shared
           component (||mu_corpus|| = 26.91). Two DIFFERENT captions arrived at cosine 0.9966
           -- the model was conditioned on a near-constant, and `rel` could not move.

    Both are properties of the output DISTRIBUTION. `(B, 50, 768)` was correct throughout and
    no assertion fired. So this gate measures the distribution, before a GPU-hour is spent.
    """
    caps, seen = [], set()
    for i in range(len(ds)):
        c = ds.caps[lang][i]
        if c and c[0] and c[0] not in seen:
            seen.add(c[0]); caps.append(c[0])
        if len(caps) >= n:
            break
    o = model.text_encoder(caps, device=dev)
    x, m = o["context"].float(), o["mask"]
    mf = m.float().unsqueeze(-1)
    norm = float((x.norm(dim=-1) * m).sum() / m.sum().clamp_min(1))
    c = (x * mf).sum(1) / mf.sum(1).clamp_min(1)              # per-caption mean token
    mu = c.mean(0)
    cn = torch.nn.functional.normalize(c, dim=1)
    off = cn @ cn.T
    eye = torch.eye(len(c), dtype=torch.bool, device=off.device)
    cos = float(off[~eye].mean())
    btwn = float((c - mu).norm(dim=1).mean() / mu.norm().clamp_min(1e-6))
    print(f"  [geometry gate] n={len(caps)}  token_norm={norm:.3f} (t5 6.758)  "
          f"cos(i,j)={cos:.4f} (t5 0.6563)  btwn/mu={btwn:.4f} (t5 0.7201)")
    if strict:
        if not 4.0 < norm < 12.0:
            raise SystemExit(f"GATE FAIL: init token norm {norm:.3f} outside t5's range. The "
                             f"DiT would read out-of-distribution context. See SSOT s14 #14.")
        if cos > 0.95:
            raise SystemExit(f"GATE FAIL: cos between different captions is {cos:.4f}. The "
                             f"adapter emits a near-constant vector, so no amount of training "
                             f"can make `rel` move. See SSOT s14 #15.")
    return {"init/token_norm": norm, "init/cos_ij": cos, "init/btwn_mu": btwn}


@torch.no_grad()
def fixed_seed_validation(model, ds, idx, langs, batch_size, dev, seeds):
    """Run 1's in-loop validation was UNINFORMATIVE: it drew fresh timesteps and noise each
    eval, so `native` oscillated in a 0.029-0.046 band with no trend across 25k steps and
    could not distinguish a better checkpoint from sampling noise (research log s5.5).

    This replays FIXED per-batch seeds, so the only thing that varies between evaluations is
    the adapter. It also computes the caption-swap sensitivity `rel` LIVE rather than only in
    a post-hoc pass, so collapse is visible during training instead of after it.
    """
    out = {}
    for lg in langs:
        own, wrong, tot_c, tot_w, n = [], [], 0.0, 0.0, 0
        for bi, s in enumerate(range(0, len(idx), batch_size)):
            sel = idx[s:s + batch_size]
            if len(sel) < 2:
                continue
            items = [ds[int(i)] for i in sel]
            b = collate(items, dev)
            caps = [ds.caps[lg][int(i)][0] if ds.caps[lg][int(i)] else "" for i in sel]
            rolled = caps[1:] + caps[:1]
            sd = seeds[bi % len(seeds)]

            torch.manual_seed(sd); torch.cuda.manual_seed_all(sd)
            b["caption"] = caps
            lc, *_ = native_loss(model, b)
            torch.manual_seed(sd); torch.cuda.manual_seed_all(sd)
            b["caption"] = rolled
            lw, *_ = native_loss(model, b)

            tot_c += float(lc) * len(sel); tot_w += float(lw) * len(sel); n += len(sel)
        if n:
            Lc, Lw = tot_c / n, tot_w / n
            out[f"val/native_{lg}"] = Lc
            out[f"val/native_wrong_{lg}"] = Lw
            out[f"val/rel_{lg}"] = (Lw - Lc) / max(Lc, 1e-9)
    return out


@torch.no_grad()
def timestep_loss_profile(model, ds, idx, lang, batch_size, dev, nbuckets=5):
    """Where in the diffusion schedule is the model weak? Run 1 never looked.

    At high tau the prediction is dominated by the prior and text barely matters; at low tau
    it is dominated by x_t. If the consistency term is doing anything, its effect should show
    up in the MIDDLE buckets -- and if `rel` is healthy only at low tau, the model is
    exploiting the input rather than reading the caption.
    """
    T = int(model.diffusion.train_timestep_map.shape[0])
    edges = np.linspace(0, T, nbuckets + 1).astype(int)
    acc = collections.defaultdict(lambda: [0.0, 0])
    for s in range(0, len(idx), batch_size):
        sel = idx[s:s + batch_size]
        b = collate([ds[int(i)] for i in sel], dev)
        b["caption"] = [ds.caps[lang][int(i)][0] if ds.caps[lang][int(i)] else "" for i in sel]
        _, diff, *_ = native_loss(model, b)
        ts = diff["timesteps"].detach().float().cpu().numpy()
        per = (diff["raw_prediction"] - diff["target"]).pow(2).mean(dim=(1, 2)).float().cpu().numpy()
        for t, v in zip(ts, per):
            k = int(np.clip(np.searchsorted(edges, t, side="right") - 1, 0, nbuckets - 1))
            acc[k][0] += float(v); acc[k][1] += 1
    return {f"val/loss_tau_b{k}": acc[k][0] / max(acc[k][1], 1)
            for k in sorted(acc) if acc[k][1]}


@torch.no_grad()
def null_context_loss(model, ds, idx, batch_size, dev, seeds):
    """L_null: native loss under the cached t5 empty-string context (SSOT s11.0 gate).

    This is the TEXT-BLIND FLOOR. L_wrong is text-present-but-wrong and is NOT the same
    quantity; every collapse-economics number depends on this one being measured.
    """
    tot, n = 0.0, 0
    enc = model.text_encoder
    for bi, s in enumerate(range(0, len(idx), batch_size)):
        sel = idx[s:s + batch_size]
        b = collate([ds[int(i)] for i in sel], dev)
        b["caption"] = [""] * len(sel)
        torch.manual_seed(seeds[bi]); torch.cuda.manual_seed_all(seeds[bi])
        ctx = enc(b["caption"], has_text=b["has_text"], force_null_text=True, device=dev)
        target, valid, hlen = model._target_sequence(b)
        conds = model._conditions(b)
        conds["text_context"], conds["text_mask"] = ctx["context"], ctx["mask"]
        d = model.diffusion.training_losses(model.denoiser, target, conds, valid,
                                            history_len=hlen)
        tot += float(d["diffusion_loss"]) * len(sel); n += len(sel)
    return tot / max(n, 1)


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/workspace/data/amass_g1_omg125")
    ap.add_argument("--exp", default="100m")
    ap.add_argument("--ckpt", default="/workspace/models/omg/checkpoints/updated/100m/sstep=170000.ckpt")
    ap.add_argument("--t5", default="/workspace/models/t5-base-local")
    ap.add_argument("--muril", default="/workspace/models/muril-base-cased")
    ap.add_argument("--self-qk", default="true")
    ap.add_argument("--cross-qk", default="true")
    ap.add_argument("--langs", default="hi", help="comma separated, e.g. hi or hi,bn,ta,te")
    ap.add_argument("--pair-mode", choices=["paraphrase", "crosslingual", "both"],
                    default="paraphrase")
    ap.add_argument("--paraphrase-frac", type=float, default=0.2,
                    help="with --pair-mode both, the probability a step uses a same-language "
                         "paraphrase pair rather than a cross-lingual one. 0.2 matches the "
                         "natural edge ratio (12 of 66 pairs at 4 langs x 3 slots). Ignored "
                         "by the other pair modes")
    ap.add_argument("--crosslingual-nonparallel", action="store_true",
                    help="force cross-lingual pairs to use DIFFERENT caption slots -- the "
                         "strong form of the claim (no sentence-level parallelism). Without "
                         "it ~1/3 of cross-lingual pairs are translations of each other")
    ap.add_argument("--steps", type=int, default=25000)
    ap.add_argument("--batch-size", type=int, default=32,
                    help="32 (not 64): two grad-carrying passes through the frozen DiT")
    ap.add_argument("--accum", type=int, default=8, help="32 x 8 = effective 256, as in Run 1")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--text-mask-prob", type=float, default=0.0,
                    help="0.0: the DiT is frozen and the CFG null is a cached t5 constant, so a "
                         "dropped-text row yields zero adapter gradient (same as Run 1)")
    ap.add_argument("--lambda-x-hi", type=float, default=0.5,
                    help="KEEP <= 0.5. Break-even against collapse sits near 2.6")
    ap.add_argument("--lambda-x-lo", type=float, default=0.2)
    ap.add_argument("--lambda-x-hold", type=int, default=5000)
    ap.add_argument("--lambda-en", type=float, default=0.1,
                    help="English-anchor weight at the START of training. HISTORY: raised to "
                         "0.5 after run ...-174910 stalled at rel=-1%%, on the theory that the "
                         "anchor was too weak to pull the adapter into t5 geometry. Direct "
                         "measurement on 2026-09-06 found the real cause was structural, not "
                         "a weight: ResidualAdapter's out_norm pinned every token to norm "
                         "sqrt(768)=27.71 against t5's 6.76, so no lambda_en could fix it. "
                         "With --t5-calibration the adapter now STARTS in t5 geometry, so the "
                         "anchor is back to 0.1 and the motion pivot stays the primary signal")
    ap.add_argument("--norm-mode", default="layernorm", choices=["layernorm", "corpus"],
                    help="layernorm = as-built (cos(i,j)=0.9966, cannot condition). "
                         "corpus = subtract MuRIL's corpus mean and per-channel std before "
                         "recolouring to t5, which recovers 13.8x the caption signal")
    ap.add_argument("--skip-geometry-gate", action="store_true",
                    help="only for deliberately running a known-bad geometry as an ablation")
    ap.add_argument("--t5-calibration", default=None,
                    help="t5 per-channel {mu,sd} from calibrate_t5_geometry.py. Initialises "
                         "the adapter's out_norm affine so its output lands in the frozen "
                         "DiT's actual input distribution instead of 4.1x outside it. "
                         "Measured effect at random init: L_correct 0.07454 -> 0.03883")
    ap.add_argument("--lambda-en-lo", type=float, default=0.1,
                    help="English-anchor weight after decay -- the motion pivot takes over "
                         "once the geometry is established")
    ap.add_argument("--lambda-en-hold", type=int, default=5000,
                    help="steps to hold --lambda-en before decaying to --lambda-en-lo")
    ap.add_argument("--tau-weight", action="store_true",
                    help="scale the consistency term by alpha_bar(tau); high-tau branches "
                         "agree for reasons unrelated to language")
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--val-windows", type=int, default=1024)
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--limit-train", type=int, default=None)
    ap.add_argument("--out", default="/workspace/runs/maila_motionpivot")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--hf-repo", default="CodeSushh/PragyaVLA-omgdit-runs")
    ap.add_argument("--wandb-project", default="maila-motionpivot")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--probe-null-only", action="store_true",
                    help="run the L_null gate and exit (SSOT s11.0)")
    a = ap.parse_args()

    langs = [x.strip() for x in a.langs.split(",") if x.strip()]
    if a.pair_mode == "crosslingual" and len(langs) < 2:
        raise SystemExit("--pair-mode crosslingual needs >=2 languages")

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device("cuda")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    run_name = a.run_name or f"maila-mp-{'-'.join(langs)}-{time.strftime('%Y%m%d-%H%M%S')}"

    tr = MultiCaptionWindowDataset(a.data, "train", langs, a.limit_train)
    va = MultiCaptionWindowDataset(a.data, "val", langs)

    model = build(a.exp, a.ckpt, a.t5, a.muril, dev,
                  a.self_qk == "true", a.cross_qk == "true", a.text_mask_prob,
                  t5_calibration=a.t5_calibration, norm_mode=a.norm_mode)
    init_geom = geometry_gate(model, va, langs[0], dev, strict=not a.skip_geometry_gate)

    # ---- SSOT s11.0 gate: the text-blind floor, before any tuning ----------
    vrng = np.random.default_rng(a.seed)
    vidx = vrng.choice(len(va), size=min(a.val_windows, len(va)), replace=False)
    vseeds = np.random.default_rng(a.seed + 1).integers(
        0, 2**31, size=(len(vidx) + a.batch_size - 1) // a.batch_size).tolist()
    l_null = null_context_loss(model, va, vidx, a.batch_size, dev, vseeds)
    print(f"\n  [GATE] L_null (cached t5 empty-string context, {len(vidx)} val windows) "
          f"= {l_null:.5f}")
    print(f"         Run 1 reference on BONES-SEED: L_correct 0.04815 / L_wrong 0.05900")
    print(f"         If L_null ~ L_wrong the native loss is a strong anchor; if it sits near "
          f"L_correct, add --lambda-en and consider the repulsion term.\n")
    (out / "l_null.json").write_text(json.dumps(
        {"l_null": l_null, "n_windows": int(len(vidx)), "seed": a.seed}, indent=2))
    if a.probe_null_only:
        return 0

    teacher = make_teacher(a.t5, dev) if max(a.lambda_en, a.lambda_en_lo) > 0 else None
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.weight_decay)

    def lr_at(s):
        if s < a.warmup:
            return a.lr * (s + 1) / a.warmup
        t = (s - a.warmup) / max(1, a.steps - a.warmup)
        return 1e-6 + 0.5 * (a.lr - 1e-6) * (1 + math.cos(math.pi * t))

    def lam_x(s):
        if s < a.lambda_x_hold:
            return a.lambda_x_hi
        t = (s - a.lambda_x_hold) / max(1, a.steps - a.lambda_x_hold)
        return a.lambda_x_hi + (a.lambda_x_lo - a.lambda_x_hi) * min(1.0, t)

    def lam_en(s):
        """Strong early, weak later -- the geometry has to be found before the motion pivot
        can mean anything. Mirrors Run 1's lambda_resp schedule (0.5 held 5k, decay to 0.1)."""
        if s < a.lambda_en_hold:
            return a.lambda_en
        t = (s - a.lambda_en_hold) / max(1, a.steps - a.lambda_en_hold)
        return a.lambda_en + (a.lambda_en_lo - a.lambda_en) * min(1.0, t)

    wb = None
    if not a.no_wandb:
        import wandb
        wb = wandb.init(project=a.wandb_project, name=run_name,
                        config={**vars(a), "train_windows": len(tr), "l_null": l_null,
                                "trainable_params": sum(p.numel() for p in params)})

    rng = np.random.default_rng(a.seed)
    hist_log = []
    t0 = time.time()
    for step in range(1, a.steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        opt.zero_grad(set_to_none=True)
        agg = {"loss": 0.0, "nat_a": 0.0, "nat_b": 0.0, "xling": 0.0, "en": 0.0}
        kind_count = collections.Counter()

        for _ in range(a.accum):
            idx = rng.integers(0, len(tr), size=a.batch_size)
            items = [tr[int(i)] for i in idx]
            batch = collate(items, dev)
            ta, tb, kinds = sample_views(items, langs, a.pair_mode, rng,
                                         a.paraphrase_frac, a.crosslingual_nonparallel)
            kind_count.update(kinds)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                state = save_rng(dev)

                batch["caption"] = ta
                l_nat_a, diff_a, target, valid, hlen = native_loss(model, batch)

                restore_rng(state, dev)          # SAME tau, eps, x_t, dropout mask
                batch["caption"] = tb
                l_nat_b, diff_b, *_ = native_loss(model, batch)

                w = 1.0
                if a.tau_weight:
                    ab = model.diffusion.base_alphas_cumprod.to(dev).index_select(
                        0, diff_a["timesteps"].long())
                    w = ab.mean().clamp(0.0, 1.0)
                l_x = F.mse_loss(diff_a["pred_x0"], diff_b["pred_x0"]) * w

                l_en = torch.zeros((), device=dev)
                if teacher is not None:
                    restore_rng(state, dev)
                    with torch.no_grad():
                        enc = teacher(batch["caption_en"], has_text=batch["has_text"], device=dev)
                        conds_en = model._conditions(batch)
                        conds_en["text_context"], conds_en["text_mask"] = enc["context"], enc["mask"]
                        d_en = model.diffusion.training_losses(
                            model.denoiser, target, conds_en, valid, history_len=hlen)
                    ref = d_en["pred_x0"].detach()
                    # anchor BOTH branches, so neither language is privileged over the other
                    l_en = 0.5 * (F.mse_loss(diff_a["pred_x0"], ref)
                                  + F.mse_loss(diff_b["pred_x0"], ref))

                loss = (0.5 * (l_nat_a + l_nat_b)
                        + lam_x(step) * l_x
                        + lam_en(step) * l_en) / a.accum
            loss.backward()

            agg["loss"] += float(loss)
            agg["nat_a"] += float(l_nat_a) / a.accum
            agg["nat_b"] += float(l_nat_b) / a.accum
            agg["xling"] += float(l_x) / a.accum
            agg["en"] += float(l_en) / a.accum

        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

        if step % a.log_every == 0 or step == 1:
            tot_k = max(sum(kind_count.values()), 1)
            sps = (time.time() - t0) / step
            # per-tensor gradient norms: if the adapter stops moving, this shows WHICH part
            gnorms = {f"gnorm/{n.split('adapter.')[-1]}": float(p.grad.norm())
                      for n, p in model.named_parameters()
                      if p.requires_grad and p.grad is not None}
            row = {"step": step, **agg, "lambda_x": lam_x(step), "lambda_en": lam_en(step),
                   "lr": lr_at(step),
                   "alpha": float(model.text_encoder.adapter.alpha),
                   "grad_norm": float(gn), "sec_per_step": sps,
                   "throughput/windows_per_s": a.batch_size * a.accum / max(sps, 1e-9),
                   "throughput/eta_hours": (a.steps - step) * sps / 3600.0,
                   "gpu/mem_alloc_gb": torch.cuda.memory_allocated() / 1e9,
                   "gpu/mem_reserved_gb": torch.cuda.memory_reserved() / 1e9,
                   "gpu/max_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
                   **gnorms,
                   **{f"frac_{k}": kind_count[k] / tot_k for k in
                      ("para", "cross_nonparallel", "cross_parallel", "degenerate")}}
            hist_log.append(row)
            mix = " ".join(f"{k[:5]}={kind_count[k]/tot_k:.2f}" for k in
                           ("para", "cross_nonparallel", "cross_parallel", "degenerate")
                           if kind_count[k])
            print(f"  {step:6d} loss={row['loss']:.5f} nat={0.5*(row['nat_a']+row['nat_b']):.5f} "
                  f"xling={row['xling']:.5f} en={row['en']:.5f} a={row['alpha']:+.4f} "
                  f"gn={row['grad_norm']:.2f} {row['sec_per_step']:.2f}s/step | {mix}", flush=True)
            if wb:
                wb.log(row, step=step)

        if step % a.val_every == 0 or step == a.steps:
            model.text_encoder.adapter.eval()
            vm = {}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                # (1) fixed-seed native + caption-swap sensitivity, per language
                vm.update(fixed_seed_validation(model, va, vidx, langs, a.batch_size, dev, vseeds))
                # (2) text-induced variance -- collapse early-warning, never optimised
                for lg in langs:
                    vm[f"val/tiv_{lg}"] = text_induced_variance(
                        model, va, vidx[:512], lg, a.batch_size, dev)
                # (3) where in the diffusion schedule the error sits
                if step % (a.val_every * 4) == 0 or step == a.steps:
                    vm.update(timestep_loss_profile(
                        model, va, vidx[:512], langs[0], a.batch_size, dev))
            model.text_encoder.adapter.train()

            # anchor every rel against the two reference points from the L_null gate
            for lg in langs:
                r = vm.get(f"val/rel_{lg}")
                if r is not None and l_null > 0:
                    vm[f"val/rel_{lg}_vs_null"] = float(
                        (vm[f"val/native_{lg}"] - l_null) / max(l_null, 1e-9))
            vm["val/l_null_reference"] = l_null

            rels = {lg: vm.get(f"val/rel_{lg}", float("nan")) for lg in langs}
            tivs = {lg: vm.get(f"val/tiv_{lg}", float("nan")) for lg in langs}
            print(f"  [val] step {step}: " +
                  "  ".join(f"{lg}: native={vm.get(f'val/native_{lg}', 0):.5f} "
                            f"rel={100*rels[lg]:+.2f}% tiv={tivs[lg]:.2e}" for lg in langs),
                  flush=True)
            if wb:
                wb.log(vm, step=step)
            if any(v < 1e-5 for v in tivs.values() if v == v):
                print("  !! WARNING text-induced variance near zero -- COLLAPSE SIGNATURE. "
                      "Lower --lambda-x or raise --lambda-en.", flush=True)
            if any(r < 0.02 for r in rels.values() if r == r):
                print("  !! WARNING caption-swap sensitivity below the step-2000 level of "
                      "Run 1 (+2.0%). The adapter may be ignoring the caption.", flush=True)

        if step % a.ckpt_every == 0 or step == a.steps:
            p = out / f"adapter_step{step:06d}.pt"
            model.text_encoder.save_adapter(
                str(p), step=step, args=vars(a), history=hist_log[-50:], l_null=l_null,
                omg_ckpt=os.path.basename(a.ckpt),
                contract={"self_qk": a.self_qk, "cross_qk": a.cross_qk})
            (out / "history.json").write_text(json.dumps(hist_log, indent=1))
            try:
                from huggingface_hub import HfApi
                api = HfApi(token=open("/root/.creds/hf_token").read().strip())
                api.create_repo(a.hf_repo, exist_ok=True, private=True)
                for f in (p, out / "history.json", out / "l_null.json"):
                    api.upload_file(path_or_fileobj=str(f),
                                    path_in_repo=f"{run_name}/{f.name}", repo_id=a.hf_repo)
                print(f"  pushed {p.name} -> {a.hf_repo}/{run_name}", flush=True)
            except Exception as e:
                print(f"  !! HF push failed: {str(e)[:160]}", flush=True)

    print(f"\ndone in {(time.time()-t0)/60:.1f} min -> {out}")
    if wb:
        wb.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
