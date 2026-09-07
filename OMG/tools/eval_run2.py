#!/usr/bin/env python3
"""Comprehensive Run 2 evaluation -- everything Run 1's plan required but never reported.

WHAT RUN 1 ACTUALLY REPORTED
    L_correct / L_wrong / rel, per checkpoint, at one CFG and one seed. Nothing else.

WHAT SSOT s9.4 REQUIRED AND RUN 1 DID NOT DELIVER, and so is here:
    * physical plausibility of GENERATED motion -- foot sliding (contact slide), jerk,
      joint-limit violations, joint jumps, ground penetration, fall rate;
    * fidelity against the ground-truth clip -- MPJPE, global MPJPE, velocity and
      acceleration error;
    * a CFG sweep, instead of assuming 2.5;
    * more than one sampling seed, so a difference between arms can be told from noise;
    * conditioning metrics for EVERY checkpoint, not only the last.

Physical and fidelity metrics deliberately call OMG's OWN implementations
(omg.benchmarks.metrics.*, omg.benchmarks.runners.h2h_retarget) rather than re-deriving the
definitions, so the numbers mean the same thing as the foundation model's own benchmark.

Retrieval / FID / diversity are NOT duplicated here: tools/three_arm_benchmark.py already
does them, including the protocol that makes a Hindi-input arm scoreable at all (generate
from Hindi, retrieve against the ENGLISH caption, so the scoring text is never the text that
drove generation).

PROTOCOL
    Checkpoint selection and the CFG sweep run on VAL. Only the selected configuration is
    scored on TEST, once. TEST is never used to choose anything.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from pathlib import Path

import numpy as np
import torch


# ------------------------------------------------------------------ conditioning
@torch.no_grad()
def conditioning(model, ds, idx, lang, dev, bs, seed, native_loss, collate):
    """L_correct / L_wrong / rel / tiv with replayed per-batch seeds.

    ONE DISTINCT SEED PER BATCH. An earlier version of this file cycled four fixed seeds
    across all 32 batches, so only 4 distinct (tau, epsilon) draws were ever sampled --
    128 noise realisations instead of 1024. Those four happened to land on low-noise
    timesteps, which pulled L_correct from 0.03317 down to 0.02483 and `rel` from +7.42%
    to +2.59% on the SAME checkpoint. Precision was not involved: bf16 autocast and fp32
    agree to 0.07 percentage points.

    TWO WRONG-CAPTION PROTOCOLS, because they do not measure the same thing:
      rel_roll  the wrong caption is the next clip in the batch (what Run 1 used, so this
                is the number comparable to Run 1s +11.92%). It UNDERSTATES sensitivity:
                a clip sometimes lands beside a near-duplicate caption, and machine
                translation collapses distinct English into identical Indic strings, so
                L_wrong falls onto L_correct for reasons that have nothing to do with the
                adapter.
      rel_draw  the wrong caption is drawn from the whole split. Measured +11.13% against
                rel_roll +7.42% on the same checkpoint.
    """
    nb = (len(idx) + bs - 1) // bs
    seeds = np.random.default_rng(seed + 1).integers(0, 2 ** 31, size=nb).tolist()
    pool_rng = np.random.default_rng(seed + 2)
    tc = tw_roll = tw_draw = 0.0
    n = 0
    tiv = []
    for bi, s0 in enumerate(range(0, len(idx), bs)):
        sel = idx[s0:s0 + bs]
        if len(sel) < 2:
            continue
        b = collate([ds[int(i)] for i in sel], dev)
        caps = [ds.caps[lang][int(i)][0] if ds.caps[lang][int(i)] else "" for i in sel]
        pool = pool_rng.choice(len(ds), size=len(sel), replace=False)
        drawn = [ds.caps[lang][int(j)][0] if ds.caps[lang][int(j)] else "" for j in pool]
        sd = int(seeds[bi])
        for caption, bucket in ((caps, "c"), (caps[1:] + caps[:1], "r"), (drawn, "d")):
            torch.manual_seed(sd)
            torch.cuda.manual_seed_all(sd)
            b["caption"] = caption
            v, *_ = native_loss(model, b)
            if bucket == "c":
                tc += float(v) * len(sel)
            elif bucket == "r":
                tw_roll += float(v) * len(sel)
            else:
                tw_draw += float(v) * len(sel)
        n += len(sel)
        o = model.text_encoder(caps, device=dev)
        m = o["mask"].float().unsqueeze(-1)
        c = (o["context"].float() * m).sum(1) / m.sum(1).clamp_min(1)
        tiv.append(float(c.var(dim=0).mean()))
    if not n:
        return {}
    Lc, Lr, Ld = tc / n, tw_roll / n, tw_draw / n
    return {"L_correct": Lc, "L_wrong_roll": Lr, "L_wrong_draw": Ld,
            "rel": (Lr - Lc) / max(Lc, 1e-9),
            "rel_draw": (Ld - Lc) / max(Lc, 1e-9),
            "tiv": float(np.mean(tiv)), "n": n, "n_batch_seeds": nb}


# ------------------------------------------------------------------ physical + fidelity
def physical_and_fidelity(gen_norm, ref_norm, rep, kin, foot_ids, fps):
    """OMG's own metric definitions, applied to generated vs ground-truth motion."""
    from omg.benchmarks.metrics.tracking import g_mpjpe, mpjpe, e_vel, e_acc
    from omg.benchmarks.runners.h2h_retarget import (
        compute_foot_sliding, compute_ground_penetration, compute_joint_jump_rate,
        compute_joint_limit_rates,
    )

    dt = 1.0 / float(fps)
    dev = gen_norm.device
    bsz = gen_norm.shape[0]
    rp = rep.default_root_pos.to(dev).view(1, 3).expand(bsz, -1)
    rq = rep.default_root_quat.to(dev).view(1, 4).expand(bsz, -1)

    def to_world(xn):
        d = rep.decode(xn)
        q = rep.compose_qpos_36(d, rp, rq)                  # (B, T, 36)
        return q, kin.forward_body_positions(q)             # (B, T, nbody, 3)

    gq, gp = to_world(gen_norm)
    _, rpos = to_world(ref_norm)
    lo = kin.joint_lower_limits.detach().cpu().numpy()
    hi = kin.joint_upper_limits.detach().cpu().numpy()

    acc: dict[str, list[float]] = {}

    def add(d):
        for k, v in d.items():
            if v is not None and math.isfinite(float(v)):
                acc.setdefault(k, []).append(float(v))

    for i in range(bsz):
        qpos = gq[i].float().cpu().numpy()
        bp = gp[i].float().cpu().numpy()
        joints = qpos[:, 7:]
        add(compute_joint_limit_rates(joints, lo, hi))
        add({"joint_jump_rate": compute_joint_jump_rate(joints)})
        add(compute_ground_penetration(bp, body_ids=list(foot_ids)))
        add(compute_foot_sliding(bp, foot_ids, dt))
        # jerk: third time-difference of world body positions, L1-averaged
        add({"body_jerk": float(np.abs(np.diff(bp, n=3, axis=0)).mean() / dt ** 3)})
        # fall: pelvis (body 0) dropping below a standing-height floor at any frame
        add({"fall_rate": float(bp[:, 0, 2].min() < 0.35)})
        rb = rpos[i].float().cpu().numpy()
        add({"mpjpe_mm": mpjpe(bp, rb), "g_mpjpe_mm": g_mpjpe(bp, rb),
             "e_vel": e_vel(bp, rb), "e_acc": e_acc(bp, rb)})
    return {k: float(np.mean(v)) for k, v in acc.items()}


@torch.no_grad()
def generate(model, ds, idx, lang, dev, bs, cfg, seed, num_frames, collate, rep):
    """Sample motion for each caption.

    motion_generator.generate() needs canon_root_pos / canon_root_quat, which the training
    collate does not build (training never generates). Every arm is anchored at the SAME
    canonical root, so differences between arms come from the caption alone and not from a
    per-clip starting pose.
    """
    gens, refs = [], []
    for s in range(0, len(idx), bs):
        sel = idx[s:s + bs]
        b = collate([ds[int(i)] for i in sel], dev)
        n = len(sel)
        b["caption"] = [ds.caps[lang][int(i)][0] if ds.caps[lang][int(i)] else "" for i in sel]
        b["canon_root_pos"] = rep.default_root_pos.to(dev).view(1, 3).expand(n, -1).clone()
        b["canon_root_quat"] = rep.default_root_quat.to(dev).view(1, 4).expand(n, -1).clone()
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        g = model.generate(b, num_frames=num_frames, cfg_scale=cfg)
        if isinstance(g, dict):
            key = next((k for k in ("motion_features", "motion", "x", "samples")
                        if k in g and torch.is_tensor(g[k])), None)
            if key is None:
                raise SystemExit(f"generate() returned keys {sorted(g)}; none is a motion "
                                 f"tensor -- refusing to guess which output to score")
            g = g[key]
        gens.append(g.float())
        # the reference is a RAW shard too, so it needs the same treatment as `g`;
        # normalising only the prediction compared a correct sample against a broken target
        refs.append(rep.normalize_features(b["motion_features"]).float()[:, :g.shape[1]])
    return torch.cat(gens), torch.cat(refs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--runs-root", default="/workspace/runs")
    ap.add_argument("--models", default="/workspace/models")
    ap.add_argument("--calibration", default="/workspace/adapter_calibration.pt")
    ap.add_argument("--norm-mode", default="corpus")
    ap.add_argument("--lang", default="hi")
    ap.add_argument("--n-cond", type=int, default=1024)
    ap.add_argument("--n-gen", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-frames", type=int, default=60)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--cfg-sweep", default="2.0,2.5,3.0")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", default="/workspace/runs/eval_report.json")
    a = ap.parse_args()

    sys.path.insert(0, "/workspace")
    from maila_train import build
    from maila_train_motionpivot import MultiCaptionWindowDataset, collate, native_loss

    dev = torch.device("cuda")
    va = MultiCaptionWindowDataset(a.data, "val", [a.lang])
    te = MultiCaptionWindowDataset(a.data, "test", [a.lang])
    ckpt = f"{a.models}/omg/checkpoints/updated/100m/sstep=170000.ckpt"
    model = build("100m", ckpt, f"{a.models}/t5-base-local", f"{a.models}/muril-base-cased",
                  dev, True, True, 0.0, t5_calibration=a.calibration, norm_mode=a.norm_mode)

    rep = model.representation                      # G1MotionRepresentation
    kin = rep.kinematics                            # G1Kinematics (URDF joint limits + FK)
    # G1Kinematics exposes body_name_to_index, not body_names. The foot link names are the
    # ones its own sole-proxy loader uses, so contact metrics index the same bodies OMG does.
    b2i = dict(kin.body_name_to_index)
    foot_names = ["left_ankle_roll_link", "right_ankle_roll_link"]
    missing = [n for n in foot_names if n not in b2i]
    if missing:
        # A wrong foot index yields plausible-looking contact-slide numbers that mean nothing,
        # so this refuses to guess rather than silently reporting a fabricated metric.
        raise SystemExit(f"foot links {missing} absent from kinematics; refusing to guess "
                         f"indices for contact-slide metrics")
    foot_ids = tuple(int(b2i[n]) for n in foot_names)

    SEED = 0     # matches the trainer: vseeds come from default_rng(seed + 1)
    vi = np.random.default_rng(0).choice(len(va), size=min(a.n_cond, len(va)), replace=False)
    ti = np.random.default_rng(0).choice(len(te), size=min(a.n_cond, len(te)), replace=False)
    report: dict = {"protocol": "selection + CFG sweep on VAL; TEST scored once",
                    "foot_ids": list(foot_ids), "foot_bodies": foot_names,
                    "runs": {}, "errors": []}
    init_sd = {k: v.clone() for k, v in model.text_encoder.adapter.state_dict().items()}

    def load(p):
        st = torch.load(p, map_location="cpu", weights_only=False)["adapter"]
        model.text_encoder.adapter.load_state_dict(st)
        model.text_encoder.adapter.eval()

    # ---- floor: the untrained adapter at the corpus-normalised init -------------
    model.text_encoder.adapter.load_state_dict(init_sd)
    report["baselines"] = {"RANDOM_corpus_init": conditioning(
        model, te, ti, a.lang, dev, a.batch_size, SEED, native_loss, collate)}

    # ---- every checkpoint, on VAL, for selection --------------------------------
    for rn in a.run:
        rows = {}
        for p in sorted(Path(a.runs_root, rn).glob("adapter_step*.pt")):
            load(p)
            rows[p.name] = conditioning(model, va, vi, a.lang, dev, a.batch_size,
                                        SEED, native_loss, collate)
            print(f"  {rn} {p.name} val rel={rows[p.name].get('rel', 0) * 100:+.2f}%",
                  flush=True)
        if not rows:
            report["errors"].append(f"{rn}: no checkpoints found")
            continue
        best = max(rows, key=lambda k: rows[k].get("rel", -9))
        report["runs"][rn] = {"val_by_checkpoint": rows, "selected": best}
        print(f"  -> {rn} selected {best} (val rel {rows[best]['rel'] * 100:+.2f}%)", flush=True)

    # ---- CFG sweep on VAL, then TEST once at the chosen CFG ---------------------
    cfgs = [float(x) for x in a.cfg_sweep.split(",")]
    seeds = [int(x) for x in a.seeds.split(",")]
    gi_v = np.random.default_rng(5).choice(len(va), size=min(a.n_gen, len(va)), replace=False)
    gi_t = np.random.default_rng(6).choice(len(te), size=min(a.n_gen, len(te)), replace=False)

    for rn in list(report["runs"]):
        load(Path(a.runs_root, rn, report["runs"][rn]["selected"]))
        report["runs"][rn]["test_conditioning"] = conditioning(
            model, te, ti, a.lang, dev, a.batch_size, SEED, native_loss, collate)

        sweep = {}
        for c in cfgs:
            try:
                g, r = generate(model, va, gi_v, a.lang, dev, a.batch_size, c,
                                seeds[0], a.num_frames, collate, rep)
                sweep[f"cfg_{c}"] = physical_and_fidelity(g, r, rep, kin, foot_ids, a.fps)
                print(f"  {rn} cfg={c} mpjpe={sweep[f'cfg_{c}'].get('mpjpe_mm', -1):.1f}mm",
                      flush=True)
            except Exception:
                report["errors"].append(f"{rn} cfg {c}: {traceback.format_exc()[-500:]}")
        report["runs"][rn]["val_cfg_sweep"] = sweep
        if not sweep:
            continue
        best_key = min(sweep, key=lambda k: sweep[k].get("mpjpe_mm", 1e9))
        best_cfg = float(best_key.split("_")[1])
        report["runs"][rn]["selected_cfg"] = best_cfg

        per_seed = {}
        for sd in seeds:
            try:
                g, r = generate(model, te, gi_t, a.lang, dev, a.batch_size, best_cfg,
                                sd, a.num_frames, collate, rep)
                per_seed[f"seed_{sd}"] = physical_and_fidelity(g, r, rep, kin, foot_ids, a.fps)
            except Exception:
                report["errors"].append(f"{rn} test seed {sd}: {traceback.format_exc()[-500:]}")
        report["runs"][rn]["test_per_seed"] = per_seed
        if per_seed:
            keys = sorted({k for v in per_seed.values() for k in v})
            report["runs"][rn]["test_mean_std"] = {
                k: {"mean": float(np.mean([v[k] for v in per_seed.values() if k in v])),
                    "std": float(np.std([v[k] for v in per_seed.values() if k in v]))}
                for k in keys}

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out}  ({len(report['errors'])} stage errors)")
    for e in report["errors"][:3]:
        print("  ERR", e[:220])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
