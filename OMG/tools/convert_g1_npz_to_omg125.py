#!/usr/bin/env python3
"""Convert the G1-retargeted AMASS corpus (.npz) into OMG's 125-D windowed representation.

INPUT  : OMG/splits_amass_g1_13k/{train,val,test,reserve}.csv  (built by build_amass_g1_splits.py)
         + per-SOURCE .npz files with keys base_frame_pos (T,3), base_frame_wxyz (T,4 wxyz),
           joint_angles (T,29)  ==  qpos_36 by construction.
OUTPUT : shard_*.npz (features) + manifest.parquet + _meta.json, the exact contract
         maila_train.py's WindowDataset already reads.

THREE STRUCTURAL FACTS, VERIFIED FROM THE CORPUS -- do not "simplify" past them
------------------------------------------------------------------------------
1. .npz files are per SOURCE MOTION, not per clip: 10,480 files serve 13,282 rows.
   Download/decode each file ONCE and cut every clip that references it.
2. 6,023 of 13,282 clips have start_s > 0. A clip is the frame range
   [start_s, end_s) INSIDE its source file. Ignoring this silently trains on the
   wrong motion for 45% of the corpus.
3. Clips from one source OVERLAP (e.g. 0.40-10.40 and 7.05-17.05 share 3.35 s).
   This is exactly why the splits are drawn over super-groups; the converter must
   never regroup them.

KNOWN CORPUS DEFECT: TRUNCATION (quantified, accepted, recorded)
-----------------------------------------------------------------
The G1 .npz exports are truncated: 32.2% of sampled sources contain exactly 225 frames,
while the CSV's clip boundaries were computed against the full AMASS timeline. CMU 15_05
defines a clip at 151.80-161.80 s against a file holding 15.00 s -- the SMPL source has
5,737 frames (191.2 s); the export kept 225.

The SMPL source (AdiShingote/amass-smpl-dataset) is INTACT: 13,879 motions across 17
subsets, 59.4% exceeding 225 frames, KIT reaching 13,182 (263.6 s). So the cap was
introduced by the retargeting export, not by AMASS.

Measured consequence of converting anyway (n=180 clips, per-subset rates + clamping):
    intact 53.9% | clamped-rounding 12.2% | clamped-real-loss 20.0%
    dropped: too short after clamp 5.0%, start past end 8.9%
    => 86.1% USABLE, 13.9% lost; of the clamped-with-loss clips, median 25% truncated.

This is ACCEPTED rather than fixed: re-exporting would recover it but costs a full
retargeting run, and the frame rates -- the part that would have been fatal -- turned out
to be recoverable without it (see SUBSET_FPS). The corpus therefore skews slightly SHORT,
because truncation removes long motions. State that in the paper's limitations; do not
pretend it is not there. Every clamp is recorded in _meta.json's truncation ledger.

CLAMPING BEHAVIOUR
------------------
Every clip is CLAMPED to the file end and the per-clip shortfall recorded in _meta.json.
Clips left shorter than one window are dropped and counted -- never silently skipped.

Resampling is linear on position/joints and sign-corrected normalized-lerp on the quaternion.

PARITY BY CONSTRUCTION
----------------------
Encoding calls OMG's OWN `model.representation` (kinematics + codec), never a
reimplementation -- the same code path the frozen checkpoint was trained against.
The first converted clip is round-tripped and the run ABORTS if error > 1e-3.
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import os
import sys
import urllib.parse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

QPOS_DIM = 36          # [root_pos(3) | root_quat_wxyz(4) | joint_dof(29)]
FEAT_DIM = 125
N_JOINTS = 29
TARGET_FPS = 30.0
LANGS = ["hi", "bn", "ta", "te"]
SLOTS = [1, 2, 3, 4]

# EXPORT FRAME RATE OF THE .npz, PER SUBSET -- measured 2026-09-05, all 16 verified.
#
# Two things are easy to conflate here, and conflating them costs you the experiment:
#   SOURCE rate  : what AMASS captured. From `motion_dt` in the SMPL .pt files -> 30 or 50.
#   EXPORT rate  : what the G1 .npz actually contains. The retargeting DOWNSAMPLED some
#                  subsets by 2x and passed others through unchanged. This is the rate the
#                  converter needs, and it is NOT derivable from the source rate.
#
# Measured two independent ways, agreeing wherever both applied:
#   (a) frame-count ratio  npz_frames / src_frames on UNTRUNCATED files, matched on the
#       FULL relative path (matching on basename alone gives 2,090 false pairs in KIT,
#       which is what made KIT look inconsistent at first).
#   (b) content correlation of the root-height trajectory against the source at ratio 1.0
#       vs 0.5 -- works on TRUNCATED files, so it covers the four subsets where every
#       sampled file hit the 225-frame cap.
#   Cross-validation: BMLmovi (a)=1.0000 std 0.0000 / (b) corr 0.951 vs 0.028 -> 1.0
#                     CMU     (a)=0.5014 std 0.0018 / (b) corr -0.044 vs 0.933 -> 0.5
#
# ACCAD and the two BML sets pass through at source rate; everything else is halved.
# That is not a rule anyone could guess -- do not "simplify" this table.
SUBSET_FPS: dict[str, float] = {
    # export == source (ratio 1.0)
    "ACCAD": 30.0,                 # (b) corr 0.988 vs 0.079
    "BMLhandball": 30.0,           # (a) ratio 1.0000, std 0.0000
    "BMLmovi": 30.0,               # (a) ratio 1.0000, std 0.0000 + (b) 0.951 vs 0.028
    # export == source / 2 (ratio 0.5)
    "BioMotionLab_NTroje": 15.0,   # (a) 0.5041 std 0.152 (noisy) -> (b) -0.013 vs 0.975
    "CMU": 15.0,                   # (a) 0.5014 std 0.0018 + (b) -0.044 vs 0.933
    "DFaust_67": 15.0,             # (a) 0.5018 std 0.0020
    "Eyes_Japan_Dataset": 15.0,    # (b) corr 0.156 vs 0.961
    "HumanEva": 15.0,              # (a) 0.5000 std 0.0004
    "MPI_HDM05": 15.0,             # (b) corr 0.176 vs 0.953
    "MPI_mosh": 15.0,              # (a) 0.5013 std 0.0011
    "SFU": 15.0,                   # (a) 0.5000 std 0.0018
    "SSM_synced": 15.0,            # (a) 0.5010 std 0.0036
    "TotalCapture": 15.0,          # (b) corr 0.196 vs 0.865
    "Transitions_mocap": 15.0,     # (a) 0.5019 std 0.0011
    "EKUT": 25.0,                  # (a) 0.5000 std 0.0012  (source 50)
    "KIT": 25.0,                   # (a) 0.5006 std 0.0009  (source 50)
}


def subset_of(npz_url: str) -> str:
    """Top-level folder in the HF repo == the AMASS subset name."""
    tail = urllib.parse.unquote(str(npz_url)).split("/resolve/main/", 1)[-1]
    return tail.split("/", 1)[0]


# ------------------------------------------------------------------ OMG representation
def load_omg_representation(exp: str, t5_path: str, device: str = "cpu"):
    """Instantiate OMG's real codec+kinematics from the same hydra config used to generate."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from hydra.utils import instantiate
    from omg.core.paths import resolve_repo_path

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(resolve_repo_path("configs/generation")),
                               version_base="1.3"):
        cfg = compose(config_name="train", overrides=[
            f"exp={exp}", "data=omg_data_lerobot", "logger=none", "trainer=1gpu",
            f"model.text_encoder.model_name={t5_path}",
        ])
    model = instantiate(cfg.model)
    return model.representation.to(device).eval()


# ------------------------------------------------------------------ npz access
def npz_local_name(npz_url: str) -> str:
    """Stable cache filename for a (URL-encoded) npz path."""
    return urllib.parse.unquote(str(npz_url)).replace("\\", "/").rsplit("/", 1)[-1]


def fetch_npz(npz_url: str, cache: Path, npz_dir: Path | None, token: str | None) -> Path | None:
    name = npz_local_name(npz_url)
    if npz_dir is not None:
        hit = npz_dir / name
        if hit.is_file():
            return hit
    dst = cache / name
    if dst.is_file() and dst.stat().st_size > 0:
        return dst
    try:
        import requests
    except ImportError as exc:
        raise SystemExit("pip install requests, or pass --npz-dir with local files") from exc
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    r = requests.get(str(npz_url), headers=headers, timeout=120)
    if r.status_code != 200:
        return None
    dst.write_bytes(r.content)
    return dst


def load_qpos36(path: Path) -> np.ndarray | None:
    """base_frame_pos(T,3) + base_frame_wxyz(T,4) + joint_angles(T,29) -> (T,36) float32."""
    try:
        d = np.load(path, allow_pickle=False)
    except Exception:
        return None
    need = ("base_frame_pos", "base_frame_wxyz", "joint_angles")
    if any(k not in d for k in need):
        return None
    p, q, j = d["base_frame_pos"], d["base_frame_wxyz"], d["joint_angles"]
    if p.ndim != 2 or p.shape[1] != 3 or q.shape[1] != 4 or j.shape[1] != N_JOINTS:
        return None
    T = min(len(p), len(q), len(j))
    return np.concatenate([p[:T], q[:T], j[:T]], axis=1).astype(np.float32)


# ------------------------------------------------------------------ resampling
def resample_qpos(qpos: np.ndarray, src_fps: float, dst_fps: float) -> np.ndarray:
    """Linear on pos/joints; sign-corrected normalized-lerp on the wxyz quaternion."""
    if abs(src_fps - dst_fps) < 1e-6:
        return qpos
    T = qpos.shape[0]
    n_out = max(int(round(T * dst_fps / src_fps)), 1)
    t_src = np.arange(T, dtype=np.float64)
    t_dst = np.linspace(0.0, T - 1, n_out, dtype=np.float64)

    out = np.empty((n_out, QPOS_DIM), dtype=np.float32)
    for c in list(range(0, 3)) + list(range(7, 36)):          # position + joints
        out[:, c] = np.interp(t_dst, t_src, qpos[:, c])

    quat = qpos[:, 3:7].astype(np.float64).copy()
    # hemisphere-align consecutive quaternions so the lerp takes the short path
    for i in range(1, T):
        if float(quat[i - 1] @ quat[i]) < 0.0:
            quat[i] = -quat[i]
    lo = np.clip(np.floor(t_dst).astype(int), 0, T - 1)
    hi = np.clip(lo + 1, 0, T - 1)
    w = (t_dst - lo)[:, None]
    q = quat[lo] * (1.0 - w) + quat[hi] * w
    n = np.linalg.norm(q, axis=1, keepdims=True)
    out[:, 3:7] = (q / np.maximum(n, 1e-8)).astype(np.float32)
    return out


# ------------------------------------------------------------------ encoding
def encode_windows(rep, qpos: np.ndarray, starts: list[int], L: int, H: int,
                   device: str = "cpu") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(N,L+H,125) canonicalized about the last history frame, plus the anchor pose."""
    win = L + H
    batch = np.stack([qpos[s:s + win] for s in starts])
    q = torch.from_numpy(batch).to(device)
    fk = rep.kinematics.forward_kinematics(q)
    comps = rep.codec.canonicalize(
        qpos_36=q,
        body_pos_w=fk["body_pos_w"],
        body_quat_w=fk["body_quat_w"],
        anchor_root_pos=q[:, L - 1, 0:3],        # anchor = LAST history frame
        anchor_root_quat=q[:, L - 1, 3:7],
        fps=torch.tensor([TARGET_FPS] * q.shape[0], device=device),
    )
    feats = rep.codec.assemble_features(comps).cpu().numpy().astype(np.float32)
    return feats, batch[:, L - 1, 0:3], batch[:, L - 1, 3:7]


def verify_roundtrip(rep, qpos: np.ndarray, L: int, H: int, device: str = "cpu") -> dict:
    """Encode then decode through OMG's OWN codec. Quaternion compared sign-invariantly."""
    feats, ap, aq = encode_windows(rep, qpos, [0], L, H, device)
    with torch.no_grad():
        comps = rep.codec.split_features(torch.from_numpy(feats).to(device))
        back = rep.codec.decode_to_world_qpos36(
            comps, torch.from_numpy(ap).to(device), torch.from_numpy(aq).to(device),
        ).cpu().numpy()[0]
    ref = qpos[:L + H]
    q_err = np.minimum(np.abs(back[:, 3:7] - ref[:, 3:7]),
                       np.abs(back[:, 3:7] + ref[:, 3:7])).max()
    return {"root_pos": float(np.abs(back[:, 0:3] - ref[:, 0:3]).max()),
            "root_quat": float(q_err),
            "joints": float(np.abs(back[:, 7:36] - ref[:, 7:36]).max())}


# ------------------------------------------------------------------ inspect
def mode_inspect(a) -> int:
    df = pd.read_csv(a.manifest, low_memory=False)
    cache = Path(a.cache_dir); cache.mkdir(parents=True, exist_ok=True)
    npz_dir = Path(a.npz_dir) if a.npz_dir else None
    token = a.hf_token or os.environ.get("HF_TOKEN")

    # sources whose clip starts at 0 give the cleanest fps estimate
    df["_start"] = pd.to_numeric(df["start_s"], errors="coerce")
    df["_end"] = pd.to_numeric(df["end_s"], errors="coerce")
    cand = df.drop_duplicates("npz_path").head(a.inspect_n)

    print(f"inspecting {len(cand)} source files from {a.manifest}\n")
    rates, shown = [], 0
    for _, r in cand.iterrows():
        p = fetch_npz(r["npz_path"], cache, npz_dir, token)
        if p is None:
            print(f"  MISS  {npz_local_name(r['npz_path'])[:60]}")
            continue
        q = load_qpos36(p)
        if q is None:
            print(f"  BAD   {npz_local_name(r['npz_path'])[:60]}")
            continue
        # implied rate: the source must be at least as long as the largest end_s
        same = df[df["npz_path"] == r["npz_path"]]
        max_end = float(same["_end"].max())
        implied = q.shape[0] / max_end if max_end > 0 else float("nan")
        rates.append(implied)
        if shown < 8:
            print(f"  T={q.shape[0]:5d}  max_end={max_end:6.2f}s  implied_fps>={implied:6.2f}  "
                  f"{npz_local_name(r['npz_path'])[:44]}")
            shown += 1

    if not rates:
        raise SystemExit("no usable npz files reached -- check --npz-dir / --hf-token")

    rates = np.array(rates)
    print(f"\n  implied fps (lower bound): median {np.median(rates):.3f}  "
          f"mean {rates.mean():.3f}  p05 {np.percentile(rates,5):.3f}  p95 {np.percentile(rates,95):.3f}")
    for guess in (20.0, 30.0, 60.0, 120.0):
        frac = float((np.abs(rates - guess) < 0.75).mean())
        flag = "  <-- LIKELY" if frac > 0.8 else ""
        print(f"    consistent with {guess:6.1f} fps : {100*frac:5.1f}%{flag}")

    q0 = load_qpos36(fetch_npz(cand.iloc[0]["npz_path"], cache, npz_dir, token))
    print("\n  qpos_36 sanity (assembled from the three npz keys):")
    print(f"    root_pos   min {q0[:,0:3].min():+.3f} max {q0[:,0:3].max():+.3f}   "
          f"(metres expected; ~x100 would mean centimetres)")
    print(f"    root_quat  norm mean {np.linalg.norm(q0[:,3:7],axis=1).mean():.5f}  (1.0 expected)")
    print(f"    joints     min {q0[:,7:36].min():+.3f} max {q0[:,7:36].max():+.3f}   "
          f"(radians expected; >6.3 would mean degrees)")
    print("\nRerun with --mode convert --src-fps <the rate confirmed above>.")
    return 0


# ------------------------------------------------------------------ convert
def mode_convert(a) -> int:
    df = pd.read_csv(a.manifest, low_memory=False)
    df["_subset"] = df["npz_path"].map(subset_of)
    fps_table = dict(SUBSET_FPS)
    if a.fps_override:
        for kv in a.fps_override.split(","):
            k, _, v = kv.partition("=")
            fps_table[k.strip()] = float(v)
    unknown = sorted(set(df["_subset"]) - set(fps_table))
    if unknown:
        counts = df["_subset"].value_counts()
        raise SystemExit(
            "REFUSING TO CONVERT: no measured source fps for subset(s) "
            + ", ".join(f"{u} ({counts[u]} clips)" for u in unknown)
            + "\n  Run --mode inspect on a manifest containing them, then add the rate via "
              "--fps-override SUBSET=RATE (comma separated). Never assume 30."
        )
    print("  per-subset source fps in use:")
    for s in sorted(set(df["_subset"])):
        print(f"     {s:24s} {fps_table[s]:6.2f}  ({int((df['_subset']==s).sum()):,} clips)")
    out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
    cache = Path(a.cache_dir); cache.mkdir(parents=True, exist_ok=True)
    npz_dir = Path(a.npz_dir) if a.npz_dir else None
    token = a.hf_token or os.environ.get("HF_TOKEN")
    win = a.hist + a.horizon

    print(f"loading OMG representation (exp={a.exp}) ...")
    rep = load_omg_representation(a.exp, a.t5, a.device)

    df["_start"] = pd.to_numeric(df["start_s"], errors="coerce").fillna(0.0)
    df["_end"] = pd.to_numeric(df["end_s"], errors="coerce")
    by_src = collections.defaultdict(list)
    for i, r in df.iterrows():
        by_src[r["npz_path"]].append(i)
    print(f"{len(df):,} clips from {len(by_src):,} source files "
          f"(each downloaded/decoded once)")

    rng = np.random.default_rng(a.seed)
    feats_all, meta, verified = [], [], None
    truncation: list[dict] = []
    shard = n_written = 0
    stats = collections.Counter()

    for si, (npz_url, rows) in enumerate(by_src.items()):
        p = fetch_npz(npz_url, cache, npz_dir, token)
        if p is None:
            stats["missing_npz"] += len(rows); continue
        q_src = load_qpos36(p)
        if q_src is None:
            stats["bad_npz"] += len(rows); continue
        src_fps = fps_table[df.loc[rows[0], "_subset"]]
        T_src = q_src.shape[0]
        q_all = resample_qpos(q_src, src_fps, TARGET_FPS)

        for i in rows:
            r = df.loc[i]
            # Slice at the SOURCE rate first so the clamp is measured against real frames,
            # then map to the resampled index space.
            e_src = int(round(float(r["_end"]) * src_fps))
            s_src = int(round(float(r["_start"]) * src_fps))
            shortfall = max(0, e_src - T_src)          # frames the CSV claims but the file lacks
            if shortfall:
                stats["clamped"] += 1
                truncation.append({"clip_id": int(r["clip_id"]), "subset": r["_subset"],
                                   "asked": e_src, "have": T_src, "short": int(shortfall),
                                   "frac_lost": round(shortfall / max(e_src - s_src, 1), 4)})
            if s_src >= T_src:
                stats["start_past_end"] += 1; continue

            scale = TARGET_FPS / src_fps
            s0 = int(round(s_src * scale))
            s1 = int(round(min(e_src, T_src) * scale))
            q = q_all[max(s0, 0):min(s1, len(q_all))]
            if q.shape[0] < win:
                stats["short"] += 1; continue
            if not np.isfinite(q).all():
                stats["nonfinite"] += 1; continue
            h = q[:, 2]                                    # pelvis height guard (metres)
            if not (0.02 < float(np.median(h)) < 3.0):
                stats["bad_height"] += 1; continue

            if verified is None:                           # gate: parity before bulk work
                verified = verify_roundtrip(rep, q, a.hist, a.horizon, a.device)
                print(f"  round-trip: {verified}")
                if max(verified.values()) > a.roundtrip_tol:
                    raise SystemExit(f"ROUND-TRIP FAILED (> {a.roundtrip_tol}): {verified}")

            n_win = min(a.windows_per_clip, (q.shape[0] - win) // a.stride + 1)
            starts = sorted(rng.choice(np.arange(0, q.shape[0] - win + 1, a.stride),
                                       size=n_win, replace=False).tolist())
            f, ap, aq = encode_windows(rep, q, starts, a.hist, a.horizon, a.device)
            for k, s in enumerate(starts):
                feats_all.append(f[k])
                m = {
                    "clip_id": int(r["clip_id"]), "source_amass": str(r["source_amass"]),
                    "npz": npz_local_name(npz_url), "window_start": int(s),
                    "split": str(r.get("split", a.split_name)),
                    "super_group": str(r.get("super_group", "")),
                    "subset": str(r["source_amass"]).split("/")[0],
                    "anchor_root_pos": ap[k].tolist(), "anchor_root_quat": aq[k].tolist(),
                }
                # slot-1 aliases keep three_arm_benchmark.py / WindowDataset working unchanged
                m["en"] = str(r["caption_1"])
                for lg in LANGS:
                    m[lg] = str(r[f"caption_1_{lg}"])
                # every caption slot, for the symmetric/paraphrase objective
                for sl in SLOTS:
                    v = r.get(f"caption_{sl}")
                    m[f"en_{sl}"] = "" if pd.isna(v) else str(v)
                    for lg in LANGS:
                        v2 = r.get(f"caption_{sl}_{lg}")
                        m[f"{lg}_{sl}"] = "" if pd.isna(v2) else str(v2)
                for lg in ["en"] + LANGS:
                    m[f"n_caps_{lg}"] = int(sum(bool(m[f"{lg}_{sl}"].strip()) for sl in SLOTS))
                meta.append(m)
            stats["ok"] += 1

            if len(feats_all) >= a.shard_size:
                np.savez_compressed(out_dir / f"shard_{shard:04d}.npz",
                                    features=np.stack(feats_all).astype(np.float32))
                shard += 1; n_written += len(feats_all); feats_all = []

        if (si + 1) % 500 == 0:
            print(f"  [{si+1:6,}/{len(by_src):,} sources] ok={stats['ok']:,} "
                  f"windows={n_written+len(feats_all):,}", flush=True)

    if feats_all:
        np.savez_compressed(out_dir / f"shard_{shard:04d}.npz",
                            features=np.stack(feats_all).astype(np.float32))
        n_written += len(feats_all)

    man = pd.DataFrame(meta)
    man.to_parquet(out_dir / "manifest.parquet", index=False)
    (out_dir / "_meta.json").write_text(json.dumps({
        "dim": FEAT_DIM, "fps": TARGET_FPS, "L": a.hist, "H": a.horizon,
        "windows": int(n_written), "clips": dict(stats),
        "subset_fps": {k: fps_table[k] for k in sorted(set(df["_subset"]))},
        "truncation": {"clips_clamped": len(truncation),
                       "worst": sorted(truncation, key=lambda d: -d["frac_lost"])[:25]},
        "roundtrip_err": verified, "exp": a.exp,
        "languages": LANGS, "caption_slots": SLOTS,
        "source_manifest": str(a.manifest),
        "encoder": "OMG model.representation (codec+kinematics) -- parity by construction",
        "note": "npz are per-SOURCE; clips sliced by [start_s,end_s] then windowed",
    }, indent=2), encoding="utf-8")

    print(f"\n{n_written:,} windows from {stats['ok']:,} clips -> {out_dir}")
    print(f"  {dict(stats)}")
    if len(man):
        print(f"  distinct hi_1 captions: {man['hi_1'].nunique():,} over {len(man):,} windows "
              f"({len(man)/max(man['hi_1'].nunique(),1):.2f} windows/caption)")
        print(f"  windows with >=2 hindi captions: "
              f"{int((man['n_caps_hi']>=2).sum()):,} ({100*(man['n_caps_hi']>=2).mean():.1f}%)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["inspect", "convert"], required=True)
    ap.add_argument("--manifest", required=True, help="a split CSV from splits_amass_g1_13k/")
    ap.add_argument("--out", help="output shard directory (convert mode)")
    ap.add_argument("--split-name", default="train")
    ap.add_argument("--npz-dir", default=None, help="local directory of .npz, checked first")
    ap.add_argument("--cache-dir", default="/workspace/cache/g1_npz")
    ap.add_argument("--hf-token", default=None, help="or set HF_TOKEN")
    ap.add_argument("--fps-override", default=None,
                    help="add/replace measured subset rates, e.g. "
                         "'MPI_HDM05=25,SFU=15'. Subsets with no rate are REFUSED")
    ap.add_argument("--hist", type=int, default=10)
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--stride", type=int, default=30)
    ap.add_argument("--windows-per-clip", type=int, default=2)
    ap.add_argument("--shard-size", type=int, default=4096)
    ap.add_argument("--roundtrip-tol", type=float, default=1e-3)
    ap.add_argument("--inspect-n", type=int, default=40)
    ap.add_argument("--exp", default="100m")
    ap.add_argument("--t5", default="/workspace/models/t5-base-local")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    if a.mode == "inspect":
        return mode_inspect(a)
    if not a.out:
        raise SystemExit("--out is required for convert")
    return mode_convert(a)


if __name__ == "__main__":
    raise SystemExit(main())
