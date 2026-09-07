#!/usr/bin/env python3
"""Convert BONES-SEED G1 CSV motions into OMG's 125-D representation.

WHY THIS EXISTS INSTEAD OF THE PRIOR `omg125.py`
------------------------------------------------
A parity test against OMG's own codec (2026-08-22) showed the prior re-implementation
disagrees on the root rotation:

    slice        max|ref - omg125|
    root_pos     2.4e-07   ok
    root_rot     5.4e-01   MISMATCH
    joints       0.0       ok
    links        4.8e-07   ok

Cause: OMG flattens the first two ROWS of R (`matrix[..., :2, :]`), the prior code the
first two COLUMNS. Columns of R are the rows of R^T, so it stored the INVERSE root
rotation in 6 of 125 channels. Its own --selftest passed because its decoder used the
same wrong convention -- internally consistent, externally wrong.

So this converter calls OMG's OWN `model.representation` (kinematics + codec). Parity
holds by construction rather than by test, and it is literally the code path the frozen
checkpoint was trained with.

TWO MODES
---------
  --mode inspect   dump the CSV schema and the proposed qpos_36 mapping. Run this FIRST.
  --mode convert   encode windows and write npz shards + a manifest.

Schema is asserted, never guessed: after the root-rotation bug, quaternion order and
joint order are the next two silent-failure candidates and must fail loudly.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

QPOS_DIM = 36          # [root_pos(3) | root_quat_wxyz(4) | joint_dof(29)]
FEAT_DIM = 125
N_JOINTS = 29
TARGET_FPS = 30.0


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
    rep = instantiate(cfg.model).representation
    rep.kinematics.to(device)
    return rep


# ------------------------------------------------------------------ CSV -> qpos_36
def _norm(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def infer_schema(columns: list[str]) -> dict:
    """Map bones-seed CSV columns onto qpos_36.

    Determined empirically 2026-08-22 (probe1-4) and by the dataset README -- none of it
    is guessed:

      root_translateX/Y/Z   CENTIMETRES   (median root Z = 77.6 -> a 0.78 m pelvis)
      root_rotateX/Y/Z      EULER, DEGREES, EXTRINSIC 'xyz'
                            (max|rot| = 180.0 -> degrees; 'zyx' floats the robot 46 cm
                             off the ground, and only 'xyz' gives a clip and its mirror
                             IDENTICAL ground clearance, which mirroring must preserve)
      29 x *_joint_dof      DEGREES  (max 165.0)  -- order already matches OMG's
                            g1_kinematics.json `joint_order` exactly, verified all 29
      source rate           120 fps  (README: "~288 hours (@ 120 fps)"; 124,554,605
                            frames / 120 / 3600 = 288.3 h, confirming it)
    """
    cols = list(columns)
    out: dict = {
        "format": "bones_seed_euler",
        "root_pos": [c for c in ("root_translateX", "root_translateY", "root_translateZ")
                     if c in cols],
        "root_euler": [c for c in ("root_rotateX", "root_rotateY", "root_rotateZ") if c in cols],
        "euler_order": "xyz",          # extrinsic
        "euler_degrees": True,
        "pos_scale": 0.01,             # cm -> m
        "dof_degrees": True,
        "joints": [c for c in cols if c.endswith("_joint_dof")],
        "src_fps": 120.0,
    }
    return out


def assert_joint_order(schema: dict, kinematics_json: str) -> None:
    """Hard-fail if the CSV joint order differs from OMG's. Never permute silently."""
    omg = json.loads(Path(kinematics_json).read_text(encoding="utf-8"))["joint_order"]
    csv = [c[:-4] for c in schema["joints"]]          # strip the '_dof' suffix
    if len(csv) != N_JOINTS:
        raise ValueError(f"expected {N_JOINTS} joint columns, got {len(csv)}")
    bad = [(i, o, c) for i, (o, c) in enumerate(zip(omg, csv)) if o != c]
    if bad:
        raise ValueError(f"joint order differs from OMG at {len(bad)} positions, "
                         f"first: idx {bad[0][0]} OMG={bad[0][1]} CSV={bad[0][2]} "
                         "-- build an explicit permutation before converting")


def euler_xyz_deg_to_quat_wxyz(eul_deg: np.ndarray) -> np.ndarray:
    """Extrinsic xyz Euler (degrees) -> unit quaternion in MuJoCo's WXYZ order."""
    from scipy.spatial.transform import Rotation as R
    q_xyzw = R.from_euler("xyz", np.asarray(eul_deg, np.float64), degrees=True).as_quat()
    q = q_xyzw[:, [3, 0, 1, 2]]
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def csv_to_qpos36(df: pd.DataFrame, schema: dict) -> np.ndarray:
    """Assemble (T,36) = [root_pos_m(3) | root_quat_wxyz(4) | joint_rad(29)]."""
    if len(schema["root_pos"]) != 3:
        raise ValueError(f"root position: expected 3 columns, matched {schema['root_pos']}")
    if len(schema["root_euler"]) != 3:
        raise ValueError(f"root euler: expected 3 columns, matched {schema['root_euler']}")
    if len(schema["joints"]) != N_JOINTS:
        raise ValueError(
            f"expected {N_JOINTS} joint columns, matched {len(schema['joints'])} "
            "-- refusing to guess joint order")

    pos = df[schema["root_pos"]].to_numpy(np.float64) * schema["pos_scale"]      # cm -> m
    quat = euler_xyz_deg_to_quat_wxyz(df[schema["root_euler"]].to_numpy(np.float64))
    dof = df[schema["joints"]].to_numpy(np.float64)
    if schema["dof_degrees"]:
        dof = np.deg2rad(dof)                                                    # deg -> rad

    # Guard the UNIT error only. A cm-vs-m mistake puts the pelvis at ~75 m, not ~0.1 m.
    # Crawling and floor-sitting legitimately sit at 0.08-0.18 m: an earlier 0.2 m floor
    # silently deleted 8/200 pilot clips, all of them Unusual Locomotion / floor motions,
    # which would have biased the corpus toward upright motion.
    med_z = float(np.median(pos[:, 2]))
    if not (0.02 < med_z < 3.0):
        raise ValueError(f"implausible pelvis height {med_z:.3f} m "
                         "-- wrong unit scale or wrong up-axis")
    if np.abs(dof).max() > 2 * np.pi:
        raise ValueError(f"|joint| max {np.abs(dof).max():.2f} rad -- dof still in degrees?")
    return np.concatenate([pos, quat, dof], -1).astype(np.float32)


def resample(qpos: np.ndarray, src_fps: float, dst_fps: float = TARGET_FPS) -> np.ndarray:
    """Linear on pos/dof, slerp-free normalized-lerp on quat (adjacent frames are close)."""
    if abs(src_fps - dst_fps) < 1e-6:
        return qpos
    t_in = np.arange(qpos.shape[0]) / src_fps
    t_out = np.arange(0.0, t_in[-1] + 1e-9, 1.0 / dst_fps)
    out = np.stack([np.interp(t_out, t_in, qpos[:, c]) for c in range(qpos.shape[1])], -1)
    q = out[:, 3:7]
    sign = np.sign((q[:, :1] != 0) * q[:, :1] + (q[:, :1] == 0))
    q = q * np.where(sign == 0, 1.0, sign)
    out[:, 3:7] = q / np.linalg.norm(q, axis=-1, keepdims=True)
    return out.astype(np.float32)


# ------------------------------------------------------------------ encoding
@torch.no_grad()
def encode_windows(rep, qpos: np.ndarray, starts: list[int], L: int, H: int,
                   device: str = "cpu") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(N,L+H,125) canonicalized about the last history frame, plus the anchor pose."""
    win = L + H
    batch = np.stack([qpos[s:s + win] for s in starts])            # (N, L+H, 36)
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
    """Encode then decode through OMG's OWN codec; returns per-block max abs error.

    `decode_to_world_qpos36` applies standardize_quaternion, which may return -q for the
    same rotation. q and -q are the SAME orientation, so the quaternion block is compared
    sign-invariantly -- otherwise a correct round-trip would report ~2.0 error.
    """
    feats, ap, aq = encode_windows(rep, qpos, [0], L, H, device)
    with torch.no_grad():
        comps = rep.codec.split_features(torch.from_numpy(feats).to(device))
        back = rep.codec.decode_to_world_qpos36(
            comps,
            torch.from_numpy(ap).to(device),
            torch.from_numpy(aq).to(device),
        ).cpu().numpy()[0]
    ref = qpos[:L + H]
    q_err = np.minimum(np.abs(back[:, 3:7] - ref[:, 3:7]),
                       np.abs(back[:, 3:7] + ref[:, 3:7])).max()
    return {
        "root_pos": float(np.abs(back[:, 0:3] - ref[:, 0:3]).max()),
        "root_quat": float(q_err),
        "joints": float(np.abs(back[:, 7:36] - ref[:, 7:36]).max()),
    }


# ------------------------------------------------------------------ modes
def mode_inspect(a) -> int:
    files = sorted(Path(a.csv_root).rglob("*.csv"))
    if not files:
        raise SystemExit(f"no CSVs under {a.csv_root}")
    print(f"found {len(files):,} CSVs; inspecting {files[0]}")
    df = pd.read_csv(files[0])
    print(f"  shape {df.shape}")
    print(f"  columns ({len(df.columns)}):")
    for i, c in enumerate(df.columns):
        print(f"    {i:3d} {c}")
    schema = infer_schema(list(df.columns))
    print("\n  mapping:")
    print(f"    root_pos   : {schema['root_pos']}  (x{schema['pos_scale']} -> metres)")
    print(f"    root_euler : {schema['root_euler']}  order={schema['euler_order']} "
          f"degrees={schema['euler_degrees']}")
    print(f"    joints     : {len(schema['joints'])} cols, degrees={schema['dof_degrees']}")
    print(f"    src_fps    : {schema['src_fps']}  -> resampled to {TARGET_FPS}")
    ok = (len(schema["root_pos"]) == 3 and len(schema["root_euler"]) == 3
          and len(schema["joints"]) == N_JOINTS)
    print(f"\n  schema usable: {ok}")
    if ok:
        assert_joint_order(schema, "assets/robots/g1/g1_kinematics.json")
        print("    joint order matches OMG g1_kinematics.json for all 29")
        q = csv_to_qpos36(df, schema)
        print(f"    qpos_36 {q.shape}  root_z mean={q[:,2].mean():.3f} m "
              f"min={q[:,2].min():.3f} max={q[:,2].max():.3f}")
        print(f"    |quat| dev from 1 = {abs(np.linalg.norm(q[:,3:7],axis=-1)-1).max():.2e}")
        print(f"    |dof| max={np.abs(q[:,7:]).max():.3f} rad "
              f"({np.rad2deg(np.abs(q[:,7:]).max()):.1f} deg)")
        r = resample(q, schema["src_fps"], TARGET_FPS)
        print(f"    after {schema['src_fps']:.0f}->{TARGET_FPS:.0f} fps: {q.shape[0]} -> "
              f"{r.shape[0]} frames")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(schema, indent=2), encoding="utf-8")
    print(f"  wrote schema -> {a.out}")
    return 0 if ok else 1


def mode_convert(a) -> int:
    schema = json.loads(Path(a.schema).read_text(encoding="utf-8"))
    assert_joint_order(schema, "assets/robots/g1/g1_kinematics.json")
    man = pd.read_csv(a.manifest, low_memory=False)
    print(f"manifest {a.manifest}: {len(man):,} rows")

    rep = load_omg_representation(a.exp, a.t5, a.device)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    win = a.hist + a.horizon

    rng = np.random.default_rng(a.seed)
    feats_all, meta, shard, n_shard = [], [], 0, 0
    stats = {"ok": 0, "missing": 0, "short": 0, "bad_schema": 0}
    verified = None

    for i, row in enumerate(man.itertuples(index=False)):
        src = Path(a.csv_root).parent / str(row.move_g1_path)
        if not src.exists():
            stats["missing"] += 1
            continue
        try:
            q = csv_to_qpos36(pd.read_csv(src), schema)
        except ValueError:
            stats["bad_schema"] += 1
            continue
        q = resample(q, a.src_fps, TARGET_FPS)
        if q.shape[0] < win:
            stats["short"] += 1
            continue

        if verified is None:                       # one hard round-trip check on real data
            verified = verify_roundtrip(rep, q, a.hist, a.horizon, a.device)
            print(f"  round-trip on first clip: root_pos={verified['root_pos']:.3e} "
                  f"root_quat={verified['root_quat']:.3e} joints={verified['joints']:.3e}",
                  flush=True)
            worst = max(verified.values())
            if worst > 1e-3:
                raise SystemExit(f"ABORT: round-trip error {worst:.3e} too large -- {verified}")

        n_win = min(a.windows_per_clip, (q.shape[0] - win) // a.stride + 1)
        starts = sorted(rng.choice(np.arange(0, q.shape[0] - win + 1, a.stride),
                                   size=n_win, replace=False).tolist())
        feats, ap, aq = encode_windows(rep, q, starts, a.hist, a.horizon, a.device)
        for k, s in enumerate(starts):
            feats_all.append(feats[k])
            meta.append({
                "move_name": row.move_name, "g1_path": str(row.move_g1_path),
                "window_start": int(s), "split": getattr(row, "split", a.split_name),
                "category": row.category, "type_of_movement": row.content_type_of_movement,
                "is_mirror": bool(row.is_mirror), "actor_uid": row.actor_uid,
                "super_group": int(row.super_group),
                "anchor_root_pos": ap[k].tolist(), "anchor_root_quat": aq[k].tolist(),
                "en": row.content_natural_desc_1, "hi": row.content_natural_desc_1_hi,
            })
        stats["ok"] += 1

        if len(feats_all) >= a.shard_size:
            np.savez_compressed(out_dir / f"shard_{shard:04d}.npz",
                                features=np.stack(feats_all).astype(np.float32))
            shard += 1
            n_shard += len(feats_all)
            feats_all = []
        if (i + 1) % 500 == 0:
            print(f"  [{i+1:6,}/{len(man):,}] ok={stats['ok']:,} windows={n_shard+len(feats_all):,}",
                  flush=True)

    if feats_all:
        np.savez_compressed(out_dir / f"shard_{shard:04d}.npz",
                            features=np.stack(feats_all).astype(np.float32))
        n_shard += len(feats_all)

    pd.DataFrame(meta).to_parquet(out_dir / "manifest.parquet", index=False)
    (out_dir / "_meta.json").write_text(json.dumps({
        "dim": FEAT_DIM, "fps": TARGET_FPS, "L": a.hist, "H": a.horizon,
        "windows": int(n_shard), "clips": stats, "src_fps": a.src_fps,
        "roundtrip_err": verified, "exp": a.exp,
        "encoder": "OMG model.representation (codec+kinematics) -- parity by construction",
    }, indent=2), encoding="utf-8")
    print(f"\n{n_shard:,} windows from {stats['ok']:,} clips -> {out_dir}")
    print(f"  {stats}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["inspect", "convert"], required=True)
    ap.add_argument("--csv-root", default="/workspace/data/bones_seed/g1")
    ap.add_argument("--schema", default="/workspace/data/bones_seed/schema.json")
    ap.add_argument("--manifest", help="one of the split CSVs")
    ap.add_argument("--split-name", default="train")
    ap.add_argument("--out", default="/workspace/data/bones_seed/schema.json")
    ap.add_argument("--exp", default="100m")
    ap.add_argument("--t5", default="/workspace/models/t5-base-local")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--src-fps", type=float, default=120.0,
                    help="BONES-SEED is 120 fps (README + 288.3h frame-count check)")
    ap.add_argument("--hist", type=int, default=10)
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--stride", type=int, default=30)
    ap.add_argument("--windows-per-clip", type=int, default=2)
    ap.add_argument("--shard-size", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    return mode_inspect(a) if a.mode == "inspect" else mode_convert(a)


if __name__ == "__main__":
    sys.exit(main())
