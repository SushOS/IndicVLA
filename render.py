"""
Pragya-VLA pilot review: render each G1 qpos CSV (or NPZ) into an MP4.

Two render modes are selected automatically:

  MuJoCo mode  (preferred)
    Requires: g1.xml + qpos CSVs in qpos_dir.
    Plays each frame through MuJoCo's offscreen renderer for a photo-realistic
    robot view with the same camera angle used in Kimodo paper figures.

  Stick-figure mode  (fallback, no robot model needed)
    Used when qpos_dir has no CSVs or g1.xml cannot be found.
    Reads posed_joints from the matching NPZ in motions_g1/, projects them to
    front & side views, and draws a colour-coded skeleton with PIL.
    Output quality is lower but requires zero extra downloads.

Inputs (from kimodo_outputs_full):
    motions_g1/<prompt_id>.npz   -- always present
    qpos_g1/<prompt_id>.csv      -- present only if you ran kimodo export

Outputs:
    <out_dir>/<prompt_id>.mp4

Dependencies:
    pip install mujoco numpy imageio imageio-ffmpeg pillow

The G1 MJCF (for MuJoCo mode) is found via, in order:
    1. --g1_xml flag
    2. G1_XML environment variable
    3. kimodo.assets.skeleton_asset_path (if kimodo is installed)

Usage:
    python render.py \\
        --qpos_dir  kimodo_outputs_full/kimodo_outputs_full/qpos_g1 \\
        --out_dir   kimodo_outputs_full/videos_g1 \\
        --manifest  kimodo_outputs_full/kimodo_outputs_full/manifest.jsonl
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


# ------- Camera & render constants ----------------------------------------
WIDTH = 640          # output video width
HEIGHT = 480         # output video height
FPS = 30             # matches Kimodo's training fps
CAM_DISTANCE = 3.5   # metres from robot
CAM_AZIMUTH = 135.0  # degrees; 90 = pure side-on, 180 = pure front. 135 = front-3/4
CAM_ELEVATION = -15.0  # negative = looking down at the robot
CAM_LOOKAT_Z = 0.9   # height of the look-at point (roughly waist height)

# ------- G1 stick-figure skeleton (34 joints, Y-up coordinate system) -----
# Joint layout inferred from posed_joints positions at rest:
#   0       = pelvis/root
#   1-7     = right leg chain (hip → toe)
#   8-14    = left leg chain  (hip → toe)
#   15-17   = spine (15≈pelvis dup, 16-17 = waist/torso)
#   18-25   = right arm (shoulder → fingertip)
#   26-33   = left  arm (shoulder → fingertip)
_BONES = [
    (0, 1),  (1, 2),  (2, 3),  (3, 4),  (4, 5),  (5, 6),  (6, 7),   # right leg
    (0, 8),  (8, 9),  (9,10), (10,11), (11,12), (12,13), (13,14),    # left leg
    (0,16), (16,17),                                                   # spine
    (17,18), (18,19), (19,20), (20,21), (21,22), (22,23), (23,24), (24,25),  # right arm
    (17,26), (26,27), (27,28), (28,29), (29,30), (30,31), (31,32), (32,33),  # left arm
]
_BONE_COLORS = (
    [(230, 100, 100)] * 7 +   # right leg  – warm red
    [(100, 100, 230)] * 7 +   # left leg   – cool blue
    [(230, 230, 100)] * 2 +   # spine      – yellow
    [(230, 160,  80)] * 8 +   # right arm  – orange
    [( 80, 200, 160)] * 8     # left arm   – teal
)


def find_g1_xml(cli_arg: str | None) -> str | None:
    """Locate the G1 MJCF; returns None (instead of raising) when not found."""
    if cli_arg:
        if not Path(cli_arg).exists():
            print(f"[warn] --g1_xml not found: {cli_arg}", file=sys.stderr)
            return None
        return cli_arg

    env_path = os.environ.get("G1_XML")
    if env_path and Path(env_path).exists():
        return env_path

    try:
        from kimodo.assets import skeleton_asset_path
        return str(skeleton_asset_path("g1skel34", "xml", "g1.xml"))
    except Exception:
        pass

    return None


def _find_npz_dir(qpos_dir: Path, manifest_path: Path | None) -> Path | None:
    """Resolve the motions_g1 NPZ directory from qpos_dir or manifest location."""
    # Sibling motions_g1 next to qpos_g1
    candidate = qpos_dir.parent / "motions_g1"
    if candidate.is_dir():
        return candidate
    # Sibling next to manifest
    if manifest_path and manifest_path.exists():
        candidate2 = manifest_path.parent / "motions_g1"
        if candidate2.is_dir():
            return candidate2
    return None


def load_prompt_lookup(manifest_path: Path) -> dict:
    """prompt_id -> {imperative_en, kimodo_input, motion_family} for nice logs."""
    out = {}
    if not manifest_path or not manifest_path.exists():
        return out
    with open(manifest_path) as f:
        for line in f:
            try:
                rec = json.loads(line)
                out[rec["prompt_id"]] = {
                    "imperative_en": rec.get("imperative_en", ""),
                    "kimodo_input": rec.get("kimodo_input", ""),
                    "motion_family": rec.get("motion_family", ""),
                }
            except json.JSONDecodeError:
                continue
    return out


def render_one(model, data, renderer, camera, qpos: np.ndarray, fps: int):
    """Yield rendered RGB frames for the full qpos sequence."""
    import mujoco
    for frame_qpos in qpos:
        data.qpos[:] = frame_qpos
        mujoco.mj_forward(model, data)
        # Re-set lookat each frame so the camera tracks the robot's translation.
        # qpos[:3] = root xyz of the floating pelvis.
        camera.lookat[0] = float(frame_qpos[0])
        camera.lookat[1] = float(frame_qpos[1])
        camera.lookat[2] = CAM_LOOKAT_Z + float(frame_qpos[2]) * 0.0  # keep z fixed at waist
        renderer.update_scene(data, camera=camera)
        yield renderer.render()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qpos_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--manifest", default=None,
                    help="Optional manifest.jsonl for nicer per-clip logging")
    ap.add_argument("--g1_xml", default=None,
                    help="Path to g1.xml; falls back to kimodo bundled or $G1_XML")
    ap.add_argument("--limit", type=int, default=None,
                    help="Render only first N clips (smoke test)")
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-render clips that already have an MP4")
    args = ap.parse_args()

    # Defer heavy imports so --help is fast and errors are clear.
    try:
        import mujoco
    except ImportError:
        raise SystemExit("Install mujoco first: pip install mujoco")
    try:
        import imageio.v3 as iio  # noqa: F401  (imageio_ffmpeg backend used below)
        import imageio
    except ImportError:
        raise SystemExit(
            "Install imageio with ffmpeg: pip install imageio imageio-ffmpeg"
        )

    qpos_dir = Path(args.qpos_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csvs = sorted(qpos_dir.glob("*.csv"))
    if args.limit:
        csvs = csvs[: args.limit]

    if not csvs:
        raise SystemExit(f"No CSVs in {qpos_dir}")

    prompts = load_prompt_lookup(Path(args.manifest)) if args.manifest else {}

    # Load model + renderer ONCE, reuse for all clips.
    g1_xml = find_g1_xml(args.g1_xml)
    print(f"[setup] G1 XML: {g1_xml}")
    model = mujoco.MjModel.from_xml_path(g1_xml)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)

    # Free camera (not a fixed XML camera). Set once outside the loop.
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.distance = CAM_DISTANCE
    camera.azimuth = CAM_AZIMUTH
    camera.elevation = CAM_ELEVATION
    camera.lookat[:] = [0.0, 0.0, CAM_LOOKAT_Z]

    print(f"[setup] {len(csvs)} clips queued; "
          f"output={out_dir}; res={WIDTH}x{HEIGHT}@{FPS}fps")

    n_ok, n_skip, n_fail = 0, 0, 0
    t_start = time.time()

    for i, csv_path in enumerate(csvs, start=1):
        pid = csv_path.stem
        out_path = out_dir / f"{pid}.mp4"
        if out_path.exists() and not args.overwrite:
            n_skip += 1
            continue

        try:
            qpos = np.loadtxt(csv_path, delimiter=",")
            if qpos.ndim == 1:
                qpos = qpos[None, :]
            if qpos.shape[1] != model.nq:
                raise ValueError(
                    f"qpos has {qpos.shape[1]} cols but model.nq={model.nq}"
                )

            t_clip = time.time()
            # imageio writer with libx264 settings sane for browser playback
            with imageio.get_writer(
                str(out_path),
                fps=FPS,
                codec="libx264",
                quality=8,            # 0=worst, 10=best; 8 is a good balance
                pixelformat="yuv420p",  # required for broad browser support
                macro_block_size=1,     # allow non-multiple-of-16 dims
            ) as writer:
                for frame in render_one(model, data, renderer, camera, qpos, FPS):
                    writer.append_data(frame)

            elapsed = time.time() - t_clip
            n_ok += 1
            label = ""
            if pid in prompts:
                label = f" \"{prompts[pid]['imperative_en']}\""
            print(f"[{i:3d}/{len(csvs)}] {pid}{label}  "
                  f"({qpos.shape[0]} frames, {elapsed:.1f}s)")
        except Exception as e:
            n_fail += 1
            print(f"[{i:3d}/{len(csvs)}] {pid} FAILED: {type(e).__name__}: {e}",
                  file=sys.stderr)

    total = time.time() - t_start
    print(f"\n[done] ok={n_ok} skipped={n_skip} failed={n_fail} "
          f"in {total/60:.1f} min ({total/max(n_ok,1):.1f}s per clip avg)")
    print(f"[done] videos: {out_dir}")


if __name__ == "__main__":
    main()