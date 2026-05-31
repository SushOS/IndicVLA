"""
play_g1_npz.py
--------------
Play a PyRoki-retargeted G1 .npz file in MuJoCo.

The PyRoki output format:
  base_frame_pos   (T, 3)  -- root XYZ in world frame
  base_frame_wxyz  (T, 4)  -- root quaternion, w-first
  joint_angles     (T, 29) -- 29 actuated DOF angles

MuJoCo G1 qpos layout (36 values):
  [0:3]  root position
  [3:7]  root quaternion (w, x, y, z)
  [7:36] joint angles (same order as URDF actuated joints)

Usage:
  python play_g1_npz.py path/to/retargeted.npz
  python play_g1_npz.py path/to/retargeted.npz --fps 30 --xml path/to/g1.xml
  python play_g1_npz.py --csv g1_chunk_0.csv
  python play_g1_npz.py --csv g1_chunk_0.csv --start 0 --end 10
"""

import argparse
import time
import csv
import tempfile
import os
import urllib.request
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer


DEFAULT_XML = str(
    Path(__file__).parent
    / "ProtoMotions/protomotions/data/assets/mjcf/g1_holo_compat.xml"
)


def load_npz(path: str) -> np.ndarray:
    f = np.load(path)
    pos = f["base_frame_pos"]    # (T, 3)
    quat = f["base_frame_wxyz"]  # (T, 4)  w-first
    angles = f["joint_angles"]   # (T, 29)
    T = pos.shape[0]
    qpos = np.concatenate([pos, quat, angles], axis=1)  # (T, 36)
    print(f"Loaded {T} frames  ({T/30:.1f}s at 30fps)")
    print(f"  root pos range:    {pos.min():.3f} .. {pos.max():.3f}")
    print(f"  joint angle range: {angles.min():.3f} .. {angles.max():.3f}")
    return qpos


def download_npz(url: str, hf_token: str = None) -> str:
    """Download a URL to a temp .npz file and return the path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
    tmp.close()
    print(f"  Downloading: {url.split('/')[-1]}")
    req = urllib.request.Request(url)
    if hf_token:
        req.add_header("Authorization", f"Bearer {hf_token}")
    with urllib.request.urlopen(req) as resp, open(tmp.name, "wb") as f:
        f.write(resp.read())
    return tmp.name


def read_csv_rows(csv_path: str, start: int, end: int):
    """Yield (index, npz_path_or_url, caption) rows from the CSV."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i < start:
                continue
            if end is not None and i >= end:
                break
            yield i, row["npz_path"], row.get("caption_1", "")


def play_qpos(model, data, qpos, fps, label=""):
    """Play a single qpos sequence in an already-open viewer context."""
    dt = 1.0 / fps
    print(f"\n[{label}]  {qpos.shape[0]} frames")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        for frame in qpos:
            if not viewer.is_running():
                return False  # user closed viewer
            data.qpos[:qpos.shape[1]] = frame
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(dt)
        # hold last frame briefly so user can see it
        end = time.time() + 1.0
        while viewer.is_running() and time.time() < end:
            viewer.sync()
            time.sleep(0.05)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("npz_path", nargs="?", help="Path to a single .npz file")
    parser.add_argument("--csv", help="Path to g1_chunk CSV to play all motions")
    parser.add_argument("--start", type=int, default=0, help="Start row index in CSV (inclusive)")
    parser.add_argument("--end",   type=int, default=None, help="End row index in CSV (exclusive)")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--xml", default=DEFAULT_XML,
                        help="Path to G1 MuJoCo XML (default: ProtoMotions g1_holo_compat.xml)")
    parser.add_argument("--loop", action="store_true", help="Loop single-file playback")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"),
                        help="HuggingFace API token for private datasets (or set HF_TOKEN env var)")
    args = parser.parse_args()

    xml_path = Path(args.xml)
    if not xml_path.exists():
        print(f"ERROR: XML not found at {xml_path}")
        print("Pass --xml path/to/g1.xml  or set DEFAULT_XML in this script.")
        return

    xml_text = xml_path.read_text(encoding="utf-8")
    floor_geom = '    <geom name="floor" type="plane" size="20 20 0.1" rgba="0.6 0.6 0.6 1"/>'
    if 'name="floor"' not in xml_text:
        xml_text = xml_text.replace("</worldbody>", f"{floor_geom}\n  </worldbody>", 1)
    tmp_xml = xml_path.parent / "_play_tmp.xml"
    tmp_xml.write_text(xml_text, encoding="utf-8")
    try:
        model = mujoco.MjModel.from_xml_path(str(tmp_xml))
    finally:
        tmp_xml.unlink(missing_ok=True)
    data = mujoco.MjData(model)

    # ---- CSV mode ----
    if args.csv:
        print(f"Playing motions from CSV: {args.csv}  (rows {args.start}..{args.end or 'end'})")
        for idx, npz_ref, caption in read_csv_rows(args.csv, args.start, args.end):
            print(f"\n=== Motion {idx}: {caption[:80]}")
            tmp_file = None
            try:
                if npz_ref.startswith("http"):
                    tmp_file = download_npz(npz_ref, hf_token=args.hf_token)
                    npz_local = tmp_file
                else:
                    npz_local = npz_ref

                if not Path(npz_local).exists():
                    print(f"  SKIP — file not found: {npz_local}")
                    continue

                qpos = load_npz(npz_local)
                if qpos.shape[1] != model.nq:
                    print(f"  WARNING: qpos dim {qpos.shape[1]} != model nq {model.nq}")

                ok = play_qpos(model, data, qpos, args.fps,
                               label=f"{idx}: {caption[:60]}")
                if not ok:
                    print("Viewer closed — stopping.")
                    break
            finally:
                if tmp_file and os.path.exists(tmp_file):
                    os.unlink(tmp_file)
        return

    # ---- Single-file mode ----
    if not args.npz_path:
        parser.error("Provide either a npz_path or --csv")

    qpos = load_npz(args.npz_path)
    if qpos.shape[1] != model.nq:
        print(f"WARNING: NPZ has {qpos.shape[1]} qpos values but model expects {model.nq}")

    dt = 1.0 / args.fps
    print(f"\nPlaying on model: {xml_path.name}  (nq={model.nq}, nv={model.nv})")
    print("Press Ctrl+C to quit.\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            for frame in qpos:
                if not viewer.is_running():
                    break
                data.qpos[:qpos.shape[1]] = frame
                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(dt)
            if not args.loop:
                while viewer.is_running():
                    viewer.sync()
                    time.sleep(0.1)


if __name__ == "__main__":
    main()
