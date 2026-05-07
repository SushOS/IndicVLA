"""
Pragya-VLA pilot: offline MuJoCo review of generated motions.

Plays back the qpos CSVs against the Unitree G1 MJCF in MuJoCo. Designed to
run on CPU on your laptop — no GPU required for replay. You can step through
the full set, mark accept/reject by pressing keys in the terminal between
clips.

Requirements (laptop side):
    pip install mujoco numpy

Usage:
    python review_mujoco.py \\
        --qpos_dir /path/to/kimodo_outputs/qpos_g1 \\
        --manifest /path/to/kimodo_outputs/manifest.jsonl \\
        --review_log /path/to/kimodo_outputs/review.jsonl

Note: qpos CSVs are produced for the G1 model only. To visually inspect the
SOMA cross-check motions, use the Kimodo demo UI (`kimodo_demo`) and load
the corresponding files from `motions_soma/` — MuJoCo replay is G1-specific.

Controls (in terminal, between clips):
    [a] = accept, motion looks plausible
    [r] = reject, motion is broken / wrong intent
    [u] = undecided, defer
    [s] = skip without recording
    [q] = quit

Notes:
    * Each clip plays once at 30 fps, then waits for your input.
    * Resume support: if review.jsonl already has a prompt_id, we skip it.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def load_manifest(manifest_path: Path) -> dict:
    """prompt_id -> generation manifest record."""
    by_id = {}
    if not manifest_path.exists():
        return by_id
    with open(manifest_path) as f:
        for line in f:
            try:
                rec = json.loads(line)
                by_id[rec["prompt_id"]] = rec
            except json.JSONDecodeError:
                continue
    return by_id


def load_reviewed_ids(review_log: Path) -> set:
    done = set()
    if not review_log.exists():
        return done
    with open(review_log) as f:
        for line in f:
            try:
                rec = json.loads(line)
                done.add(rec["prompt_id"])
            except json.JSONDecodeError:
                continue
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qpos_dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--review_log", required=True)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--loops", type=int, default=2,
                    help="Loop each clip this many times before asking for input")
    args = ap.parse_args()

    # Lazy import — gives a clean message if mujoco isn't installed
    try:
        import mujoco
        import mujoco.viewer
    except ImportError:
        print("ERROR: mujoco not installed. Run: pip install mujoco")
        sys.exit(1)

    # The G1 XML ships inside the kimodo package assets.
    # If reviewing on a machine without kimodo installed, point this at a
    # standalone copy of g1.xml (e.g. from unitree_mujoco repo).
    try:
        from kimodo.assets import skeleton_asset_path
        g1_xml = str(skeleton_asset_path("g1skel34", "xml", "g1.xml"))
    except ImportError:
        print("kimodo package not found locally. Set G1_XML_PATH env var to "
              "your standalone g1.xml.", file=sys.stderr)
        import os
        g1_xml = os.environ.get("G1_XML_PATH")
        if not g1_xml:
            sys.exit(1)

    manifest = load_manifest(Path(args.manifest))
    reviewed = load_reviewed_ids(Path(args.review_log))
    qpos_dir = Path(args.qpos_dir)

    csv_files = sorted(qpos_dir.glob("*.csv"))
    todo = [f for f in csv_files if f.stem not in reviewed]
    print(f"{len(csv_files)} total, {len(reviewed)} already reviewed, "
          f"{len(todo)} remaining")

    model = mujoco.MjModel.from_xml_path(g1_xml)
    data = mujoco.MjData(model)

    with open(args.review_log, "a") as rl:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            for i, csv_file in enumerate(todo, start=1):
                pid = csv_file.stem
                rec = manifest.get(pid, {})
                qpos = np.loadtxt(csv_file, delimiter=",")
                if qpos.ndim == 1:
                    qpos = qpos[None, :]

                print(f"\n[{i}/{len(todo)}] {pid}")
                print(f"  imperative: {rec.get('imperative_en', '?')}")
                print(f"  kimodo:     {rec.get('kimodo_input', '?')}")
                print(f"  family:     {rec.get('motion_family', '?')}")
                print(f"  frames:     {qpos.shape[0]} ({qpos.shape[0]/args.fps:.1f}s)")

                # Play it (loops times)
                for _ in range(args.loops):
                    if not viewer.is_running():
                        break
                    for frame in qpos:
                        if not viewer.is_running():
                            break
                        data.qpos[:] = frame
                        mujoco.mj_forward(model, data)
                        viewer.sync()
                        time.sleep(1.0 / args.fps)

                # Decision
                while True:
                    ans = input("  decision [a]ccept / [r]eject / [u]ndecided "
                                "/ [s]kip / [q]uit: ").strip().lower()
                    if ans in {"a", "r", "u", "s", "q"}:
                        break
                if ans == "q":
                    print("Quitting.")
                    return
                if ans == "s":
                    continue
                decision_map = {"a": "accept", "r": "reject", "u": "undecided"}
                rl.write(json.dumps({
                    "prompt_id": pid,
                    "decision": decision_map[ans],
                    "reviewer_note": "",
                }) + "\n")
                rl.flush()

    print("Review complete.")


if __name__ == "__main__":
    main()
