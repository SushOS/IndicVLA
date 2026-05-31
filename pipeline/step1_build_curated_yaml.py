"""
build_curated_yaml.py
---------------------
Reads HumanML3D/index.csv and emits amass_humanml3d_curated.yaml for
ProtoMotions' convert_amass_to_proto.py.

Skips HumanAct12 clips (not an AMASS subdataset).
Converts HumanML3D 20-fps frame indices to seconds in the original AMASS file,
accounting for the per-dataset head-trims applied by HumanML3D's
raw_pose_processing.ipynb before saving pose_data/*.npy.

Usage:
    python build_curated_yaml.py [--index INDEX_CSV] [--out OUT_YAML]
"""

import argparse
import csv
import yaml
from pathlib import Path

# Seconds trimmed from the START of each dataset sequence by HumanML3D's
# raw_pose_processing.ipynb (cell-18) before saving pose_data/*.npy.
# The index.csv start/end frames are relative to AFTER this trim.
DATASET_TRIM_SECONDS = {
    "Eyes_Japan_Dataset": 3.0,
    "MPI_HDM05": 3.0,
    "TotalCapture": 1.0,
    "MPI_Limits": 1.0,
    "Transitions_mocap": 0.5,
}

HUMANML3D_FPS = 20  # HumanML3D downsamples all sequences to 20 fps


def source_path_to_amass_rel(source_path: str) -> str:
    """Map a HumanML3D pose_data path to an AMASS-root-relative .npz path."""
    return (source_path
            .replace("./pose_data/", "")
            .replace("_poses.npy", "_poses.npz"))


def dataset_name_from_path(source_path: str) -> str:
    """Extract the top-level AMASS subdataset folder name."""
    parts = source_path.replace("./pose_data/", "").split("/")
    return parts[0] if parts else ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="D:/HumanML3d/HumanML3D/index.csv",
                        help="Path to HumanML3D index.csv")
    parser.add_argument("--out", default="D:/HumanML3d/amass_humanml3d_curated.yaml",
                        help="Output YAML path")
    parser.add_argument("--amass-root", default="D:/HumanML3d/amass_data",
                        help="Root of downloaded AMASS data (used only for existence warnings)")
    args = parser.parse_args()

    index_path = Path(args.index)
    output_path = Path(args.out)
    amass_root = Path(args.amass_root)

    motions = []
    skipped_humanact12 = 0
    missing_files = []

    with open(index_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sp = row["source_path"]

            if "humanact12" in sp.lower():
                skipped_humanact12 += 1
                continue

            amass_rel = source_path_to_amass_rel(sp).replace("\\", "/")
            dataset = dataset_name_from_path(sp)
            trim_s = DATASET_TRIM_SECONDS.get(dataset, 0.0)

            start_frame = int(row["start_frame"])
            end_frame = int(row["end_frame"])

            start_s = round(trim_s + start_frame / HUMANML3D_FPS, 4)
            end_s = round(trim_s + end_frame / HUMANML3D_FPS, 4) if end_frame >= 0 else 9999.0

            motions.append({
                "file": amass_rel,
                "sub_motions": [
                    {"timings": {"start": start_s, "end": end_s}}
                ],
            })

            if not (amass_root / amass_rel).exists():
                missing_files.append(amass_rel)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump({"motions": motions}, f, default_flow_style=False, sort_keys=False)

    print(f"Wrote {len(motions)} entries -> {output_path}")
    print(f"Skipped {skipped_humanact12} HumanAct12 clips (no AMASS source)")

    if missing_files:
        pct = 100 * len(missing_files) / len(motions)
        print(f"WARNING: {len(missing_files)} / {len(motions)} AMASS files not found locally "
              f"({pct:.1f}%) — download AMASS subdatasets first.")
        sample = missing_files[:5]
        for p in sample:
            print(f"  missing: {p}")
        if len(missing_files) > 5:
            print(f"  ... and {len(missing_files) - 5} more")


if __name__ == "__main__":
    main()
