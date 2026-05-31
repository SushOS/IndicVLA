"""
build_text_sidecar.py
---------------------
Joins HumanML3D/index.csv with HumanML3D/HumanML3D/texts.zip to produce
proto-g1/text_annotations.json.

JSON structure:
{
  "CMU/80/80_63_poses.npz": [
    {
      "clip_id": "000004",
      "start_s": 26.8,
      "end_s": 36.8,
      "captions": ["a person walks forward", ...]
    },
    ...
  ],
  ...
}

The top-level key is the AMASS-root-relative .npz path — the same string used
as "file" in amass_humanml3d_curated.yaml — so you can JOIN the two outputs
by motion path.  Multiple HumanML3D clips can share the same source .npz file
(different time windows), so each key maps to a list.

Skips HumanAct12 clips (same as build_curated_yaml.py).

Usage:
    python build_text_sidecar.py [--index INDEX_CSV] [--texts TEXTS_ZIP] [--out OUT_JSON]
"""

import argparse
import csv
import json
import zipfile
from pathlib import Path

DATASET_TRIM_SECONDS = {
    "Eyes_Japan_Dataset": 3.0,
    "MPI_HDM05": 3.0,
    "TotalCapture": 1.0,
    "MPI_Limits": 1.0,
    "Transitions_mocap": 0.5,
}

HUMANML3D_FPS = 20


def source_path_to_amass_rel(source_path: str) -> str:
    return (source_path
            .replace("./pose_data/", "")
            .replace("_poses.npy", "_poses.npz")
            .replace("\\", "/"))


def dataset_name_from_path(source_path: str) -> str:
    parts = source_path.replace("./pose_data/", "").split("/")
    return parts[0] if parts else ""


def parse_captions(lines: list[bytes]) -> list[str]:
    """Each line: 'caption text#token_ids#pos_tags#...'; keep only the caption."""
    captions = []
    for line in lines:
        text = line.decode("utf-8").strip()
        if text:
            captions.append(text.split("#")[0].strip())
    return captions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="D:/HumanML3d/HumanML3D/index.csv")
    parser.add_argument("--texts", default="D:/HumanML3d/HumanML3D/HumanML3D/texts.zip")
    parser.add_argument("--out", default="D:/HumanML3d/proto-g1/text_annotations.json")
    args = parser.parse_args()

    index_path = Path(args.index)
    texts_zip_path = Path(args.texts)
    output_path = Path(args.out)

    if not texts_zip_path.exists():
        print(f"ERROR: texts.zip not found at {texts_zip_path}")
        print("Expected location: HumanML3D/HumanML3D/texts.zip (inside the cloned repo)")
        return

    annotations: dict[str, list] = {}
    skipped_humanact12 = 0
    missing_txt = 0

    with zipfile.ZipFile(texts_zip_path) as zf:
        zip_names = set(zf.namelist())

        with open(index_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                sp = row["source_path"]

                if "humanact12" in sp.lower():
                    skipped_humanact12 += 1
                    continue

                new_name = row["new_name"]          # e.g. "000004.npy"
                clip_id = new_name.replace(".npy", "")  # e.g. "000004"
                txt_key = f"texts/{clip_id}.txt"

                if txt_key not in zip_names:
                    missing_txt += 1
                    continue

                with zf.open(txt_key) as fh:
                    captions = parse_captions(fh.readlines())

                if not captions:
                    continue

                motion_key = source_path_to_amass_rel(sp)
                dataset = dataset_name_from_path(sp)
                trim_s = DATASET_TRIM_SECONDS.get(dataset, 0.0)

                start_frame = int(row["start_frame"])
                end_frame = int(row["end_frame"])
                start_s = round(trim_s + start_frame / HUMANML3D_FPS, 4)
                end_s = round(trim_s + end_frame / HUMANML3D_FPS, 4) if end_frame >= 0 else None

                annotations.setdefault(motion_key, []).append({
                    "clip_id": clip_id,
                    "start_s": start_s,
                    "end_s": end_s,
                    "captions": captions,
                })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=2)

    total_clips = sum(len(v) for v in annotations.values())
    print(f"Wrote {total_clips} annotated clips across {len(annotations)} motion files -> {output_path}")
    print(f"Skipped {skipped_humanact12} HumanAct12 clips")
    if missing_txt:
        print(f"WARNING: {missing_txt} clips had no matching .txt in texts.zip")


if __name__ == "__main__":
    main()
