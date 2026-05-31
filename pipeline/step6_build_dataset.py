"""
build_dataset_excel.py
----------------------
Joins retargeted G1 .npz files with HumanML3D text annotations
and writes a single Excel file — one row per HumanML3D clip.

Columns:
  npz_path        - HF URL (or local path if files exist) of the retargeted G1 .npz
  clip_id         - HumanML3D clip ID (e.g. "000004")
  source_amass    - original AMASS source path (relative, e.g. CMU/80/80_63_poses.npz)
  start_s         - start time (seconds) of this clip within the full .npz sequence
  end_s           - end time  (seconds) of this clip within the full .npz sequence
  duration_s      - clip length in seconds
  caption_1..4    - individual text captions from HumanML3D
  all_captions    - all captions joined by " | "

File discovery strategy:
  1. If --retargeted-dir exists and contains .npz files → use local paths
  2. Otherwise → list files from HF repo via API (no download required)
     npz_path column will contain the HF resolve URL for each file.

Usage:
  python build_dataset_excel.py
  python build_dataset_excel.py --retargeted-dir D:/HumanML3d/retargeted_g1 --out my_dataset.xlsx
"""

import argparse
import json
import os
import sys
from pathlib import Path
from urllib.parse import quote

import pandas as pd

# Fix Windows console encoding so Unicode prints work
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    from huggingface_hub import HfApi
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "huggingface_hub", "-q"])
    from huggingface_hub import HfApi

# ── Config ────────────────────────────────────────────────────────────────────
HF_TOKEN = os.environ.get("HF_TOKEN", os.environ["HF_TOKEN"])
HF_REPO  = "AdiShingote/pragya-vla-g1-motion-dataset"

ALL_DATASETS = [
    "ACCAD", "BMLhandball", "BMLmovi", "BioMotionLab_NTroje", "CMU",
    "DFaust_67", "EKUT", "Eyes_Japan_Dataset", "HumanEva", "KIT",
    "MPI_HDM05", "MPI_mosh", "SFU", "SSM_synced", "TotalCapture", "Transitions_mocap",
]

# Same character substitutions applied by convert_amass_to_proto.py
# when creating keypoints filenames from AMASS paths.
CHAR_SUBS = str.maketrans({"-": "_", " ": "_", "(": "_", ")": "_"})


# ── Filename conversion ───────────────────────────────────────────────────────

def amass_key_to_expected_basename(key: str) -> str:
    """
    Convert a text_annotations.json key to the expected HF file basename.

    The keypoints pipeline ran on Windows: it embedded the full Windows path
    into each keypoints filename. After retargeting on Linux, the '_retargeted'
    suffix was appended, preserving that Windows-path-based name.

    Examples
    --------
    "KIT/3/kick_high_left02_poses.npz"
      → "_D:\\HumanML3d\\amass_data\\KIT\\3\\kick_high_left02_poses_keypoints_retargeted.npz"

    "Eyes_Japan_Dataset/hamada/pose-06-hangon-hamada_poses.npz"
      → "_D:\\HumanML3d\\amass_data\\Eyes_Japan_Dataset\\hamada\\pose_06_hangon_hamada_poses_keypoints_retargeted.npz"
    """
    # Convert forward slashes → backslashes (Windows path style), strip .npz
    win_stem = key.replace("/", "\\").replace(".npz", "")
    # Apply same char substitutions the keypoints script used
    win_stem = win_stem.translate(CHAR_SUBS)
    return f"_D:\\HumanML3d\\amass_data\\{win_stem}_keypoints_retargeted.npz"


# ── HF index builder ──────────────────────────────────────────────────────────

def build_hf_index(token: str, repo_id: str) -> dict:
    """
    List every *_keypoints_retargeted.npz from the HF repo (no download).
    Returns  { basename → resolve_url }
    where basename = the HF file path after stripping the "DATASET/" prefix,
    i.e. "_D:\\HumanML3d\\amass_data\\...\\foo_keypoints_retargeted.npz"
    """
    api = HfApi(token=token)
    index: dict[str, str] = {}

    print(f"Indexing files from HF repo: {repo_id}")
    for ds in ALL_DATASETS:
        print(f"  {ds:30s}", end="", flush=True)
        try:
            items = list(api.list_repo_tree(
                repo_id=repo_id,
                repo_type="dataset",
                path_in_repo=ds,
                recursive=False,
            ))
            npz = [f for f in items if hasattr(f, "path") and f.path.endswith(".npz")]
            for item in npz:
                # item.path = "ACCAD/_D:\HumanML3d\..._keypoints_retargeted.npz"
                basename = item.path[len(ds) + 1:]          # strip "DATASET/" prefix
                encoded  = quote(item.path, safe="/")
                url = (f"https://huggingface.co/datasets/{repo_id}"
                       f"/resolve/main/{encoded}")
                index[basename] = url
            print(f"  {len(npz):>5} files")
        except Exception as exc:
            print(f"  ERROR: {exc}")

    print(f"\n  Total indexed: {len(index):,} files")
    return index


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build Excel dataset from G1 retargeted motions + HumanML3D captions")
    parser.add_argument(
        "--retargeted-dir",
        default="D:/HumanML3d/retargeted_g1",
        help="Local directory with retargeted .npz files (searched recursively). "
             "If absent or empty, files are discovered from HF instead.",
    )
    parser.add_argument(
        "--text-json",
        default="D:/HumanML3d/proto-g1/text_annotations.json",
        help="text_annotations.json built by build_text_sidecar.py",
    )
    parser.add_argument(
        "--out",
        default="D:/HumanML3d/g1_humanml3d_dataset.xlsx",
        help="Output Excel file path",
    )
    parser.add_argument("--hf-repo",  default=HF_REPO,  help="Source HF dataset repo")
    parser.add_argument("--hf-token", default=HF_TOKEN, help="HuggingFace read token")
    args = parser.parse_args()

    retargeted_dir = Path(args.retargeted_dir)
    text_json_path = Path(args.text_json)
    output_path    = Path(args.out)

    # ── Load annotations ──────────────────────────────────────────────────────
    print(f"Loading annotations from {text_json_path} ...")
    with open(text_json_path, encoding="utf-8") as fh:
        annotations: dict = json.load(fh)
    print(f"  {len(annotations):,} AMASS source files  |  "
          f"{sum(len(v) for v in annotations.values()):,} clips total")

    # ── Build file index: basename → path-or-URL ──────────────────────────────
    local_npz = list(retargeted_dir.rglob("*_keypoints_retargeted.npz")) if retargeted_dir.exists() else []

    if local_npz:
        print(f"\nUsing local files from {retargeted_dir}")
        # Local files: key on their base filename (same convention as HF basenames)
        existing = {p.name: str(p.resolve()) for p in local_npz}
        print(f"  {len(existing):,} .npz files found")
    else:
        print(f"\nNo local files — fetching file index from HF ...")
        existing = build_hf_index(args.hf_token, args.hf_repo)

    # ── Join annotations with retargeted files ────────────────────────────────
    print("\nJoining annotations with retargeted files ...")
    rows = []
    missing_npz = 0
    missing_keys: list[str] = []

    for amass_key, clips in annotations.items():
        expected_name = amass_key_to_expected_basename(amass_key)
        npz_ref = existing.get(expected_name)

        if npz_ref is None:
            missing_npz += 1
            missing_keys.append(amass_key)
            continue

        for clip in clips:
            captions = clip.get("captions", [])
            start_s  = clip.get("start_s", 0.0)
            end_s    = clip.get("end_s")

            rows.append({
                "npz_path":     npz_ref,
                "clip_id":      clip["clip_id"],
                "source_amass": amass_key,
                "start_s":      start_s,
                "end_s":        end_s,
                "duration_s":   round(end_s - start_s, 3) if end_s is not None else None,
                "caption_1":    captions[0] if len(captions) > 0 else "",
                "caption_2":    captions[1] if len(captions) > 1 else "",
                "caption_3":    captions[2] if len(captions) > 2 else "",
                "caption_4":    captions[3] if len(captions) > 3 else "",
                "all_captions": " | ".join(captions),
            })

    # ── Diagnostics if nothing matched ────────────────────────────────────────
    if not rows:
        print("\n[ERROR] No annotations matched any retargeted file.")
        print(f"  Annotations total : {len(annotations):,}")
        print(f"  Files indexed     : {len(existing):,}")
        if existing:
            sample_key = next(iter(annotations))
            expected   = amass_key_to_expected_basename(sample_key)
            actual     = next(iter(existing))
            print(f"\n  Sample annotation key : {sample_key}")
            print(f"  Expected basename     : {expected!r}")
            print(f"  Sample actual key     : {actual!r}")
        return

    # ── Write Excel ───────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Writing Excel → {output_path} ...")
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Dataset")

        ws = writer.sheets["Dataset"]
        for col in ws.columns:
            max_len = max(len(str(cell.value or "")) for cell in col)
            ws.column_dimensions[col[0].column_letter].width = min(max_len + 2, 80)

    # ── Summary ───────────────────────────────────────────────────────────────
    total_annotations = sum(len(v) for v in annotations.values())
    print()
    print("=" * 55)
    print(f"  Annotations in JSON     : {total_annotations:>7,}")
    print(f"  AMASS files with .npz   : {len(annotations):>7,}")
    print(f"  AMASS files missing .npz: {missing_npz:>7,}")
    print(f"  Excel rows written      : {len(rows):>7,}")
    print(f"  Output                  : {output_path}")
    print("=" * 55)

    if missing_keys:
        sample = missing_keys[:5]
        print(f"\n  First {len(sample)} unmatched keys (of {missing_npz}):")
        for k in sample:
            print(f"    {k}")


if __name__ == "__main__":
    main()
