"""
retry_hf_push.py
----------------
Re-pushes retargeted .npz files that failed due to HF rate limits.
Uses upload_folder() so ALL files go in ONE commit — no per-file limit.

Run this on Vast.ai:
    python /workspace/retry_hf_push.py

Or to target specific datasets:
    python /workspace/retry_hf_push.py --datasets DFaust_67 EKUT HumanEva

Or to push ALL datasets (safe to re-run — HF just overwrites identical files):
    python /workspace/retry_hf_push.py --all
"""

import argparse
import os
import sys
import time
from pathlib import Path

try:
    from huggingface_hub import HfApi, create_repo
except ImportError:
    os.system(f"{sys.executable} -m pip install huggingface_hub -q")
    from huggingface_hub import HfApi, create_repo

# ── Config ────────────────────────────────────────────────────────────────────
HF_TOKEN     = os.environ.get("HF_TOKEN", os.environ["HF_TOKEN"])
HF_REPO_OUT  = "AdiShingote/pragya-vla-g1-motion-dataset"
RETARGET_DIR = Path("/workspace/retargeted_g1")

# Datasets that failed due to 429 rate limit (from pipeline completion summary)
FAILED_DATASETS = [
    "DFaust_67",
    "EKUT",
    "HumanEva",
    "MPI_HDM05",
    "SFU",
    "SSM_synced",
    "Transitions_mocap",
    "MPI_mosh",
]

# All known datasets (for --all flag)
ALL_DATASETS = [
    "ACCAD",
    "BMLhandball",
    "BMLmovi",
    "BioMotionLab_NTroje",
    "CMU",
    "DFaust_67",
    "EKUT",
    "Eyes_Japan_Dataset",
    "HumanEva",
    "KIT",
    "MPI_HDM05",
    "MPI_Limits",
    "MPI_mosh",
    "SFU",
    "SSM_synced",
    "TotalCapture",
    "Transitions_mocap",
]

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Re-push retargeted .npz to HuggingFace")
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        help="Space-separated list of dataset names to push. "
             "Default: only the ones that previously failed due to rate limits.",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Push ALL datasets (safe to re-run).",
    )
    parser.add_argument(
        "--retarget-dir", type=str, default=str(RETARGET_DIR),
        help=f"Root folder containing per-dataset subdirectories. Default: {RETARGET_DIR}",
    )
    args = parser.parse_args()

    retarget_dir = Path(args.retarget_dir)
    if not retarget_dir.exists():
        print(f"ERROR: Retarget directory not found: {retarget_dir}")
        sys.exit(1)

    if args.all:
        target_datasets = ALL_DATASETS
    elif args.datasets:
        target_datasets = args.datasets
    else:
        target_datasets = FAILED_DATASETS

    api = HfApi(token=HF_TOKEN)

    # Ensure output repo exists
    print(f"Connecting to HF repo: {HF_REPO_OUT}")
    try:
        api.create_repo(
            repo_id=HF_REPO_OUT,
            repo_type="dataset",
            private=True,
            exist_ok=True,
        )
        print(f"  Repo OK: https://huggingface.co/datasets/{HF_REPO_OUT}\n")
    except Exception as e:
        print(f"  WARNING: Could not create/verify repo: {e}\n")

    print(f"Datasets to push: {target_datasets}\n")
    print("=" * 60)

    results = {}
    for dataset in target_datasets:
        folder = retarget_dir / dataset
        if not folder.exists():
            print(f"[SKIP] {dataset}: folder not found at {folder}")
            results[dataset] = "SKIP (folder missing)"
            continue

        npz_files = list(folder.glob("*.npz"))
        if not npz_files:
            print(f"[SKIP] {dataset}: no .npz files found in {folder}")
            results[dataset] = "SKIP (no .npz files)"
            continue

        total_mb = sum(f.stat().st_size for f in npz_files) / 1024**2
        print(f"[{dataset}] Uploading {len(npz_files)} files  ({total_mb:.1f} MB)  via upload_folder()...")
        t0 = time.time()

        try:
            api.upload_folder(
                folder_path=str(folder),
                path_in_repo=dataset,          # stores as /dataset/*.npz in repo
                repo_id=HF_REPO_OUT,
                repo_type="dataset",
                ignore_patterns=["*.log", "*.json", "*.txt"],
                commit_message=f"Add {dataset} retargeted motions ({len(npz_files)} files)",
            )
            elapsed = time.time() - t0
            speed = total_mb / elapsed
            print(f"  [OK] {elapsed:.0f}s  ({speed:.1f} MB/s)  —  {len(npz_files)} files pushed")
            results[dataset] = f"OK ({len(npz_files)} files)"
        except Exception as e:
            print(f"  [FAIL] {e}")
            results[dataset] = f"FAIL: {e}"
            # Wait a bit before next attempt if rate limited
            if "429" in str(e) or "rate" in str(e).lower():
                print("  Rate limit hit — waiting 60 seconds before continuing...")
                time.sleep(60)

    # Summary
    print()
    print("=" * 60)
    print("PUSH SUMMARY")
    print("=" * 60)
    ok = sum(1 for v in results.values() if v.startswith("OK"))
    fail = sum(1 for v in results.values() if v.startswith("FAIL"))
    skip = sum(1 for v in results.values() if v.startswith("SKIP"))
    for dataset, status in results.items():
        tag = "✓" if status.startswith("OK") else ("✗" if status.startswith("FAIL") else "–")
        print(f"  {tag}  {dataset:30s}  {status}")
    print()
    print(f"  OK: {ok}   FAILED: {fail}   SKIPPED: {skip}")
    print()
    if fail:
        print("Re-run with same command to retry failed datasets.")
    else:
        print(f"All done! Dataset: https://huggingface.co/datasets/{HF_REPO_OUT}")
        print()
        print("Next step (on local machine):")
        print("  python download_retargeted_from_hf.py")


if __name__ == "__main__":
    main()
