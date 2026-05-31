"""
download_retargeted_from_hf.py
-------------------------------
Pulls the retargeted G1 .npz files from HuggingFace
(AdiShingote/pragya-vla-g1-motion-dataset) to your local machine.

Run this locally on Windows after the Vast.ai pipeline completes.

Usage:
    python download_retargeted_from_hf.py
    python download_retargeted_from_hf.py --dataset CMU      # one dataset only
    python download_retargeted_from_hf.py --flat             # download to flat dir
"""

import argparse
import os
import sys
from pathlib import Path

try:
    from huggingface_hub import snapshot_download, list_repo_tree, hf_hub_download
except ImportError:
    os.system(f"{sys.executable} -m pip install huggingface_hub -q")
    from huggingface_hub import snapshot_download, list_repo_tree, hf_hub_download

HF_TOKEN    = os.environ.get("HF_TOKEN", os.environ["HF_TOKEN"])
HF_REPO_OUT = "AdiShingote/pragya-vla-g1-motion-dataset"

# Local destination: keeps per-dataset sub-folders
LOCAL_DIR   = Path("D:/HumanML3d/retargeted_g1")

def main():
    parser = argparse.ArgumentParser(description="Download retargeted G1 .npz from HuggingFace")
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Download only this dataset (e.g. CMU). Default: all datasets.",
    )
    parser.add_argument(
        "--flat", action="store_true",
        help="Flatten to a single directory (no per-dataset subdirs). "
             "Useful for build_dataset_excel.py.",
    )
    parser.add_argument(
        "--flat-dir", type=str, default="D:/HumanML3d/retargeted_g1_flat",
        help="Destination when --flat is used.",
    )
    args = parser.parse_args()

    if args.flat:
        dest = Path(args.flat_dir)
        dest.mkdir(parents=True, exist_ok=True)
        print(f"Downloading ALL retargeted .npz → {dest}  (flat)")
        print(f"Source: https://huggingface.co/datasets/{HF_REPO_OUT}\n")

        snapshot_download(
            repo_id=HF_REPO_OUT,
            repo_type="dataset",
            local_dir=str(dest / "_hf_staging"),
            token=HF_TOKEN,
            ignore_patterns=["*.md", ".gitattributes"],
        )

        # Flatten: move all .npz from subdirs to dest
        staging = dest / "_hf_staging"
        npz_files = list(staging.rglob("*.npz"))
        print(f"Flattening {len(npz_files)} .npz files...")
        for f in npz_files:
            target = dest / f.name
            if not target.exists():
                import shutil
                shutil.copy2(f, target)
        import shutil
        shutil.rmtree(staging, ignore_errors=True)
        total = len(list(dest.glob("*.npz")))
        print(f"\nDone. {total} .npz files in {dest}")

    elif args.dataset:
        dest = LOCAL_DIR / args.dataset
        dest.mkdir(parents=True, exist_ok=True)
        print(f"Downloading dataset: {args.dataset} → {dest}")
        print(f"Source: https://huggingface.co/datasets/{HF_REPO_OUT}/{args.dataset}/\n")

        snapshot_download(
            repo_id=HF_REPO_OUT,
            repo_type="dataset",
            local_dir=str(dest),
            token=HF_TOKEN,
            allow_patterns=[f"{args.dataset}/*.npz"],
            ignore_patterns=["*.md", ".gitattributes"],
        )

        # Move files from subdir up to dest if snapshot created a subdir
        subdir = dest / args.dataset
        if subdir.exists():
            import shutil
            for f in subdir.glob("*.npz"):
                shutil.move(str(f), str(dest / f.name))
            subdir.rmdir()

        n = len(list(dest.glob("*.npz")))
        print(f"Done. {n} .npz files in {dest}")

    else:
        # Download everything, keeping per-dataset structure
        LOCAL_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Downloading ALL retargeted .npz → {LOCAL_DIR}")
        print(f"Source: https://huggingface.co/datasets/{HF_REPO_OUT}\n")
        print("NOTE: This may take a few minutes (~340 MB total estimated)\n")

        snapshot_download(
            repo_id=HF_REPO_OUT,
            repo_type="dataset",
            local_dir=str(LOCAL_DIR),
            token=HF_TOKEN,
            ignore_patterns=["*.md", ".gitattributes"],
        )

        # Summary
        total = sum(1 for _ in LOCAL_DIR.rglob("*.npz"))
        total_mb = sum(f.stat().st_size for f in LOCAL_DIR.rglob("*.npz")) / 1024**2
        print(f"\nDownload complete!")
        print(f"  Location : {LOCAL_DIR}")
        print(f"  Files    : {total} .npz")
        print(f"  Size     : {total_mb:.1f} MB")
        print()
        print("Per-dataset breakdown:")
        for subdir in sorted(LOCAL_DIR.iterdir()):
            if subdir.is_dir() and not subdir.name.startswith("."):
                n = len(list(subdir.glob("*.npz")))
                mb = sum(f.stat().st_size for f in subdir.glob("*.npz")) / 1024**2
                print(f"  {subdir.name:30s} {n:5d} files  {mb:7.1f} MB")

        print(f"\nNext step: run build_dataset_excel.py")
        print(f"  python build_dataset_excel.py --retargeted-dir {LOCAL_DIR}")


if __name__ == "__main__":
    main()
