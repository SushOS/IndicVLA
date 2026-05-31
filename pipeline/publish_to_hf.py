"""
create_professional_dataset.py
-------------------------------
Republishes the G1 motion dataset to a new HF repo with a clean,
professional commit history suitable for research paper citation.

Creates : AdiShingote/PragyaVLA-G1-motions-dataset  (private)
Source  : AdiShingote/pragya-vla-g1-motion-dataset  (existing)

Run:
    python create_professional_dataset.py

If local retargeted files already exist at D:/HumanML3d/retargeted_g1/
they are used directly (fast). Otherwise each dataset is streamed from
the source HF repo before being re-uploaded (slower but self-contained).
"""

import os, sys, time, shutil
from pathlib import Path
from urllib.parse import quote

try:
    from huggingface_hub import HfApi
    import requests
except ImportError:
    os.system(f"{sys.executable} -m pip install huggingface_hub requests -q")
    from huggingface_hub import HfApi
    import requests

# ── Config ────────────────────────────────────────────────────────────────────
HF_TOKEN    = os.environ.get("HF_TOKEN", os.environ["HF_TOKEN"])
HF_USERNAME = "AdiShingote"
SOURCE_REPO = f"{HF_USERNAME}/pragya-vla-g1-motion-dataset"
NEW_REPO    = f"{HF_USERNAME}/PragyaVLA-G1-motions-dataset"
LOCAL_FILES = Path("D:/HumanML3d/retargeted_g1")   # use local if present
STAGING     = Path("D:/HumanML3d/_hf_staging_pro")  # temp download dir

# Ordered for clean commit history (small → large)
DATASETS = [
    ("HumanEva",            28,    "HumanEva synchronized video and motion capture"),
    ("SSM_synced",          30,    "Synchronized SSM motion dataset"),
    ("TotalCapture",        37,    "TotalCapture full-body motion capture"),
    ("SFU",                 44,    "Simon Fraser University (SFU) motion capture"),
    ("MPI_mosh",            77,    "MPI MoSh motion shape reconstruction"),
    ("Transitions_mocap",   110,   "Human motion transitions dataset"),
    ("DFaust_67",           129,   "Dynamic FAUST (DFaust) 3D human body dataset"),
    ("MPI_HDM05",           215,   "MPI HDM05 motion capture database"),
    ("ACCAD",               252,   "American College of Sports Medicine (ACCAD)"),
    ("EKUT",                348,   "Karlsruhe Institute of Technology (EKUT) motions"),
    ("Eyes_Japan_Dataset",  750,   "Eyes Japan large-scale motion capture"),
    ("BMLhandball",         649,   "BML Handball sport motions"),
    ("BMLmovi",             1801,  "BML MoVi large-scale motion capture corpus"),
    ("CMU",                 2082,  "Carnegie Mellon University (CMU) mocap database"),
    ("BioMotionLab_NTroje", 2958,  "BioMotion Lab / Nikolaus Troje (NTroje) recordings"),
    ("KIT",                 4231,  "KIT Motion Database (Karlsruhe Institute of Technology)"),
]

# ── README ────────────────────────────────────────────────────────────────────
README_MD = """\
---
license: cc-by-nc-4.0
task_categories:
  - robotics
  - reinforcement-learning
language:
  - en
tags:
  - motion-capture
  - robotics
  - humanoid
  - motion-retargeting
  - unitree-g1
  - amass
  - smpl
  - physical-simulation
  - vla
size_categories:
  - 10K<n<100K
---

# PragyaVLA G1 Motion Dataset

## Overview

A large-scale dataset of **13,741 human motion sequences** retargeted to the
**Unitree G1 humanoid robot**, designed for Vision-Language-Action (VLA) model
training and robot motion policy learning.

Motion sequences are sourced from the
[AMASS](https://amass.is.tue.mpg.de/) human motion corpus and retargeted to
the Unitree G1 skeleton using
[PyRoki](https://github.com/chungmin99/pyroki), a differentiable robot
kinematics optimization library.

---

## Dataset Statistics

| Source Dataset         | Sequences | Source Description                                  |
|------------------------|----------:|-----------------------------------------------------|
| ACCAD                  |       252 | American College of Sports Medicine                 |
| BMLhandball            |       649 | BML Handball sport motions                          |
| BMLmovi                |     1,801 | BML MoVi large-scale motion capture corpus          |
| BioMotionLab_NTroje    |     2,958 | BioMotion Lab / Nikolaus Troje recordings           |
| CMU                    |     2,082 | Carnegie Mellon University mocap database           |
| DFaust_67              |       129 | Dynamic FAUST 3D human body dataset                 |
| EKUT                   |       348 | University of Tübingen motion dataset               |
| Eyes_Japan_Dataset     |       750 | Eyes Japan large-scale motion capture               |
| HumanEva               |        28 | HumanEva synchronized video and mocap               |
| KIT                    |     4,231 | KIT Motion Database                                 |
| MPI_HDM05              |       215 | MPI HDM05 motion capture database                   |
| MPI_mosh               |        77 | MPI MoSh motion shape reconstruction                |
| SFU                    |        44 | Simon Fraser University motion capture              |
| SSM_synced             |        30 | Synchronized SSM motion dataset                     |
| TotalCapture           |        37 | TotalCapture full-body motion capture               |
| Transitions_mocap      |       110 | Human motion transitions dataset                    |
| **Total**              | **13,741**|                                                     |

---
"""

# ── Helpers ───────────────────────────────────────────────────────────────────
def download_dataset_http(api, source_repo, ds_name, staging, token):
    """
    Download all .npz files for a dataset using direct HTTP requests.
    Avoids snapshot_download which fails on Windows when HF filenames
    contain backslashes or colons (embedded Windows paths).
    Files are saved with sanitized names (: and \\ replaced with _).
    """
    out_dir = staging / ds_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # List files from HF API (no local paths involved)
    try:
        items = [
            f for f in api.list_repo_tree(
                repo_id=source_repo, repo_type="dataset",
                path_in_repo=ds_name, recursive=False,
            )
            if hasattr(f, "path") and f.path.endswith(".npz")
        ]
    except Exception as e:
        print(f"    Cannot list {ds_name}: {e}")
        return out_dir, []

    if not items:
        return out_dir, []

    headers = {"Authorization": f"Bearer {token}"}
    downloaded = []

    for i, hf_file in enumerate(items):
        # Strip the dataset prefix ("HumanEva/") to get the raw filename
        filename_in_ds = hf_file.path[len(ds_name) + 1:]
        # Sanitize: \ and : are invalid in Windows filenames/paths
        safe_name = filename_in_ds.replace("\\", "_").replace(":", "_")
        local_path = out_dir / safe_name

        if local_path.exists():
            downloaded.append(local_path)
            continue

        # URL-encode the HF path (preserves / as separator, encodes \ : etc.)
        encoded = quote(hf_file.path, safe="/")
        url = (f"https://huggingface.co/datasets/{source_repo}"
               f"/resolve/main/{encoded}")

        try:
            r = requests.get(url, headers=headers, timeout=120)
            r.raise_for_status()
            local_path.write_bytes(r.content)
            downloaded.append(local_path)
        except Exception as e:
            print(f"    Failed {hf_file.path}: {e}")

        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(items)} downloaded...")

    return out_dir, downloaded


def push_with_retry(api, folder_path, path_in_repo, repo_id, commit_message):
    """Upload folder to HF, waiting 65 min on 429 rate limit."""
    while True:
        try:
            api.upload_folder(
                folder_path=str(folder_path),
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=commit_message,
                ignore_patterns=["*.log", "*.json", "*.txt", "*.md", "*.py"],
            )
            return True
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                print(f"    Rate limited — waiting 65 min before retry...")
                time.sleep(65 * 60)
            else:
                print(f"    FAILED: {e}")
                return False


def main():
    api = HfApi(token=HF_TOKEN)
    STAGING.mkdir(parents=True, exist_ok=True)

    # ── 1. Create new repo ────────────────────────────────────────────────────
    print(f"\nCreating repo: {NEW_REPO}")
    api.create_repo(
        repo_id=NEW_REPO,
        repo_type="dataset",
        private=True,
        exist_ok=True,
    )
    print(f"  https://huggingface.co/datasets/{NEW_REPO}")

    # ── 2. Push README as initial commit ──────────────────────────────────────
    readme_path = STAGING / "README.md"
    readme_path.write_text(README_MD, encoding="utf-8")
    print("\nPushing dataset card (README)...")
    api.upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=NEW_REPO,
        repo_type="dataset",
        commit_message="Initial commit: Add dataset card and documentation",
    )
    print("  ✓ README committed")

    # ── 3. Push each dataset ──────────────────────────────────────────────────
    print(f"\nPushing {len(DATASETS)} datasets...\n")
    total_files = 0

    for ds_name, expected_count, description in DATASETS:
        # Prefer local files; fall back to downloading from source repo
        local_ds = LOCAL_FILES / ds_name
        if local_ds.exists() and list(local_ds.glob("*.npz")):
            source_dir = local_ds
            files = list(source_dir.glob("*.npz"))
            print(f"[{ds_name}]  {len(files)} files  (local)")
        else:
            # snapshot_download fails on Windows when HF filenames contain
            # colons/backslashes (embedded Windows paths). Use HTTP instead.
            print(f"[{ds_name}]  Downloading via HTTP (Windows-safe)...")
            source_dir, files = download_dataset_http(
                api, SOURCE_REPO, ds_name, STAGING, HF_TOKEN
            )
            print(f"  Downloaded {len(files)} files")

        if not files:
            print(f"  SKIP — no .npz files found\n")
            continue

        n = len(files)
        commit_msg = (
            f"Add {ds_name}: {n:,} retargeted Unitree G1 motion sequences "
            f"({description})"
        )
        t0 = time.time()
        ok = push_with_retry(api, source_dir, ds_name, NEW_REPO, commit_msg)
        elapsed = time.time() - t0

        if ok:
            total_files += n
            print(f"  ✓ {n:,} files pushed  ({elapsed:.0f}s)  "
                  f"[cumulative: {total_files:,}]\n")
        else:
            print(f"  ✗ Push failed for {ds_name}\n")

    # ── 4. Summary ────────────────────────────────────────────────────────────
    print("=" * 60)
    print(f"Dataset published!")
    print(f"  URL   : https://huggingface.co/datasets/{NEW_REPO}")
    print(f"  Files : {total_files:,} .npz motion sequences")
    print(f"  Vis.  : {len(DATASETS) + 1} commits (1 README + 1 per dataset)")
    print("=" * 60)

    # Clean up staging dir
    if STAGING.exists():
        shutil.rmtree(STAGING, ignore_errors=True)


if __name__ == "__main__":
    main()
