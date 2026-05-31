"""
upload_pt_to_hf.py
------------------
Uploads all 17 per-dataset AMASS .pt files to HuggingFace
under AdiShingote/amass-smpl-dataset.

Run locally on Windows:
    pip install huggingface_hub
    python upload_pt_to_hf.py
"""
import os
import sys
import time
from pathlib import Path

try:
    from huggingface_hub import HfApi, create_repo
except ImportError:
    print("Installing huggingface_hub...")
    os.system(f"{sys.executable} -m pip install huggingface_hub -q")
    from huggingface_hub import HfApi, create_repo

HF_TOKEN    = os.environ.get("HF_TOKEN", os.environ["HF_TOKEN"])
HF_USERNAME = "AdiShingote"
REPO_ID     = f"{HF_USERNAME}/amass-smpl-dataset"
PT_DIR      = Path("D:/HumanML3d/amass_pt_parts")

api = HfApi(token=HF_TOKEN)

# ── Create repo (no-op if already exists) ───────────────────────────────────
print(f"Creating / connecting to repo: {REPO_ID}")
api.create_repo(
    repo_id=REPO_ID,
    repo_type="dataset",
    private=True,
    exist_ok=True,
)
print(f"Repo ready: https://huggingface.co/datasets/{REPO_ID}\n")

# ── Upload each .pt file ─────────────────────────────────────────────────────
pt_files = sorted(PT_DIR.glob("*.pt"))
if not pt_files:
    print(f"ERROR: No .pt files found in {PT_DIR}")
    sys.exit(1)

total_bytes = sum(f.stat().st_size for f in pt_files)
uploaded_bytes = 0

print(f"Found {len(pt_files)} files  |  Total: {total_bytes/1024**3:.2f} GB\n")

for i, pt_file in enumerate(pt_files, 1):
    size_mb = pt_file.stat().st_size / 1024**2
    print(f"[{i}/{len(pt_files)}] Uploading {pt_file.name}  ({size_mb:.1f} MB)...")
    t0 = time.time()

    try:
        api.upload_file(
            path_or_fileobj=str(pt_file),
            path_in_repo=pt_file.name,
            repo_id=REPO_ID,
            repo_type="dataset",
        )
        elapsed = time.time() - t0
        speed_mb = size_mb / elapsed
        uploaded_bytes += pt_file.stat().st_size
        pct = uploaded_bytes / total_bytes * 100
        print(f"  OK  {elapsed:.0f}s  ({speed_mb:.1f} MB/s)  |  overall {pct:.1f}% done")
    except Exception as e:
        print(f"  FAILED: {e}")
        print("  Will continue with remaining files — re-run script to retry failed files.")

print("\n" + "="*60)
print("Upload complete!")
print(f"Dataset URL: https://huggingface.co/datasets/{REPO_ID}")
print("="*60)
print("\nNext: SCP the pipeline scripts to Vast.ai, then run:")
print("  bash /workspace/setup_vastai_envs.sh")
print("  python /workspace/vastai_retarget_pipeline.py")
