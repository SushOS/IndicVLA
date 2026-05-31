"""
build_pt_parts.py
-----------------
Builds one amass_smpl_<DATASET>.pt file per AMASS subdataset.
Each is small enough to fit in RAM, unlike the full 13,879-motion package.

On Vast.ai run extract_retargeting_input_keypoints_from_packaged_motionlib.py
once per .pt file into the same keypoints output directory.

Usage:
    python build_pt_parts.py
"""
import subprocess
import sys
from pathlib import Path
import time

AMASS_DATA  = Path("D:/HumanML3d/amass_data")
OUTPUT_DIR  = Path("D:/HumanML3d/amass_pt_parts")
PYTHON      = Path("D:/HumanML3d/protomotions_env/Scripts/python.exe")
MOTION_LIB  = Path("D:/HumanML3d/ProtoMotions/protomotions/components/motion_lib.py")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Datasets sorted smallest → largest (so we can verify early ones quickly)
datasets = sorted(
    [d for d in AMASS_DATA.iterdir() if d.is_dir()],
    key=lambda d: sum(1 for _ in d.rglob("*.motion"))
)

total = len(datasets)
for i, dataset_dir in enumerate(datasets, 1):
    name = dataset_dir.name
    out_pt = OUTPUT_DIR / f"{name}.pt"

    motion_count = sum(1 for _ in dataset_dir.rglob("*.motion"))
    if motion_count == 0:
        print(f"[{i}/{total}] SKIP {name} — no .motion files")
        continue

    if out_pt.exists():
        print(f"[{i}/{total}] SKIP {name} — {out_pt.name} already exists")
        continue

    print(f"\n[{i}/{total}] Building {out_pt.name}  ({motion_count} motions)...")
    t0 = time.time()

    result = subprocess.run(
        [str(PYTHON), str(MOTION_LIB),
         "--motion-path", str(dataset_dir),
         "--output-file", str(out_pt)],
        env={**__import__("os").environ, "PYTHONPATH": "D:/HumanML3d/ProtoMotions"},
        capture_output=False,   # print live
    )

    elapsed = time.time() - t0
    size_mb = out_pt.stat().st_size / 1024 / 1024 if out_pt.exists() else 0

    if result.returncode == 0 and out_pt.exists():
        print(f"  -> Done in {elapsed:.0f}s  |  {size_mb:.1f} MB  |  {out_pt.name}")
    else:
        print(f"  -> FAILED (exit {result.returncode}) — skipping")

print("\n=== All done ===")
for pt in sorted(OUTPUT_DIR.glob("*.pt")):
    print(f"  {pt.name:40s}  {pt.stat().st_size/1024/1024:7.1f} MB")
