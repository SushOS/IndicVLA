"""
merge_chunks.py — merges g1_chunk_*.csv back into g1_dataset_robot_v2.csv.
Run AFTER all parallel translate workers have finished.

Usage:  python merge_chunks.py
"""
import sys
import pandas as pd
from pathlib import Path

OUTD  = Path("D:/HumanML3d")
FINAL = OUTD / "g1_dataset_robot_v2.csv"

chunks = sorted(OUTD.glob("g1_chunk_*.csv"))
if not chunks:
    print("ERROR: No g1_chunk_*.csv files found in D:/HumanML3d/")
    sys.exit(1)

print(f"Found {len(chunks)} chunk files:")
dfs = []
for p in chunks:
    df = pd.read_csv(p, dtype=str)
    print(f"  {p.name}  ({len(df):,} rows)")
    dfs.append(df)

merged = pd.concat(dfs, ignore_index=True)

# Sanity check
TRANS_COLS = [f"caption_{c}_{l}" for c in range(1, 5) for l in ("hi", "bn", "ta")]
print(f"\nMerged: {len(merged):,} rows  |  {len(merged.columns)} columns")
print("Translation fill rates:")
for col in TRANS_COLS:
    if col in merged.columns:
        filled = merged[col].apply(lambda v: pd.notna(v) and str(v).strip() != "").sum()
        pct    = 100 * filled / len(merged)
        print(f"  {col:20s}  {filled:>7,} / {len(merged):>7,}  ({pct:.1f}%)")

merged.to_csv(FINAL, index=False)
print(f"\nSaved -> {FINAL}")

# Optional: remove chunk files
answer = input("Delete chunk files? [y/N] ").strip().lower()
if answer == "y":
    for p in chunks:
        p.unlink()
    print("Chunk files deleted.")
