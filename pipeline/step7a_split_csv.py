"""
split_csv.py — splits g1_dataset_robot_v2.csv into N chunk files.
Already-translated cells are preserved in each chunk.
Run ONCE before launching parallel workers.

Usage:  python split_csv.py [N]   (default N=4)
"""
import sys
import pandas as pd
from pathlib import Path

CSV  = "D:/HumanML3d/g1_dataset_robot_v2.csv"
OUTD = Path("D:/HumanML3d")

n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
df = pd.read_csv(CSV, dtype=str)
total = len(df)
chunk_size = (total + n - 1) // n          # ceiling division

print(f"Splitting {total:,} rows into {n} chunks of ~{chunk_size} rows each")
for i in range(n):
    start = i * chunk_size
    end   = min(start + chunk_size, total)
    chunk = df.iloc[start:end].copy()
    out   = OUTD / f"g1_chunk_{i}.csv"
    chunk.to_csv(out, index=False)
    print(f"  chunk {i}: rows {start:,}–{end-1:,}  ({len(chunk):,} rows)  -> {out.name}")

print("\nDone. Now open 4 terminals and run one command per terminal:")
for i in range(n):
    print(f"  python D:/HumanML3d/translate_captions.py --csv D:/HumanML3d/g1_chunk_{i}.csv")
print("\nWhen ALL 4 finish, run:")
print("  python D:/HumanML3d/merge_chunks.py")
