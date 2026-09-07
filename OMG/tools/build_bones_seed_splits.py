#!/usr/bin/env python3
"""Build leakage-safe, diversity-stratified splits from the BONES-SEED metadata.

WHY THE OBVIOUS SPLIT IS WRONG
------------------------------
`seed_metadata_v004__FULLY_TRANSLATED.csv` has 142,220 rows but only 7,620 distinct
`content_name` values and only **5,272 distinct captions**. Three structures make a naive
row-level or content_name-level split leak:

1. Mirror pairs (`body_check` / `body_check_M`) carry the SAME caption. Splitting on
   content_name puts identical text in train and test.
2. 2,338 captions are shared by more than one content_name (covering 4,686 of them), so
   even mirror-aware grouping leaks.
3. `Baseline` is 16.1% of rows from just 26 concepts (~880 rows each) with ordinary
   locomotion captions. Row-proportional sampling floods the set with near-duplicates.

Because the Hindi caption is derived from the English one, ANY caption appearing in both
train and test means the evaluated text was memorised. So the atomic unit here is a
**super-group**: content_names transitively merged by shared caption OR shared mirror base.
Splits are drawn over super-groups; rows are only ever sampled inside an assigned group.

ALLOCATION
----------
Row counts are a bad diversity signal (see Baseline). Concept counts alone would
under-represent genuinely large families like locomotion. Budget per category therefore
uses the geometric mean of volume and diversity:

    weight(category) = sqrt(n_rows * n_supergroups)

Inside a category the budget is spread across super-groups round-robin (widest coverage
first), and inside a super-group rows are picked to balance mirror state and spread across
distinct actors.

OUTPUT
------
train (default 30,000 rows) plus val / test / annotation-reserve drawn from DISJOINT
super-groups, so the held-out sets share no caption with training.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MIRROR_SUFFIX = re.compile(r"_M$")

ID_COLS = [
    "move_name", "filename", "move_g1_path", "move_duration_frames",
    "package", "category", "content_type_of_movement", "content_body_position",
    "content_horizontal_move", "content_vertical_move", "content_props",
    "content_complex_action", "content_repeated_action",
    "is_neutral", "is_mirror", "take_actor", "actor_uid", "actor_gender",
    "content_name",
]
DESC_COLS = [f"content_natural_desc_{i}" for i in (1, 2, 3, 4)]
TRANS_COLS = [f"content_natural_desc_{i}_{lg}" for i in (1, 2, 3, 4)
              for lg in ("hi", "bn", "ta", "te")]


class DisjointSet:
    def __init__(self, items):
        self.parent = {x: x for x in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def build_super_groups(concepts: pd.DataFrame) -> pd.Series:
    """content_name -> super_group id, merging on shared caption OR shared mirror base."""
    ds = DisjointSet(concepts["content_name"].tolist())
    for key in ("cap_key", "mirror_base"):
        for _, grp in concepts.groupby(key)["content_name"]:
            names = grp.tolist()
            for other in names[1:]:
                ds.union(names[0], other)
    roots = {n: ds.find(n) for n in concepts["content_name"]}
    codes = {r: i for i, r in enumerate(sorted(set(roots.values())))}
    return pd.Series({n: codes[r] for n, r in roots.items()}, name="super_group")


def assign_splits(groups: pd.DataFrame, fracs: dict[str, float], seed: int) -> pd.Series:
    """Stratify super-groups across splits WITHIN each category, largest-first for stability."""
    rng = np.random.default_rng(seed)
    out = {}
    for cat, sub in groups.groupby("category", sort=True):
        sub = sub.sample(frac=1.0, random_state=int(rng.integers(1 << 31)))
        sub = sub.sort_values("n_rows", ascending=False, kind="mergesort")
        # deal round-robin weighted by target fraction: keeps small categories represented
        names, quota = list(fracs), {k: 0.0 for k in fracs}
        for gid in sub["super_group"]:
            pick = min(names, key=lambda k: quota[k] / max(fracs[k], 1e-9))
            out[gid] = pick
            quota[pick] += 1.0
    return pd.Series(out, name="split")


def allocate_category_budget(stats: pd.DataFrame, budget: int) -> dict[str, int]:
    """weight = sqrt(rows * supergroups): geometric mean of volume and diversity."""
    w = np.sqrt(stats["n_rows"].to_numpy(float) * stats["n_groups"].to_numpy(float))
    w = w / w.sum()
    raw = w * budget
    alloc = np.floor(raw).astype(int)
    # hand out the remainder by largest fractional part
    for i in np.argsort(-(raw - alloc))[: budget - int(alloc.sum())]:
        alloc[i] += 1
    # never ask a category for more rows than it has
    alloc = np.minimum(alloc, stats["n_rows"].to_numpy())
    return dict(zip(stats.index, alloc.tolist()))


def sample_rows(df: pd.DataFrame, group_ids: list[int], budget: int, seed: int) -> pd.Index:
    """Round-robin across super-groups; inside a group balance mirror state and actors."""
    rng = np.random.default_rng(seed)
    pools: dict[int, list] = {}
    for gid in group_ids:
        sub = df[df["super_group"] == gid]
        if sub.empty:
            continue
        # Rotate actors first, then INTERLEAVE mirror states (F,T,F,T...) so that any
        # truncation of this pool keeps both the 50/50 mirror balance of the source and a
        # spread of actors. Sorting by is_mirror instead of interleaving skews ~67/33,
        # which would bias the left/right test cohort.
        sub = sub.sample(frac=1.0, random_state=int(rng.integers(1 << 31)))
        sub = sub.assign(_a=sub.groupby(["is_mirror", "actor_uid"]).cumcount())
        sub = sub.sort_values(["_a"], kind="mergesort")
        sub = sub.assign(_m=sub.groupby("is_mirror").cumcount())
        sub = sub.sort_values(["_m", "is_mirror"], kind="mergesort")
        pools[gid] = sub.index.tolist()

    picked: list = []
    order = sorted(pools, key=lambda g: -len(pools[g]))
    while len(picked) < budget and order:
        progressed = False
        for gid in list(order):
            if not pools[gid]:
                order.remove(gid)
                continue
            picked.append(pools[gid].pop(0))
            progressed = True
            if len(picked) >= budget:
                break
        if not progressed:
            break
    return pd.Index(picked)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="seed_metadata_v004__FULLY_TRANSLATED.csv.xls")
    ap.add_argument("--out-dir", default="splits_bones_seed_30k")
    ap.add_argument("--train-rows", type=int, default=30000)
    ap.add_argument("--val-rows", type=int, default=3000)
    ap.add_argument("--test-rows", type=int, default=3000)
    ap.add_argument("--reserve-rows", type=int, default=3000,
                    help="held-out pool reserved for native Hindi annotation (plan 6.1)")
    ap.add_argument("--min-frames", type=int, default=70,
                    help="10 history + 60 predicted horizon")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    use = ID_COLS + DESC_COLS + TRANS_COLS
    print(f"reading {a.csv} ...", flush=True)
    df = pd.read_csv(a.csv, usecols=use, low_memory=False)
    print(f"  {len(df):,} rows", flush=True)

    n0 = len(df)
    df = df[pd.to_numeric(df["move_duration_frames"], errors="coerce") >= a.min_frames]
    df = df[df["content_natural_desc_1"].astype(str).str.strip() != ""]
    df = df[df["content_natural_desc_1_hi"].notna()]
    print(f"  {len(df):,} usable after >= {a.min_frames} frames + caption + hi-translation "
          f"({n0 - len(df):,} dropped)", flush=True)

    df["cap_key"] = df["content_natural_desc_1"].astype(str).str.strip().str.lower()
    df["mirror_base"] = df["content_name"].astype(str).str.replace(MIRROR_SUFFIX, "", regex=True)

    concepts = df.drop_duplicates("content_name")[
        ["content_name", "cap_key", "mirror_base", "category"]]
    sg = build_super_groups(concepts)
    df["super_group"] = df["content_name"].map(sg)
    print(f"  {df['content_name'].nunique():,} content_names -> {df['super_group'].nunique():,} "
          f"leakage-safe super-groups", flush=True)

    groups = (df.groupby("super_group")
                .agg(n_rows=("move_name", "size"),
                     category=("category", lambda s: s.mode().iat[0]))
                .reset_index())

    total = a.train_rows + a.val_rows + a.test_rows + a.reserve_rows
    fracs = {"train": a.train_rows / total, "val": a.val_rows / total,
             "test": a.test_rows / total, "reserve": a.reserve_rows / total}
    groups["split"] = groups["super_group"].map(assign_splits(groups, fracs, a.seed))

    budgets = {"train": a.train_rows, "val": a.val_rows,
               "test": a.test_rows, "reserve": a.reserve_rows}
    frames, report = [], {}
    for split, budget in budgets.items():
        gsub = groups[groups["split"] == split]
        stats = (df[df["super_group"].isin(gsub["super_group"])]
                 .groupby("category")
                 .agg(n_rows=("move_name", "size"), n_groups=("super_group", "nunique")))
        alloc = allocate_category_budget(stats, budget)

        idx = []
        for cat, cat_budget in alloc.items():
            if cat_budget <= 0:
                continue
            gids = gsub[gsub["category"] == cat]["super_group"].tolist()
            idx.extend(sample_rows(df[df["category"] == cat], gids, cat_budget, a.seed))
        part = df.loc[pd.Index(idx)].copy()
        part["split"] = split
        frames.append(part)
        report[split] = {
            "rows": int(len(part)),
            "super_groups": int(part["super_group"].nunique()),
            "distinct_captions": int(part["cap_key"].nunique()),
            "categories": int(part["category"].nunique()),
            "types_of_movement": int(part["content_type_of_movement"].nunique()),
            "actors": int(part["actor_uid"].nunique()),
        }
        print(f"  {split:8s} {len(part):6,} rows | {part['super_group'].nunique():5,} groups | "
              f"{part['cap_key'].nunique():5,} captions", flush=True)

    all_parts = pd.concat(frames, ignore_index=True)

    # ---- hard leakage assertions: no super-group and no caption may cross a split
    xg = all_parts.groupby("super_group")["split"].nunique()
    xc = all_parts.groupby("cap_key")["split"].nunique()
    assert (xg == 1).all(), f"super-group leak: {(xg > 1).sum()} groups span splits"
    assert (xc == 1).all(), f"CAPTION LEAK: {(xc > 1).sum()} captions span splits"
    print("  leakage check PASSED (0 shared super-groups, 0 shared captions)", flush=True)

    drop = ["cap_key", "mirror_base"]
    for split in budgets:
        p = out / f"{split}.csv"
        all_parts[all_parts["split"] == split].drop(columns=drop).to_csv(p, index=False)
        print(f"  wrote {p}", flush=True)

    report["_meta"] = {
        "source_csv": a.csv, "seed": a.seed, "min_frames": a.min_frames,
        "source_rows": int(n0), "usable_rows": int(len(df)),
        "total_super_groups": int(df["super_group"].nunique()),
        "grouping": "content_names merged transitively by shared caption OR mirror base",
        "allocation": "per-category weight = sqrt(n_rows * n_super_groups)",
    }
    (out / "split_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out}/split_report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
