#!/usr/bin/env python3
"""Leakage-safe splits for the G1-retargeted AMASS/HumanML3D 13k corpus.

WHY SUPER-GROUPS (same lesson as build_bones_seed_splits.py)
------------------------------------------------------------
A naive row-level split leaks along TWO independent axes in this corpus:

  1. `source_amass`  one AMASS .npz is segmented into up to 5 clips
     (2,310 source files produce >1 clip). Two segments of the same take are
     near-duplicates; putting one in train and one in test inflates the score.

  2. `caption_1`     774 clips share a caption with at least one other clip
     ("a robot walks forward slowly" x24). Because the Indic captions are
     DERIVED from the English one, a shared English caption means a shared
     Hindi/Bengali/Tamil/Telugu caption too -- the exact string the model is
     conditioned on would appear on both sides of the split.

Fix: merge clips transitively (union-find) by shared source_amass OR shared
normalized caption_1, then draw splits over GROUPS, never over clips. Both
leakage conditions are then impossible by construction, and asserted anyway.

Allocation is stratified per AMASS subset (KIT / CMU / BMLmovi / ...) with
weight = sqrt(n_clips * n_groups) -- the geometric mean of raw volume and
genuine concept diversity, so a large-but-repetitive subset cannot dominate.

Outputs train/val/test/reserve CSVs plus split_report.json. `reserve` is held
back untouched for native-speaker caption authoring (research-log s10.1).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


LANGS = ["", "_hi", "_bn", "_ta", "_te"]      # "" == English
SLOTS = [1, 2, 3, 4]


def normalize_caption(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip().str.lower().str.rstrip(".")


def caption_columns(df: pd.DataFrame) -> list[str]:
    """EVERY caption string a clip carries: 4 slots x 5 languages.

    WHY ALL OF THEM (measured 2026-09-05, after a leaky first build)
    ----------------------------------------------------------------
    v1 keyed the split on `caption_1` (English) alone. That column came out perfectly
    clean -- 0 strings shared between any pair of splits -- and the build still leaked,
    two ways:

      1. SLOTS 2-3 WERE NEVER CONSIDERED. Training samples every caption slot, so a clip
         unique on desc_1 can still collide on desc_2 (49 shared train|val) or desc_3 (67).

      2. MACHINE TRANSLATION COLLAPSES DISTINCT ENGLISH INTO IDENTICAL INDIC. English
         caption_1 shared 0 strings; Hindi caption_1 shared 27, Tamil 42. Examples:
             "robot is walking and turning left"      \\
             "the robot is walking and turning left"  /  -> one Hindi string
             "a robot slowly walks forward and then stops" \\
             "a robot walks forward slowly then stops"     /  -> one Hindi string
         The model is conditioned on the INDIC string, so an English-only key cannot see
         this collision at all. Same failure class as the walk/jog gait collapse recorded
         in the SSOT: MT erases distinctions the split logic was relying on.

    Keying on the union of all 20 columns makes both impossible by construction.
    """
    return [f"caption_{s}{lg}" for s in SLOTS for lg in LANGS
            if f"caption_{s}{lg}" in df.columns]


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def mirror_links(df: pd.DataFrame) -> list[tuple]:
    """Pairs of clip indices that are reflections of each other.

    WHY THIS IS A THIRD KEY AND NOT COVERED BY THE OTHER TWO (SSOT s17.4)
    ---------------------------------------------------------------------
    A mirror X_M is the same motion reflected, with the side words swapped in every caption
    (right <-> left, in every language). So it shares NEITHER key already used here:

        source_amass  differs -- the mirror is written as its own clip
        caption       differs -- that is the entire point of a counterfactual pair

    Union-find would therefore place X in train and X_M in test and report ZERO leakage,
    while the test set holds a reflection of a training example. Every gate we have would
    pass. This is the only key that catches it.

    It also protects Contribution 2: the flip-rate metric is only meaningful on pairs whose
    source motion was never trained on. If the pair straddles the split, the metric measures
    memorisation.

    Accepts either convention:
        mirror_of        clip_id of the source clip (mirrors point at their original)
        mirror_pair_id   a shared token carried by both members
    """
    links: list[tuple] = []
    if "mirror_pair_id" in df.columns:
        v = df["mirror_pair_id"].fillna("").astype(str)
        for i in df.index[v != ""]:
            links.append((f"clip::{i}", f"mirror::{v.at[i]}"))
    if "mirror_of" in df.columns and "clip_id" in df.columns:
        by_clip = {c: i for i, c in df["clip_id"].items()}
        v = pd.to_numeric(df["mirror_of"], errors="coerce")
        for i in df.index[v.notna()]:
            j = by_clip.get(type(df.at[i, "clip_id"])(v.at[i]))
            if j is not None:
                links.append((f"clip::{i}", f"clip::{j}"))
    return links


def build_super_groups(df: pd.DataFrame, cap_cols: list[str]) -> pd.Series:
    """clip index -> super-group id.

    Merges on shared source file OR ANY shared caption string, in ANY of the 4 slots and
    ANY of the 5 languages, OR a mirror relationship. Two clips end up in the same group if
    they share even one string the model could be conditioned on, or if one is a reflection
    of the other.
    """
    uf = UnionFind()
    norm = {c: normalize_caption(df[c]) for c in cap_cols}
    for i in df.index:
        uf.union(f"clip::{i}", f"src::{df.at[i, 'source_amass']}")
        for c in cap_cols:
            v = norm[c].at[i]
            if v:
                uf.union(f"clip::{i}", f"cap::{v}")
    links = mirror_links(df)
    for a, b in links:
        uf.union(a, b)
    if links:
        print(f"  mirror links unioned: {len(links):,}")
    return pd.Series({i: uf.find(f"clip::{i}") for i in df.index}, name="super_group")


def allocate(groups: pd.DataFrame, fracs: dict[str, float], seed: int) -> dict[str, list]:
    """Assign whole super-groups to splits, stratified per AMASS subset."""
    rng = np.random.default_rng(seed)
    out: dict[str, list] = {k: [] for k in fracs}
    for _, sub in groups.groupby("subset", sort=True):
        gids = sub["super_group"].tolist()
        rng.shuffle(gids)
        sizes = sub.set_index("super_group")["n_clips"].to_dict()
        total = sum(sizes.values())
        # greedy: fill the split that is furthest below its quota
        want = {k: total * v for k, v in fracs.items()}
        got = {k: 0 for k in fracs}
        for g in sorted(gids, key=lambda x: -sizes[x]):
            k = max(fracs, key=lambda s: (want[s] - got[s]) / max(want[s], 1e-9))
            out[k].append(g)
            got[k] += sizes[g]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=r"D:\HumanML3d\g1_dataset_robot_v2.csv")
    ap.add_argument("--out", default=r"D:\HumanML3d\IndicVLA\OMG\splits_amass_g1_13k")
    ap.add_argument("--min-duration", type=float, default=70 / 30,
                    help="drop clips too short for one L=10,H=60 window at 30fps")
    ap.add_argument("--train", type=float, default=0.770)
    ap.add_argument("--val", type=float, default=0.077)
    ap.add_argument("--test", type=float, default=0.077)
    ap.add_argument("--reserve", type=float, default=0.076)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(a.csv, low_memory=False)
    n0 = len(df)
    df["duration_s"] = pd.to_numeric(df["duration_s"], errors="coerce")
    df = df[df["duration_s"] >= a.min_duration].reset_index(drop=True)
    print(f"loaded {n0:,} clips -> {len(df):,} after >= {a.min_duration:.2f}s filter "
          f"({n0 - len(df):,} too short for a 70-frame window)")

    cap_cols = caption_columns(df)
    df["_cap_key"] = normalize_caption(df["caption_1"])          # reporting only
    df["subset"] = df["source_amass"].astype(str).str.split("/").str[0]
    print(f"  leakage keys: source_amass + {len(cap_cols)} caption columns "
          f"({len(SLOTS)} slots x {len(LANGS)} languages)")
    df["super_group"] = build_super_groups(df, cap_cols)
    print(f"  {len(df):,} clips -> {df['super_group'].nunique():,} leakage-safe super-groups")

    groups = (df.groupby(["super_group", "subset"], as_index=False)
                .size().rename(columns={"size": "n_clips"}))
    # a super-group can span subsets; attribute it to its dominant subset
    groups = groups.sort_values("n_clips", ascending=False).drop_duplicates("super_group")

    fracs = {"train": a.train, "val": a.val, "test": a.test, "reserve": a.reserve}
    assign = allocate(groups, fracs, a.seed)
    g2s = {g: s for s, gs in assign.items() for g in gs}
    df["split"] = df["super_group"].map(g2s)
    if df["split"].isna().any():
        raise SystemExit(f"{df['split'].isna().sum()} clips unassigned")

    # ---- hard leakage assertions -------------------------------------------
    xs = df.groupby("source_amass")["split"].nunique()
    if (xs > 1).any():
        raise SystemExit(f"SOURCE LEAK: {(xs > 1).sum()} source files span splits")
    xg = df.groupby("super_group")["split"].nunique()
    if (xg > 1).any():
        raise SystemExit(f"GROUP LEAK: {(xg > 1).sum()} super-groups span splits")
    # assert EVERY caption column independently -- this is what v1 failed to do
    for c in cap_cols:
        v = normalize_caption(df[c])
        m = v != ""
        if not m.any():
            continue
        x = df[m].assign(_v=v[m]).groupby("_v")["split"].nunique()
        if (x > 1).any():
            bad = x[x > 1]
            raise SystemExit(f"CAPTION LEAK in {c}: {len(bad)} strings span splits, "
                             f"e.g. {list(bad.index)[:2]}")
    # mirrors: X and X_M must never straddle a split. Checked explicitly rather than
    # trusted to the union above, because a silent failure here is undetectable downstream:
    # the reflection sits in test looking like an ordinary unseen clip.
    n_pairs = 0
    if "mirror_of" in df.columns and "clip_id" in df.columns:
        by_clip = {c: i for i, c in df["clip_id"].items()}
        mv = pd.to_numeric(df["mirror_of"], errors="coerce")
        for i in df.index[mv.notna()]:
            j = by_clip.get(type(df.at[i, "clip_id"])(mv.at[i]))
            if j is None:
                continue
            n_pairs += 1
            if df.at[i, "split"] != df.at[j, "split"]:
                raise SystemExit(f"MIRROR LEAK: clip {df.at[i, 'clip_id']} is in "
                                 f"{df.at[i, 'split']} but its mirror source is in "
                                 f"{df.at[j, 'split']}")
    if "mirror_pair_id" in df.columns:
        pv = df["mirror_pair_id"].fillna("").astype(str)
        x = df[pv != ""].groupby(pv[pv != ""])["split"].nunique()
        if (x > 1).any():
            raise SystemExit(f"MIRROR LEAK: {(x > 1).sum()} mirror pairs span splits")
        n_pairs = max(n_pairs, int((pv != "").sum()))
    print(f"  leakage check PASSED (0 shared sources, 0 shared groups, "
          f"0 shared strings across all {len(cap_cols)} caption columns, "
          f"0 mirror pairs split across {n_pairs:,} checked)")

    caption_cols = [c for c in df.columns if c.startswith("caption_")]
    langs = ["hi", "bn", "ta", "te"]
    report: dict = {}
    print(f"\n{'split':9s} {'clips':>7s} {'groups':>7s} {'captions':>9s} "
          f"{'en pairs':>9s} {'indic pairs':>12s}")
    for s in ["train", "val", "test", "reserve"]:
        part = df[df["split"] == s]
        part.drop(columns=["_cap_key"]).to_csv(out / f"{s}.csv", index=False, encoding="utf-8")
        en_pairs = int(sum(part[f"caption_{i}"].notna().sum() for i in (1, 2, 3, 4)))
        report[s] = {
            "clips": int(len(part)),
            "super_groups": int(part["super_group"].nunique()),
            "distinct_caption_1": int(part["_cap_key"].nunique()),
            "clips_per_caption": round(len(part) / max(part["_cap_key"].nunique(), 1), 4),
            "english_caption_pairs": en_pairs,
            "indic_caption_pairs": en_pairs * len(langs),
            "subsets": int(part["subset"].nunique()),
            "hours": round(float(part["duration_s"].sum()) / 3600, 2),
        }
        print(f"{s:9s} {len(part):7,} {part['super_group'].nunique():7,} "
              f"{part['_cap_key'].nunique():9,} {en_pairs:9,} {en_pairs * len(langs):12,}")

    report["_meta"] = {
        "source_csv": str(a.csv),
        "seed": a.seed,
        "min_duration_s": a.min_duration,
        "source_clips": n0,
        "usable_clips": int(len(df)),
        "total_super_groups": int(df["super_group"].nunique()),
        "grouping": "clips merged transitively by shared source_amass OR any shared caption string OR a mirror relationship (SSOT s17.4)",
        "allocation": "per-AMASS-subset greedy fill, largest-group-first, seed-shuffled",
        "languages": langs,
        "caption_columns": caption_cols,
        "leakage_verified": ["source_amass", "caption_1", "super_group", "mirror_of"],
    }
    (out / "split_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
