#!/usr/bin/env python3
"""Extract a Run-3 corpus from THU-MARS/OMG-Data into the pipeline's existing npz contract.

WHY THIS SHAPE
--------------
OMG-Data publishes 798,181 episodes already in Unitree G1 `qpos_36` at 30 fps
(`observation.state[36]` = root_pos(3) + root_quat(4) + joints(29)). That is exactly the
input `convert_g1_npz_to_omg125.py` consumes, so this writer emits
`base_frame_pos / base_frame_wxyz / joint_angles` npz files and a manifest CSV, and the
ALREADY-VALIDATED converter and split builder run downstream unchanged.

That is deliberate. This project has shipped four shape-correct / scale-wrong bugs
(SSOT #14, #15, #21, #24), every one of them at a boundary where a new data path met an old
one. Writing npz and reusing the validated converter adds no new decode path.

Two whole pipeline stages disappear here:
  * no SMPL->G1 retargeting -- OMG-Data is natively G1;
  * no SUBSET_FPS table -- OMG-Data is uniformly 30 fps, so the per-subset fps measurement
    that produced a wrong --src-fps flag and a basename-collision bug is simply not needed.

CORPUS (SSOT s17.3, Option D): humanml + omomo + 100style + motionllama
  56,338 episodes | 55,286 four-language-safe | 25,811 leakage-safe groups | all real capture
MotionGV is excluded: 0.0% ground contact across 10 rendered episodes against AMASS's 11.6%.
AMASS is excluded: 69% overlap with Run 2 and fewer groups than Option D despite more rows.

THE GATE IS TELUGU, NOT HINDI (SSOT s17.2)
  measured MuRIL fertility -- bn 1.000, ta 1.095, hi 1.158, te 1.278
  => English must be <= 39 t5 tokens for all four translations to fit MuRIL's 50-token window.

GROUPING FOR LEAKAGE
  `source_amass` is set to "<dataset>/<source motion>" with the __seg suffix stripped, so every
  segment of one source motion unions into a single super-group in build_amass_g1_splits.py.
  Mirror pairs need a THIRD key (`mirror::<pair_id>`) which the mirror stage adds -- X and X_M
  share neither source_id nor caption, so without it a reflection of a training clip can land
  in test and pass every existing gate.

MODES
  plan     report episode counts, token-gate survival and exact download size. No download.
  extract  download only the data files the selection touches, write npz + manifest.csv.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = "THU-MARS/OMG-Data"
OPTION_D = ["humanml", "omomo", "100style", "motionllama"]
FERTILITY = {"bn": 1.000, "ta": 1.095, "hi": 1.158, "te": 1.278}   # measured, SSOT s17.2
N_JOINTS = 29


def load_episode_meta(cache: Path, token: str) -> pd.DataFrame:
    """The 8 episode shards, cached. ~800K rows, cheap enough to hold in memory."""
    from huggingface_hub import hf_hub_download

    fs = []
    for k in range(8):
        fs.append(hf_hub_download(REPO, "meta/episodes/chunk-000/file-%03d.parquet" % k,
                                  repo_type="dataset", token=token, local_dir=str(cache)))
    d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    d["base"] = d["omg/dataset"].astype(str).str.replace(r"_(train|val|test)$", "", regex=True)
    d["split"] = d["omg/dataset"].astype(str).str.extract(r"_(train|val|test)$")[0]
    d["motion"] = d["omg/source_id"].astype(str).str.split("__seg").str[0]
    d["caption"] = d["tasks"].apply(lambda t: t[0] if len(t) else "")
    return d


def apply_token_gate(d: pd.DataFrame, gate: int, sample_only: int | None = None):
    """Keep captions whose English fits ALL FOUR Indic scripts inside MuRIL's 50 tokens."""
    from transformers import AutoTokenizer

    tk = AutoTokenizer.from_pretrained("t5-base", legacy=True)
    caps = d["caption"].astype(str)
    uniq = pd.Index(caps.unique())
    if sample_only and len(uniq) > sample_only:          # plan mode: estimate, do not tokenise 500K
        u = pd.Index(np.random.default_rng(0).choice(uniq, sample_only, replace=False))
        lens = {c: len(tk(c, add_special_tokens=True)["input_ids"]) for c in u}
        rate = float(np.mean([v <= gate for v in lens.values()]))
        return None, rate
    lens = {c: len(tk(c, add_special_tokens=True)["input_ids"]) for c in uniq}
    keep = caps.map(lens) <= gate
    return keep, float(keep.mean())


def qpos_to_npz_fields(q: np.ndarray) -> dict:
    """qpos_36 -> the converter's npz contract. Validated, not assumed."""
    if q.ndim != 2 or q.shape[1] != 36:
        raise ValueError("expected (T,36) qpos, got %s" % (q.shape,))
    if not np.isfinite(q).all():
        raise ValueError("non-finite values in qpos")
    quat = q[:, 3:7]
    n = np.linalg.norm(quat, axis=1)
    if np.abs(n - 1.0).max() > 1e-2:
        raise ValueError("root quaternion not unit norm (max dev %.3e)" % np.abs(n - 1).max())
    return {"base_frame_pos": q[:, 0:3].astype(np.float32),
            "base_frame_wxyz": (quat / n[:, None]).astype(np.float32),
            "joint_angles": q[:, 7:7 + N_JOINTS].astype(np.float32)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["plan", "extract"], default="plan")
    ap.add_argument("--datasets", default=",".join(OPTION_D))
    ap.add_argument("--gate", type=int, default=39, help="English t5 tokens; Telugu-bound")
    ap.add_argument("--out", default="D:/HumanML3d/omg_run3")
    ap.add_argument("--cache", default="D:/HumanML3d/omg_cache",
                    help="parquet cache, needs ~14 GB -- keep it off C:")
    ap.add_argument("--token", default=None)
    ap.add_argument("--limit", type=int, default=0, help="cap episodes, for a pilot")
    ap.add_argument("--min-frames", type=int, default=70, help="L+H; shorter cannot window")
    ap.add_argument("--no-resume", action="store_true",
                    help="re-extract everything, ignoring npz already on disk")
    a = ap.parse_args()

    token = a.token or __import__("os").environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("pass --token or set HF_TOKEN")
    cache = Path(a.cache); cache.mkdir(parents=True, exist_ok=True)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    bases = [x.strip() for x in a.datasets.split(",") if x.strip()]

    print("reading episode metadata ...", flush=True)
    d = load_episode_meta(cache, token)
    sel = d[d["base"].isin(bases) & d["omg/has_text"] & (d["caption"].str.len() > 0)].copy()
    sel = sel[sel["length"] >= a.min_frames]
    print("  %d episodes in %s (>= %d frames, has_text)" % (len(sel), bases, a.min_frames))

    from huggingface_hub import HfApi
    api = HfApi(token=token)
    sizes = {f.rfilename: (f.size or 0)
             for f in api.repo_info(REPO, repo_type="dataset", files_metadata=True).siblings}
    files = sorted(sel["data/file_index"].unique())
    dl = sum(sizes.get("data/chunk-000/file-%03d.parquet" % f, 0) for f in files)

    if a.mode == "plan":
        _, rate = apply_token_gate(sel, a.gate, sample_only=8000)
        print("\n%-14s %9s %9s %10s %11s" % ("dataset", "episodes", "motions", "hours", "files"))
        print("-" * 58)
        for b in bases:
            s = sel[sel["base"] == b]
            print("%-14s %9d %9d %10.1f %11d"
                  % (b, len(s), s["motion"].nunique(), s["length"].sum() / 30 / 3600,
                     s["data/file_index"].nunique()))
        print("-" * 58)
        print("%-14s %9d %9d %10.1f %11d"
              % ("TOTAL", len(sel), sel["motion"].nunique(),
                 sel["length"].sum() / 30 / 3600, len(files)))
        print("\n  four-language gate <= %d t5 tokens -> %.1f%% survive (~%d episodes)"
              % (a.gate, 100 * rate, int(len(sel) * rate)))
        print("  download: %d of 61 data files = %.1f GB" % (len(files), dl / 1e9))
        print("  splits (OMG, pre-repair):", sel["split"].value_counts().to_dict())
        print("\n  run with --mode extract to download and write npz + manifest")
        return 0

    print("\napplying four-language token gate (<= %d t5 tokens) ..." % a.gate, flush=True)
    keep, rate = apply_token_gate(sel, a.gate)
    sel = sel[keep].copy()
    # itertuples() renames columns that are not valid Python identifiers, so "omg/source_id"
    # silently becomes a positional name and r.omg/source_id is unreachable. Rename first.
    sel = sel.rename(columns={"omg/source_id": "omg_source_id",
                              "omg/dataset": "omg_dataset"})
    print("  %d episodes survive (%.1f%%)" % (len(sel), 100 * rate))
    if a.limit:
        sel = sel.groupby("base", group_keys=False).apply(
            lambda g: g.head(max(1, int(a.limit * len(g) / len(sel))))).reset_index(drop=True)
        print("  pilot cap: %d episodes" % len(sel))

    npz_dir = out / "npz"; npz_dir.mkdir(parents=True, exist_ok=True)
    want = {int(e): r for e, r in zip(sel["episode_index"], sel.itertuples())}

    def npz_path_for(r, ep):
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", str(r.omg_source_id))
        return npz_dir / ("%s/%s__ep%d.npz" % (r.base, safe, ep))

    def manifest_row(r, ep, n_frames, dst):
        return {"clip_id": ep,
                # segments of one source motion must union into ONE super-group
                "source_amass": "%s/%s" % (r.base, r.motion),
                "npz_path": str(dst.resolve()),
                "start_s": 0.0, "end_s": round(n_frames / 30.0, 4),
                "duration_s": round(n_frames / 30.0, 4),
                "caption_1": r.caption,
                "omg_dataset": r.omg_dataset, "omg_source_id": r.omg_source_id,
                "omg_split": r.split, "frames": n_frames}

    # RESUME. Without it a pause costs every episode already written, and this run was in
    # fact paused at ~28k of 45k. An npz counts as done only if it LOADS and its frame count
    # matches the metadata: a process killed mid-write can leave a truncated file, and a
    # silently truncated motion entering the corpus is exactly the failure class this
    # project keeps paying for.
    resume_rows, todo, bad = [], {}, 0
    if a.no_resume:
        todo = dict(want)
        print("--no-resume: re-extracting all %d episodes" % len(todo), flush=True)
    else:
        print("scanning existing npz for resume ...", flush=True)
        for ep, r in want.items():
            dst = npz_path_for(r, ep)
            if not dst.exists() or dst.stat().st_size == 0:
                todo[ep] = r
                continue
            try:
                with np.load(dst) as z:
                    n = int(z["base_frame_pos"].shape[0])
                if n != int(r.length):
                    raise ValueError("frame count mismatch")
            except Exception:                                         # noqa: BLE001
                bad += 1
                todo[ep] = r
                continue
            resume_rows.append(manifest_row(r, ep, int(r.length), dst))
        print("  %d already valid, %d to extract%s"
              % (len(resume_rows), len(todo),
                 (", %d truncated/unreadable -> redo" % bad) if bad else ""), flush=True)

    file_of = dict(zip(sel["episode_index"].astype(int), sel["data/file_index"].astype(int)))
    files = sorted({file_of[ep] for ep in todo}) if todo else []
    print("\nextracting from %d data files ..." % len(files), flush=True)

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    rows = list(resume_rows)
    stats = {"written": 0, "bad_qpos": 0, "short": 0, "frame_mismatch": 0,
             "resumed": len(resume_rows)}
    for fi in files:
        name = "data/chunk-000/file-%03d.parquet" % fi
        p = hf_hub_download(REPO, name, repo_type="dataset", token=token, local_dir=str(cache))
        pf = pq.ParquetFile(p)
        need = {ep for ep in todo if file_of[ep] == fi}
        if not need:
            continue
        buf: dict[int, list] = {}
        for rg in range(pf.num_row_groups):
            t = pf.read_row_group(rg, columns=["episode_index", "frame_index",
                                               "observation.state"]).to_pandas()
            t = t[t["episode_index"].isin(need)]
            for ep, g in t.groupby("episode_index"):
                buf.setdefault(int(ep), []).append(g)
        for ep, parts in buf.items():
            g = pd.concat(parts).sort_values("frame_index")
            q = np.stack(g["observation.state"].to_numpy()).astype(np.float32)
            r = todo[ep]
            if len(q) != int(r.length):
                stats["frame_mismatch"] += 1
                continue
            if len(q) < a.min_frames:
                stats["short"] += 1
                continue
            try:
                fields = qpos_to_npz_fields(q)
            except ValueError:
                stats["bad_qpos"] += 1
                continue
            safe = re.sub(r"[^A-Za-z0-9_.-]", "_", str(r.omg_source_id))
            rel = "%s/%s__ep%d.npz" % (r.base, safe, ep)
            dst = npz_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dst, **fields)
            rows.append(manifest_row(r, ep, len(q), dst))
            stats["written"] += 1
        print("  file-%03d -> %d episodes (running total %d)" % (fi, len(buf), stats["written"]),
              flush=True)

    man = pd.DataFrame(rows)
    man.to_csv(out / "manifest.csv", index=False, encoding="utf-8")
    rep = {"repo": REPO, "datasets": bases, "gate_t5_tokens": a.gate,
           "fertility_measured": FERTILITY, "episodes_written": stats["written"],
           "source_motions": int(man["source_amass"].nunique()) if len(man) else 0,
           "hours": round(float(man["frames"].sum()) / 30 / 3600, 2) if len(man) else 0.0,
           "omg_splits": man["omg_split"].value_counts().to_dict() if len(man) else {},
           "rejected": {k: v for k, v in stats.items() if k != "written"},
           "note": "start_s=0 per episode: OMG-Data is already segmented, so the converter's "
                   "clip-slicing is a no-op and SUBSET_FPS is unnecessary (uniform 30 fps)."}
    (out / "extract_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print("\nwrote %d npz + manifest.csv to %s" % (stats["written"], out))
    print("  source motions: %d | hours: %.1f" % (rep["source_motions"], rep["hours"]))
    print("  rejected:", rep["rejected"])
    print("\nnext: translate caption_1 -> hi/bn/ta/te, then build_amass_g1_splits.py "
          "(ADD the mirror:: union key), then convert_g1_npz_to_omg125.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
