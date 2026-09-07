#!/usr/bin/env python3
"""Capture an immutable provenance lock for a MAILA Run 2 experiment.

WHY
---
Run 1's hardest-won lesson is that the things which silently corrupt an experiment are
invisible to the obvious checks: an attention flag that changes numerics without changing
any parameter shape (research log s5.2), a codec convention that passes its own selftest
(s4.3), a state_dict key that loads into a randomly-initialised layer (s11.3), a dropout
flag left on for one row of a comparison table (SSOT error 9). None of those show up in a
loss curve. They show up months later when a number cannot be reproduced.

So before a single training step runs, this writes RUN_LOCK.json: the git commit, the
SHA-256 of every weight and data file the run touches, resolved hyperparameters, package
versions, GPU identity, and the data statistics that define the corpus. It is pushed to
the Hugging Face run repo FIRST, so the lock exists even if the box dies mid-run.

Nothing here is secret: tokens are read from the environment and never recorded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SECRET_HINTS = ("token", "key", "secret", "password", "cred")


def sha256(path: Path, limit_mb: int | None = None) -> dict:
    """Full-file SHA-256. `limit_mb` hashes only the first N MB for very large files."""
    h = hashlib.sha256()
    n = 0
    cap = None if limit_mb is None else limit_mb * 1024 * 1024
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
            n += len(chunk)
            if cap is not None and n >= cap:
                break
    return {"sha256": h.hexdigest(), "bytes": path.stat().st_size,
            "hashed_bytes": n, "partial": cap is not None and n >= cap}


def hash_tree(root: Path, patterns: list[str]) -> dict:
    out = {}
    for pat in patterns:
        for p in sorted(root.glob(pat)):
            if p.is_file():
                out[str(p.relative_to(root))] = sha256(p)
    return out


def git_state(repo: Path) -> dict:
    def run(*args):
        try:
            return subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                                  text=True, timeout=30).stdout.strip()
        except Exception:
            return ""
    dirty = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "describe": run("describe", "--always", "--dirty"),
        "dirty": bool(dirty),
        "dirty_files": [l[3:] for l in dirty.splitlines()][:60],
        "remote": run("remote", "get-url", "origin"),
    }


def package_versions() -> dict:
    mods = ["torch", "transformers", "numpy", "pandas", "huggingface_hub", "hydra",
            "pytorch_lightning", "wandb", "scipy", "pyarrow", "omegaconf"]
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for m in mods:
        try:
            out[m] = __import__(m).__version__
        except Exception:
            out[m] = None
    return out


def gpu_state() -> dict:
    info = {}
    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        info["torch_cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            info["device_name"] = torch.cuda.get_device_name(0)
            info["device_count"] = torch.cuda.device_count()
            info["capability"] = list(torch.cuda.get_device_capability(0))
            info["total_memory_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
    except Exception as e:
        info["error"] = str(e)[:200]
    try:
        info["nvidia_smi"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        pass
    return info


def data_stats(shard_root: Path) -> dict:
    """Per-split window/caption statistics, read from what the converter actually wrote."""
    out = {}
    for split in ("train", "val", "test", "reserve"):
        d = shard_root / split
        if not d.is_dir():
            continue
        entry: dict = {}
        meta = d / "_meta.json"
        if meta.is_file():
            entry["meta"] = json.loads(meta.read_text())
            entry["meta_sha256"] = sha256(meta)["sha256"]
        shards = sorted(d.glob("shard_*.npz"))
        entry["shards"] = len(shards)
        entry["shard_sha256"] = {s.name: sha256(s, limit_mb=64)["sha256"] for s in shards}
        man = d / "manifest.parquet"
        if man.is_file():
            entry["manifest_sha256"] = sha256(man)["sha256"]
            try:
                import pandas as pd
                m = pd.read_parquet(man)
                entry["windows"] = int(len(m))
                for lg in ("en", "hi", "bn", "ta", "te"):
                    col = f"{lg}_1"
                    if col in m.columns:
                        entry[f"distinct_{lg}_1"] = int(m[col].nunique())
                        entry[f"windows_per_caption_{lg}"] = round(
                            len(m) / max(m[col].nunique(), 1), 4)
                    ncol = f"n_caps_{lg}"
                    if ncol in m.columns:
                        entry[f"windows_with_2plus_{lg}"] = int((m[ncol] >= 2).sum())
                if "source_amass" in m.columns:
                    entry["distinct_sources"] = int(m["source_amass"].nunique())
                if "super_group" in m.columns:
                    entry["distinct_super_groups"] = int(m["super_group"].nunique())
            except Exception as e:
                entry["manifest_read_error"] = str(e)[:200]
        out[split] = entry
    return out


def leakage_check(shard_root: Path) -> dict:
    """Re-assert at LOCK time that the shards themselves share no caption or source."""
    try:
        import pandas as pd
    except Exception as e:
        return {"error": str(e)[:120]}
    mans = {}
    for split in ("train", "val", "test", "reserve"):
        p = shard_root / split / "manifest.parquet"
        if p.is_file():
            mans[split] = pd.read_parquet(p)
    res, ok = {}, True
    names = sorted(mans)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            cap = src = None
            if "hi_1" in mans[a] and "hi_1" in mans[b]:
                cap = len(set(mans[a]["hi_1"]) & set(mans[b]["hi_1"]))
            if "source_amass" in mans[a] and "source_amass" in mans[b]:
                src = len(set(mans[a]["source_amass"]) & set(mans[b]["source_amass"]))
            res[f"{a}|{b}"] = {"shared_captions": cap, "shared_sources": src}
            if (cap or 0) > 0 or (src or 0) > 0:
                ok = False
    res["PASS"] = ok
    return res


def redact(d: dict) -> dict:
    return {k: ("<redacted>" if any(h in k.lower() for h in SECRET_HINTS) else v)
            for k, v in d.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default="/workspace/IndicVLA")
    ap.add_argument("--shard-root", default="/workspace/data/amass_g1_omg125")
    ap.add_argument("--omg-ckpt", default="/workspace/models/omg/checkpoints/updated/100m/sstep=170000.ckpt")
    ap.add_argument("--evaluator", default="/workspace/models/omg/evaluator/step_004000.pt")
    ap.add_argument("--muril", default="/workspace/models/muril-base-cased")
    ap.add_argument("--t5", default="/workspace/models/t5-base-local")
    ap.add_argument("--stats", default=None, help="g1_125d_stats.json used by the codec")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--args-json", default=None, help="resolved training args as JSON")
    ap.add_argument("--out", default="RUN_LOCK.json")
    ap.add_argument("--hf-repo", default="PragyaVLA/maila-run2-motionpivot")
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args()

    repo = Path(a.repo_root)
    shard_root = Path(a.shard_root)

    weights = {}
    for name, p in [("omg_ckpt", a.omg_ckpt), ("evaluator", a.evaluator)]:
        pp = Path(p)
        if pp.is_file():
            weights[name] = {"path": str(pp), **sha256(pp)}
    for name, p in [("muril", a.muril), ("t5_base", a.t5)]:
        pp = Path(p)
        if pp.is_dir():
            weights[name] = {"path": str(pp),
                             "files": hash_tree(pp, ["*.bin", "*.safetensors", "*.json",
                                                     "*.model", "*.txt"])}
    if a.stats and Path(a.stats).is_file():
        weights["g1_125d_stats"] = {"path": a.stats, **sha256(Path(a.stats))}

    lock = {
        "schema": "maila.run_lock.v1",
        "run_name": a.run_name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "git": git_state(repo),
        "code_sha256": hash_tree(repo / "OMG" / "tools", ["*.py"]),
        "docs_sha256": hash_tree(repo / "OMG", ["MAILA_SSOT_*.md", "MAILA_RESEARCH_LOG_*.md"]),
        "splits_sha256": hash_tree(repo / "OMG" / "splits_amass_g1_13k", ["*.csv", "*.json"]),
        "weights": weights,
        "packages": package_versions(),
        "gpu": gpu_state(),
        "data": data_stats(shard_root),
        "leakage_recheck": leakage_check(shard_root),
        "env": redact({k: v for k, v in os.environ.items()
                       if k.startswith(("CUDA", "HF_", "WANDB", "TORCH", "PYTHON", "OMG_"))}),
        "contract": {
            "self_attention_qk_norm": True,
            "cross_attention_qk_norm": True,
            "source": "EXP-F, research log s5.2: true/true wins by +16.1 GEN-R@1 on updated/100m; "
                      "REFERENCE R@1=0.6680 gate passed in both arms",
        },
    }
    if a.args_json and Path(a.args_json).is_file():
        lock["train_args"] = redact(json.loads(Path(a.args_json).read_text()))

    out = Path(a.out)
    out.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    print(f"wrote {out}  ({out.stat().st_size:,} bytes)")
    print(f"  git      : {lock['git'].get('describe')}  dirty={lock['git'].get('dirty')}")
    print(f"  gpu      : {lock['gpu'].get('device_name')} / {lock['gpu'].get('nvidia_smi','')}")
    print(f"  torch    : {lock['packages'].get('torch')}  transformers "
          f"{lock['packages'].get('transformers')}")
    for s, e in lock["data"].items():
        print(f"  data[{s:7s}]: {e.get('windows','?')} windows, "
              f"{e.get('shards','?')} shards, hi/caption={e.get('windows_per_caption_hi','?')}")
    print(f"  leakage  : {'PASS' if lock['leakage_recheck'].get('PASS') else '*** FAIL ***'}")
    if not lock["leakage_recheck"].get("PASS", True):
        raise SystemExit("LEAKAGE DETECTED IN CONVERTED SHARDS -- do not train")

    if a.push:
        from huggingface_hub import HfApi
        tok = os.environ.get("HF_TOKEN") or Path("/root/.creds/hf_token").read_text().strip()
        api = HfApi(token=tok)
        api.create_repo(a.hf_repo, repo_type="dataset", exist_ok=True, private=True)
        api.upload_file(path_or_fileobj=str(out), repo_type="dataset",
                        path_in_repo=f"{a.run_name}/RUN_LOCK.json", repo_id=a.hf_repo)
        print(f"  pushed -> {a.hf_repo}/{a.run_name}/RUN_LOCK.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
