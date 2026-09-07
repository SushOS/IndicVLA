#!/usr/bin/env python3
"""Mirror run artifacts to the Hub continuously, independently of the trainer.

WHY A SEPARATE DAEMON
---------------------
The trainer pushes its own checkpoints, and on 2026-09-06 those pushes had been failing for
hours with `403 Private repository storage limit reached` while training reported success.
Nothing noticed, because the trainer treats a push failure as a warning and continues.

Two lessons, both applied here:
  * the box is workspace_is_volume=false -- a recycle erases everything, so artifacts must
    leave continuously, not at the end (SSOT s14 #7);
  * a push that "succeeded" is not an artifact that EXISTS. This verifies by size against
    the Hub's own file listing, and teardown_gate.py later re-downloads and checksums.

Idempotent: re-uploads only files absent from the Hub or whose size differs.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path


def targets(work: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    runs = work / "runs"
    if runs.is_dir():
        for p in runs.rglob("*"):
            if p.is_file() and p.suffix in {".pt", ".json", ".csv", ".txt", ".jsonl", ".md"}:
                out[f"runs/{p.relative_to(runs).as_posix()}"] = p
    logs = work / "logs"
    if logs.is_dir():
        for p in sorted(logs.glob("*.log")):
            out[f"logs/{p.name}"] = p
    for name in ("adapter_calibration.pt", "adapter_calibration.json",
                 "t5_channel_calibration.pt", "t5_channel_calibration.json",
                 "RUN_NAME.env", "TEARDOWN_MANIFEST.json"):
        p = work / name
        if p.is_file():
            out[f"provenance/{name}"] = p
    data = work / "data" / "amass_g1_omg125"
    if data.is_dir():
        for p in sorted(data.rglob("_meta.json")):
            out[f"data/{p.parent.name}_meta.json"] = p
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/workspace")
    ap.add_argument("--repo", default="CodeSushh/maila-run2-motionpivot")
    ap.add_argument("--repo-type", default="dataset")
    ap.add_argument("--interval", type=int, default=600, help="0 = one pass and exit")
    a = ap.parse_args()

    from huggingface_hub import HfApi
    tok = os.environ.get("HF_TOKEN") or Path("/root/.creds/hf_token").read_text().strip()
    api = HfApi(token=tok)
    api.create_repo(a.repo, repo_type=a.repo_type, exist_ok=True, private=True)
    work = Path(a.work)

    while True:
        try:
            info = api.repo_info(a.repo, repo_type=a.repo_type, files_metadata=True)
            have = {f.rfilename: (f.size or 0) for f in info.siblings}
        except Exception as e:                                        # noqa: BLE001
            print(f"[sync] listing failed: {type(e).__name__}: {e}", flush=True)
            have = {}

        want = targets(work)
        todo = [d for d, p in want.items()
                if have.get(d) != p.stat().st_size]
        pushed = failed = 0
        for dst in sorted(todo):
            try:
                api.upload_file(path_or_fileobj=str(want[dst]), path_in_repo=dst,
                                repo_id=a.repo, repo_type=a.repo_type)
                pushed += 1
            except Exception as e:                                    # noqa: BLE001
                failed += 1
                print(f"[sync] FAILED {dst}: {type(e).__name__}: {str(e)[:140]}", flush=True)
        print(f"[sync] {time.strftime('%H:%M:%S')} tracked={len(want)} "
              f"pushed={pushed} failed={failed} already={len(want) - len(todo)}", flush=True)
        if a.interval <= 0:
            return 1 if failed else 0
        time.sleep(a.interval)


if __name__ == "__main__":
    raise SystemExit(main())
