#!/usr/bin/env python3
"""Verify EVERYTHING survives off-box, then and only then permit instance destruction.

WHY THIS EXISTS
---------------
The Vast.ai instance runs with workspace_is_volume=false: `vastai destroy` erases the
container and every byte in /workspace with no recovery path. Run 1 already lost a verbose
per-arm log this way (SSOT s14 error #7) even though its JSONs survived.

An upload returning HTTP 200 is NOT proof the bytes are on the Hub and readable. This gate
therefore RE-DOWNLOADS every required artifact with force_download=True and compares SHA-256
against the local file. A file that differs, is truncated, or 404s fails the gate.

Exit 0 == safe to destroy. Any other exit == do not destroy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path


def sha256(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def collect(work: Path, run_names: list[str]) -> dict[str, Path]:
    """repo-relative destination -> local path.

    Deliberately delegates to sync_artifacts.targets() rather than re-deriving the layout.
    An earlier version built its own paths ("<run>/checkpoints/x.pt") while the sync daemon
    wrote "runs/<run>/x.pt", so the gate would have verified a set of files that did not
    overlap what was actually uploaded -- and passed while proving nothing. One manifest,
    one source of truth.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, "/workspace")
    from sync_artifacts import targets

    want = targets(work)
    for rn in run_names:
        rd = work / "runs" / rn
        if not rd.is_dir():
            raise SystemExit(f"GATE FAIL: run directory missing: {rd}")
        cks = [d for d in want if d.startswith(f"runs/{rn}/") and "adapter_step" in d]
        if not cks:
            raise SystemExit(f"GATE FAIL: {rn} has no adapter_step*.pt checkpoints")
    return want


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/workspace")
    ap.add_argument("--run", action="append", required=True,
                    help="run name; repeat for each arm")
    ap.add_argument("--repo", default="CodeSushh/maila-run2-motionpivot")
    ap.add_argument("--repo-type", default="dataset")
    ap.add_argument("--require-eval", action="store_true",
                    help="fail unless each run has an eval_report.json")
    ap.add_argument("--push-missing", action="store_true",
                    help="upload anything absent or mismatched, then re-verify")
    ap.add_argument("--manifest-out", default="/workspace/TEARDOWN_MANIFEST.json")
    a = ap.parse_args()

    from huggingface_hub import HfApi, hf_hub_download
    tok = os.environ.get("HF_TOKEN") or Path("/root/.creds/hf_token").read_text().strip()
    api = HfApi(token=tok)

    want = collect(Path(a.work), a.run)

    if a.require_eval:
        for rn in a.run:
            if not any(k.startswith(f"runs/{rn}/") and k.endswith("eval_report.json")
                       for k in want):
                raise SystemExit(f"GATE FAIL: {rn} has no eval_report.json -- evaluation "
                                 f"has not been run, so results are not complete")

    print(f"  {len(want)} artifacts required across {len(a.run)} run(s)")
    local = {dst: sha256(p) for dst, p in want.items()}

    remote = set(api.list_repo_files(a.repo, repo_type=a.repo_type))
    if a.push_missing:
        todo = [d for d in want if d not in remote]
        for dst in todo:
            api.upload_file(path_or_fileobj=str(want[dst]), path_in_repo=dst,
                            repo_id=a.repo, repo_type=a.repo_type)
        if todo:
            print(f"  uploaded {len(todo)} missing artifacts")
            remote = set(api.list_repo_files(a.repo, repo_type=a.repo_type))

    ok, bad = [], []
    with tempfile.TemporaryDirectory() as td:
        for dst, want_sha in sorted(local.items()):
            if dst not in remote:
                bad.append((dst, "ABSENT on hub")); continue
            try:
                # force_download: a cached copy would verify our own local bytes, not the hub's
                got = hf_hub_download(a.repo, dst, repo_type=a.repo_type, token=tok,
                                      local_dir=td, force_download=True)
            except Exception as e:                                  # noqa: BLE001
                bad.append((dst, f"download failed: {type(e).__name__}")); continue
            if sha256(Path(got)) != want_sha:
                bad.append((dst, "SHA-256 MISMATCH"))
            else:
                ok.append(dst)
            Path(got).unlink(missing_ok=True)

    manifest = {
        "repo": f"{a.repo_type}s/{a.repo}", "runs": a.run,
        "verified": len(ok), "failed": len(bad),
        "artifacts": {d: local[d] for d in sorted(local)},
        "failures": [{"path": d, "reason": r} for d, r in bad],
        "verification": "re-downloaded from the Hub with force_download and SHA-256 compared",
    }
    Path(a.manifest_out).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"  verified on hub: {len(ok)}/{len(local)}")
    if bad:
        print(f"\n  GATE FAIL -- {len(bad)} artifact(s) not safely on the Hub:")
        for d, r in bad[:20]:
            print(f"    {r:24s} {d}")
        print("\n  DO NOT DESTROY THE INSTANCE.")
        return 2

    try:
        api.upload_file(path_or_fileobj=a.manifest_out, path_in_repo="TEARDOWN_MANIFEST.json",
                        repo_id=a.repo, repo_type=a.repo_type)
    except Exception as e:                                          # noqa: BLE001
        print(f"  manifest upload failed ({type(e).__name__}) -- not fatal, gate already passed")

    print(f"\n  GATE PASS -- all {len(ok)} artifacts re-downloaded from the Hub and "
          f"SHA-256 verified.\n  Safe to destroy instance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
