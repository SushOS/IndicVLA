#!/usr/bin/env python3
"""
vastai_retarget_pipeline.py
---------------------------
End-to-end pipeline that runs on a Vast.ai GPU instance.

For each of the 17 AMASS datasets:
  1. Extract keypoints from .pt MotionLib   [CPU, 'protomotions' conda env]
  2. Retarget human motion to Unitree G1    [GPU, 'pyroki' conda env + JAX]
  3. Push retargeted .npz files to HuggingFace

Fully resumable: re-run at any point to continue where it left off.
Per-dataset state is saved in /workspace/pipeline_state.json so you can
also kill and restart without losing progress.

JAX compilation cache is written to /workspace/jax_cache, so only the
FIRST dataset ever needs to JIT-compile. All 16 subsequent ones reuse it.

Usage:
    python vastai_retarget_pipeline.py                      # full pipeline
    python vastai_retarget_pipeline.py --skip-download      # skip HF pull
    python vastai_retarget_pipeline.py --datasets CMU KIT   # specific datasets
    python vastai_retarget_pipeline.py --skip-extract       # keypoints already done
    python vastai_retarget_pipeline.py --skip-retarget      # retargeting already done
    python vastai_retarget_pipeline.py --skip-push          # no HF upload
    python vastai_retarget_pipeline.py --force              # ignore saved state

Run inside a screen session so SSH disconnect doesn't kill it:
    screen -S pipeline
    python vastai_retarget_pipeline.py
    Ctrl-A D   # detach
    screen -r pipeline  # reattach
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# ── HuggingFace Config ────────────────────────────────────────────────────────
HF_TOKEN    = os.environ.get("HF_TOKEN", os.environ["HF_TOKEN"])
HF_REPO_IN  = "AdiShingote/amass-smpl-dataset"           # 17 input .pt files
HF_REPO_OUT = "AdiShingote/pragya-vla-g1-motion-dataset" # retargeted .npz output

# ── Paths (on Vast.ai) ────────────────────────────────────────────────────────
WORKSPACE     = Path("/workspace")
PT_DIR        = WORKSPACE / "amass_pt_parts"   # downloaded .pt files
KEYPOINTS_DIR = WORKSPACE / "keypoints"        # intermediate per-dataset keypoints
RETARGET_DIR  = WORKSPACE / "retargeted_g1"    # final .npz output per dataset
PROTO_DIR     = WORKSPACE / "ProtoMotions"     # cloned repo
SCRIPTS_DIR   = PROTO_DIR / "data" / "scripts" # keypoint_utils.py lives here
LOG_DIR       = WORKSPACE / "logs"
JAX_CACHE_DIR = WORKSPACE / "jax_cache"
STATE_FILE    = WORKSPACE / "pipeline_state.json"

# Key scripts (inside ProtoMotions)
EXTRACT_SCRIPT  = SCRIPTS_DIR / "extract_retargeting_input_keypoints_from_packaged_motionlib.py"
RETARGET_SCRIPT = PROTO_DIR / "pyroki" / "batch_retarget_to_g1_from_keypoints.py"

# Conda environment names — must match what setup_vastai_envs.sh creates
PROTO_ENV  = "protomotions"   # PyTorch + protomotions package (CPU)
PYROKI_ENV = "pyroki"         # JAX[cuda12] + pyroki package  (GPU)


# ── State management ──────────────────────────────────────────────────────────
def load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {}


def save_state(state: dict) -> None:
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def mark_done(state: dict, dataset: str, step: str) -> None:
    state.setdefault(dataset, {})[step] = True
    save_state(state)


def is_done(state: dict, dataset: str, step: str) -> bool:
    return state.get(dataset, {}).get(step, False)


# ── Helpers ───────────────────────────────────────────────────────────────────
def ts() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def fmt_elapsed(seconds: float) -> str:
    return str(datetime.timedelta(seconds=int(seconds)))


def run(cmd, desc="", **kwargs) -> tuple:
    """Run subprocess, stream output live, return (returncode, elapsed_sec)."""
    t0 = time.time()
    label = f"[{ts()}]"
    if desc:
        print(f"\n{label} {desc}")
    print(f"{label} $ {' '.join(str(c) for c in cmd)}\n")
    sys.stdout.flush()
    result = subprocess.run(cmd, **kwargs)
    elapsed = time.time() - t0
    return result.returncode, elapsed


# ── Step A: Download all .pt files from HuggingFace ──────────────────────────
def download_pt_files() -> bool:
    PT_DIR.mkdir(parents=True, exist_ok=True)
    existing = sorted(PT_DIR.glob("*.pt"))
    if existing:
        print(f"[{ts()}] {len(existing)} .pt files already in {PT_DIR} — skipping download.")
        for f in existing:
            print(f"  {f.name:40s} {f.stat().st_size/1024**3:.2f} GB")
        return True

    print(f"\n[{ts()}] ══ Downloading .pt files from HuggingFace ══")
    print(f"  Source : {HF_REPO_IN}")
    print(f"  Dest   : {PT_DIR}")
    print(f"  NOTE   : ~12 GB transfer, may take 10-30 min depending on HF CDN")

    code, elapsed = run(
        [
            sys.executable, "-c",
            (
                "from huggingface_hub import snapshot_download; "
                f"snapshot_download("
                f"  repo_id='{HF_REPO_IN}', "
                f"  repo_type='dataset', "
                f"  local_dir='{PT_DIR}', "
                f"  token='{HF_TOKEN}', "
                f"  ignore_patterns=['*.md', '.gitattributes']"
                f")"
            ),
        ],
        desc="Pulling from HuggingFace...",
    )

    pt_files = sorted(PT_DIR.glob("*.pt"))
    print(f"[{ts()}] Download done: {len(pt_files)} .pt files  ({fmt_elapsed(elapsed)})")
    return code == 0 and len(pt_files) > 0


# ── Step B.1: Extract keypoints from a .pt file ───────────────────────────────
def extract_keypoints(dataset_name: str, pt_file: Path, keypoints_dir: Path) -> bool:
    keypoints_dir.mkdir(parents=True, exist_ok=True)

    # Skip if already done (extract script has its own skip-existing logic)
    existing_npy = list(keypoints_dir.glob("*.npy"))
    if existing_npy:
        print(f"[{ts()}] [{dataset_name}] {len(existing_npy)} .npy files already exist in {keypoints_dir}")
        print(f"[{ts()}] [{dataset_name}] Running extract anyway (will skip existing, pick up new ones)...")

    env = {
        **os.environ,
        # protomotions package must be findable
        "PYTHONPATH": f"{PROTO_DIR}:{SCRIPTS_DIR}",
    }

    code, elapsed = run(
        [
            "conda", "run", "--no-capture-output", "-n", PROTO_ENV,
            "python", str(EXTRACT_SCRIPT),
            str(pt_file),
            "--output-path",     str(keypoints_dir),
            "--skeleton-format", "smpl",
            "--skip-freq",       "1",   # process ALL motions (default=35 subsamples!)
            "--start-idx",       "0",   # include motion index 0
        ],
        desc=f"[{dataset_name}] Extracting keypoints...",
        env=env,
        cwd=str(PROTO_DIR),     # MJCF paths are relative to ProtoMotions root; keypoint_utils found via PYTHONPATH
    )

    n_npy = len(list(keypoints_dir.glob("*.npy")))
    ok = code == 0 and n_npy > 0
    print(f"[{ts()}] [{dataset_name}] Keypoints: {n_npy} .npy files  exit={code}  ({fmt_elapsed(elapsed)})")
    if not ok:
        print(f"  WARNING: exit code {code} or zero output files — check logs above.")
    return ok


# ── Step B.2: PyRoki retargeting ──────────────────────────────────────────────
def retarget_dataset(
    dataset_name: str,
    keypoints_dir: Path,
    retargeted_dir: Path,
    log_file: Path,
) -> bool:
    retargeted_dir.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    n_input = len(list(keypoints_dir.glob("*.npy")))
    n_done  = len(list(retargeted_dir.glob("*.npz")))
    print(f"[{ts()}] [{dataset_name}] Retargeting: {n_input} keypoint files, {n_done} already retargeted")

    if n_input == 0:
        print(f"  ERROR: No .npy files in {keypoints_dir}. Run extract step first.")
        return False

    env = {
        **os.environ,
        "PYTHONPATH": str(PROTO_DIR),
        # Persist JAX JIT-compiled kernels across dataset runs (only compiles once!)
        "JAX_COMPILATION_CACHE_DIR": str(JAX_CACHE_DIR),
        # Avoid pre-allocating all GPU VRAM — leaves room for large datasets
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        # CUDA path hint (adjust if needed)
        "XLA_FLAGS": "--xla_gpu_cuda_data_dir=/usr/local/cuda",
    }

    code, elapsed = run(
        [
            "conda", "run", "--no-capture-output", "-n", PYROKI_ENV,
            "python", str(RETARGET_SCRIPT),
            "--no-visualize",
            "--keypoints-folder-path", str(keypoints_dir),
            "--output-dir",            str(retargeted_dir),
            "--skip-existing",         # resume: skip already-retargeted motions
            "--log-file",              str(log_file),
            "--source-type",           "smpl",
            "--input-fps",             "30",
            # target-raw-frames=450 (default) = 15s @ 30fps, suits HumanML3D clips
            # subsample-factor=1  (default) = no frame subsampling
        ],
        desc=f"[{dataset_name}] PyRoki G1 retargeting (JAX GPU)...",
        env=env,
        cwd=str(PROTO_DIR / "pyroki"),
    )

    n_npz = len(list(retargeted_dir.glob("*.npz")))
    ok = n_npz > 0
    print(
        f"[{ts()}] [{dataset_name}] Retargeting: {n_npz} .npz files  "
        f"exit={code}  ({fmt_elapsed(elapsed)})"
    )
    if code != 0:
        print(f"  WARNING: non-zero exit. Check log: {log_file}")
    return ok


# ── Step B.3: Push retargeted .npz files to HuggingFace ─────────────────────
def push_to_hf(dataset_name: str, retargeted_dir: Path) -> bool:
    npz_files = sorted(retargeted_dir.glob("*.npz"))
    if not npz_files:
        print(f"[{ts()}] [{dataset_name}] No .npz files to push — skipping HF upload.")
        return True

    total_mb = sum(f.stat().st_size for f in npz_files) / 1024**2
    print(f"\n[{ts()}] [{dataset_name}] Pushing {len(npz_files)} .npz ({total_mb:.1f} MB) to HuggingFace...")
    print(f"  Repo: {HF_REPO_OUT}")

    # Build upload script as a heredoc string to avoid shell quoting issues
    upload_code = f"""
import os, time, sys
from huggingface_hub import HfApi
from pathlib import Path

api = HfApi(token="{HF_TOKEN}")
api.create_repo(
    repo_id="{HF_REPO_OUT}",
    repo_type="dataset",
    private=True,
    exist_ok=True,
)

npz_files = sorted(Path("{retargeted_dir}").glob("*.npz"))
total = len(npz_files)
failed = 0

for i, f in enumerate(npz_files, 1):
    t0 = time.time()
    try:
        api.upload_file(
            path_or_fileobj=str(f),
            path_in_repo="{dataset_name}/" + f.name,
            repo_id="{HF_REPO_OUT}",
            repo_type="dataset",
        )
        elapsed = time.time() - t0
        print(f"  [{{i}}/{{total}}] OK  {{f.name}}  ({{elapsed:.1f}}s)", flush=True)
    except Exception as e:
        failed += 1
        print(f"  [{{i}}/{{total}}] FAIL  {{f.name}}: {{e}}", flush=True)

print(f"Upload done: {{total - failed}} OK, {{failed}} failed.")
sys.exit(0 if failed == 0 else 1)
"""

    # Write to a temp file to avoid -c string length limits
    tmp_script = WORKSPACE / f"_hf_push_{dataset_name}.py"
    tmp_script.write_text(upload_code)

    code, elapsed = run(
        [sys.executable, str(tmp_script)],
        desc=f"[{dataset_name}] Uploading to HuggingFace...",
    )

    tmp_script.unlink(missing_ok=True)
    print(f"[{ts()}] [{dataset_name}] HF push done  exit={code}  ({fmt_elapsed(elapsed)})")
    return code == 0


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Vast.ai AMASS → G1 retargeting pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip HF download (use .pt files already in /workspace/amass_pt_parts)",
    )
    parser.add_argument(
        "--datasets", nargs="*", metavar="DATASET",
        help="Process only these datasets (e.g. --datasets CMU KIT). Default: all.",
    )
    parser.add_argument(
        "--skip-extract", action="store_true",
        help="Skip keypoint extraction (keypoints already extracted).",
    )
    parser.add_argument(
        "--skip-retarget", action="store_true",
        help="Skip PyRoki retargeting (already done).",
    )
    parser.add_argument(
        "--skip-push", action="store_true",
        help="Skip uploading to HuggingFace after each dataset.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Ignore pipeline_state.json; redo all steps.",
    )
    args = parser.parse_args()

    # Create all workspace directories
    for d in [PT_DIR, KEYPOINTS_DIR, RETARGET_DIR, LOG_DIR, JAX_CACHE_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    state = {} if args.force else load_state()

    print(f"\n[{ts()}] ══════════════════════════════════════════════════════")
    print(f"[{ts()}] Vast.ai AMASS → Unitree G1 Retargeting Pipeline")
    print(f"[{ts()}] ══════════════════════════════════════════════════════")
    print(f"  Input  HF repo : {HF_REPO_IN}")
    print(f"  Output HF repo : {HF_REPO_OUT}")
    print(f"  Workspace      : {WORKSPACE}")
    print(f"  State file     : {STATE_FILE}")

    # ── Step A: Download .pt files ────────────────────────────────────────────
    if not args.skip_download and not is_done(state, "_global", "download"):
        ok = download_pt_files()
        if ok:
            mark_done(state, "_global", "download")
        else:
            print(f"ERROR: Failed to download .pt files from HuggingFace.")
            sys.exit(1)
    else:
        print(f"\n[{ts()}] Skipping HF download (--skip-download or already done).")

    # ── Find .pt files ────────────────────────────────────────────────────────
    pt_files = sorted(PT_DIR.glob("*.pt"))
    if not pt_files:
        print(f"\nERROR: No .pt files found in {PT_DIR}")
        print("  Run without --skip-download, or manually copy .pt files there.")
        sys.exit(1)

    if args.datasets:
        pt_files = [f for f in pt_files if f.stem in args.datasets]
        if not pt_files:
            print(f"ERROR: None of the specified datasets found. Available: {[f.stem for f in sorted(PT_DIR.glob('*.pt'))]}")
            sys.exit(1)

    print(f"\n[{ts()}] Processing {len(pt_files)} datasets:")
    for f in pt_files:
        size_gb = f.stat().st_size / 1024**3
        print(f"  {f.stem:30s} {size_gb:.2f} GB")

    # ── Per-dataset loop ──────────────────────────────────────────────────────
    results: dict = {}
    pipeline_t0 = time.time()

    for i, pt_file in enumerate(pt_files, 1):
        dname        = pt_file.stem
        kp_dir       = KEYPOINTS_DIR / dname
        ret_dir      = RETARGET_DIR  / dname
        log_file     = LOG_DIR / f"{dname}_retarget.log"
        results[dname] = {}

        print(f"\n[{ts()}] ── [{i}/{len(pt_files)}]  {dname}  ──────────────────────────────")

        # B.1 — Extract keypoints
        if args.skip_extract or is_done(state, dname, "extract"):
            n = len(list(kp_dir.glob("*.npy"))) if kp_dir.exists() else 0
            print(f"[{ts()}] SKIP extract ({n} .npy already in {kp_dir})")
            results[dname]["extract"] = "skipped"
        else:
            ok = extract_keypoints(dname, pt_file, kp_dir)
            results[dname]["extract"] = ok
            if ok:
                mark_done(state, dname, "extract")
            else:
                print(f"  WARNING: keypoint extraction failed for {dname}, continuing...")

        # B.2 — PyRoki retargeting
        if args.skip_retarget or is_done(state, dname, "retarget"):
            n = len(list(ret_dir.glob("*.npz"))) if ret_dir.exists() else 0
            print(f"[{ts()}] SKIP retarget ({n} .npz already in {ret_dir})")
            results[dname]["retarget"] = "skipped"
        else:
            ok = retarget_dataset(dname, kp_dir, ret_dir, log_file)
            results[dname]["retarget"] = ok
            if ok:
                mark_done(state, dname, "retarget")
            else:
                print(f"  WARNING: retargeting returned non-zero for {dname}, continuing...")

        # B.3 — Push to HuggingFace
        if args.skip_push or is_done(state, dname, "push"):
            print(f"[{ts()}] SKIP HF push for {dname}")
            results[dname]["push"] = "skipped"
        else:
            ok = push_to_hf(dname, ret_dir)
            results[dname]["push"] = ok
            if ok:
                mark_done(state, dname, "push")

    # ── Final summary ─────────────────────────────────────────────────────────
    total_elapsed = time.time() - pipeline_t0
    total_npz     = sum(1 for _ in RETARGET_DIR.rglob("*.npz"))

    print(f"\n[{ts()}] ══════════════════════════════════════════════════════")
    print(f"[{ts()}] PIPELINE COMPLETE  (total: {fmt_elapsed(total_elapsed)})")
    print(f"[{ts()}] ══════════════════════════════════════════════════════")
    print(f"  {'Dataset':30s}  {'npz':>5}  extract   retarget  push")
    print(f"  {'-'*30}  {'-'*5}  {'-'*8}  {'-'*8}  {'-'*8}")
    for dname, steps in results.items():
        ret_dir = RETARGET_DIR / dname
        n_npz = len(list(ret_dir.glob("*.npz"))) if ret_dir.exists() else 0
        row = "  {:<30}  {:>5}  {:<8}  {:<8}  {:<8}".format(
            dname, n_npz,
            str(steps.get("extract", "-"))[:8],
            str(steps.get("retarget", "-"))[:8],
            str(steps.get("push", "-"))[:8],
        )
        print(row)

    print(f"\n  Total retargeted .npz : {total_npz}")
    print(f"  Output HF repo        : https://huggingface.co/datasets/{HF_REPO_OUT}")
    print(f"\n[{ts()}] All done.")


if __name__ == "__main__":
    main()
