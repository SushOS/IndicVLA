"""
Pragya-VLA pilot: upload generated motion data to a HuggingFace Hub dataset repo.

What it uploads:
    motions_g1/   *.npz      — Kimodo native G1 motions
    qpos_g1/      *.csv      — MuJoCo qpos (36 cols) for G1
    motions_soma/ *.npz      — Kimodo native SOMA motions
    manifest.jsonl           — full generation log
    prompts_rewritten.csv    — source prompts + Kimodo rewrites (audit trail)
    README.md                — auto-generated dataset card

Why HF Hub instead of laptop disk:
    * Vast.ai instances are ephemeral; once the rental ends, /workspace is gone.
    * HF Hub gives you free private dataset hosting and version history (git-LFS
      under the hood).
    * The same repo will host the 5,000-prompt full corpus later — start the
      naming/versioning convention now.

Setup (one-time, before first upload):
    1. Get an HF write-token: https://huggingface.co/settings/tokens
       Type: "Write". Save it.
    2. On the Vast.ai box:
           pip install huggingface_hub
           export HF_TOKEN=hf_xxxxxxxxxxxx
       (or pass --token directly to this script — but env var is safer)
    3. Pick a repo name. Convention: <username>/pragya-vla-pilot-200-T1

Usage:
    python upload_to_hf.py \\
        --out_dir /workspace/kimodo_outputs \\
        --csv /workspace/prompts_rewritten.csv \\
        --repo_id <yourusername>/pragya-vla-pilot-200-T1 \\
        --private
"""
import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, create_repo


README_TEMPLATE = """\
---
license: apache-2.0
tags:
- pragya-vla
- kimodo
- motion-generation
- multilingual
- robotics
- unitree-g1
language:
- en
- hi
- bn
- te
---

# {repo_id}

Pilot motion-generation outputs for the **Pragya-VLA** project, generated with
[Kimodo](https://github.com/nv-tlabs/kimodo).

## What's in this repo

- **`motions_g1/`** — {n_g1} Kimodo `.npz` motions on the Unitree G1 humanoid robot skeleton.
- **`qpos_g1/`** — {n_qpos} MuJoCo `qpos` CSVs (36 columns: 3 root pos + 4 root quat + 29 joint angles), one per G1 motion. Replay with `mujoco.MjModel.from_xml_path(g1.xml)` and stepping `data.qpos[:] = frame`.
- **`motions_soma/`** — {n_soma} Kimodo `.npz` motions on the NVIDIA SOMA humanoid mesh, generated from the same prompts as G1 for cross-validation.
- **`manifest.jsonl`** — Per-prompt generation record: input text, Kimodo rewrite, seeds, file paths, status, errors.
- **`prompts_rewritten.csv`** — Source prompts (robot-imperative English) plus the descriptive Kimodo input form used for generation.

## Generation parameters

- **Tier**: T1 (atomic primitives) — single-action prompts, 3-second duration each.
- **Models**: `Kimodo-G1-RP-v1` and `Kimodo-SOMA-RP-v1.1`.
- **Diffusion steps**: 100 (DDIM, official default).
- **Samples per prompt**: 1.
- **Seeds**: deterministic per prompt (`SHA256(prompt_id)`).
- **Post-processing**: disabled for G1 per official Kimodo guidance; SOMA also untouched here.

## Prompt structure

The Pragya-VLA paper specifies a 7-tier prompt taxonomy. This pilot covers tier T1 only
(200 prompts), partitioned across motion families:
- `basic_locomotion` (155): turns, steps, postural shifts, stops.
- `posture_transition` (25): rise, crouch, squat, bow, lean.
- `social_gesture` (14): clap, salute, nod, shrug, thumbs up, point.
- `safety_intervention` (6): weight shift, hand lift, chin lift.

Each prompt was authored as an imperative robot command (e.g. `"Stop."`, `"Raise your left arm."`)
and rewritten into descriptive form (`"A person stops walking and stands still."`,
`"A person raises their left arm."`) before being passed to Kimodo, which was trained on
descriptive captions.

## Multilingual coverage

The original prompt source (`translations_200.xlsx`, not bundled here) contains literal and
robot-natural translations into Hindi, Bengali, and Telugu. Kimodo itself only consumes
English; the multilingual variants are reserved for downstream OpenVLA fine-tuning.

## Generated

{generated_at} UTC.

## Cite

Generated as part of Pragya-VLA — Instruction-Finetuned Vision-Language-Action Model with
Locomotion-Aware Chain-of-Thought (BITS Pilani Goa).

Kimodo: Rempe et al., *Kimodo: Scaling Controllable Human Motion Generation*, NVIDIA, 2026.
"""


def count_files(directory: Path, ext: str) -> int:
    if not directory.exists():
        return 0
    return sum(1 for _ in directory.glob(f"*{ext}"))


def manifest_summary(manifest_path: Path) -> dict:
    """Quick stats from the manifest for the README card."""
    if not manifest_path.exists():
        return {"total": 0, "ok": 0, "partial": 0}
    by_id = {}
    with open(manifest_path) as f:
        for line in f:
            try:
                rec = json.loads(line)
                by_id[rec["prompt_id"]] = rec
            except json.JSONDecodeError:
                continue
    total = len(by_id)
    ok = sum(1 for r in by_id.values() if r.get("status") == "ok")
    partial = total - ok
    return {"total": total, "ok": ok, "partial": partial}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True, help="Local kimodo_outputs/ folder")
    ap.add_argument("--csv", required=True, help="prompts_rewritten.csv path")
    ap.add_argument("--repo_id", required=True, help="HF Hub repo, e.g. user/pragya-vla-pilot-200-T1")
    ap.add_argument("--token", default=None, help="HF write token (or set HF_TOKEN env var)")
    ap.add_argument("--private", action="store_true", help="Create as private repo")
    ap.add_argument("--commit_message", default=None)
    args = ap.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("Provide --token or set HF_TOKEN env var.")

    out_dir = Path(args.out_dir)
    if not out_dir.exists():
        raise SystemExit(f"out_dir not found: {out_dir}")

    # Stage prompts CSV alongside the outputs so the upload is one folder
    src_csv = Path(args.csv)
    if src_csv.exists():
        dst_csv = out_dir / "prompts_rewritten.csv"
        if not dst_csv.exists() or dst_csv.stat().st_mtime < src_csv.stat().st_mtime:
            shutil.copy2(src_csv, dst_csv)
            print(f"[stage] copied {src_csv} -> {dst_csv}")

    # Build the dataset card
    manifest_path = out_dir / "manifest.jsonl"
    summary = manifest_summary(manifest_path)
    n_g1 = count_files(out_dir / "motions_g1", ".npz")
    n_qpos = count_files(out_dir / "qpos_g1", ".csv")
    n_soma = count_files(out_dir / "motions_soma", ".npz")

    readme = README_TEMPLATE.format(
        repo_id=args.repo_id,
        n_g1=n_g1,
        n_qpos=n_qpos,
        n_soma=n_soma,
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
    )
    readme_extras = (
        f"\n## Manifest summary\n\n"
        f"- Total prompts: {summary['total']}\n"
        f"- Fully successful (both models): {summary['ok']}\n"
        f"- Partial / failed: {summary['partial']}\n"
    )
    (out_dir / "README.md").write_text(readme + readme_extras, encoding="utf-8")
    print(f"[stage] wrote README.md")

    # Push
    api = HfApi(token=token)
    print(f"[hf] ensuring repo exists: {args.repo_id} (private={args.private})")
    create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
        token=token,
    )

    msg = args.commit_message or (
        f"Pragya-VLA pilot T1 upload — "
        f"{summary['ok']}/{summary['total']} prompts ok "
        f"({n_g1} G1 npz, {n_qpos} G1 qpos, {n_soma} SOMA npz)"
    )
    print(f"[hf] uploading {out_dir} -> {args.repo_id}")
    print(f"[hf] commit: {msg}")
    api.upload_folder(
        folder_path=str(out_dir),
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message=msg,
        token=token,
        # Skip transient files
        ignore_patterns=[".DS_Store", "__pycache__", "*.pyc", "*.tmp"],
    )
    print(f"[hf] done. View: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
