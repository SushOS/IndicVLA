"""
Pragya-VLA pilot: dual-model batch motion generation with Kimodo.

Generates each prompt on BOTH:
  * Kimodo-G1-RP-v1     (Unitree G1 humanoid robot — produces 36-col MuJoCo qpos CSV)
  * Kimodo-SOMA-RP-v1.1 (NVIDIA SOMA humanoid mesh — for visual cross-check)

The two models share a single LLM2Vec text encoder (~14 GB) to save VRAM.
With the 5090 (32 GB) this leaves comfortable headroom even with both
diffusion denoisers resident.

Outputs per prompt_id:
    motions_g1/<prompt_id>.npz       Kimodo native (replays in demo UI)
    qpos_g1/<prompt_id>.csv          MuJoCo qpos for G1 (36 cols)
    motions_soma/<prompt_id>.npz     Kimodo native (replays in demo UI)
    manifest.jsonl                   one row per prompt; tracks both models

Usage on Vast.ai:
    python generate_pilot_batch.py \\
        --csv prompts_rewritten.csv \\
        --out /workspace/kimodo_outputs

Resume: skips per-model work where the manifest already records status="ok".
If only G1 succeeded but SOMA failed, only SOMA reruns on the next pass.
"""
import argparse
import csv
import hashlib
import json
import os
import time
import traceback
from pathlib import Path

import torch

from kimodo import load_model
from kimodo.exports.motion_io import save_kimodo_npz
from kimodo.exports.mujoco import MujocoQposConverter
from kimodo.tools import seed_everything


MODELS = [
    {
        "key": "g1",
        "name": "Kimodo-G1-RP-v1",
        "motions_subdir": "motions_g1",
        "qpos_subdir": "qpos_g1",
        "save_qpos_csv": True,
    },
    {
        "key": "soma",
        "name": "Kimodo-SOMA-RP-v1.1",
        "motions_subdir": "motions_soma",
        "qpos_subdir": None,
        "save_qpos_csv": False,
    },
]


def stable_seed(prompt_id: str) -> int:
    h = hashlib.sha256(prompt_id.encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big") & 0x7FFFFFFF


def load_manifest(manifest_path: Path) -> dict:
    """prompt_id -> latest record (later writes supersede earlier ones)."""
    by_id = {}
    if not manifest_path.exists():
        return by_id
    with open(manifest_path) as f:
        for line in f:
            try:
                rec = json.loads(line)
                by_id[rec["prompt_id"]] = rec
            except json.JSONDecodeError:
                continue
    return by_id


def needs_run(prompt_id: str, manifest: dict, model_key: str) -> bool:
    rec = manifest.get(prompt_id)
    if rec is None:
        return True
    pm = rec.get("per_model", {}).get(model_key, {})
    return pm.get("status") != "ok"


def generate_one(model, text, num_frames, seed, diffusion_steps):
    seed_everything(seed)
    return model(
        [text],
        [num_frames],
        constraint_lst=[],
        num_denoising_steps=diffusion_steps,
        num_samples=1,
        multi_prompt=True,
        num_transition_frames=5,
        post_processing=False,
        return_numpy=True,
    )


def squeeze_sample(output: dict) -> dict:
    return {
        k: (v[0] if hasattr(v, "shape") and v.ndim and v.shape[0] == 1 else v)
        for k, v in output.items()
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--diffusion_steps", type=int, default=100)
    ap.add_argument("--limit", type=int, default=None,
                    help="Generate only first N prompts (for smoke test)")
    ap.add_argument("--only_model", choices=["g1", "soma"], default=None,
                    help="Run only this model (recovery / debug)")
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    for m in MODELS:
        (out_root / m["motions_subdir"]).mkdir(exist_ok=True)
        if m["qpos_subdir"]:
            (out_root / m["qpos_subdir"]).mkdir(exist_ok=True)

    manifest_path = out_root / "manifest.jsonl"
    manifest = load_manifest(manifest_path)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[setup] device: {device}")

    active_models = MODELS if args.only_model is None else [
        m for m in MODELS if m["key"] == args.only_model
    ]

    # Load each model; share the text encoder across all of them
    loaded = {}
    qpos_converters = {}
    shared_text_encoder = None

    for m in active_models:
        t0 = time.time()
        if shared_text_encoder is None:
            print(f"[setup] loading {m['name']} (with new text encoder)")
            model, resolved = load_model(
                m["name"], device=device, default_family="Kimodo",
                return_resolved_name=True,
            )
            shared_text_encoder = model.text_encoder
        else:
            print(f"[setup] loading {m['name']} (sharing text encoder)")
            model, resolved = load_model(
                m["name"], device=device, default_family="Kimodo",
                return_resolved_name=True,
                text_encoder=shared_text_encoder,
            )
        print(f"[setup]   loaded in {time.time() - t0:.1f}s; resolved={resolved}; fps={model.fps}")
        loaded[m["key"]] = {"model": model, "resolved": resolved, "cfg": m}
        if m["save_qpos_csv"]:
            qpos_converters[m["key"]] = MujocoQposConverter(model.skeleton)

    # Prompts
    with open(args.csv) as f:
        prompts = list(csv.DictReader(f))
    if args.limit:
        prompts = prompts[: args.limit]

    needed = []
    for p in prompts:
        pending = [m["key"] for m in active_models
                   if needs_run(p["prompt_id"], manifest, m["key"])]
        if pending:
            needed.append((p, pending))

    print(f"[setup] {len(prompts)} prompts total, {len(needed)} have pending work")

    t_loop = time.time()
    with open(manifest_path, "a") as mf:
        for idx, (p, pending_keys) in enumerate(needed, start=1):
            pid = p["prompt_id"]
            text = p["kimodo_input"]
            duration_s = float(p["duration_s"])
            seed = stable_seed(pid)

            # Carry forward per-model results from prior runs (so partial
            # successes are not lost when only the failed model is rerun).
            base_rec = manifest.get(pid, {})
            per_model = dict(base_rec.get("per_model", {}))

            t_prompt = time.time()
            for mk in pending_keys:
                m_info = loaded[mk]
                model = m_info["model"]
                cfg = m_info["cfg"]
                num_frames = int(duration_s * model.fps)

                t_m = time.time()
                try:
                    output = generate_one(model, text, num_frames, seed, args.diffusion_steps)
                    single = squeeze_sample(output)
                    npz_path = out_root / cfg["motions_subdir"] / f"{pid}.npz"
                    save_kimodo_npz(str(npz_path), single)

                    qpos_csv_path = ""
                    if cfg["save_qpos_csv"]:
                        conv = qpos_converters[mk]
                        qpos = conv.dict_to_qpos(output, device)
                        qpos_csv_path = str(out_root / cfg["qpos_subdir"] / f"{pid}.csv")
                        conv.save_csv(qpos, qpos_csv_path)

                    per_model[mk] = {
                        "status": "ok",
                        "model": m_info["resolved"],
                        "npz_path": str(npz_path),
                        "qpos_csv_path": qpos_csv_path,
                        "num_frames": num_frames,
                        "elapsed_s": round(time.time() - t_m, 3),
                        "error": None,
                    }
                except Exception as e:
                    per_model[mk] = {
                        "status": "fail",
                        "model": m_info["resolved"],
                        "npz_path": "",
                        "qpos_csv_path": "",
                        "num_frames": num_frames,
                        "elapsed_s": round(time.time() - t_m, 3),
                        "error": f"{type(e).__name__}: {e}",
                        "traceback": traceback.format_exc(),
                    }

            all_required = [m["key"] for m in MODELS]  # full set, not just active
            all_ok = all(per_model.get(mk, {}).get("status") == "ok" for mk in all_required)
            rec = {
                "prompt_id": pid,
                "status": "ok" if all_ok else "partial",
                "kimodo_input": text,
                "imperative_en": p["imperative_en"],
                "motion_family": p["motion_family"],
                "primitive_tag": p["primitive_tag"],
                "duration_s": duration_s,
                "seed": seed,
                "diffusion_steps": args.diffusion_steps,
                "per_model": per_model,
                "elapsed_s": round(time.time() - t_prompt, 3),
            }
            mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            mf.flush()
            os.fsync(mf.fileno())

            avg = (time.time() - t_loop) / idx
            eta_min = avg * (len(needed) - idx) / 60
            status_str = " ".join(
                f"{mk}={per_model.get(mk, {}).get('status', '-')}"
                for mk in [m["key"] for m in active_models]
            )
            print(f"[{idx:3d}/{len(needed)}] {pid} | {status_str} | "
                  f"{rec['elapsed_s']:.1f}s | avg {avg:.1f}s | ETA {eta_min:.1f} min")

    print(f"\n[done] processed {len(needed)} prompts in {(time.time()-t_loop)/60:.1f} min")
    print(f"[done] manifest: {manifest_path}")
    for m in active_models:
        print(f"[done] {m['key']:5s} motions: {out_root / m['motions_subdir']}/")
        if m["qpos_subdir"]:
            print(f"[done] {m['key']:5s} qpos:    {out_root / m['qpos_subdir']}/")


if __name__ == "__main__":
    main()
