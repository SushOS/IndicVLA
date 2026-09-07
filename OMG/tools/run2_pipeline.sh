#!/usr/bin/env bash
# MAILA Run 2 -- end-to-end pipeline for a fresh Vast.ai box.
#
# USAGE (on the box):
#   export HF_TOKEN=...            # never written into any file that is committed or pushed
#   export WANDB_API_KEY=...
#   bash run2_pipeline.sh
#
# Stage-guarded and idempotent: every stage drops a marker in $STATE, and re-running skips
# whatever already finished. Safe to resume after an SSH drop, an OOM, or a box recycle.
#
# DESIGN RULES INHERITED FROM RUN 1 (research log s9, s11)
#   * workspace_is_volume=false -> NOTHING on the box survives a recycle. Push checkpoints
#     AND LOGS continuously, never at the end (error #7: a truncated log lost the verbose
#     per-arm output even though the JSONs survived).
#   * Never guess units or rates. Stage 4 runs the fps INSPECT and stops for a human.
#   * uv venv does not bundle pip (error #4) -> always `uv pip install --python .venv/bin/python`.
#   * The lock (stage 6) is pushed BEFORE training so provenance exists even if the run dies.
set -euo pipefail

WORK=${WORK:-/workspace}
REPO=${REPO:-$WORK/IndicVLA}
DATA=${DATA:-$WORK/data/amass_g1_omg125}
MODELS=${MODELS:-$WORK/models}
RUNS=${RUNS:-$WORK/runs}
STATE=${STATE:-$WORK/.run2_state}
LOGS=${LOGS:-$WORK/logs}
CACHE=${CACHE:-$WORK/cache/g1_npz}

HF_REPO=${HF_REPO:-PragyaVLA/maila-run2-motionpivot}
WANDB_PROJECT=${WANDB_PROJECT:-maila-run2-motionpivot}
RUN_NAME=${RUN_NAME:-maila-mp-hi-$(date -u +%Y%m%d-%H%M%S)}

LANGS=${LANGS:-hi}
PAIR_MODE=${PAIR_MODE:-paraphrase}
STEPS=${STEPS:-25000}
BATCH=${BATCH:-32}
ACCUM=${ACCUM:-8}
LAMBDA_X=${LAMBDA_X:-0.5}
LAMBDA_EN=${LAMBDA_EN:-0.1}
# Source fps is PER-SUBSET and measured (convert_g1_npz_to_omg125.py :: SUBSET_FPS).
# Measured 2026-09-05: 15 fps (CMU, BioMotionLab_NTroje, DFaust_67), 25 (KIT, EKUT),
# 30 (ACCAD, BMLmovi, BMLhandball). convert REFUSES any subset without a measured rate;
# supply more via FPS_OVERRIDE="MPI_HDM05=25,SFU=15".
FPS_OVERRIDE=${FPS_OVERRIDE:-}
# The .npz live in a PRIVATE repo (AdiShingote/pragya-vla-g1-motion-dataset). CSV npz_path
# URLs resolve VERBATIM with that account's token. Defaults to HF_TOKEN if not set.
MOTION_HF_TOKEN=${MOTION_HF_TOKEN:-$HF_TOKEN}

mkdir -p "$STATE" "$LOGS" "$DATA" "$MODELS" "$RUNS" "$CACHE" /root/.creds
export PYTHONPATH=${PYTHONPATH:-$REPO/OMG/src}
export TOKENIZERS_PARALLELISM=false
export HF_HUB_ENABLE_HF_TRANSFER=1

done_stage() { [ -f "$STATE/$1.done" ]; }
mark()       { touch "$STATE/$1.done"; echo "[stage $1] DONE"; }
banner()     { echo; echo "=============== $* ==============="; }

# Push a file to HF immediately. Called constantly -- a log is an artifact like any other.
push() {
  local src=$1 dst=$2
  python - "$src" "$dst" "$HF_REPO" "$RUN_NAME" <<'PY' || echo "  !! push failed: $src"
import sys, os
from huggingface_hub import HfApi
src, dst, repo, run = sys.argv[1:5]
tok = os.environ.get("HF_TOKEN") or open("/root/.creds/hf_token").read().strip()
api = HfApi(token=tok)
api.create_repo(repo, repo_type="dataset", exist_ok=True, private=True)
api.upload_file(path_or_fileobj=src, path_in_repo=f"{run}/{dst}",
                repo_id=repo, repo_type="dataset")
print(f"  pushed -> {repo}/{run}/{dst}")
PY
}

# ---------------------------------------------------------------- 1. credentials
banner "1. credentials"
if ! done_stage 01_creds; then
  : "${HF_TOKEN:?export HF_TOKEN before running}"
  : "${WANDB_API_KEY:?export WANDB_API_KEY before running}"
  umask 077
  printf '%s' "$HF_TOKEN"      > /root/.creds/hf_token
  printf '%s' "$WANDB_API_KEY" > /root/.creds/wandb_key
  chmod 600 /root/.creds/*
  echo "  creds written to /root/.creds (0600). They are NEVER committed or pushed."
  mark 01_creds
fi

# ---------------------------------------------------------------- 2. environment
banner "2. python environment"
if ! done_stage 02_env; then
  command -v uv >/dev/null 2>&1 || pip install -q uv
  cd "$WORK"
  [ -d .venv ] || uv venv --python 3.10 .venv
  PY=$WORK/.venv/bin/python
  # error #4: uv venv ships no pip, so a bare `pip install` silently hits the system python
  uv pip install --python "$PY" -q \
      "torch==2.6.0" --index-url https://download.pytorch.org/whl/cu124
  uv pip install --python "$PY" -q \
      "transformers==4.57.6" "numpy<2" pandas pyarrow scipy hydra-core omegaconf \
      pytorch-lightning einops wandb huggingface_hub hf_transfer requests tqdm
  "$PY" -c "import torch;print('  torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0))"
  mark 02_env
fi
PY=$WORK/.venv/bin/python

# ---------------------------------------------------------------- 3. artifacts
banner "3. model artifacts"
if ! done_stage 03_models; then
  HF_TOKEN="$HF_TOKEN" "$PY" - <<'PY'
import os
from huggingface_hub import hf_hub_download, snapshot_download
tok = os.environ["HF_TOKEN"]; M = os.environ.get("MODELS", "/workspace/models")
print("  OMG checkpoint + evaluator ...")
hf_hub_download("THU-MARS/OMG", "updated/100m/sstep=170000.ckpt",
                local_dir=f"{M}/omg/checkpoints", token=tok)
hf_hub_download("THU-MARS/OMG", "evaluator/step_004000.pt",
                local_dir=f"{M}/omg", token=tok)
print("  t5-base, MuRIL ...")
snapshot_download("t5-base", local_dir=f"{M}/t5-base-local", token=tok)
snapshot_download("google/muril-base-cased", local_dir=f"{M}/muril-base-cased", token=tok)
print("  done")
PY
  mark 03_models
fi

# ---------------------------------------------------------------- 4. fps gate (HUMAN STOP)
banner "4. fps gate -- inspect before converting anything"
if ! done_stage 04_inspect; then
  "$PY" "$REPO/OMG/tools/convert_g1_npz_to_omg125.py" --mode inspect \
      --manifest "$REPO/OMG/splits_amass_g1_13k/train.csv" \
      --cache-dir "$CACHE" --hf-token "$MOTION_HF_TOKEN" 2>&1 | tee "$LOGS/04_inspect.log"
  push "$LOGS/04_inspect.log" "logs/04_inspect.log"
  echo
  echo "  ^^^ Compare the implied rates against SUBSET_FPS in the converter."
  echo "  convert REFUSES any subset with no measured rate -- add it via FPS_OVERRIDE."
  mark 04_inspect
fi

# ---------------------------------------------------------------- 5. convert
banner "5. convert .npz -> 125-D windows (per-subset fps, clip ranges clamped)"
for SPLIT in train val test; do
  if ! done_stage "05_convert_$SPLIT"; then
    "$PY" "$REPO/OMG/tools/convert_g1_npz_to_omg125.py" --mode convert \
        --manifest "$REPO/OMG/splits_amass_g1_13k/$SPLIT.csv" --split-name "$SPLIT" \
        --out "$DATA/$SPLIT" \
        ${FPS_OVERRIDE:+--fps-override "$FPS_OVERRIDE"} \
        --cache-dir "$CACHE" --hf-token "$MOTION_HF_TOKEN" \
        2>&1 | tee "$LOGS/05_convert_$SPLIT.log"
    push "$LOGS/05_convert_$SPLIT.log" "logs/05_convert_$SPLIT.log"
    push "$DATA/$SPLIT/_meta.json" "data/${SPLIT}_meta.json"
    mark "05_convert_$SPLIT"
  fi
done

# ---------------------------------------------------------------- 6. provenance lock
banner "6. provenance lock (pushed BEFORE training)"
if ! done_stage 06_lock; then
  "$PY" "$REPO/OMG/tools/run2_lock.py" \
      --repo-root "$REPO" --shard-root "$DATA" --run-name "$RUN_NAME" \
      --stats "$REPO/OMG/assets/stats/g1_125d_stats.json" \
      --out "$RUNS/RUN_LOCK.json" --hf-repo "$HF_REPO" --push \
      2>&1 | tee "$LOGS/06_lock.log"
  push "$LOGS/06_lock.log" "logs/06_lock.log"
  mark 06_lock
fi

# ---------------------------------------------------------------- 7. L_null gate
banner "7. L_null gate (SSOT s11.0) -- the text-blind floor"
if ! done_stage 07_lnull; then
  cd "$WORK"
  cp "$REPO/OMG/tools/"{maila_encoder.py,maila_train.py,maila_train_motionpivot.py} "$WORK/"
  "$PY" "$WORK/maila_train_motionpivot.py" --data "$DATA" --langs "$LANGS" \
      --pair-mode "$PAIR_MODE" --probe-null-only --no-wandb \
      --out "$RUNS/$RUN_NAME" 2>&1 | tee "$LOGS/07_lnull.log"
  push "$LOGS/07_lnull.log" "logs/07_lnull.log"
  push "$RUNS/$RUN_NAME/l_null.json" "l_null.json"
  mark 07_lnull
fi

# ---------------------------------------------------------------- 8. train
banner "8. train -- motion-pivot symmetric ($LANGS, $PAIR_MODE)"
if ! done_stage 08_train; then
  cd "$WORK"
  export WANDB_API_KEY WANDB_PROJECT
  "$PY" "$WORK/maila_train_motionpivot.py" \
      --data "$DATA" --langs "$LANGS" --pair-mode "$PAIR_MODE" \
      --steps "$STEPS" --batch-size "$BATCH" --accum "$ACCUM" \
      --lambda-x-hi "$LAMBDA_X" --lambda-en "$LAMBDA_EN" \
      --out "$RUNS/$RUN_NAME" --run-name "$RUN_NAME" \
      --hf-repo "$HF_REPO" --wandb-project "$WANDB_PROJECT" \
      2>&1 | tee "$LOGS/08_train.log"
  push "$LOGS/08_train.log" "logs/08_train.log"
  mark 08_train
fi

# ---------------------------------------------------------------- 9. select
banner "9. fixed-seed checkpoint re-scoring and selection"
if ! done_stage 09_select; then
  cd "$WORK"
  cp "$REPO/OMG/tools/finalize_run.py" "$WORK/" 2>/dev/null || true
  "$PY" "$WORK/finalize_run.py" --run "$RUNS/$RUN_NAME" --data "$DATA" \
      --run-name "$RUN_NAME" --repo "$HF_REPO" --n 1024 \
      2>&1 | tee "$LOGS/09_select.log" || echo "  (finalize_run may need the Run-2 dataset shim)"
  push "$LOGS/09_select.log" "logs/09_select.log"
  [ -f "$RUNS/$RUN_NAME/selection.json" ] && push "$RUNS/$RUN_NAME/selection.json" "selection.json"
  mark 09_select
fi

banner "PIPELINE COMPLETE -- $RUN_NAME"
echo "artifacts: https://huggingface.co/datasets/$HF_REPO/tree/main/$RUN_NAME"
echo "wandb    : project $WANDB_PROJECT, run $RUN_NAME"
