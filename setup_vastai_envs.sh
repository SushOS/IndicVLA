#!/bin/bash
# setup_vastai_envs.sh
# --------------------
# One-time environment setup for the Vast.ai GPU instance.
# Run this ONCE after renting the instance, before starting the pipeline.
#
# What it does:
#   1. Installs huggingface_hub in the base conda env
#   2. Clones ProtoMotions repo (or updates if already present)
#   3. Creates 'protomotions' conda env (PyTorch + protomotions package)
#   4. Creates 'pyroki' conda env (JAX[cuda12] + pyroki package)
#   5. Creates all workspace directories
#   6. Copies pipeline script to /workspace
#
# Usage (run on Vast.ai via SSH):
#   bash setup_vastai_envs.sh
#
# NOTE: This takes ~10-15 minutes on first run due to conda env creation.
# Run it inside a screen session:
#   screen -S setup
#   bash setup_vastai_envs.sh
#   Ctrl-A D  (detach, check later with: screen -r setup)

set -e  # exit on any error

WORKSPACE=/workspace
PROTO_DIR=$WORKSPACE/ProtoMotions

# ╔══════════════════════════════════════════════════════╗
# ║  EDIT THIS: your ProtoMotions repo URL               ║
# ╠══════════════════════════════════════════════════════╣
# ║  Options:                                            ║
# ║    1. Public repo: https://github.com/USER/REPO.git  ║
# ║    2. NVlabs original: NVlabs/ProtoMotions not yet   ║
# ║       public — use your fork instead                 ║
# ╚══════════════════════════════════════════════════════╝
PROTO_REPO_URL="https://github.com/addy-codes1/Protomotions-Env.git"

echo ""
echo "══════════════════════════════════════════════════════"
echo "  Vast.ai Environment Setup"
echo "  Workspace: $WORKSPACE"
echo "══════════════════════════════════════════════════════"
echo ""

# ── 0. Verify conda is available ─────────────────────────────────────────────
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found. Is this the right Vast.ai image?"
    echo "  Use an image with 'pytorch' or 'cuda' in the name, e.g.:"
    echo "  vastai/pytorch:2.1.0-cuda12.1-py310"
    exit 1
fi
echo "[OK] conda found: $(conda --version)"

# ── 1. Install huggingface_hub in base env ────────────────────────────────────
echo ""
echo "── Installing huggingface_hub in base env..."
pip install huggingface_hub -q
echo "[OK] huggingface_hub installed"

# ── 2. Clone / update ProtoMotions ───────────────────────────────────────────
echo ""
echo "── Setting up ProtoMotions repo..."
if [ ! -d "$PROTO_DIR" ]; then
    echo "Cloning from $PROTO_REPO_URL ..."
    git clone "$PROTO_REPO_URL" "$PROTO_DIR"
    echo "[OK] Cloned to $PROTO_DIR"
else
    echo "[OK] ProtoMotions already at $PROTO_DIR (not re-cloning)"
    echo "  Run 'git -C $PROTO_DIR pull' if you want to update."
fi

# ── 3. Create 'protomotions' conda env (CPU, PyTorch) ────────────────────────
echo ""
echo "── Creating 'protomotions' conda env..."
if conda env list | grep -q "^protomotions "; then
    echo "[OK] 'protomotions' env already exists — skipping creation"
else
    conda create -n protomotions python=3.10 -y

    # PyTorch with CUDA 12 (compatible with CUDA 13.x driver via backward compat)
    conda run -n protomotions pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu121 -q

    # Install protomotions package from local clone
    conda run -n protomotions pip install -e "$PROTO_DIR" -q

    # Extra deps used by the keypoint extraction script + protomotions internals
    conda run -n protomotions pip install typer tqdm numpy easydict \
        lightning tensordict wandb omegaconf hydra-core pyyaml -q

    echo "[OK] 'protomotions' env ready"
fi

# Smoke test
echo "  Smoke test: importing torch..."
conda run -n protomotions python -c "import torch; print(f'    torch {torch.__version__}')"
echo "  Smoke test: importing protomotions..."
conda run -n protomotions python -c \
    "import sys; sys.path.insert(0, '$PROTO_DIR'); from protomotions.components.motion_lib import MotionLib; print('    protomotions OK')"

# ── 4. Create 'pyroki' conda env (GPU, JAX) ───────────────────────────────────
echo ""
echo "── Creating 'pyroki' conda env..."
if conda env list | grep -q "^pyroki "; then
    echo "[OK] 'pyroki' env already exists — skipping creation"
else
    conda create -n pyroki python=3.10 -y

    # JAX with CUDA 12 wheels (works with CUDA 12.x and 13.x drivers)
    conda run -n pyroki pip install "jax[cuda12]" -q

    # PyRoki — not on PyPI, install from GitHub (chungmin99/pyroki)
    conda run -n pyroki pip install git+https://github.com/chungmin99/pyroki.git -q

    # PyRoki dependencies
    conda run -n pyroki pip install jaxlie jaxls jax-dataclasses yourdfpy -q

    # Numpy (pin to 1.x for compatibility with some JAX versions)
    conda run -n pyroki pip install "numpy<2.0" -q

    echo "[OK] 'pyroki' env ready"
fi

# Smoke test
echo "  Smoke test: importing jax + GPU check..."
conda run -n pyroki python -c "
import jax
devs = jax.devices()
print(f'    jax {jax.__version__}  devices={devs}')
if not any('gpu' in str(d).lower() or 'cuda' in str(d).lower() for d in devs):
    print('    WARNING: No GPU detected by JAX! Check CUDA setup.')
else:
    print('    GPU detected OK')
"

echo "  Smoke test: importing pyroki..."
conda run -n pyroki python -c "import pyroki; print(f'    pyroki OK')"

# ── 5. Create workspace directories ──────────────────────────────────────────
echo ""
echo "── Creating workspace directories..."
mkdir -p $WORKSPACE/amass_pt_parts
mkdir -p $WORKSPACE/keypoints
mkdir -p $WORKSPACE/retargeted_g1
mkdir -p $WORKSPACE/logs
mkdir -p $WORKSPACE/jax_cache
echo "[OK] Directories ready:"
ls -la $WORKSPACE/ | grep -E "amass|keypoints|retargeted|logs|jax_cache"

# ── 6. Copy pipeline scripts from HuggingFace ────────────────────────────────
# (These scripts are uploaded to the dataset repo for easy access on Vast.ai)
echo ""
echo "── Downloading pipeline scripts..."
python3 -c "
from huggingface_hub import hf_hub_download
import shutil, os
token = os.environ['HF_TOKEN']
scripts = ['vastai_retarget_pipeline.py']
for script in scripts:
    try:
        path = hf_hub_download(
            repo_id='AdiShingote/amass-smpl-dataset',
            filename=f'scripts/{script}',
            repo_type='dataset',
            token=token,
            local_dir='/workspace'
        )
        print(f'  [OK] Downloaded {script}')
    except Exception as e:
        print(f'  [SKIP] {script}: {e}')
" 2>/dev/null || echo "  (Script download from HF skipped — copy manually if needed)"

echo ""
echo "══════════════════════════════════════════════════════"
echo "  Setup complete!"
echo ""
echo "  Next steps:"
echo "    cd /workspace"
echo "    screen -S pipeline"
echo "    python vastai_retarget_pipeline.py"
echo "    Ctrl-A D   (detach; reconnect with: screen -r pipeline)"
echo ""
echo "  Or, to run specific datasets only:"
echo "    python vastai_retarget_pipeline.py --datasets HumanEva SSM_synced"
echo ""
echo "  Monitor progress:"
echo "    tail -f /workspace/logs/CMU_retarget.log"
echo "══════════════════════════════════════════════════════"
