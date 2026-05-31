# Full Dataset Retargeting on Vast.ai GPU

> Run the complete 13,879-motion AMASS → Unitree G1 retargeting pipeline on a cloud GPU instance,
> then bring back the retargeted `.npz` files + combine with text annotations into a single Excel file.

---

## Overview

| Where | What |
|---|---|
| **Local (done)** | AMASS → 13,879 `.motion` files |
| **Local (in progress)** | 13,879 `.motion` → `amass_smpl.pt` (~3.3 GB) |
| **Vast.ai GPU** | `amass_smpl.pt` → keypoints → retargeted G1 `.npz` files |
| **Local (final)** | Download `.npz` files → run `build_dataset_excel.py` → full Excel |

**Why GPU?** PyRoki uses JAX trajectory optimisation. On CPU each motion takes 30–60 seconds (after JIT warm-up). On an RTX 4090 with `jax[cuda12]` each motion takes ~3–6 seconds — roughly 10–20x faster.

**Estimated GPU wall-clock:** ~15–20 hours for 13,879 motions on RTX 4090.  
**Cost:** ~$8–16 at $0.50–0.80/hr.

---

## Step 1 — Wait for amass_smpl.pt to finish building (local)

The MotionLib build is currently running in the background. Watch the log:

```powershell
Get-Content "D:\HumanML3d\motion_lib_build.log" -Tail 5 -Wait
```

When it finishes you should see something like:
```
Loaded X/13879 motion files ...
Saving to D:\HumanML3d\amass_smpl.pt
```

Check the final file size:
```powershell
(Get-Item "D:\HumanML3d\amass_smpl.pt").Length / 1MB
# Expected: ~3,200-3,500 MB
```

---

## Step 2 — Provision a Vast.ai Instance

### 2.1 Instance spec (recommended)

| Parameter | Value |
|---|---|
| GPU | **RTX 4090** (24 GB VRAM) |
| RAM | ≥ 32 GB |
| Disk | ≥ 150 GB |
| CUDA | **12.x** (must be ≥ 12.0) |
| Python | 3.10 or 3.12 |
| OS | Ubuntu 22.04 |

**Why RTX 4090?** JAX single-GPU performance is best on consumer-class 4090 (16,384 CUDA cores, 82.6 TFLOPS FP32). A100/H100 would be faster but cost 3-5x more per hour for this workload.

### 2.2 Steps on Vast.ai website

1. Go to https://vast.ai/console/create/ → click **"Templates"**
2. Select **"PyTorch"** (base image with CUDA 12 + conda pre-installed)
3. Filter GPU: `RTX 4090`, sort by price
4. Pick an instance — recommended: ≥32 GB RAM, ≥150 GB disk
5. Under **"Storage"** set disk to **150 GB**
6. Click **"Rent"**, wait for instance to start (1-3 minutes)
7. Copy the **SSH command** shown in the dashboard (format: `ssh -p NNNNN root@x.x.x.x`)

### 2.3 Connect via SSH

Open a **new** PowerShell window on your local machine (keep the one running build tasks separate):

```powershell
# Paste the SSH command from Vast.ai dashboard, e.g.:
ssh -p 12345 root@123.456.78.90
```

---

## Step 3 — Set Up Environments on Vast.ai

All commands below run **on the Vast.ai SSH session** (Linux).

### 3.1 Install Miniconda (if not pre-installed)

```bash
# Check if conda exists
which conda

# If not found, install:
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda init bash && source ~/.bashrc
```

### 3.2 Clone repositories

```bash
cd /workspace    # or $HOME — use a location with ≥150 GB free

# ProtoMotions
git clone https://github.com/NVlabs/ProtoMotions.git
cd ProtoMotions
git checkout main

# PyRoki (NOT on PyPI — must clone)
cd /workspace
git clone https://github.com/chungmin99/pyroki.git
```

### 3.3 Create ProtoMotions env

```bash
conda create -n protomotions python=3.10 -y
conda activate protomotions

cd /workspace/ProtoMotions

# Install PyTorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install ProtoMotions + dependencies
pip install -e .
pip install -r requirements_mujoco.txt
pip install easydict
```

### 3.4 Create PyRoki env  (KEY DIFFERENCE from local: jax[cuda12])

```bash
conda create -n pyroki python=3.10 -y
conda activate pyroki

cd /workspace/pyroki
pip install -e .

# *** CRITICAL GPU CHANGE ***
# Local CPU install:   pip install "jax[cpu]"
# Vast.ai GPU install: pip install "jax[cuda12]"   <-- this line only
pip install "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Verify GPU is visible to JAX
python -c "import jax; print(jax.devices())"
# Should print: [CudaDevice(id=0)]
```

---

## Step 4 — Transfer amass_smpl.pt to Vast.ai

Run this **on your local machine** (in a separate PowerShell window):

```powershell
# Replace PORT and IP with values from your Vast.ai dashboard
$VAST_PORT = "12345"
$VAST_IP   = "123.456.78.90"

# Upload amass_smpl.pt (~3.3 GB)
scp -P $VAST_PORT "D:\HumanML3d\amass_smpl.pt" root@${VAST_IP}:/workspace/amass_smpl.pt
```

Expected transfer time: ~5–20 minutes depending on your upload speed.

---

## Step 5 — Run the Pipeline on Vast.ai

All commands below run on the **Vast.ai SSH session**.

### Step A — Extract keypoints from amass_smpl.pt

```bash
conda activate protomotions
cd /workspace/ProtoMotions
export PYTHONPATH=/workspace/ProtoMotions

python data/scripts/extract_retargeting_input_keypoints_from_packaged_motionlib.py \
    /workspace/amass_smpl.pt \
    --output-path /workspace/keypoints-for-retarget \
    --skeleton-format smpl \
    --start-idx 0 \
    --skip-freq 1

# Output: /workspace/keypoints-for-retarget/   (~5 GB, one .npy per motion)
# Time: ~30-60 minutes
```

### Step B — PyRoki retarget to G1 (GPU-accelerated)

```bash
conda activate pyroki
cd /workspace/ProtoMotions
export PYTHONPATH=/workspace/ProtoMotions

python pyroki/batch_retarget_to_g1_from_keypoints.py \
    --keypoints-folder-path /workspace/keypoints-for-retarget \
    --source-type smpl \
    --subsample-factor 1 \
    --output-dir /workspace/retargeted-g1 \
    --no-visualize \
    --skip-existing

# Output: /workspace/retargeted-g1/   (one _keypoints_retargeted.npz per motion)
# Time: ~15-20 hours on RTX 4090
# First motion: ~5-10 min (JAX JIT compilation), subsequent: ~3-6 sec each
```

> **Tip:** Run this inside a `screen` or `tmux` session so it survives SSH disconnects:
> ```bash
> screen -S pyroki
> # run the command above
> # detach: Ctrl+A then D
> # reattach: screen -r pyroki
> ```

### Step C — Extract contact labels

```bash
# Still in pyroki env
python pyroki/batch_retarget_to_g1_from_keypoints.py \
    --keypoints-folder-path /workspace/keypoints-for-retarget \
    --source-type smpl \
    --subsample-factor 1 \
    --save-contacts-only \
    --contacts-dir /workspace/contacts \
    --skip-existing

# Time: ~15-20 hours (same loop, lighter compute)
```

> **Run Steps B and C sequentially** (don't run simultaneously — JAX will compete for GPU memory).

### Step D — Convert to ProtoMotions format (optional — only needed if you want proto-g1.pt for training)

```bash
conda activate protomotions
cd /workspace/ProtoMotions
export PYTHONPATH=/workspace/ProtoMotions

python data/scripts/convert_pyroki_retargeted_robot_motions_to_proto.py \
    --retargeted-motion-dir /workspace/retargeted-g1 \
    --output-dir /workspace/proto-g1 \
    --robot-type g1 \
    --contact-labels-dir /workspace/contacts \
    --apply-motion-filter \
    --force-remake

# Time: ~30-60 minutes
```

### Step E — Package final MotionLib (optional — only needed for training)

```bash
python protomotions/components/motion_lib.py \
    --motion-path /workspace/proto-g1 \
    --output-file /workspace/proto-g1.pt

# Time: ~10-15 minutes
# Output: /workspace/proto-g1.pt  (ready for G1 motion-tracking training)
```

---

## Step 6 — Download Retargeted Files Back to Local Machine

Run this **on your local machine**:

```powershell
$VAST_PORT = "12345"
$VAST_IP   = "123.456.78.90"

# Create local destination directory
New-Item -ItemType Directory -Force "D:\HumanML3d\retargeted-g1-full"

# Download all retargeted .npz files
# Using scp with recursive flag:
scp -P $VAST_PORT -r root@${VAST_IP}:/workspace/retargeted-g1 "D:\HumanML3d\retargeted-g1-full"

# Alternatively with rsync (faster, resumable — use Git Bash or WSL):
rsync -avz --progress -e "ssh -p $VAST_PORT" \
    root@${VAST_IP}:/workspace/retargeted-g1/ \
    "D:/HumanML3d/retargeted-g1-full/"
```

Expected download size: ~2–7 GB (13,879 files × ~150–500 KB each).

---

## Step 7 — Build the Final Excel File (local)

Once the `.npz` files are downloaded, run on your local machine:

```powershell
& "D:\HumanML3d\protomotions_env\Scripts\python.exe" "D:\HumanML3d\build_dataset_excel.py" `
    --retargeted-dir "D:\HumanML3d\retargeted-g1-full" `
    --text-json "D:\HumanML3d\proto-g1\text_annotations.json" `
    --out "D:\HumanML3d\g1_humanml3d_dataset_full.xlsx"
```

Expected output:
```
Found 13XXX retargeted .npz files in D:\HumanML3d\retargeted-g1-full
Annotations in JSON     : 13425
AMASS files missing .npz:    0  (not yet retargeted)
Rows written            : ~13425
Output                  : D:\HumanML3d\g1_humanml3d_dataset_full.xlsx
```

The final Excel file will have ~13,425 rows with columns:
`npz_path | clip_id | source_amass | start_s | end_s | duration_s | caption_1 | caption_2 | caption_3 | all_captions`

---

## Summary: What Changes for GPU vs Local

| Item | Local (CPU) | Vast.ai (GPU) |
|---|---|---|
| JAX install | `pip install jax[cpu]` | `pip install "jax[cuda12]"` |
| Per-motion PyRoki time | ~30–60 sec | ~3–6 sec |
| Total PyRoki time (13,879 motions) | ~100–200 hours | ~15–20 hours |
| SMPL `.pkl` models needed | Yes (for conversion) | **No** (not needed — `.pt` file already preprocessed) |
| ProtoMotions env | Same | Same (`pip install torch --index-url https://download.pytorch.org/whl/cu121`) |
| Other pipeline steps | Identical | Identical |

---

## Checklist

- [ ] `amass_smpl.pt` build complete on local machine
- [ ] Vast.ai RTX 4090 instance provisioned (CUDA 12.x)
- [ ] `protomotions` conda env created on Vast.ai
- [ ] `pyroki` conda env created on Vast.ai with `jax[cuda12]`
- [ ] `jax.devices()` shows `[CudaDevice(id=0)]`
- [ ] `amass_smpl.pt` uploaded to Vast.ai
- [ ] Step A (keypoint extraction) complete
- [ ] Step B (PyRoki retargeting) complete — running in `screen`/`tmux`
- [ ] Step C (contact extraction) complete
- [ ] Retargeted `.npz` files downloaded locally
- [ ] `build_dataset_excel.py` run → `g1_humanml3d_dataset_full.xlsx` verified
- [ ] Vast.ai instance stopped/destroyed (to stop billing)

---

## Estimated Timeline

| Step | Time |
|---|---|
| Build amass_smpl.pt (local, in progress) | ~15–30 min |
| Upload to Vast.ai (3.3 GB) | ~10–20 min |
| Env setup on Vast.ai | ~15 min |
| Step A: keypoint extraction | ~30–60 min |
| Step B: PyRoki retargeting (GPU) | **~15–20 hours** |
| Step C: contact extraction | ~8–12 hours |
| Step D+E: convert + package (optional) | ~1 hour |
| Download retargeted .npz (local) | ~20–40 min |
| Build Excel (local) | < 1 min |
| **Total (excluding optional D+E)** | **~36–54 hours** |

> Steps B and C dominate. Once Step B starts, you can leave it running overnight.
> Vast.ai instance cost at ~$0.60/hr × 36 hours = **~$22 total**.
