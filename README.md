# AMASS → Unitree G1 Retargeting Pipeline

<p align="center">
  <strong>Retarget large-scale human motion capture data to the Unitree G1 humanoid robot — with multilingual text annotations.</strong>
</p>

<p align="center">
  <a href="https://huggingface.co/datasets/AdiShingote/PragyaVLA-G1-motions-dataset">
    <img src="https://img.shields.io/badge/🤗_Dataset-PragyaVLA--G1-blue" alt="HuggingFace Dataset"/>
  </a>
  <a href="https://amass.is.tue.mpg.de/">
    <img src="https://img.shields.io/badge/Source-AMASS-orange" alt="AMASS"/>
  </a>
  <a href="https://github.com/NVlabs/ProtoMotions">
    <img src="https://img.shields.io/badge/Framework-ProtoMotions-green" alt="ProtoMotions"/>
  </a>
  <img src="https://img.shields.io/badge/Sequences-13%2C741-purple" alt="Sequences"/>
  <img src="https://img.shields.io/badge/Languages-EN_|_HI_|_BN_|_TA_|_TE-red" alt="Languages"/>
</p>

---

## Overview

This pipeline takes raw **AMASS** human motion-capture data, retargets every sequence to the **Unitree G1 humanoid robot** skeleton using differentiable inverse kinematics, and packages the result as a research-ready dataset with multilingual text annotations.

It was developed as part of the **PragyaVLA** project — a Vision-Language-Action model for Indian-language robot control.

### What you get

| Output | Description |
|--------|-------------|
| **13,741 `.npz` files** | One per retargeted motion sequence; G1 joint angles at each frame |
| **`g1_dataset.xlsx` / `.csv`** | Annotated dataset: HF URL + HumanML3D clip ID + timestamps + captions |
| **Multilingual captions** | English + Hindi + Bengali + Tamil + Telugu for every clip |
| **HuggingFace dataset** | [`AdiShingote/PragyaVLA-G1-motions-dataset`](https://huggingface.co/datasets/AdiShingote/PragyaVLA-G1-motions-dataset) |

### Source datasets (all from AMASS / HumanML3D)

| Dataset | Sequences |
|---------|----------:|
| KIT | 4,231 |
| BioMotionLab_NTroje | 2,958 |
| CMU | 2,082 |
| BMLmovi | 1,801 |
| Eyes_Japan_Dataset | 750 |
| BMLhandball | 649 |
| EKUT | 348 |
| ACCAD | 252 |
| MPI_HDM05 | 215 |
| Transitions_mocap | 110 |
| DFaust_67 | 129 |
| MPI_mosh | 77 |
| SFU | 44 |
| TotalCapture | 37 |
| SSM_synced | 30 |
| HumanEva | 28 |
| **Total** | **13,741** |

---

## Architecture

```
AMASS raw data (.npz, SMPL-H params)
        │
        ▼
[Step 1]  pipeline/step1_build_curated_yaml.py
          Filter AMASS to the HumanML3D-curated subset
        │
        ▼
[Step 2]  pipeline/step2_build_pt_parts.py
          Convert AMASS sequences → ProtoMotions .pt motion library
        │
        ▼
[Step 3]  pipeline/step3_retarget_pipeline.py        ← GPU required
          PyRoki differentiable IK: SMPL keypoints → G1 joint angles
          Output: retargeted_g1/{DATASET}/{motion}_keypoints_retargeted.npz
        │
        ▼
[Step 4]  pipeline/step4_push_to_hf.py
          Upload retargeted .npz files to HuggingFace
        │
        ▼
[Step 5]  pipeline/step5_build_annotations.py
          Extract HumanML3D text captions → text_annotations.json
        │
        ▼
[Step 6]  pipeline/step6_build_dataset.py
          Join retargeted files + captions → Excel + CSV
        │
        ▼
[Step 7]  pipeline/step7_translate_captions.py       ← parallel workers
          Translate captions to Hindi / Bengali / Tamil / Telugu
```

---

## Prerequisites

### 1. Data

- **AMASS** — register and download from [amass.is.tue.mpg.de](https://amass.is.tue.mpg.de/) (SMPL-H format for each sub-dataset)
- **HumanML3D** — clone [github.com/EricGuo5513/HumanML3D](https://github.com/EricGuo5513/HumanML3D) to get `index.csv` and `texts/`

### 2. Framework

```bash
git clone https://github.com/addy-codes1/Protomotions-Env.git
cd Protomotions-Env
pip install -e .
pip install -r requirements_mujoco.txt
```

### 3. Python dependencies

```bash
pip install -r requirements.txt
```

### 4. GPU compute

Step 3 requires a GPU. We used an **H100 on [Vast.ai](https://vast.ai)** (~$2–3/hr).
See [`docs/vastai_guide.md`](docs/vastai_guide.md) for full setup.

### 5. Environment variables

```bash
export HF_TOKEN=hf_...           # HuggingFace write token
export AMASS_DIR=/path/to/amass_data
export HUMANML3D_DIR=/path/to/HumanML3D
```

---

## Step-by-Step Usage

### Step 1 — Filter AMASS to HumanML3D subset

```bash
python pipeline/step1_build_curated_yaml.py \
    --amass-dir $AMASS_DIR \
    --humanml3d-index $HUMANML3D_DIR/index.csv \
    --out amass_humanml3d_curated.yaml
```

### Step 2 — Build ProtoMotions motion library

```bash
python pipeline/step2_build_pt_parts.py \
    --amass-dir $AMASS_DIR \
    --curated-yaml amass_humanml3d_curated.yaml \
    --out-dir amass_pt_parts/
```

### Step 3 — Retarget to G1 (GPU required)

```bash
# On Vast.ai — one-time environment setup:
bash setup_vastai_envs.sh

# Retarget all datasets with parallel workers:
python pipeline/step3_retarget_pipeline.py \
    --keypoints-dir /workspace/keypoints \
    --output-dir /workspace/retargeted_g1 \
    --workers 6

# Or retarget a single dataset directly:
python /path/to/Protomotions-Env/pyroki/batch_retarget_to_g1_from_keypoints.py \
    --keypoints-folder-path /workspace/keypoints/CMU \
    --output-dir /workspace/retargeted_g1/CMU \
    --no-visualize --skip-existing \
    --subsample-factor 2 \
    --source-type smpl \
    --input-fps 30
```

Expected output per sequence: `retargeted_g1/{DATASET}/{motion}_keypoints_retargeted.npz`

Timing: ~6–8 hours for 13,741 sequences on an H100 with 6 parallel workers.

### Step 4 — Upload to HuggingFace

```bash
python pipeline/step4_push_to_hf.py \
    --retarget-dir /workspace/retargeted_g1 \
    --hf-repo YOUR_USERNAME/your-dataset-name \
    --all
```

Uses `upload_folder()` (one commit per dataset) to stay within HF's 128 commits/hr rate limit. Auto-retries on 429 errors after 65 minutes.

### Step 5 — Build text annotations

```bash
python pipeline/step5_build_annotations.py \
    --humanml3d-dir $HUMANML3D_DIR \
    --out proto-g1/text_annotations.json
```

### Step 6 — Build annotated dataset

```bash
python pipeline/step6_build_dataset.py \
    --hf-repo YOUR_USERNAME/your-dataset-name \
    --text-json proto-g1/text_annotations.json \
    --out g1_dataset.xlsx
```

Output columns: `npz_path` (HF URL), `clip_id`, `source_amass`, `start_s`, `end_s`, `duration_s`, `caption_1`–`4`, `all_captions`.

### Step 7 — Translate captions (parallel)

```bash
# Split for 4 parallel workers:
python pipeline/step7a_split_csv.py

# Run one command per terminal (simultaneously):
python pipeline/step7_translate_captions.py --csv g1_chunk_0.csv
python pipeline/step7_translate_captions.py --csv g1_chunk_1.csv
python pipeline/step7_translate_captions.py --csv g1_chunk_2.csv
python pipeline/step7_translate_captions.py --csv g1_chunk_3.csv

# Merge when all 4 are done:
python pipeline/step7b_merge_chunks.py
```

Adds columns: `caption_1_hi`, `caption_1_bn`, `caption_1_ta`, `caption_1_te` (and likewise for captions 2–4).

Resumable — Ctrl-C saves progress. Re-running skips already-translated cells.

---

## Output Format

Each retargeted `.npz` file:

```python
import numpy as np
data = np.load("motion_keypoints_retargeted.npz")
# data["dof_pos"]   — shape (T, 29): joint angles in radians, 15 fps
# data["root_pos"]  — shape (T, 3):  root XYZ position in world frame
# data["root_rot"]  — shape (T, 4):  root quaternion (x, y, z, w)
```

The G1 has **29 actuated DOFs**: 12 leg + 3 waist + 14 arm joints.

---

## Hardware & Time Budget

| Step | Hardware | Time |
|------|----------|------|
| Steps 1–2 (AMASS → .pt) | Any CPU | ~2–4 hrs |
| Step 3 (retargeting) | H100 GPU, 6 workers | ~6–8 hrs |
| Step 4 (HF upload) | Any, good internet | ~2–3 hrs |
| Steps 5–6 (annotations + dataset) | Any CPU | ~15 min |
| Step 7 (translation, 4 workers) | Any CPU | ~4–5 hrs |
| **Total** | | **~18–24 hrs** |

---

## Dataset

The output is publicly available for research:

**[HuggingFace: AdiShingote/PragyaVLA-G1-motions-dataset](https://huggingface.co/datasets/AdiShingote/PragyaVLA-G1-motions-dataset)**

```python
# Download a single sequence
from huggingface_hub import hf_hub_download
import numpy as np

path = hf_hub_download(
    repo_id="AdiShingote/PragyaVLA-G1-motions-dataset",
    repo_type="dataset",
    filename="KIT/_D:\\HumanML3d\\amass_data\\KIT\\3\\kick_high_left02_poses_keypoints_retargeted.npz",
)
data = np.load(path)
print(data["dof_pos"].shape)   # (T, 29)
```

---

## Project Context

This pipeline is part of **PragyaVLA** — a Vision-Language-Action model for robot control via Indian language instructions.

Related repositories:
- **[SushOS/IndicVLA](https://github.com/SushOS/IndicVLA)** — multilingual instruction dataset
- **[addy-codes1/Protomotions-Env](https://github.com/addy-codes1/Protomotions-Env)** — ProtoMotions fork with full pipeline scripts

---

## Acknowledgements

- **[AMASS](https://amass.is.tue.mpg.de/)** (Mahmood et al., 2019)
- **[HumanML3D](https://github.com/EricGuo5513/HumanML3D)** (Guo et al., CVPR 2022)
- **[ProtoMotions](https://github.com/NVlabs/ProtoMotions)** (NVIDIA)
- **[PyRoki](https://github.com/chungmin99/pyroki)** — differentiable robot kinematics
- **[IndicTrans2](https://github.com/AI4Bharat/IndicTrans2)** (AI4Bharat)

---

## Citation

```bibtex
@misc{pragyavla2025,
  title        = {PragyaVLA: Vision-Language-Action for Indian Language Robot Control},
  author       = {Shingote, Aditya and Suresh, Sushmit and others},
  year         = {2025},
  howpublished = {\url{https://github.com/SushOS/IndicVLA/tree/g1-retargeting-pipeline}},
}
```

---

## License

Pipeline code: **Apache 2.0**

Retargeted motion data inherits **CC BY-NC 4.0** from AMASS (non-commercial research use only).
