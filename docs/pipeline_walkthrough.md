# HumanML3D → Unitree G1 Retargeting Pipeline

> An end-to-end, reproducible plan for obtaining the **HumanML3D** dataset and retargeting all of its motions to the **Unitree G1** (34-link kinematic tree / 29 actuated DOFs) using **ProtoMotions** + **PyRoki**.
> 
> Companion doc: [PROTOMOTIONS_CODEBASE_REFERENCE.md](PROTOMOTIONS_CODEBASE_REFERENCE.md).

---

## Table of Contents

1. [TL;DR Path Recommendation](#1-tldr-path-recommendation)
2. [What HumanML3D Actually Is](#2-what-humanml3d-actually-is)
3. [The HumanML3D Repo — File-by-File](#3-the-humanml3d-repo--file-by-file)
4. [What "Unitree G1 with 34-joint skeleton" Means in ProtoMotions](#4-what-unitree-g1-with-34-joint-skeleton-means-in-protomotions)
5. [Two Retargeting Strategies](#5-two-retargeting-strategies)
6. [Data Acquisition Prerequisites](#6-data-acquisition-prerequisites)
7. [End-to-End Pipeline — Recommended (Strategy A)](#7-end-to-end-pipeline--recommended-strategy-a)
8. [End-to-End Pipeline — HumanML3D-Filtered (Strategy B)](#8-end-to-end-pipeline--humanml3d-filtered-strategy-b)
9. [Validation, Debugging, and Quality Checks](#9-validation-debugging-and-quality-checks)
10. [Training G1 on the Retargeted Motions](#10-training-g1-on-the-retargeted-motions)
11. [Disk-Space, Compute, and Time Budget](#11-disk-space-compute-and-time-budget)
12. [Gotchas and Common Pitfalls](#12-gotchas-and-common-pitfalls)
13. [Appendix A — Index of Source Datasets in HumanML3D](#13-appendix-a--index-of-source-datasets-in-humanml3d)
14. [Appendix B — Format Cheat Sheet (SMPL / HumanML3D / Keypoints / .motion)](#14-appendix-b--format-cheat-sheet)

---

## 1. TL;DR Path Recommendation

You said "retarget HumanML3D to Unitree G1 with a 34-joint skeleton using ProtoMotions." The cleanest, most reproducible path is:

> **Strategy A (recommended):** Download AMASS yourself, run **ProtoMotions's** `convert_amass_to_motionlib.py` (skipping HumanML3D's pipeline entirely), then run `scripts/retarget_amass_to_robot.sh ... g1` to retarget via PyRoki.
> 
> **Strategy B:** Use HumanML3D's `index.csv` as a curation filter on top of AMASS so you retarget exactly the 14,616 HumanML3D-curated clips (plus get HumanML3D's text annotations). The HumanML3D notebooks themselves remain useful only for *(a)* obtaining the pre-curated index and *(b)* keeping the source→clip mappings — the actual retargeting still goes through ProtoMotions.

The HumanML3D `new_joints/*.npy` files (root-centered 22-joint positions, normalized-to-face-+Z) are **not directly usable** by ProtoMotions' PyRoki retargeter — they have been stripped of global root rotation/translation and lack the (T, 18, 3, 3) orientation matrices the retargeter expects. You need to start from AMASS poses.

The Unitree G1 model used by ProtoMotions has:

- **29 actuated DOFs** (12 leg + 3 waist + 14 arm)
- **33 rigid bodies** in the training MJCF (`mjcf/g1_holo_compat.xml`)
- **43 links** in the retargeting URDF (`urdf/for_retargeting/g1.urdf`, including IMU + sensor + decorative links)
- "34" in your phrasing = **33 link bodies + 1 free-floating root joint** (the way URDF/SDF count kinematic-tree entities). PyRoki retargets all 29 actuated DOFs to match 15 semantic keypoints.

---

## 2. What HumanML3D Actually Is

**HumanML3D** (Guo et al., CVPR 2022 — paper "Generating Diverse and Natural 3D Human Motions From Text") is **not a new motion-capture corpus**. It is:

1. A **curated 14,616-clip subset** drawn from AMASS (17 sub-datasets) + HumanAct12, downsampled to 20 fps, with each clip trimmed to 2–10 seconds. Total: 28.59 hours of motion, 5,371-word vocabulary.
2. A **263-D motion-representation pipeline** (a deterministic feature extractor) that turns the 22-joint SMPL skeleton into a translation-, root-rotation-invariant feature vector for text→motion learning.
3. A **text-annotation corpus**: 44,970 sentences (3–4 per clip) collected on Amazon Mechanical Turk.

Each clip is referenced by a 6-digit ID and stored as:
- `new_joints/XXXXXX.npy` — `[T, 22, 3]` 3D joint positions in the HumanML3D normalized frame (root XZ-centered, face direction → +Z, floor at y=0)
- `new_joint_vecs/XXXXXX.npy` — `[T, 263]` motion feature vector
- `texts/XXXXXX.txt` — 3–4 lines of text annotations + POS tags + start/end time markers

Mirrored ("M") variants (`M000000.npy`) double the dataset to 29,232 clips.

**Critical**: the repo **does not distribute the raw motion data** because the AMASS license forbids redistribution. You must obtain AMASS yourself and re-run the notebooks. The repo ships **only one** sample file (`012314.npy`) in `new_joints/` and `new_joint_vecs/` for verification.

### Index: who feeds HumanML3D?

`index.csv` (14,616 rows) maps each clip ID to its source. Top sources (by row prefix in `index.csv`):

- KIT (KIT-ML) — many entries
- humanact12 (HumanAct12 — already in `pose_data/humanact12.zip`)
- CMU, BMLmovi, BioMotionLab_NTroje (a.k.a. BMLrub), MPI_HDM05, MPI_mosh, MPI_Limits (PosePrior), Eyes_Japan_Dataset, SSM_synced, TotalCapture, TCD_handMocap, SFU, EKUT, HumanEva, BMLhandball, Transitions_mocap, DFaust_67, ACCAD

See [Appendix A](#13-appendix-a--index-of-source-datasets-in-humanml3d).

---

## 3. The HumanML3D Repo — File-by-File

```
HumanML3D/
├── README.md                          # data-acquisition instructions
├── LICENSE                            # MIT
├── environment.yaml                   # conda env "torch_render" (Python 3.7.10)
├── index.csv                          # 14,616 rows: source_path,start_frame,end_frame,new_name
├── dataset_showcase.png               # paper figure
│
├── paramUtil.py                       # skeleton tables (t2m, KIT)
├── text_process.py                    # text tokenization helpers
│
├── common/
│   ├── __init__.py
│   ├── quaternion.py                  # qmul_np, qrot_np, qbetween_np, quaternion_to_cont6d_np, cont6d_to_matrix_np, qfix, ...
│   └── skeleton.py                    # class Skeleton: inverse_kinematics_np, forward_kinematics(_np / _cont6d{,_np}), get_offsets_joints
│
├── human_body_prior/                  # full SMPL+H body model code (from MPI prior code)
│   ├── body_model/                    # BodyModel class (loads SMPL+H + DMPL)
│   ├── data/                          # dataloaders for SMPL+H
│   ├── models/                        # neural priors (unused by HumanML3D notebooks)
│   ├── tools/                         # omni_tools (copy2cpu, etc.)
│   ├── train/                         # training scripts (unused)
│   └── visualizations/                # rendering helpers
│
├── pose_data/
│   └── humanact12.zip                 # 1191 HumanAct12 clips already shipped
│
├── raw_pose_processing.ipynb          # NOTEBOOK 1: AMASS .npz → joints/*.npy [T,22,3]
├── motion_representation.ipynb        # NOTEBOOK 2: joints/*.npy → new_joints/*.npy + new_joint_vecs/*.npy [T,263]
├── cal_mean_variance.ipynb            # NOTEBOOK 3: compute Mean.npy and Std.npy (groupwise std)
├── animation.ipynb                    # OPTIONAL: render mp4 animations
│
└── HumanML3D/                         # final output directory (mostly empty in repo)
    ├── new_joints/012314.npy          # one sample (3D positions)
    ├── new_joint_vecs/012314.npy      # one sample (263-dim feature)
    ├── Mean.npy, Std.npy              # 263-dim normalization stats
    ├── texts.zip                      # text annotations (44,970 sentences)
    ├── all.txt / train.txt / val.txt / test.txt / train_val.txt   # split lists
```

### 3.1 `raw_pose_processing.ipynb` — Stage 1 (AMASS → joints)

For each AMASS `.npz` (which contains `poses` axis-angle [T_src, 156], `trans` [T_src, 3], `betas` [16], `gender`, `mocap_framerate`):

1. Loads SMPL+H male/female body model from `body_models/smplh/{male,female}/model.npz` and DMPL from `body_models/dmpls/{male,female}/model.npz`.
2. **Downsamples** to 20 fps: `down_sample = int(fps / 20)`; takes every Nth frame.
3. Splits poses: `root_orient` (3), `pose_body` (63), `pose_hand` (90).
4. Runs SMPL+H **forward pass** → joint positions `Jtr` (52 joints with hands).
5. Multiplies by `trans_matrix = [[1,0,0],[0,0,1],[0,1,0]]` (Y/Z axis swap to ProtoMotions-friendly coordinate convention).
6. Saves `./pose_data/<dataset>/<file>.npy` with shape `[T_20fps, 52, 3]`.

Then iterates over `index.csv` (14,616 rows):
- Reads the corresponding `pose_data/.../<file>.npy`.
- Applies dataset-specific frame trims (Eyes_Japan: skip first 3 s; MPI_HDM05: skip 3 s; TotalCapture: skip 1 s; MPI_Limits: skip 1 s; Transitions: skip 0.5 s).
- Slices `[start_frame:end_frame]`.
- Mirrors `data[..., 0] *= -1` and calls `swap_left_right` to produce the M-prefixed version.
- Writes `./joints/XXXXXX.npy` and `./joints/MXXXXXX.npy` with shape `[T, 22, 3]` (first 22 SMPL joints — no fingers).

### 3.2 `motion_representation.ipynb` — Stage 2 (joints → HumanML3D features)

For each `./joints/*.npy`:

1. **Skeleton uniforming** via `uniform_skeleton()`: scale all clips to a uniform target skeleton (derived from `000021.npy`).
2. **Floor alignment**: subtract `min y` so feet rest at y=0.
3. **XZ centering**: shift root XZ to origin (preserve y).
4. **Face direction normalization**:
   - Across-vector = `(r_hip - l_hip) + (r_shoulder - l_shoulder)` (joints 2, 1, 17, 16).
   - Forward = cross([0,1,0], across).
   - Rotate motion so forward → +Z.
5. **Foot-contact detection** (`foot_detect`): velocity² threshold 0.002 m²/frame → binary contact per ankle/toe.
6. **Root rotation extraction**: IK → quaternions, convert root to continuous 6D, compute angular vel.
7. **Local joint transform** (`get_rifke`): root-center XZ and rotate joints into root-local frame.
8. **Assemble 263-D feature**:
   ```
   [0:1]    r_velocity_y  (1)  — root angular vel around Y
   [1:3]    l_velocity    (2)  — root XZ linear velocity
   [3:4]    root_y        (1)  — root height
   [4:67]   ric_data     (63)  — 21 local joint positions × 3
   [67:193] rot_data    (126)  — 21 joint 6D rotations × 6
   [193:259] local_vel   (66)  — 22 joint velocities × 3 (root-local frame)
   [259:263] foot_contact (4)  — [L_ankle, L_toe, R_ankle, R_toe]
   ```
9. Writes:
   - `./HumanML3D/new_joints/XXXXXX.npy` — `[T, 22, 3]` joint positions in HumanML3D normalized frame.
   - `./HumanML3D/new_joint_vecs/XXXXXX.npy` — `[T-1, 263]` feature vector.

### 3.3 `cal_mean_variance.ipynb` — Stage 3 (compute z-score stats)

- Concatenates all `new_joint_vecs/*.npy` (skipping NaN files: `007975.npy`, `M007975.npy`).
- Computes per-feature mean / std.
- **Group-normalizes** the std (one shared scalar per semantic group: root vel, root pos, ric, rot, local_vel, foot_contact) to avoid blowing up small features.
- Saves `Mean.npy` (263), `Std.npy` (263).

### 3.4 `animation.ipynb` (optional)

Renders mp4 animations of motions via matplotlib 3D. Requires `ffmpeg==4.3.1`. Not needed for retargeting.

### 3.5 `common/quaternion.py` — utility highlights

| Function | Purpose |
|---|---|
| `qinv_np(q)` | quaternion conjugate (xyz negated) |
| `qmul_np(q, r)` | quaternion multiplication |
| `qrot_np(q, v)` | rotate vector v by quaternion q |
| `qbetween_np(v0, v1)` | smallest rotation v0→v1 |
| `qfix(q)` | enforce continuity (flip sign if dot < 0) |
| `quaternion_to_cont6d_np` | quat → 6D continuous representation (first two cols of R) |
| `cont6d_to_matrix_np` | 6D → 3×3 rotation matrix via Gram-Schmidt |

### 3.6 `common/skeleton.py` — `Skeleton` class

- `inverse_kinematics_np(joints, face_joint_idx, smooth_forward=False)` — IK from positions to quat_params using a chosen face direction (face_joint_idx = `[r_hip, l_hip, sdr_r, sdr_l]`).
- `forward_kinematics{,_np,_cont6d,_cont6d_np}` — FK with full or no root rotation.
- `get_offsets_joints[_batch]` — derive bone offsets from a reference pose.

### 3.7 `paramUtil.py` — skeleton tables

`t2m_kinematic_chain` (22-joint SMPL, used by HumanML3D):

| Chain | Indices | Bodies |
|---|---|---|
| Right leg | `[0, 2, 5, 8, 11]` | pelvis → R_hip → R_knee → R_ankle → R_foot |
| Left leg  | `[0, 1, 4, 7, 10]` | pelvis → L_hip → L_knee → L_ankle → L_foot |
| Spine     | `[0, 3, 6, 9, 12, 15]` | pelvis → spine1 → spine2 → spine3 → neck → head |
| Right arm | `[9, 14, 17, 19, 21]` | spine3 → R_collar → R_shoulder → R_elbow → R_wrist |
| Left arm  | `[9, 13, 16, 18, 20]` | spine3 → L_collar → L_shoulder → L_elbow → L_wrist |

Face joint indices: `[2, 1, 17, 16]` (r_hip, l_hip, r_shoulder, l_shoulder).

---

## 4. What "Unitree G1 with 34-joint skeleton" Means in ProtoMotions

### 4.1 The two G1 asset files

| File | Bodies | Joints | Used for |
|---|---|---|---|
| `protomotions/data/assets/mjcf/g1_holo_compat.xml` | **33** | 46 (incl. tendons) | Training (default for `G1RobotConfig`) |
| `protomotions/data/assets/mjcf/g1_bm.xml`, `g1_bm_box_feet.xml`, `g1_bm_no_mesh_box_feet.xml` | 33 | 29 | Alternate training MJCFs |
| `protomotions/data/assets/urdf/for_retargeting/g1.urdf` | **43 links** (+ world) | 42 | PyRoki retargeting only |

**Actuated DOFs** (in all variants): **29**.

```
Legs   12  = 6 left + 6 right  (hip pitch/roll/yaw, knee, ankle pitch/roll)
Waist   3  = waist_yaw, waist_roll, waist_pitch
Arms   14  = 7 left + 7 right  (shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw)
Total  29
```

### 4.2 Why "34" matches your terminology

The training MJCF body list (output of `grep '<body name=' g1_holo_compat.xml`):

```
1  pelvis
2  head
3-8   left leg (hip_pitch_link → ankle_roll_link)              [6]
9-14  right leg                                                [6]
15-17 waist_yaw_link, waist_roll_link, torso_link              [3]
18-25 left arm (shoulder_pitch → wrist_yaw + rubber_hand)      [8]
26-33 right arm                                                [8]
────────────────────────────────────────────────────────────
33 rigid bodies
+ 1 free-floating root joint (the 7-DOF base in `nq`)
= 34 kinematic-tree entities
```

This is consistent with how some Unitree documentation counts the G1 skeleton (each link counts, plus the floating base). In ProtoMotions' robot config (`protomotions/robot_configs/g1.py`), this is `G1RobotConfig` pointing to `mjcf/g1_holo_compat.xml`.

### 4.3 The 15 retargeting keypoints

PyRoki's `get_humanoid_retarget_indices()` maps **15 human semantic joints** to G1 link names:

```python
("pelvis",          "pelvis_contour_link"),
("left_hip",        "left_hip_pitch_link"),    ("right_hip",   "right_hip_pitch_link"),
("left_knee",       "left_knee_link"),         ("right_knee",  "right_knee_link"),
("left_ankle",      "left_ankle_roll_link"),   ("right_ankle", "right_ankle_roll_link"),
("left_foot",       "left_foot_link"),         ("right_foot",  "right_foot_link"),
("left_shoulder",   "left_shoulder_pitch_link"), ("right_shoulder", "right_shoulder_pitch_link"),
("left_elbow",      "left_elbow_link"),        ("right_elbow", "right_elbow_link"),
("left_wrist",      "left_wrist_yaw_link"),    ("right_wrist", "right_wrist_yaw_link"),
```

These 15 keypoints + 3 auxiliary points (2 hand offsets + 1 pelvis-forward) make up the 18-keypoint format PyRoki actually solves against.

### 4.4 G1 size / scaling factors

PyRoki applies these scales to the input keypoints to bring them into G1's body proportions before solving:

```python
# G1 (smaller than human SMPL average)
simplified_keypoints_lower_body_local *= [0.9, 0.9, 0.85]
simplified_keypoints_upper_body_local *= [0.9, 0.9, 0.8]
```

### 4.5 Auxiliary hand offset (G1)

The "hand auxiliary point" for matching wrist orientation is `wrist + R_wrist @ [0, 0, 0.14]` (G1 specific, vs `[0, 0, 0.2]` for H1_2).

---

## 5. Two Retargeting Strategies

### 5.1 Strategy A (recommended) — ProtoMotions pipeline on raw AMASS

Don't run HumanML3D's notebooks at all (unless you need the text annotations). Run ProtoMotions's own `convert_amass_to_motionlib.py` on the raw AMASS `.npz` files. **Why**:

- ProtoMotions's pipeline uses SMPL **.pkl** (v1.1.0) models, not HumanML3D's SMPL+H **.npz** + DMPL.
- ProtoMotions targets 30 fps, while HumanML3D downsamples to 20 fps.
- ProtoMotions preserves global root rotation + translation; HumanML3D normalizes them out (you'd have to invert).
- PyRoki retargeting expects `(T, 18, 3)` positions **+** `(T, 18, 3, 3)` orientations + foot contacts — HumanML3D's `new_joints/*.npy` only has positions.
- The PyRoki convenience script `scripts/retarget_amass_to_robot.sh` is a one-line invocation.

You retarget all 11,000+ AMASS clips (more than HumanML3D's 14,616 mirrored, because HumanML3D filters and crops). If you want **exactly** HumanML3D's 14,616 clip subset, see Strategy B.

### 5.2 Strategy B — HumanML3D-curated subset

Use `index.csv` to build a YAML manifest that selects only AMASS files (and the right time windows) that HumanML3D included. Then run ProtoMotions's converter on this curated list. Text annotations from `texts.zip` can be matched back via the `new_name` column in `index.csv`.

- Pro: exactly 14,616 motions ↔ exactly 44,970 text annotations.
- Pro: cropped to motion-meaningful segments (per the HumanML3D paper's manual review).
- Con: you write a small adapter script to map `index.csv` → ProtoMotions YAML.
- HumanAct12 (`pose_data/humanact12.zip`, 1,191 clips) is **not** in AMASS. To include it, convert it via the same SMPL+H FK that HumanML3D's notebook does, then feed the output through ProtoMotions's AMASS converter (treating it as another sub-dataset). Or just skip HumanAct12 (~8% of HumanML3D).

---

## 6. Data Acquisition Prerequisites

### 6.1 AMASS sub-datasets (free with academic registration)

Register at https://amass.is.tue.mpg.de/ → Download → choose **SMPL+H G** archive for each dataset. HumanML3D uses these 17 subsets (unzip each into its bracketed folder name):

| HumanML3D label | AMASS folder | Notes |
|---|---|---|
| ACCAD | ACCAD | |
| HDM05 | MPI_HDM05 | skip first 3 s per clip |
| TCDHands | TCD_handMocap | |
| SFU | SFU | |
| BMLmovi | BMLmovi | |
| CMU | CMU | |
| Mosh | MPI_mosh | |
| EKUT | EKUT | |
| KIT | KIT | |
| Eyes_Japan_Dataset | Eyes_Japan_Dataset | skip first 3 s |
| BMLhandball | BMLhandball | |
| Transitions | Transitions_mocap | skip first 0.5 s |
| PosePrior | MPI_Limits | skip first 1 s |
| HumanEva | HumanEva | |
| SSM | SSM_synced | |
| DFaust | DFaust_67 | |
| TotalCapture | TotalCapture | skip first 1 s |
| BMLrub | BioMotionLab_NTroje | |

Approximate total size after unzip: **40–50 GB**.

### 6.2 Body models (free with academic registration)

**For HumanML3D's stage-1 notebook** (SMPL+H + DMPL, `.npz`):

- https://mano.is.tue.mpg.de/download.php → "Extended SMPL+H model used in AMASS project"
- https://smpl.is.tue.mpg.de/download.php → "DMPLs compatible with SMPL"

Unzip into `HumanML3D/body_models/`:

```
body_models/
├── smplh/
│   ├── male/model.npz
│   ├── female/model.npz
│   └── neutral/model.npz
└── dmpls/
    ├── male/model.npz
    ├── female/model.npz
    └── neutral/model.npz
```

**For ProtoMotions** (SMPL `.pkl` v1.1.0):

- https://smpl.is.tue.mpg.de/download.php → SMPL v1.1.0

Rename and place under `ProtoMotions/data/smpl/`:

```
ProtoMotions/data/smpl/
├── SMPL_NEUTRAL.pkl   <- basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl
├── SMPL_MALE.pkl      <- basicmodel_m_lbs_10_207_0_v1.1.0.pkl
└── SMPL_FEMALE.pkl    <- basicmodel_f_lbs_10_207_0_v1.1.0.pkl
```

### 6.3 Python environments (two separate ones)

**Env 1 — `protomotions`** (Python 3.10):

```bash
conda create -n protomotions python=3.10
conda activate protomotions
cd D:/HumanML3d/ProtoMotions
pip install -e .
# Pick ONE simulator's requirements:
pip install -r requirements_isaacgym.txt   # for IsaacGym training
# or
pip install -r requirements_newton.txt     # for Newton training (+ pip install "newton[examples]" first)
# or for CPU-only validation:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements_mujoco.txt
```

**Env 2 — `pyroki`** (Python 3.10, separate from ProtoMotions because of JAX/CUDA pinning):

```bash
conda create -n pyroki python=3.10
conda activate pyroki
git clone https://github.com/chungmin99/pyroki.git
cd pyroki
pip install -e .
# PyRoki pulls in jax, jaxlie, jaxls automatically.
```

**Env 3 — `torch_render`** (only if you want to run HumanML3D's stage-1 notebook, Python 3.7):

```bash
cd D:/HumanML3d/HumanML3D
conda env create -f environment.yaml
conda activate torch_render
```

Strategy A (recommended) skips Env 3 entirely.

---

## 7. End-to-End Pipeline — Recommended (Strategy A)

The full path. Roughly 4–8 hours wall-clock on a single A100 for the heavy AMASS retarget, depending on `--skip-freq`.

### Step 0 — Lay out directories

```
D:/HumanML3d/
├── amass_data/                    # raw AMASS (you populate this from step 6.1)
│   ├── ACCAD/...
│   ├── KIT/...
│   └── BioMotionLab_NTroje/...
├── ProtoMotions/                  # already cloned
│   └── data/smpl/                 # SMPL .pkl files (from step 6.2)
├── HumanML3D/                     # already cloned (only needed for Strategy B)
└── retargeting_workspace/
    ├── amass_proto/               # output of step 2
    ├── amass_smpl_train.pt        # packaged MotionLib (after step 3)
    ├── keypoints-for-retarget/    # output of step 4
    ├── pyroki-retargeted-g1/      # output of step 5
    ├── contacts/                  # output of step 6
    ├── proto-g1/                  # output of step 7
    └── proto-g1.pt                # FINAL — feed into training
```

### Step 1 — Sanity-check the YAML manifests

ProtoMotions ships pre-built manifests in `data/yaml_files/`. Inspect:

```bash
ls D:/HumanML3d/ProtoMotions/data/yaml_files/amass_smpl_*.yaml
# amass_smpl_train.yaml, amass_smpl_test.yaml, amass_smpl_validation.yaml
```

Each YAML lists `<dataset>/<subject>/<clip>_poses.motion` entries with FPS, sampling weight, and `sub_motions.timings.{start,end}` to exclude T-pose calibration frames. You can use these as-is or copy and edit.

### Step 2 — Convert AMASS to ProtoMotions `.motion` files

```bash
conda activate protomotions
cd D:/HumanML3d/ProtoMotions

python data/scripts/convert_amass_to_proto.py \
    /path/to/amass_data \
    --humanoid-type smpl \
    --output-fps 30 \
    --motion-config data/yaml_files/amass_smpl_train.yaml \
    --motion-config data/yaml_files/amass_smpl_test.yaml \
    --motion-config data/yaml_files/amass_smpl_validation.yaml \
    --force-remake
```

Under the hood (per `docs/source/getting_started/amass_preparation.rst`):
- Loads each `.npz` (`poses` axis-angle, `trans` translations, `mocap_framerate`).
- Downsamples (largest divisor of source FPS that is ≥ 30).
- Runs SMPL FK via the SMPL `.pkl` model → world body positions/rotations.
- Computes lin/ang velocities by finite differences (with multi-horizon noise filtering, see `protomotions/components/pose_lib.py`).
- Detects ground contact (velocity < 0.15 m/s **and** height < 0.1 m).
- Vertically shifts so the lowest joint matches a toe offset (0.015 m for SMPL, 0.017 m for SMPL-X).
- Saves `<motion_root>/<dataset>/<subject>/<clip>_poses.motion` (one per source `.npz`).

### Step 3 — Package into a MotionLib `.pt`

```bash
python protomotions/components/motion_lib.py \
    --motion-path data/yaml_files/amass_smpl_train.yaml \
    --output-file D:/HumanML3d/retargeting_workspace/amass_smpl_train.pt \
    --device cpu
```

This concatenates per-clip tensors into a single `.pt` indexed by `length_starts` (see [components/motion_lib.py](ProtoMotions/protomotions/components/motion_lib.py)).

### Step 4 — Extract retargeting keypoints

```bash
python data/scripts/extract_retargeting_input_keypoints_from_packaged_motionlib.py \
    D:/HumanML3d/retargeting_workspace/amass_smpl_train.pt \
    --output-path D:/HumanML3d/retargeting_workspace/keypoints-for-retarget/ \
    --skeleton-format smpl \
    --start-idx 0 \
    --skip-freq 1
```

For each motion, writes a `.npy` with:

```python
{
  "positions":       (T, 18, 3),       # 15 base + 3 aux keypoints, world frame
  "orientations":    (T, 18, 3, 3),    # rotation matrices
  "left_foot_contacts":  (T, 2),       # [ankle, toebase]
  "right_foot_contacts": (T, 2),
}
```

The 18 keypoints (in order) are: pelvis, left/right hip, left/right knee, left/right ankle, left/right foot (toebase), left/right shoulder, left/right elbow, left/right wrist + 3 aux (left/right hand_aux, pelvis_aux).

Tip: for a smoke test, use `--skip-freq 50` to process ~1/50th of the dataset first.

### Step 5 — Run PyRoki retargeting (G1)

```bash
conda activate pyroki
cd D:/HumanML3d/ProtoMotions   # PyRoki script lives in the ProtoMotions repo

python pyroki/batch_retarget_to_g1_from_keypoints.py \
    --keypoints-folder-path D:/HumanML3d/retargeting_workspace/keypoints-for-retarget/ \
    --output-dir D:/HumanML3d/retargeting_workspace/pyroki-retargeted-g1/ \
    --source-type smpl \
    --subsample-factor 1 \
    --no-visualize \
    --skip-existing
```

What this does (`pyroki/batch_retarget_to_g1_from_keypoints.py`):

- Loads G1 URDF from `protomotions/data/assets/urdf/for_retargeting/g1.urdf` (43 links, the version with sensor mounts).
- For each motion:
  - Pads/trims to 450 frames (15 s @ 30 fps) for JAX batch compilation.
  - Scales lower-body keypoints by `[0.9, 0.9, 0.85]`, upper-body by `[0.9, 0.9, 0.8]` (G1 is smaller than human).
  - Sets up JAX least-squares optimization (variables: 29 joint angles per frame + SE3 root pose per frame + 15×15 scale matrix).
  - **Cost terms** with default weights:
    - `local_alignment`        (1.0) — relative joint positions + bone directions
    - `global_alignment`       (4.0) — absolute keypoint positions in world
    - `root_smoothness`        (1.0) — SE3 root motion smoothness
    - `joint_smoothness`       (4.0) — joint angle smoothness over time
    - `limit_cost`             — soft joint limits from URDF
    - `joint_rest_penalty`     (1.0) — keep waist/wrists near 0
    - `joint_vel_limit`        (50.0) — max 20 rad/s
    - `foot_contact`           (30.0) — when contact: penalize foot velocity + Z height drift
    - `foot_tilt`              (1.0) — keep foot flat in contact (R[2,2] ≈ 1)
    - `self_collision`         (0.0) — disabled by default
  - Runs `jaxls.LeastSquaresProblem.solve()` (max 800 iterations).
- Writes `<clip>_retargeted.npz`:
  ```python
  {
    "base_frame_pos":  (T, 3),     # root world position (meters)
    "base_frame_wxyz": (T, 4),     # root quaternion (wxyz)
    "joint_angles":    (T, 29),    # actuated DOFs (G1 ordering)
  }
  ```

### Step 6 — Extract contact labels (separate pass)

```bash
# Still in pyroki env
python pyroki/batch_retarget_to_g1_from_keypoints.py \
    --keypoints-folder-path D:/HumanML3d/retargeting_workspace/keypoints-for-retarget/ \
    --source-type smpl \
    --subsample-factor 1 \
    --save-contacts-only \
    --contacts-dir D:/HumanML3d/retargeting_workspace/contacts/ \
    --skip-existing
```

Why a second pass? Contact labels are taken from the **source SMPL motion**, not re-computed on the retargeted robot — the retargeted motion has imperfect contacts because the IK is soft on the foot. The script reuses the same loader (with the 5-frame contact-smoothing crossfade) and writes per-clip `<clip>_contacts.npz`.

### Step 7 — Convert PyRoki output → ProtoMotions `.motion`

```bash
conda activate protomotions
python data/scripts/convert_pyroki_retargeted_robot_motions_to_proto.py \
    --retargeted-motion-dir D:/HumanML3d/retargeting_workspace/pyroki-retargeted-g1/ \
    --output-dir D:/HumanML3d/retargeting_workspace/proto-g1/ \
    --robot-type g1 \
    --contact-labels-dir D:/HumanML3d/retargeting_workspace/contacts/ \
    --apply-motion-filter \
    --force-remake
```

What this does:
1. Loads G1 MJCF kinematic info (`mjcf/g1_bm_box_feet.xml`) via `pose_lib.extract_kinematic_info()`.
2. For each `.npz`: builds qpos `[root_pos(3), root_wxyz(4), joint_angles(29)]`.
3. Runs FK + multi-horizon velocity computation → full `RobotState`.
4. Applies `fix_height_per_frame` + global `fix_height(offset=0.04)` so feet don't penetrate the ground.
5. Loads matching `_contacts.npz`, downsamples to output FPS, sets `motion.rigid_body_contacts[:, left_foot_idx] = lcontact > 0.5` and similarly for right.
6. (Optional) `--apply-motion-filter` skips motions failing height/velocity sanity checks.
7. Saves `<clip>.motion`.

### Step 8 — Package the G1 MotionLib

```bash
python protomotions/components/motion_lib.py \
    --motion-path D:/HumanML3d/retargeting_workspace/proto-g1/ \
    --output-file D:/HumanML3d/retargeting_workspace/proto-g1.pt \
    --device cpu
```

Now you have **`proto-g1.pt`** — the G1-retargeted MotionLib, ready for training.

### Step 9 — One-shot alternative

All of Steps 4–8 can be run with a single convenience script:

```bash
./scripts/retarget_amass_to_robot.sh \
    /path/to/conda/envs/protomotions/bin/python \
    /path/to/conda/envs/pyroki/bin/python \
    D:/HumanML3d/retargeting_workspace/amass_smpl_train.pt \
    g1 \
    1   # skip_freq (use 50 for fast test)
```

On Windows replace the `.sh` with a manual sequence of the python calls above, or use Git Bash / WSL.

---

## 8. End-to-End Pipeline — HumanML3D-Filtered (Strategy B)

If you specifically want only the 14,616 HumanML3D-curated clips (plus their text annotations), build a custom YAML manifest from `index.csv`:

### Step 1 — Run AMASS download + HumanML3D's notebook 1

Follow [Section 3.1](#31-raw_pose_processingipynb--stage-1-amass--joints). This gives you `./joints/XXXXXX.npy` files (14,616 + 14,616 mirrored).

### Step 2 — Map `index.csv` → ProtoMotions YAML

Write a short Python script (place under `D:/HumanML3d/HumanML3D_curation.py`):

```python
import csv, os, yaml
from pathlib import Path

HUMANML3D = Path("D:/HumanML3d/HumanML3D")
AMASS_ROOT = Path("D:/HumanML3d/amass_data")
FPS = 20  # HumanML3D's chosen output FPS — but we'll bump to 30 for ProtoMotions

motions = []
with open(HUMANML3D / "index.csv") as f:
    for row in csv.DictReader(f):
        sp = row["source_path"]
        if "humanact12" in sp:
            continue   # HumanAct12 is not in AMASS — skip or handle separately
        # source_path format: ./pose_data/<DATASET>/<subject>/<clip>_poses.npy
        rel = sp.removeprefix("./pose_data/").replace("_poses.npy", "_poses.motion")
        start_s = int(row["start_frame"]) / FPS
        end_frame = int(row["end_frame"])
        end_s = None if end_frame == -1 else end_frame / FPS
        motions.append({
            "file": rel,
            "fps": 30.0,                           # for ProtoMotions output_fps
            "weight": 1.0,
            "sub_motions": [{
                "timings": {"start": start_s,
                            "end": end_s if end_s is not None else 999.0},
            }],
        })

with open("amass_humanml3d_curated.yaml", "w") as f:
    yaml.safe_dump({"motions": motions}, f)
print(f"Wrote {len(motions)} entries.")
```

This produces `amass_humanml3d_curated.yaml` — a ProtoMotions manifest that selects exactly the AMASS clips HumanML3D used (minus HumanAct12, which is not in AMASS).

### Step 3 — Run ProtoMotions's pipeline on the curated manifest

Exactly as Strategy A from Step 2 onward, but feed the curated YAML:

```bash
python data/scripts/convert_amass_to_proto.py /path/to/amass_data \
    --humanoid-type smpl --output-fps 30 \
    --motion-config amass_humanml3d_curated.yaml --force-remake

python protomotions/components/motion_lib.py \
    --motion-path amass_humanml3d_curated.yaml \
    --output-file amass_humanml3d_curated.pt --device cpu

./scripts/retarget_amass_to_robot.sh \
    <proto_python> <pyroki_python> amass_humanml3d_curated.pt g1 1
```

### Step 4 (optional) — Carry through text annotations

`texts.zip` in the HumanML3D repo contains `XXXXXX.txt` files matching `new_name` in `index.csv`. If you need text annotations alongside the retargeted motions:

1. Unzip `texts.zip` → `texts/`.
2. After Step 3, you have `proto-g1/<dataset>/<subject>/<clip>_poses.motion` files. Build a sidecar mapping from `index.csv` (`source_path` ↔ `new_name`) so you can look up the text for any given retargeted motion.
3. Store the mapping as JSON next to your `.motion` files, or use it in your data-loader.

### Strategy B caveats

- **HumanAct12 (≈1,191 of 14,616 ≈ 8%) is not in AMASS.** It uses its own action-to-motion format. To include it: extract poses via the SMPL+H model (the HumanML3D notebook does this), then write a small adapter to produce the same `.npz` layout AMASS uses (`poses`, `trans`, `betas`, `mocap_framerate`), and add them to the AMASS root dir so step 2 picks them up. Or just skip them.
- **HumanML3D's frame trims** (e.g., "Eyes_Japan: skip first 3 s") are baked into the `start_frame` column of `index.csv`. The above script preserves them.
- **Mirroring**: HumanML3D doubles the dataset via left/right mirror. ProtoMotions handles this differently — domain randomization + random-direction terrain. You probably don't need the HumanML3D-style explicit mirrors.
- **FPS mismatch**: HumanML3D uses 20 fps; ProtoMotions trainers default to 30 fps. Use `--output-fps 30` so the rest of the pipeline aligns.

---

## 9. Validation, Debugging, and Quality Checks

### 9.1 Visualize the source motion library (SMPL)

```bash
python examples/motion_libs_visualizer.py \
    --motion_files D:/HumanML3d/retargeting_workspace/amass_smpl_train.pt \
    --robot smpl \
    --simulator isaaclab
```

Controls: **R** next motion, **1/2** speed up/down, **3/4** smoothness threshold for jitter highlighting.

### 9.2 Visualize the retargeted G1 motion library

```bash
python examples/motion_libs_visualizer.py \
    --motion_files D:/HumanML3d/retargeting_workspace/proto-g1.pt \
    --robot g1 \
    --simulator isaacgym
```

Or side-by-side comparison:

```bash
python examples/motion_libs_visualizer.py \
    --motion_files amass_smpl_train.pt proto-g1.pt \
    --robot g1 \
    --simulator isaacgym
```

### 9.3 Spot-check forward kinematics on a single motion

ProtoMotions ships a Newton FK test you can adapt:

```bash
pytest protomotions/tests/test_newton_simulator_fk.py \
    -k g1 --motion-file proto-g1.pt
```

### 9.4 Common quality red flags

- **Foot sliding**: tune up `foot_contact` weight in PyRoki's `RetargetingWeights` or apply `--apply-motion-filter` more aggressively in Step 7.
- **Sudden flips / 180° rotations**: usually means a bad contact label. Re-run Step 6 with `--save-contacts-only` and check the smoothing window in `load_motion_data()`.
- **Robot height off**: the `fix_height(offset=0.04)` in Step 7 assumes G1; if you see persistent floating/clipping, edit the offset in `convert_pyroki_retargeted_robot_motions_to_proto.py`.
- **Joint limits violated**: increase `limit_cost` weight in PyRoki (it's already a hard-ish soft penalty).
- **Out-of-distribution motions** (handstands, gymnastics): may simply not map well to G1's 29 DOFs. Filter them out or accept some failures.

---

## 10. Training G1 on the Retargeted Motions

Once you have `proto-g1.pt`, the standard ProtoMotions training command (see [PROTOMOTIONS_CODEBASE_REFERENCE.md §5](PROTOMOTIONS_CODEBASE_REFERENCE.md)) for motion-tracking:

```bash
conda activate protomotions
python protomotions/train_agent.py \
    --robot-name g1 \
    --simulator isaacgym \
    --experiment-path examples/experiments/mimic/mlp.py \
    --experiment-name g1_humanml3d_tracker \
    --motion-file D:/HumanML3d/retargeting_workspace/proto-g1.pt \
    --num-envs 4096 \
    --batch-size 16384
```

For BeyondMimic-style settings (production-grade, used for real G1 deployment), use the reference experiment config:

```bash
python protomotions/train_agent.py \
    --robot-name g1 \
    --simulator isaaclab \
    --experiment-path data/pretrained_models/motion_tracker/g1-bones-deploy/experiment_config.py \
    --experiment-name g1_humanml3d_bm \
    --motion-file D:/HumanML3d/retargeting_workspace/proto-g1.pt \
    --num-envs 4096 \
    --batch-size 16384
```

Sim-to-sim test in Newton/MuJoCo (CPU-only) after training:

```bash
python protomotions/inference_agent.py \
    --checkpoint results/g1_humanml3d_bm/last.ckpt \
    --motion-file D:/HumanML3d/retargeting_workspace/proto-g1.pt \
    --simulator newton --num-envs 16
```

ONNX export for real-robot deployment:

```bash
python deployment/export_bm_tracker_onnx.py \
    --checkpoint results/g1_humanml3d_bm/last.ckpt
```

(See [g1_deployment tutorial](ProtoMotions/docs/source/tutorials/workflows/g1_deployment.rst) for the full RoboJuDo deployment pipeline.)

---

## 11. Disk-Space, Compute, and Time Budget

| Step | Disk | GPU? | Wall-clock (rough) |
|---|---|---|---|
| AMASS download | 40–50 GB | — | hours of network |
| SMPL/SMPL+H/DMPL models | <500 MB | — | minutes |
| HumanML3D Stage 1 (Strategy B only) | +30 GB (joints/) | CUDA recommended | 2–6 h on one GPU |
| ProtoMotions AMASS conversion | +20 GB (.motion files) | — | ~30–90 min on CPU |
| MotionLib packaging | ≈5 GB per `.pt` | — | ~10 min |
| Keypoint extraction | +5 GB | — | ~30 min |
| **PyRoki retargeting (G1)** | +5 GB | GPU helps | **3–6 h on A100; longer on CPU** |
| Contact extraction | +500 MB | GPU helps | ~15 min |
| Conversion to .motion | +20 GB | — | ~30 min |
| Final MotionLib pack | ≈5 GB | — | ~10 min |
| G1 mimic training (mlp.py, 4096 envs) | — | 1–4 GPUs | ~12 h to 99% success |

If GPU-poor, use `--skip-freq 50` for the PyRoki step to make the first retarget complete in ~30 min while you iterate on the rest.

---

## 12. Gotchas and Common Pitfalls

1. **Two Python envs are required.** ProtoMotions and PyRoki pin different JAX/CUDA versions; mixing them fails at import. Keep them in separate conda envs and pass both Python paths to `retarget_amass_to_robot.sh`.

2. **SMPL `.pkl` ≠ SMPL+H `.npz`.** ProtoMotions wants SMPL v1.1.0 `.pkl`; HumanML3D wants SMPL+H + DMPL `.npz`. If you mix them up, the AMASS converter will silently produce wrong joint positions.

3. **Coordinate frame conventions.** AMASS axis-angles are in SMPL's native frame. HumanML3D applies a `[[1,0,0],[0,0,1],[0,1,0]]` axis swap; ProtoMotions applies `Rx(-π/2) @ Ry(-π/2)`. Don't try to feed HumanML3D-frame joint positions back into ProtoMotions's converter — you'll get rotated motions.

4. **HumanML3D's mirrors are duplicates.** Don't double-count them when budgeting the retarget — most of the time you only need one direction; ProtoMotions handles left/right symmetry through training augmentation.

5. **Free-floating root is NOT a DOF.** The G1 has 29 actuated DOFs + 6-DOF floating root in `nq`, so `nq = 7 (pos+quat) + 29 = 36` and `nv = 6 + 29 = 35`. "34" in the model description refers to the kinematic-tree entity count (33 rigid bodies + 1 root joint).

6. **PyRoki's URDF (43 links) ≠ training MJCF (33 bodies).** The retargeting URDF includes IMU/d435/mid360 sensor links that the training MJCF strips. The shared meaningful subset is the 33 actuated/visual bodies. Don't try to use the URDF for training — joint indexing won't match `g1.py`'s `common_naming_to_robot_body_names`.

7. **Contact labels come from the SMPL source, not the retargeted G1.** This is by design (Step 6 re-runs the script with `--save-contacts-only`). Don't compute contacts from the G1 trajectory — they'll be noisier.

8. **`--apply-motion-filter` silently drops motions** that fail height/velocity sanity checks. If your output has fewer `.motion` files than input `.npz` files, that's why. Check the script's logs.

9. **PyRoki batches to 450 frames (15 s @ 30 fps).** Motions longer than 15 s get **trimmed**. If you need longer clips, edit `target_raw_frames` in `batch_retarget_to_g1_from_keypoints.py`.

10. **The `paramUtil.py::t2m_kinematic_chain` lists joints 0–21 in non-trivial order.** The chain `[0, 2, 5, 8, 11]` is the *right* leg (because SMPL puts index 2 = right_hip, 1 = left_hip). When debugging, double-check left vs. right.

11. **HumanML3D notebook 1 requires Python 3.7 and Matplotlib 3.3.4** (per the README — `pip install matplotlib==3.3.4` to avoid the "small deviation" issue documented in [#21](https://github.com/EricGuo5513/HumanML3D/issues/21)).

12. **Windows path separators.** All the scripts use forward slashes and bash. On native Windows, run via Git Bash or WSL, or convert the convenience scripts into PowerShell.

---

## 13. Appendix A — Index of Source Datasets in HumanML3D

From `index.csv` (sampled — these are the 17 AMASS subsets + HumanAct12 referenced by the 14,616 rows):

| AMASS folder | HumanML3D rows | License | Notes |
|---|---|---|---|
| KIT | ~3,500 | KIT (free, register) | most-represented |
| CMU | ~2,000 | CMU (free) | classic mocap |
| BMLmovi | ~1,500 | free with registration | full-body, 90 subjects |
| BioMotionLab_NTroje (BMLrub) | ~1,300 | free with registration | walking, locomotion |
| MPI_HDM05 | ~700 | HDM05 (free for academic) | dance, sports |
| Eyes_Japan_Dataset | ~700 | free | varied actions |
| MPI_mosh | ~500 | MPI (free) | morphed motions |
| TotalCapture | ~400 | free | multi-modal mocap |
| MPI_Limits (PosePrior) | ~300 | free | extreme poses |
| TCD_handMocap | ~200 | TCD (free) | hand-focused (not used) |
| SFU | ~200 | free | varied |
| DFaust_67 | ~150 | free | scanning subjects |
| EKUT | ~150 | free | locomotion |
| HumanEva | ~100 | free | classic eval set |
| BMLhandball | ~100 | free | handball-specific |
| Transitions_mocap | ~100 | free | transitions |
| SSM_synced | ~50 | free | synchronized markers |
| ACCAD | ~50 | free | small set |
| **humanact12 (NOT AMASS)** | ~1,200 | from HumanAct12 paper | shipped as `pose_data/humanact12.zip` |

Counts are approximate; exact counts can be derived from `index.csv` with a one-liner:
```bash
awk -F, 'NR>1 {n=split($1,a,"/"); print a[3]}' index.csv | sort | uniq -c | sort -rn
```

---

## 14. Appendix B — Format Cheat Sheet

### 14.1 Raw AMASS `.npz`

```python
{
  "poses":           (T_src, 156),   # axis-angle, root(3) + body(63) + lhand(45) + rhand(45)
  "trans":           (T_src, 3),     # root translation
  "betas":           (16,),          # shape
  "gender":          b"male" | b"female" | b"neutral",
  "mocap_framerate": float,          # e.g. 120.0
  "dmpls":           (T_src, 8),     # optional, soft-tissue DMPL params
}
```

### 14.2 HumanML3D `./joints/XXXXXX.npy`

`np.ndarray` shape `(T, 22, 3)`, float32. World-frame joint positions at 20 fps, after axis swap and dataset-specific frame trims. Includes mirrored variants `MXXXXXX.npy`.

### 14.3 HumanML3D `./HumanML3D/new_joints/XXXXXX.npy`

`np.ndarray` shape `(T, 22, 3)`, float32. **Root XZ-centered, face-direction-aligned to +Z, floor at y=0.** Lost: global root rotation, global root translation in XZ.

### 14.4 HumanML3D `./HumanML3D/new_joint_vecs/XXXXXX.npy`

`np.ndarray` shape `(T-1, 263)`, float32. Channel layout:

```
[0:1]    r_angular_velocity_Y    (root yaw rate via arcsin(qz))
[1:3]    root_linear_velocity_XZ
[3:4]    root_y_height
[4:67]   ric_data                 (21 local joint positions × 3)
[67:193] rot_data                 (21 joint 6D rotations × 6)
[193:259] local_vel               (22 joint vels × 3 in local frame)
[259:263] foot_contacts           (L_ankle, L_toe, R_ankle, R_toe)
```

### 14.5 ProtoMotions keypoint `.npy` (PyRoki input)

`np.savez`-style dictionary with:

```python
{
  "positions":           (T, 18, 3),       # 15 base + 3 aux, world frame, meters
  "orientations":        (T, 18, 3, 3),    # rotation matrices
  "left_foot_contacts":  (T, 2),           # [ankle, toebase], in [0, 1]
  "right_foot_contacts": (T, 2),
}
```

Keypoint order: pelvis, left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle, left_foot, right_foot, left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist, left_hand_aux, right_hand_aux, pelvis_aux.

### 14.6 PyRoki retargeted `.npz`

```python
{
  "base_frame_pos":  (T, 3),        # world position
  "base_frame_wxyz": (T, 4),        # quaternion wxyz
  "joint_angles":    (T, 29),       # G1 DOFs (URDF joint order)
}
```

### 14.7 ProtoMotions `.motion` (after Step 7)

Pickled `RobotState`-dict from `pose_lib.fk_from_transforms_with_velocities`:

```python
{
  "dof_pos":                (T, 29),
  "dof_vel":                (T, 29),
  "rigid_body_pos":         (T, 33, 3),
  "rigid_body_rot":         (T, 33, 4),    # xyzw
  "rigid_body_vel":         (T, 33, 3),
  "rigid_body_ang_vel":     (T, 33, 3),
  "rigid_body_contacts":    (T, 33),       # bool
  "fps":                    float,         # 30.0
}
```

### 14.8 ProtoMotions MotionLib `.pt` (after Step 8)

Pickled dict containing:

```python
{
  # Concatenated frame tensors:
  "gts":           (sum_T, 33, 3),      # global body positions
  "grs":           (sum_T, 33, 4),      # global body rotations (xyzw)
  "gvs":           (sum_T, 33, 3),      # global body linear velocities
  "gavs":          (sum_T, 33, 3),      # global body angular velocities
  "dps":           (sum_T, 29),         # DOF positions
  "dvs":           (sum_T, 29),         # DOF velocities
  "contacts":      (sum_T, 33),         # optional, bool
  # Per-motion metadata:
  "length_starts":     (N_motions,),    # start frame index of each motion
  "motion_lengths":    (N_motions,),    # length in seconds
  "motion_num_frames": (N_motions,),
  "motion_dt":         (N_motions,),    # 1/fps
  "motion_weights":    (N_motions,),    # sampling weights
  "motion_files":      [str, ...],      # source file names
}
```

This is what gets passed as `--motion-file` to `train_agent.py`. Loadable in O(1) via `length_starts` indexing.

---

*End of pipeline reference.*
