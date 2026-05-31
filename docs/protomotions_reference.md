# ProtoMotions 3 — Codebase Deep Dive

> A thorough, file-by-file, function-by-function reference for the NVlabs/ProtoMotions repository (`https://github.com/NVlabs/ProtoMotions`). Synthesized after a full pull at commit `a87d2f3` (May 2026).

---

## Table of Contents

1. [What ProtoMotions Is](#1-what-protomotions-is)
2. [Repository Layout](#2-repository-layout)
3. [Architecture at a Glance](#3-architecture-at-a-glance)
4. [The Configuration System](#4-the-configuration-system)
5. [Entry Points (`train_agent.py`, `inference_agent.py`, `train_slurm.py`)](#5-entry-points)
6. [`protomotions/envs/` — The MDP Layer](#6-protomotionsenvs--the-mdp-layer)
7. [`protomotions/simulator/` — Multi-Simulator Abstraction](#7-protomotionssimulator--multi-simulator-abstraction)
8. [`protomotions/agents/` — RL Algorithms](#8-protomotionsagents--rl-algorithms)
9. [`protomotions/components/` — Motion, Pose, Scene, Terrain](#9-protomotionscomponents--motion-pose-scene-terrain)
10. [`protomotions/robot_configs/` — Per-Robot Configuration](#10-protomotionsrobot_configs--per-robot-configuration)
11. [`protomotions/utils/` — Helpers](#11-protomotionsutils--helpers)
12. [`protomotions/tests/`](#12-protomotionstests)
13. [`examples/` — Experiments, Tutorials, Benchmarks](#13-examples--experiments-tutorials-benchmarks)
14. [`deployment/` — ONNX Export & Real-Robot Bridge](#14-deployment--onnx-export--real-robot-bridge)
15. [`scripts/` — CLI Helpers](#15-scripts--cli-helpers)
16. [`pyroki/` — Retargeting via PyRoki](#16-pyroki--retargeting-via-pyroki)
17. [`usd_convert/` — MJCF→USD Conversion](#17-usd_convert--mjcfusd-conversion)
18. [`docs/` and `data/`](#18-docs-and-data)
19. [Build, Dependencies & Containers](#19-build-dependencies--containers)
20. [Gotchas, Conventions, and Tips](#20-gotchas-conventions-and-tips)

---

## 1. What ProtoMotions Is

ProtoMotions 3 is a **GPU-accelerated simulation and RL framework** for training physically simulated digital humans and humanoid robots (SMPL, SMPLX, AMP humanoid, Unitree H1.2, Unitree G1, SOMA, RigV1, etc.).

It is unique in the space because it is **simulator-agnostic** and **algorithm-agnostic**: the same experiment file can be trained or evaluated under IsaacGym, IsaacLab, Newton, Genesis, or MuJoCo, with PPO, AMP, ASE, Mimic/ADD, or MaskedMimic.

The framework's three core abstractions are:

- **`Simulator`** — an abstract physics-engine wrapper (~17 abstract methods) that hides per-engine quirks (quaternion conventions, body/DOF ordering, friction modes, etc.) behind unified state dataclasses.
- **`MdpComponent` + `FieldPath`** — a declarative configuration of observations, rewards, and terminations as **pure tensor functions** bound to **typed context paths**. The output is ONNX-exportable and `torch.compile`-friendly.
- **`BaseAgent`** — a generic on-policy RL training loop using PyTorch Lightning Fabric for distributed training; specialized into PPO → AMP → ASE / Mimic-ADD / MaskedMimic.

License: **Apache-2.0**. Python ≥ 3.8. Sibling/lightweight project: [MimicKit](https://github.com/xbpeng/MimicKit).

Authors: Chen Tessler, Yifeng Jiang, Xue Bin Peng, Erwin Coumans, Yi Shi, Haotian Zhang, Davis Rempe, Gal Chechik, Sanja Fidler.

---

## 2. Repository Layout

```
ProtoMotions/
├── CLAUDE.md                  # Internal guide for Claude / contributors
├── README.md                  # Public README with task gallery
├── CONTRIBUTING.md            # DCO sign-off requirement
├── LICENSE.md                 # Apache-2.0
├── setup.py                   # `protomotions` package (v3.1, py>=3.8)
├── Dockerfile.isaacgym        # CUDA 12.1 + IsaacGym Preview 4 image
├── Dockerfile.isaaclab        # Based on nvcr.io/nvidian/isaac-lab 2.3.0
├── Dockerfile.newton          # CUDA 12.4 + Newton 1.0 (PyPI) image
├── requirements_*.txt         # Per-backend pinned deps
├── .pre-commit-config.yaml    # Ruff + license-header + typos hooks
├── .gitattributes             # Git LFS rules for *.npy/*.pt/*.ckpt/...
│
├── protomotions/              # Python package — see §6–12
│   ├── train_agent.py
│   ├── inference_agent.py
│   ├── train_slurm.py
│   ├── agents/                # RL algorithms (BaseAgent → PPO → AMP → ASE / Mimic / MaskedMimic)
│   ├── envs/                  # MDP layer (BaseEnv, MdpComponent, obs, rewards, terminations, control)
│   ├── simulator/             # Multi-engine abstraction (base + IsaacGym/Lab/Newton/Genesis/MuJoCo)
│   ├── components/            # MotionLib, PoseLib, SceneLib, terrains
│   ├── robot_configs/         # Per-robot configs (SMPL, G1, H1.2, SOMA23, …)
│   ├── utils/                 # Shared helpers
│   ├── tests/                 # Pytest suite
│   └── data/assets/           # MJCF / URDF / mesh / USD / checkerboard textures
│
├── data/                      # Motion data, pretrained models, conversion scripts, YAML manifests
├── examples/                  # Experiment files, tutorials, benchmarks, visualizers
├── deployment/                # ONNX export + standalone MuJoCo deploy test
├── scripts/                   # Bash + Python helpers (retargeting, search, joint monkey, …)
├── pyroki/                    # JAX-based retargeter (replaces older Mink dep)
├── usd_convert/               # MJCF → USDA pipeline for IsaacLab
└── docs/                      # Sphinx docs (built to https://protomotions.github.io)
```

---

## 3. Architecture at a Glance

Three layered loops, with strict dependencies running top-down:

```
┌────────────────────────────────────────────────────────────┐
│  Agent (PPO / AMP / ASE / Mimic / MaskedMimic)             │
│    fit():  rollout → normalize → advantages → minibatch    │
└──────────────────────────────┬─────────────────────────────┘
                               │ env.step(action) / env.reset()
┌──────────────────────────────▼─────────────────────────────┐
│  BaseEnv (the MDP layer)                                   │
│    process_action → simulator.step → post_physics_step     │
│     → context.build → obs/rew/term components → reset      │
│   Components are MdpComponents bound by FieldPath to       │
│   EnvContext (CurrentStateView, HistoricalView, MimicCtx…) │
└──────────────────────────────┬─────────────────────────────┘
                               │ apply_control / get_state
┌──────────────────────────────▼─────────────────────────────┐
│  Simulator (IsaacGym / IsaacLab / Newton / Genesis / MuJoCo│
│    State exchanged via RobotState/ObjectState dataclasses  │
│    with StateConversion {COMMON ↔ SIMULATOR} ordering.     │
└────────────────────────────────────────────────────────────┘
```

### Agent hierarchy

```
BaseAgent (Lightning Fabric, ExperienceBuffer, fit loop)
├── PPO          actor-critic, GAE, clipped surrogate
│   ├── AMP      + discriminator, replay buffer, style reward
│   │   └── ASE  + MI encoder, latent skill codes, diversity
│   └── Mimic/ADD  pose tracking diff obs (extends AMP)
└── MaskedMimic  expert distillation (BC, not RL)
```

Every model subclasses `TensorDictModuleBase`; shapes are inferred lazily via `nn.LazyLinear`.

---

## 4. The Configuration System

**Configs are Python files, not YAML.** Experiments live under `examples/experiments/`. Every experiment file must implement the contract documented in [`examples/experiments/format.py`](#examples-experiments-format):

| Function | Returns | Called during |
|---|---|---|
| `configure_robot_and_simulator(robot_cfg, simulator_cfg, args)` | mutates in-place | training only |
| `terrain_config(args)` | `TerrainConfig` or `None` | training only |
| `scene_lib_config(args)` | `SceneLibConfig` | training only |
| `motion_lib_config(args)` | `MotionLibConfig` | training only |
| `env_config(robot_cfg, args)` | `EnvConfig` | training only |
| `agent_config(robot_cfg, env_cfg, args)` | `PPOAgentConfig` (or subclass) | training only |
| `apply_inference_overrides(...)` (optional) | mutates in-place | inference only |

### Build pipeline (training, first run)

1. `robot_factory()` and `simulator_factory()` create base configs from CLI args.
2. `configure_robot_and_simulator()` from the experiment customizes them.
3. `env_config()` builds the environment config (action functions, observation/reward/termination components, control components).
4. `agent_config()` builds the agent config (actor/critic networks, optimizers, evaluator, optional discriminator/MI components).
5. CLI `--overrides key=value` are applied **and saved permanently** into `resolved_configs.pt`.
6. Everything is pickled to `resolved_configs.pt` (full Python objects, exact reproducibility) and `resolved_configs.yaml` (best-effort human-readable copy).
7. `experiment_config.py` (a copy of the experiment file) is saved alongside for traceability.

### Resume / inference path

- **Resume mode**: configs are loaded **directly from pickle** — the experiment file is NOT re-executed. CLI `--overrides` are **ignored** (use a new experiment name for temporary changes).
- **Inference**: loads `resolved_configs_inference.pt`, then runs `apply_inference_overrides()` if present, then applies CLI overrides. `configure_robot_and_simulator()` is NOT called (already baked in).

All configs use `_target_` strings for dynamic instantiation, e.g. `_target_: "protomotions.agents.ppo.agent.PPO"`. Instantiation happens via `protomotions.utils.hydra_replacement.instantiate()`.

---

## 5. Entry Points

### `protomotions/train_agent.py`

Main training script. Documents the entire config pipeline in its docstring.

CLI flags:
- `--robot-name` — smpl, g1, h1_2, soma23, smplx, amp, rigv1
- `--simulator` — isaacgym | isaaclab | newton | genesis | mujoco
- `--experiment-path` — path to experiment `.py`
- `--experiment-name` — directory under `results/`
- `--motion-file` — motion library (`.pt` / `.yaml` / `.motion`)
- `--num-envs` — parallel environments
- `--batch-size` — minibatch size for optimization
- `--overrides key.subkey=value` — config field overrides (saved permanently)
- `--headless` — disable visualization
- `--create-config-only` — generate configs without training (useful for migrating old checkpoints)

Outputs into `results/<experiment-name>/`:
- `resolved_configs.pt` (full pickle, **load with `torch.load(..., weights_only=False)`**)
- `resolved_configs.yaml` (best-effort human reference)
- `resolved_configs_inference.pt` (sometimes a separate inference-tuned snapshot)
- `experiment_config.py` (a literal copy of the experiment file)
- `config.yaml` (CLI args)
- `last.ckpt`, optionally `best.ckpt`, and periodic `epoch=*.ckpt`

### `protomotions/inference_agent.py`

Visualize/evaluate trained policies. CLI flags `--checkpoint`, `--simulator`, `--num-envs`, `--motion-file`, `--full-eval`, `--headless`.

Interactive keys (when not headless):
- **J** apply random forces (throw a projectile)
- **R** reset all envs
- **O** toggle camera
- **L** start/stop video recording
- **Q** quit

Override priority: CLI > `apply_inference_overrides()` > frozen configs.

### `protomotions/train_slurm.py`

SLURM/cluster launcher template. Edit `CLUSTER_LOGIN_NODE`, `CLUSTER_BASE_DIR`, container images (Singularity `.sif` / Enroot `.sqsh`), and Python paths. Synchronizes code via rsync, generates a SLURM job script, supports distributed loading via `.slurmrank.pt` per-rank motion files.

---

## 6. `protomotions/envs/` — The MDP Layer

This is the heart of the framework. The job here is to wrap a `Simulator` into an RL environment whose observations, rewards, and terminations are **declared as data**, not coded into the env class.

### 6.1 `envs/base_env/env.py` — `BaseEnv`

The single environment class for all tasks. Subclasses override hooks rather than rewriting `step()`.

**Key state buffers** (all `[num_envs, ...]`):
- `rew_buf` — accumulated reward this step
- `reset_buf`, `terminate_buf` — reset and (hard) termination flags
- `progress_buf` — episode step counter
- `respawn_root_offset` — terrain-height correction for spawn z
- `_current_raw_action`, `_current_processed_action`
- `prev_contact_force_magnitudes` — for impact-penalty deltas
- `state_history` — `StateHistoryBuffer` (if `num_state_history_steps > 0`)
- `_current_context` — cached `EnvContext` (rebuilt each step, invalidated after)

**`step(action)` flow**:

1. `_process_action(action, ctx)` → calls `action_config["fn"](action, **other_kwargs)` returning a dict with `processed_action`, optional `stiffness_targets`, `damping_targets`.
2. `simulator.step(processed_action, markers_callback=get_markers_state)` runs N physics substeps with decimation.
3. `post_physics_step()`:
   - `progress_buf += 1`
   - `state_history.rotate_and_update()` (if enabled)
   - `motion_manager.post_physics_step()` (if mimic-style task)
   - `control_manager.step()` (mimic, steering, path, masked_mimic, kinematic_replay)
   - `_current_context = _build_global_context()` — pulls robot state, builds `CurrentStateView`, `HistoricalView`, control-specific contexts (`mimic`, `steering`, `path`, `masked_mimic`), wraps in `EnvContext`.
   - `compute_observations(env_ids, ctx)` → `_process_observations` → `ComponentManager.execute_all(observation_components, ctx)`.
   - `compute_reward(ctx)` → `_process_rewards` → executes reward components, then `combine_rewards()` with grace-period zeroing and multiplicative/additive merging.
   - `check_resets_and_terminations(ctx)` → max-length check + control-component terminations + termination components, then `combine_terminations()` (logical OR, `fail_above` inverter).
   - Save flattened RobotState in `extras["raw/*"]`, persist contact forces for next step.
4. If `reset_buf.any()`, call `reset_envs(reset_ids)` (initial pose comes from default DOF / reference motion / scene).
5. Return `obs, rew_buf, reset_buf, terminate_buf, extras`.

### 6.2 `envs/mdp_component.py` — `MdpComponent`

The cornerstone abstraction. An `MdpComponent` binds a pure tensor function to context paths.

```python
MdpComponent(
    compute_func=pure_tensor_function,            # Level 1: tensors only
    dynamic_vars={"dof_pos": EnvContext.current.dof_pos},  # resolved at runtime
    static_params={"scale": 1.0, "weight": -1.0}, # compile-time constants
)
```

**Compute levels**:
1. **Level 1 — pure tensor**: `compute_func` takes only tensors, returns a tensor. ONNX-exportable, `torch.compile`-friendly.
2. **Level 2 — aggregated**: components combined by `ComponentManager` into a per-step dict; still pure.
3. **Level 3 — side effects**: reward/termination combiners apply `weight`, `multiplicative`, `zero_during_grace_period`, `min_value`, `max_value`, `use_region_weights`, `threshold`, `fail_above`. Those metadata keys are filtered out before `compute_func` is called.

**Methods**:
- `compute(ctx)` — resolves dynamic_vars, calls compute_func with static_params.
- `resolve_args(ctx)` — separates Python path resolution from tensor compute (so only the latter is compiled).
- `get_bindings_dict()` — exposes path strings (e.g. `"current.rigid_body_pos"`) for ONNX export.
- `_ensure_device(resolved)` — lazy migration of static tensor params to device on first call.

### 6.3 `envs/context_paths.py` and `envs/context_views.py`

**`FieldPath[T]`** — a descriptor with dual access:
- Class access: `EnvContext.current.dof_pos` → returns a `FieldPath` whose `.path == "current.dof_pos"`.
- Instance access: `ctx.current.dof_pos` → returns the actual tensor.

**`NestedField[T]`** — descriptor for nested view objects; propagates parent path so child FieldPaths resolve to e.g. `"current.dof_pos"`, `"historical.rigid_body_vel"`, etc.

**`resolve_path(ctx, "a.b.c")`** — walks dot-separated paths on a context instance.

**Views**:
- `CurrentStateView` — wraps a single `RobotState`, with precomputed `anchor_pos/rot/vel/ang_vel`, `root_local_ang_vel`, etc.
- `HistoricalView` — wraps `StateHistoryBuffer`, exposes clean and `noisy_historical_*` variants.
- Control context views: `MimicContext`, `MaskedMimicContext`, `SteeringContext`, `PathContext` — populated by their respective control components.
- `EnvContext` — top-level namespace with `current`, `noisy`, `historical`, `noisy_historical`, `previous_action`, `current_processed_action`, `previous_processed_action`, `ground_heights`, `body_contacts`, `current_contact_force_magnitudes`, `dt`, `mimic`, `masked_mimic`, `steering`, `path`.

### 6.4 `envs/component_manager.py` — `ComponentManager`

Executes a dict of named MdpComponents. Caches a `torch.compile`d wrapper around each `compute_func` (path resolution stays uncompiled, only the pure tensor body is compiled, avoiding dynamo tracing through descriptors). Exposes `execute_all(components, ctx)`, `execute_single`, `clear_cache`.

### 6.5 `envs/component_factories.py`

Pre-baked MdpComponent factories to cut boilerplate. Note: the module's `__all__` has known F822 lint errors — leave them alone unless fixing intentionally.

Examples: `max_coords_obs_factory(use_noisy, local_obs, root_height_obs, observe_contacts)`, `reduced_coords_obs_factory(...)`, action factories, reward factories.

### 6.6 `envs/action/action_functions.py`

Pure functions converting policy outputs to simulator targets.

- `normalized_pd_fixed_gains_action(action, pd_action_offset, stiffness, damping)` → dict with `processed_action` (PD targets), `stiffness_targets`, `damping_targets`.
- Tanh → [-1,1] → scaled by DOF range → offset by default DOF positions. Stiffness/damping fixed per-DOF or learned per-env.
- `make_pd_action_config(robot_cfg)` helper packages this into an `action_config` dict consumed by `_process_action()`.

### 6.7 `envs/obs/` — Observation kernels

| File | Kernel | What it returns |
|---|---|---|
| `humanoid.py` | `compute_humanoid_max_coords_observations` | Body pos/rot (6D tan-norm) / vel / ang_vel + optional root height & contact flags |
| `humanoid.py` | `compute_humanoid_reduced_coords_observations` | DOF pos+vel, anchor rot, root local ang_vel; compact; optional root vel/height |
| `humanoid.py` | `dof_to_local`, `dof_to_obs` | Convert DOF angles → local quat or 6D tan-norm |
| `humanoid_historical.py` | `compute_historical_*` | Stack historical state snapshots into a temporal context vector |
| `target_poses.py` | `build_max_coords_target_poses_future_rel` | Relative pose deltas between consecutive future ref frames |
| `masked_mimic.py` | `compute_target_poses_only` | Sparse target poses, zeroing out masked-out bodies |
| `steering.py` | — | Egocentric direction/speed/facing features (5D) |
| `path.py` | — | Distance-to-path + trajectory samples |
| `state_history_buffer.py` | `StateHistoryBuffer` class | Circular ring buffer of past state snapshots, with `rotate_and_update`, `reset_from_single_state`, `reset_from_states` and clean+noisy property views |

### 6.8 `envs/rewards/`

- `base.py`: primitives — `mean_squared_error_exp(actual, target, coefficient)`, `rotation_error_exp`, `power_consumption_sum`, `delta_norm`, `delta_logmeanexp`.
- `tracking.py`: `compute_gt_rew` (pos), `compute_gr_rew` (rot), `compute_gv_rew` (vel), `compute_gav_rew` (ang_vel), `compute_rh_rew` (root-height penalty).
- `regularization.py`: `compute_action_smoothness`, `compute_action_smoothness_logmeanexp`, `compute_power_consumption`, `compute_soft_dof_pos_limit_rew`, `compute_contact_match_rew`, `compute_contact_force_penalty`.
- `task.py`: `compute_heading_alignment_rew`, `compute_speed_tracking_rew`, `compute_path_distance_rew`, etc.

### 6.9 `envs/terminations/`

- `base.py`: `check_fall_contact_term` (non-allowed body parts touching ground), `check_height_term`, `check_max_length_term`.
- `tracking.py`: tracking-error termination thresholds.
- `task.py`: goal-dependent terminations (path-deviation, etc.).

### 6.10 `envs/control/`

Stateful task managers. They drive the *task* — `BaseEnv` orchestrates the physics, but the control component decides "where should the agent go now."

- `base.py`: `ControlComponentConfig` (with `_target_`), `ControlComponent` ABC with `step`, `reset`, `check_resets_and_terminations`, `populate_context(ctx)`, `get_context`, `create_visualization_markers`, `get_markers_state`.
- `manager.py`: `ControlManager` — orchestrates a dict of control components. Aggregates marker definitions, ORs terminations, calls `populate_context` on every component.
- `mimic_control.py`: `MimicControlConfig(bootstrap_on_episode_end, future_steps)` + `MimicControl`. Queries `motion_manager`, exposes `MimicContext` with `ref_state` and stacked future-pose features.
- `masked_mimic_control.py`: extends MimicControl with random body masks + time offsets (`time_alpha/beta`, `repeat_mask_probability`, `visible_target_pose_prob`). Visualization uses color-coded markers (blue=near, yellow=med, red=far).
- `steering_control.py`: `SteeringControlConfig(tar_speed_min/max, heading_change_steps_min/max, random_heading_probability, enable_rand_facing)` → `SteeringContext.tar_dir`, `tar_speed`, `tar_face_dir`.
- `path_follower_control.py`: `PathFollowerControlConfig(num_traj_samples, traj_sample_timestep, enable_path_termination, fail_dist, fail_height_dist)` → `PathContext.tar_pos`, `traj_samples`. Generates procedural paths (lines, circles).
- `kinematic_replay_control.py`: kinematic-only control (no physics) for mocap verification.

### 6.11 `envs/base_env/utils.py` — Reward / Termination combiners

- `combine_rewards(raw, configs, grace_mask, region_weights, num_envs, device)` → applies multiplicative-then-additive merging, clamping, zeroing during grace period, optional anatomical region weights. Returns `(combined_reward, logging_dict)`.
- `combine_terminations(raw, configs, num_envs, device)` → OR all conditions, invert if `fail_above=True`, return `(reset_buf, terminate_buf, logging_dict)`. Separating reset (any episode end) from terminate (true failure) is important for GAE bootstrapping.

---

## 7. `protomotions/simulator/` — Multi-Simulator Abstraction

### 7.1 Base class — `simulator/base_simulator/`

`simulator.py` defines `Simulator(RecordingMixin, ABC)` with ~17 abstract methods (see table below) plus shared infrastructure: domain randomization, projectiles, action processing, two-phase init, marker management.

**Two-phase initialization**:
1. Constructor: stores config, robot config, scene_lib, terrain, num_envs, dt. **No GPU memory yet.**
2. `_initialize_with_markers(visualization_markers)` (called by env after task creates markers): runs `_create_simulation()`, then `_finalize_setup()` builds conversion tensors.

**State dataclasses** (`simulator_state.py`):
- `RobotState` — full rigid-body state for **reads**: `dof_pos`, `dof_vel`, `dof_forces`, `rigid_body_pos/rot/vel/ang_vel`, `rigid_body_contacts`, `rigid_body_contact_forces`. Each field is `[batch, ...]`.
- `ResetState` — minimal for **writes**: `root_pos/rot/vel/ang_vel` + `dof_pos/vel`. FK is computed by the simulator. **Never** pass `rigid_body_*` to resetters.
- `RootOnlyState` — convenience for root-only reads.
- `ObjectState` — `[batch, num_objects, ...]` for scene objects.
- `StateConversion` enum: `COMMON` | `SIMULATOR`.

**`DataConversionMapping`**: precomputed reorder/quat-format tensors:
- `body_convert_to_common[sim_idx] = common_idx`, `body_convert_to_sim[common_idx] = sim_idx`
- `dof_convert_to_sim`, `dof_convert_to_common`
- `sim_w_last: bool` (True = sim uses xyzw natively)

`RobotState.convert_to_common(mapping)` reorders bodies/DOFs by tensor indexing and applies `wxyz_to_xyzw` if needed; `convert_to_sim()` is the inverse.

### 7.2 Quaternion convention

| Simulator | Native | `w_last` | Conversion on I/O |
|---|---|---|---|
| IsaacGym | xyzw | `True` | none |
| IsaacLab | wxyz | `False` | wxyz↔xyzw |
| Newton | xyzw | `True` | none |
| Genesis | xyzw | `True` | none |
| MuJoCo | wxyz | `False` | wxyz↔xyzw |

`COMMON` format is always xyzw. Conversions live in `protomotions/utils/rotations.py` (`wxyz_to_xyzw`, `xyzw_to_wxyz`).

### 7.3 Friction combine modes

- PhysX (IsaacGym, IsaacLab): **AVERAGE** `(robot + terrain) / 2`
- Newton, MuJoCo: **MAX** `max(robot, terrain)`

`base_simulator/utils.py::convert_friction_for_simulator(terrain_cfg, sim_cfg)` automatically adjusts friction values when swapping simulators. Newton workaround: robot friction set near epsilon so terrain friction dominates under MAX.

### 7.4 Control modes (`ControlType` enum)

- **`BUILT_IN_PD`** — pass position targets to simulator-native PD via `_apply_simulator_pd_targets(pd_targets)`.
- **`PROPORTIONAL`** — custom PD: `torque = kp*(target-pos) - kd*vel` computed in `_apply_control`, then `_apply_simulator_torques`.
- **`TORQUE`** — direct torque (action is torque), clamped to effort limits.

Action-noise domain randomization is applied **inside `_apply_control()`** before mode-specific dispatch.

### 7.5 Domain randomization (`base_simulator/config.py`)

| Config class | What it randomizes | When applied |
|---|---|---|
| `ActionNoiseDomainRandomizationConfig` | Uniform noise on DOF action targets | `_apply_control` (every step) |
| `FrictionDomainRandomizationConfig` | Static/dynamic friction + restitution per body | simulator init (bucketed across envs) |
| `CenterOfMassDomainRandomizationConfig` | Per-axis COM offsets per body | simulator init |
| `ObjectAssetDomainRandomizationConfig` | Scene-object friction/restitution/mass/density/COM | simulator init |
| `PushDomainRandomizationConfig` | Random root velocity impulses at random intervals | `step()` via `_apply_push_if_due` → `_apply_root_velocity_impulse` |

### 7.6 Projectile system (J-key throwing)

`ProjectileConfig`: `num_projectiles`, `cube_half_size_range`, `density`, `spawn_distance_range`, `spawn_height_range`, `speed_range`, `direction_noise_std`, `hide_z`, `hide_delay`.

Methods: `_init_projectiles`, `_throw_projectile` (aimed at robot root with Gaussian noise, leads target by robot XY velocity), `_update_projectiles` (hide after `hide_delay`), `_hide_projectiles_for_envs` (uses per-env spacing to dodge a PhysX actor-aliasing bug — see fix `156733c`).

### 7.7 The 17 abstract methods

| Method | Purpose |
|---|---|
| `_create_simulation()` | build the physics scene |
| `_get_sim_body_ordering()` → `SimBodyOrdering` | sim's native body/DOF order |
| `_physics_step()` | one physics tick |
| `_set_simulator_env_state(ResetState, ObjectState, env_ids)` | reset envs (root + DOF only) |
| `_get_simulator_root_state(env_ids)` | root state |
| `_get_simulator_bodies_state(env_ids)` | full rigid-body state |
| `_get_simulator_dof_state(env_ids)` | DOF pos/vel |
| `_get_simulator_dof_forces(env_ids)` | joint forces |
| `_get_simulator_dof_limits_for_verification()` | for sanity checks |
| `_get_simulator_bodies_contact_buf(env_ids)` | per-body contact forces |
| `_get_simulator_object_root_state(env_ids)` | scene object states |
| `_get_simulator_object_contact_buf(env_ids)` | scene object contacts |
| `_apply_simulator_pd_targets(...)` | BUILT_IN_PD mode |
| `_apply_simulator_torques(...)` | PROPORTIONAL/TORQUE modes |
| `_apply_root_velocity_impulse(lin_vel, ang_vel, env_ids)` | push DR |
| `_create_projectiles(ProjectileConfig)` | spawn cubes |
| `_set_projectile_root_states(...)` | move projectiles |
| `_get_projectile_positions_rotations()` | for viz |
| `_write_viewport_to_file(filename)` | screenshot |
| `_init_camera()` | viewport setup |
| `_update_simulator_markers(markers_state)` | refresh debug markers |

### 7.8 Concrete backends

| Backend | File | Notable details |
|---|---|---|
| **IsaacGym** | `isaacgym/simulator.py` | `gymapi.Gym`, multi-env GPU, contact sensors per body, URDF projectiles, w_last=True, friction AVERAGE |
| **IsaacLab** | `isaaclab/simulator.py` | USD-based, `SimulationContext` + `InteractiveScene`, w_last=False, Omniverse rich viz, perspective viewer in `utils/perspective_viewer.py`, scene config in `utils/scene.py` |
| **Newton** | `newton/simulator.py` | Warp-based solver, custom PD torque kernel `compute_pd_torques_kernel`, friction MAX (terrain dominant), no scene objects, migrated to PyPI Newton 1.0 (commit `b59d44f`) |
| **Genesis** | `genesis/simulator.py` | community-contributed example for new simulator authors, w_last=True, no scene objects |
| **MuJoCo** | `mujoco/simulator.py` | CPU-only, `num_envs == 1` enforced, w_last=False, friction MAX, `use_implicit_pd` flag (position actuators vs explicit PD), EMA action smoothing, no scene objects |

### 7.9 Factory — `simulator/factory.py`

- `get_simulator_config_class(name)` → `IsaacGymSimulatorConfig` / `IsaacLabSimulatorConfig` / ...
- `simulator_config(name, robot_cfg, headless, num_envs, experiment_name)` → assembled config, pulling per-engine sim params from `robot_cfg.simulation_params.<name>`.
- `update_simulator_config_for_test(current, new_sim, robot_cfg)` — swap engines at inference time (train on isaacgym, eval on newton/mujoco).

### 7.10 Import order

`protomotions/utils/simulator_imports.py::import_simulator_before_torch(name)` **must** be called before `import torch` when `name` is `isaacgym` or `isaaclab`. Returns `AppLauncher` for IsaacLab (user then runs `app = AppLauncher(args); app.launch()`), else `None`.

---

## 8. `protomotions/agents/` — RL Algorithms

All algorithms share an `ExperienceBuffer`, Lightning Fabric distribution, optional reward normalization (running mean/std), and a hook-based training loop in `BaseAgent.fit()`.

### 8.1 `agents/base_agent/agent.py` — `BaseAgent`

Abstract base class. The training loop is fully implemented; subclasses fill in algorithm-specific hooks.

Lifecycle hooks subclasses **must** implement:
- `create_model()` → `TensorDictModule` policy (and value net, discriminator, etc.)
- `create_optimizers(model)` → list/dict of Fabric-distributed optimizers
- `perform_optimization_step(batch_dict, batch_idx)` → loss dict for logging
- `register_algorithm_experience_buffer_keys()` — register extra keys in `ExperienceBuffer`

Hooks subclasses **may** override:
- `record_rollout_step(next_obs, actions, rewards, dones, ...)` — log algorithm-specific data during rollout
- `pre_process_dataset()` — compute advantages, normalize targets, etc., before minibatching
- `add_agent_info_to_obs(obs)` — augment obs (latents, discriminator features, mimic diff)
- `post_env_step_modifications(dones, terminated, extras)` — early termination conditions etc.

`fit()` skeleton:
```
for epoch in range(num_epochs):
    rollout (no_grad):
        for step in range(num_steps):
            obs = add_agent_info_to_obs(obs)
            out = model(obs)
            actions = out["action"]
            next_obs, rewards, dones, extras = env.step(actions)
            record_rollout_step(next_obs, actions, rewards, dones, ...)
            collect_rollout_step(...)
    pre_process_dataset()                       # GAE, advantage norm, etc.
    dataset = DictDataset(experience_buffer.make_dict())
    for batch_idx, batch in enumerate(dataset):
        log_dict = perform_optimization_step(batch, batch_idx)
    if eval_due: evaluator.run(); maybe save best.ckpt
    save last.ckpt
```

### 8.2 `agents/ppo/`

- `agent.py::PPO(BaseAgent)` — clipped surrogate objective with separate actor/critic optimizers, optional L2C2 (Lipschitz on noisy obs), entropy bonus when `learnable_std=True`. Hyperparameters: `tau` (GAE λ), `e_clip` (PPO ε), `gamma` (discount), `batch_size`, `advantage_normalization`, `adaptive_lr` (KL-based LR scheduling), optional clipped value loss.
- `model.py`:
  - `PPOActor(TensorDictModuleBase)` — MLP mu network + learnable/fixed `logstd`. Forward writes `action`, `mean_action`, `neglogp`.
  - `PPOModel(BaseModel)` — composes actor + critic; out_keys `[action, mean_action, neglogp, value]`.
- `utils.py::discount_values(...)` — GAE advantages, iterates backward through the rollout: `δ_t = r + γ·V(s') − V(s)`, `A_t = δ_t + γλ(1−done)A_{t+1}`.

### 8.3 `agents/amp/`

Adversarial Motion Priors. Adds a binary discriminator that distinguishes agent transitions from expert (reference) transitions, and a style reward `−log(1−p)` where `p = σ(disc_logits)`.

- `agent.py::AMP(PPO)`:
  - `amp_replay_buffer: ReplayBuffer` — circular agent-transition store for discriminator training
  - `running_amp_reward_norm: RewardRunningMeanStd`
  - `update_disc_replay_buffer`, `get_expert_disc_obs(num_samples)` — samples motion IDs/times and computes reference observations via the configured `reference_obs_components`.
  - `register_algorithm_experience_buffer_keys` → `amp_rewards`, `next_disc_value`, `disc_returns`
  - `post_env_step_modifications` — early-terminate episodes that accumulate too many "bad" disc predictions
- `model.py::Discriminator` — binary classifier; `compute_disc_reward(disc_logits)` returns the style reward.
- `model.py::AMPModel(PPOModel)` — adds the discriminator network.

### 8.4 `agents/ase/`

Adversarial Skill Embeddings. Extends AMP with a **latent skill code** sampled on the unit sphere, an **MI encoder** that predicts the latent from the agent's behavior, and an **MI reward** that encourages each skill to produce distinguishable behaviors.

- `agent.py::ASE(AMP)`:
  - `latents: [num_envs, latent_dim]` and `latent_reset_steps: StepTracker`
  - `add_agent_info_to_obs` — re-samples latents for done envs, injects `obs["latents"]`
  - `sample_latents(n)` — Gaussian → unit-norm
  - `register_algorithm_experience_buffer_keys` → `mi_rewards`, `next_mi_value`, `mi_returns`
  - `perform_optimization_step` → AMP step + MI critic step
- `model.py::ASEDiscriminatorEncoder(Discriminator)` — discriminator with MI encoder head (encoder weights initialized U[-0.1, 0.1]).
- `model.py::ASEModel(AMPModel)` — policy conditioned on latents.

### 8.5 `agents/mimic/agent_add.py`

ADD = Adversarial Discriminator Diffusion (or "AMP-with-tracking-diff"). About 50 LOC; demonstrates how modular the framework is.

- `MimicADD(AMP)`:
  - `add_agent_info_to_obs` — samples motion, computes reference pose via `compute_humanoid_max_coords_observations`, computes current pose, writes `obs["mimic_target_poses_diff"] = ref − current`.
  - `get_expert_disc_obs` — pads with zeros for the tracking-diff slot.

### 8.6 `agents/masked_mimic/`

Not RL — supervised expert distillation.

- `agent.py::MaskedMimic(BaseAgent)`:
  - Loads a pre-trained "expert" policy via `expert_model_path`
  - During rollout: query both expert (on full obs) and student (on masked obs)
  - Loss: MSE(student.action, expert.action)
  - Optional VAE noise via `vae_noise` for stochastic distillation
- `model.py::FeedForwardModel`, `MaskedMimicModel` — feed-forward and optionally VAE-augmented student networks.

### 8.7 `agents/common/`

Shared NN building blocks, all `TensorDictModuleBase`.

- `mlp.py::MLPWithConcat(TensorDictModuleBase)` — concatenates `in_keys`, applies optional `NormObsBase` normalization, runs an MLP built from `nn.LazyLinear` layers with optional LayerNorm + activation, writes `out_keys[0]`.
- `mlp.py::build_mlp(config)` — builds the `nn.Sequential`.
- `common.py::NormObsBase` — wraps `RunningMeanStd`, normalizes & updates stats in train mode.
- `common.py::ObsProcessor` — applies a chain of tensor ops (permute/reshape/squeeze/unsqueeze/expand/normalize/forward) configured by `ModuleOperationConfig`.
- `common.py::ModuleContainer` — sequential `TensorDictModule` chain with data-flow validation (each model's required `in_keys` must already be available).
- `common.py::weight_init`, `get_params`, `apply_module_operations`.

### 8.8 `agents/utils/`

- `data.py::ExperienceBuffer(nn.Module)` — `[num_steps, num_envs, *shape]` storage, with `register_key`, `update_data`, `batch_update_data`, `make_dict()` (flattens to `[num_steps*num_envs, ...]`). `swap_and_flatten01` helper.
- `data.py::DictDataset(Dataset)` — minibatch shuffler over the flattened buffer.
- `normalization.py::RunningMeanStd` — Welford online algorithm in float64; supports Fabric all-reduce; clamp + lazy shape init.
- `normalization.py::RewardRunningMeanStd` — adds reward-specific update accounting for episode termination and discount.
- `replay_buffer.py::ReplayBuffer(nn.Module)` — circular buffer for AMP/ASE.
- `metering.py::TimeReport`, `TensorAverageMeterDict`.

### 8.9 Model invariants

- All models extend `TensorDictModuleBase`; declare `in_keys` (read) and `out_keys` (write).
- `nn.LazyLinear` everywhere → shapes materialize on the first forward, **before** Fabric wraps the model (a dummy forward is run during `setup()`).
- Reward normalization (optional) operates on the running mean/std and is persisted via `state_dict`.

---

## 9. `protomotions/components/` — Motion, Pose, Scene, Terrain

These four libraries handle the data pipeline outside the simulator.

### 9.1 `components/motion_lib.py` — `MotionLib`

Concatenated-tensor motion store with O(1) indexing.

**`MotionLibConfig`**: `motion_file: Optional[str]`, `get_motion_state_use_blend: bool = True`.

**Per-frame tensors** (all concatenated across motions, indexed via `length_starts`):
- `gts` — global body **positions** `[T, B, 3]`
- `grs` — global body **rotations** (xyzw) `[T, B, 4]`
- `gvs` — global body **linear velocities** `[T, B, 3]`
- `gavs` — global body **angular velocities** `[T, B, 3]`
- `dps` — DOF positions `[T, D]`
- `dvs` — DOF velocities `[T, D]`
- `contacts` (optional) — `[T, B]` per-body contact label
- `lrs` (optional) — local body rotations

**Per-motion metadata**: `motion_lengths` (seconds), `motion_num_frames`, `motion_dt = 1/fps`, `motion_weights` (sampling), `length_starts` (start frame for each motion).

**Methods**:
- `get_motion_state(motion_ids, motion_times, joint_3d_format='exp_map')` → `RobotState` with SLERP for quats and linear blend for positions; uses `calc_frame_blend()` on times.
- `get_motion_state_exact_frame(motion_ids, frame_indices)` → no blending.
- `get_motion_length(motion_ids)`, `get_motion_num_frames(motion_ids)`, `num_motions()`.
- `save_to_file(path)`, `load_from_file(path)` — packaged `.pt` (fast).
- `_load_motions(motion_file)` — fallback path for `.motion` files / directories.
- `_fetch_motion_files(motion_file)` — YAML parser, returns `(file_list, weights)`.
- `process_packaged_motion_file_name_multi_gpu(motion_file)` — replaces a `"slurmrank"` token with the current rank for distributed loading.
- `smooth_contacts(window_size)` — moving-average smoothing that respects motion boundaries.
- `translate_all_motions_to_origin(target_xy)` — shift root positions, preserve z.

`MotionLib.empty(device)` creates a no-op library for tasks without reference motion (steering, path).

### 9.2 `components/pose_lib.py`

Forward kinematics + helper analysis.

**`KinematicInfo`** (extracted from MJCF via `extract_kinematic_info(mjcf_path)`):
- `body_names`, `dof_names`, `parent_indices`
- `local_pos: [B, 3]`, `local_rot_ref_mat: [B, 3, 3]` (from MJCF quat wxyz)
- `hinge_axes_map: Dict[int, [num_dofs_at_body, 3]]`
- `nq`, `nv`, `num_bodies`, `num_dofs`, `dof_limits_lower/upper`

**`ControlInfo`**: `stiffness`, `damping`, `armature`, `friction`, `effort_limit`, `velocity_limit` per DOF (from MJCF actuators).

**Functions**:
- `extract_kinematic_info(mjcf_path)` — parses MJCF via `dm_control.mjcf`, builds the full kinematic tree.
- `extract_control_info(asset)` — same for actuators.
- `compute_joint_loss_weights(kinematic_info, discount=0.9, min_weight=0.01)` — heuristic: proximal joints (more descendants) → higher weight; exponential decay.
- `compute_body_density_weights(kinematic_info, discount=0.9)` — penalize dense regions (e.g., finger chains).
- `compute_region_uniform_weights(kinematic_info)` — auto-discovers anatomical regions by tracing each leaf node back to root.
- `extract_transforms_from_qpos(kinematic_info, qpos)` → `(root_pos, joint_rot_mats)`. Handles 1-DOF hinges and 3-DOF coupled rotations.
- `compute_forward_kinematics_from_transforms(...)` → `(world_pos, world_rot_mat)`.
- `compute_cartesian_velocity(pos, fps, velocity_max_horizon=1)` — **multi-horizon minimum-magnitude finite differences** for mocap-noise filtering: compute Δp over horizons 1..H, pick the min per frame/body. A 2 cm jitter at 30 fps shows up as 1 m/s over one frame and 0.3 m/s over three frames, so the minimum filters it out.
- `compute_angular_velocity(rot_mats, fps, velocity_max_horizon=1)` — same idea using rotation logarithm.
- `compute_kinematics_velocities(world_pos, world_rot_mat, fps, horizon)` — combined.
- `fk_batch_mjcf_with_velocities(kinematic_info, qpos, fps, compute_velocities, velocity_max_horizon=3)` → `RobotState` — full one-shot pipeline.
- `fk_from_transforms_with_velocities(...)` — when you already have transforms.

### 9.3 `components/scene_lib.py`

Object management with optional motion + collision pointclouds.

**`ObjectOptions`** — physics/material properties: `fix_base_link`, `vhacd_enabled/params`, `density|mass` (exclusive, default `1000 kg/m³`), `damping`, `max_angular_velocity`, `friction`, `restitution`, `color`. Methods: `to_dict`, `physics_material_kwargs`, `single_friction`, `with_asset_property_overrides`.

**`SceneObject`** (base): `translation`, `rotation` (xyzw, static `[4]` or motion `[T, 4]`), `fps`, `options`, `object_dims: [6]` bbox, `object_pointcloud: [N, 3]`, `object_pointcloud_normals: [N, 3]`, `is_first_instance`/`first_instance_id`/`instance_id` for asset reuse. Methods: `has_motion()`, `start_pose()`, `calculate_dimensions()`, `compute_pointcloud(num_samples)`.

Subclasses: `MeshSceneObject(object_path)`, `BoxSceneObject(dims)`, `SphereSceneObject(radius)`, `CylinderSceneObject(radius, height)`.

**`Scene`**: list of `SceneObject`.

**`SceneLibConfig`**: `scene_file`, `subset_method` (FIRST/RANDOM/SEQUENTIAL/custom), `replicate_method`, `pointcloud_samples_per_object`, `num_objects_per_env`.

**`SceneLib`**:
- Internal state mirrors `MotionLib` design: concatenated `_object_translations`/`_object_rotations`, `_motion_lengths`, `_motion_starts`, `_motion_dts`, `_motion_num_frames`, `_object_pointclouds`, `_object_pointcloud_normals`.
- `_scene_to_original_scene_id`, `_is_static_object`, `_scene_offsets` — manage replication and placement on terrain.
- `save_scenes_to_file(scenes, path)` (static), `get_scene_for_env(env_id)`.
- Reserves space in the terrain's flat **object playground** for each scene.

### 9.4 `components/terrains/`

Procedural height-field generator + scene placement manager.

**`TerrainConfig`**: `map_length/width`, `border_size`, `num_levels` (curriculum), `num_terrains` (variations per level), `terrain_proportions: [smooth_slope, rough_slope, stairs_up, stairs_down, discrete, stepping_stones, poles, flat]`, `horizontal_scale`, `vertical_scale`, `num_samples_per_axis`, `spacing_between_scenes`, `minimal_humanoid_spacing`, `terrain_path`, `load_terrain`, `save_terrain`, `sim_config: TerrainSimConfig` (friction, restitution, height_offset, combine_mode).

**`Terrain`** (`terrain.py`):
- `height_field_raw: np.ndarray[int16][tot_rows, tot_cols]`
- `walkable_field_raw`, `ceiling_field_raw`, `flat_field_raw` masks
- `height_samples: torch.Tensor` (scaled), `vertices/triangles` (trimesh), `scene_placement_map`.
- Layout (top-down):
  ```
  [Border] [Terrain Grid (varied)] [Object Playground (flat z=0)] [Border]
  ```
- Methods: `curriculum`, `generate_subterrains`, `init_height_points`, `get_ground_heights(xy)`, `get_heights_jit(xy)`, `find_terrain_height_for_max_below_body(rigid_body_pos)`, `save_terrain`, `load_terrain`.

**Subterrain generators** (`subterrain_generator.py`): `random_uniform_subterrain`, `pyramid_sloped_subterrain`, `pyramid_stairs_subterrain`, `stepping_stones_subterrain`, `discrete_obstacles_subterrain`, `poles_subterrain` — each mutates a `SubTerrain.height_field_raw`.

**`SubTerrain`** (`subterrain.py`): one terrain patch — `height_field_raw[width, length]`, ceiling/walkable masks, scaling, name.

**`shape_utils.py`** — primitive shape drawers backed by skimage/scipy: `draw_disk`, `draw_circle`, `draw_curve`, `draw_polygon`, `draw_ellipse`.

---

## 10. `protomotions/robot_configs/` — Per-Robot Configuration

### 10.1 `factory.py`

`robot_config(robot_name, **updates) -> RobotConfig` — dispatches to the per-robot subclass (SMPL, SMPLX, AMP, G1, H1_2, RigV1, Soma23). After construction it applies `config.update_fields(**updates)`.

### 10.2 `base.py`

- **`SimulatorParams`** — nested fields for each engine's physics params.
- **`InitState`** — initial root position.
- **`ControlType`** Enum: `BUILT_IN_PD`, `TORQUE`, `PROPORTIONAL` (with `from_str` for case-insensitive parsing).
- **`RobotAssetConfig`** — `asset_file_name` (`.xml` MJCF only, validated), `self_collisions`, `max_linear_velocity`, `max_angular_velocity`, `density`, `damping`.
- **`ControlConfig`** — `control_type`, `override_control_info` (regex → `ControlInfo`), `soft_pos_limit` (default 0.9), `control_info` (populated from MJCF via `pose_lib.extract_control_info`).
- **`RobotConfig`** — the headline dataclass:
  - `asset: RobotAssetConfig`
  - `common_naming_to_robot_body_names: Dict[str, List[str]]` — semantic mapping (e.g. `"all_left_foot_bodies": ["L_Ankle", "L_Toe"]`). **Values MUST be lists.**
  - `default_root_height`, `default_dof_pos` (can be regex-pattern dict resolved in `__post_init__`), `contact_bodies`, `trackable_bodies_subset`, `non_termination_contact_bodies`
  - `kinematic_info: KinematicInfo` — extracted at `__post_init__` via `pose_lib.extract_kinematic_info()`
  - `number_of_actions`, `anchor_body_index` — derived
  - `simulation_params: SimulatorParams` — per-engine overrides
  - `update_fields(**kwargs)` — updates and reprocesses dependent mappings
- **`abstract_names_to_body_names(names, robot_config)`** — expands abstract names like `"all"`, `"root"`, or `"all_left_foot_bodies"` into concrete lists.

### 10.3 Per-robot files

| File | Robot | Notes |
|---|---|---|
| `smpl.py` | SMPL humanoid | `mjcf/smpl_humanoid.xml`, regex-based stiffness/damping per joint group, trackable bodies (Pelvis, L/R Ankle, L/R Hand, Head). IsaacGym 60 fps / 2 dec / 2 substeps; IsaacLab 120 fps / 4 dec. |
| `smplx.py` | SMPL-X | adds finger joints |
| `g1.py` | Unitree G1 | armature constants (`ARMATURE_5020`, `ARMATURE_7520_14`, …), regex `DEFAULT_JOINT_POS` (standing pose like `".*_hip_pitch_joint": -0.312`), control parameters derived from BeyondMimic |
| `h1_2.py` | Unitree H1.2 | analogous to G1 |
| `soma23.py` | SOMA-X skeleton (NVlabs) | for the BONES dataset |
| `amp.py` | AMP humanoid | classic Peng et al. character |
| `rigv1.py` | RigV1 | research humanoid |

Robot MJCF assets live under `protomotions/data/assets/mjcf/` (`amp_humanoid.xml`, `amp_humanoid_sword_shield.xml`, `g1_bm.xml`, `g1_bm_box_feet.xml`, `g1_holo*.xml`, `h1_2*.xml`, `rigv1_humanoid.xml`, `smpl_humanoid.xml`, `smplx_humanoid.xml`, `soma23_humanoid.xml`).

---

## 11. `protomotions/utils/` — Helpers

| File | What it provides |
|---|---|
| `simulator_imports.py` | `import_simulator_before_torch(name)` — must run before `import torch` for IsaacGym/Lab. Returns `AppLauncher` for IsaacLab. |
| `component_builder.py` | `build_terrain_from_config`, `build_scene_lib_from_config`, `build_motion_lib_from_config`, `build_simulator_from_config`, `build_all_components(...)` — one-call factory that returns `{terrain, scene_lib, motion_lib, simulator}`. |
| `config_builder.py` | `build_standard_configs(args, terrain_fn, scene_fn, motion_fn, env_fn, configure_fn=None, agent_fn=None)` — runs the experiment functions in dependency order. |
| `config_utils.py` | `import_experiment_relative_eval_overrides(rel_path)` — imports `apply_inference_overrides` from a relative experiment module using `inspect.stack()`. |
| `inference_utils.py` | `apply_all_inference_overrides(...)`, `apply_backward_compatibility_fixes(...)` — for loading old checkpoints. |
| `torch_utils.py` | `grad_norm(params)`, `to_torch(x, dtype, device, requires_grad)`, `seeding(seed, torch_deterministic)`. |
| `rotations.py` | TorchScript-compiled: `normalize`, `wxyz_to_xyzw`, `xyzw_to_wxyz`, `_sqrt_positive_part`, `quat_mul(a, b, w_last)`, `quat_conjugate`, `quat_rotate`, `quat_rotate_inverse`, matrix↔quat conversions. |
| `motion_interpolation_utils.py` | `interpolate_pos(p0, p1, blend)` (lerp), `interpolate_quat(q0, q1, blend)` (SLERP) — same code used by training and deployment. |
| `mesh_utils.py` | `as_mesh(scene_or_mesh)` (Trimesh Scene→Mesh), `compute_bounding_box(mesh)`. |
| `hydra_replacement.py` | `get_class("a.b.C")`, `instantiate(cfg, **kwargs)` — lightweight `_target_` instantiation without bringing in Hydra. |
| `export_utils.py` | `ONNXExportWrapper(nn.Module)` — bridges TensorDict ↔ positional tensors for `torch.onnx.export`. Helpers for exporting PPO models / observation graphs / unified pipelines. `_resolve_context_path(path, ctx)`. |
| `fabric_config.py` | `FabricConfig` dataclass: `accelerator`, `devices`, `num_nodes`, `strategy`, `precision`, loggers. |

---

## 12. `protomotions/tests/`

| File | What it asserts |
|---|---|
| `test_newton_simulator_fk.py` | Newton FK matches MotionLib reference. CLI: `--motion-file`, `--robot g1`, `--frame-idx 50`. |
| `test_isaacgym_friction_mode.py` | Empirically confirms IsaacGym uses **AVERAGE** friction (slides a box across ground frictions {0.4, 1.6}). |
| `test_newton_contact_body_labels.py` | Newton contact body labels round-trip correctly. |
| `test_object_asset_domain_randomization.py` | Object asset DR distributes properties across buckets. |
| `test_scene_object_options.py` | `ObjectOptions` defaults, density/mass exclusivity. |

Run via `pytest protomotions/tests/` or single files.

---

## 13. `examples/` — Experiments, Tutorials, Benchmarks

### 13.1 `examples/experiments/`

#### `format.py`

The contract — every experiment file should mirror these signatures:

```python
def configure_robot_and_simulator(robot_cfg, simulator_cfg, args): ...
def terrain_config(args) -> Optional[TerrainConfig]: ...
def scene_lib_config(args) -> SceneLibConfig: ...
def motion_lib_config(args) -> MotionLibConfig: ...
def env_config(robot_cfg, args) -> EnvConfig: ...
def agent_config(robot_cfg, env_cfg, args) -> PPOAgentConfig: ...
def apply_inference_overrides(robot_cfg, simulator_cfg, env_cfg, agent_cfg,
                              terrain_cfg, motion_lib_cfg, scene_lib_cfg, args): ...
```

#### `mimic/mlp.py` — full-body motion tracking

- **Control**: `MimicControlConfig(bootstrap_on_episode_end=True)`.
- **Obs**: `max_coords_obs` + `previous_actions` + `mimic_target_poses` (future ref).
- **Termination**: `tracking_error` (threshold 0.5, removed at inference).
- **Rewards**: gt/gr/gv/gav tracking + action smoothness + contact matching + power regularization.
- **Actor**: 6×1024 MLP relu; **Critic**: 4×1024.
- **Optimizers**: Adam (actor 2e-5, critic 1e-4).
- **Evaluator**: `MimicEvaluatorConfig` with motion-weight rules (success discount 0.999) — failures get sampled more.
- `configure_robot_and_simulator`: adds contact sensors on `["all_left_foot_bodies", "all_right_foot_bodies"]`.
- `apply_inference_overrides`: relaxes termination, sets max episode length to a very large number, forces motion resampling/reset.
- `MimicMotionManagerConfig(init_start_prob=0.2 train | 1.0 inference)`.

#### `amp/mlp.py` — Adversarial Motion Prior

Pure-locomotion AMP. Dilated history `[1, 2, 3, 4, 8, 16, 32]` for both actor and discriminator. Obs: `max_coords_obs` + `historical_max_coords_obs`. No control components.

#### `steering/mlp.py` — Locomotion steering (uses AMP)

- **Control**: `SteeringControlConfig()`.
- **Obs**: `max_coords_obs` + `historical_max_coords_obs` + `steering` (5-D feature vector pulled from `EnvContext.steering`).
- **Rewards**: heading + velocity rewards in `rewards/task.py`.

#### `masked_mimic/transformer.py`

Transformer student learning from a pretrained tracker — pure BC. References frozen expert via `MaskedMimic` agent.

#### `add/mlp.py`

ADD variant — extends AMP with target pose-diff observations.

#### `ase/mlp.py`

ASE — AMP + MI encoder + latent skill codes.

#### `path_follower/mlp.py`

Navigation along procedural paths via `PathFollowerControlConfig`.

### 13.2 Tutorials (`examples/tutorial/`)

`0_create_simulator.py` → `1_add_terrain.py` → `2_load_robot.py` → `3_create_scene.py` → `4_basic_env.py` → `5_motion_manager.py` → `6_mimic_control.py` → `7_deepmimic.py`. Each step adds one layer.

### 13.3 Benchmarks (`examples/benchmark/`)

- `isaacgym_bench.py`, `isaaclab_bench.py`, `genesis_bench.py` — measure steps/sec per simulator.

### 13.4 Visualizers

- `env_kinematic_playback.py` — FK-only motion playback (no physics).
- `motion_libs_visualizer.py` — interactive viewer over packaged `.pt` libraries.
- `random_pose_visualizer.py` — sample random poses within joint limits.

---

## 14. `deployment/` — ONNX Export & Real-Robot Bridge

### 14.1 `deployment/export_bm_tracker_onnx.py`

Exports BeyondMimic-style tracker policies to a single ONNX bundle without launching a simulator.

`export_tracker(checkpoint, output_dir, validate=True)` workflow:
1. Load `resolved_configs_inference.pt` (fallback to `resolved_configs.pt`).
2. **Auto-detect actor obs keys** from `agent_config.model.actor.in_keys`.
3. Extract robot dimensions.
4. Build `MockContext` (sub-mocks: `_MockState`, `_MockMimic`, `_MockHistorical`) for shape tracing.
5. Create `ObservationExportModule` — runs the actor obs MdpComponents on the mock.
6. Reconstruct the actor network, load weights, force `nn.LazyLinear` materialization with a dummy forward.
7. Build `ActionExportModule` — applies PD transforms to actor output.
8. Compose `UnifiedPipelineModule` (obs → actor → action).
9. `torch.onnx.export` with dynamic batch axis.
10. Validate via `onnxruntime`.
11. Emit a deployment-contract YAML containing:
    - Policy inputs/outputs (semantic key + shape)
    - Joint/body names, stiffness, damping, effort limits
    - `control_dt`, `physics_dt`, decimation
    - Future step indices
    - Checkpoint path and control type

Helper: `_resolve_attr_path(path, obj)`.

### 14.2 `deployment/motion_utils.py` — `MotionPlayer`

Lightweight, dependency-minimal motion playback for deployment.

Three input formats:
- `.motion` (single `RobotState` dict)
- packaged `.pt` (multi-motion library; needs `motion_index`)
- cached `.pt` (pre-resampled to control rate)

Properties: `total_frames`, `num_bodies`, `num_dofs`, `control_dt`.

Methods: `get_state_at_frame(frame_idx)` → dict (`dof_pos/vel`, `body_rot/pos/vel/ang_vel`); `get_future_references(frame_idx, step_indices)`; `cache_to_file(out_path)` — pre-resample to the deployment rate and write NumPy arrays (eliminating torch from later runs).

Two modes: **interpolation** (uses `calc_frame_blend` + `interpolate_pos` + `interpolate_quat` from `protomotions.utils.motion_interpolation_utils` — exact same code as training) or **cached** (NumPy indexing only).

### 14.3 `deployment/state_utils.py`

Bridges raw simulator state to ONNX inputs. Provides both NumPy (deployment) and PyTorch (export) variants.

NumPy:
- `mujoco_wxyz_to_xyzw(wxyz)` — MuJoCo native to ProtoMotions format.
- `compute_anchor_rot_np(rigid_body_rot, anchor_body_index)` — pulls the IMU body's orientation (e.g., G1's `torso_link` at index 16).
- `compute_root_local_ang_vel_np(rigid_body_rot, rigid_body_ang_vel, root_body_index=0)` — rotates root angular velocity from world → local frame. **Only use when the source is world-frame** (MuJoCo `data.cvel`); if it's already local (`qvel` or IMU gyro), use directly.
- `_quat_rotate_inverse_np(q_xyzw, v)`.
- `compute_yaw_offset_np(robot_quat, motion_quat)`, `apply_heading_offset_np(offset, body_rots)` — heading alignment between physical robot and reference motion.

PyTorch (used at export time): `compute_anchor_rot`, `compute_root_local_ang_vel`.

⚠️ **Body indexing pitfall**: `anchor_rot` uses the **anchor body** (e.g. torso, index 16 for G1). `root_local_ang_vel` uses the **root body** (pelvis, index 0). They are different bodies; mixing them silently produces wrong observations.

### 14.4 `deployment/test_tracker_mujoco.py`

Stand-alone MuJoCo deployment demo — the **deployment contract** in code form. At 50 Hz control:
1. Read `qpos`, `qvel`, `xquat`, `cvel` from MuJoCo.
2. Derive `anchor_rot` and `root_local_ang_vel`.
3. Query future motion frames from `MotionPlayer`.
4. Run ONNX inference → PD targets + stiffness + damping.
5. Apply acceleration clamp + EMA action filter.
6. Step MuJoCo for decimation substeps.

Conventions: `data.xquat[body_id + 1]` (body 0 is world). `data.cvel[body_id + 1, 0:3]` is world-frame ang vel. Reduced-coords obs are position-invariant → no motion realignment needed.

---

## 15. `scripts/` — CLI Helpers

| File | Purpose |
|---|---|
| `retarget_amass_to_robot.sh`, `retarget_single_motion_to_robot.sh` | Batch / single-clip retargeting wrappers around `pyroki/` scripts. |
| `joint_monkey.py` | IsaacGym visualizer that cycles each DOF through its range, helpful for verifying joint limits. 36-env 6×6 grid. Supports `--show_axis`. Robots: `h1_2`, `g1`, `smplx_humanoid`, `rigv1_humanoid`. |
| `smoke_test.sh` | Minimal end-to-end run to confirm install. |
| `search_motions.py` | `main()` — substring-search a packaged `.pt` library and print matching indices. |
| `subset_motion_lib.py` | `subset_motion_lib(input, output, sample_every=200)` — pick every N-th motion, rebuild `length_starts` and reindex frame tensors. |
| `optimize_blend_weights.py` | Map Blender mesh skinning weights to MJCF rigid bodies via nearest-neighbor in world space. Parses MJCF (`parse_mjcf_visual` → `BodyVisualInfo`), loads STL meshes, computes rest-pose world transforms, KD-tree matches Blender vertices to STL bodies and writes 1.0 rigid weights. Auto-relaunches inside Blender if needed. |
| `analyze_mimic_most_failed_motions.py` | Inspect motion weights from training checkpoints to find motions the agent struggles with. |
| `create_video.sh` | Helper for assembling rendered frames into mp4. |

---

## 16. `pyroki/` — Retargeting via PyRoki

As of v3, retargeting uses [PyRoki](https://github.com/chungmin99/pyroki) (JAX-based IK) rather than the older [Mink](https://github.com/kevinzakka/mink).

- `batch_retarget_to_g1_from_keypoints.py` — retarget AMASS motions to Unitree G1. Uses `pk.ikdc` / `pk.solve`. Defines link-name mappings (e.g. `"left_hip" → "left_hip_pitch_link"`), direct limb-pair constraints (shoulder–elbow–wrist, hip–knee–ankle–foot). `get_humanoid_retarget_indices()` maps AMASS keypoint indices → G1 joint indices.
- `batch_retarget_to_h1_2_from_keypoints.py` — same for H1.2.

The pipeline starts from extracted keypoints (use `data/scripts/extract_keypoints_from_single_motion.py` and `extract_retargeting_input_keypoints_from_packaged_motionlib.py`) and produces retargeted `.motion` files (see `data/scripts/convert_pyroki_retargeted_robot_motions_to_proto.py`).

---

## 17. `usd_convert/` — MJCF → USDA Conversion

For IsaacLab, which works in USD.

- `convert_robot_mjcf_to_usda.py` — top-level wrapper:
  1. Verify MJCF is flat (no `<default>`, `class=`, or `<freejoint>`) via `verify_mjcf_is_flat(path)` (run `flatten_mjcf.py` first if not).
  2. Strip MuJoCo-only elements (`<contact>`, `<sensor>`, `<tendon>`) into a temp cleaned XML.
  3. Invoke Isaac Lab's MJCF importer as a subprocess with `--make-instanceable --headless`.
  4. Patch missing visual meshes into the base USD.
  5. Edit the USDA: remove `_cleaned` suffix, add `over "worldBody" (active = false)` to deactivate the extra articulation root.
  6. Delete the temp file.
- `convert_mjcf_to_usd.py` — low-level Isaac Lab importer wrapper invoked by the above.
- `flatten_mjcf.py` — expand all `<default>` class inheritance and `class=`/`childclass=` references inline.
- `convert_objects_to_usd.py` — primitives/meshes → USD.
- `patch_usd_visual_meshes.py` — fix dropped visual meshes after import.

Limitations: tendons / sensors / actuator ctrlrange / custom contact friction / joint solimplimit are all dropped (PhysX has no equivalents). Marker bodies cause cosmetic PhysX inertia warnings.

---

## 18. `docs/` and `data/`

### 18.1 `docs/`

Sphinx project (built and deployed to `https://protomotions.github.io` via `.github/workflows/deploy-docs.yml`).

- `docs/source/conf.py` — Sphinx config.
- `docs/source/_ext/dataclass_metadata.py` — custom extension that documents dataclass fields including ProtoMotions config structures.

### 18.2 `data/`

| Subdir | Contents |
|---|---|
| `data/__init__.py` | (empty package marker) |
| `data/static/` | GIFs / images for README |
| `data/smpl/`, `data/soma/`, `data/g1-kimodo-generated/`, `data/soma-kimodo-generated/` | reference motions per skeleton |
| `data/motion_for_trackers/` | small seed motion packs (e.g. `g1_bones_seed_mini.pt`, `soma23_bones_seed_mini.pt`) |
| `data/pretrained_models/` | shipped checkpoints (`motion_tracker/g1-bones-deploy/last.ckpt`, `motion_tracker/soma-bones/last.ckpt`, etc.) |
| `data/yaml_files/` | motion-set manifests (`amass_smpl_train.yaml`, `g1_phuma_train.yaml`, `motion_fps_amass.yaml`, …) |
| `data/scripts/` | data converters (see below) |

**`data/scripts/`** — extensive conversion utilities, summary:

- `bvh.py`, `convert_soma23_bvh_to_proto.py` — BVH parsing and conversion.
- `compose_mjcf_from_neutral_joints.py` — build a MJCF from neutral-pose joint specs.
- `contact_detection.py`, `recompute_contacts_from_packaged_motionlib.py` — derive per-body contact labels from world positions.
- `convert_amass_to_motionlib.py`, `convert_amass_to_proto.py` — AMASS humans into ProtoMotions formats.
- `convert_g1_csv_to_proto.py` — SEED G1 CSV → `.motion`.
- `convert_phuma_to_motionlib.py`, `convert_phuma_to_proto.py` — PHUMA dataset.
- `convert_pyroki_retargeted_robot_motions_to_proto.py` — PyRoki output → `.motion`.
- `convert_rigv1_npz_to_proto.py`, `convert_soma23_npz_to_proto.py`, `convert_soma23_to_proto.py` — per-format converters.
- `create_motion_fps_yaml.py`, `create_motion_split_yaml.py` — generate manifests.
- `extract_keypoints_from_single_motion.py`, `extract_retargeting_input_keypoints_from_packaged_motionlib.py` — produce keypoints for retargeting.
- `keypoint_utils.py` — shared keypoint helpers.
- `motion_filter.py` — filter motions by length, contact, etc.
- `process_grab.py` — preprocess the GRAB grasping dataset.

### 18.3 `protomotions/data/assets/`

| Subdir | Contents |
|---|---|
| `mjcf/` | All robot MJCFs (see §10.3) |
| `urdf/` | URDF variants when needed (Newton/IsaacGym) |
| `usd/` | USDA exports for IsaacLab |
| `mesh/` | STL meshes per robot (G1, H1.2, …) — many tracked via Git LFS |
| `checkerboard/` | ground textures |

---

## 19. Build, Dependencies & Containers

### 19.1 `setup.py`

```python
setup(
    name="protomotions",
    version="3.1",
    packages=["protomotions"],
    python_requires=">=3.8",
)
```

The package is intentionally tiny — heavy installs go into the per-backend `requirements_*.txt`.

### 19.2 Per-backend requirements

| File | Highlights |
|---|---|
| `requirements_isaacgym.txt` | `torch>=2.2`, `lightning>=2.3`, `tensordict>=0.5.0`, `wandb>=0.13.4`, `dm_control>=1.0`, `omegaconf==2.3.0` |
| `requirements_isaaclab.txt` | `tensordict==0.9.0`, `lightning`, `hydra-core==1.3.2`, `wandb==0.15.12`, `setuptools==69.5.1`, `dm_control>=1.0` |
| `requirements_newton.txt` | `mujoco==3.5.0` (pinned for Newton 1.0), `tensordict==0.9.0`, `lightning`, `openmesh==1.2.1`, `dm_control==1.0.37` |
| `requirements_genesis.txt` | `genesis-world`, `torch==2.5.0`, `lightning==2.5.0.post0`, `mink==0.0.7`, `open3d==0.19.0` |
| `requirements_mujoco.txt` | `mujoco>=3.0.0`, `dm_control>=1.0`, CPU-only setup |

Install (representative):
```bash
pip install -e .
pip install -r requirements_isaacgym.txt    # or _isaaclab / _newton / _genesis / _mujoco
```

For MuJoCo CPU-only:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -e .
pip install -r requirements_mujoco.txt
# MuJoCo backend supports num_envs=1 only.
```

For Newton: `pip install "newton[examples]"` first.

### 19.3 Dockerfiles

| File | Base | What it does |
|---|---|---|
| `Dockerfile.isaacgym` | `nvidia/cuda:12.1.0-devel-ubuntu20.04` | Python 3.8, downloads IsaacGym Preview 4, installs torch + reqs (strips `^torch` line from reqs since IsaacGym brings its own). Note: protomotions itself is NOT copied — it's rsync'd at run time by `train_slurm.py` so containers run fresh code. |
| `Dockerfile.isaaclab` | `nvcr.io/nvidian/isaac-lab:IsaacLab-release-2.3.0-latest-release-5-1` | Adds cmake, installs reqs via `/workspace/isaaclab/isaaclab.sh -p -m pip`. Sets EULA env vars. |
| `Dockerfile.newton` | `nvidia/cuda:12.4.0-devel-ubuntu22.04` | Python 3.10 + uv venv, torch CUDA 12.4, `newton[examples]`, requirements_newton.txt. Starts Xvfb for headless GL. |

### 19.4 `.pre-commit-config.yaml`

- License-header insertion (Apache-2.0) on every `.py` (except `setup.py`)
- Ruff lint + format
- Typos spell-checker
- Trailing-whitespace and EOF fixes
- check-yaml
- check-added-large-files (1000 KB cap)

⚠️ **Do NOT run `pre-commit run --all-files`** — many files don't conform yet and you'll touch 100+ unrelated files. Target specific files: `pre-commit run --files file1 file2 ...`.

### 19.5 `.github/workflows/`

- `deploy-docs.yml` — builds and deploys the Sphinx docs to GitHub Pages.

### 19.6 `.gitattributes` — Git LFS

`*.npy`, `*.npz`, `*.pkl`, `*.json`, `*.csv`, `*.obj`, `*.stl`, `*.usd*`, `*.gif`, `*.ckpt`, `*.pt`, `*.mp4`, `*.blend*`, `*.fbx`, `*.onnx`, plus `data/**/*.yaml` and `data/yaml_files/*.yaml` are all in LFS.

If you encounter the post-clone warning *"Encountered N files that should have been pointers, but weren't"* for STL meshes (e.g., `H1_2/*.STL`), it's a known LFS-pointer issue for those specific large mesh files and doesn't block code understanding or training.

---

## 20. Gotchas, Conventions, and Tips

### 20.1 Import order

`isaacgym` and `isaaclab.app.AppLauncher` **must be imported before `torch`**. Use `protomotions.utils.simulator_imports.import_simulator_before_torch(name)` at the top of any entry point that may use them. For Newton / Genesis / MuJoCo it's a no-op.

### 20.2 Config persistence

`resolved_configs.pt` is a plain pickle, not a torch tensor file. Use `torch.load(path, weights_only=False)`.

CLI `--overrides` are **saved permanently** into `resolved_configs.pt` on first training run; on resume the experiment file is **not** re-imported.

### 20.3 Robot body-name mappings

`common_naming_to_robot_body_names` values must be **lists**, never single strings — even for one body. The `abstract_names_to_body_names()` helper depends on it.

### 20.4 Reset states are root + DOF only

Never pass `rigid_body_pos/rot/vel` to `_set_simulator_env_state` — the simulator runs FK from the root and DOF state. Use `ResetState`, not `RobotState`, for writes.

### 20.5 Quaternion ordering

COMMON is xyzw. If `simulator_config.w_last == False`, conversion happens in `RobotState.convert_to_common()` / `convert_to_sim()`. The custom `rotations.quat_mul(a, b, w_last)` takes the convention as an argument.

### 20.6 Body / DOF reordering

Each simulator has its own native ordering. The conversion tensors (`body_convert_to_common`, `dof_convert_to_sim`, etc.) are built **once** in `_finalize_setup()` and applied via fancy indexing — no loops.

### 20.7 Friction combine modes

PhysX (IsaacGym, IsaacLab) uses AVERAGE; Newton, MuJoCo use MAX. `convert_friction_for_simulator()` auto-corrects when you swap simulators. For Newton specifically, robot friction is set near zero so the MAX rule makes terrain friction effective.

### 20.8 Anchor body ≠ root body

- **Root** = the free body (pelvis, index 0). Used for global root pose / vel.
- **Anchor** = the IMU body (e.g., G1 `torso_link`, index 16). Used for orientation references that match a physical IMU.

`anchor_rot` and `root_local_ang_vel` reference different bodies. Mixing them silently breaks deployment.

### 20.9 MuJoCo restrictions

CPU-only and `num_envs == 1`. EMA filtering and either implicit PD (position actuators) or explicit PD kernel.

### 20.10 LazyLinear gotcha

All MLPs use `nn.LazyLinear`. Shapes materialize on the first forward, which `BaseAgent.setup()` runs before Fabric wraps the model with DDP. If you write a custom model, make sure a dummy forward runs before distributed wrapping.

### 20.11 Pre-existing lint errors

`protomotions/envs/component_factories.py` has known F822 errors in `__all__`. Leave them unless you're explicitly fixing them.

### 20.12 Pre-commit

Don't `pre-commit run --all-files`. Target files: `pre-commit run --files <a.py> <b.py>` or `pre-commit run` for staged only.

### 20.13 Apache-2.0 license header

Every new `.py` (except `setup.py`) needs the **full** Apache-2.0 header — not an abbreviated SPDX-only stub. The pre-commit hook will add it on first staging.

### 20.14 Sign-off requirement

All commits must be signed off (`git commit -s`) per the DCO in `CONTRIBUTING.md`.

### 20.15 Distributed motion loading

`MotionLib.process_packaged_motion_file_name_multi_gpu(motion_file)` replaces `"slurmrank"` in filenames with the current rank — useful for sharding the BONES dataset (~142K motions) across GPUs.

### 20.16 Reward grace periods

`combine_rewards()` zeros out designated rewards (power, contact-force penalty, etc.) for the first N steps after reset to avoid penalizing the unrealistic teleport-spawn frame. Mark a reward with `static_params={"zero_during_grace_period": True}`.

### 20.17 Reset vs terminate

`reset_buf = True` whenever an episode ends (success, max length, or failure).
`terminate_buf = True` only on **failure**. GAE bootstrapping uses `terminate_buf` to decide whether `V(s')` should be zeroed.

### 20.18 Two-phase simulator init

```python
sim = SimulatorClass(config, robot_config, terrain, device, scene_lib)   # shell
# env builds markers...
sim._initialize_with_markers(markers)                                    # allocates GPU mem
```

Skipping the second call will leave the simulator unusable.

### 20.19 Projectile actor-aliasing fix

PhysX gets confused when many rigid bodies share the same world position. Projectiles must be hidden at per-env z offsets (`env_id * spacing + proj_idx`), not all at the same `hide_z`. See commit `156733c`.

### 20.20 Multi-horizon velocity filtering

`pose_lib.compute_cartesian_velocity(..., velocity_max_horizon=H)` picks the minimum-magnitude finite difference over horizons 1..H per frame/body. This kills mocap jitter (which spikes over short horizons but smooths over longer ones) without losing real motion.

---

## Quick-Start Cheat Sheet

```bash
# Install (IsaacGym example)
pip install -e .
pip install -r requirements_isaacgym.txt

# Train
python protomotions/train_agent.py \
    --robot-name g1 \
    --simulator isaacgym \
    --experiment-path examples/experiments/mimic/mlp.py \
    --experiment-name my_g1_run \
    --motion-file data/motion_for_trackers/g1_bones_seed_mini.pt \
    --num-envs 4096 \
    --batch-size 16384

# Train with overrides (permanently saved)
python protomotions/train_agent.py ... \
    --overrides agent.config.learning_rate=0.0001 env.max_episode_length=1000

# Evaluate pretrained G1
python protomotions/inference_agent.py \
    --checkpoint data/pretrained_models/motion_tracker/g1-bones-deploy/last.ckpt \
    --motion-file data/motion_for_trackers/g1_bones_seed_mini.pt \
    --simulator isaacgym --num-envs 16

# Sim-to-sim test (trained in IsaacGym, evaluated in Newton)
python protomotions/inference_agent.py \
    --checkpoint <...> \
    --motion-file <...> \
    --simulator newton --num-envs 16

# CPU MuJoCo inference (1 env)
python protomotions/inference_agent.py \
    --checkpoint <...> \
    --motion-file <...> \
    --simulator mujoco --num-envs 1

# Export ONNX for real-robot deployment
python deployment/export_bm_tracker_onnx.py \
    --checkpoint data/pretrained_models/motion_tracker/g1-bones-deploy/last.ckpt

# Run a specific test
pytest protomotions/tests/test_newton_simulator_fk.py
```

---

*End of reference.*
