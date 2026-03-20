# VLA Isaac Lab AbsIK Pipeline

This document records all modifications made to integrate the **OpenPI JAX checkpoint (`pi05_libero_absik`)** into the verl VLA pipeline running in **Isaac Lab** with the **DiffIK absolute action** controller.

## Quick Start

To use PyTorch inference instead (requires converted safetensors checkpoint):

```bash
cd /root/code/verl_new
bash verl/experimental/vla/run_pi05_libero_sac.sh
```

For reference, JAX model based inference pipeline is established. To run, you will need to first download JAX model from https://huggingface.co/china-sae-robotics/pi05_libero_isaaclab_absik-ckpt30000/tree/main, then use the follow command to run:

```bash
cd /root/code/verl
USE_JAX_INFERENCE=true bash verl/experimental/vla/run_pi05_libero_sac.sh
```

## Environment Setup

### Prerequisites

| Component | Version / Path | Notes |
|-----------|---------------|-------|
| Python | 3.11 | Isaac Sim bundled Python |
| Isaac Sim | `/workspace/isaaclab/_isaac_sim/` | Must be pre-installed |
| CUDA | 12.x (bundled with Isaac Sim) | GPU required (tested on NVIDIA L20) |
| PyTorch | 2.7.0+cu128 | Isaac Sim bundled |
| verl | editable install from this repo | `pip install -e .` |

### JAX Dependencies (for JAX inference path only)

```bash
pip install -r verl/experimental/vla/requirements_isaac_jax.txt
```

**Critical version constraints:**

- **numpy must be < 2.0**: Isaac Lab, verl, and many other packages are incompatible with numpy 2.x. JAX 0.5.3 works with numpy 1.26.x despite not officially advertising it.
- **jax-cuda12-pjrt and jax-cuda12-plugin must match jaxlib version (0.5.3)**: The default pip resolution may install version 0.9.x which is incompatible with jaxlib 0.5.3 and will cause silent GPU failures.
- **Do NOT install uvloop**: It was sometimes pulled in as a transitive dependency and breaks `asyncio.get_event_loop()` in Ray workers. If accidentally installed, run `pip uninstall uvloop`.

### LD_LIBRARY_PATH for JAX CUDA

JAX needs to find CUDA/cuDNN libraries from the Isaac Sim bundle. The run script sets this automatically, but if running manually:

```bash
NVIDIA_BASE=/workspace/isaaclab/_isaac_sim/exts/omni.isaac.ml_archive/pip_prebundle/nvidia
CUDNN_LIB=/workspace/isaaclab/_isaac_sim/kit/python/lib/python3.11/site-packages/nvidia/cudnn/lib
CUDA_LIBS=""
for d in $NVIDIA_BASE/*/lib; do CUDA_LIBS="$d:$CUDA_LIBS"; done
export LD_LIBRARY_PATH="$CUDA_LIBS$CUDNN_LIB:$LD_LIBRARY_PATH"
```

### OpenPI Source Code (for JAX inference path only)

The JAX inference path requires the OpenPI source code (not installed as a package, just source on PYTHONPATH):

```bash
git clone https://github.com/Physical-Intelligence/openpi.git /root/openpi
cd /root/openpi && git checkout e6b0441
```

Verified working commit: `e6b0441` (Implement gsutil download for GCS URLs (#901)). No modifications to OpenPI source are needed. The run script adds it to PYTHONPATH automatically.

### Required Data Files

| Path | Description |
|------|-------------|
| `$HOME/data/pi05_libero_absik/checkpoint-30000/` | OpenPI JAX checkpoint (Orbax OCDBT format) |
| `$HOME/data/pi05_libero_torch/` | PyTorch converted checkpoint (safetensors) |
| `$HOME/data/libero_rl/train.parquet` | Training data |
| `$HOME/data/libero_rl/test.parquet` | Test data |
| HDF5 demo files (see below) | For reproducible environment reset |

HDF5 demo files are searched in these paths (first match wins):

- `/root/data/IsaacLabPlayGround_Dataset/libero/assembled_hdf5/`
- `/root/RobotLearningLab/benchmarks/datasets/libero/assembled_hdf5/`

LIBERO config and USD assets:

- `/root/RobotLearningLab/benchmarks/datasets/libero/config`
- `/root/RobotLearningLab/benchmarks/datasets/libero/USD`

## RobotLearningLab Dependency

### Repository and Branch

| Item | Value |
|------|-------|
| Repo | `https://github.com/nvidia-china-sae/RobotLearningLab` |
| Branch | `yujie/update/2.3.0-verl` |

The pipeline depends on `isaaclab_playground` (installed via `pip install -e` from `source/isaaclab_playground/`) and the `isaaclab` framework (from `source/isaaclab/`).

### Files Used at Runtime

**isaaclab_playground code (8 key files):**

| File | Purpose |
|------|---------|
| `tasks/manipulation/libero/config/franka/__init__.py` | Gym registration for `Isaac-Libero-Franka-IK-Abs-v0` |
| `tasks/manipulation/libero/config/franka/franka_libero_env_cfg.py` | Scene, controller, light, observation, termination configs |
| `tasks/manipulation/libero/mdp/events.py` | Object pose randomization on env reset |
| `tasks/manipulation/libero/mdp/observations.py` | Contact force, object grasped observations |
| `tasks/manipulation/libero/mdp/terminations.py` | `libero_goals_reached` task success check |
| `assets/robots/franka.py` | `FRANKA_PANDA_LIBERO_HIGH_PD_CFG` robot config |
| `utils/decorators.py` | `@subtask_termination` decorator |

**Data files:**

| Path | Purpose |
|------|---------|
| `benchmarks/datasets/libero/config/libero_object.json` | Task definitions (workspace, objects, regions, goals) |
| `benchmarks/datasets/libero/USD/` | USD scene assets (floor, objects) |
| `benchmarks/datasets/libero/assembled_hdf5/` | HDF5 demo files for fixed-state reset |

### Modifications in `b97fa8bb` (vs upstream `update/2.3.0`)

**1. Wrist camera observation (`franka_libero_env_cfg.py`)**

The upstream code creates the `eye_in_hand_cam` CameraCfg (hardware) but does not register it as an observation. The commit adds a `wrist_image` ObsTerm so the observation manager collects wrist RGB images for Pi0 input.

**2. Partial env reset fix (`events.py`)**

Removed the `len(env_list) != num_envs` early return in `randomize_object_pose_by_groups`. Without this fix, when individual environments complete tasks and auto-reset mid-rollout, objects are placed at origin (0,0,0) and explode due to interpenetration. See [Auto-Reset Object Explosion](#6-auto-reset-object-explosion-mid-rollout) below.

**3. Light intensity (`franka_libero_env_cfg.py`)**

Changed DomeLight intensity from 1000 to 200 across all four scene types (KitchenTable, LivingRoomTable, Floor, StudyTable) for better visual quality in saved videos.

### Older Commit Pitfalls

If using an older RobotLearningLab commit (before `b97fa8bb`), the following issues will occur:

| Issue | Symptom | Fix |
|-------|---------|-----|
| Missing `wrist_image` ObsTerm | Pi0 receives no wrist camera image; may silently use zeros or crash | Add `wrist_image = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("eye_in_hand_cam"), "data_type": "rgb"})` to `ObservationsCfg.policy` |
| Partial reset early return | Objects fly off screen when individual envs auto-reset after task completion | Remove the `if len(env_list) != env.cfg.scene.num_envs: return` guard in `randomize_object_pose_by_groups` |
| Light intensity 1000 | Overexposed/washed-out video output | Change `DomeLightCfg(intensity=1000.0)` to `intensity=200.0` in all SceneCfg classes |

## Architecture

### Two Inference Paths

```
run_pi05_libero_sac.sh
  └─ main_sac.py
       └─ fsdp_workers.py (RobActorRolloutRefWorker)
            ├─ USE_JAX_INFERENCE=true
            │    └─ PI0JaxRolloutRob          [sac/naive_rollout_jax.py]
            │         └─ OpenPIJaxPolicy      [models/openpi_jax/openpi_jax_policy.py]
            │              └─ openpi Policy    (JAX checkpoint, Orbax, diffusion ODE)
            │
            └─ USE_JAX_INFERENCE=false (default)
                 └─ PI0RolloutRob             [sac/naive_rollout_pi05.py]
                      └─ PI0ForActionPrediction [models/pi0_torch/modeling_pi0_torch.py]
                           └─ PI0Model         (PyTorch safetensors, diffusion ODE)
```

### Action Flow: Model Output → Isaac Lab

```
Model output (7-dim): [delta_pos(3), delta_axisangle(3), gripper(1)]
         │
         │  delta-to-absolute conversion (actions[:, :6] += state[:, :6])
         ▼
Absolute action (7-dim): [abs_pos(3), abs_axisangle(3), gripper(1)]
         │
         │  axis-angle → quaternion conversion (isaac_env._convert_actions_for_ik_abs)
         ▼
Isaac Lab input (8-dim): [abs_pos(3), quat_wxyz(4), gripper(1)]
         │
         │  DiffIK controller (use_relative_mode=False)
         ▼
Joint torques → Robot motion
```

## Code Changes Summary

All changes are relative to commit `b738b410` (fix: resolve franka can not move and interact with isaac).

### New Files

| File | Purpose |
|------|---------|
| `models/openpi_jax/__init__.py` | Package init, exports `OpenPIJaxPolicy` |
| `models/openpi_jax/openpi_jax_policy.py` | JAX inference wrapper: loads OpenPI checkpoint, runs diffusion inference, applies delta-to-absolute conversion |
| `sac/naive_rollout_jax.py` | SAC rollout class for JAX path (`PI0JaxRolloutRob`): bridges verl DataProto with JAX policy |

### Modified Files

| File | Change | Purpose |
|------|--------|---------|
| `run_pi05_libero_sac.sh` | Added process cleanup (ray/omni/isaac pkill) at script start | Prevent GPU OOM from zombie processes |
| | Added JAX config variables (`USE_JAX_INFERENCE`, `JAX_CHECKPOINT_DIR`, etc.) | Configurable JAX inference |
| | Added `LD_LIBRARY_PATH` and `PYTHONPATH` for JAX/OpenPI | JAX CUDA and OpenPI source access |
| | Changed `SIM_TYPE` default from `libero` to `isaac` | Use Isaac Sim by default |
| | Changed `NUM_ENV=1`, `TRAIN_BATCH_SIZE=4`, `MAX_EPISODE_STEPS=210` | Single-env debug config, reduced rollout length |
| | Added Hydra overrides for JAX config passthrough | Pass JAX settings to workers |
| `fsdp_workers.py` | Added JAX/PyTorch inference path branching based on `use_jax_inference` | Load either JAX or PyTorch policy |
| | Strip JAX-specific keys before passing config to HFModelConfig | Prevent unknown config errors |
| | Added `asyncio.get_event_loop()` RuntimeError fallback | Fix crash when uvloop is installed |
| `envs/isaac_env/isaac_env.py` | Changed task name from `Isaac-Libero-Franka-OscPose-v0` to `Isaac-Libero-Franka-IK-Abs-v0` | Use DiffIK absolute controller |
| | Removed `LIBERO_OSC_TYPE=pose_rel` and `osc_type` assertion | Not needed for IK-Abs |
| | Added `_quat2axisangle()` / `_axisangle2quat()` / `_convert_actions_for_ik_abs()` | Convert between axis-angle (model) and quaternion (Isaac Lab) |
| | Fixed state observation: extract `eef_pos(3) + axisangle(3) + gripper(1)` = 7-dim | Was incorrectly passing raw 7-dim quaternion pose without conversion |
| | Added action logging (first 3 steps) for VLA absolute action and EEF pose | Debug verification |
| | Added HDF5-based `_find_hdf5_file()` / `_load_hdf5_initial_state()` | Reproducible environment reset from demo data |
| | Modified `reset_envs_to_state_ids()` to use `env.reset_to()` with HDF5 initial state | Correct initial arm position |
| | Fixed stabilization loop: use current EEF pose as hold action for IK-Abs | Prevent arm from drifting to origin during stabilization |
| `config/rob_sac_trainer.yaml` | Changed `task_suite_name` from `libero_spatial` to `libero_object` | Correct task suite |
| `env_loop.py` | Added `asyncio.get_event_loop()` RuntimeError fallback | Same uvloop fix |
| `workers/env/env_worker.py` | Minor comment fix | Typo |

## Key Fix: Delta-to-Absolute Action Conversion

### The Problem

The `pi05_libero_absik` model's `norm_stats.json` reveals that action statistics have very small ranges centered around zero (e.g., position mean ~0.005), while state statistics have absolute ranges (e.g., position mean ~0.45). This means the model outputs **delta actions** (relative to current state), not absolute actions.

However, the Isaac Lab `IK-Abs` controller expects **absolute end-effector poses**. Feeding raw delta outputs causes the arm to move toward the origin and flip upside down.

### The Fix

After model inference produces unnormalized actions, add the current robot state to convert deltas to absolute:

```python
# In openpi_jax_policy.py infer_from_dataproto():
for i in range(actions_np.shape[0]):
    actions_np[i, :, :6] += states[i, :6]
```

Only the first 6 dimensions (position + axis-angle rotation) are converted; the gripper dimension (index 6) is left as-is because it's already absolute (0 or 1).

### Why This Happens

In the OpenPI training pipeline, the `AbsoluteActions` transform (in `openpi/transforms.py`) handles this conversion automatically. When `extra_delta_transform=False` (as in `pi05_libero` config), it means the training data already contains absolute actions, but the normalization statistics are computed on the **delta** between consecutive actions, making the model effectively predict deltas. The `AbsoluteActions` output transform adds the state back during inference.

In our pipeline, we bypass OpenPI's transform chain and apply the conversion manually.

## Environment Reset Modes

The pipeline supports two ways to initialize environment state at the start of each rollout, controlled by the `RESET_MODE` environment variable (passed via `env.train.reset_mode` Hydra config).

### `hdf5` — Fixed Initial State from Demo Data (default)

```bash
RESET_MODE=hdf5 bash verl/experimental/vla/run_pi05_libero_sac.sh
```

Loads the exact robot and object poses from an HDF5 demo file and calls `env.reset_to()` to place the scene into that state. All parallel environments receive the same initial state (broadcast via `_expand_state`). This mode is useful for:

- **Reproducible evaluation**: every run starts from the same configuration, making success rate comparisons fair.
- **Debugging**: eliminates initialization variance as a failure source.

If the HDF5 file is not found, it falls back to the `random` behavior automatically.

### `random` — Isaac Lab Built-in Randomization

```bash
RESET_MODE=random bash verl/experimental/vla/run_pi05_libero_sac.sh
```

Uses `env.reset()` only, which triggers Isaac Lab's `EventTerm` pipeline:

1. `reset_scene_to_default` — resets robot joints to the default pose.
2. `randomize_object_pose_by_groups` — places each object within a narrow pose range defined in the task config (typically ~1 cm variation per axis).

This mode is preferred for **RL training** because the small initial-state variation improves policy robustness. Note that the pose ranges are intentionally narrow (aligned with LIBERO demo distributions), so the randomization is minor — objects land in roughly the same spot each time, with slight perturbation.

### Common to Both Modes

After the initial state is established (by either method), the pipeline runs a 10-step **arm stabilization loop** that holds the current EEF pose. This prevents the arm from drifting during the physics settling phase. See [Arm Stabilization After Reset](#7-arm-stabilization-after-reset).

## Known Limitations

### 1. JAX Mode is Rollout-Only (No RL Training)

The JAX inference path (`USE_JAX_INFERENCE=true`) only supports rollout (action generation). SAC critic evaluation and RL training are **skipped** — `naive_rollout_jax.py` returns dummy zero values for `critic_value`, `images`, `lang_tokens`, etc.

**Why:** The JAX model (~6.2 GB) and PyTorch FSDP model (~9.7 GB) both reside on the same GPU. When FSDP attempts to unshard parameters for critic evaluation, the combined memory exceeds the L20's 44 GB VRAM, causing CUDA OOM:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1006.00 MiB.
GPU 0 has a total capacity of 44.42 GiB of which 627.75 MiB is free.
```

**Workaround:** The JAX rollout class (`PI0JaxRolloutRob`) bypasses all PyTorch model forward passes. This is sufficient for evaluating inference quality (generating videos), but cannot do SAC training. For RL training, use the PyTorch inference path.

## Common Pitfalls

### 1. GPU OOM from Zombie Processes

**Symptom:** `Failed to get DOF velocities from backend` during `env.reset()`.

**Cause:** Previous Ray/Isaac Sim processes didn't fully terminate (Ctrl+C is not sufficient), consuming GPU memory.

**Fix:** The run script now includes aggressive cleanup at the start:
```bash
ray stop --force
pkill -9 -f "ray::"
pkill -9 -f "omni\."
pkill -9 -f "isaac_sim"
```

### 2. Logger Visibility in Ray Workers

**Symptom:** `logging.getLogger(__name__)` messages from model code don't appear in Ray worker output.

**Cause:** Ray workers have their own logging configuration that may filter module-level loggers.

**Workaround:** Use `print(..., flush=True)` for critical debug messages in Ray workers, or use `logging.getLogger(__file__)`.

### 3. asyncio Event Loop in Ray

**Symptom:** `RuntimeError: There is no current event loop in thread 'MainThread'`

**Cause:** Installing JAX dependencies may pull in `uvloop`, which changes asyncio's event loop behavior.

**Fix:** Added try/except fallback in `fsdp_workers.py` and `env_loop.py`:
```python
try:
    loop = asyncio.get_event_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
```

### 4. PyTorch Checkpoint norm_stats Must Match JAX Checkpoint

The PyTorch checkpoint (`$HOME/data/pi05_libero_torch/`) was converted from a JAX checkpoint using the [giga-models conversion script](https://github.com/open-gigaai/giga-models/tree/main/projects/vla/pi0). The conversion script only handles model weights — it does **not** transfer `norm_stats`.

**Symptom:** PyTorch inference produces nonsensical actions even though model weights are correct.

**Cause:** The default `config.json` shipped with the converted checkpoint contained norm_stats copied from the official OpenPI LIBERO config (likely trained on MuJoCo OSC-relative actions). These do not match the `pi05_libero_absik` model which was trained on Isaac Lab absolute actions.

| | Wrong (original) action mean pos | Correct (absik) action mean pos |
|---|---|---|
| dim 0-2 | [0.027, 0.089, -0.100] | [0.005, 0.006, -0.005] |

**Fix:** Replace `state_norm_stats` and `action_norm_stats` in `config.json` with values from the JAX checkpoint:

```
Source: $HOME/data/pi05_libero_absik/checkpoint-30000/assets/all_libero_suites/norm_stats.json
Target: $HOME/data/pi05_libero_torch/config.json
```

The original config was backed up as `config_backup.json`.

### 5. Multi-Env Broadcast Error on HDF5 Reset

**Symptom:** `RuntimeError: output with shape [1, 3] doesn't match the broadcast shape [8, 3]` during `env.reset_to()`.

**Cause:** HDF5 initial state has batch dimension 1, but `interactive_scene.py` expects batch dimension matching `num_envs`. The `initial_state` dict is deeply nested (`state["articulation"]["robot"]["root_pose"]`), requiring recursive expansion.

**Fix:** `isaac_env.py` uses `_expand_state()` to recursively expand all tensors with batch dim 1 to `num_envs`:
```python
@staticmethod
def _expand_state(state, num_envs):
    if isinstance(state, torch.Tensor):
        if state.shape[0] == 1:
            return state.expand(num_envs, *state.shape[1:]).contiguous()
        return state
    if isinstance(state, dict):
        return {k: IsaacEnv._expand_state(v, num_envs) for k, v in state.items()}
    return state
```

### 6. Auto-Reset Object Explosion Mid-Rollout

**Symptom:** After a task succeeds, objects in that environment scatter violently ("explode") before the next observation.

**Cause:** Isaac Lab auto-resets completed environments inside `env.step()`. The reset triggers two events in sequence:
1. `reset_scene_to_default` — moves all objects to their `init_state` default position, which is `(0, 0, 0)` for dynamically-added LIBERO objects
2. `randomize_object_pose_by_groups` — was supposed to scatter objects to correct positions, but had an early return: `if len(env_ids) != num_envs: return` that skipped randomization for partial resets

Result: all objects pile up at origin → PhysX detects interpenetration → objects launch at high velocity.

**Fix:** Remove the early return guard in `events.py` (done in RobotLearningLab commit `b97fa8bb`). This allows object randomization even when only a subset of environments reset.

**Impact:** The explosion only affects post-completion steps which are masked out during training (via `compute_response_mask`), so it does not corrupt training data. The fix is primarily for visual quality in saved videos.

### 7. Arm Stabilization After Reset (Zero Action ≠ Hold Position)

**Symptom:** After `env.reset()`, the arm immediately flips upside down or crashes into the table during the 10-step stabilization loop.

**Cause:** The original code (designed for MuJoCo with OSC relative control) sends **zero actions** during the 10 stabilization steps after reset. In relative mode, zero means "don't move" — this is correct. However, in Isaac Lab's IK-Abs (absolute) mode, zero means "move to position (0,0,0) with identity rotation", which causes the arm to slam toward the world origin.

**Fix:** Detect whether the task uses absolute actions (`IK-Abs`) and, if so, read the current EEF pose and send it back as the hold action:
```python
if is_abs_action:
    eef_pose = raw_obs["policy"]["eef_pose"].to(self.device)
    gripper_pos = raw_obs["policy"]["gripper_pos"].to(self.device)
    hold_action = torch.cat([eef_pose, gripper_pos[..., 0:1]], dim=-1)
```
For non-absolute tasks (e.g., MuJoCo OSC-relative), the original zero-action logic is preserved.

### 8. Light Intensity Change Not Taking Effect

**Symptom:** `DomeLightCfg(intensity=200.0)` is set correctly in all four `SceneCfg` classes, and `__post_init__` debug prints confirm `intensity=200`, yet the rendered video still appears as bright as `intensity=1000`.

**Cause:** Isaac Sim / Kit may cache scene lighting state internally (e.g., Fabric/USD stage cache). The Python-level config is correct, but the renderer sometimes uses a stale value from a previous run. This is non-deterministic — it happens intermittently, especially when switching between JAX and PyTorch inference modes or restarting without a full process cleanup.

**Workaround:** Re-run the pipeline. In most cases, the correct intensity is picked up on the next launch. Ensure all Isaac Sim / Ray processes are fully killed before restarting (the run script's `pkill -9` block helps). Clearing `__pycache__` directories under `isaaclab_playground` may also help:
```bash
find /root/RobotLearningLab/source/isaaclab_playground -type d -name __pycache__ -exec rm -rf {} +
```

## Configuration Reference

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_JAX_INFERENCE` | `false` | Set to `true` to use JAX checkpoint inference |
| `JAX_CHECKPOINT_DIR` | `$HOME/data/pi05_libero_absik/checkpoint-30000` | Path to OpenPI JAX checkpoint |
| `JAX_CONFIG_NAME` | `pi05_libero` | OpenPI training config name |
| `SFT_MODEL_PATH` | `$HOME/data/pi05_libero_torch` | Path to PyTorch model (used for tokenizer even in JAX mode) |
| `SIM_TYPE` | `isaac` | Simulator type (`isaac` or `libero`) |
| `RESET_MODE` | `random` | Environment reset mode: `random` (Isaac Lab randomization) or `hdf5` (fixed demo state) |
| `NUM_ENV` | `1` | Number of parallel environments per worker |
| `MAX_EPISODE_STEPS` | `210` | Max steps per episode (rollout iterations = this / `NUM_ACTION_CHUNKS`) |
| `NUM_ACTION_CHUNKS` | `10` | Action chunk size |

### Key Parameters

| Parameter | Value | Constraint |
|-----------|-------|------------|
| `TRAIN_BATCH_SIZE` | 4 | Must satisfy: `TRAIN_BATCH_SIZE * ROLLOUT_N == NUM_ENV_GPUS * NUM_STAGE * NUM_ENV` |
| `ROLLOUT_N` | 1 | Response number per prompt |
| `NUM_STAGE` | 2 | Pipeline stages |
| `NUM_ENV_GPUS` | 2 | GPUs for environment workers |
| `NUM_ROLLOUT_GPUS` | 2 | GPUs for rollout workers (= `NUM_GPUS - NUM_ENV_GPUS`) |
