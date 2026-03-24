# VLA Isaac Lab AbsIK Pipeline

Run the **Pi0.5 VLA model** in **Isaac Lab** with the **DiffIK absolute action** controller, using the verl RL framework.

## Repository Info

| Repo | Branch |
|------|--------|
| [**verl**](https://github.com/Miical/verl/tree/yujie/fix-isaac-verl) (this repo) | `yujie/fix-isaac-verl` |
| [**RobotLearningLab**](https://github.com/nvidia-china-sae/RobotLearningLab/tree/yujie/update/2.3.0-verl) | `yujie/update/2.3.0-verl` (commit `b97fa8bb`) |

## Option A: Clean Build (Recommended)

Build from the official Isaac Lab base image. All dependencies are installed by the Dockerfile; repos are mounted from the host at runtime.

### 1. Prepare Host Data

Clone repos:

```bash
git clone https://github.com/Miical/verl.git ~/iCode/RL/verl
cd ~/iCode/RL/verl && git checkout yujie/fix-isaac-verl

git clone https://github.com/nvidia-china-sae/RobotLearningLab.git ~/iCode/RL/RobotLearningLab
cd ~/iCode/RL/RobotLearningLab && git checkout yujie/update/2.3.0-verl
```

Download USD scene assets (not tracked in git):

```bash
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'china-sae-robotics/RobotLearningLab_Dataset',
    repo_type='dataset',
    allow_patterns=['libero/USD/**'],
    local_dir='/tmp/hf_libero'
)
"
cp -r /tmp/hf_libero/libero/USD ~/iCode/RL/RobotLearningLab/benchmarks/datasets/libero/USD
```

Prepare model checkpoint and RL data:

```bash
# PyTorch checkpoint (~13G) — download from HuggingFace:
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'china-sae-robotics/pi05_libero_torch',
    local_dir='$HOME/iDataset/VLA/openpi/checkpoint/pi05_libero_torch'
)
"
# Then fix config.json (see "Fix PyTorch Checkpoint norm_stats" below)

# RL prompt parquet files — already included in this repo:
mkdir -p ~/iDataset/VLA/openpi/libero_rl
cp verl/experimental/vla/libero_rl_train.parquet ~/iDataset/VLA/openpi/libero_rl/train.parquet
cp verl/experimental/vla/libero_rl_test.parquet ~/iDataset/VLA/openpi/libero_rl/test.parquet
```

### 2. Build

```bash
cd ~/iCode/RL/verl/docker
DOCKER_BUILDKIT=1 docker build -f Dockerfile.isaaclab323 -t verl-isaac-vla:latest .
```

The base image `nvcr.io/nvidia/isaac-lab:2.3.0` will be pulled automatically if not already present.

### 3. Run

A convenience script is provided:

```bash
cd ~/iCode/RL/verl/docker && ./run_isaaclab323.sh
```

Or run manually:

```bash
docker run \
    --entrypoint /root/entrypoint.sh \
    -e "ACCEPT_EULA=Y" \
    -it --gpus all \
    -v /dev/shm:/dev/shm:rw \
    -v ~/iCode/RL/verl:/root/code/verl \
    -v ~/iCode/RL/RobotLearningLab:/root/RobotLearningLab \
    -v ~/iDataset/VLA/openpi/checkpoint/pi05_libero_torch:/root/data/pi05_libero_torch \
    -v ~/iDataset/VLA/openpi/libero_rl:/root/data/libero_rl \
    verl-isaac-vla:latest \
    bash
```

Inside the container:

```bash
cd /root/code/verl
bash verl/experimental/vla/run_pi05_libero_sac.sh
```

### Host → Container Mount Mapping

| Host Path | Container Path | Content |
|---|---|---|
| `~/iCode/RL/verl` | `/root/code/verl` | verl framework (branch `yujie/fix-isaac-verl`) |
| `~/iCode/RL/RobotLearningLab` | `/root/RobotLearningLab` | Isaac Lab playground + USD assets (branch `yujie/update/2.3.0-verl`) |
| `~/iDataset/VLA/openpi/checkpoint/pi05_libero_torch` | `/root/data/pi05_libero_torch` | PyTorch model checkpoint (13G) |
| `~/iDataset/VLA/openpi/libero_rl` | `/root/data/libero_rl` | train/test parquet files |

### Dockerfile.isaaclab323 vs Dockerfile.isaaclab230

`Dockerfile.isaaclab323` is based on `Dockerfile.isaaclab230` with the following changes:

| Change | Why |
|--------|-----|
| Removed `COPY RobotLearningLab/` and its `pip install -e` | Repos are mounted at runtime via `-v`, not baked into the image |
| Added VLA pip deps (`diffusers`, `draccus`, `einops`, `wandb`, `lerobot`, etc.) | Required by the PyTorch VLA inference pipeline |
| Added pip re-bootstrap after `pip install --upgrade` | Upgrading `packaging` corrupts `pip._vendor.packaging`; fixed by re-installing pip via `get-pip.py` |
| Pin `numpy==1.26.4` after all pip installs | Later deps pull in numpy 2.x which breaks Isaac Sim's C extensions (pinocchio, etc.) |
| `libgl1-mesa-glx` → `libgl1` | `libgl1-mesa-glx` is unavailable on Ubuntu 24.04 (base image OS) |
| Added `entrypoint.sh` with `ENTRYPOINT`/`CMD` | Symlinks mounted repos and runs `pip install -e` at container start; also patches `torch._vendor.packaging._structures` (broken symlink from packaging upgrade) |

### What the entrypoint does

1. Restores `torch._vendor.packaging._structures.py` if the symlink is broken (caused by packaging upgrade)
2. Symlinks mounted `RobotLearningLab/source/isaaclab_playground` into `/workspace/isaaclab/source/`
3. Runs `pip install -e` (no-deps) for both `verl` and `isaaclab_playground`
4. Executes the user command (`bash` by default)

---

## Option B: Using the Pre-built verl Image

If you already have access to the pre-built image `vemlp-demo-cn-beijing.cr.volces.com/verl/verl-isaac-fix:v0.1`, you can skip the Dockerfile build. This image already has all system and pip dependencies installed.

```bash
docker run --entrypoint bash \
    -e "ACCEPT_EULA=Y" \
    -it --gpus all \
    -v /dev/shm:/dev/shm:rw \
    --rm \
    vemlp-demo-cn-beijing.cr.volces.com/verl/verl-isaac-fix:v0.1
```

This image already has all dependencies, repos, and data pre-installed. Refer to the sections below for repo setup, data preparation, and running instructions.

## Fix PyTorch Checkpoint norm_stats

**This must be done before running anything with PyTorch inference.**

The PyTorch checkpoint (`$HOME/data/pi05_libero_torch/`) was converted from the JAX checkpoint, but the conversion script does not transfer `norm_stats`. The default `config.json` ships with wrong norm_stats (from official OpenPI LIBERO, trained on MuJoCo), causing nonsensical actions.

**Fix:** Replace `config.json` in your PyTorch checkpoint directory with the corrected version shipped in this repo:

```bash
cp verl/experimental/vla/pi05_libero_torch_config.json $HOME/data/pi05_libero_torch/config.json
```

The corrected config contains `state_norm_stats` and `action_norm_stats` extracted from the JAX checkpoint (`$HOME/data/pi05_libero_absik/checkpoint-30000/assets/all_libero_suites/norm_stats.json`).

## Quick Start (inside container)

```bash
cd /root/code/verl
bash verl/experimental/vla/run_pi05_libero_sac.sh
```

Videos are saved to `$HOME/video/`.

## Environment Prerequisites (without Docker)

| Component | Version / Path | Notes |
|-----------|---------------|-------|
| Python | 3.11 | Isaac Sim bundled Python |
| Isaac Sim | `/workspace/isaaclab/_isaac_sim/` | Must be pre-installed |
| CUDA | 12.x (bundled with Isaac Sim) | GPU required (tested on NVIDIA L20) |
| PyTorch | 2.7.0+cu128 | Isaac Sim bundled |
| verl | editable install from this repo | `pip install -e .` |
| RobotLearningLab | `pip install -e` from `source/isaaclab_playground/` | See repo info above |

**Critical:** numpy must be < 2.0. Isaac Lab and verl are incompatible with numpy 2.x.

### Required Data Files

| Path | Description |
|------|-------------|
| `$HOME/data/pi05_libero_torch/` | PyTorch model checkpoint (safetensors + corrected config.json) |
| `$HOME/data/libero_rl/train.parquet` | Prompt metadata for RL rollouts (task_id, state_id, language instruction) |
| `$HOME/data/libero_rl/test.parquet` | Test prompt metadata |

From **RobotLearningLab**:

| Path | Description |
|------|-------------|
| `benchmarks/datasets/libero/config/` | Task definitions (e.g., `libero_object.json`) |
| `benchmarks/datasets/libero/USD/` | USD scene assets — download from [HuggingFace](https://huggingface.co/datasets/china-sae-robotics/RobotLearningLab_Dataset/tree/main/libero/USD) |
| `benchmarks/datasets/libero/assembled_hdf5/` | HDF5 demo files (for `RESET_MODE=hdf5` only) |

## Action Flow

```
Model output (7-dim): [delta_pos(3), delta_axisangle(3), gripper(1)]
         │
         │  delta-to-absolute: actions[:, :6] += state[:, :6]
         ▼
Absolute action (7-dim): [abs_pos(3), abs_axisangle(3), gripper(1)]
         │
         │  axis-angle → quaternion: isaac_env._convert_actions_for_ik_abs()
         ▼
Isaac Lab input (8-dim): [abs_pos(3), quat_wxyz(4), gripper(1)]
         │
         │  DiffIK controller (use_relative_mode=False)
         ▼
Joint torques → Robot motion
```

The model predicts **delta** actions (relative to current state). The pipeline adds the current EEF state to convert to absolute poses before feeding to Isaac Lab's DiffIK controller.

## Environment Reset Modes

Controlled by `RESET_MODE` environment variable (default: `random`).

| Mode | Command | Description |
|------|---------|-------------|
| `random` | `bash run_pi05_libero_sac.sh` | Isaac Lab built-in randomization within narrow pose ranges (~1cm). Preferred for RL training. |
| `hdf5` | `RESET_MODE=hdf5 bash run_pi05_libero_sac.sh` | Load exact initial state from HDF5 demo data. Useful for reproducible evaluation. |

Both modes run a 10-step arm stabilization loop after reset (hold current EEF pose).

## Configuration Reference

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SFT_MODEL_PATH` | `$HOME/data/pi05_libero_torch` | Path to PyTorch model checkpoint |
| `SIM_TYPE` | `isaac` | Simulator type (`isaac` or `libero`) |
| `RESET_MODE` | `random` | Reset mode: `random` or `hdf5` |
| `NUM_ENV` | `8` | Number of parallel environments per worker |
| `MAX_EPISODE_STEPS` | `300` | Max steps per episode |
| `NUM_ACTION_CHUNKS` | `10` | Action chunk size |

### Key Parameters

| Parameter | Value | Constraint |
|-----------|-------|------------|
| `TRAIN_BATCH_SIZE` | 4 | Must satisfy: `TRAIN_BATCH_SIZE * ROLLOUT_N == NUM_ENV_GPUS * NUM_STAGE * NUM_ENV` |
| `ROLLOUT_N` | 8 | Response number per prompt (`== NUM_ENV` for Isaac) |
| `NUM_STAGE` | 2 | Pipeline stages |
| `NUM_ENV_GPUS` | 2 | GPUs for environment workers |
| `NUM_ROLLOUT_GPUS` | 2 | GPUs for rollout workers (`= NUM_GPUS - NUM_ENV_GPUS`) |

## Key Code Changes

All changes are relative to commit `b738b410` (fix: resolve franka can not move and interact with isaac).

### `envs/isaac_env/isaac_env.py`

| Change | Why |
|--------|-----|
| Task name → `Isaac-Libero-Franka-IK-Abs-v0` | Use DiffIK absolute controller instead of OSC |
| Added `_convert_actions_for_ik_abs()` (axis-angle → quaternion) | Isaac Lab IK-Abs expects quaternion input |
| Fixed state extraction: `eef_pos(3) + axisangle(3) + gripper(1)` | Was passing raw quaternion pose without conversion |
| Stabilization loop: hold current EEF pose instead of zero action | Zero = move to origin in absolute mode |
| Added dual reset mode (`hdf5` / `random`) | Configurable initial state strategy |
| OSC backward compatibility (`LIBERO_OSC_TYPE` auto-set) | Won't break if switching back to OSC task |

### `sac/naive_rollout_pi05.py`

| Change | Why |
|--------|-----|
| `output.action[:, :, :6] += raw_state[:, :6]` after `sample_actions()` | Delta-to-absolute conversion for PyTorch inference |

### `fsdp_workers.py`

| Change | Why |
|--------|-----|
| JAX/PyTorch inference path branching | Support both backends |
| `asyncio.get_event_loop()` fallback | Fix crash from uvloop |

### `run_pi05_libero_sac.sh`

| Change | Why |
|--------|-----|
| Process cleanup (`pkill -9`) at script start | Prevent GPU OOM from zombies |
| `SIM_TYPE=isaac`, `RESET_MODE`, JAX config variables | Configurable pipeline |

### `config/rob_sac_trainer.yaml`

| Change | Why |
|--------|-----|
| `task_suite_name`: `libero_spatial` → `libero_object` | Correct task suite |

## Common Pitfalls

### 1. GPU OOM from Zombie Processes

**Symptom:** `Failed to get DOF velocities from backend` during `env.reset()`.

**Fix:** The run script includes aggressive cleanup at the start (`ray stop --force`, `pkill -9`). If problems persist, manually run `nvidia-smi` and kill stale processes.

### 2. Arm Flies to Origin After Reset

**Symptom:** Arm flips upside down during the 10-step stabilization loop after `env.reset()`.

**Cause:** Original MuJoCo code sends zero actions to "hold position". In IK-Abs mode, zero = move to `(0,0,0)`.

**Fix:** Already handled — `_stabilize_arm()` reads current EEF pose and sends it as the hold action.

### 3. Objects Explode on Auto-Reset

**Symptom:** After task completion, objects scatter violently in the environment.

**Cause:** Isaac Lab's auto-reset places objects at origin `(0,0,0)`, and `randomize_object_pose_by_groups` was skipped for partial resets.

**Fix:** Removed the early return guard in `events.py` (RobotLearningLab commit `b97fa8bb`). Does not affect training data (post-completion steps are masked).

### 4. Multi-Env HDF5 Broadcast Error

**Symptom:** `RuntimeError: shape [1, 3] doesn't match broadcast shape [8, 3]` during `env.reset_to()`.

**Fix:** `_expand_state()` recursively expands all tensors from batch dim 1 to `num_envs`. Already implemented.

### 5. Light Intensity Intermittently Wrong

**Symptom:** `DomeLightCfg(intensity=200)` is set and prints correctly, but video still looks overexposed.

**Workaround:** Re-run the pipeline. Isaac Sim may cache scene state internally. Clearing `__pycache__` under `isaaclab_playground` may help:
```bash
find /root/RobotLearningLab/source/isaaclab_playground -type d -name __pycache__ -exec rm -rf {} +
```

### 6. asyncio Event Loop Crash

**Symptom:** `RuntimeError: There is no current event loop in thread 'MainThread'`

**Cause:** `uvloop` (transitive dependency) changes asyncio behavior. If installed, run `pip uninstall uvloop`.

**Fix:** Already handled — try/except fallback in `fsdp_workers.py` and `env_loop.py`.

---

## Appendix: JAX Inference Path (Backup)

The JAX path was developed first for validation. It is **rollout-only** (no RL training) due to GPU memory constraints.

### Setup

Install JAX dependencies:
```bash
pip install -r verl/experimental/vla/requirements_isaac_jax.txt
```

**Version constraints:** `jax==0.5.3`, `jaxlib==0.5.3`, `jax-cuda12-pjrt==0.5.3`, `jax-cuda12-plugin==0.5.3`. Mismatched versions cause silent GPU failures.

Clone OpenPI source:
```bash
git clone https://github.com/Physical-Intelligence/openpi.git /root/openpi
cd /root/openpi && git checkout e6b0441
```

Download JAX checkpoint from `https://huggingface.co/china-sae-robotics/pi05_libero_isaaclab_absik-ckpt30000/tree/main` to `$HOME/data/pi05_libero_absik/checkpoint-30000/`.

### Run

```bash
cd /root/code/verl
USE_JAX_INFERENCE=true bash verl/experimental/vla/run_pi05_libero_sac.sh
```

### JAX-Specific Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_JAX_INFERENCE` | `false` | Set to `true` for JAX path |
| `JAX_CHECKPOINT_DIR` | `$HOME/data/pi05_libero_absik/checkpoint-30000` | JAX checkpoint path |
| `JAX_CONFIG_NAME` | `pi05_libero` | OpenPI config name |

### Limitation: Rollout-Only (No Training)

The JAX model (~6.2 GB) + PyTorch FSDP model (~9.7 GB) exceed L20's 44 GB VRAM when both are on GPU. JAX mode skips SAC critic and training, returning dummy zeros. Sufficient for evaluating inference quality but not for RL training. Use the PyTorch path for training.

### JAX-Specific Files

| File | Purpose |
|------|---------|
| `models/openpi_jax/openpi_jax_policy.py` | JAX inference wrapper, delta-to-absolute conversion |
| `sac/naive_rollout_jax.py` | JAX rollout class (`PI0JaxRolloutRob`) |
| `requirements_isaac_jax.txt` | Pinned JAX dependency versions |
