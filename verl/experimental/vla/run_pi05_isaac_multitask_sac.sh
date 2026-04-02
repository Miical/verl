set -x

# Clean up lingering processes from previous runs to free GPU memory
ray stop --force 2>/dev/null || true
pkill -9 -f "ray::" 2>/dev/null || true
pkill -9 -f "omni\." 2>/dev/null || true
pkill -9 -f "isaac_sim" 2>/dev/null || true
sleep 2

libero_train_path=$HOME/data/libero_rl/train.parquet
libero_test_path=$HOME/data/libero_rl/test.parquet

train_files=$libero_train_path
test_files=$libero_test_path

OUTPUT_DIR=${MLP_MODEL_OUTPUT:-"$HOME/models/vla_libero_multitask_sac"}
VIDEO_OUTPUT=${MLP_MODEL_OUTPUT:-"$HOME"}/video
SFT_MODEL_PATH=${SFT_MODEL_PATH:-"$HOME/data/pi05_libero_torch"}
TOKENIZER_PATH="$SFT_MODEL_PATH"

# JAX inference config (set USE_JAX_INFERENCE=true to use JAX checkpoint directly)
USE_JAX_INFERENCE=${USE_JAX_INFERENCE:-"false"}
JAX_CHECKPOINT_DIR=${JAX_CHECKPOINT_DIR:-"$HOME/data/pi05_libero_absik/checkpoint-30000"}
JAX_CONFIG_NAME=${JAX_CONFIG_NAME:-"pi05_libero"}
ABSOLUTE_ACTION_MODE=${ABSOLUTE_ACTION_MODE:-"auto"}

# LD_LIBRARY_PATH for JAX CUDA support
NVIDIA_BASE=/workspace/isaaclab/_isaac_sim/exts/omni.isaac.ml_archive/pip_prebundle/nvidia
CUDNN_LIB=/workspace/isaaclab/_isaac_sim/kit/python/lib/python3.11/site-packages/nvidia/cudnn/lib
CUDA_LIBS=""
for d in $NVIDIA_BASE/*/lib; do CUDA_LIBS="$d:$CUDA_LIBS"; done
export LD_LIBRARY_PATH="$CUDA_LIBS$CUDNN_LIB:$LD_LIBRARY_PATH"

# PYTHONPATH for openpi (JAX model source)
export PYTHONPATH="/root/openpi/src:/root/openpi/packages/openpi-client/src:$PYTHONPATH"

# Physical Node Config
NUM_NODES=1                                    # number of nodes
NUM_GPUS=${NUM_GPUS:-8}                        # total number of gpus per node

# Role Config
NUM_ENV_GPUS=2                                 # number of gpus for env workers per node (each runs an Isaac instance)
NUM_ROLLOUT_GPUS=$((NUM_GPUS - NUM_ENV_GPUS))  # number of gpus for FSDP training  [=6]

# Task / Environment Config
TASK_SUITE=${TASK_SUITE:-"libero_spatial"}     # "" = all 40 tasks; "libero_10" = 10 tasks; "libero_10,libero_object" = 20
GROUP_SIZE=${GROUP_SIZE:-18}                   # envs per task per stage (each sim gets GROUP_SIZE*NUM_TASKS envs)
NUM_TASKS=${NUM_TASKS:-10}                     # number of tasks in the suite
NUM_ENV=$((NUM_TASKS * GROUP_SIZE))            # envs per stage per worker [=180]

# Rollout Config (MultiTask Rolling Mode)
TRAIN_BATCH_SIZE=1                             # batch size for dataloaders per step
ROLLOUT_N=$NUM_ENV                             # == NUM_ENV for isaac_multitask rolling mode
NUM_STAGE=2                                    # 2-stage pipeline: overlap VLA inference with env stepping

ROLLOUT_HORIZON=6                              # rolling steps per training iteration
NUM_ACTION_CHUNKS=10                           # env steps per rolling step (action chunk size)
# per iteration: ROLLOUT_HORIZON * NUM_ACTION_CHUNKS = 60 env steps
# total envs across both stages = NUM_ENV * NUM_ENV_GPUS * NUM_STAGE = 180*2*2 = 720
# but effective batch per rollout = NUM_ENV * NUM_ENV_GPUS = 360 (same throughput)

TASK_NAME=${TASK_NAME:-"Isaac-Libero-Franka-Abs-IK-Camera-All-Tasks-v0"}

# Training Config
# Constraints (all must hold):
#   1. GROUP_SIZE % NUM_ENV_GPUS == 0             (GROUP_SIZE splits evenly across sim instances)
#   2. NUM_ENV % NUM_ROLLOUT_GPUS == 0            (DataProto splits evenly across training GPUs)
#   3. (MINI_BATCH_SIZE * NUM_ENV) % (NUM_ROLLOUT_GPUS * MICRO_BATCH_SIZE) == 0
# FSDP normalization: normalized_batch = MINI_BATCH_SIZE * NUM_ENV / NUM_ROLLOUT_GPUS
#   this is the replay pool sample size per GPU; normalized_batch / MICRO_BATCH_SIZE = grad accum steps
MINI_BATCH_SIZE=1024                            # replay pool sample size
MICRO_BATCH_SIZE=128                             # reduced from 256 to avoid OOM during actor update

SIM_TYPE="isaac_multitask"
PROJECT_NAME="vla_libero_multitask_RL"
EXPERIMENT_NAME="${SIM_TYPE}_sac_rolling"

ISSC_PYTHON="/workspace/isaaclab/_isaac_sim/python.sh"
PYTHON=python
if [ -f "$ISSC_PYTHON" ]; then
    PYTHON=$ISSC_PYTHON
fi

# avoiding warnings
mkdir -p /root/LIBERO/libero/libero/../datasets 2>/dev/null || true
gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -n 1)

# force osmesa in Hopper
if echo "$gpu_name" | grep "NVIDIA H"; then
    echo "enable MUJOCO_GL=osmesa in Hopper"
    export MUJOCO_GL=osmesa
fi

export VERL_LOGGING_LEVEL=INFO

$PYTHON -m verl.experimental.vla.main_sac \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_batch_size=4 \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    env.train.num_envs=$NUM_ENV \
    data.max_prompt_length=256 \
    data.max_response_length=128 \
    env.rollout.pipeline_stage_num=$NUM_STAGE \
    env.train.simulator_type=$SIM_TYPE \
    env.train.rollout_horizon=$ROLLOUT_HORIZON \
    env.actor.model.num_action_chunks=$NUM_ACTION_CHUNKS \
    env.actor.model.action_dim=7 \
    env.train.only_eval=False \
    env.train.max_episode_steps=$MAX_EPISODE_STEPS \
    env.train.video_cfg.save_video=True \
    env.train.video_cfg.video_base_dir=${VIDEO_OUTPUT} \
    env.train.seed=42 \
    +env.train.task_name=$TASK_NAME \
    +env.train.group_size=$GROUP_SIZE \
    env.train.task_suite_name="$TASK_SUITE" \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[SiglipEncoderLayer,GemmaDecoderLayerWithExpert] \
    actor_rollout_ref.model.path=$SFT_MODEL_PATH \
    actor_rollout_ref.model.tokenizer_path=$TOKENIZER_PATH \
    actor_rollout_ref.rollout.mode=async_envloop \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.actor.optim.warmup_style=constant \
    actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.strategy=fsdp2 \
    critic.strategy=fsdp2 \
    actor_rollout_ref.actor.grad_clip=1 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.num_images_in_input=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.trust_remote_code=False \
    +actor_rollout_ref.model.override_config.attn_implementation=eager \
    actor_rollout_ref.actor.entropy_coeff=0. \
    actor_rollout_ref.rollout.temperature=1.6 \
    actor_rollout_ref.rollout.prompt_length=512 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=hf \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    +actor_rollout_ref.model.use_jax_inference=$USE_JAX_INFERENCE \
    +actor_rollout_ref.model.jax_checkpoint_dir=$JAX_CHECKPOINT_DIR \
    +actor_rollout_ref.model.jax_config_name=$JAX_CONFIG_NAME \
    +actor_rollout_ref.model.absolute_action_mode=$ABSOLUTE_ACTION_MODE \
    +actor_rollout_ref.algorithm='sac' \
    algorithm.kl_ctrl.kl_coef=0.00 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$OUTPUT_DIR \
    trainer.resume_mode=auto \
    trainer.n_gpus_per_node=$NUM_GPUS \
    +trainer.n_env_gpus_per_node=$NUM_ENV_GPUS \
    +trainer.n_rollout_gpus_per_node=$NUM_ROLLOUT_GPUS \
    +trainer.rollout_interval=10 \
    trainer.nnodes=$NUM_NODES \
    trainer.save_freq=100 \
    trainer.test_freq=-1 \
    trainer.total_epochs=100 \
    trainer.val_only=False \
    trainer.val_before_train=False
