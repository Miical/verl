#!/usr/bin/env bash
set -euo pipefail
set -x

# Critic-only-ish debug script for PI0.5 + LIBERO SAC.
# Goal: minimize actor updates and focus on critic learning behavior.
#
# Usage:
#   bash verl/experimental/vla/run_pi05_libero_sac_critic_debug.sh
#
# Optional env overrides:
#   SFT_MODEL_PATH=/path/to/pi05_ckpt
#   TRAIN_FILES=/path/to/train.parquet
#   VAL_FILES=/path/to/test.parquet
#   VIDEO_OUTPUT=/tmp/critic_debug_videos
#   OUTPUT_DIR=/tmp/critic_debug_ckpt

TRAIN_FILES=${TRAIN_FILES:-"$HOME/data/libero_rl/train.parquet"}
VAL_FILES=${VAL_FILES:-"$HOME/data/libero_rl/test.parquet"}

OUTPUT_DIR=${OUTPUT_DIR:-"$HOME/models/vla_libero_sac_critic_debug"}
VIDEO_OUTPUT=${VIDEO_OUTPUT:-"$HOME/video_critic_debug"}
SFT_MODEL_PATH=${SFT_MODEL_PATH:-"$HOME/data/pi05_libero_torch"}
TOKENIZER_PATH="$SFT_MODEL_PATH"

# ---- Hardware layout (smaller defaults for quick diagnosis) ----
NUM_NODES=${NUM_NODES:-1}
NUM_GPUS=${NUM_GPUS:-2}
NUM_ENV_GPUS=${NUM_ENV_GPUS:-1}
NUM_ROLLOUT_GPUS=${NUM_ROLLOUT_GPUS:-1}

# ---- Rollout/Env ----
SIM_TYPE=${SIM_TYPE:-"libero"}
NUM_STAGE=${NUM_STAGE:-1}
NUM_ENV=${NUM_ENV:-4}
NUM_ACTION_CHUNKS=${NUM_ACTION_CHUNKS:-10}
MAX_EPISODE_STEPS=${MAX_EPISODE_STEPS:-300}

# ---- Data/Batch ----
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-4}
ROLLOUT_N=${ROLLOUT_N:-1}
MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-32}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-8}

# ---- Training horizon ----
TOTAL_EPOCHS=${TOTAL_EPOCHS:-15}
SAVE_FREQ=${SAVE_FREQ:-5}
ROLLOUT_INTERVAL=${ROLLOUT_INTERVAL:-100}

# Key trick for critic diagnosis:
# 1) warmup critic for first 500 global steps
# 2) then update actor every step by default
ACTOR_UPDATE_INTERVAL=${ACTOR_UPDATE_INTERVAL:-1}
CRITIC_LR=${CRITIC_LR:-1e-4}
CRITIC_WARMUP_STEPS=${CRITIC_WARMUP_STEPS:-500}

PROJECT_NAME=${PROJECT_NAME:-"vla_libero_critic_debug"}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"${SIM_TYPE}_sac_critic_focus"}

ISSC_PYTHON="/workspace/isaaclab/_isaac_sim/python.sh"
PYTHON=python
if [ -f "$ISSC_PYTHON" ]; then
    PYTHON=$ISSC_PYTHON
fi

mkdir -p /root/LIBERO/libero/libero/../datasets
mkdir -p "$VIDEO_OUTPUT"
mkdir -p "$OUTPUT_DIR"

gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -n 1)
if echo "$gpu_name" | grep "NVIDIA H"; then
    echo "enable MUJOCO_GL=osmesa in Hopper"
    export MUJOCO_GL=osmesa
fi

export VERL_LOGGING_LEVEL=INFO

$PYTHON -m verl.experimental.vla.main_sac \
    data.train_files="$TRAIN_FILES" \
    data.val_files="$VAL_FILES" \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_batch_size=$VAL_BATCH_SIZE \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    env.train.num_envs=$NUM_ENV \
    data.max_prompt_length=256 \
    data.max_response_length=128 \
    env.rollout.pipeline_stage_num=$NUM_STAGE \
    env.train.simulator_type=$SIM_TYPE \
    env.actor.model.num_action_chunks=$NUM_ACTION_CHUNKS \
    env.actor.model.action_dim=7 \
    env.train.only_eval=False \
    env.train.max_episode_steps=$MAX_EPISODE_STEPS \
    env.train.video_cfg.save_video=True \
    env.train.video_cfg.video_base_dir="$VIDEO_OUTPUT" \
    env.train.seed=42 \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[SiglipEncoderLayer,GemmaDecoderLayerWithExpert] \
    actor_rollout_ref.model.path="$SFT_MODEL_PATH" \
    actor_rollout_ref.model.tokenizer_path="$TOKENIZER_PATH" \
    actor_rollout_ref.rollout.mode=async_envloop \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.actor.optim.warmup_style=constant \
    actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.strategy=fsdp2 \
    critic.strategy=fsdp2 \
    actor_rollout_ref.actor.grad_clip=1 \
    actor_rollout_ref.actor.num_images_in_input=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.trust_remote_code=False \
    +actor_rollout_ref.model.override_config.attn_implementation=eager \
    actor_rollout_ref.actor.entropy_coeff=0. \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.prompt_length=512 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=hf \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    +actor_rollout_ref.algorithm='sac' \
    actor_rollout_ref.actor.critic_lr=$CRITIC_LR \
    actor_rollout_ref.actor.critic_warmup_steps=$CRITIC_WARMUP_STEPS \
    actor_rollout_ref.actor.actor_update_interval=$ACTOR_UPDATE_INTERVAL \
    algorithm.kl_ctrl.kl_coef=0.00 \
    trainer.logger=['console'] \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.default_local_dir="$OUTPUT_DIR" \
    trainer.n_gpus_per_node=$NUM_GPUS \
    +trainer.n_env_gpus_per_node=$NUM_ENV_GPUS \
    +trainer.n_rollout_gpus_per_node=$NUM_ROLLOUT_GPUS \
    trainer.nnodes=$NUM_NODES \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=-1 \
    trainer.total_epochs=$TOTAL_EPOCHS \
    +trainer.rollout_interval=$ROLLOUT_INTERVAL \
    trainer.val_only=False \
    trainer.val_before_train=False
