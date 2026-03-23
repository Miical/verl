#!/usr/bin/env bash
set -ex
# =========================
# 1) online single-task parquet
# =========================
TASK_SUITE_NAME=${TASK_SUITE_NAME:-libero_10}
SINGLE_TASK_ID=${SINGLE_TASK_ID:-3}
ONLINE_DATA_DIR=${ONLINE_DATA_DIR:-$HOME/data/libero_10_task3/libero_10_task3}

libero_train_path=${TRAIN_FILES:-${ONLINE_DATA_DIR}/train.parquet}
libero_test_path=${TEST_FILES:-${ONLINE_DATA_DIR}/test.parquet}

# =========================
# 2) offline RLPD dataset
# IMPORTANT:
#   data.rlpd_files must be a LeRobot dataset ROOT directory.
#   Raw LIBERO *_demo.hdf5 cannot be passed directly to main_sac.py.
# =========================
RLPD_FILES=${RLPD_FILES:-/shared_disk/users/angen.ye/code/hil-serl/datasets/LIBERO-dataset/libero_10_task3}
RLPD_BATCH_SIZE=${RLPD_BATCH_SIZE:-32}
ENABLE_RLPD=${ENABLE_RLPD:-False}




EXPERIMENT_NAME=${EXPERIMENT_NAME:-libero10_task3_rlpd_smoke_bad_head_train_no_rlpd2}
OUTPUT_DIR=${OUTPUT_DIR:-/shared_disk/users/angen.ye/code/hil-serl/model/verl_fintune_model/libero10_task3_rlpd_smoke_bad_head_train_no_rlpd_2}
VIDEO_OUTPUT="${OUTPUT_DIR}/video"
# SFT_MODEL_PATH=${SFT_MODEL_PATH:-/shared_disk/users/angen.ye/code/hil-serl/model/torch_pi05_base}
SFT_MODEL_PATH="/shared_disk/users/angen.ye/code/hil-serl/model/pi05_libero_torch_bad_head_v3"
# SFT_MODEL_PATH="/shared_disk/users/angen.ye/code/hil-serl/model/pi05_libero_torch"
TOKENIZER_PATH=${TOKENIZER_PATH:-$SFT_MODEL_PATH}

NUM_NODES=${NUM_NODES:-1}
NUM_GPUS=${NUM_GPUS:-4}
NUM_ENV_GPUS=${NUM_ENV_GPUS:-2}
NUM_ROLLOUT_GPUS=$((NUM_GPUS - NUM_ENV_GPUS))

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
ROLLOUT_N=${ROLLOUT_N:-1}
NUM_STAGE=${NUM_STAGE:-2}
NUM_ENV=${NUM_ENV:-8}

NUM_ACTION_CHUNKS=${NUM_ACTION_CHUNKS:-8}
ACTION_DIM=${ACTION_DIM:-7}
MAX_EPISODE_STEPS=${MAX_EPISODE_STEPS:-512}

MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-128}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-8}

SIM_TYPE=${SIM_TYPE:-libero}
PROJECT_NAME=${PROJECT_NAME:-vla_libero_RL}


ISSC_PYTHON="/workspace/isaaclab/_isaac_sim/python.sh"
PYTHON=python
if [ -f "$ISSC_PYTHON" ]; then
    PYTHON=$ISSC_PYTHON
fi

mkdir -p /root/LIBERO/libero/libero/../datasets
mkdir -p "$OUTPUT_DIR"

gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -n 1)
if echo "$gpu_name" | grep "NVIDIA H"; then
    echo "enable MUJOCO_GL=osmesa in Hopper"
    export MUJOCO_GL=osmesa
fi

export VERL_LOGGING_LEVEL=INFO

$PYTHON -m verl.experimental.vla.main_sac \
    data.train_files="$libero_train_path" \
    data.val_files="$libero_test_path" \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_batch_size=4 \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    env.train.num_envs=$NUM_ENV \
    env.train.task_suite_name=$TASK_SUITE_NAME \
    data.max_prompt_length=256 \
    data.max_response_length=128 \
    env.rollout.pipeline_stage_num=$NUM_STAGE \
    env.train.simulator_type=$SIM_TYPE \
    env.actor.model.num_action_chunks=$NUM_ACTION_CHUNKS \
    env.actor.model.action_dim=$ACTION_DIM \
    env.train.only_eval=False \
    env.train.max_episode_steps=$MAX_EPISODE_STEPS \
    env.train.video_cfg.save_video=True \
    env.train.video_cfg.video_base_dir=${VIDEO_OUTPUT} \
    env.train.seed=42 \
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
    +actor_rollout_ref.model.override_config.flow_logprob_mode=path_exact \
    +actor_rollout_ref.model.override_config.flow_logprob_reduction=mean_exec_path \
    +actor_rollout_ref.model.override_config.flow_logprob_exec_steps=$NUM_ACTION_CHUNKS \
    +actor_rollout_ref.model.override_config.flow_logprob_exec_dim=$ACTION_DIM \
    +actor_rollout_ref.model.override_config.flow_logprob_clip=10.0 \
    +actor_rollout_ref.model.override_config.dataset_type=libero \
    +actor_rollout_ref.model.override_config.rlpd_single_task=True \
    +actor_rollout_ref.model.override_config.rlpd_single_task_id=$SINGLE_TASK_ID \
    actor_rollout_ref.actor.sac.n_step_return=10 \
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
    +actor_rollout_ref.algorithm=sac \
    algorithm.kl_ctrl.kl_coef=0.00 \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$OUTPUT_DIR \
    trainer.n_gpus_per_node=$NUM_GPUS \
    +trainer.n_env_gpus_per_node=$NUM_ENV_GPUS \
    +trainer.n_rollout_gpus_per_node=$NUM_ROLLOUT_GPUS \
    +trainer.rollout_interval=20 \
    +trainer.rlpd_enable=$ENABLE_RLPD \
    +data.rlpd_files="$RLPD_FILES" \
    +data.rlpd_batch_size=$RLPD_BATCH_SIZE \
    trainer.nnodes=$NUM_NODES \
    trainer.save_freq=300 \
    trainer.test_freq=-1 \
    trainer.total_epochs=500 \
    trainer.val_only=False \
    trainer.val_before_train=False \
    actor_rollout_ref.actor.critic_warmup_steps=300 \
    actor_rollout_ref.actor.sac.auto_entropy=True \
    actor_rollout_ref.actor.sac.initial_alpha=-4.6 \
    actor_rollout_ref.actor.sac.target_entropy=-2.8 \
    actor_rollout_ref.actor.sac.bc_loss_coef=0 \
    actor_rollout_ref.actor.replay_pool_save_dir="$OUTPUT_DIR/replay_pools" \
    # +trainer.offline_only=True 
