#!/usr/bin/env bash
set -x
set -e

# =========================================================
# 目标：
# 真机 online + rollout
# 同时混入 rlpd 数据
# actor 先用 bc loss 验证链路是否走通
# =========================================================

export HOME=/shared_disk/users/weijie.ke
export HYDRA_OUTPUT_DIR=/shared_disk/users/weijie.ke/verl/outputs

# =========================================================
# Ray / 调试
# =========================================================
export RAY_RUNTIME_ENV_CACHE_TTL_SECONDS=0
export RAY_memory_usage_threshold=0.98
export RAY_record_ref_creation_sites=1
export VERL_RAY_GET_TIMEOUT=1200
export VERL_DEBUG_RPC=1
export VERL_LOGGING_LEVEL=INFO

# =========================================================
# 在线 prompt 数据（online 分支仍然要）
# =========================================================
LIBERO_TRAIN_PATH="$HOME/data/libero_rl/train.parquet"
LIBERO_TEST_PATH="$HOME/data/libero_rl/test.parquet"
TRAIN_FILES="$LIBERO_TRAIN_PATH"
TEST_FILES="$LIBERO_TEST_PATH"

# =========================================================
# RLPD / LeRobot 数据（用于混入 bc）
# =========================================================
RLPD_FILES="/shared_disk/users/yejun.zeng/datasets/huggingface/lerobot/catch_bowl"
RLPD_BATCH_SIZE=64

# =========================================================
# 模型路径
# 注意：这里先沿用你 offline 能跑通的目录
# =========================================================
# SFT_MODEL_PATH="/shared_disk/users/weijie.ke/weight/giga-openpi/pick_catch_bowl_new_yag/3w"
# TOKENIZER_PATH="$SFT_MODEL_PATH"
# NORM_PATH="/shared_disk/users/yejun.zeng/datasets/huggingface/lerobot/catch_bowl/meta/norm.json"

SFT_MODEL_PATH="/shared_disk/users/weijie.ke/weight/giga-openpi/install_belt_joint/1k"
TOKENIZER_PATH="$SFT_MODEL_PATH"
NORM_PATH="/shared_disk/users/yejun.zeng/datasets/huggingface/lerobot/install_belt/meta/norm.json"
# =========================================================
# 输出目录
# =========================================================
OUTPUT_DIR="/shared_disk/users/weijie.ke/online_robot_rlpd_runs/catch_bowl_robot_online_bc_323_test"
VIDEO_OUTPUT="/shared_disk/users/weijie.ke/verl/video"
mkdir -p "$OUTPUT_DIR"

# =========================================================
# 资源配置
# 这是“先验证能不能走通”的保守配置
# =========================================================
NUM_NODES=1
NUM_GPUS=1
NUM_ENV_GPUS=1
NUM_ROLLOUT_GPUS=1
NUM_SIM_NODES=1
ENABLE_DISAGG_SIM=True

# =========================================================
# rollout / env 配置
# =========================================================
TRAIN_BATCH_SIZE=1
ROLLOUT_N=1
NUM_STAGE=1
NUM_ENV=1

# 建议先和数据/action chunk 对齐
ACTION_CHUNK=50
MAX_EPISODE_STEPS=300
STEP_SIZE=1

# =========================================================
# 训练配置
# =========================================================
TOTAL_EPOCHS=5
ROLLOUT_INTERVAL=1
MICRO_BATCH_SIZE=8
MINI_BATCH_SIZE=16
LR=1e-5

PROJECT_NAME="online_robot_rlpd"
EXPERIMENT_NAME="catch_bowl_robot_online_bc_smoke"

# =========================================================
# 真机配置
# =========================================================
SIM_TYPE="robot"
ROBOT_CONFIG_PATH="/home/agilex-home/agilex/keweijie/verl/verl/experimental/vla/envs/robot_env/robot/controller/piper/config/bipiper_gym_pico.json"

# =========================================================
# Python
# =========================================================
ISSC_PYTHON="/workspace/isaaclab/_isaac_sim/python.sh"
PYTHON=python
if [ -f "$ISSC_PYTHON" ]; then
    PYTHON=$ISSC_PYTHON
fi

# =========================================================
# MuJoCo / Hopper
# =========================================================
mkdir -p /root/LIBERO/libero/libero/../datasets 2>/dev/null || true
gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -n 1)
if echo "$gpu_name" | grep "NVIDIA H"; then
    echo "enable MUJOCO_GL=osmesa in Hopper"
    export MUJOCO_GL=osmesa
fi

# 如有需要你自己改
# export CUDA_VISIBLE_DEVICES=4,5

# =========================================================
# 主程序
# =========================================================
$PYTHON -m verl.experimental.vla.main_sac \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.default_local_dir="$OUTPUT_DIR" \
    trainer.logger=['console',"tensorboard"] \
    trainer.nnodes=$NUM_NODES \
    trainer.n_gpus_per_node=$NUM_GPUS \
    +trainer.n_env_gpus_per_node=$NUM_ENV_GPUS \
    +trainer.n_rollout_gpus_per_node=$NUM_ROLLOUT_GPUS \
    +trainer.rollout_interval=$ROLLOUT_INTERVAL \
    +trainer.rlpd_enable=True \
    trainer.save_freq=100 \
    trainer.test_freq=-1 \
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.val_only=False \
    trainer.val_before_train=False \
    \
    +data.robot_online_num_samples=1 \
    +data.robot_online_prompt="catch_bowl" \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_batch_size=1 \
    data.max_prompt_length=256 \
    data.max_response_length=128 \
    +data.action_chunk=$ACTION_CHUNK \
    +data.rlpd_files="$RLPD_FILES" \
    +data.rlpd_batch_size=$RLPD_BATCH_SIZE \
    \
    env.train.simulator_type=$SIM_TYPE \
    env.train.num_envs=$NUM_ENV \
    env.rollout.pipeline_stage_num=$NUM_STAGE \
    env.disagg_sim.enable=$ENABLE_DISAGG_SIM \
    env.disagg_sim.nnodes=$NUM_SIM_NODES \
    +env.train.step_size=$STEP_SIZE \
    +env.train.action_chunk=$ACTION_CHUNK \
    +env.train.env.name="gym_testenv" \
    +env.train.robot_config_path="$ROBOT_CONFIG_PATH" \
    +env.train.env.processor.gripper.use_gripper=True \
    +env.train.env.processor.gripper.gripper_penalty=0.0 \
    +env.train.env.processor.reset.terminate_on_success=True \
    +env.train.env.processor.observation.display_cameras=False \
    +env.train.env.device="cpu" \
    env.actor.model.num_action_chunks=$ACTION_CHUNK \
    env.actor.model.action_dim=14 \
    env.train.only_eval=False \
    env.train.max_episode_steps=$MAX_EPISODE_STEPS \
    env.train.video_cfg.save_video=False \
    env.train.video_cfg.video_base_dir="$VIDEO_OUTPUT" \
    env.train.seed=42 \
    env.train.reward_coef=1.0 \
    \
    +actor_rollout_ref.algorithm='sac' \
    actor_rollout_ref.model.path="$SFT_MODEL_PATH" \
    actor_rollout_ref.model.tokenizer_path="$TOKENIZER_PATH" \
    +actor_rollout_ref.model.override_config.dataset_type=lerobot \
    +actor_rollout_ref.model.override_config.attn_implementation=eager \
    +actor_rollout_ref.model.override_config.flow_logprob_mode=path_exact \
    +actor_rollout_ref.model.override_config.flow_sigma_init=0.05 \
    +actor_rollout_ref.model.override_config.flow_sigma_min=0.001 \
    +actor_rollout_ref.model.override_config.flow_sigma_max=0.5 \
    +actor_rollout_ref.model.override_config.norm_stats_path="$NORM_PATH" \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.trust_remote_code=False \
    \
    actor_rollout_ref.rollout.mode=async_envloop \
    actor_rollout_ref.rollout.name=hf \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.temperature=1.6 \
    actor_rollout_ref.rollout.prompt_length=512 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.free_cache_engine=False \
    \
    actor_rollout_ref.actor.strategy=fsdp2 \
    critic.strategy=fsdp2 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.actor.critic_warmup_steps=0 \
    actor_rollout_ref.actor.sac.bc_loss_coef=1.0 \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.actor.optim.warmup_style=constant \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.num_images_in_input=3 \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[SiglipEncoderLayer,GemmaDecoderLayerWithExpert] \
    actor_rollout_ref.actor.replay_pool_save_dir="$OUTPUT_DIR/replay_pools" \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.00