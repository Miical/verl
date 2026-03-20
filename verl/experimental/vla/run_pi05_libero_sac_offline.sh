#!/usr/bin/env bash
set -x
set -e

RLPD_FILES="/shared_disk/users/yejun.zeng/datasets/huggingface/lerobot/catch_bowl"
# SFT_MODEL_PATH="/shared_disk/users/angen.ye/code/hil-serl/model/torch_pi05_base"
# SFT_MODEL_PATH="/shared_disk/users/weijie.ke/weight/giga-openpi/install_belt_joint/3w"
SFT_MODEL_PATH="/shared_disk/users/weijie.ke/weight/giga-openpi/pick_catch_bowl_new_yag/3w"
NORM_PATH="/shared_disk/users/yejun.zeng/datasets/huggingface/lerobot/catch_bowl/meta/norm.json"
TOKENIZER_PATH="$SFT_MODEL_PATH"


OUTPUT_DIR="/shared_disk/users/weijie.ke/offline_rlpd_runs/catch_bowl_offline_only_320_cb3w"

NUM_NODES=1
NUM_GPUS=4
NUM_ENV_GPUS=0
NUM_ROLLOUT_GPUS=4

RLPD_BATCH_SIZE=256
TOTAL_EPOCHS=50
MICRO_BATCH_SIZE=8
LR=1e-5

PROJECT_NAME="offline_rlpd"
EXPERIMENT_NAME="catch_bowl_offline_only"

ISSC_PYTHON="/workspace/isaaclab/_isaac_sim/python.sh"
PYTHON=python
if [ -f "$ISSC_PYTHON" ]; then
    PYTHON=$ISSC_PYTHON
fi

export VERL_LOGGING_LEVEL=INFO
mkdir -p "$OUTPUT_DIR"

gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -n 1)
if echo "$gpu_name" | grep "NVIDIA H"; then
    echo "enable MUJOCO_GL=osmesa in Hopper"
    export MUJOCO_GL=osmesa
fi

$PYTHON -m verl.experimental.vla.main_sac \
    +trainer.offline_only=True \
    +trainer.rlpd_enable=True \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$OUTPUT_DIR \
    trainer.logger=['console',"tensorboard"] \
    trainer.nnodes=$NUM_NODES \
    trainer.n_gpus_per_node=$NUM_GPUS \
    +trainer.n_env_gpus_per_node=$NUM_ENV_GPUS \
    +trainer.n_rollout_gpus_per_node=$NUM_ROLLOUT_GPUS \
    trainer.save_freq=100 \
    trainer.test_freq=-1 \
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.val_only=False \
    trainer.val_before_train=False \
    +data.rlpd_files="$RLPD_FILES" \
    +data.rlpd_batch_size=$RLPD_BATCH_SIZE \
    +actor_rollout_ref.algorithm='sac' \
    +actor_rollout_ref.model.override_config.dataset_type=lerobot \
    +actor_rollout_ref.model.override_config.attn_implementation=eager \
    +actor_rollout_ref.model.override_config.flow_logprob_mode=path_exact \
    +actor_rollout_ref.model.override_config.flow_sigma_init=0.05 \
    +actor_rollout_ref.model.override_config.flow_sigma_min=0.001 \
    +actor_rollout_ref.model.override_config.flow_sigma_max=0.5 \
    actor_rollout_ref.model.path="$SFT_MODEL_PATH" \
    actor_rollout_ref.model.tokenizer_path="$TOKENIZER_PATH" \
    actor_rollout_ref.actor.strategy=fsdp2 \
    critic.strategy=fsdp2 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.actor.critic_warmup_steps=0 \
    actor_rollout_ref.actor.sac.bc_loss_coef=1.0 \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.actor.optim.warmup_style=constant \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[SiglipEncoderLayer,GemmaDecoderLayerWithExpert] \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.trust_remote_code=False \
    actor_rollout_ref.actor.replay_pool_save_dir="$OUTPUT_DIR/replay_pools" \
    +actor_rollout_ref.model.override_config.norm_stats_path="$NORM_PATH"
    # +actor_rollout_ref.model.override_config.critic_action_steps=50 \
    # +actor_rollout_ref.model.override_config.critic_action_dim=14
