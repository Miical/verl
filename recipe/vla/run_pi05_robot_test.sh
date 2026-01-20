set -x
export HOME=/shared_disk/users/weijie.ke
export HYDRA_OUTPUT_DIR=/shared_disk/users/weijie.ke/verl/outputs
export RAY_RUNTIME_ENV_CACHE_TTL_SECONDS=0
export RAY_memory_usage_threshold=0.98
export RAY_record_ref_creation_sites=1
# Increase timeout for model initialization (large models need more time)
export VERL_RAY_GET_TIMEOUT=1200  # 20 minutes instead of default 5 minutes
export VERL_DEBUG_RPC=1  # Enable debug output to see where it's stuck

# Data path for agilex dual-arm robot dataset
DATA_PATH=/shared_disk/users/yejun.zeng/datasets/huggingface/lerobot/catch_bowl/
TEST_DATA_PATH=${TEST_DATA_PATH:-"$DATA_PATH"}
libero_train_path=/shared_disk/users/yejun.zeng/datasets/huggingface/lerobot/catch_bowl/
libero_test_path=/shared_disk/users/yejun.zeng/datasets/huggingface/lerobot/catch_bowl/

export CUDA_VISIBLE_DEVICES=0,1
train_files=$libero_train_path
test_files=$libero_test_path

OUTPUT_DIR=${MLP_MODEL_OUTPUT:-"$HOME/verl/models/vla_robot_grpo"}
VIDEO_OUTPUT=${MLP_MODEL_OUTPUT:-"$HOME/verl"}/video
SFT_MODEL_PATH=${SFT_MODEL_PATH:-"$HOME/weight/giga-openpi/pick_catch_bowl_model"}
TOKENIZER_PATH=${TOKENIZER_PATH:-"/shared_disk/models/huggingface/models--google--paligemma-3b-pt-224"}

# Disaggregation mode configuration
# NUM_NODES: Number of train/rollout nodes (main nodes)
# NUM_SIM_NODES: Number of simulation/environment nodes (robot/env nodes)
NUM_NODES=1
NUM_SIM_NODES=1  # Enable 1 simulation node (node B - robot side)
ENABLE_DISAGG_SIM=True  # Enable disaggregation: node A for train, node B for env

NUM_GPUS=1
NUM_ROLLOUT_GPUS=1  # Use 1 GPU on node A for rollout
NUM_ENV_GPUS=1      # Use 1 GPU on node B for environment

# rollout.n should equal to num_envs for real robot
ROLLOUT_N=1
# test means test_env using LeRobot dataset replay (also works for real robot)
SIM_TYPE="test"
PROJECT_NAME="vla_test_env_RL"
EXPERIMENT_NAME="${SIM_TYPE}_reinforce_plus_plus"

ISSC_PYTHON="/workspace/isaaclab/_isaac_sim/python.sh"
PYTHON=python
if [ -f "$ISSC_PYTHON" ]; then
    PYTHON=$ISSC_PYTHON
fi

# Real robot config file path
# 注意: 端侧机器路径与主机不同，使用端侧的路径
ROBOT_CONFIG_PATH=${ROBOT_CONFIG_PATH:-"/home/agilex-home/agilex/keweijie/verl/recipe/vla/envs/test_env/robot/controller/piper/config/bipiper_gym_pico.json"}

# Test env specific config (for backward compatibility)
TEST_STEP_SIZE=${TEST_STEP_SIZE:-1}
TEST_ACTION_CHUNK=${TEST_ACTION_CHUNK:-50}

$PYTHON -m recipe.vla.main_sac \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=1 \
    data.val_batch_size=1 \
    data.custom_cls.path="pkg://recipe.vla.dataset.lerobot_dataset" \
    data.custom_cls.name="LeRobotRLDataset" \
    +data.action_chunk=$TEST_ACTION_CHUNK \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    env.train.num_envs=$ROLLOUT_N \
    data.max_prompt_length=256 \
    data.max_response_length=128 \
    env.rollout.pipeline_stage_num=1 \
    env.train.simulator_type=$SIM_TYPE \
    env.disagg_sim.enable=$ENABLE_DISAGG_SIM \
    env.disagg_sim.nnodes=$NUM_SIM_NODES \
    +env.train.data_path="$TEST_DATA_PATH" \
    +env.train.step_size=$TEST_STEP_SIZE \
    +env.train.action_chunk=$TEST_ACTION_CHUNK \
    +env.train.robot_config_path="$ROBOT_CONFIG_PATH" \
    +env.train.env.name="gym_testenv" \
    +env.train.env.data_path="$TEST_DATA_PATH" \
    +env.train.env.step_size=$TEST_STEP_SIZE \
    +env.train.env.action_chunk=$TEST_ACTION_CHUNK \
    +env.train.env.robot.type="piper" \
    +env.train.env.robot.config="$ROBOT_CONFIG_PATH" \
    +env.train.env.teleop.type=null \
    +env.train.env.teleop.config=null \
    +env.train.env.processor.gripper.use_gripper=True \
    +env.train.env.processor.gripper.gripper_penalty=0.0 \
    +env.train.env.processor.reset.terminate_on_success=True \
    +env.train.env.processor.observation.display_cameras=False \
    +env.train.env.device="cpu" \
    env.actor.model.num_action_chunks=10 \
    env.actor.model.action_dim=16 \
    env.train.only_eval=True \
    env.train.max_episode_steps=300 \
    env.train.video_cfg.save_video=False \
    env.train.video_cfg.video_base_dir=${VIDEO_OUTPUT} \
    env.train.seed=42 \
    env.train.reward_coef=1.0 \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[SiglipEncoderLayer,GemmaDecoderLayerWithExpert] \
    actor_rollout_ref.model.path=$SFT_MODEL_PATH \
    actor_rollout_ref.model.tokenizer_path=$TOKENIZER_PATH \
    actor_rollout_ref.rollout.mode=async_envloop \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.actor.optim.warmup_style=constant \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.strategy=fsdp \
    critic.strategy=fsdp \
    actor_rollout_ref.actor.grad_clip=1 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.num_images_in_input=3 \
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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.00 \
    trainer.logger=['console'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$OUTPUT_DIR \
    trainer.n_gpus_per_node=$NUM_GPUS \
    +trainer.n_env_gpus_per_node=$NUM_ENV_GPUS \
    +trainer.n_rollout_gpus_per_node=$NUM_ROLLOUT_GPUS \
    trainer.nnodes=$NUM_NODES \
    trainer.save_freq=30 \
    trainer.test_freq=-1 \
    trainer.total_epochs=10 \
    trainer.val_only=True \
    algorithm.adv_estimator=reinforce_plus_plus \
    trainer.val_before_train=False

