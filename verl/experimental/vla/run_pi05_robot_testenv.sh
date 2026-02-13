set -x

export HOME=/shared_disk/users/weijie.ke
export HYDRA_OUTPUT_DIR=/shared_disk/users/weijie.ke/verl/outputs

# ============================================================================
# Ray 超时和调试配置 (从脚本 A 引入，解决 OwnerDiedError)
# ============================================================================
export RAY_RUNTIME_ENV_CACHE_TTL_SECONDS=0
export RAY_memory_usage_threshold=0.98
export RAY_record_ref_creation_sites=1
# Increase timeout for model initialization (large models need more time)
export VERL_RAY_GET_TIMEOUT=1200  # 20 minutes instead of default 5 minutes
export VERL_DEBUG_RPC=1  # Enable debug output to see where it's stuck

libero_train_path=$HOME/data/libero_rl/train.parquet
libero_test_path=$HOME/data/libero_rl/test.parquet
train_files=$libero_train_path
test_files=$libero_test_path
train_files=$libero_train_path
test_files=$libero_test_path

# ============================================================================
# 模型和输出路径配置
# ============================================================================
OUTPUT_DIR=${MLP_MODEL_OUTPUT:-"$HOME/verl/models/vla_robot_sac"}
VIDEO_OUTPUT=${MLP_MODEL_OUTPUT:-"$HOME/verl"}/video
SFT_MODEL_PATH=${SFT_MODEL_PATH:-"$HOME/weight/giga-openpi/pick_catch_bowl_model"}
TOKENIZER_PATH=${TOKENIZER_PATH:-"/shared_disk/models/huggingface/models--google--paligemma-3b-pt-224"}

# ============================================================================
# 物理节点配置 (保持脚本 B 的架构)
# ============================================================================
NUM_NODES=1                                    # 节点数量
NUM_GPUS=1  
NUM_SIM_NODES=1                                   # 每个节点的总 GPU 数
ENABLE_DISAGG_SIM=True
 # 环境仿真节点数量=======================================================================
# 分布式解耦配置 (从脚本 A 引入 Disaggregation 模式)
# ============================================================================
# 从脚本 A 引入：启用解耦模式，将训练和环境仿真分离到不同的 GPU 组


# ============================================================================
# GPU 角色分配配置 (保持脚本 B 的动态计算逻辑)
# ============================================================================
NUM_ENV_GPUS=1                                 # 每个节点用于环境 worker 的 GPU 数
NUM_ROLLOUT_GPUS=1  # 每个节点用于 rollout worker 的 GPU 数

# ============================================================================
# Rollout 配置 (融合两个脚本的设置)
# ============================================================================
# 注意：TRAIN_BATCH_SIZE * ROLLOUT_N == NUM_ENV_GPUS * NUM_STAGE * NUM_ENV
TRAIN_BATCH_SIZE=1                            # 每步的数据加载器批次大小
ROLLOUT_N=1                                    # 每个 prompt 的响应数量 (用于 GRPO)
NUM_STAGE=1                                    # 流水线阶段数量
NUM_ENV=1                                      # 每个环境 worker 的环境数量

# 从脚本 A 引入：真机环境使用更大的 action chunk
NUM_ACTION_CHUNKS=10                           # action chunk 数量（脚本 A 使用 50）
MAX_EPISODE_STEPS=300                          # 每个环境的最大 episode 步数（从脚本 A 引入）
                                               # max_interactions = MAX_EPISODE_STEPS / num_action_chunks

# 从脚本 A 引入：测试环境特定的步长和 action chunk 配置
TEST_STEP_SIZE=${TEST_STEP_SIZE:-1}
TEST_ACTION_CHUNK=${TEST_ACTION_CHUNK:-50}

# ============================================================================
# 训练配置 (保持脚本 B 的 SAC 配置)
# ============================================================================
MINI_BATCH_SIZE=16                            # mini batch 大小（每个 GPU 的批次大小）
                                               # 在 SAC 中目前无效
                                               # SAC 中等于 (max_interactions - 1) * TRAIN_BATCH_SIZE * ROLLOUT_N / NUM_ROLLOUT_GPUS
MICRO_BATCH_SIZE=8                             # micro batch 大小（每个 GPU，用于梯度累积）

# ============================================================================
# 环境类型配置 (从脚本 A 引入真机环境类型)
# ============================================================================
# 从脚本 A 引入：设置为 robot 表示真机/测试环境
SIM_TYPE="robot"
PROJECT_NAME="vla_robot_RL"
EXPERIMENT_NAME="${SIM_TYPE}_sac_disagg"
# Real robot config file path
ROBOT_CONFIG_PATH=${ROBOT_CONFIG_PATH:-"/home/agilex-home/agilex/keweijie/verl/verl/experimental/vla/envs/robot_env/robot/controller/piper/config/bipiper_gym_pico.json"}
# ============================================================================
# Python 解释器选择 (保持兼容 Isaac Sim)
# ============================================================================
ISSC_PYTHON="/workspace/isaaclab/_isaac_sim/python.sh"
PYTHON=python
if [ -f "$ISSC_PYTHON" ]; then
    PYTHON=$ISSC_PYTHON
fi

# ============================================================================
# MuJoCo 配置 (针对 Hopper GPU)
# ============================================================================
# 避免警告
mkdir -p /root/LIBERO/libero/libero/../datasets 2>/dev/null || true
gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -n 1)

# 在 Hopper GPU 上强制使用 osmesa
if echo "$gpu_name" | grep "NVIDIA H"; then
    echo "enable MUJOCO_GL=osmesa in Hopper"
    export MUJOCO_GL=osmesa
fi

export VERL_LOGGING_LEVEL=INFO
export CUDA_VISIBLE_DEVICES=4,5

# ============================================================================
# 主程序调用 (整合两个脚本的参数)
# ============================================================================
$PYTHON -m verl.experimental.vla.main_sac \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_batch_size=1 \
    data.max_prompt_length=256 \
    data.max_response_length=128 \
    +data.action_chunk=$TEST_ACTION_CHUNK \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    env.train.num_envs=$NUM_ENV \
    env.rollout.pipeline_stage_num=$NUM_STAGE \
    env.train.simulator_type=$SIM_TYPE \
    `# 从脚本 A 引入：启用解耦模式配置` \
    env.disagg_sim.enable=$ENABLE_DISAGG_SIM \
    env.disagg_sim.nnodes=$NUM_SIM_NODES \
    `# 从脚本 A 引入：真机环境的数据路径和步长配置` \
    +env.train.step_size=$TEST_STEP_SIZE \
    +env.train.action_chunk=$TEST_ACTION_CHUNK \
    `# 真机环境配置：使用 robot_config_path 直接加载完整配置` \
    `# 注意：当使用 robot_config_path 时，env.train.env 配置会被 JSON 文件覆盖` \
    +env.train.env.name="gym_testenv" \
    `# use real robot config` \
    `# +env.train.robot_config_path="$ROBOT_CONFIG_PATH"` \
    `# use test env config` \
    +env.train.env.robot.type=null \
    +env.train.env.robot.config=null \
    +env.train.env.teleop.type=null \
    +env.train.env.teleop.config=null \
    `# 从脚本 A 引入：夹爪处理器配置` \
    +env.train.env.processor.gripper.use_gripper=True \
    +env.train.env.processor.gripper.gripper_penalty=0.0 \
    `# 从脚本 A 引入：重置处理器配置（成功后终止）` \
    +env.train.env.processor.reset.terminate_on_success=True \
    `# 从脚本 A 引入：观测处理器配置（不显示相机画面）` \
    +env.train.env.processor.observation.display_cameras=False \
    `# 从脚本 A 引入：设备配置（CPU 用于真机环境）` \
    +env.train.env.device="cpu" \
    `# 从脚本 A 引入：动作维度为 16（双臂机器人配置）` \
    env.actor.model.num_action_chunks=$NUM_ACTION_CHUNKS \
    env.actor.model.action_dim=16 \
    env.train.only_eval=False \
    env.train.max_episode_steps=$MAX_EPISODE_STEPS \
    env.train.video_cfg.save_video=False \
    env.train.video_cfg.video_base_dir=${VIDEO_OUTPUT} \
    env.train.seed=42 \
    env.train.reward_coef=1.0 \
    `# 保持脚本 B 的 FSDP2 策略和优化器配置` \
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
    `# 保持脚本 B 的 FSDP2 策略（更先进）` \
    actor_rollout_ref.actor.strategy=fsdp2 \
    critic.strategy=fsdp2 \
    actor_rollout_ref.actor.grad_clip=1 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    `# 从脚本 A 引入：使用 3 个输入图像（双臂机器人有多个相机）` \
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
    `# 从脚本 A 引入：真机环境使用较低的 GPU 内存占用率（0.7 vs 0.9）` \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    `# 保持脚本 B 的 SAC 算法配置` \
    +actor_rollout_ref.algorithm='sac' \
    algorithm.kl_ctrl.kl_coef=0.00 \
    trainer.logger=['console'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$OUTPUT_DIR \
    trainer.n_gpus_per_node=$NUM_GPUS \
    `# 保持脚本 B 的 GPU 分配逻辑` \
    +trainer.n_env_gpus_per_node=$NUM_ENV_GPUS \
    +trainer.n_rollout_gpus_per_node=$NUM_ROLLOUT_GPUS \
    +trainer.rollout_interval=30 \
    trainer.nnodes=$NUM_NODES \
    trainer.save_freq=30 \
    trainer.test_freq=-1 \
    trainer.total_epochs=100 \
    trainer.val_only=True \
    trainer.val_before_train=False
