set -x
libero_train_path=$HOME/data/libero_rl/train.parquet
libero_test_path=$HOME/data/libero_rl/test.parquet

train_files=$libero_train_path
test_files=$libero_test_path
# rlpd_files="/file_system/liujincheng/datasets/20251027T005_install_belt_cyt001_01"
# Priority: explicit env RLPD_FILES > pre-set shell variable rlpd_files > default.
rlpd_files=${RLPD_FILES:-${rlpd_files:-"/shared_disk/users/angen.ye/code/hil-serl/datasets/LIBERO-dataset/libero_10"}}

OUTPUT_DIR=${MLP_MODEL_OUTPUT:-"/shared_disk/users/angen.ye/code/hil-serl/model/verl_fintune_model/test302_libero4"}
VIDEO_OUTPUT="${OUTPUT_DIR}/video"
RL_DEBUG_DIR=${RL_DEBUG_DIR:-"${OUTPUT_DIR}/rl_debug_once"}
SFT_MODEL_PATH=${SFT_MODEL_PATH:-"/shared_disk/users/angen.ye/code/hil-serl/model/pi05_libero_torch"}
TOKENIZER_PATH="$SFT_MODEL_PATH"

# Physical Node Config
NUM_NODES=1                                    # number of nodes
NUM_GPUS=4                                     # total number of gpus per node

# Role Config
NUM_ENV_GPUS=2                                 # number of gpus for env workers per node
NUM_ROLLOUT_GPUS=$((NUM_GPUS - NUM_ENV_GPUS))  # number of gpus for rollout workers per node

# Rollout Config
# NOTE: TRAIN_BATCH_SIZE * ROLLOUT_N == NUM_ENV_GPUS * NUM_STAGE * NUM_ENV
TRAIN_BATCH_SIZE=32                            # batch size for dataloaders per step
ROLLOUT_N=1                                    # response number for each prompt (for GRPO)
NUM_STAGE=2                                    # number of pipeline stages
NUM_ENV=8                                      # number of envs per env worker

NUM_ACTION_CHUNKS=10                           # number of action chunks
MAX_EPISODE_STEPS=500                           # max episode steps for each env
                                               # max_interactions = MAX_EPISODE_STEPS / num_action_chunks

# Training Config
MINI_BATCH_SIZE=128                            # mini batch size (batch size per GPU, automatically multiplied by ROLLOUT_N)
                                               # invalid in SAC, currently
                                               # In SAC, it equal to (max_interactions - 1) * TRAIN_BATCH_SIZE * ROLLOUT_N / NUM_ROLLOUT_GPUS
MICRO_BATCH_SIZE=8                             # micro batch size (per GPU, for gradient accumulation, should divide MINI_BATCH_SIZE)
RLPD_BATCH_SIZE=$(((MAX_EPISODE_STEPS / NUM_ACTION_CHUNKS - 1) * TRAIN_BATCH_SIZE * ROLLOUT_N))  # batch size for RLPD data loader



# isaac or libero
# libero means original libero benchmark with mujoco sim
# isaac means libero benchmark using isaac sim
SIM_TYPE=${SIM_TYPE:-"libero"}
PROJECT_NAME="vla_libero_RL"
EXPERIMENT_NAME="${SIM_TYPE}_reinforce_plus_plus"

ISSC_PYTHON="/workspace/isaaclab/_isaac_sim/python.sh"
PYTHON=python
if [ -f "$ISSC_PYTHON" ]; then
    PYTHON=$ISSC_PYTHON
fi

# avoiding warnings
mkdir /root/LIBERO/libero/libero/../datasets
gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -n 1)

# force osmesa in Hopper
if echo "$gpu_name" | grep "NVIDIA H"; then
    echo "enable MUJOCO_GL=osmesa in Hopper"
    export MUJOCO_GL=osmesa
fi

export VERL_LOGGING_LEVEL=INFO

ACTOR_LOSS_TYPE=${ACTOR_LOSS_TYPE:-sac}
EXPORT_ROLLOUT_HDF5_DIR=${EXPORT_ROLLOUT_HDF5_DIR:-"${rlpd_files}"}
EXPORT_ROLLOUT_MAX_DEMOS=${EXPORT_ROLLOUT_MAX_DEMOS:-0}
EXPORT_ROLLOUT_EXIT_AFTER_DUMP=${EXPORT_ROLLOUT_EXIT_AFTER_DUMP:-0}

# One-shot RL debug dump for checking rollout-vs-dataset alignment.
# Set RL_DEBUG_DUMP=1 to enable and dump on step 1.
RL_DEBUG_DUMP=${RL_DEBUG_DUMP:-0}
RL_DEBUG_DUMP_INTERVAL=0
if [ "$RL_DEBUG_DUMP" = "1" ]; then
    RL_DEBUG_DUMP_INTERVAL=1
fi

echo "[run_pi05_libero_sac] Using rlpd_files=${rlpd_files}"
echo "[run_pi05_libero_sac] Using export_rollout_hdf5_dir=${EXPORT_ROLLOUT_HDF5_DIR}"
echo "[run_pi05_libero_sac] actor_loss_type=${ACTOR_LOSS_TYPE}, export_rollout_max_demos=${EXPORT_ROLLOUT_MAX_DEMOS}, rl_debug_dump=${RL_DEBUG_DUMP}"

$PYTHON -m verl.experimental.vla.main_sac \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    +data.rlpd_files="$rlpd_files" \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_batch_size=4 \
    +data.rlpd_batch_size=$RLPD_BATCH_SIZE \
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
    actor_rollout_ref.model.override_config.dataset_type=libero \
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
    +actor_rollout_ref.algorithm='sac' \
    algorithm.kl_ctrl.kl_coef=0.00 \
    trainer.logger=['console'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$OUTPUT_DIR \
    trainer.n_gpus_per_node=$NUM_GPUS \
    +trainer.n_env_gpus_per_node=$NUM_ENV_GPUS \
    +trainer.n_rollout_gpus_per_node=$NUM_ROLLOUT_GPUS \
    +trainer.rollout_interval=60 \
    +trainer.rlpd_enable=True \
    +trainer.rl_debug_dump_interval=$RL_DEBUG_DUMP_INTERVAL \
    +trainer.rl_debug_dump_num_samples=4 \
    +trainer.rl_debug_dump_dir=${RL_DEBUG_DIR} \
    trainer.nnodes=$NUM_NODES \
    trainer.save_freq=30 \
    trainer.test_freq=-1 \
    trainer.total_epochs=100 \
    trainer.val_only=False \
    trainer.val_before_train=False \
    actor_rollout_ref.actor.sac.actor_loss_type=${ACTOR_LOSS_TYPE} \
    +trainer.export_rollout_hdf5_dir=${EXPORT_ROLLOUT_HDF5_DIR} \
    +trainer.export_rollout_max_demos=${EXPORT_ROLLOUT_MAX_DEMOS} \
    +trainer.export_rollout_exit_after_dump=${EXPORT_ROLLOUT_EXIT_AFTER_DUMP}
