# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import logging
import os
import pdb
import datasets
import hydra
import ray
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.ppo.ray_trainer import ResourcePoolManager
from verl.trainer.ppo.utils import Role

from recipe.vla.sac.sac_ray_trainer import RobRayPPOTrainer

logger = logging.getLogger(__name__)


def calculate_reward(data: DataProto, return_dict: bool = False) -> torch.Tensor:
    complete_tensor = data.batch["complete"]
    batch_size, num_steps = complete_tensor.shape[:2]
    traj_has_complete = torch.any(complete_tensor, dim=(1, 2))  # shape: [batch_size]
    reward_per_traj = traj_has_complete.float()
    reward_per_step = reward_per_traj.unsqueeze(1).expand(batch_size, num_steps)
    if return_dict:
        return {"reward_tensor": reward_per_step}
    else:
        return reward_per_step


@hydra.main(config_path="config", config_name="rob_sac_trainer", version_base=None)
def main(config):
    import sys
    import signal
    
    print(f"[Main] Driver process starting, PID={os.getpid()}", flush=True)
    
    if not ray.is_initialized():
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        logger.info(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))
    
    # 方案：不使用 TaskRunner Actor，直接在 Driver 进程中运行
    # 这样可以确保 Driver 进程不会提前退出，ObjectRef 的 owner 始终存活
    # 
    # 原来使用 TaskRunner Actor 的问题是：
    # 1. Driver 创建 TaskRunner Actor
    # 2. TaskRunner Actor 创建其他 Workers
    # 3. 如果 Driver 退出，TaskRunner 会被杀死，导致 OwnerDiedError
    #
    # 现在直接在 Driver 中运行，避免这个问题
    
    print(f"[Main] Running TaskRunner directly in driver process...", flush=True)
    
    try:
        task_runner = TaskRunner()
        task_runner.run(config)
    except Exception as e:
        print(f"[Main] FATAL ERROR: {type(e).__name__}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    finally:
        print(f"[Main] Driver process finishing...", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()

    # ray.timeline(filename="/tmp/ray_timeline.json")


class TaskRunner:
    """Ray Actor for executing distributed training tasks.
    
    Using an Actor instead of a Task ensures stable ObjectRef ownership,
    preventing OwnerDiedError when passing data between distributed workers.
    """
    
    def run(self, config):
        import os
        import sys
        import traceback
        
        # 设置异常钩子，确保所有异常都被打印
        def exception_hook(exc_type, exc_value, exc_tb):
            print(f"[TaskRunner] UNCAUGHT EXCEPTION: {exc_type.__name__}: {exc_value}", flush=True)
            traceback.print_exception(exc_type, exc_value, exc_tb)
            sys.stdout.flush()
            sys.stderr.flush()
        
        sys.excepthook = exception_hook
        
        print(f"[TaskRunner] Starting on PID={os.getpid()}, Node={ray.util.get_node_ip_address()}", flush=True)
        
        # print initial config
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_local_path_from_hdfs

        pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
        OmegaConf.resolve(config)

        # download the checkpoint from hdfs
        local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

        # instantiate tokenizer
        from verl.utils import hf_tokenizer

        tokenizer = hf_tokenizer(local_path)

        # define worker classes
        if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from recipe.vla.workers.env.env_worker import EnvWorker
            from verl.single_controller.ray import RayWorkerGroup

            from .fsdp_workers import RobActorRolloutRefWorker

            ray_worker_group_cls = RayWorkerGroup

        else:
            raise NotImplementedError

        role_worker_mapping = {
            # Role.Critic: ray.remote(RobActorRolloutRefWorker),
            Role.ActorRollout: ray.remote(resources={"node:A": 0.1})(RobActorRolloutRefWorker),
            # Role.ActorRollout: ray.remote(RobActorRolloutRefWorker),
            # Role.RefPolicy: ray.remote(RobActorRolloutRefWorker),
            Role.Env: ray.remote(resources={"node:B": 0.1})(EnvWorker),
            # Role.Env: ray.remote(EnvWorker),
        }

        train_rollout_pool_id = "train_rollout_pool"

        num_nodes_actor_rollout = config.trainer.nnodes
        train_rollout_gpu_num = config.trainer.n_rollout_gpus_per_node
        env_gpu_num = config.trainer.n_env_gpus_per_node
        if config.env.disagg_sim.enable:
            # disaggregated sim and actor rollout
            num_nodes_sim = config.env.disagg_sim.nnodes
        else:
            # colocated sim and actor rollout
            num_nodes_sim = config.trainer.nnodes
        # print(f"num_nodes_sim: {num_nodes_sim}")
        # pdb.set_trace()
        
        # Build resource pool spec and mapping
        resource_pool_spec = {
            train_rollout_pool_id: [train_rollout_gpu_num] * num_nodes_actor_rollout,
        }
        mapping = {
            Role.ActorRollout: train_rollout_pool_id,
            # Role.Critic: global_pool_id,
            # Role.RefPolicy: global_pool_id,
        }
        
        # Always create separate env_gpu_pool for Env workers
        # This ensures Env workers are scheduled to the correct node (nodeB with robot hardware)
        # even when they don't need GPUs
        resource_pool_spec["env_gpu_pool"] = [max(env_gpu_num, 1)] * num_nodes_sim
        mapping[Role.Env] = "env_gpu_pool"
        
        print(f"resource_pool_spec: {resource_pool_spec}")
        print(f"env_gpu_num: {env_gpu_num}, num_nodes_sim: {num_nodes_sim}")
        print(f"mapping: {mapping}")

        reward_fn = calculate_reward
        val_reward_fn = calculate_reward

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        # Create training and validation datasets.
        # Always use LeRobotRLDataset if specified (including val_only mode)
        # because env_loop requires state_ids from the dataset
        collate_fn = None
        train_dataset = None
        val_dataset = None
        
        # Use custom LeRobot dataset if specified
        if config.data.get("custom_cls", {}).get("path") == "recipe.vla.dataset.lerobot_dataset":
            from recipe.vla.dataset.lerobot_dataset import LeRobotRLDataset, collate_fn as lerobot_collate_fn
            from verl.utils import hf_processor
            
            processor = hf_processor(local_path, trust_remote_code=config.data.get("trust_remote_code", False))
            
            # Always create datasets - env_loop needs state_ids even in val_only mode
            train_dataset = LeRobotRLDataset(
                data_files=config.data.train_files,
                tokenizer=tokenizer,
                config=config.data,
                processor=processor,
                max_samples=config.data.get("train_max_samples", -1),
            )
            val_dataset = LeRobotRLDataset(
                data_files=config.data.val_files,
                tokenizer=tokenizer,
                config=config.data,
                processor=processor,
                max_samples=config.data.get("val_max_samples", -1),
            )
            collate_fn = lerobot_collate_fn
        elif config.trainer.val_only is not True:
            # Only load parquet datasets for non-val_only mode without LeRobot
            train_dataset = datasets.load_dataset("parquet", data_files=config.data.train_files)["train"]
            val_dataset = datasets.load_dataset("parquet", data_files=config.data.val_files)["train"]

        logger.info("Creating RobRayPPOTrainer...")
        try:
            trainer = RobRayPPOTrainer(
                config=config,
                tokenizer=tokenizer,
                role_worker_mapping=role_worker_mapping,
                resource_pool_manager=resource_pool_manager,
                ray_worker_group_cls=ray_worker_group_cls,
                reward_fn=reward_fn,
                val_reward_fn=val_reward_fn,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                collate_fn=collate_fn,
            )
            logger.info("Trainer created, initializing workers...")
            trainer.init_workers()
            logger.info("Workers initialized, starting training...")
            trainer.fit()
            logger.info("Training completed!")
        except Exception as e:
            print(f"[TaskRunner] FATAL ERROR in trainer: {type(e).__name__}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
            raise


if __name__ == "__main__":
    main()
