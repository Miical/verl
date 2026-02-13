# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
import sys
import traceback
from pprint import pprint

import datasets
import hydra
import ray
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.experimental.vla.sac.sac_ray_trainer import RobRaySACTrainer
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.ppo.ray_trainer import ResourcePoolManager
from verl.trainer.ppo.utils import Role
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_local_path_from_hdfs

logger = logging.getLogger(__name__)


def calculate_reward(data: DataProto, return_dict: bool = False) -> torch.Tensor:
    complete_tensor = data.batch["complete"]
    reward_per_step = complete_tensor.float()
    if return_dict:
        return {"reward_tensor": reward_per_step}
    else:
        return reward_per_step


@hydra.main(config_path="config", config_name="rob_sac_trainer", version_base=None)
def main(config):
    print(f"[Main] Driver process starting, PID={os.getpid()}", flush=True)
    
    if not ray.is_initialized():
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        logger.info(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))
    
    # 直接在 Driver 进程中运行训练逻辑
    # 这样可以确保 Driver 进程不会提前退出，ObjectRef 的 owner 始终存活
    # 避免了使用 @ray.remote 装饰的 main_task 导致的 OwnerDiedError
    
    print(f"[Main] Running TaskRunner directly in driver process...", flush=True)
    
    try:
        task_runner = TaskRunner()
        task_runner.run(config)
    except Exception as e:
        print(f"[Main] FATAL ERROR: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    finally:
        print(f"[Main] Driver process finishing...", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()


class TaskRunner:
    """普通 Python 类，用于组织训练初始化流程。
    
    注意：不要给这个类加 @ray.remote 装饰器。
    直接在 Driver 进程中运行，确保 ObjectRef 的 owner 稳定，
    防止 OwnerDiedError。
    """
    
    def run(self, config):
        # 设置异常钩子，确保所有异常都被打印
        def exception_hook(exc_type, exc_value, exc_tb):
            print(f"[TaskRunner] UNCAUGHT EXCEPTION: {exc_type.__name__}: {exc_value}", flush=True)
            traceback.print_exception(exc_type, exc_value, exc_tb)
            sys.stdout.flush()
            sys.stderr.flush()
        
        sys.excepthook = exception_hook
        
        print(f"[TaskRunner] Starting on PID={os.getpid()}, Node={ray.util.get_node_ip_address()}", flush=True)
        
        # print initial config
        pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
        OmegaConf.resolve(config)

        # download the checkpoint from hdfs
        local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

        # instantiate tokenizer
        tokenizer = hf_tokenizer(local_path)

        # define worker classes
        if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.experimental.vla.workers.env.env_worker import EnvWorker
            from verl.single_controller.ray import RayWorkerGroup

            from .fsdp_workers import RobActorRolloutRefWorker

            ray_worker_group_cls = RayWorkerGroup
        else:
            raise NotImplementedError

        role_worker_mapping = {
            # Role.ActorRollout: ray.remote(RobActorRolloutRefWorker),
            Role.ActorRollout: ray.remote(resources={"node:A": 0.1})(RobActorRolloutRefWorker),
            # Role.Env: ray.remote(EnvWorker),
            Role.Env: ray.remote(resources={"node:B": 0.1})(EnvWorker),
        }

        # setup resource pool manager
        train_rollout_gpu_num = config.trainer.n_rollout_gpus_per_node
        train_rollout_nodes_num = config.trainer.nnodes
        env_gpu_num = config.trainer.n_env_gpus_per_node
        
        # Calculate number of simulation nodes based on disaggregation config
        if config.env.disagg_sim.enable:
            # disaggregated sim and actor rollout
            num_nodes_sim = config.env.disagg_sim.nnodes
        else:
            # colocated sim and actor rollout
            num_nodes_sim = config.trainer.nnodes

        resource_pool_spec = {
            "train_rollout_pool": [train_rollout_gpu_num] * train_rollout_nodes_num,
        }
        mapping = {
            Role.ActorRollout: "train_rollout_pool",
        }
        
        # Always create separate env_gpu_pool for Env workers
        # This ensures Env workers are scheduled to the correct node (nodeB with robot hardware)
        # even when they don't need GPUs
        resource_pool_spec["env_gpu_pool"] = [max(env_gpu_num, 1)] * num_nodes_sim
        mapping[Role.Env] = "env_gpu_pool"
        
        print(f"resource_pool_spec: {resource_pool_spec}")
        print(f"env_gpu_num: {env_gpu_num}, num_nodes_sim: {num_nodes_sim}")
        print(f"mapping: {mapping}")
        
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        # create datasets
        train_dataset = datasets.load_dataset("parquet", data_files=config.data.train_files)["train"]
        val_dataset = datasets.load_dataset("parquet", data_files=config.data.val_files)["train"]

        # instantiate trainer and start training
        try:
            trainer = RobRaySACTrainer(
                config=config,
                tokenizer=tokenizer,
                role_worker_mapping=role_worker_mapping,
                resource_pool_manager=resource_pool_manager,
                ray_worker_group_cls=ray_worker_group_cls,
                reward_fn=calculate_reward,
                val_reward_fn=calculate_reward,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
            )

            trainer.init_workers()
            trainer.fit()
        except Exception as e:
            print(f"[TaskRunner] FATAL ERROR in trainer: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
            raise


if __name__ == "__main__":
    main()
