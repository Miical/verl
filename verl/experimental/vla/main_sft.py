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
from pprint import pprint

import hydra
import ray
from omegaconf import OmegaConf

from verl.experimental.vla.sft.sft_ray_trainer import RobRaySFTTrainer
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.ppo.ray_trainer import ResourcePoolManager
from verl.trainer.ppo.utils import Role

logger = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="rob_sft_trainer", version_base=None)
def main(config):
    if not ray.is_initialized():
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        logger.info(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))
    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
        from verl.single_controller.ray import RayWorkerGroup

        from .fsdp_workers import RobActorRolloutRefWorker

        ray_worker_group_cls = RayWorkerGroup
    else:
        raise NotImplementedError

    role_worker_mapping = {
        Role.Actor: ray.remote(RobActorRolloutRefWorker),
    }

    train_rollout_gpu_num = config.trainer.n_gpus_per_node
    train_rollout_nodes_num = config.trainer.nnodes

    resource_pool_spec = {
        "train_rollout_pool": [train_rollout_gpu_num] * train_rollout_nodes_num,
    }
    resource_pool_kwargs = {
        "train_rollout_pool": {"use_gpu": True},
    }
    mapping = {
        Role.Actor: "train_rollout_pool",
    }
    resource_pool_manager = ResourcePoolManager(
        resource_pool_spec=resource_pool_spec,
        mapping=mapping,
        resource_pool_kwargs=resource_pool_kwargs,
    )

    trainer = RobRaySFTTrainer(
        config=config,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=ray_worker_group_cls,
    )
    trainer.init_workers()
    trainer.fit()


if __name__ == "__main__":
    main()
