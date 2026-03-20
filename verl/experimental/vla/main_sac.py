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
from pprint import pprint

import datasets
import hydra
import ray
import torch
from omegaconf import OmegaConf, open_dict

from verl import DataProto
from verl.experimental.vla.dataloader import build_dataloader_components
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
    return reward_per_step


def _build_robot_online_dummy_dataset(config):
    configured_num_samples = int(OmegaConf.select(config, "data.robot_online_num_samples") or 1)
    configured_prompt = OmegaConf.select(config, "data.robot_online_prompt")
    configured_task_id = int(OmegaConf.select(config, "data.robot_online_task_id") or 0)

    batch_size = int(config.data.get("gen_batch_size", config.data.train_batch_size))
    num_samples = max(configured_num_samples, batch_size)
    fixed_prompt = configured_prompt or OmegaConf.select(config, "env.train.env.task_description") or "catch_bowl"

    rows = []
    for idx in range(num_samples):
        rows.append(
            {
                "data_source": "robot_online",
                "prompt": fixed_prompt,
                "state_ids": 0,
                "task_ids": configured_task_id,
                "ability": "robot",
                "extra_info": {
                    "split": "train",
                    "state_ids": 0,
                    "index": idx,
                    "task": fixed_prompt,
                    "task_ids": configured_task_id,
                    "task_suite_name": "real_robot",
                },
            }
        )

    return datasets.Dataset.from_list(rows)


def _maybe_set_dataset_norm_stats_path(config):
    dataset_type = OmegaConf.select(config, "actor_rollout_ref.model.override_config.dataset_type")
    if dataset_type != "lerobot":
        return

    current = OmegaConf.select(config, "actor_rollout_ref.model.override_config.norm_stats_path")
    if current:
        return

    rlpd_root = OmegaConf.select(config, "data.rlpd_files")
    if not rlpd_root:
        return

    candidates = [
        os.path.join(rlpd_root, "meta", "giga_pi05_norm_stats.json"),
        os.path.join(rlpd_root, "meta", "norm_stats.json"),
        os.path.join(rlpd_root, "meta", "norm.json"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            with open_dict(config.actor_rollout_ref.model.override_config):
                config.actor_rollout_ref.model.override_config.norm_stats_path = candidate
            logger.info(f"Using dataset norm stats from: {candidate}")
            return

    logger.warning(
        "Could not auto-detect dataset norm stats under rlpd_files/meta; "
        "falling back to checkpoint-provided state/action stats."
    )


@hydra.main(config_path="config", config_name="rob_sac_trainer", version_base=None)
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

    offline_only = bool(config.trainer.get("offline_only", False))

    _maybe_set_dataset_norm_stats_path(config)

    # propagate offline flag into actor_rollout worker config so worker-side code can branch on it
    with open_dict(config.actor_rollout_ref):
        config.actor_rollout_ref.offline_only = offline_only
        if offline_only:
            # rollout should never be built in offline-only mode, but keep TP conservative anyway
            config.actor_rollout_ref.rollout.tensor_model_parallel_size = 1

    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)
    tokenizer_src = config.actor_rollout_ref.model.get("tokenizer_path") or config.actor_rollout_ref.model.path
    tokenizer_local_path = copy_local_path_from_hdfs(tokenizer_src)
    tokenizer = hf_tokenizer(
        tokenizer_local_path,
        trust_remote_code=config.actor_rollout_ref.model.get("trust_remote_code", False),
    )

    if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.single_controller.ray import RayWorkerGroup
        from .fsdp_workers import RobActorRolloutRefWorker

        ray_worker_group_cls = RayWorkerGroup
        EnvWorker = None
        if not offline_only:
            from verl.experimental.vla.workers.env.env_worker import EnvWorker
    else:
        raise NotImplementedError

    role_worker_mapping = {Role.ActorRollout: ray.remote(RobActorRolloutRefWorker)}
    if not offline_only:
        role_worker_mapping[Role.Env] = ray.remote(EnvWorker)

    train_rollout_gpu_num = config.trainer.n_rollout_gpus_per_node
    train_rollout_nodes_num = config.trainer.nnodes

    resource_pool_spec = {"train_rollout_pool": [train_rollout_gpu_num] * train_rollout_nodes_num}
    mapping = {Role.ActorRollout: "train_rollout_pool"}

    if not offline_only:
        env_gpu_num = config.trainer.n_env_gpus_per_node
        env_nodes_num = config.env.disagg_sim.nnodes if config.env.disagg_sim.enable else config.trainer.nnodes
        resource_pool_spec["env_gpu_pool"] = [env_gpu_num] * env_nodes_num
        mapping[Role.Env] = "env_gpu_pool"

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    rlpd_dataset = None
    rlpd_sampler = None
    rlpd_collate_fn = None
    collate_fn = None
    train_sampler = None

    if config.trainer.rlpd_enable:
        dataset_type = config.actor_rollout_ref.model.override_config.dataset_type
        rlpd_root = config.data.rlpd_files
        rlpd_repo_id = OmegaConf.select(config, "data.rlpd_repo_id")
        if rlpd_repo_id is None:
            if dataset_type == "lerobot":
                rlpd_repo_id = os.path.basename(str(rlpd_root).rstrip("/")) or "lerobot"
            else:
                rlpd_repo_id = "parquet"

        rlpd_dataset, rlpd_sampler, rlpd_collate_fn = build_dataloader_components(
            dataset_type=dataset_type,
            repo_id=rlpd_repo_id,
            root=rlpd_root,
        )

    simulator_type = OmegaConf.select(config, "env.train.simulator_type")

    if offline_only:
        assert config.trainer.rlpd_enable, "offline_only=True requires trainer.rlpd_enable=True"
        assert rlpd_dataset is not None, "offline_only=True requires a valid rlpd_dataset"
        train_dataset = rlpd_dataset
        val_dataset = rlpd_dataset
        collate_fn = rlpd_collate_fn
        train_sampler = rlpd_sampler
    elif simulator_type == "robot":
        train_dataset = _build_robot_online_dummy_dataset(config)
        val_dataset = _build_robot_online_dummy_dataset(config)
    else:
        train_dataset = datasets.load_dataset("parquet", data_files=config.data.train_files)["train"]
        val_dataset = datasets.load_dataset("parquet", data_files=config.data.val_files)["train"]

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
        collate_fn=collate_fn,
        train_sampler=train_sampler,
        rlpd_dataset=rlpd_dataset if config.trainer.rlpd_enable else None,
        rlpd_sampler=rlpd_sampler if config.trainer.rlpd_enable else None,
        rlpd_collate_fn=rlpd_collate_fn if config.trainer.rlpd_enable else None,
    )

    trainer.init_workers()
    trainer.fit()


if __name__ == "__main__":
    main()
