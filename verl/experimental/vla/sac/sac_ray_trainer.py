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

import asyncio
import itertools
import uuid
from pprint import pprint
from typing import Optional

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager
from verl.trainer.ppo.utils import Role, WorkerType
from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics


REQUIRED_TRAIN_BATCH_KEYS = [
    "a0.full_action",
    "a1.full_action",
    "a0.action_loss_mask",
    "a1.action_loss_mask",
    "s0.states",
    "s1.states",
    "s0.images",
    "s1.images",
    "s0.image_masks",
    "s1.image_masks",
    "s0.lang_tokens",
    "s1.lang_tokens",
    "s0.lang_masks",
    "s1.lang_masks",
    "dones",
    "valids",
    "rewards",
    "positive_sample_mask",
    "task_ids",
]


def compute_avg_positive_trajectory_length(batch: DataProto) -> float:
    dones = batch.batch["dones"].bool()
    positive_mask = batch.batch["positive_sample_mask"]
    positive_traj = positive_mask.any(dim=1)

    if positive_traj.sum() == 0:
        return 0.0

    done_idx = torch.argmax(dones.int(), dim=1)
    traj_lens = done_idx + 1
    return traj_lens[positive_traj].float().mean().item()


def flatten_trajectories(data: DataProto) -> DataProto:
    batch_size, num_steps = data.batch["action"].shape[:2]
    new_batch_fields = {}

    for key, tensor in data.batch.items():
        if len(tensor.shape) >= 2 and tensor.shape[0] == batch_size and tensor.shape[1] == num_steps:
            new_shape = (batch_size * num_steps, *tensor.shape[2:])
            new_batch_fields[key] = tensor.reshape(new_shape)
        elif len(tensor.shape) == 1 and tensor.shape[0] == batch_size:
            new_batch_fields[key] = tensor.repeat_interleave(num_steps)
        else:
            new_batch_fields[key] = tensor

    return DataProto.from_dict(tensors=new_batch_fields, meta_info=data.meta_info)


def add_transition_prefixes(data: DataProto) -> DataProto:
    batch = data.batch
    step_key = "action" if "action" in batch else "full_action"
    if step_key not in batch:
        return data

    num_steps = batch[step_key].shape[1]
    if num_steps <= 1:
        return data

    def drop_last(tensor: torch.Tensor) -> torch.Tensor:
        return tensor[:, :-1, ...]

    def shift_next(tensor: torch.Tensor) -> torch.Tensor:
        return tensor[:, 1:, ...]

    state_keys = ["states", "images", "image_masks", "lang_tokens", "lang_masks"]
    action_keys = ["full_action", "action"]

    for key in state_keys:
        if key in batch:
            batch[f"s0.{key}"] = drop_last(batch[key])
            batch[f"s1.{key}"] = shift_next(batch[key])

    for key in action_keys:
        if key in batch:
            batch[f"a0.{key}"] = drop_last(batch[key])
            batch[f"a1.{key}"] = shift_next(batch[key])

    batch_size = batch[step_key].shape[0]
    for key, tensor in list(batch.items()):
        if tensor.ndim >= 2 and tensor.shape[0] == batch_size and tensor.shape[1] == num_steps:
            batch[key] = drop_last(tensor)

    return data


def select_required_train_fields(data: DataProto, tag: str) -> DataProto:
    missing = [k for k in REQUIRED_TRAIN_BATCH_KEYS if k not in data.batch]
    if missing:
        raise KeyError(f"{tag} is missing required train keys: {missing}")
    return data.select(REQUIRED_TRAIN_BATCH_KEYS)


class RobRaySACTrainer(RayPPOTrainer):
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        rlpd_dataset: Optional[Dataset] = None,
        rlpd_sampler: Optional[Sampler] = None,
        rlpd_collate_fn=None,
        device_name=None,
    ):
        super().__init__(
            config=config,
            tokenizer=tokenizer,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            processor=processor,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=device_name,
        )
        # SAC 的 critic 在 actor_module 内部，不存在独立 critic worker group
        self.use_critic = False

        # 工程兜底：即使有人误访问也不会 AttributeError
        self.critic_wg = None
        
        self.rlpd_dataloader = None
        if self.config.trainer.rlpd_enable:
            assert rlpd_dataset is not None, "rlpd_dataset must be provided when rlpd_enable is True"
            self.rlpd_dataloader = StatefulDataLoader(
                rlpd_dataset,
                batch_size=self.config.data.rlpd_batch_size,
                sampler=rlpd_sampler,
                collate_fn=rlpd_collate_fn,
                num_workers=self.config.data.dataloader_num_workers,
                drop_last=True,
            )
            assert len(self.rlpd_dataloader) >= 1, "RLPD dataloader is empty!"
            print(f"Size of RLPD dataloader: {len(self.rlpd_dataloader)}")
            print(f"Total training steps: {self.config.trainer.total_epochs * len(self.rlpd_dataloader)}")

    def _start_profiling(self, do_profile: bool) -> None:
        super()._start_profiling(do_profile)
        if do_profile and hasattr(self, "env_wg") and self.env_wg is not None:
            self.env_wg.start_profile(role="env", profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        super()._stop_profiling(do_profile)
        if do_profile and hasattr(self, "env_wg") and self.env_wg is not None:
            self.env_wg.stop_profile()

    def init_workers(self):
        self.resource_pool_manager.create_resource_pool()
        offline_only = bool(self.config.trainer.get("offline_only", False))

        if (not offline_only) and self.config.env.disagg_sim.enable:
            self.resource_pool_manager.get_resource_pool(Role.Env).accelerator_type = "sim"
            self.resource_pool_manager.get_resource_pool(Role.ActorRollout).accelerator_type = "train_rollout"

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}
        resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
        actor_rollout_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[Role.ActorRollout],
            config=self.config.actor_rollout_ref,
            role="actor_rollout",
        )
        self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls

        if not offline_only:
            assert Role.Env in self.role_worker_mapping
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Env)
            env_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.Env], config=self.config.env)
            self.resource_pool_to_cls[resource_pool]["env"] = env_cls

        all_wg = {}
        wg_kwargs = {}
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options") is not None
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls, **wg_kwargs)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()
        self.env_wg = all_wg.get("env", None)

        self.async_rollout_mode = False
        if (not offline_only) and self.config.actor_rollout_ref.rollout.mode == "async_envloop":
            from verl.experimental.vla.env_loop import EnvLoop
            self.async_rollout_mode = True
            self.async_rollout_manager = EnvLoop(config=self.config, rollout_wg=self.actor_rollout_wg, env_wg=self.env_wg)

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        batch_keys_to_pop = []
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys())
        gen_batch = batch.pop(batch_keys=batch_keys_to_pop, non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop))
        return gen_batch

    def _reset_envs(self, gen_batch: DataProto) -> asyncio.Future:
        initial_state_ids = gen_batch.non_tensor_batch["state_ids"]
        task_ids = gen_batch.non_tensor_batch["task_ids"]
        reset_prompts = DataProto.from_dict(non_tensors={"state_ids": initial_state_ids, "task_ids": task_ids})
        return self.env_wg.reset_envs_to_state_ids(reset_prompts)

    def fit(self):
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()
        offline_only = bool(self.config.trainer.get("offline_only", False))

        if (not offline_only) and self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if offline_only:
            assert self.rlpd_dataloader is not None, "offline_only=True requires rlpd_dataloader"
            self.total_training_steps = self.config.trainer.total_epochs * len(self.rlpd_dataloader)
        else:
            self.total_training_steps = self.config.trainer.total_epochs * len(self.train_dataloader) * self.config.trainer.rollout_interval
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = self.global_steps in self.config.global_profiler.steps if self.config.global_profiler.steps is not None else False
        next_step_profile = False

        rlpd_dataloader_iter = None
        if self.config.trainer.rlpd_enable:
            assert self.rlpd_dataloader is not None
            rlpd_dataloader_iter = itertools.cycle(self.rlpd_dataloader)

        for epoch in range(self.config.trainer.total_epochs):
            if offline_only:
                assert self.rlpd_dataloader is not None
                print(f"Starting offline-only epoch {epoch}, rlpd dataloader length: {len(self.rlpd_dataloader)}")
                for offline_step, rlpd_batch_raw in enumerate(self.rlpd_dataloader):
                    metrics = {}
                    timing_raw = {}
                    is_last_step = self.global_steps >= self.total_training_steps

                    with marked_timer("start_profile", timing_raw):
                        self._start_profiling((not prev_step_profile and curr_step_profile) if self.config.global_profiler.profile_continuous_steps else curr_step_profile)

                    with marked_timer("step", timing_raw):
                        rlpd_batch = DataProto.from_single_dict(rlpd_batch_raw)
                        rlpd_batch = self.actor_rollout_wg.process_dataset_batch(rlpd_batch).get()
                        rlpd_batch = select_required_train_fields(rlpd_batch, tag="offline-only RLPD batch")
                        rlpd_batch.meta_info["global_token_num"] = [0]
                        rlpd_batch.meta_info["global_steps"] = self.global_steps
                        metrics["data/offline_reward_mean"] = rlpd_batch.batch["rewards"].float().mean().item()
                        metrics["data/offline_done_ratio"] = rlpd_batch.batch["dones"].float().mean().item()
                        metrics["data/offline_positive_ratio"] = rlpd_batch.batch["positive_sample_mask"].float().mean().item()

                        with marked_timer("update_actor", timing_raw, color="red"):
                            actor_output = self.actor_rollout_wg.update_actor(rlpd_batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    esi_close_to_expiration = should_save_ckpt_esi(max_steps_duration=self.max_steps_duration, redundant_time=self.config.trainer.esi_redundant_time)
                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration):
                        if esi_close_to_expiration:
                            print("Force saving checkpoint: ESI instance expiration approaching.")
                        with marked_timer("save_checkpoint", timing_raw, color="green"):
                            self._save_checkpoint()

                    with marked_timer("stop_profile", timing_raw):
                        next_step_profile = self.global_steps + 1 in self.config.global_profiler.steps if self.config.global_profiler.steps is not None else False
                        self._stop_profiling((curr_step_profile and not next_step_profile) if self.config.global_profiler.profile_continuous_steps else curr_step_profile)
                        prev_step_profile = curr_step_profile
                        curr_step_profile = next_step_profile

                    steps_duration = timing_raw["step"]
                    self.max_steps_duration = max(self.max_steps_duration, steps_duration)
                    metrics.update({"training/global_step": self.global_steps, "training/epoch": epoch, "training/offline_step": offline_step})
                    logger.log(data=metrics, step=self.global_steps)
                    progress_bar.update(1)
                    self.global_steps += 1

                    if hasattr(self.config.actor_rollout_ref.actor, "profiler") and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory":
                        self.actor_rollout_wg.dump_memory_snapshot(tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}")

                    if is_last_step:
                        pprint(f"Final validation metrics: {last_val_metrics}")
                        progress_bar.close()
                        return
                continue

            train_iter = iter(self.train_dataloader)
            next_batch_dict = next(train_iter)
            dataloader_len = len(self.train_dataloader)
            print(f"Starting epoch {epoch}, dataloader length: {dataloader_len}")

            for dataloader_step in range(dataloader_len):
                for training_step in range(self.config.trainer.rollout_interval):
                    metrics = {}
                    timing_raw = {}
                    task_ids_from_dataloader = None
                    need_rollout = training_step == 0
                    is_last_step = self.global_steps >= self.total_training_steps

                    with marked_timer("start_profile", timing_raw):
                        self._start_profiling((not prev_step_profile and curr_step_profile) if self.config.global_profiler.profile_continuous_steps else curr_step_profile)

                    if need_rollout:
                        batch_dict = next_batch_dict
                        try:
                            next_batch_dict = next(train_iter)
                        except StopIteration:
                            next_batch_dict = None

                        task_ids_from_dataloader = [batch_dict["extra_info"][task_i]["task_ids"] for task_i in range(len(batch_dict["extra_info"]))]
                        batch: DataProto = DataProto.from_single_dict(batch_dict)
                        batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch))], dtype=object)
                        gen_batch = self._get_gen_batch(batch)
                        gen_batch.meta_info["global_steps"] = self.global_steps
                        gen_batch.meta_info["do_sample"] = True
                        gen_batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
                        gen_batch.meta_info["prompt_length"] = self.config.actor_rollout_ref.rollout.prompt_length
                        gen_batch.meta_info["eos_token_id"] = self.tokenizer.eos_token_id
                        gen_batch.meta_info["n_samples"] = self.config.actor_rollout_ref.rollout.n
                        gen_batch.meta_info["pad_token_id"] = self.tokenizer.pad_token_id
                        gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                        if dataloader_step == 0:
                            reset_future = self._reset_envs(gen_batch)

                    with marked_timer("step", timing_raw):
                        if need_rollout:
                            with marked_timer("gen", timing_raw, color="red"):
                                batch = self.async_rollout_manager.generate_sequences(gen_batch, reset_future)

                            if dataloader_step != dataloader_len - 1:
                                next_batch: DataProto = DataProto.from_single_dict(next_batch_dict)
                                next_gen_batch = self._get_gen_batch(next_batch)
                                next_gen_batch = next_gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                                reset_future = self._reset_envs(next_gen_batch)

                            complete_any = batch.batch["complete"].any(dim=-1)
                            dones_step = complete_any.clone()
                            dones_step[:, -2] = True
                            batch.batch["dones"] = dones_step.float()
                            sparse_rewards = complete_any.float()
                            batch.batch["valids"] = (~batch.batch["complete"]).any(dim=-1).float()
                            step_penalty = float(self.config.env.train.get("step_penalty", 0.0))
                            batch.batch["rewards"] = sparse_rewards - step_penalty * batch.batch["valids"]
                            batch.batch["rewards"][:, -2] = -1.0
                            batch.batch["positive_sample_mask"] = sparse_rewards.any(dim=-1).unsqueeze(-1).repeat_interleave(batch.batch["action"].shape[1], dim=-1)
                            average_reward = sparse_rewards.any(dim=-1).mean(dtype=torch.float32).item()
                            metrics["data/trajectory_avg_reward"] = average_reward
                            metrics["data/avg_positive_trajectory_length"] = compute_avg_positive_trajectory_length(batch)
                            batch = add_transition_prefixes(batch)
                            assert task_ids_from_dataloader is not None
                            rollout_level_task_ids = [task_id for task_id in task_ids_from_dataloader for _ in range(self.config.actor_rollout_ref.rollout.n)]
                            batch.batch["task_ids"] = torch.tensor(rollout_level_task_ids, dtype=torch.long, device=batch.batch["dones"].device)
                            batch = flatten_trajectories(batch)
                            batch.meta_info["global_token_num"] = [0]
                            rl_batch = select_required_train_fields(batch, tag="online RL batch")

                            if self.config.trainer.rlpd_enable:
                                assert rlpd_dataloader_iter is not None
                                rlpd_batch_raw = next(rlpd_dataloader_iter)
                                rlpd_batch = DataProto.from_single_dict(rlpd_batch_raw)
                                rlpd_batch = self.actor_rollout_wg.process_dataset_batch(rlpd_batch).get()
                                rlpd_batch = select_required_train_fields(rlpd_batch, tag="RLPD batch")
                                train_batch = DataProto.concat([rl_batch, rlpd_batch])
                                train_batch.meta_info["global_token_num"] = [0]
                                train_batch.meta_info["global_steps"] = self.global_steps
                            else:
                                train_batch = rl_batch
                                train_batch.meta_info["global_token_num"] = [0]
                                train_batch.meta_info["global_steps"] = self.global_steps

                            with marked_timer("update_actor", timing_raw, color="red"):
                                actor_output = self.actor_rollout_wg.update_actor(train_batch)
                            actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        else:
                            with marked_timer("update_actor", timing_raw, color="red"):
                                actor_output = self.actor_rollout_wg.update_actor(DataProto(meta_info={"empty_batch": True, "global_steps": self.global_steps, "global_token_num": [0]}))
                            actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    esi_close_to_expiration = should_save_ckpt_esi(max_steps_duration=self.max_steps_duration, redundant_time=self.config.trainer.esi_redundant_time)
                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration):
                        if esi_close_to_expiration:
                            print("Force saving checkpoint: ESI instance expiration approaching.")
                        with marked_timer("save_checkpoint", timing_raw, color="green"):
                            self._save_checkpoint()

                    with marked_timer("stop_profile", timing_raw):
                        next_step_profile = self.global_steps + 1 in self.config.global_profiler.steps if self.config.global_profiler.steps is not None else False
                        self._stop_profiling((curr_step_profile and not next_step_profile) if self.config.global_profiler.profile_continuous_steps else curr_step_profile)
                        prev_step_profile = curr_step_profile
                        curr_step_profile = next_step_profile

                    steps_duration = timing_raw["step"]
                    self.max_steps_duration = max(self.max_steps_duration, steps_duration)
                    metrics.update({"training/global_step": self.global_steps, "training/epoch": epoch})
                    logger.log(data=metrics, step=self.global_steps)
                    progress_bar.update(1)
                    self.global_steps += 1

                    if hasattr(self.config.actor_rollout_ref.actor, "profiler") and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory":
                        self.actor_rollout_wg.dump_memory_snapshot(tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}")

                    if is_last_step:
                        pprint(f"Final validation metrics: {last_val_metrics}")
                        progress_bar.close()
                        return

                    if hasattr(self.train_dataset, "on_batch_end"):
                        self.train_dataset.on_batch_end(batch=batch)
