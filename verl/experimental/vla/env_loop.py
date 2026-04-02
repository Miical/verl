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
import logging
import os

import numpy as np
import torch
from omegaconf import DictConfig

from verl import DataProto
from verl.single_controller.ray import RayWorkerGroup

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class EnvLoop:
    """An env loop manages interactions between models and vectorized environments.

    Supports two modes:
      - **episode mode** (original): resets all envs per training step, runs a full
        episode of ``max_episode_steps // num_action_chunks`` macro-steps.
      - **rolling mode** (``rolling_horizon > 0``): envs persist across training
        steps with IsaacLab auto-reset.  Each ``generate_sequences`` call runs a
        short window of ``rolling_horizon`` macro-steps.
    """

    def __init__(self, env_wg: RayWorkerGroup, rollout_wg: RayWorkerGroup, config: DictConfig):
        self.env_wg = env_wg
        self.rollout_wg = rollout_wg
        self.config = config

        self.stage_num = config.env.rollout.pipeline_stage_num
        self.num_envs_per_worker = config.env.train.num_envs
        self.action_dim = config.env.actor.model.action_dim
        self.num_action_chunks = config.env.actor.model.num_action_chunks

        self.total_envs = self.env_wg.world_size * self.num_envs_per_worker
        if self.total_envs % self.stage_num != 0:
            raise ValueError(f"Total envs ({self.total_envs}) must be divisible by stage_num ({self.stage_num})")
        self.envs_per_stage = self.total_envs // self.stage_num

        self.rolling_horizon = int(config.env.train.get("rollout_horizon", 0))
        self.rolling_mode = self.rolling_horizon > 0

        if self.rolling_mode:
            self.max_interactions = self.rolling_horizon
            self._last_obs: DataProto | None = None
        else:
            self.max_interactions = config.env.train.max_episode_steps // self.num_action_chunks

        self.env_wg.init_worker()
        self.env_wg.init_simulator()

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def generate_sequences(self, prompts: DataProto, reset_future=None) -> DataProto:
        """Run a rollout window and return the collected trajectories.

        In **episode mode** (original), ``reset_future`` must be provided and
        is awaited to get the post-reset observations.

        In **rolling mode**, ``reset_future`` is ignored.  On the very first
        call the envs are reset internally; subsequent calls reuse the stored
        last observation.
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        self.rollout_wg.switch_to_rollout()

        if self.rolling_mode:
            output = loop.run_until_complete(self._run_rolling(prompts))
        else:
            assert reset_future is not None, "Episode mode requires reset_future"
            reset_results = reset_future.get()
            output = loop.run_until_complete(self.run(prompts, reset_results))

        self.rollout_wg.switch_to_train()
        return output

    # ------------------------------------------------------------------
    # Episode mode (original)
    # ------------------------------------------------------------------

    async def run(self, prompts: DataProto, reset_results: DataProto) -> DataProto:
        """Full-episode rollout (original behaviour)."""
        initial_state_ids = prompts.non_tensor_batch["state_ids"]

        staged_obs = self._restructure_obs_data(reset_results)
        trajectories = {i: [] for i in range(self.stage_num)}
        rollout_futures = {}

        for stage_id in range(self.stage_num):
            trajectories[stage_id].append({})
            vla_input = staged_obs[stage_id]
            vla_input.meta_info = prompts.meta_info
            rollout_futures[stage_id] = self.rollout_wg.generate_sequences(vla_input)

        async def _stage_loop(stage_id: int):
            for step_idx in range(self.max_interactions):
                if stage_id == 0:
                    logger.info(f"[{step_idx}/{self.max_interactions}] rollout step")
                action_result: DataProto = await asyncio.to_thread(rollout_futures[stage_id].get)

                trajectories[stage_id][-1]["action"] = action_result
                action_data = DataProto.from_dict(
                    non_tensors={
                        "actions": action_result.batch["action"].cpu().numpy(),
                        "critic_values": action_result.batch["critic_value"].cpu().numpy(),
                    },
                    meta_info={"stage_id": stage_id},
                )

                env_ref = self.env_wg.env_interact_step(action_data)
                env_result: DataProto = await asyncio.to_thread(env_ref.get)

                trajectories[stage_id][-1]["rew"] = env_result.batch["rews"]
                trajectories[stage_id][-1]["done"] = env_result.batch["terminations"]

                next_obs = DataProto(
                    batch=env_result.batch.select("full_image", "wrist_image", "state"),
                    non_tensor_batch={"task_descriptions": env_result.non_tensor_batch["task_descriptions"]},
                )

                if step_idx < self.max_interactions - 1:
                    trajectories[stage_id].append({})
                    vla_input = next_obs
                    vla_input.meta_info = prompts.meta_info
                    rollout_futures[stage_id] = self.rollout_wg.generate_sequences(vla_input)

        await asyncio.gather(*[asyncio.create_task(_stage_loop(sid)) for sid in range(self.stage_num)])
        self.env_wg.finish_rollout()

        return self._collate_trajectories(trajectories, initial_state_ids, meta_info=prompts.meta_info)

    # ------------------------------------------------------------------
    # Rolling mode
    # ------------------------------------------------------------------

    async def _run_rolling(self, prompts: DataProto) -> DataProto:
        """Short-horizon rolling rollout with persistent envs and auto-reset."""
        if self._last_obs is None:
            reset_result = await asyncio.to_thread(self._initial_reset)
            self._last_obs = self._restructure_obs_data(reset_result)

        staged_obs = self._last_obs
        trajectories = {i: [] for i in range(self.stage_num)}
        rollout_futures = {}

        for stage_id in range(self.stage_num):
            trajectories[stage_id].append({})
            vla_input = staged_obs[stage_id]
            vla_input.meta_info = prompts.meta_info
            rollout_futures[stage_id] = self.rollout_wg.generate_sequences(vla_input)

        last_obs_per_stage: dict[int, DataProto] = {}

        async def _stage_loop(stage_id: int):
            for step_idx in range(self.rolling_horizon):
                if stage_id == 0:
                    logger.info(f"[rolling {step_idx}/{self.rolling_horizon}] rollout step")

                action_result: DataProto = await asyncio.to_thread(rollout_futures[stage_id].get)

                trajectories[stage_id][-1]["action"] = action_result
                action_data = DataProto.from_dict(
                    non_tensors={
                        "actions": action_result.batch["action"].cpu().numpy(),
                        "critic_values": action_result.batch["critic_value"].cpu().numpy(),
                    },
                    meta_info={"stage_id": stage_id},
                )

                env_ref = self.env_wg.env_interact_step(action_data)
                env_result: DataProto = await asyncio.to_thread(env_ref.get)

                trajectories[stage_id][-1]["rew"] = env_result.batch["rews"]
                trajectories[stage_id][-1]["terminated"] = env_result.batch["terminations"]
                trajectories[stage_id][-1]["truncated"] = env_result.batch["truncations"]
                trajectories[stage_id][-1]["success"] = env_result.batch["successes"]
                if "valid_mask" in env_result.batch:
                    trajectories[stage_id][-1]["valid_mask"] = env_result.batch["valid_mask"]
                trajectories[stage_id][-1]["task_assignment_keys"] = env_result.non_tensor_batch["task_assignment_keys"]

                next_obs = DataProto(
                    batch=env_result.batch.select("full_image", "wrist_image", "state"),
                    non_tensor_batch={"task_descriptions": env_result.non_tensor_batch["task_descriptions"]},
                )

                if step_idx < self.rolling_horizon - 1:
                    trajectories[stage_id].append({})
                    vla_input = next_obs
                    vla_input.meta_info = prompts.meta_info
                    rollout_futures[stage_id] = self.rollout_wg.generate_sequences(vla_input)

                last_obs_per_stage[stage_id] = next_obs

        await asyncio.gather(*[asyncio.create_task(_stage_loop(sid)) for sid in range(self.stage_num)])
        self.env_wg.finish_rollout()

        self._last_obs = [last_obs_per_stage[sid] for sid in range(self.stage_num)]

        return self._collate_rolling_trajectories(trajectories, meta_info=prompts.meta_info)

    def _initial_reset(self) -> DataProto:
        """Reset all envs once (called on very first rolling rollout)."""
        num_total = self.total_envs * self.stage_num
        dummy_state_ids = np.zeros(num_total, dtype=object)
        dummy_task_ids = np.zeros(num_total, dtype=object)
        reset_prompts = DataProto.from_dict(
            non_tensors={"state_ids": dummy_state_ids, "task_ids": dummy_task_ids}
        )
        result = self.env_wg.reset_envs_to_state_ids(reset_prompts)
        return result.get() if hasattr(result, "get") else result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _restructure_obs_data(self, data_proto: DataProto) -> list[DataProto]:
        """Reshapes flat observation data from env_wg into a list of per-stage DataProto objects."""
        num_workers = self.env_wg.world_size

        staged_data = [[] for _ in range(self.stage_num)]
        chunks = data_proto.chunk(num_workers)
        for worker_chunk in chunks:
            stage_chunks = worker_chunk.chunk(self.stage_num)
            for stage_id, data in enumerate(stage_chunks):
                staged_data[stage_id].append(data)

        return [DataProto.concat(data_list) for data_list in staged_data]

    # ------------------------------------------------------------------
    # Collation
    # ------------------------------------------------------------------

    def _collate_trajectories(self, trajectories: dict, initial_state_ids: np.ndarray, meta_info) -> DataProto:
        """Collates trajectory data for episode mode (original)."""
        flat_trajs = [{} for _ in range(len(trajectories[0]))]
        for stage_id in range(self.stage_num):
            for step_idx, step_data in enumerate(trajectories[stage_id]):
                if not flat_trajs[step_idx]:
                    flat_trajs[step_idx] = step_data
                else:
                    for key, value in step_data.items():
                        if isinstance(value, DataProto):
                            flat_trajs[step_idx][key] = DataProto.concat([flat_trajs[step_idx][key], value])
                        elif isinstance(value, torch.Tensor):
                            flat_trajs[step_idx][key] = torch.cat([flat_trajs[step_idx][key], value], dim=0)

        batch_dict = {}
        action_batch_keys = list(flat_trajs[0]["action"].batch.keys())
        for key in action_batch_keys:
            per_step_values = [step["action"].batch[key] for step in flat_trajs]
            batch_dict[key] = torch.stack(per_step_values, dim=1)

        batch_dict["complete"] = torch.stack([step["done"] for step in flat_trajs], dim=1).squeeze(-1)
        batch_dict["env_state_id"] = torch.from_numpy(initial_state_ids.astype(int))

        return DataProto.from_single_dict(batch_dict, meta_info=meta_info)

    def _collate_rolling_trajectories(self, trajectories: dict, meta_info) -> DataProto:
        """Collates trajectory data for rolling mode.

        Includes ``env_rewards``, ``terminated``, ``truncated``, and ``success``
        from IsaacLab so the trainer can use them directly.
        """
        flat_trajs = [{} for _ in range(len(trajectories[0]))]
        for stage_id in range(self.stage_num):
            for step_idx, step_data in enumerate(trajectories[stage_id]):
                if not flat_trajs[step_idx]:
                    flat_trajs[step_idx] = step_data
                else:
                    for key, value in step_data.items():
                        if isinstance(value, DataProto):
                            flat_trajs[step_idx][key] = DataProto.concat([flat_trajs[step_idx][key], value])
                        elif isinstance(value, torch.Tensor):
                            flat_trajs[step_idx][key] = torch.cat([flat_trajs[step_idx][key], value], dim=0)
                        elif isinstance(value, np.ndarray):
                            flat_trajs[step_idx][key] = np.concatenate([flat_trajs[step_idx][key], value])
                        elif isinstance(value, list):
                            flat_trajs[step_idx][key] = flat_trajs[step_idx][key] + value

        batch_dict = {}
        action_batch_keys = list(flat_trajs[0]["action"].batch.keys())
        for key in action_batch_keys:
            per_step_values = [step["action"].batch[key] for step in flat_trajs]
            batch_dict[key] = torch.stack(per_step_values, dim=1)

        batch_dict["env_rewards"] = torch.stack([step["rew"] for step in flat_trajs], dim=1)
        batch_dict["terminated"] = torch.stack([step["terminated"] for step in flat_trajs], dim=1)
        batch_dict["truncated"] = torch.stack([step["truncated"] for step in flat_trajs], dim=1)
        batch_dict["success"] = torch.stack([step["success"] for step in flat_trajs], dim=1)
        if "valid_mask" in flat_trajs[0]:
            batch_dict["valid_mask"] = torch.stack([step["valid_mask"] for step in flat_trajs], dim=1)

        done_any = batch_dict["terminated"] | batch_dict["truncated"]
        batch_dict["complete"] = done_any.squeeze(-1) if done_any.dim() > 2 else done_any

        result = DataProto.from_single_dict(batch_dict, meta_info=meta_info)

        task_assignment_keys = flat_trajs[0].get("task_assignment_keys", [])
        if len(task_assignment_keys) > 0:
            result.non_tensor_batch["task_assignment_keys"] = np.array(task_assignment_keys, dtype=object)
        return result
