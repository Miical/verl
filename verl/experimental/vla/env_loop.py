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

OBS_KEY = "obs"
ACTION_KEY = "action"
FEEDBACK_KEY = "feedback"

FEEDBACK_FIELDS = ["rewards", "terminations", "truncations"]


class EnvLoop:
    """An env loop manages interactions between models and vectorized environments. It's designed for computationally
    intensive environments, such as robotics simulators."""

    def __init__(self, env_wg: RayWorkerGroup, rollout_wg: RayWorkerGroup, config: DictConfig):
        """
        Initialize the EnvLoop.

        Args:
            env_wg (RayWorkerGroup): Environment worker group.
            rollout_wg (RayWorkerGroup): Rollout worker group for model inference.
            config (DictConfig): YAML config.
        """
        self.env_wg = env_wg
        self.rollout_wg = rollout_wg
        self.config = config

        # Extract relevant configuration
        self.max_interactions = config.env.train.max_episode_steps // config.env.actor.model.num_action_chunks
        self.stage_num = config.env.rollout.pipeline_stage_num
        self.num_envs_per_worker = config.env.train.num_envs
        self.action_dim = config.env.actor.model.action_dim
        self.num_action_chunks = config.env.actor.model.num_action_chunks
        self.single_env_rollout = bool(config.env.train.get("single_env_rollout", False))
        if self.single_env_rollout:
            assert self.stage_num == 1, "single_env_rollout only supports pipeline_stage_num == 1"
        # Derived properties
        self.total_envs = self.env_wg.world_size * self.num_envs_per_worker
        if self.total_envs % self.stage_num != 0:
            raise ValueError(f"Total envs ({self.total_envs}) must be divisible by stage_num ({self.stage_num})")
        self.envs_per_stage = self.total_envs // self.stage_num

        self.env_wg.init_worker()
        self.env_wg.init_simulator()

    def generate_sequences(self, prompts: DataProto, reset_future: asyncio.Future) -> DataProto:
        """Split input batch and dispatch to env loop workers.

        Args:
            prompts (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
        """

        reset_results = reset_future.get()

        loop = asyncio.get_event_loop()
        self.rollout_wg.switch_to_rollout()
        output = loop.run_until_complete(self.run(prompts, reset_results))
        self.rollout_wg.switch_to_train()
        # TODO(caiyunke.astra): add timing metrics
        return output

    async def run(self, prompts: DataProto, reset_results: DataProto) -> DataProto:
        """
        Run the environment interaction loop.
        This method orchestrates a pipelined process:
        1. Resets environments to specified initial states.
        2. In a loop, it gets actions from the rollout workers and applies them to the environments.
        3. Collects all trajectory data (observations, actions, rewards, dones).
        4. Formats and returns the collected trajectories as a single batch.
        Args:
            prompts (DataProto): Contains initial state IDs and other settings.
                                 - 'non_tensor_batch.state_ids': A numpy array of state IDs to reset envs.
        Returns:
            DataProto: A batch containing the complete trajectories.
        """
        trajectories = {i: [] for i in range(self.stage_num)}  

        initial_state_ids = prompts.non_tensor_batch["state_ids"]
        staged_obs = self._restructure_obs_data(reset_results)
        for stage_id in range(self.stage_num):
            trajectories[stage_id].append({OBS_KEY: staged_obs[stage_id]})
        if self.single_env_rollout:
            staged_obs = [staged_obs[0].repeat(repeat_times=len(prompts), interleave=True)]

        # --- Pipeline state ---
        rollout_futures = {}
        for stage_id in range(self.stage_num):
            vla_input = staged_obs[stage_id]
            vla_input.meta_info = prompts.meta_info  # Pass along rollout config
            rollout_futures[stage_id] = self.rollout_wg.generate_sequences(vla_input)

        async def _stage_loop(stage_id: int):
            for step_idx in range(self.max_interactions):
                # get actions from rollout worker
                if stage_id == 0:
                    logger.info(f"[{step_idx}/{self.max_interactions - 1}] rollout step")
                action_result: DataProto = await asyncio.to_thread(rollout_futures[stage_id].get)

                # send actions to env worker 
                stored_action_result = action_result[:1] if self.single_env_rollout else action_result
                trajectories[stage_id][-1][ACTION_KEY] = stored_action_result
                action_batch_size = len(action_result)
                if self.single_env_rollout:
                    env_action = action_result.batch["action"][:1].cpu().numpy()
                    env_critic_values = action_result.batch["critic_value"][:1].cpu().numpy()
                else:
                    env_action = action_result.batch["action"].cpu().numpy()
                    env_critic_values = action_result.batch["critic_value"].cpu().numpy()
                action_data = DataProto.from_dict(
                    non_tensors={
                        "actions": env_action,
                        "critic_values": env_critic_values,
                    },
                    meta_info={"stage_id": stage_id},
                )
                env_ref = self.env_wg.env_interact_step(action_data)

                # get results from env worker
                env_result: DataProto = await asyncio.to_thread(env_ref.get)
                feedback = DataProto.from_dict(tensors={field: env_result.batch[field] for field in FEEDBACK_FIELDS})
                next_obs = DataProto.from_dict(
                    tensors={key[len(OBS_KEY + "."):]: value for key, value in env_result.batch.items() if key.startswith(OBS_KEY + ".")},  
                    non_tensors={key[len(OBS_KEY + "."):]: value for key, value in env_result.non_tensor_batch.items() if key.startswith(OBS_KEY + ".")}
                )
                trajectories[stage_id][-1][FEEDBACK_KEY] = feedback
                if step_idx < self.max_interactions - 1:
                    trajectories[stage_id].append({OBS_KEY: next_obs})

                # send next obs to rollout worker for next step
                if step_idx < self.max_interactions - 1:
                    vla_input = next_obs
                    if self.single_env_rollout:
                        vla_input = vla_input.repeat(repeat_times=action_batch_size, interleave=True)
                    vla_input.meta_info = prompts.meta_info
                    rollout_futures[stage_id] = self.rollout_wg.generate_sequences(vla_input)

        await asyncio.gather(*[asyncio.create_task(_stage_loop(sid)) for sid in range(self.stage_num)])
        self.env_wg.finish_rollout()

        return self._collate_trajectories(trajectories, initial_state_ids, meta_info=prompts.meta_info)

    def _restructure_obs_data(self, data_proto: DataProto) -> list[DataProto]:
        """Reshapes flat observation data from env_wg into a list of per-stage DataProto objects."""
        # env_wg returns a flat batch ordered by [worker0_stage0, worker0_stage1, ...,
        # worker1_stage0, worker1_stage1, ...]
        # First, un-flatten by worker, then by stage

        num_workers = self.env_wg.world_size

        staged_data = [[] for _ in range(self.stage_num)]
        chunks = data_proto.chunk(num_workers)
        for worker_chunk in chunks:
            stage_chunks = worker_chunk.chunk(self.stage_num)
            for stage_id, data in enumerate(stage_chunks):
                staged_data[stage_id].append(data)

        # Concatenate data from all workers for each stage
        return [DataProto.concat(data_list) for data_list in staged_data]

    def _collate_trajectories(self, trajectories: dict, initial_state_ids: np.ndarray, meta_info) -> DataProto:
        """
        Collates the collected trajectory data into the final batch format.
        """
        flat_trajs = [{} for _ in range(len(trajectories[0]))]
        for stage_id in range(self.stage_num):
            for step_idx, step_data in enumerate(trajectories[stage_id]):
                if not flat_trajs[step_idx]:  # if dict is empty
                    flat_trajs[step_idx] = step_data
                else:
                    # Concatenate DataProto objects
                    for key, value in step_data.items():
                        if isinstance(value, DataProto):
                            flat_trajs[step_idx][key] = DataProto.concat([flat_trajs[step_idx][key], value])
                        elif isinstance(value, torch.Tensor):
                            flat_trajs[step_idx][key] = torch.cat([flat_trajs[step_idx][key], value], dim=0)

        batch_dict = {}
        for field_key in [OBS_KEY, ACTION_KEY, FEEDBACK_KEY]:
            for key in flat_trajs[0][field_key].batch.keys():
                per_step_values = [step[field_key].batch[key] for step in flat_trajs]
                batch_dict[f"{field_key}.{key}"] = torch.stack(per_step_values, dim=1)
            for key in flat_trajs[0][field_key].non_tensor_batch.keys():
                per_step_values = [step[field_key].non_tensor_batch[key] for step in flat_trajs]
                batch_dict[f"{field_key}.{key}"] = np.stack(per_step_values, axis=1)


        return DataProto.from_single_dict(batch_dict, meta_info=meta_info)
