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
INTERVENTION_INFO_KEY = "intervention_info"

FEEDBACK_FIELDS = ["rewards", "terminations", "truncations"]



def get_dataproto_from_prefix(data: DataProto, prefix: str, separator: str = "") -> DataProto:
    """Extract a sub-DataProto from a DataProto based on a given prefix.

    Args:
        data: The input DataProto containing various keys.
        prefix: The prefix string to filter keys.
        separator: Optional separator appended after prefix when matching keys.
    Returns:
        A DataProto containing tensor and non-tensor entries whose keys start
        with the specified prefix. The prefix is removed from the result keys.
    """

    match_prefix = prefix if not separator or prefix.endswith(separator) else f"{prefix}{separator}"
    prefix_length = len(match_prefix)
    tensor_batch = {}
    non_tensor_batch = {}
    for key in data.batch.keys():
        if key.startswith(match_prefix):
            new_key = key[prefix_length:]
            tensor_batch[new_key] = data.batch[key]
    for key in data.non_tensor_batch.keys():
        if key.startswith(match_prefix):
            new_key = key[prefix_length:]
            non_tensor_batch[new_key] = data.non_tensor_batch[key]
    return DataProto.from_dict(tensors=tensor_batch, non_tensors=non_tensor_batch, meta_info=data.meta_info)


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

    def _extract_intervention_obs_at(self, intervention_info: DataProto, obs_idx: int) -> DataProto:
        tensor_batch = {
            key.removeprefix(OBS_KEY + "."): value[:, obs_idx]
            for key, value in intervention_info.batch.items()
            if key.startswith(OBS_KEY + ".")
        }
        non_tensor_batch = {
            key.removeprefix(OBS_KEY + "."): value[:, obs_idx]
            for key, value in intervention_info.non_tensor_batch.items()
            if key.startswith(OBS_KEY + ".")
        }
        return DataProto.from_dict(tensors=tensor_batch, non_tensors=non_tensor_batch)

    def _expand_single_env_intervention_steps(
        self,
        current_obs: DataProto,
        rollout_action: DataProto,
        feedback: DataProto,
        intervention_info: DataProto,
        max_chunks: int,
    ) -> list[dict]:
        total_steps = intervention_info.batch["actions"].shape[1]
        aligned_steps = (total_steps // self.num_action_chunks) * self.num_action_chunks
        if aligned_steps == 0 or max_chunks <= 0:
            return []

        num_chunks = min(aligned_steps // self.num_action_chunks, max_chunks)
        critic_value_template = rollout_action.batch["critic_value"][:1]

        expanded_steps = []
        chunk_obs = current_obs
        for chunk_idx in range(num_chunks):
            start = chunk_idx * self.num_action_chunks
            end = start + self.num_action_chunks

            action_chunk = intervention_info.batch["actions"][:, start:end].clone()
            if chunk_idx == 0:
                intervention_mask = intervention_info.batch["is_intervention"][:, start:end].to(torch.bool)
                rollout_action_chunk = rollout_action.batch["action"][:1, : action_chunk.shape[1], : action_chunk.shape[2]]
                action_chunk = torch.where(
                    intervention_mask.unsqueeze(-1).to(action_chunk.device),
                    action_chunk,
                    rollout_action_chunk.to(device=action_chunk.device, dtype=action_chunk.dtype),
                )

            action_dp = DataProto.from_dict(
                tensors={
                    "action": action_chunk,
                    "critic_value": critic_value_template.new_zeros(critic_value_template.shape),
                }
            )
            feedback_dp = DataProto.from_dict(
                tensors={field: feedback.batch[field][:, start:end] for field in FEEDBACK_FIELDS}
            )
            expanded_steps.append({OBS_KEY: chunk_obs, ACTION_KEY: action_dp, FEEDBACK_KEY: feedback_dp})

            if chunk_idx < num_chunks - 1:
                chunk_obs = self._extract_intervention_obs_at(intervention_info, chunk_idx)

        return expanded_steps

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
            step_idx = 0
            while step_idx < self.max_interactions:
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

                # get results from env worker and process human intervention
                env_result: DataProto = await asyncio.to_thread(env_ref.get)
                feedback = DataProto.from_dict(tensors={field: env_result.batch[field] for field in FEEDBACK_FIELDS})
                intervention_info = get_dataproto_from_prefix(env_result, INTERVENTION_INFO_KEY, ".")
                next_obs = get_dataproto_from_prefix(env_result, OBS_KEY, ".")

                expanded_steps = []
                effective_steps = 1
                current_slot = trajectories[stage_id].pop()
                has_intervention = self.single_env_rollout and intervention_info.batch is not None
                if has_intervention:
                    expanded_steps = self._expand_single_env_intervention_steps(
                        current_obs=current_slot[OBS_KEY],
                        rollout_action=stored_action_result,
                        feedback=feedback,
                        intervention_info=intervention_info,
                        max_chunks=self.max_interactions - step_idx,
                    )

                if expanded_steps:
                    trajectories[stage_id].extend(expanded_steps)
                    effective_steps = len(expanded_steps)
                else:
                    current_slot[FEEDBACK_KEY] = feedback
                    current_slot[INTERVENTION_INFO_KEY] = intervention_info
                    trajectories[stage_id].append(current_slot)

                step_idx += effective_steps
                if step_idx < self.max_interactions:
                    trajectories[stage_id].append({OBS_KEY: next_obs})

                # send next obs to rollout worker for next step
                if step_idx < self.max_interactions:
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

        print(f"Final batch dict: {DataProto.from_single_dict(batch_dict, meta_info=meta_info)}")
        return DataProto.from_single_dict(batch_dict, meta_info=meta_info)
