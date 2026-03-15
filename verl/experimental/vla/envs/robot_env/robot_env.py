# Copyright 2025 The RLinf Authors.
# Lightweight robot environment adapter for VERL.

import logging
from typing import Optional

import gymnasium as gym
import numpy as np
import torch

from .robot.controller.piper.piper_env import RealRobotEnvWrapper as PiperEnv
from verl.experimental.vla.envs.action_utils import list_of_dict_to_dict_of_list, to_tensor

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class RealRobotEnv(gym.Env):
    """Minimal adapter that plugs PiperEnv into VERL env worker API."""

    def __init__(self, cfg, rank, world_size):
        self.rank = rank
        self.cfg = cfg
        self.world_size = world_size
        self.num_envs = getattr(cfg, "num_envs", 1)

        self.auto_reset = getattr(cfg, "auto_reset", True)
        self.ignore_terminations = getattr(cfg, "ignore_terminations", False)
        self.max_episode_steps = getattr(cfg, "max_episode_steps", 1000)

        self.robot_backends = [PiperEnv(cfg, rank, world_size) for _ in range(self.num_envs)]
        self.task_descriptions = ["" for _ in range(self.num_envs)]
        self._elapsed_steps = np.zeros(self.num_envs, dtype=np.int32)

        logger.info(
            "[RealRobotEnv] initialized rank=%s num_envs=%s auto_reset=%s",
            rank,
            self.num_envs,
            self.auto_reset,
        )

    @property
    def elapsed_steps(self):
        return self._elapsed_steps

    def get_all_state_ids(self) -> np.ndarray:
        # real robot mode: no dataset state indexing
        return np.array([0], dtype=np.int32)

    def _wrap_obs(self, obs_list):
        images_list = []
        state_list = []

        for env_i, obs in enumerate(obs_list):
            images_list.append(obs["images"])
            state_list.append(obs["state"])
            self.task_descriptions[env_i] = obs.get("task_description", self.task_descriptions[env_i])

        images_dict = list_of_dict_to_dict_of_list(images_list)
        images_and_states = {}

        # pad encoded image tensors to same length per camera
        for key, tensors in images_dict.items():
            if not tensors:
                continue
            max_len = max(int(t.shape[-1]) for t in tensors)
            padded = []
            for t in tensors:
                if int(t.shape[-1]) < max_len:
                    pad = torch.zeros((t.shape[0], max_len - t.shape[-1]), dtype=t.dtype)
                    t = torch.cat([t, pad], dim=-1)
                padded.append(t)
            images_and_states[key] = torch.cat(padded, dim=0)

        images_and_states["state"] = to_tensor(np.stack(state_list, axis=0))

        return {
            "images_and_states": images_and_states,
            "task_descriptions": self.task_descriptions.copy(),
        }

    def reset(self, env_idx: Optional[np.ndarray] = None, reset_state_ids=None, options: Optional[dict] = None):
        del reset_state_ids, options
        if env_idx is None:
            env_idx = np.arange(self.num_envs)
        elif isinstance(env_idx, int):
            env_idx = np.asarray([env_idx])
        else:
            env_idx = np.asarray(env_idx)

        obs_list = []
        for eid in env_idx:
            obs_list.append(self.robot_backends[int(eid)].reset())
            self._elapsed_steps[int(eid)] = 0

        # if partial reset, merge untouched env observations from step-time cache is not needed in current flow
        # env_worker uses full reset at startup and auto-reset through step/chunk_step
        if len(env_idx) != self.num_envs:
            full_obs = [None] * self.num_envs
            for i, eid in enumerate(env_idx):
                full_obs[int(eid)] = obs_list[i]
            for i in range(self.num_envs):
                if full_obs[i] is None:
                    full_obs[i] = self.robot_backends[i].reset()
                    self._elapsed_steps[i] = 0
            obs_list = full_obs

        return self._wrap_obs(obs_list), {}

    def _auto_reset_done_envs(self, dones: np.ndarray, obs, infos):
        done_ids = np.where(dones)[0]
        if len(done_ids) == 0:
            return obs, infos

        final_obs = obs
        new_obs, new_infos = self.reset(env_idx=done_ids)
        infos.update(new_infos)
        infos["final_observation"] = final_obs
        infos["_final_observation"] = dones
        infos["final_info"] = {"episode": {}}
        infos["_final_info"] = dones
        return new_obs, infos

    def step(self, actions=None, auto_reset=True):
        if actions is None:
            obs, infos = self.reset()
            z = np.zeros(self.num_envs, dtype=bool)
            return obs, None, to_tensor(z), to_tensor(z), infos

        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()

        obs_list = []
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        terminations = np.zeros(self.num_envs, dtype=bool)
        truncations = np.zeros(self.num_envs, dtype=bool)
        intervene_actions = []
        intervene_flags = []

        self._elapsed_steps += 1

        for eid in range(self.num_envs):
            single_action = actions[eid] if actions.ndim > 1 else actions
            obs, reward, term, trunc, info = self.robot_backends[eid].step(single_action)
            obs_list.append(obs)
            rewards[eid] = reward
            terminations[eid] = term
            truncations[eid] = trunc
            intervene_actions.append(info.get("intervene_action", np.zeros(14, dtype=np.float32)))
            intervene_flags.append(bool(info.get("intervene_flag", False)))

        truncations = np.logical_or(truncations, self._elapsed_steps >= self.max_episode_steps)
        obs = self._wrap_obs(obs_list)

        infos = {
            "intervene_action": to_tensor(np.stack(intervene_actions, axis=0)),
            "intervene_flag": to_tensor(np.asarray(intervene_flags, dtype=bool)),
            "episode": {
                "episode_len": to_tensor(self._elapsed_steps.copy()),
                "return": to_tensor(rewards.copy()),
                "reward": to_tensor(rewards.copy()),
                "success_once": to_tensor(terminations.copy()),
            },
        }

        if self.ignore_terminations:
            terminations[:] = False

        dones = np.logical_or(terminations, truncations)
        if auto_reset and self.auto_reset and dones.any():
            obs, infos = self._auto_reset_done_envs(dones, obs, infos)

        return obs, to_tensor(rewards), to_tensor(terminations), to_tensor(truncations), infos

    def chunk_step(self, chunk_actions):
        # chunk_actions: [num_envs, chunk, action_dim]
        chunk = int(chunk_actions.shape[1])
        all_rewards = []
        all_terms = []
        all_truncs = []
        all_states = []
        last_obs = None
        last_infos = {}

        for i in range(chunk):
            obs, rew, term, trunc, infos = self.step(chunk_actions[:, i], auto_reset=False)
            last_obs, last_infos = obs, infos
            all_rewards.append(rew)
            all_terms.append(term)
            all_truncs.append(trunc)
            all_states.append(obs["images_and_states"]["state"])

        chunk_rewards = torch.stack(all_rewards, dim=1)
        raw_terms = torch.stack(all_terms, dim=1)
        raw_truncs = torch.stack(all_truncs, dim=1)

        # return last-step images + full chunk states
        chunk_obs = {
            "images_and_states": {
                "head_image": last_obs["images_and_states"].get("head_image"),
                "left_wrist_image": last_obs["images_and_states"].get("left_wrist_image"),
                "right_wrist_image": last_obs["images_and_states"].get("right_wrist_image"),
                "state": torch.stack(all_states, dim=1),
            },
            "task_descriptions": last_obs["task_descriptions"],
        }

        past_terms = raw_terms.any(dim=1)
        past_truncs = raw_truncs.any(dim=1)
        dones = torch.logical_or(past_terms, past_truncs)

        # env worker expects done only at last chunk step in auto_reset mode
        if self.auto_reset or self.ignore_terminations:
            chunk_terms = torch.zeros_like(raw_terms)
            chunk_truncs = torch.zeros_like(raw_truncs)
            chunk_terms[:, -1] = past_terms
            chunk_truncs[:, -1] = past_truncs
        else:
            chunk_terms = raw_terms
            chunk_truncs = raw_truncs

        if dones.any() and self.auto_reset:
            chunk_obs, last_infos = self._auto_reset_done_envs(dones.cpu().numpy(), chunk_obs, last_infos)

        return chunk_obs, chunk_rewards, chunk_terms, chunk_truncs, last_infos

    def reset_envs_to_state_ids(self, state_ids_list, task_ids_list):
        del state_ids_list, task_ids_list
        obs, _ = self.reset()
        task_descriptions = obs["task_descriptions"]
        return [
            {"images_and_states": obs["images_and_states"], "task_descriptions": task_descriptions},
            {"task_descriptions": task_descriptions},
        ]

    def close(self):
        for backend in self.robot_backends:
            backend.close()

    def get_state(self) -> bytes:
        return b""

    def load_state(self, state_buffer: bytes):
        del state_buffer
        return
