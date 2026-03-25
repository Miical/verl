# Copyright 2024 Bytedance Ltd. and/or its affiliates

import logging
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import sys
from typing import Optional

import gymnasium as gym
import numpy as np
import torch

from .ipc_channel import clear_ipc, send_obj

logger = logging.getLogger(__name__)
_LEROBOT_RUNTIME_SESSION_PREFIX = "lerobot_runtime"


class LeRobotEnv(gym.Env):
    """Scaffold environment for future LeRobot integration.

    This implementation only provides the interface expected by EnvWorker so
    the simulator type can be wired through the system before the real runtime
    is implemented.
    """

    def __init__(self, cfg, rank, world_size, stage_id: int = 0):
        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size
        self.stage_id = stage_id
        self.num_envs = cfg.num_envs
        assert self.num_envs == 1, f"LeRobotEnv only supports a single real-world environment, got {self.num_envs}"
        self.max_episode_steps = int(cfg.max_episode_steps)
        self.action_dim = int(getattr(cfg, "action_dim", 7))
        self.state_dim = int(getattr(cfg, "state_dim", 8))
        init_params = getattr(cfg, "init_params", None)
        self.image_height = int(getattr(init_params, "camera_heights", 256) if init_params is not None else 256)
        self.image_width = int(getattr(init_params, "camera_widths", 256) if init_params is not None else 256)
        self.video_cfg = cfg.video_cfg
        self.video_cnt = 0
        self.render_images = []

        self._episode_steps = np.zeros(self.num_envs, dtype=np.int32)
        self._episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self._state_ids = np.zeros(self.num_envs, dtype=np.int64)
        self._task_ids = np.zeros(self.num_envs, dtype=np.int64)
        self.task_descriptions = [self._task_description(0) for _ in range(self.num_envs)]
        self.total_num_group_envs = max(self.num_envs * self.world_size * 32, 1024)

        lerobot_config_path = getattr(cfg, "lerobot_config_path", None)
        if not lerobot_config_path:
            raise ValueError("`cfg.lerobot_config_path` is required for LeRobotEnv.")
        if not Path(lerobot_config_path).exists():
            raise FileNotFoundError(f"LeRobot config not found: {lerobot_config_path}")
        self._ensure_tmux_lerobot_runtime(lerobot_config_path)

    def _ensure_tmux_lerobot_runtime(self, lerobot_config_path: str) -> None:
        if shutil.which("tmux") is None:
            raise RuntimeError("`tmux` is required for LeRobotEnv runtime process.")

        session_name = f"{_LEROBOT_RUNTIME_SESSION_PREFIX}_{socket.gethostname()}_rank{self.rank}_stage{self.stage_id}"
        runtime_cmd = (
            f"{sys.executable} -u -m verl.experimental.vla.envs.lerobot_env.lerobot_runtime "
            f"--config_path {shlex.quote(str(lerobot_config_path))} "
            f"--rank {self.rank} --stage_id {self.stage_id}"
        )

        has_session = subprocess.run(
            ["tmux", "has-session", "-t", session_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if has_session.returncode == 0:
            logger.info("LeRobot runtime tmux session already exists: %s", session_name)
            return

        clear_ipc(rank=self.rank, stage_id=self.stage_id)

        subprocess.run(
            ["tmux", "new-session", "-d", "-s", session_name, runtime_cmd],
            check=True,
        )
        logger.info("Started LeRobot runtime in tmux session: %s", session_name)

    def _task_description(self, task_id: int) -> str:
        return f"lerobot_task_{task_id}"

    def _build_obs(self):
        full_image = torch.zeros((self.num_envs, self.image_height, self.image_width, 3), dtype=torch.uint8)
        wrist_image = torch.zeros((self.num_envs, self.image_height, self.image_width, 3), dtype=torch.uint8)
        state = torch.zeros((self.num_envs, self.state_dim), dtype=torch.float32)
        state[:, 0] = torch.as_tensor(self._state_ids, dtype=torch.float32)
        state[:, 1] = torch.as_tensor(self._task_ids, dtype=torch.float32)
        state[:, 2] = torch.as_tensor(self._episode_steps, dtype=torch.float32)
        return {
            "images_and_states": {
                "full_image": full_image,
                "wrist_image": wrist_image,
                "state": state,
            },
            "task_descriptions": list(self.task_descriptions),
        }

    def get_all_state_ids(self):
        return np.arange(self.total_num_group_envs)

    def reset_envs_to_state_ids(self, state_ids_list, task_ids_list):
        self._state_ids = np.asarray(state_ids_list, dtype=np.int64)
        self._task_ids = np.asarray(task_ids_list, dtype=np.int64)
        self._episode_steps[:] = 0
        self._episode_returns[:] = 0.0
        self.task_descriptions = [self._task_description(task_id) for task_id in self._task_ids]
        reply = send_obj(
            type="reset",
            content={
                "state_ids": self._state_ids.tolist(),
                "task_ids": self._task_ids.tolist(),
            },
            rank=self.rank,
            stage_id=self.stage_id,
            timeout_s=10.0,
        )
        if not isinstance(reply, dict) or reply.get("status") != "ok":
            raise RuntimeError(f"Invalid runtime reply: {reply}")
        return self._build_obs(), {}
    
    def step(self, action):
        reply = send_obj(
            type="step",
            content={
                "actions": action.tolist(),
            },
            rank=self.rank,
            stage_id=self.stage_id,
            timeout_s=10.0,
        )

        if not isinstance(reply, dict) or reply.get("status") != "ok":
            raise RuntimeError(f"Invalid runtime reply: {reply}")
        

    def chunk_step(self, chunk_actions, chunk_values=None):
        chunk_size = chunk_actions.shape[1]
        for step_idx in range(chunk_size):
            action = chunk_actions[:, step_idx, :]
            self.step(action)



        chunk_size = int(chunk_actions.shape[1])
        rewards = torch.zeros((self.num_envs, chunk_size), dtype=torch.float32)
        terminations = torch.zeros((self.num_envs, chunk_size), dtype=torch.bool)
        truncations = torch.zeros((self.num_envs, chunk_size), dtype=torch.bool)

        for step_idx in range(chunk_size):
            self._episode_steps += 1
            reached_limit = self._episode_steps >= self.max_episode_steps
            truncations[:, step_idx] = torch.as_tensor(reached_limit, dtype=torch.bool)

        self._episode_returns += rewards.sum(dim=1).numpy()
        infos = {}
        if truncations[:, -1].any():
            done_mask = truncations[:, -1]
            infos["final_info"] = {
                "episode": {
                    "return": torch.as_tensor(self._episode_returns, dtype=torch.float32),
                    "episode_len": torch.as_tensor(self._episode_steps, dtype=torch.int32),
                    "success_once": torch.zeros(self.num_envs, dtype=torch.bool),
                }
            }
            self._episode_steps[done_mask.numpy()] = 0
            self._episode_returns[done_mask.numpy()] = 0.0

        return self._build_obs(), rewards, terminations, truncations, infos

    def flush_video(self, video_sub_dir: Optional[str] = None):
        logger.debug("LeRobotEnv.flush_video is a no-op scaffold")

    def load_state(self, state_buffer: bytes):
        logger.debug("LeRobotEnv.load_state is a no-op scaffold")

    def close(self):
        return None
