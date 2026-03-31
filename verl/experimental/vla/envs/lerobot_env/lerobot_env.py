# Copyright 2024 Bytedance Ltd. and/or its affiliates

import logging
import os
import shlex
import shutil
import socket
import subprocess
import sys
from typing import Optional

import gymnasium as gym
import numpy as np
import torch

from verl.experimental.vla.envs.action_utils import put_info_on_image, save_rollout_video, to_tensor

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
        self._runtime_session_name = f"{_LEROBOT_RUNTIME_SESSION_PREFIX}_{socket.gethostname()}_rank{self.rank}_stage{self.stage_id}"

        self._episode_steps = np.zeros(self.num_envs, dtype=np.int32)
        self._episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self._state_ids = np.zeros(self.num_envs, dtype=np.int64)
        self._task_ids = np.zeros(self.num_envs, dtype=np.int64)
        self.task_descriptions = [self._task_description(0) for _ in range(self.num_envs)]
        self.total_num_group_envs = max(self.num_envs * self.world_size * 32, 1024)

        lerobot_config_path = getattr(cfg, "lerobot_config_path", None)
        if not lerobot_config_path:
            raise ValueError("`cfg.lerobot_config_path` is required for LeRobotEnv.")
        if not os.path.exists(lerobot_config_path):
            raise FileNotFoundError(f"LeRobot config not found: {lerobot_config_path}")
        self._ensure_tmux_lerobot_runtime(lerobot_config_path)

    def _ensure_tmux_lerobot_runtime(self, lerobot_config_path: str) -> None:
        if shutil.which("tmux") is None:
            raise RuntimeError("`tmux` is required for LeRobotEnv runtime process.")

        runtime_cmd = (
            f"exec {sys.executable} -u -m verl.experimental.vla.envs.lerobot_env.lerobot_runtime "
            f"--config_path {shlex.quote(str(lerobot_config_path))} "
            f"--rank {self.rank} --stage_id {self.stage_id} "
            f"--owner_pid {os.getpid()}"
        )

        has_session = self._tmux_has_session()
        if has_session.returncode == 0:
            self._enable_tmux_log_forwarding()
            logger.info(
                "LeRobot runtime tmux session already exists: %s (forwarding to pid=%s stdout)",
                self._runtime_session_name,
                os.getpid(),
            )
            return

        clear_ipc(rank=self.rank, stage_id=self.stage_id)

        subprocess.run(
            ["tmux", "new-session", "-d", "-s", self._runtime_session_name],
            check=True,
        )
        self._enable_tmux_log_forwarding()
        subprocess.run(
            ["tmux", "send-keys", "-t", self._runtime_session_name, runtime_cmd, "C-m"],
            check=True,
        )
        logger.info(
            "Started LeRobot runtime in tmux session: %s (forwarding to pid=%s stdout)",
            self._runtime_session_name,
            os.getpid(),
        )

    def _tmux_has_session(self) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["tmux", "has-session", "-t", self._runtime_session_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )

    def _assert_runtime_alive(self) -> None:
        has_session = self._tmux_has_session()
        if has_session.returncode != 0:
            raise RuntimeError(
                f"LeRobot runtime tmux session exited unexpectedly: {self._runtime_session_name}"
            )

    def _enable_tmux_log_forwarding(self) -> None:
        stdout_path = f"/proc/{os.getpid()}/fd/1"
        prefix = f"[LeRobot runtime rank={self.rank} stage={self.stage_id}] "
        pipe_cmd = f"awk -v p={shlex.quote(prefix)} '{{print p $0; fflush();}}' >> {shlex.quote(stdout_path)}"
        subprocess.run(["tmux", "pipe-pane", "-t", self._runtime_session_name], check=True)
        subprocess.run(
            ["tmux", "pipe-pane", "-t", self._runtime_session_name, pipe_cmd],
            check=True,
        )

    def _task_description(self, task_id: int) -> str:
        return f"lerobot_task_{task_id}"

    def _wrap_runtime_obs(self, runtime_obs: dict) -> dict:
        return {
            "images_and_states": to_tensor(
                {
                    "full_image": runtime_obs["observation.images.top"].permute(0, 2, 3, 1),
                    "wrist_image": runtime_obs["observation.images.wrist"].permute(0, 2, 3, 1),
                    "state": runtime_obs["observation.state"],
                }
            ),
            "task_descriptions": list(self.task_descriptions),
        }

    def add_new_frames(self, obs, plot_infos):
        info_item = {k: v if np.size(v) == 1 else v[0] for k, v in plot_infos.items()}
        top_image = (obs["images_and_states"]["full_image"][0].detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        wrist_image = (obs["images_and_states"]["wrist_image"][0].detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        top_image = put_info_on_image(top_image, info_item)
        wrist_image = put_info_on_image(wrist_image, info_item)
        self.render_images.append(np.concatenate([top_image, wrist_image], axis=1))

    def get_all_state_ids(self):
        return np.arange(self.total_num_group_envs)

    def reset_envs_to_state_ids(self, state_ids_list, task_ids_list):
        self._state_ids = np.asarray(state_ids_list, dtype=np.int64)
        self._task_ids = np.asarray(task_ids_list, dtype=np.int64)
        self._episode_steps[:] = 0
        self._episode_returns[:] = 0.0
        self.task_descriptions = [self._task_description(task_id) for task_id in self._task_ids]
        self._assert_runtime_alive()
        obs = send_obj(
            type="reset",
            content={
                "state_ids": self._state_ids.tolist(),
                "task_ids": self._task_ids.tolist(),
            },
            rank=self.rank,
            stage_id=self.stage_id
        )
        obs = self._wrap_runtime_obs(obs)
        return obs, {}
    
    def step(self, action, critic_values=None):
        self._assert_runtime_alive()
        reply = send_obj(
            type="step",
            content={
                "actions": action.tolist(),
            },
            rank=self.rank,
            stage_id=self.stage_id,
            timeout_s=10.0,
        )

        if not isinstance(reply, dict):
            raise RuntimeError(f"Invalid runtime reply: {reply}")
        obs = self._wrap_runtime_obs(reply["obs"])
        reward = to_tensor(np.asarray([reply["reward"]], dtype=np.float32))
        is_intervention = reply["extra_info"]["is_intervention"]
        executed_action = reply["extra_info"]["executed_action"]

        self._episode_steps += 1
        self._episode_returns += reward.numpy()

        terminations = to_tensor(np.asarray([reply["done"]], dtype=bool))
        raw_truncations = np.asarray([reply["truncated"]], dtype=bool)
        raw_truncations = np.logical_or(raw_truncations, self._episode_steps >= self.max_episode_steps)
        truncations = to_tensor(raw_truncations)

        if self.video_cfg.save_video:
            plot_infos = {
                "rewards": reward.numpy(),
                "terminations": terminations.numpy(),
                "truncations": truncations.numpy(),
                "is_intervention": is_intervention,
                "task": self.task_descriptions,
            }
            if critic_values is not None:
                plot_infos["critic_value"] = np.asarray(critic_values, dtype=np.float32)
            self.add_new_frames(obs, plot_infos)

        infos = {}
        infos["is_intervention"] = is_intervention
        infos["executed_action"] = executed_action
        done_mask = torch.logical_or(terminations, truncations)
        if done_mask.any():
            infos["final_info"] = {
                "episode": {
                    "return": torch.as_tensor(self._episode_returns, dtype=torch.float32),
                    "episode_len": torch.as_tensor(self._episode_steps, dtype=torch.int32),
                    "success_once": terminations.clone(),
                }
            }
            self._episode_steps[done_mask.numpy()] = 0
            self._episode_returns[done_mask.numpy()] = 0.0

        return obs, reward, terminations, truncations, infos

    def chunk_step(self, chunk_actions, chunk_values=None):
        chunk_size = chunk_actions.shape[1]
        chunk_rewards = []
        raw_chunk_terminations = []
        raw_chunk_truncations = []
        intervention_info = {"obs": [], "actions": [], "is_intervention": []}
        last_step_is_intervention = False

        step_idx = 0
        while step_idx < chunk_size or last_step_is_intervention:
            extracted_obs, step_reward, terminations, truncations, infos = self.step(
                chunk_actions[:, min(step_idx, chunk_size - 1), :],
                critic_values= None if last_step_is_intervention else chunk_values
            )

            chunk_rewards.append(step_reward)
            raw_chunk_terminations.append(terminations)
            raw_chunk_truncations.append(truncations)
            intervention_info["actions"].append(to_tensor(infos["executed_action"]).unsqueeze(0))
            intervention_info["is_intervention"].append(to_tensor(infos["is_intervention"]).unsqueeze(0))
            if (step_idx + 1) % chunk_size == 0:
                intervention_info["obs"].append(extracted_obs)

            last_step_is_intervention = infos["is_intervention"]
            step_idx += 1

        chunk_rewards = torch.stack(chunk_rewards, dim=1)
        chunk_terminations = torch.stack(raw_chunk_terminations, dim=1)
        chunk_truncations = torch.stack(raw_chunk_truncations, dim=1)

        infos = {}
        intervention_obs = {
            f"obs.{key}": torch.stack([step_obs["images_and_states"][key] for step_obs in intervention_info["obs"]], dim=1)
            for key in intervention_info["obs"][0]["images_and_states"]
        }

        is_intervention_tensor = torch.stack(intervention_info["is_intervention"], dim=1)
        if is_intervention_tensor.any():
            infos["intervention_info"] = {
                **intervention_obs,
                "obs.task_descriptions": np.array(
                    [step_obs["task_descriptions"] for step_obs in intervention_info["obs"]], dtype=object
                ).transpose(1, 0),
                "actions": torch.stack(intervention_info["actions"], dim=1),
                "is_intervention": is_intervention_tensor
            }
        return extracted_obs, chunk_rewards, chunk_terminations, chunk_truncations, infos

    def flush_video(self, video_sub_dir: Optional[str] = None):
        output_dir = os.path.join(self.video_cfg.video_base_dir, f"rank_{self.rank}")
        if video_sub_dir is not None:
            output_dir = os.path.join(output_dir, f"{video_sub_dir}")
        save_rollout_video(
            self.render_images,
            output_dir=output_dir,
            video_name=f"{self.video_cnt}",
        )
        self.video_cnt += 1
        self.render_images = []

    def load_state(self, state_buffer: bytes):
        logger.debug("LeRobotEnv.load_state is a no-op scaffold")

    def close(self):
        if shutil.which("tmux") is not None and self._tmux_has_session().returncode == 0:
            subprocess.run(
                ["tmux", "kill-session", "-t", self._runtime_session_name],
                check=False,
            )
            logger.info("Stopped LeRobot runtime tmux session: %s", self._runtime_session_name)
        clear_ipc(rank=self.rank, stage_id=self.stage_id)
        return None
