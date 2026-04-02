# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""IsaacLab multi-task environment wrapper for verl rolling short-horizon rollouts.

Key design decisions vs the original IsaacEnv:
  - The env is created **once** and persists across training steps (no USD stage rebuild).
  - Relies on IsaacLab's built-in auto-reset: terminated/truncated sub-envs are
    reset internally by ManagerBasedRLEnv.step(); the returned obs is post-reset.
  - Rewards, terminated, and truncated tensors are passed through directly from
    IsaacLab without recalculation.
  - Supports MultiTask (GroupedManagerBasedRLEnv with per-group assets/rewards)
    as well as single-task ManagerBasedRLEnv configs.
"""

import logging
import os
from typing import Optional

import gymnasium as gym
import numpy as np
import torch

from verl.experimental.vla.envs.action_utils import (
    put_info_on_image,
    save_rollout_video,
    tile_images,
    to_tensor,
)

logger = logging.getLogger(__name__)


class IsaacMultiTaskEnv:
    """Persistent Isaac Lab environment wrapper for rolling short-horizon rollouts.

    Unlike IsaacEnv which rebuilds the USD stage on every reset, this wrapper
    creates the environment once during ``__init__`` and keeps it alive.
    Sub-envs that terminate are auto-reset by IsaacLab internally.
    """

    def __init__(self, cfg, rank: int, world_size: int):
        self.rank = rank
        self.cfg = cfg
        self.world_size = world_size
        self.seed = cfg.seed + rank
        self.num_envs = cfg.num_envs
        self.action_dim = cfg.get("action_dim", 7)
        self.device = cfg.get("device", "cuda:0")
        self.camera_names = cfg.init_params.camera_names

        self.video_cfg = cfg.video_cfg
        self.render_images: list[np.ndarray] = []
        self.video_cnt = 0

        os.environ["LIBERO_CONFIG_DIR"] = "/root/RobotLearningLab/benchmarks/datasets/libero/config"
        os.environ["LIBERO_ASSETS_DATA_DIR"] = "/root/RobotLearningLab/benchmarks/datasets/libero/USD"
        os.environ.setdefault(
            "LIBERO_ASSEMBLED_DATASET_DIR",
            "/root/RobotLearningLab/benchmarks/datasets/libero/assembled_hdf5",
        )

        group_size = cfg.get("group_size", 1)
        if world_size > 1:
            group_size = group_size // world_size
        os.environ["GROUP_SIZE"] = str(group_size)

        task_suite = cfg.get("task_suite_name", "")
        if task_suite:
            os.environ["LIBERO_TASK_SUITE"] = task_suite

        from isaaclab.app import AppLauncher

        launch_args = {"headless": True, "enable_cameras": True}
        app_launcher = AppLauncher(**launch_args)
        self.app = app_launcher.app

        import isaaclab_playground.tasks.manipulation.libero.config.franka  # noqa: F401

        self.task_name = cfg.get("task_name", "Isaac-Libero-Franka-Joint-Position-All-Tasks-v0")
        self.env = None
        self.env_cfg = None

        self._is_ik_abs = "Abs-IK" in self.task_name or "IK-Abs" in self.task_name

        self._init_metrics()

        logger.info(
            f"IsaacMultiTaskEnv app launched: task={self.task_name}, "
            f"num_envs={self.num_envs}, device={self.device}"
        )

    def init_env(self):
        """Create the IsaacLab environment (heavy — builds USD scenes).

        Called separately from __init__ so that it runs through the command
        queue (no 180-second timeout) rather than during subprocess startup.
        """
        from isaaclab_tasks.utils import parse_env_cfg

        env_cfg = parse_env_cfg(self.task_name)
        env_cfg.sim.device = self.device
        env_cfg.sim.physx.enable_ccd = True
        env_cfg.observations.policy.concatenate_terms = False

        self.env = gym.make(self.task_name, cfg=env_cfg).unwrapped
        self.env_cfg = env_cfg
        self.num_envs = env_cfg.scene.num_envs

        self._build_task_descriptions()
        self._init_metrics()

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

        if self.video_cfg.save_video:
            video_dir = os.path.join(self.video_cfg.video_base_dir, f"rank_{self.rank}")
            os.makedirs(video_dir, exist_ok=True)

        logger.info(
            f"IsaacMultiTaskEnv env created: task={self.task_name}, "
            f"num_envs={self.num_envs}, device={self.device}"
        )

    # ------------------------------------------------------------------
    # Task descriptions
    # ------------------------------------------------------------------

    def _build_task_descriptions(self):
        """Build per-env language instructions and task assignment keys from the env config."""
        if hasattr(self.env_cfg, "libero_config"):
            lc = self.env_cfg.libero_config
            if hasattr(lc, "get_task_info_for_env"):
                self.task_descriptions = [
                    lc.get_task_info_for_env(i).get("language_instruction", "")
                    for i in range(self.num_envs)
                ]
            elif hasattr(lc, "task_info"):
                desc = lc.task_info.get("language_instruction", "")
                self.task_descriptions = [desc] * self.num_envs
            else:
                self.task_descriptions = [""] * self.num_envs

            if hasattr(lc, "env_task_assignments"):
                self.task_assignment_keys = list(lc.env_task_assignments[:self.num_envs])
            else:
                self.task_assignment_keys = [str(i) for i in range(self.num_envs)]
        else:
            self.task_descriptions = [""] * self.num_envs
            self.task_assignment_keys = [str(i) for i in range(self.num_envs)]

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _init_metrics(self):
        self.cumulative_rewards = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.success_flags = np.zeros(self.num_envs, dtype=bool)

    def _update_metrics(self, rewards: np.ndarray, terminated: np.ndarray):
        self.cumulative_rewards += rewards
        self.episode_lengths += 1
        self.success_flags |= terminated.astype(bool)

        reset_mask = terminated.astype(bool)
        if reset_mask.any():
            self.cumulative_rewards[reset_mask] = 0.0
            self.episode_lengths[reset_mask] = 0
            self.success_flags[reset_mask] = False

    # ------------------------------------------------------------------
    # Coordinate conversions
    # ------------------------------------------------------------------

    @staticmethod
    def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
        """(w, x, y, z) quaternion -> axis-angle.  Batched: (..., 4) -> (..., 3)."""
        w = np.clip(quat[..., 0:1], -1.0, 1.0)
        xyz = quat[..., 1:4]
        angle = 2.0 * np.arccos(np.abs(w))
        den = np.sqrt(1.0 - w * w)
        small = den < 1e-8
        return np.where(small, np.zeros_like(xyz), xyz / den * angle * np.sign(w))

    @staticmethod
    def _axisangle2quat(axisangle: torch.Tensor) -> torch.Tensor:
        """Axis-angle (..., 3) -> (w, x, y, z) quaternion (..., 4)."""
        angle = torch.norm(axisangle, dim=-1, keepdim=True).clamp(min=1e-8)
        axis = axisangle / angle
        half = angle * 0.5
        return torch.cat([torch.cos(half), axis * torch.sin(half)], dim=-1)

    def _convert_actions_for_ik_abs(self, actions: torch.Tensor) -> torch.Tensor:
        """pos(3) + axis-angle(3) + gripper(1) -> pos(3) + quat(4) + gripper(1)."""
        pos = actions[..., :3]
        quat = self._axisangle2quat(actions[..., 3:6])
        gripper = actions[..., 6:7]
        return torch.cat([pos, quat, gripper], dim=-1)

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def _wrap_obs(self, raw_obs: dict) -> dict:
        images_and_states = self._extract_image_and_state(raw_obs)
        return {
            "images_and_states": to_tensor(images_and_states),
            "task_descriptions": list(self.task_descriptions),
            "task_assignment_keys": list(self.task_assignment_keys),
        }

    def _extract_image_and_state(self, obs: dict) -> dict:
        images: dict[str, np.ndarray] = {}
        available_cameras = list(self.env.scene.keys())

        for cam_name in self.camera_names:
            for key in available_cameras:
                if key.startswith(cam_name):
                    rgb = self.env.scene[key].data.output["rgb"]
                    images[cam_name] = rgb.cpu().numpy()
                    break

        assert self.camera_names[0] in images, (
            f"Primary camera {self.camera_names[0]} not found in scene"
        )

        eef_pose = obs["policy"]["eef_pose"].cpu().numpy()
        eef_pos = eef_pose[..., :3]
        eef_axisangle = self._quat2axisangle(eef_pose[..., 3:7])
        gripper_pos = obs["policy"]["gripper_pos"].cpu().numpy()
        gripper_state = gripper_pos[..., 0:1]

        output: dict = {
            "full_image": images[self.camera_names[0]],
            "state": np.concatenate([eef_pos, eef_axisangle, gripper_state], axis=-1),
        }

        wrist_key = None
        for name in ("eye_in_hand_cam", "robot0_eye_in_hand"):
            if name in images:
                wrist_key = name
                break
        if wrist_key is None:
            for cam_key in available_cameras:
                if "eye_in_hand" in cam_key or "wrist" in cam_key:
                    rgb = self.env.scene[cam_key].data.output["rgb"]
                    output["wrist_image"] = rgb.cpu().numpy()
                    wrist_key = cam_key
                    break
            assert wrist_key is not None, "Wrist camera not found in scene"
        else:
            output["wrist_image"] = images[wrist_key]

        return output

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self, env_ids=None):
        """Reset (some or all) envs.  Normally only called once at startup."""
        if env_ids is not None:
            env_ids_tensor = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        else:
            env_ids_tensor = None
        raw_obs, infos = self.env.reset(env_ids=env_ids_tensor)
        self._init_metrics()
        return self._wrap_obs(raw_obs), infos

    def step(self, actions: torch.Tensor):
        """Single env step.  Returns IsaacLab signals directly."""
        if self._is_ik_abs:
            actions = self._convert_actions_for_ik_abs(actions.to(self.device))
        else:
            actions = actions.to(self.device)

        raw_obs, reward_buf, terminated, truncated, extras = self.env.step(actions)

        # get_term("success") reads _term_dones which is NOT cleared on
        # auto-reset, so it stays True for all subsequent steps until a
        # different termination term fires.  Mask with ``terminated`` to
        # only report success at the actual episode boundary.
        success = self.env.termination_manager.get_term("success") & terminated

        reward_np = reward_buf.cpu().numpy()
        terminated_np = terminated.cpu().numpy()
        success_np = success.cpu().numpy()
        self._update_metrics(reward_np, success_np)

        obs = self._wrap_obs(raw_obs)
        return (
            obs,
            to_tensor(reward_np),
            to_tensor(terminated_np),
            to_tensor(truncated.cpu().numpy()),
            to_tensor(success_np),
            extras,
        )

    def chunk_step(self, chunk_actions: np.ndarray | torch.Tensor, chunk_values=None):
        """Execute an action chunk (num_envs, chunk_size, action_dim).

        When an env terminates mid-chunk, remaining steps use a *hold action*
        (current EEF pose for IK-Abs, zeros otherwise) so the robot stays still
        instead of executing stale actions from the previous episode.  A
        ``valid_mask`` tensor marks post-reset steps as invalid so the trainer
        can exclude them from loss computation.

        Returns:
            (obs, rewards, terminated, truncated, infos)
            infos contains 'successes' and 'valid_mask' tensors.
        """
        if isinstance(chunk_actions, np.ndarray):
            chunk_actions = torch.from_numpy(chunk_actions)
        chunk_size = chunk_actions.shape[1]

        chunk_rewards = []
        chunk_terminated = []
        chunk_truncated = []
        chunk_success = []

        reset_mask = torch.zeros(self.num_envs, dtype=torch.bool)
        hold_actions = torch.zeros(self.num_envs, chunk_actions.shape[-1])
        valid_mask = torch.ones(self.num_envs, chunk_size, dtype=torch.bool)

        for i in range(chunk_size):
            step_values = None
            if chunk_values is not None:
                if chunk_values.ndim == 1:
                    step_values = chunk_values
                elif chunk_values.ndim == 2:
                    step_values = chunk_values[:, i]

            step_actions = chunk_actions[:, i].clone()
            if reset_mask.any():
                step_actions[reset_mask] = hold_actions[reset_mask]
                valid_mask[reset_mask, i] = False

            obs, reward, terminated, truncated, success, extras = self.step(step_actions)

            done = (terminated.bool() | truncated.bool()).cpu()
            newly_done = done & ~reset_mask
            if newly_done.any():
                reset_mask |= newly_done
                if self._is_ik_abs:
                    eef_state = obs["images_and_states"]["state"]
                    if isinstance(eef_state, torch.Tensor):
                        hold_actions[newly_done] = eef_state[newly_done].cpu()
                    else:
                        hold_actions[newly_done] = torch.from_numpy(eef_state[newly_done])

            if self.video_cfg.save_video:
                plot_infos: dict = {
                    "rewards": reward.numpy() if isinstance(reward, torch.Tensor) else reward,
                    "terminations": terminated,
                }
                if step_values is not None:
                    plot_infos["critic_value"] = np.asarray(step_values, dtype=np.float32)
                self._add_video_frames(obs, plot_infos)

            chunk_rewards.append(reward)
            chunk_terminated.append(terminated)
            chunk_truncated.append(truncated)
            chunk_success.append(success)

        extras["successes"] = torch.stack(chunk_success, dim=1)
        extras["valid_mask"] = valid_mask
        return (
            obs,
            torch.stack(chunk_rewards, dim=1),
            torch.stack(chunk_terminated, dim=1),
            torch.stack(chunk_truncated, dim=1),
            extras,
        )

    # ------------------------------------------------------------------
    # Video recording
    # ------------------------------------------------------------------

    def _add_video_frames(self, obs: dict, plot_infos: dict):
        images = []
        for env_id, img in enumerate(obs["images_and_states"]["full_image"]):
            info_item = {
                k: (v if np.size(v) == 1 else v[env_id])
                for k, v in plot_infos.items()
            }
            img_np = img.cpu().numpy() if isinstance(img, torch.Tensor) else img
            images.append(put_info_on_image(img_np, info_item))
        full_image = tile_images(images, nrows=max(1, int(np.sqrt(self.num_envs))))
        self.render_images.append(full_image)

    def flush_video(self, video_sub_dir: Optional[str] = None):
        if not self.render_images:
            return
        output_dir = os.path.join(self.video_cfg.video_base_dir, f"rank_{self.rank}")
        if video_sub_dir is not None:
            output_dir = os.path.join(output_dir, video_sub_dir)
        save_rollout_video(self.render_images, output_dir=output_dir, video_name=str(self.video_cnt))
        self.video_cnt += 1
        self.render_images = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self):
        if self.env is not None:
            self.env.close()
            self.app.close()

    def load_state(self, state_buffer: bytes):
        pass

    def get_state(self):
        return None
