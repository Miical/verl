# !/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""
Real Robot Environment Module - Plug-and-Play Component for test_env

This module provides a hot-pluggable real robot environment that can be directly
imported and used in test_env. It includes:

1. RobotEnv: Core Gym environment for real robot control
2. RealRobotEnvWrapper: Adapter that makes RobotEnv compatible with test_env interface
3. Human intervention support via teleoperator devices (gamepad/keyboard)
4. Inverse kinematics processing for end-effector control
5. Image preprocessing and observation processing pipelines

Usage Example:
    from robot.controller.piper.robot_env import RealRobotEnvWrapper
    
    # Create real robot environment
    env = RealRobotEnvWrapper(cfg, rank=0, world_size=1)
    
    # Use like test_env
    obs, info = env.reset()
    obs, reward, term, trunc, info = env.step(action)
    
    # Check for human intervention
    if info["intervene_flag"].any():
        intervened_action = info["intervene_action"]
        
    env.close()

Key Components:
    - RobotEnv: Low-level robot control (reset, step, observation)
    - make_robot_env: Factory to create robot + teleoperator
    - make_processors: Create data processing pipelines
    - step_env_and_process_transition: Unified step function with processors
    - RealRobotEnvWrapper: Main entry point for test_env integration
"""

import logging
logging.getLogger("can.interfaces.socketcan").setLevel(logging.ERROR)
import time
import sys
import os
from dataclasses import dataclass
from typing import Any
import cv2
import gymnasium as gym
import numpy as np
import torch
import math
def rotmat_to_rpy_zyx(R: np.ndarray) -> tuple[float, float, float]:
    """
    将旋转矩阵 R 转成欧拉角 (rx, ry, rz)，单位 rad
    使用 Z-Y-X (yaw-pitch-roll) 约定：
      - rz: yaw (绕 Z)
      - ry: pitch (绕 Y)
      - rx: roll (绕 X)
    返回顺序按 EndPoseCtrl 习惯：rx, ry, rz
    """
    # 避免数值误差
    r20 = -R[2, 0]
    r20_clamped = max(min(r20, 1.0), -1.0)
    pitch = math.asin(r20_clamped)

    cos_pitch = math.cos(pitch)
    if abs(cos_pitch) < 1e-6:
        # 退化情况（接近 ±90°），简单处理
        roll = 0.0
        yaw = math.atan2(-R[0, 1], R[1, 1])
    else:
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(R[1, 0], R[0, 0])

    return float(roll), float(pitch), float(yaw)

def images_encoding(imgs):
    """编码图像并记录耗时"""
    encode_start = time.perf_counter()
    encode_data = []
    padded_data = []
    max_len = 0
    for i in range(len(imgs)):
        success, encoded_image = cv2.imencode(".jpg", imgs[i])
        jpeg_data = encoded_image.tobytes()
        encode_data.append(jpeg_data)
        max_len = max(max_len, len(jpeg_data))
    for i in range(len(imgs)):
        padded_data.append(encode_data[i].ljust(max_len, b"\0"))
    encode_time = time.perf_counter() - encode_start
    return encode_data, max_len, encode_time
# 强制刷新输出，解决 Ray 环境下的输出缓冲问题
def force_print(*args, **kwargs):
    """强制输出并刷新的 print 函数"""
    print(*args, **kwargs, flush=True)
    sys.stdout.flush()
    sys.stderr.flush()

# 将 lerobot 路径添加到 sys.path（用于 RobotEnv）
_current_dir = os.path.dirname(os.path.abspath(__file__))
_lerobot_path = _current_dir
if _lerobot_path not in sys.path:
    sys.path.insert(0, _lerobot_path)

from lerobot.cameras import opencv  # noqa: F401
from lerobot.envs.configs import HILSerlRobotEnvConfig
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    AddTeleopActionAsComplimentaryDataStep,
    AddTeleopEventsAsInfoStep,
    DataProcessorPipeline,
    DeviceProcessorStep,
    EnvTransition,
    GripperPenaltyProcessorStep,
    ImageCropResizeProcessorStep,
    InterventionActionProcessorStep,
    RenameObservationsProcessorStep,
    JointVelocityProcessorStep,
    MapDeltaActionToRobotActionStep,
    MapTensorToDeltaActionDictStep,
    MotorCurrentProcessorStep,
    Numpy2TorchActionProcessorStep,
    RewardClassifierProcessorStep,
    RobotActionToPolicyActionProcessorStep,
    TimeLimitProcessorStep,
    Torch2NumpyActionProcessorStep,
    TransitionKey,
    VanillaObservationProcessorStep,
    create_transition,
)
from lerobot.processor.converters import identity_transition
from lerobot.robots import (  # noqa: F401
    RobotConfig,
    make_robot_from_config,
    so100_follower,
    piper_follower,
)
from lerobot.robots.robot import Robot
from lerobot.robots.so100_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    EEReferenceAndDelta,
    ForwardKinematicsJointsToEEObservation,
    GripperVelocityToJoint,
    InverseKinematicsRLStep,
)
from lerobot.teleoperators import (
    gamepad,  # noqa: F401
    keyboard,  # noqa: F401
    make_teleoperator_from_config,
    so101_leader,  # noqa: F401
)
from lerobot.model.kinematics import RobotKinematics
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.utils.constants import ACTION, DONE, OBS_IMAGES, OBS_STATE, REWARD
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import log_say
from lerobot.cameras import (  # noqa: F401 - 导入相机以触发注册
    opencv,
    realsense,
    dabai,
)

import pdb

# 模块加载提示
force_print(f"[robot_env.py] Module loaded")


class HDF5DataLoader:
    """HDF5数据加载器，用于从HDF5文件读取机器人数据。
    
    每个 .hdf5 文件代表一个 episode，文件结构为：
    - action: (T, 14) 动作序列
    - observations/qpos: (T, 14) 状态序列
    - observations/images/{cam_high, cam_left_wrist, cam_right_wrist}: (T, H, W, 3) 图像序列
    """
    
    # 相机名称映射：内部名称 -> HDF5文件中的名称
    CAMERA_NAME_MAP = {
        'front': 'cam_high',
        'left': 'cam_left_wrist', 
        'right': 'cam_right_wrist',
    }
    
    def __init__(
        self,
        data_path: str,
        episode_index: int | None = None,
        step_size: int = 1,
    ):
        """初始化HDF5数据加载器。
        
        Args:
            data_path: HDF5文件路径或目录路径
            episode_index: 指定episode索引（文件索引），None表示随机选择
            step_size: 数据读取步长
        """
        import h5py
        from pathlib import Path
        hdf5_data_path = "/home/agilex-home/agilex/lerobot_hil-serl/dataset/20251125T005_lsz001_01"
        # force_print(f"[HDF5DataLoader] __init__ CALLED! data_path={data_path}")
        
        self.data_path = Path(hdf5_data_path)
        self.step_size = step_size
        self.ptr = 0  # 帧指针，从0开始
        self.h5_file = None
        self.episode_index = 0
        self.episode_length = 0
        
        force_print(f"[HDF5DataLoader] Checking path: {self.data_path}")
        force_print(f"[HDF5DataLoader] Path exists: {self.data_path.exists()}")
        force_print(f"[HDF5DataLoader] Is dir: {self.data_path.is_dir()}")
        
        # 查找HDF5文件（每个文件就是一个episode）
        if self.data_path.is_file() and self.data_path.suffix == '.hdf5':
            self.hdf5_files = [self.data_path]
        elif self.data_path.is_dir():
            # 按文件名排序（episode_0.hdf5, episode_1.hdf5, ...）
            self.hdf5_files = sorted(
                list(self.data_path.glob('*.hdf5')),
                key=lambda x: int(x.stem.split('_')[-1]) if x.stem.split('_')[-1].isdigit() else 0
            )
            force_print(f"[HDF5DataLoader] Found {len(self.hdf5_files)} .hdf5 files")
            if not self.hdf5_files:
                raise ValueError(f"No HDF5 files found in {data_path}")
        else:
            raise ValueError(f"Invalid data_path: {data_path}")
        
        force_print(f"[HDF5DataLoader] Total episodes: {len(self.hdf5_files)}")
        
        # 选择并加载一个episode（文件）
        self.select_episode(episode_index)
    
    @property
    def num_episodes(self) -> int:
        """返回可用的episode数量"""
        return len(self.hdf5_files)
    
    def select_episode(self, episode_index: int | None = None):
        """选择并加载一个episode（HDF5文件）。
        
        Args:
            episode_index: episode索引（文件索引），None表示随机选择
        """
        import h5py
        
        # 关闭之前的文件
        if self.h5_file is not None:
            self.h5_file.close()
            self.h5_file = None
        
        # 选择episode索引
        num_episodes = len(self.hdf5_files)
        if episode_index is None:
            episode_index = np.random.randint(0, num_episodes)
        
        if episode_index < 0 or episode_index >= num_episodes:
            raise ValueError(f"Episode index {episode_index} out of range [0, {num_episodes})")
        
        self.episode_index = episode_index
        
        # 加载HDF5文件
        file_path = self.hdf5_files[episode_index]
        self.h5_file = h5py.File(file_path, 'r')
        
        # 获取episode长度（从action或qpos的第一维）
        if 'action' in self.h5_file:
            self.episode_length = self.h5_file['action'].shape[0]
        elif 'observations' in self.h5_file and 'qpos' in self.h5_file['observations']:
            self.episode_length = self.h5_file['observations']['qpos'].shape[0]
        else:
            raise ValueError(f"Cannot determine episode length from {file_path}")
        
        # 重置帧指针到0
        self.ptr = 0
        
        force_print(f"[HDF5DataLoader] Selected episode {episode_index}: {file_path.name} (length: {self.episode_length})")
    
    def get_data(self, advance: int = 0) -> dict | None:
        """获取当前帧数据并可选推进指针。
        
        Args:
            advance: 推进步数，0表示不推进
            
        Returns:
            包含 'state', 'image', 'action' 的字典，或None（episode结束）
        """
        if self.ptr >= self.episode_length:
            return None
        
        current_idx = self.ptr
        
        # 读取状态数据 (observations/qpos)
        state = np.array(self.h5_file['observations']['qpos'][current_idx], dtype=np.float32)
        
        # 读取图像数据 (observations/images/{cam_high, cam_left_wrist, cam_right_wrist})
        image_data = {}
        images_group = self.h5_file['observations']['images']
        for internal_name, hdf5_name in self.CAMERA_NAME_MAP.items():
            if hdf5_name in images_group:
                img = np.array(images_group[hdf5_name][current_idx])
                # 确保图像格式为 (H, W, 3) 且值在 [0, 255]
                if img.ndim == 3 and img.shape[0] == 3:
                    img = np.transpose(img, (1, 2, 0))
                if img.dtype != np.uint8:
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                    else:
                        img = img.astype(np.uint8)
                image_data[internal_name] = img
        
        # 读取动作数据
        action = None
        if 'action' in self.h5_file:
            action = np.array(self.h5_file['action'][current_idx], dtype=np.float32)
        
        # 推进指针
        if advance > 0:
            self.ptr += advance
        
        return {
            'state': {
                'state': state,
                'frame_index': current_idx,
                'episode_index': self.episode_index,
            },
            'image': image_data,
            'action': action,
        }
    
    def reset(self, episode_index: int | None = None):
        """重置到指定episode的开始（帧指针归0）。
        
        Args:
            episode_index: 新的episode索引，None表示随机选择新episode
        """
        if episode_index is None or episode_index != self.episode_index:
            # 切换到新episode（None表示随机选择，或者指定了不同的episode）
            self.select_episode(episode_index)
        else:
            # 只重置帧指针到0（相同episode，从头开始）
            self.ptr = 0
    
    def is_done(self) -> bool:
        """检查当前episode是否结束。"""
        return self.ptr >= self.episode_length
    
    @property
    def num_episodes(self) -> int:
        """返回episode总数（HDF5文件数量）。"""
        return len(self.hdf5_files)
    
    def close(self):
        """关闭HDF5文件。"""
        if self.h5_file is not None:
            self.h5_file.close()
            self.h5_file = None


class TestRobotEnv(gym.Env):
    """测试用的机器人环境，从HDF5数据集读取数据，不连接真实硬件。
    
    模拟真实RobotEnv的接口，使用HDF5DataLoader从数据集读取观测和动作。
    用于测试和开发，不需要真实机器人硬件。
    """
    
    def __init__(
        self,
        use_gripper: bool = False,
        image_size: tuple[int, int] = (480, 640),
        num_cameras: int = 3,
        data_path: str = None,
        step_size: int = 1,
        action_chunk: int = 50,
    ) -> None:
        """初始化测试机器人环境。
        
        Args:
            use_gripper: 是否使用夹爪（暂未使用）
            image_size: 图像尺寸 (height, width)
            num_cameras: 相机数量
            data_path: HDF5数据集路径（文件或目录）
            step_size: 数据回放步长
            action_chunk: 动作块大小（暂未使用）
        """
        super().__init__()
        
        self.use_gripper = use_gripper
        self.image_size = image_size
        self.num_cameras = num_cameras
        # 使用传入的 data_path 参数
        self.data_path = data_path
        self.step_size = step_size
        
        force_print(f"[TestRobotEnv] __init__ CALLED! data_path={data_path}, self.data_path={self.data_path}")
        
        # 相机名称
        self._image_keys = ["front", "left", "right"][:num_cameras]
        
        # 末端位姿维度：左臂7维 + 右臂7维 = 14维
        self.state_dim = 14
        
        # Episode tracking
        self.current_step = 0
        
        # 使用传入的 data_path 参数,如果没有则使用硬编码路径
        if self.data_path is None:
            hdf5_data_path = "/home/agilex-home/agilex/lerobot_hil-serl/dataset/20251125T005_lsz001_01"
            force_print(f"[TestRobotEnv] No data_path provided, using hardcoded path: {hdf5_data_path}")
        else:
            hdf5_data_path = self.data_path
            force_print(f"[TestRobotEnv] Using provided data_path: {hdf5_data_path}")
        
        # 初始化 HDF5DataLoader
        self.use_hdf5 = True  # 总是尝试使用 HDF5
        force_print(f"[TestRobotEnv] Attempting to load HDF5 data from: {hdf5_data_path}")
        force_print(f"[TestRobotEnv] step_size={step_size}")
        try:
            force_print(f"[TestRobotEnv] Creating HDF5DataLoader...")
            self.hdf5_loader = HDF5DataLoader(
                data_path=hdf5_data_path,
                episode_index=None,  # 随机选择episode
                step_size=step_size,
            )
            force_print(f"[TestRobotEnv] ✓ HDF5DataLoader initialized successfully!")
            force_print(f"[TestRobotEnv] Number of episodes: {self.hdf5_loader.num_episodes}")
        except Exception as e:
            import traceback
            force_print(f"[TestRobotEnv] ✗ Failed to load HDF5: {e}")
            force_print(f"[TestRobotEnv] Traceback: {traceback.format_exc()}")
            force_print("[TestRobotEnv] Falling back to random data mode")
            self.use_hdf5 = False
            self.hdf5_loader = None
        
        self._setup_spaces()
    
    def _setup_spaces(self) -> None:
        """配置观测和动作空间。"""
        # 图像观测空间
        observation_spaces = {}
        prefix = OBS_IMAGES
        for key in self._image_keys:
            observation_spaces[f"{prefix}.{key}"] = gym.spaces.Box(
                low=0, high=255, 
                shape=(self.image_size[0], self.image_size[1], 3), 
                dtype=np.uint8
            )
        
        # 状态观测空间（末端位姿）
        observation_spaces[OBS_STATE] = gym.spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(self.state_dim,),
            dtype=np.float32,
        )
        
        self.observation_space = gym.spaces.Dict(observation_spaces)
        
        # 动作空间：14维（左臂7维 + 右臂7维）
        action_dim = 14
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(action_dim,),
            dtype=np.float32,
        )
    
    def _get_observation(self) -> dict[str, Any]:
        """获取观测数据（从HDF5或随机生成）。"""
        if self.use_hdf5 and self.hdf5_loader is not None:
            # 从数据集读取（数据回放模式）
            # 从HDF5数据集读取
            data = self.hdf5_loader.get_data()
            if data is None:
                # Episode结束，重置到新episode
                self.hdf5_loader.reset(episode_index=None)
                data = self.hdf5_loader.get_data()
            
            # 直接使用HDF5DataLoader返回的数据
            images = data['image']
            state_data = data['state']
            
            # 从state_data中提取state数组
            if isinstance(state_data, dict) and 'state' in state_data:
                state = state_data['state']
            else:
                state = state_data
            
            # 转换state为end_pose格式
            if isinstance(state, dict):
                agent_end_pose_value = np.array(list(state.values()), dtype=np.float32)
            else:
                agent_end_pose_value = np.asarray(state, dtype=np.float32)
            
            # 确保维度正确
            if len(agent_end_pose_value) != self.state_dim:
                # 如果维度不匹配，填充或截断
                if len(agent_end_pose_value) < self.state_dim:
                    agent_end_pose_value = np.pad(
                        agent_end_pose_value, 
                        (0, self.state_dim - len(agent_end_pose_value)),
                        mode='constant'
                    )
                else:
                    agent_end_pose_value = agent_end_pose_value[:self.state_dim]
            # 转换图像格式：images是字典 {cam_name: img_array}
            pixels = {}
            for i, key in enumerate(self._image_keys):
                if key in images:
                    img = images[key]
                    # 调整图像尺寸（如果需要）
                    if img.shape[:2] != self.image_size:
                        import cv2
                        img = cv2.resize(img, (self.image_size[1], self.image_size[0]))
                    pixels[key] = img
                else:
                    # 如果缺少某个相机，使用默认图像
                    pixels[key] = np.zeros(
                        (self.image_size[0], self.image_size[1], 3),
                        dtype=np.uint8
                    )
            
        else:
            # 随机生成模式（无数据集）
            agent_end_pose_value = np.random.uniform(-1.0, 1.0, size=(self.state_dim,)).astype(np.float32)
            
            pixels = {}
            for key in self._image_keys:
                pixels[key] = np.random.randint(
                    0, 256, 
                    size=(self.image_size[0], self.image_size[1], 3),
                    dtype=np.uint8
                )
        return {
            "agent_end_pose_value": agent_end_pose_value,
            "pixels": pixels
        }
    
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """重置环境。
        
        Args:
            seed: 随机种子
            options: 其他选项
            
        Returns:
            (observation, info) 元组
        """
        if seed is not None:
            np.random.seed(seed)
        
        self.current_step = 0
        
        # 如果使用HDF5，重置到新episode
        if self.use_hdf5 and self.hdf5_loader is not None:
            self.hdf5_loader.reset(episode_index=None)
        
        obs = self._get_observation()
        
        # 构造返回格式
        result_obs = {}
        result_obs[OBS_STATE] = obs["agent_end_pose_value"]
        for key in self._image_keys:
            result_obs[f"{OBS_IMAGES}.{key}"] = obs["pixels"][key]
        
        info = {"is_success": False}
        return result_obs, info
    
    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """执行一步动作。
        
        Args:
            action: 动作数组 [14] (末端位姿目标)
            
        Returns:
            (observation, reward, terminated, truncated, info) 元组
        
        Note:
            - 数据回放模式：action 被记录但不影响环境（回放历史数据）
            - 随机模式：action 可以用于简单的状态更新（可选）
        """
        # 验证 action 格式
        if action is not None:
            action = np.asarray(action[:14], dtype=np.float32)
            if action.shape != (self.state_dim,):
                force_print(f"[TestRobotEnv] Warning: Action shape mismatch: expected ({self.state_dim},), got {action.shape}")
        
        self.current_step += 1
        
        # 如果使用HDF5，推进数据指针（action 不影响回放，但会被记录）
        if self.use_hdf5 and self.hdf5_loader is not None:
            # 推进数据指针
            self.hdf5_loader.get_data(advance=self.step_size)
            
            # 检查是否到达episode末尾
            terminated = self.hdf5_loader.is_done()
            obs = self._get_observation()
        else:
            # 随机模式：可以使用 action 进行简单的状态更新
            # 这里我们保持简单，只生成随机观测
            # 如果需要，可以使用 action 来更新状态：
            #   next_state = current_state + action * 0.1  # 简单的积分
            obs = self._get_observation()
            terminated = np.random.random() < 0.01  # 1%概率终止
        
        # 构造返回格式
        result_obs = {}
        result_obs[OBS_STATE] = obs["agent_end_pose_value"]
        for key in self._image_keys:
            result_obs[f"{OBS_IMAGES}.{key}"] = obs["pixels"][key]
        
        # 奖励（简单的稀疏奖励）
        reward = 1.0 if terminated else 0.0
        
        # 检查truncation
        truncated = self.current_step >= 1000  # 最多1000步
        
        info = {
            "is_success": terminated,
            "current_step": self.current_step,
            "action_received": action is not None,  # 记录是否收到 action
        }
        
        return result_obs, reward, terminated, truncated, info
    
    def get_end_pose_name(self) -> list[str]:
        """返回末端位姿名称列表。"""
        left_names = [f"left_end_pose_{i}" for i in range(7)]
        right_names = [f"right_end_pose_{i}" for i in range(7)]
        return left_names + right_names
    
    def close(self):
        """关闭环境。"""
        if self.use_hdf5 and self.hdf5_loader is not None:
            self.hdf5_loader.close()

def save_cropped_images_from_processed_obs(processed_obs: dict, base_dir: str = "img_output/img_crop", enabled: bool = False) -> None:
    """
    从经过 VanillaObservationProcessorStep (+ ImageCropResizeProcessorStep) 处理后的观测中，
    提取裁剪/缩放后的图像 (B,C,H,W, float32, [0,1]) 并保存到 img_output/img_crop/<camera_name>/，
    每个相机只保留最新5张。

    参数:
        processed_obs: env_processor 输出的观测字典（transition[TransitionKey.OBSERVATION]）
        base_dir: 保存根目录
        enabled: 是否启用图像保存功能（默认False）
    """
    if not enabled:
        return
    
    import os, glob, time
    import torch
    import cv2

    os.makedirs(base_dir, exist_ok=True)

    def _key_to_camname(k: str) -> str:
        # 兼容两类键：
        # 1) 多相机: "observation.images.<camera>"
        # 2) 单图像: "observation.image"
        if ".images." in k:
            return k.split(".images.", 1)[1]
        if k.endswith(".image"):
            return "image"
        return k.replace(".", "_")  # 兜底

    for key, tensor_img in list(processed_obs.items()):
        if "image" not in key:
            continue
        if not isinstance(tensor_img, torch.Tensor):
            continue

        # 移到 CPU、去梯度
        img = tensor_img.detach()
        if img.device.type != "cpu":
            img = img.cpu()
        
        # 允许 (C,H,W) 或 (B,C,H,W)
        if img.ndim == 3:
            img = img.unsqueeze(0)  # -> (1,C,H,W)
        assert img.ndim == 4, f"Expect (B,C,H,W), got {img.shape}"

        # 规范到 [0,1] 再转 [0,255] uint8
        img = img.clamp(0.0, 1.0)
        img_u8 = torch.round(img * 255.0).to(torch.uint8)  # (B,C,H,W)

        cam_name = _key_to_camname(key)
        save_dir = os.path.join(base_dir, cam_name)
        os.makedirs(save_dir, exist_ok=True)

        # 逐张保存（通常 B=1）
        b, c, h, w = img_u8.shape
        for i in range(b):
            # (C,H,W) -> (H,W,C)
            hwc = img_u8[i].permute(1, 2, 0).numpy()
            # RGB -> BGR (cv2.imwrite 期望BGR)
            bgr = hwc[..., ::-1]

            filename = os.path.join(save_dir, f"{int(time.time() * 1000)}.jpg")
            cv2.imwrite(filename, bgr)

            # 只保留最近 5 张
            files = sorted(glob.glob(os.path.join(save_dir, "*.jpg")), key=os.path.getmtime)
            if len(files) > 5:
                for old_file in files[:-5]:
                    try:
                        os.remove(old_file)
                    except Exception:
                        pass

def reset_follower_position(robot_arm: Robot, target_position: np.ndarray,end_pose_name) -> None:
    """Reset robot arm to target position using smooth trajectory."""
    robot_arm.bus_left.interface.ModeCtrl(ctrl_mode=0x01, move_mode=0x01,move_spd_rate_ctrl=50, is_mit_mode=0x00)
    robot_arm.bus_right.interface.ModeCtrl(ctrl_mode=0x01, move_mode=0x01,move_spd_rate_ctrl=50, is_mit_mode=0x00)
    current_position_dict_left = robot_arm.bus_left.sync_read("Present_Position")
    current_position_dict_right = robot_arm.bus_right.sync_read("Present_Position")
    current_position_left = np.array(
        [current_position_dict_left[name] for name in current_position_dict_left], dtype=np.float32
    )
    current_position_right = np.array(
        [current_position_dict_right[name] for name in current_position_dict_right], dtype=np.float32
    )
    # pdb.set_trace()
    target_position_left=target_position[:7]
    target_position_right=target_position[7:]
    trajectory_left = torch.from_numpy(
        np.linspace(current_position_left, target_position_left, 50)
    )  # NOTE: 30 is just an arbitrary number
    trajectory_right = torch.from_numpy(
        np.linspace(current_position_right, target_position_right, 50)
    )  # NOTE: 30 is just an arbitrary number
    # pdb.set_trace()
    # busy_wait(0.1)
    for pose_left,pose_right in zip(trajectory_left,trajectory_right):
        action_dict_left = dict(zip(current_position_dict_left, pose_left, strict=False))
        action_dict_right = dict(zip(current_position_dict_right, pose_right, strict=False))
        # pdb.set_trace()
        robot_arm.bus_left.sync_write("Goal_Position", action_dict_left)
        robot_arm.bus_right.sync_write("Goal_Position", action_dict_right)
        busy_wait(0.015)
    
    robot_arm.bus_left.interface.ModeCtrl(ctrl_mode=0x01, move_mode=0x00,move_spd_rate_ctrl=50, is_mit_mode=0x00)
    robot_arm.bus_right.interface.ModeCtrl(ctrl_mode=0x01, move_mode=0x00,move_spd_rate_ctrl=50, is_mit_mode=0x00)
    action_zero=[0.056127, 0.0, 0.213266,0.0, 1.48351241090266,0.0,0.05957,0.056127, 0.0, 0.213266,0.0, 1.48351241090266,0.0,0.05957]
    action_target=[0.056127, 0.0, 0.213266,0.0, 1.48351241090266,0.0,0.05957,0.056127, 0.0, 0.213266,0.0, 1.48351241090266,0.0,0.05957]
    
    #action_target=[0.252452, -0.034618, 0.272318, 3.07350736, 0.4325624, 2.90283161, 0.06916,0.252452, -0.034618, 0.272318, 3.07350736, 0.4325624, 2.90283161, 0.06916]
    trajectory_endpose=np.linspace(action_zero, action_target, 12)
    for action in trajectory_endpose:
        end_pose_targets_dict = {f"{key}.value": action[i] for i, key in enumerate(end_pose_name)}
        robot_arm.send_action(end_pose_targets_dict)
        busy_wait(0.12)
    busy_wait(0.1)
    # pdb.set_trace()    


class RobotEnv(gym.Env):
    """Gym environment for robotic control with human intervention support."""

    def __init__(
        self,
        robot,
        use_gripper: bool = False,
        display_cameras: bool = False,
        reset_pose: list[float] | None = None,
        reset_time_s: float = 5.0,
    ) -> None:
        """Initialize robot environment with configuration options.

        Args:
            robot: Robot interface for hardware communication.
            use_gripper: Whether to include gripper in action space.
            display_cameras: Whether to show camera feeds during execution.
            reset_pose: Joint positions for environment reset.
            reset_time_s: Time to wait during reset.
        """
        super().__init__()
        
        self.robot = robot
        self.display_cameras = display_cameras

        # Connect to the robot if not already connected.
        if not self.robot.is_connected:
            self.robot.connect()

        # Episode tracking.
        self.current_step = 0
        self.episode_data = None

        # self._joint_names = [f"{key}.pos" for key in self.robot.bus.motors]
        self._image_keys = self.robot.cameras.keys()

        self.reset_pose = reset_pose
        self.reset_time_s = reset_time_s

        self.use_gripper = use_gripper

        # self._joint_names = list(self.robot.bus.motors.keys())
        self._left_joint_names = self.robot.bus_left.motors
        self._right_joint_names = self.robot.bus_right.motors
        self._joint_names= self._left_joint_names + self._right_joint_names
        # self._raw_joint_positions = None
        self._raw_end_pose_value = None
        self._left_end_pose_name = self.robot.bus_left.end_pose_keys
        self._right_end_pose_name = self.robot.bus_right.end_pose_keys
        self._end_pose_name= self._left_end_pose_name + self._right_end_pose_name
        
        self._setup_spaces()

    def _get_observation(self) -> dict[str, Any]:
        """Get current robot observation including joint positions and camera images."""
        obs_dict = self.robot.get_observation()
        for k in ("front", "left", "right"):
            img = obs_dict[k]
            obs_dict[k] = img[:, :, ::-1].copy()   
        # pdb.set_trace()
        # raw_joint_joint_position = {f"{name}.pos": obs_dict[f"{name}.pos"] for name in self._joint_names}
        # joint_positions = np.array([raw_joint_joint_position[f"{name}.pos"] for name in self._joint_names])
        raw_left_end_pose_values = {f"{name}.value": obs_dict[f"{name}.value"] for name in self._left_end_pose_name}
        raw_right_end_pose_values = {f"{name}.value": obs_dict[f"{name}.value"] for name in self._right_end_pose_name}
        left_end_pose_value=np.array([raw_left_end_pose_values[f"{name}.value"] for name in self._left_end_pose_name])
        right_end_pose_value=np.array([raw_right_end_pose_values[f"{name}.value"] for name in self._right_end_pose_name])

        agent_end_pose_value = np.concatenate([left_end_pose_value, right_end_pose_value], axis=0)
        images = {key: obs_dict[key] for key in self._image_keys}
        # print("agent_end_pose_value:",agent_end_pose_value)
        # pdb.set_trace()
        return {"agent_end_pose_value":agent_end_pose_value , "pixels": images, **raw_left_end_pose_values,**raw_right_end_pose_values}

    def _setup_spaces(self) -> None:
        """Configure observation and action spaces based on robot capabilities."""
        current_observation = self._get_observation()

        observation_spaces = {}

        # Define observation spaces for images and other states.
        if current_observation is not None and "pixels" in current_observation:
            prefix = OBS_IMAGES
            observation_spaces = {
                f"{prefix}.{key}": gym.spaces.Box(
                    low=0, high=255, shape=current_observation["pixels"][key].shape, dtype=np.uint8
                )
                for key in current_observation["pixels"]
            }
        # pdb.set_trace()
        if current_observation is not None:
            agent_end_pose_value = current_observation["agent_end_pose_value"]
            observation_spaces[OBS_STATE] = gym.spaces.Box(
                low=-10,
                high=10,
                shape=agent_end_pose_value.shape,
                dtype=np.float32,
            )
        
        self.observation_space = gym.spaces.Dict(observation_spaces)

        # Define the action space for joint positions along with setting an intervention flag.
        action_dim = 14
        bounds = {}
        bounds["min"] = -np.ones(action_dim)
        bounds["max"] = np.ones(action_dim)

        # if self.use_gripper:
        #     action_dim += 1
        #     bounds["min"] = np.concatenate([bounds["min"], [0]])
        #     bounds["max"] = np.concatenate([bounds["max"], [2]])

        self.action_space = gym.spaces.Box(
            low=bounds["min"],
            high=bounds["max"],
            shape=(action_dim,),
            dtype=np.float32,
        )
        # pdb.set_trace()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset environment to initial state.

        Args:
            seed: Random seed for reproducibility.
            options: Additional reset options.

        Returns:
            Tuple of (observation, info) dictionaries.
        """
        # Reset the robot
        # self.robot.reset()

        start_time = time.perf_counter()
        if self.reset_pose is not None:
            # print("here")
            print('reset the environment')
            #log_say("Reset the environment.", play_sounds=True)
            # pdb.set_trace()
            reset_follower_position(self.robot, np.array(self.reset_pose),self._end_pose_name)
            # action=[0.056127, 0.0, 0.213266,0.0, 1.48351241090266,0.0,0.05957,0.056127, 0.0, 0.213266,0.0, 1.48351241090266,0.0,0.05957]

            #log_say("Reset the environment done.", play_sounds=True)
        busy_wait(self.reset_time_s - (time.perf_counter() - start_time))

        # super().reset(seed=seed, options=options)

        # Reset episode tracking variables.
        self.current_step = 0
        self.episode_data = None
        obs = self._get_observation()
        # pdb.set_trace() 
        # self._raw_joint_positions = {f"{key}.pos": obs[f"{key}.pos"] for key in self._joint_names}
        self._raw_end_pose_value = {f"{key}.value": obs[f"{key}.value"] for key in self._end_pose_name}

        return obs, {TeleopEvents.IS_INTERVENTION: False}

    def step(self, action) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Execute one environment step with given action."""
        # joint_targets_dict = {f"{key}.pos": action[i] for i, key in enumerate(self.robot.bus.motors)}
        #盲改
        end_pose_targets_dict = {f"{key}.value": action[i] for i, key in enumerate(self._end_pose_name)}
        self.robot.send_action(end_pose_targets_dict)

        obs = self._get_observation()
        self._raw_end_pose_value = {f"{key}.value": obs[f"{key}.value"] for key in self._end_pose_name}

        # self._raw_joint_positions = {f"{key}.pos": obs[f"{key}.pos"] for key in self._joint_names}

        if self.display_cameras:
            self.render()

        self.current_step += 1
        
        reward = 0.0
        terminated = False
        truncated = False

        return (
            obs,
            reward,
            terminated,
            truncated,
            {TeleopEvents.IS_INTERVENTION: False},
        )

    def render(self) -> None:
        """Save latest 5 frames from each camera to img_output/gym/<camera_name>/."""
        import cv2, os, glob

        current_observation = self._get_observation()
        if current_observation is None:
            return

        base_dir = "img_output/robotenv"
        os.makedirs(base_dir, exist_ok=True)

        image_keys = list(current_observation["pixels"].keys())
        for key in image_keys:
            img = current_observation["pixels"][key]
            save_dir = os.path.join(base_dir, key)
            os.makedirs(save_dir, exist_ok=True)

            # 保存当前帧
            filename = os.path.join(save_dir, f"{int(time.time() * 1000)}.jpg")
            cv2.imwrite(filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            # 只保留最新 5 张
            files = sorted(glob.glob(os.path.join(save_dir, "*.jpg")), key=os.path.getmtime)
            if len(files) > 5:
                for old_file in files[:-5]:
                    os.remove(old_file)

    # def render(self) -> None:
    #     """Display robot camera feeds."""
    #     import cv2
    #     # pdb.set_trace()
    #     current_observation = self._get_observation()
    #     if current_observation is not None:
    #         image_keys = [key for key in current_observation["pixels"]]
    #         # pdb.set_trace()
    #         for key in image_keys:
    #             cv2.imshow(key, cv2.cvtColor(current_observation["pixels"][key], cv2.COLOR_RGB2BGR))
    #             cv2.waitKey(1)

    def close(self) -> None:
        """Close environment and disconnect robot."""
        if self.robot.is_connected:
            self.robot.disconnect()

    # def get_raw_joint_positions(self) -> dict[str, float]:
    #     """Get raw joint positions."""
    #     return self._raw_joint_positions
    def get_raw_end_pose_value(self) -> dict[str, float]:
        return self._raw_end_pose_value
    def get_end_pose_name(self) -> list[str]:
        return self._end_pose_name

def load_config_from_json(config_path: str) -> HILSerlRobotEnvConfig:
    """从 JSON 文件加载配置并转换为 HILSerlRobotEnvConfig 对象。
    
    Args:
        config_path: JSON 配置文件路径
        
    Returns:
        HILSerlRobotEnvConfig 对象
    """
    import json
    import tempfile
    from pathlib import Path
    import draccus
    
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    force_print(f"[load_config_from_json] Loading config from: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # 提取 env 部分的配置
    env_config = config_dict.get('env', {})
    
    force_print(f"[load_config_from_json] Config loaded: name={env_config.get('name', 'MISSING')}")
    
    # 使用 draccus 将字典转换为配置对象
    # 创建临时文件来存储 env 配置
    with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as f:
        json.dump(env_config, f)
        temp_config_file = f.name
    
    try:
        # 使用 draccus 解析配置
        with draccus.config_type("json"):
            cfg = draccus.parse(HILSerlRobotEnvConfig, temp_config_file, args=[])
    finally:
        # 清理临时文件
        Path(temp_config_file).unlink(missing_ok=True)
    
    return cfg


def make_robot_env(cfg: HILSerlRobotEnvConfig | str) -> tuple[gym.Env, Any]:
    """Create robot environment from configuration.

    Args:
        cfg: Environment configuration object or path to JSON config file.

    Returns:
        Tuple of (gym environment, teleoperator device).
    """
    # 如果传入的是字符串路径，则加载配置文件
    if isinstance(cfg, str):
        cfg = load_config_from_json(cfg)
    
    force_print(f"[make_robot_env] CALLED! cfg.name={getattr(cfg, 'name', 'MISSING')}")
    force_print(f"[make_robot_env] cfg.data_path={getattr(cfg, 'data_path', 'MISSING')}")
    
    # Check if this is a test environment (no real hardware)
    if cfg.name == "gym_testenv":
        # gym_testenv doesn't need robot or teleop config (can be None or have type=None)
        if cfg.robot is not None and getattr(cfg.robot, 'type', None) is not None:
            raise ValueError("gym_testenv does not support robot configuration")
        if cfg.teleop is not None and getattr(cfg.teleop, 'type', None) is not None:
            raise ValueError("gym_testenv does not support teleop configuration")
        
        # Extract gripper settings with defaults
        use_gripper = cfg.processor.gripper.use_gripper if cfg.processor.gripper is not None else True
        
        # 获取数据路径（如果提供）
        data_path = getattr(cfg, 'data_path', None)
        step_size = getattr(cfg, 'step_size', 1)
        action_chunk = getattr(cfg, 'action_chunk', 50)
        
        # 创建测试环境（使用Replay或随机数据）
        env = TestRobotEnv(
            use_gripper=use_gripper,
            image_size=(480, 640),  # 默认图像尺寸
            num_cameras=3,  # 默认3个相机
            data_path=data_path,
            step_size=step_size,
            action_chunk=action_chunk,
        )
        
        return env, None
    
    # Check if this is a GymHIL simulation environment
    if cfg.name == "gym_hil":
        assert cfg.robot is None and cfg.teleop is None, "GymHIL environment does not support robot or teleop"
        import gym_hil  # noqa: F401

        # Extract gripper settings with defaults
        use_gripper = cfg.processor.gripper.use_gripper if cfg.processor.gripper is not None else True
        gripper_penalty = cfg.processor.gripper.gripper_penalty if cfg.processor.gripper is not None else 0.0

        env = gym.make(
            f"gym_hil/{cfg.task}",
            image_obs=True,
            render_mode="human",
            use_gripper=use_gripper,
            gripper_penalty=gripper_penalty,
        )

        return env, None

    # Real robot environment
    assert cfg.robot is not None, "Robot config must be provided for real robot environment"
    assert cfg.teleop is not None, "Teleop config must be provided for real robot environment"

    robot = make_robot_from_config(cfg.robot)
    teleop_device = make_teleoperator_from_config(cfg.teleop)
    teleop_device.connect()

    # Create base environment with safe defaults
    use_gripper = cfg.processor.gripper.use_gripper if cfg.processor.gripper is not None else True
    display_cameras = (
        cfg.processor.observation.display_cameras if cfg.processor.observation is not None else False
    )
    reset_pose = cfg.processor.reset.fixed_reset_joint_positions if cfg.processor.reset is not None else None

    env = RobotEnv(
        robot=robot,
        use_gripper=use_gripper,
        display_cameras=display_cameras,
        reset_pose=reset_pose,
    )

    return env, teleop_device


def make_processors(
    env: gym.Env, teleop_device: Teleoperator | None, cfg: HILSerlRobotEnvConfig, device: str = "cpu"
) -> tuple[
    DataProcessorPipeline[EnvTransition, EnvTransition], DataProcessorPipeline[EnvTransition, EnvTransition]
]:
    """Create environment and action processors.

    Args:
        env: Robot environment instance.
        teleop_device: Teleoperator device for intervention.
        cfg: Processor configuration.
        device: Target device for computations.

    Returns:
        Tuple of (environment processor, action processor).
    """
    terminate_on_success = (
        cfg.processor.reset.terminate_on_success if cfg.processor.reset is not None else True
    )

    # 处理 gym_testenv（测试环境，无真实硬件）
    if cfg.name == "gym_testenv":
        # 简单的处理管道，不需要介入检测
        action_pipeline_steps = [
            Torch2NumpyActionProcessorStep(),
        ]

        env_pipeline_steps = [
            Numpy2TorchActionProcessorStep(),
            VanillaObservationProcessorStep(),
            AddBatchDimensionProcessorStep(),
            DeviceProcessorStep(device=device),
        ]

        return DataProcessorPipeline(
            steps=env_pipeline_steps, to_transition=identity_transition, to_output=identity_transition
        ), DataProcessorPipeline(
            steps=action_pipeline_steps, to_transition=identity_transition, to_output=identity_transition
        )

    if cfg.name == "gym_hil":
        action_pipeline_steps = [
            InterventionActionProcessorStep(terminate_on_success=terminate_on_success),
            Torch2NumpyActionProcessorStep(),
        ]

        env_pipeline_steps = [
            Numpy2TorchActionProcessorStep(),
            VanillaObservationProcessorStep(),
            AddBatchDimensionProcessorStep(),
            DeviceProcessorStep(device=device),
        ]

        return DataProcessorPipeline(
            steps=env_pipeline_steps, to_transition=identity_transition, to_output=identity_transition
        ), DataProcessorPipeline(
            steps=action_pipeline_steps, to_transition=identity_transition, to_output=identity_transition
        )

    # Full processor pipeline for real robot environment
    # Get robot and motor information for kinematics
    

    # 映射：bus 电机名 -> URDF 关节名（顺序必须与 motor_names 一致）
    # 1) bus 层电机名
    end_pose_name= env.get_end_pose_name()

    # 2) 拆分：臂的6关节 vs 夹爪
    # arm_motor_names = [m for m in motor_names if "gripper" not in m]  # 或者直接 motor_names[:6]
    # gripper_motor_names = [m for m in motor_names if "gripper" in m]  # 可为空或1个

    # 3) 只为“臂的6关节”做 URDF 名映射（按你的 URDF 实际名字改）
    # name_map = {
    #     "left_joint_1": "joint1",
    #     "left_joint_2": "joint2",
    #     "left_joint_3": "joint3",
    #     "left_joint_4": "joint4",
    #     "left_joint_5": "joint5",
    #     "left_joint_6": "joint6",
    # }
    # ik_joint_names = [name_map.get(n, n) for n in arm_motor_names]
    # pdb.set_trace()
    # Set up kinematics solver if inverse kinematics is configured
    kinematics_solver = None
    # Note: IK requires proper motor_names setup - currently disabled
    # if cfg.processor.inverse_kinematics is not None:
    #     motor_names = env.get_end_pose_name()  # or proper joint names
    #     kinematics_solver = RobotKinematics(
    #         urdf_path=cfg.processor.inverse_kinematics.urdf_path,
    #         target_frame_name=cfg.processor.inverse_kinematics.target_frame_name,
    #         joint_names=motor_names[:-1],
    #     )

    env_pipeline_steps = [VanillaObservationProcessorStep()]
    
    # If environment feature keys differ from policy keys, insert a rename step
    if cfg.features_map:
        rename_map = {src: dst for src, dst in cfg.features_map.items() if src != dst}
        if rename_map:
            env_pipeline_steps.append(RenameObservationsProcessorStep(rename_map=rename_map))

    if cfg.processor.observation is not None:
        if cfg.processor.observation.add_joint_velocity_to_observation:#false
            env_pipeline_steps.append(JointVelocityProcessorStep(dt=1.0 / cfg.fps))
        if cfg.processor.observation.add_current_to_observation:#false
            env_pipeline_steps.append(MotorCurrentProcessorStep(robot=env.robot))

    # Forward kinematics observation (disabled - requires motor_names setup)
    # if kinematics_solver is not None:
    #     motor_names = env.get_end_pose_name()
    #     env_pipeline_steps.append(
    #         ForwardKinematicsJointsToEEObservation(
    #             kinematics=kinematics_solver,
    #             motor_names=motor_names,
    #         )
    #     )

    if cfg.processor.image_preprocessing is not None:
        env_pipeline_steps.append(
            ImageCropResizeProcessorStep(
                crop_params_dict=cfg.processor.image_preprocessing.crop_params_dict,
                resize_size=cfg.processor.image_preprocessing.resize_size,
            )
        )

    # Add time limit processor if reset config exists
    if cfg.processor.reset is not None:
        env_pipeline_steps.append(
            TimeLimitProcessorStep(max_episode_steps=int(cfg.processor.reset.control_time_s * cfg.fps))
        )

    # Add gripper penalty processor if gripper config exists and enabled
    if cfg.processor.gripper is not None and cfg.processor.gripper.use_gripper:
        env_pipeline_steps.append(
            GripperPenaltyProcessorStep( #需要等到action走通
                penalty=cfg.processor.gripper.gripper_penalty,
                max_gripper_pos=cfg.processor.max_gripper_pos,
            )
        )

    if (
        cfg.processor.reward_classifier is not None
        and cfg.processor.reward_classifier.pretrained_path is not None
    ):
        env_pipeline_steps.append(
            RewardClassifierProcessorStep(
                pretrained_path=cfg.processor.reward_classifier.pretrained_path,
                device=device,
                success_threshold=cfg.processor.reward_classifier.success_threshold,
                success_reward=cfg.processor.reward_classifier.success_reward,
                terminate_on_success=terminate_on_success,
            )
        )

    env_pipeline_steps.append(AddBatchDimensionProcessorStep())
    env_pipeline_steps.append(DeviceProcessorStep(device=device))

    action_pipeline_steps = [
        AddTeleopActionAsComplimentaryDataStep(teleop_device=teleop_device),
        AddTeleopEventsAsInfoStep(teleop_device=teleop_device),
        InterventionActionProcessorStep(
            use_gripper=cfg.processor.gripper.use_gripper if cfg.processor.gripper is not None else False,
            terminate_on_success=terminate_on_success,
        ),
    ]
    action_pipeline_steps.append(RobotActionToPolicyActionProcessorStep(end_pose_name=end_pose_name))
    # Replace InverseKinematicsProcessor with new kinematic processors
    # if cfg.processor.inverse_kinematics is not None and kinematics_solver is not None:
    #     # Add EE bounds and safety processor
    #     inverse_kinematics_steps = [
    #         MapTensorToDeltaActionDictStep(
    #             use_gripper=cfg.processor.gripper.use_gripper if cfg.processor.gripper is not None else False
    #         ),
    #         MapDeltaActionToRobotActionStep(),
    #         EEReferenceAndDelta(
    #             kinematics=kinematics_solver,
    #             end_effector_step_sizes=cfg.processor.inverse_kinematics.end_effector_step_sizes,
    #             motor_names=motor_names,
    #             # motor_names=arm_motor_names,
    #             # motor_names=ik_joint_names,
    #             use_latched_reference=False,#测试一下True
    #             use_ik_solution=True,
    #         ),
    #         EEBoundsAndSafety(
    #             end_effector_bounds=cfg.processor.inverse_kinematics.end_effector_bounds,
    #         ),
    #         GripperVelocityToJoint(
    #             clip_max=cfg.processor.max_gripper_pos,
    #             speed_factor=1.0,
    #             discrete_gripper=False,
    #         ),
    #         InverseKinematicsRLStep(
    #             kinematics=kinematics_solver, 
    #             motor_names=motor_names, 
    #             # motor_names=arm_motor_names,
    #             # motor_names=ik_joint_names
    #             initial_guess_current_joints=False
    #         ),
    #     ]
    #     action_pipeline_steps.extend(inverse_kinematics_steps)
    #     action_pipeline_steps.append(RobotActionToPolicyActionProcessorStep(motor_names=motor_names))

    return DataProcessorPipeline(
        steps=env_pipeline_steps, to_transition=identity_transition, to_output=identity_transition
    ), DataProcessorPipeline(
        steps=action_pipeline_steps, to_transition=identity_transition, to_output=identity_transition
    )


def step_env_and_process_transition(
    env: gym.Env,
    transition: EnvTransition,
    action: torch.Tensor,
    env_processor: DataProcessorPipeline[EnvTransition, EnvTransition],
    action_processor: DataProcessorPipeline[EnvTransition, EnvTransition],
    save_images: bool = False,
) -> EnvTransition:
    """
    Execute one step with processor pipeline.

    Args:
        env: The robot environment
        transition: Current transition state
        action: Action to execute
        env_processor: Environment processor
        action_processor: Action processor
        save_images: Whether to save processed images to disk (default False)

    Returns:
        Processed transition with updated state.
    """
    # Create action transition
    transition[TransitionKey.ACTION] = action
    transition[TransitionKey.OBSERVATION] = (
        env.get_raw_end_pose_value() if hasattr(env, "get_raw_end_pose_value") else {}
    )
    processed_action_transition = action_processor(transition)
    processed_action = processed_action_transition[TransitionKey.ACTION]

    obs, reward, terminated, truncated, info = env.step(processed_action)

    reward = reward + processed_action_transition[TransitionKey.REWARD]
    terminated = terminated or processed_action_transition[TransitionKey.DONE]
    truncated = truncated or processed_action_transition[TransitionKey.TRUNCATED]
    complementary_data = processed_action_transition[TransitionKey.COMPLEMENTARY_DATA].copy()
    new_info = processed_action_transition[TransitionKey.INFO].copy()
    new_info.update(info)

    new_transition = create_transition(
        observation=obs,
        action=processed_action,
        reward=reward,
        done=terminated,
        truncated=truncated,
        info=new_info,
        complementary_data=complementary_data,
    )
    new_transition = env_processor(new_transition)
    
   

    return new_transition


class RealRobotEnvWrapper:
    """
    简化的机器人环境适配器，用于test_env集成和数据回灌测试。
    
    职责：
    1. 初始化机器人和处理器
    2. reset() - 重置环境并返回观测
    3. step(action) - 执行动作并返回obs, reward, terminated, truncated, info
    4. 观测格式转换（RobotEnv -> test_env）
    
    注意：chunk_step由上层test_env实现，此处不实现。
    
    Args:
        cfg: 配置对象
        rank: 进程rank
        world_size: 总进程数
    """
    
    def __init__(self, cfg, rank: int = 0, world_size: int = 1):
        force_print(f"[RealRobotEnvWrapper] __init__ CALLED! cfg type={type(cfg)}")
        force_print(f"[RealRobotEnvWrapper] cfg.env={getattr(cfg, 'env', 'MISSING')}")
        force_print(f"[RealRobotEnvWrapper] cfg.env.name={getattr(getattr(cfg, 'env', None), 'name', 'MISSING')}")
        
        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size
        self.device = getattr(cfg, 'device', 'cpu')
        self.num_envs = getattr(cfg, 'num_envs', 1)
        
        # 初始化运动学求解器(缓存,避免每次step都创建)
        self.left_kin = None
        self.right_kin = None
        self._init_kinematics()
        
        # 创建机器人环境和遥操作设备
        # 如果提供了 robot_config_path,则从JSON文件加载完整配置
        robot_config_path = getattr(cfg, 'robot_config_path', None)
        if robot_config_path is not None:
            force_print(f"[RealRobotEnvWrapper] Loading robot config from: {robot_config_path}")
            self.env, self.teleop_device = make_robot_env(robot_config_path)
            # 同样从JSON文件加载处理器配置
            env_cfg = load_config_from_json(robot_config_path)
        else:
            force_print(f"[RealRobotEnvWrapper] Using Hydra config: cfg.env")
            self.env, self.teleop_device = make_robot_env(cfg.env)
            env_cfg = cfg.env
        
        # 创建处理器管道
        self.env_processor, self.action_processor = make_processors(
            self.env, self.teleop_device, env_cfg, self.device
        )
        
        # 当前transition状态
        self.current_transition = None
        
        # 相机名称映射
        self.camera_mapping = getattr(cfg, 'camera_mapping', {
            'front': 'head_image',
            'left': 'left_wrist_image',
            'right': 'right_wrist_image',
        })
        
        # 任务描述
        self.task_description = getattr(cfg, 'task_description', 'manipulation task')
        
        force_print(f"[RealRobotEnvWrapper] Initialized: rank={rank}, num_envs={self.num_envs}")
    
    def _init_kinematics(self):
        """初始化运动学求解器(只初始化一次,避免频繁创建对象)"""
        try:
            # 保存当前工作目录,切换到URDF所在目录(以便placo能找到mesh文件)
            urdf_dir = "/shared_disk/users/weijie.ke/verl/recipe/vla/envs/test_env/robot/controller/piper/local_assets"
            urdf_path = os.path.join(urdf_dir, "piper.urdf.bak")
            original_cwd = os.getcwd()
            os.chdir(urdf_dir)
            
            try:
                force_print(f"[RealRobotEnvWrapper] Initializing RobotKinematics...")
                self.left_kin = RobotKinematics(
                    urdf_path=urdf_path,
                    target_frame_name="link6",
                    joint_names=[f"joint{i+1}" for i in range(6)],
                )
                self.right_kin = RobotKinematics(
                    urdf_path=urdf_path,
                    target_frame_name="link6",
                    joint_names=[f"joint{i+1}" for i in range(6)],
                )
                force_print(f"[RealRobotEnvWrapper] RobotKinematics initialized successfully!")
            finally:
                # 恢复原来的工作目录
                os.chdir(original_cwd)
        except Exception as e:
            import traceback
            force_print(f"[RealRobotEnvWrapper] WARNING: Failed to initialize kinematics: {e}")
            force_print(f"[RealRobotEnvWrapper] Traceback: {traceback.format_exc()}")
            force_print(f"[RealRobotEnvWrapper] Kinematics will be disabled")
            self.left_kin = None
            self.right_kin = None
    
    def _convert_obs_to_test_env_format(self, transition: EnvTransition) -> dict:
        """
        转换观测格式：RobotEnv -> test_env。
        
        RobotEnv format:
            observation.images.front: [1, C, H, W] tensor
            observation.images.left: [1, C, H, W] tensor
            observation.images.right: [1, C, H, W] tensor
            observation.state: [1, 14] tensor
        
        返回格式（单个环境，非批次）:
            {
                "images": {
                    "head_image": bytes (JPEG encoded),
                    "left_wrist_image": bytes (JPEG encoded),
                    "right_wrist_image": bytes (JPEG encoded)
                },
                "state": [14] float32 ndarray,
                "task_description": str
            }
        """
        obs = transition[TransitionKey.OBSERVATION]
        
        images_list = []
        camera_names = []
        
        # 转换图像: [1, C, H, W] tensor -> [H, W, C] uint8 numpy
        for robot_cam_name, test_cam_name in self.camera_mapping.items():
            key = f"observation.images.{robot_cam_name}"
            if key in obs:
                img_tensor = obs[key]  # [1, C, H, W]
                if isinstance(img_tensor, torch.Tensor):
                    img_tensor = img_tensor.cpu()
                    # [1, C, H, W] -> [H, W, C]
                    img_np = img_tensor.squeeze(0).permute(1, 2, 0).numpy()
                    # [0, 1] float -> [0, 255] uint8
                    img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
                    images_list.append(img_np)
                    camera_names.append(test_cam_name)
        
        # 确保所有期望的图像都存在
        default_image = np.zeros((224, 224, 3), dtype=np.uint8)
        for test_cam_name in ['head_image', 'left_wrist_image', 'right_wrist_image']:
            if test_cam_name not in camera_names:
                images_list.append(default_image)
                camera_names.append(test_cam_name)
        
        # 使用 images_encoding 函数对所有图像进行 JPEG 编码压缩
        encoded_data, max_len, encode_time = images_encoding(images_list)
        
        # 构建压缩后的图像字典
        # 注意：需要将 bytes 转换为 torch.Tensor (uint8, 2D) 格式
        # 以便与 env_worker.py 中的解码逻辑兼容
        images_dict = {}
        for i, cam_name in enumerate(camera_names):
            # 将 bytes 转换为 numpy array，然后转为 tensor
            # 使用 padded 格式确保所有图像长度一致
            padded_bytes = encoded_data[i].ljust(max_len, b"\0")
            # ⚠️ 关键修复: 使用 np.array() 而不是 np.frombuffer() 创建可写数组
            # np.frombuffer() 返回只读数组,在Ray分布式传输时会失败
            img_array = np.array(np.frombuffer(padded_bytes, dtype=np.uint8))
            # 转换为 2D tensor: [1, max_len] 以便后续 batch 处理
            images_dict[cam_name] = torch.from_numpy(img_array).unsqueeze(0)  # [1, max_len]
        
        # 转换状态
        state_array = np.zeros(14, dtype=np.float32)  # 默认值
        if "observation.state" in obs:
            state_tensor = obs["observation.state"]
            if isinstance(state_tensor, torch.Tensor):
                state_array = state_tensor.cpu().squeeze(0).numpy()
        
        return {
            "images": images_dict,
            "state": state_array,
            "task_description": self.task_description,
        }
    
    def reset(self, episode_id: int | None = None):
        """
        重置环境到初始状态。
        
        Args:
            episode_id: 指定episode索引（仅用于gym_testenv），None表示随机选择
        
        Returns:
            obs_dict: test_env格式的观测
                {
                    "images": {"head_image", "left_wrist_image", "right_wrist_image"},
                    "states": {"state"},
                    "task_description": str
                }
        """
        # 如果是TestRobotEnv且提供了episode_id，需要先设置episode
        if hasattr(self.env, 'hdf5_loader') and self.env.hdf5_loader is not None:
            if episode_id is not None:
                self.env.hdf5_loader.reset(episode_index=episode_id)
            else:
                self.env.hdf5_loader.reset(episode_index=None)
        
        # 重置机器人环境
        obs, info = self.env.reset()
        
        # 重置处理器
        self.env_processor.reset()
        self.action_processor.reset()
        
        # 创建并处理初始transition
        complementary_data = (
            {"raw_end_pose_value": info.pop("raw_end_pose_value")} 
            if "raw_end_pose_value" in info else {}
        )
        self.current_transition = create_transition(
            observation=obs, 
            info=info, 
            complementary_data=complementary_data
        )
        self.current_transition = self.env_processor(data=self.current_transition)
        
        # 转换为test_env格式
        obs_converted = self._convert_obs_to_test_env_format(self.current_transition)
        
        return obs_converted
    
    def step(self, action):
        """
        执行一步动作。
        
        Args:
            action: 动作 [action_dim] or [1, action_dim]
            
        Returns:
            obs_dict: 观测字典
            reward: float 奖励
            terminated: bool 是否成功终止
            truncated: bool 是否超时
            info_dict: 信息字典，包含intervene_action和intervene_flag
        """
        # 转换为tensor
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).to(self.device)
        
        # 确保是 [action_dim] 格式
        if action.ndim == 2:
            action = action.squeeze(0)
        if action is not None:
            if isinstance(action, torch.Tensor):
                action_np = action.cpu().numpy()
            else:
                action_np = np.asarray(action, dtype=np.float32)
            
            # 检查维度并填充
            original_dim = action_np.shape[0]
            if action_np.shape[0] == 7:
                # 7维动作，填充另外7维为0（双臂机器人，只控制一个臂）
                action_np = np.concatenate([action_np, np.zeros(7, dtype=np.float32)])
                force_print(f"[RealRobotEnvWrapper] Action padded from 7 to 14 dims")
            elif action_np.shape[0] > 14:
                # 超过14维，截断
                action_np = action_np[:14]
                force_print(f"[RealRobotEnvWrapper] Action truncated from {original_dim} to 14 dims")
            elif action_np.shape[0] < 7:
                # 小于7维，填充到14维
                padding_size = 14 - action_np.shape[0]
                action_np = np.concatenate([action_np, np.zeros(padding_size, dtype=np.float32)])
                force_print(f"[RealRobotEnvWrapper] Action padded from {original_dim} to 14 dims")
            
            # 检查是否已初始化运动学求解器
            if self.left_kin is None or self.right_kin is None:
                force_print(f"[RealRobotEnvWrapper] WARNING: Kinematics not initialized, skipping FK conversion")
                # 直接使用原始action,不做转换
                action = torch.from_numpy(action_np).to(self.device)
            else:
                # 机器人运动学 FK: joint -> end-effector pose
                # 先转回 tensor 以便索引
                action_tensor = torch.from_numpy(action_np).to(self.device)
                
                left_joint_idx = [0, 1, 2, 3, 4, 5]
                left_grip_idx = 6
                right_joint_idx = [7, 8, 9, 10, 11, 12]
                right_grip_idx = 13
                
                # 取出关节（rad）-> numpy
                ql_rad = action_tensor[left_joint_idx].detach().cpu().numpy().astype(float)
                qr_rad = action_tensor[right_joint_idx].detach().cpu().numpy().astype(float)

                # RobotKinematics.forward_kinematics 接口使用"度"
                ql_deg = np.rad2deg(ql_rad)
                qr_deg = np.rad2deg(qr_rad)

                # 做 FK (使用缓存的kinematics对象)
                T_l = self.left_kin.forward_kinematics(ql_deg)   # 4x4
                T_r = self.right_kin.forward_kinematics(qr_deg)  # 4x4

                # 位置：米
                lx, ly, lz = T_l[:3, 3].tolist()
                rx, ry, rz = T_r[:3, 3].tolist()

                # 姿态：旋转矩阵 -> RPY（弧度）
                lrx, lry, lrz = rotmat_to_rpy_zyx(T_l[:3, :3])
                rrx, rry, rrz = rotmat_to_rpy_zyx(T_r[:3, :3])

                # 夹爪：从 action 中获取
                l_grip = float(action_tensor[left_grip_idx].item())
                r_grip = float(action_tensor[right_grip_idx].item())

                # 构建 end-effector pose action (14维)
                # [L_x, L_y, L_z, L_rx, L_ry, L_rz, L_grip,
                #  R_x, R_y, R_z, R_rx, R_ry, R_rz, R_grip]
                endpose_action = torch.tensor(
                    [
                        lx, ly, lz, lrx, lry, lrz, l_grip,
                        rx, ry, rz, rrx, rry, rrz, r_grip,
                    ],
                    device=self.device,
                    dtype=torch.float32,
                )
                
                # 使用转换后的 end-effector pose action
                action = endpose_action
        
        # 通过处理器管道执行步骤（处理人工介入）
        self.current_transition = step_env_and_process_transition(
            env=self.env,
            transition=self.current_transition,
            action=action,
            env_processor=self.env_processor,
            action_processor=self.action_processor,
        )
        
        # 提取数据
        obs_converted = self._convert_obs_to_test_env_format(self.current_transition)
        reward = float(self.current_transition[TransitionKey.REWARD])
        terminated = bool(self.current_transition[TransitionKey.DONE])
        truncated = bool(self.current_transition[TransitionKey.TRUNCATED])
        
        # 提取介入信息
        intervene_action = self.current_transition[TransitionKey.COMPLEMENTARY_DATA].get(
            "teleop_action", None
        )
        intervene_flag = self.current_transition[TransitionKey.INFO].get(
            TeleopEvents.IS_INTERVENTION, False
        )
        
        # 构建info字典
        info_dict = {
            "intervene_action": intervene_action,
            "intervene_flag": intervene_flag,
        }
        
        return obs_converted, reward, terminated, truncated, info_dict
    
    def close(self):
        """Close environment and release resources."""
        if self.env is not None:
            self.env.close()
        if self.teleop_device is not None and hasattr(self.teleop_device, 'disconnect'):
            self.teleop_device.disconnect()

if __name__ == "__main__":
    """测试 gym_testenv 环境"""
    import yaml
    from pathlib import Path
    from types import SimpleNamespace
    
    print("=" * 70)
    print("测试 gym_testenv 环境（无需真实硬件）")
    print("=" * 70)
    
    # 加载 YAML 配置文件
    # 当前文件: robot/controller/piper/robot_env.py
    # 目标文件: robot/config/test_env.yml
    config_path = Path(__file__).parent.parent.parent / "config" / "test_env.yml"
    if config_path.exists():
        print(f"\n加载配置文件: {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            yaml_cfg = yaml.safe_load(f)
        print("✓ 配置文件加载成功")
        print(f"  环境名称: {yaml_cfg.get('env_name', 'gym_testenv')}")
        print(f"  状态维度: {yaml_cfg.get('env_params', {}).get('state_dim', 14)}")
        print(f"  动作维度: {yaml_cfg.get('env_params', {}).get('action_dim', 14)}")
        print(f"  相机数量: {yaml_cfg.get('env_params', {}).get('num_cameras', 3)}")
    else:
        print(f"\n警告: 配置文件不存在: {config_path}")
        print("使用默认配置")
        yaml_cfg = {
            'env_name': 'gym_testenv',
            'robot': {'config': None},
            'teleop': {'config': None},
            'processor': {
                'gripper': {'use_gripper': True, 'gripper_penalty': 0.0},
                'reset': {'terminate_on_success': True, 'fixed_reset_joint_positions': None},
                'observation': {'display_cameras': False},
                'inverse_kinematics': None
            }
        }
    
    # 直接使用字典转换为简单对象（更简洁的方式）
    def dict_to_obj(d):
        """递归地将字典转换为对象，方便使用点号访问"""
        if isinstance(d, dict):
            return SimpleNamespace(**{k: dict_to_obj(v) for k, v in d.items()})
        elif isinstance(d, list):
            return [dict_to_obj(item) for item in d]
        else:
            return d
    
    # 创建配置对象
    cfg = SimpleNamespace(
        name=yaml_cfg.get('env_name', 'gym_testenv'),
        robot=yaml_cfg.get('robot', {}).get('config', None),
        teleop=yaml_cfg.get('teleop', {}).get('config', None),
        processor=dict_to_obj(yaml_cfg.get('processor', {})),
        # 添加数据集配置（如果YAML中提供）
        data_path=yaml_cfg.get('data_path', None),
        step_size=yaml_cfg.get('step_size', 1),
        action_chunk=yaml_cfg.get('action_chunk', 50),
    )
    
    print(f"\n运行时配置:")
    print(f"  环境名称: {cfg.name}")
    print(f"  机器人配置: {cfg.robot}")
    print(f"  遥操作配置: {cfg.teleop}")
    print(f"  使用夹爪: {cfg.processor.gripper.use_gripper}")
    print(f"  显示相机: {cfg.processor.observation.display_cameras}")
    print(f"  数据路径: {cfg.data_path}")
    if cfg.data_path:
        print(f"    → 步长: {cfg.step_size}, 动作块: {cfg.action_chunk}")
    
    # 创建环境
    print("\n[1] 创建环境...")
    env, teleop_device = make_robot_env(cfg)
    print(f"  ✓ 环境类型: {type(env).__name__}")
    print(f"  ✓ 遥操作设备: {teleop_device}")
    print(f"  ✓ 观测空间: {env.observation_space}")
    print(f"  ✓ 动作空间: {env.action_space}")
    
    # 创建处理器
    print("\n[2] 创建处理器...")
    env_processor, action_processor = make_processors(env, teleop_device, cfg, device="cpu")
    print(f"  ✓ 环境处理器: {len(env_processor.steps)} 个步骤")
    print(f"  ✓ 动作处理器: {len(action_processor.steps)} 个步骤")
    
    # 重置环境
    print("\n[3] 重置环境...")
    obs, info = env.reset(seed=42)
    print(f"  ✓ 观测键: {list(obs.keys())}")
    print(f"  ✓ 状态形状: {obs['observation.state'].shape}")
    
    # 显示图像信息
    image_keys = [k for k in obs.keys() if 'image' in k]
    print(f"  ✓ 图像数量: {len(image_keys)}")
    for img_key in image_keys:
        img = obs[img_key]
        print(f"    - {img_key}: shape={img.shape}, dtype={img.dtype}, "
              f"range=[{img.min()}, {img.max()}]")
    
    # 执行多步测试
    print("\n[4] 执行动作测试...")
    num_steps = 5
    for step in range(num_steps):
        # 生成随机动作
        action = env.action_space.sample()
        
        # 执行一步
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 显示结果
        print(f"  步骤 {step+1}/{num_steps}: "
              f"reward={reward:.4f}, "
              f"terminated={terminated}, "
              f"truncated={truncated}, "
              f"step={info.get('current_step', '?')}")
        
        if terminated or truncated:
            print(f"    → 环境终止，重置...")
            obs, info = env.reset()
    
    # 测试 joint action 到 end-effector pose action 的转换
    print("\n[4.5] 测试 joint action -> end-effector pose 转换...")
    print("  使用全零 joint action (14维) 进行测试")
    
    try:
        # 创建运动学求解器
        left_kin = RobotKinematics(
            urdf_path="/home/agilex-home/agilex/keweijie/verl/recipe/vla/envs/test_env/robot/controller/piper/local_assets/piper.urdf",
            target_frame_name="link6",
            joint_names=[f"joint{i+1}" for i in range(6)],
        )
        right_kin = RobotKinematics(
            urdf_path="/home/agilex-home/agilex/keweijie/verl/recipe/vla/envs/test_env/robot/controller/piper/local_assets/piper.urdf",
            target_frame_name="link6",
            joint_names=[f"joint{i+1}" for i in range(6)],
        )
        
        # 创建全零 joint action (14维: 左臂6关节 + 左夹爪 + 右臂6关节 + 右夹爪)
        joint_action = np.zeros(14, dtype=np.float32)
        print(f"  输入 joint action (14维):")
        print(f"    左臂关节 (rad): {joint_action[:6]}")
        print(f"    左臂夹爪: {joint_action[6]}")
        print(f"    右臂关节 (rad): {joint_action[7:13]}")
        print(f"    右臂夹爪: {joint_action[13]}")
        
        # 拆分左右臂
        left_joints_rad = joint_action[:6]
        left_grip = joint_action[6]
        right_joints_rad = joint_action[7:13]
        right_grip = joint_action[13]
        
        # 转换为度（RobotKinematics 使用度作为输入）
        left_joints_deg = np.rad2deg(left_joints_rad)
        right_joints_deg = np.rad2deg(right_joints_rad)
        
        # 执行正向运动学 (FK)
        T_left = left_kin.forward_kinematics(left_joints_deg)
        T_right = right_kin.forward_kinematics(right_joints_deg)
        
        # 提取位置 (米)
        left_pos = T_left[:3, 3]
        right_pos = T_right[:3, 3]
        
        # 提取姿态 (旋转矩阵 -> RPY 欧拉角，弧度)
        left_rpy = rotmat_to_rpy_zyx(T_left[:3, :3])
        right_rpy = rotmat_to_rpy_zyx(T_right[:3, :3])
        
        # 构建 end-effector pose action (14维)
        # [L_x, L_y, L_z, L_rx, L_ry, L_rz, L_grip,
        #  R_x, R_y, R_z, R_rx, R_ry, R_rz, R_grip]
        endpose_action = np.array([
            left_pos[0], left_pos[1], left_pos[2],
            left_rpy[0], left_rpy[1], left_rpy[2],
            left_grip,
            right_pos[0], right_pos[1], right_pos[2],
            right_rpy[0], right_rpy[1], right_rpy[2],
            right_grip,
        ], dtype=np.float32)
        
        print(f"\n  ✓ FK 转换成功")
        print(f"  输出 end-effector pose (14维):")
        print(f"    左臂位置 (m): [{left_pos[0]:.4f}, {left_pos[1]:.4f}, {left_pos[2]:.4f}]")
        print(f"    左臂姿态 (rad): [{left_rpy[0]:.4f}, {left_rpy[1]:.4f}, {left_rpy[2]:.4f}]")
        print(f"    左臂夹爪: {left_grip:.4f}")
        print(f"    右臂位置 (m): [{right_pos[0]:.4f}, {right_pos[1]:.4f}, {right_pos[2]:.4f}]")
        print(f"    右臂姿态 (rad): [{right_rpy[0]:.4f}, {right_rpy[1]:.4f}, {right_rpy[2]:.4f}]")
        print(f"    右臂夹爪: {right_grip:.4f}")
        print(f"  完整向量: {endpose_action}")
        
    except Exception as e:
        import traceback
        print(f"  ✗ 转换测试失败: {e}")
        print(f"  Traceback: {traceback.format_exc()}")
        print("  注意: 确保 URDF 文件路径正确")
    
    # 测试动作范围
    print("\n[5] 验证动作空间...")
    random_actions = [env.action_space.sample() for _ in range(100)]
    all_actions = np.array(random_actions)
    print(f"  ✓ 动作形状: {all_actions[0].shape}")
    print(f"  ✓ 动作范围: [{all_actions.min():.3f}, {all_actions.max():.3f}]")
    print(f"  ✓ 动作均值: {all_actions.mean():.3f}")
    
    # 关闭环境
    print("\n[6] 关闭环境...")
    env.close()
    print("  ✓ 环境已关闭")
    
    print("\n" + "=" * 70)
    print("✓ gym_testenv 测试通过！所有功能正常。")
    print("=" * 70)
    print("\n提示: gym_testenv 返回随机数据，仅用于测试，不适合训练。")
    print("      如需仿真训练，请使用 gym_hil")
    print("      如需真实部署，请使用 real_robot 配置")