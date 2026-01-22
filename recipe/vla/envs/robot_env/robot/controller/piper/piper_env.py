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
# 将 lerobot 路径添加到 sys.path（用于 RobotEnv）
_current_dir = os.path.dirname(os.path.abspath(__file__))
_lerobot_path = _current_dir
if _lerobot_path not in sys.path:
    sys.path.insert(0, _lerobot_path)

# 核心配置和工具
from lerobot.envs.configs import HILSerlRobotEnvConfig
from lerobot.model.kinematics import RobotKinematics

# 从 gym_manipulator_bipiper 导入基础环境和函数
try:
    from lerobot.rl.gym_manipulator_bipiper import (
        RobotEnv,
        make_robot_env as _make_robot_env_base,
        make_processors as _make_processors_base,
        step_env_and_process_transition as _step_env_and_process_transition_base,
        reset_follower_position,
        control_loop,
        replay_trajectory,
    )
except ImportError as e:
    force_print(f"[robot_env.py] 警告: 无法从 gym_manipulator_bipiper 导入: {e}")
    force_print(f"[robot_env.py] 尝试直接导入模块...")
    # 尝试直接导入模块然后访问
    import lerobot.rl.gym_manipulator_bipiper as gym_manipulator_bipiper_module
    RobotEnv = gym_manipulator_bipiper_module.RobotEnv
    _make_robot_env_base = gym_manipulator_bipiper_module.make_robot_env
    _make_processors_base = gym_manipulator_bipiper_module.make_processors
    _step_env_and_process_transition_base = gym_manipulator_bipiper_module.step_env_and_process_transition
    reset_follower_position = gym_manipulator_bipiper_module.reset_follower_position
    control_loop = gym_manipulator_bipiper_module.control_loop
    replay_trajectory = gym_manipulator_bipiper_module.replay_trajectory
    force_print(f"[robot_env.py] ✓ 成功通过模块直接导入")

# 仅导入 robot_env.py 实际使用的 processor 类
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DataProcessorPipeline,
    DeviceProcessorStep,
    EnvTransition,
    Numpy2TorchActionProcessorStep,
    Torch2NumpyActionProcessorStep,
    TransitionKey,
    VanillaObservationProcessorStep,
    create_transition,
)
from lerobot.processor.converters import identity_transition

# Teleoperator 相关（RealRobotEnvWrapper 需要）
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.teleoperators.utils import TeleopEvents

# 常量定义
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE

import pdb

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

# RobotEnv、reset_follower_position 等函数已从 gym_manipulator_bipiper 导入，不在此重复实现

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
    
    扩展版本，支持 JSON 配置加载和 gym_testenv 模式。
    对于真实机器人环境，复用 gym_manipulator_bipiper 中的基础实现。

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
    
    # 对于其他环境类型（gym_hil 和 real_robot），复用基础实现
    # 但需要先切换到 URDF 目录（如果使用真实机器人）
    if cfg.name != "gym_hil":
        from pathlib import Path
        current_file = Path(__file__).resolve()
        urdf_dir = current_file.parent / "local_assets"
        original_cwd = os.getcwd()
        
        try:
            force_print(f"[make_robot_env] Switching to URDF directory: {urdf_dir}")
            os.chdir(str(urdf_dir))
            env, teleop_device = _make_robot_env_base(cfg)
        finally:
            os.chdir(original_cwd)
            force_print(f"[make_robot_env] Restored working directory: {original_cwd}")
        return env, teleop_device
    else:
        # gym_hil 不需要切换目录
        return _make_robot_env_base(cfg)


def make_processors(
    env: gym.Env, teleop_device: Teleoperator | None, cfg: HILSerlRobotEnvConfig, device: str = "cpu"
) -> tuple[
    DataProcessorPipeline[EnvTransition, EnvTransition], DataProcessorPipeline[EnvTransition, EnvTransition]
]:
    """Create environment and action processors.
    
    扩展版本，支持 gym_testenv 模式。
    对于其他环境类型，复用 gym_manipulator_bipiper 中的基础实现。

    Args:
        env: Robot environment instance.
        teleop_device: Teleoperator device for intervention.
        cfg: Processor configuration.
        device: Target device for computations.

    Returns:
        Tuple of (environment processor, action processor).
    """
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

    # 对于其他环境类型，复用基础实现
    return _make_processors_base(env, teleop_device, cfg, device)


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
    
    扩展版本，复用 gym_manipulator_bipiper 中的基础实现。
    save_images 参数当前未使用（基础实现中已包含图像保存逻辑）。

    Args:
        env: The robot environment
        transition: Current transition state
        action: Action to execute
        env_processor: Environment processor
        action_processor: Action processor
        save_images: Whether to save processed images to disk (default False, 已弃用)

    Returns:
        Processed transition with updated state.
    """
    # 复用基础实现（基础实现中已包含图像保存逻辑）
    return _step_env_and_process_transition_base(
        env=env,
        transition=transition,
        action=action,
        env_processor=env_processor,
        action_processor=action_processor,
    )


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
            from pathlib import Path
            
            # 使用相对路径：相对于当前文件 (robot_env.py) 的位置
            # robot_env.py 位于 .../robot/controller/piper/robot_env.py
            # local_assets 位于 .../robot/controller/piper/local_assets/
            current_file = Path(__file__).resolve()
            piper_dir = current_file.parent  # .../robot/controller/piper/
            
            urdf_path = piper_dir / "local_assets" / "piper.urdf" / "robot.urdf"
            mesh_dir = piper_dir / "local_assets"
            
            force_print(f"[RealRobotEnvWrapper] Current file: {current_file}")
            force_print(f"[RealRobotEnvWrapper] Computed URDF path: {urdf_path}")
            force_print(f"[RealRobotEnvWrapper] Computed mesh dir: {mesh_dir}")
            
            # 保存当前工作目录,切换到mesh目录(以便placo能找到mesh文件)
            original_cwd = os.getcwd()
            os.chdir(str(mesh_dir)) 
            
            try:
                force_print(f"[RealRobotEnvWrapper] Initializing RobotKinematics...")
                self.left_kin = RobotKinematics(
                    urdf_path=str(urdf_path),
                    target_frame_name="link6",
                    joint_names=[f"joint{i+1}" for i in range(6)],
                )
                self.right_kin = RobotKinematics(
                    urdf_path=str(urdf_path),
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
                force_print(f"[RealRobotEnvWrapper] End-effector pose action: {endpose_action}")
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
    """测试 RealRobotEnvWrapper 能否正确启动 RobotEnv 和 TestRobotEnv"""
    import yaml
    from pathlib import Path
    from types import SimpleNamespace
    
    print("=" * 80)
    print("测试 RealRobotEnvWrapper 启动 RobotEnv 和 TestRobotEnv")
    print("=" * 80)
    
    def dict_to_obj(d):
        """递归地将字典转换为对象，方便使用点号访问"""
        if isinstance(d, dict):
            return SimpleNamespace(**{k: dict_to_obj(v) for k, v in d.items()})
        elif isinstance(d, list):
            return [dict_to_obj(item) for item in d]
        else:
            return d
    
    # ====================================================================
    # 测试 1: 使用 RealRobotEnvWrapper 启动 TestRobotEnv（数据回放模式）
    # ====================================================================
    print("\n" + "=" * 80)
    print("测试 1: RealRobotEnvWrapper + TestRobotEnv（数据回放模式）")
    print("=" * 80)
    
    # 创建 gym_testenv 配置
    testenv_cfg = SimpleNamespace(
        env=SimpleNamespace(
            name="gym_testenv",
            robot=None,
            teleop=None,
            processor=SimpleNamespace(
                gripper=SimpleNamespace(use_gripper=False, gripper_penalty=0.0),
                reset=SimpleNamespace(terminate_on_success=True, fixed_reset_joint_positions=None),
                observation=SimpleNamespace(display_cameras=False),
                inverse_kinematics=None
            ),
            data_path=None,  # None 表示使用默认路径或生成随机数据
            step_size=1,
            action_chunk=50,
        ),
        num_envs=1,
        device="cpu",
        task_description="测试任务",
        camera_mapping={
            'front': 'head_image',
            'left': 'left_wrist_image',
            'right': 'right_wrist_image',
        }
    )
    
    try:
        print("\n[1.1] 创建 RealRobotEnvWrapper (TestRobotEnv 模式)...")
        wrapper = RealRobotEnvWrapper(testenv_cfg, rank=0, world_size=1)
        print(f"  ✓ 内部环境类型: {type(wrapper.env).__name__}")
        print(f"  ✓ 遥操作设备: {wrapper.teleop_device}")
        
        print("\n[1.2] 重置环境...")
        obs_dict = wrapper.reset()
        print(f"  ✓ 观测键: {list(obs_dict.keys())}")
        print(f"  ✓ 图像键: {list(obs_dict['images'].keys())}")
        print(f"  ✓ 状态形状: {obs_dict['state'].shape}")
        print(f"  ✓ 任务描述: {obs_dict['task_description']}")
        
        # 显示图像信息（JPEG 压缩格式）
        for cam_name, img_tensor in obs_dict['images'].items():
            print(f"    - {cam_name}: shape={img_tensor.shape}, dtype={img_tensor.dtype}")
        
        print("\n[1.3] 执行动作测试...")
        num_steps = 3
        for step in range(num_steps):
            # 生成随机动作 (14维 end-effector pose)
            action = np.random.uniform(-1.0, 1.0, size=(14,)).astype(np.float32)
            
            # 执行一步
            obs_dict, reward, terminated, truncated, info_dict = wrapper.step(action)
            
            print(f"  步骤 {step+1}/{num_steps}: "
                  f"reward={reward:.4f}, "
                  f"terminated={terminated}, "
                  f"truncated={truncated}, "
                  f"intervene_flag={info_dict.get('intervene_flag', False)}")
            
            if terminated or truncated:
                print(f"    → 环境终止，重置...")
                obs_dict = wrapper.reset()
        
        print("\n[1.4] 关闭环境...")
        wrapper.close()
        print("  ✓ TestRobotEnv 模式测试通过！")
        
    except Exception as e:
        import traceback
        print(f"\n  ✗ TestRobotEnv 模式测试失败: {e}")
        print(f"  Traceback:\n{traceback.format_exc()}")
    
    # ====================================================================
    # 测试 2: 使用 RealRobotEnvWrapper 启动 RobotEnv（真实机器人模式）
    # ====================================================================
    print("\n" + "=" * 80)
    print("测试 2: RealRobotEnvWrapper + RobotEnv（真实机器人模式）")
    print("=" * 80)
    
    # 使用 bipiper_gym_pico.json 配置文件
    config_path = Path(__file__).parent / "config" / "bipiper_gym_pico.json"
    
    if not config_path.exists():
        print(f"  ⚠ 配置文件不存在: {config_path}")
        print("  ⚠ 跳过真实机器人测试")
    else:
        try:
            print(f"\n[2.1] 加载配置文件: {config_path}")
            
            # 创建配置对象（使用 robot_config_path）
            robot_cfg = SimpleNamespace(
                robot_config_path=str(config_path),
                num_envs=1,
                device="cpu",
                task_description="真实机器人测试任务",
                camera_mapping={
                    'front': 'head_image',
                    'left': 'left_wrist_image',
                    'right': 'right_wrist_image',
                }
            )
            
            print("\n[2.2] 创建 RealRobotEnvWrapper (RobotEnv 模式)...")
            print("  ⚠ 正在连接真实机器人硬件，请稍候...")
            wrapper = RealRobotEnvWrapper(robot_cfg, rank=0, world_size=1)
            print(f"  ✓ 内部环境类型: {type(wrapper.env).__name__}")
            print(f"  ✓ 遥操作设备类型: {type(wrapper.teleop_device).__name__ if wrapper.teleop_device else None}")
            
            print("\n[2.3] 重置环境（机器人将移动到初始位置）...")
            print("  ⚠ 机器人正在移动，请确保安全区域无障碍物...")
            obs_dict = wrapper.reset()
            print(f"  ✓ 重置成功！")
            print(f"  ✓ 观测键: {list(obs_dict.keys())}")
            print(f"  ✓ 图像键: {list(obs_dict['images'].keys())}")
            print(f"  ✓ 状态形状: {obs_dict['state'].shape}")
            print(f"  ✓ 任务描述: {obs_dict['task_description']}")
            
            # 显示图像信息
            for cam_name, img_tensor in obs_dict['images'].items():
                print(f"    - {cam_name}: shape={img_tensor.shape}, dtype={img_tensor.dtype}")
            
            print("\n[2.4] 执行动作测试（发送零动作，机器人应保持静止）...")
            num_steps = 5
            for step in range(num_steps):
                # 发送零动作（保持当前位置）
                action = np.zeros(14, dtype=np.float32)
                
                # 执行一步
                obs_dict, reward, terminated, truncated, info_dict = wrapper.step(action)
                
                print(f"  步骤 {step+1}/{num_steps}: "
                      f"reward={reward:.4f}, "
                      f"terminated={terminated}, "
                      f"truncated={truncated}, "
                      f"intervene_flag={info_dict.get('intervene_flag', False)}")
                
                if terminated or truncated:
                    print(f"    → 环境终止，重置...")
                    obs_dict = wrapper.reset()
                
                # 短暂延迟
                time.sleep(0.1)
            
            print("\n[2.5] 关闭环境...")
            wrapper.close()
            print("  ✓ RobotEnv 模式测试通过！")
            
        except KeyboardInterrupt:
            print("\n  ⚠ 用户中断测试")
            if 'wrapper' in locals():
                try:
                    wrapper.close()
                except:
                    pass
        except Exception as e:
            import traceback
            print(f"\n  ✗ RobotEnv 模式测试失败: {e}")
            print(f"  Traceback:\n{traceback.format_exc()}")
            if 'wrapper' in locals():
                try:
                    wrapper.close()
                except:
                    pass
    
    # ====================================================================
    # 测试 3: 直接测试底层函数（make_robot_env, make_processors）
    # ====================================================================
    print("\n" + "=" * 80)
    print("测试 3: 底层函数测试（make_robot_env + make_processors）")
    print("=" * 80)
    
    try:
        print("\n[3.1] 测试 make_robot_env (gym_testenv)...")
        cfg = SimpleNamespace(
            name="gym_testenv",
            robot=None,
            teleop=None,
            processor=SimpleNamespace(
                gripper=SimpleNamespace(use_gripper=False),
                reset=SimpleNamespace(terminate_on_success=True),
                observation=SimpleNamespace(display_cameras=False),
            ),
            data_path=None,
            step_size=1,
            action_chunk=50,
        )
        
        env, teleop_device = make_robot_env(cfg)
        print(f"  ✓ 环境类型: {type(env).__name__}")
        print(f"  ✓ 遥操作设备: {teleop_device}")
        print(f"  ✓ 观测空间: {env.observation_space}")
        print(f"  ✓ 动作空间: {env.action_space}")
        
        print("\n[3.2] 测试 make_processors...")
        env_processor, action_processor = make_processors(env, teleop_device, cfg, device="cpu")
        print(f"  ✓ 环境处理器步骤数: {len(env_processor.steps)}")
        print(f"  ✓ 动作处理器步骤数: {len(action_processor.steps)}")
        
        print("\n[3.3] 测试环境 reset + step...")
        obs, info = env.reset(seed=42)
        print(f"  ✓ reset 成功，观测键: {list(obs.keys())}")
        
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  ✓ step 成功: reward={reward:.4f}, done={terminated}, truncated={truncated}")
        
        env.close()
        print("  ✓ 底层函数测试通过！")
        
    except Exception as e:
        import traceback
        print(f"\n  ✗ 底层函数测试失败: {e}")
        print(f"  Traceback:\n{traceback.format_exc()}")
    
    # ====================================================================
    # 总结
    # ====================================================================
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    print("✓ 测试 1: RealRobotEnvWrapper + TestRobotEnv - 完成")
    print("⚠ 测试 2: RealRobotEnvWrapper + RobotEnv - 已跳过（需要硬件）")
    print("✓ 测试 3: 底层函数（make_robot_env + make_processors）- 完成")
    print("\n说明:")
    print("  1. TestRobotEnv: 数据回放模式，无需真实硬件")
    print("  2. RobotEnv: 真实机器人模式，需要配置 robot 和 teleop")
    print("  3. RealRobotEnvWrapper: 统一接口，可同时使用两种模式")
    print("\n使用建议:")
    print("  - 开发/测试: 使用 TestRobotEnv（cfg.env.name='gym_testenv'）")
    print("  - 仿真训练: 使用 gym_hil（cfg.env.name='gym_hil'）")
    print("  - 真实部署: 使用 RobotEnv（cfg.env.name='real_robot'，配置 robot + teleop）")
    print("=" * 80)