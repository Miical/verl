# Copyright 2025 The RLinf Authors.
# Test environment that interfaces with robot_env for robot control.
# Uses RealRobotEnvWrapper to interface with robot hardware or gym_testenv

import copy
import logging
import os
import time
from typing import Optional, Dict, List

import cv2
import gymnasium as gym
import numpy as np
import torch
import sys
import ray

def images_decoding(encoded_data, valid_len=None):
    """解码图像并记录耗时"""
    decode_start = time.perf_counter()
    imgs = []
    for data in encoded_data:
        if valid_len is not None:
            data = data[:valid_len]
        nparr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        imgs.append(img)
    decode_time = time.perf_counter() - decode_start
    return imgs, decode_time


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
# 强制刷新输出（解决 Ray 环境输出缓冲问题）
def force_print(*args, **kwargs):
    print(*args, **kwargs, flush=True)
    sys.stdout.flush()
    sys.stderr.flush()

force_print(f"[test_env.py] Importing RealRobotEnvWrapper...")
from .robot.controller.piper.robot_env import RealRobotEnvWrapper as RobotEnv
force_print(f"[test_env.py] RobotEnv = {RobotEnv}")
from recipe.vla.envs.action_utils import (
    list_of_dict_to_dict_of_list,
    put_info_on_image,
    save_rollout_video,
    tile_images,
    to_tensor,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # 设置日志级别

class TestEnv(gym.Env):
    """Test environment that interfaces with robot_env for robot control.
    
    This environment wraps RealRobotEnvWrapper to provide a unified interface
    for verl training pipeline. Supports both real robot hardware and gym_testenv
    for data replay testing.
    
    Args:
        cfg: Configuration object containing:
            - num_envs: Number of parallel environments
            - max_episode_steps: Maximum steps per episode
            - seed: Random seed
            - video_cfg: Video recording configuration
            - env: Robot environment configuration (for RealRobotEnvWrapper)
        rank: Current process rank
        world_size: Total number of processes
    """
    
    def __init__(self, cfg, rank, world_size):
        force_print(f"[TestEnv] __init__ CALLED! rank={rank}, world_size={world_size}")
        self.rank = rank
        self.cfg = cfg
        self.world_size = world_size
        self.seed = getattr(cfg, 'seed', 42) + rank
        self.num_envs = getattr(cfg, 'num_envs', 1)
        force_print(f"[TestEnv] num_envs={self.num_envs}, seed={self.seed}")
        
        # 检测是否使用真实机器人模式（通过检查 robot_config_path 或 env.name）
        # 真实机器人模式：不使用 hdf5_loader，每次 reset 都是真实的机器人重置
        self.use_real_robot = self._detect_real_robot_mode(cfg)
        force_print(f"[TestEnv] use_real_robot={self.use_real_robot}")
        
        # Distributed inference related configs
        self.auto_reset = getattr(cfg, 'auto_reset', True)
        self.ignore_terminations = getattr(cfg, 'ignore_terminations', False)
        self.use_fixed_reset_state_ids = getattr(cfg, 'use_fixed_reset_state_ids', False)
        self.group_size = getattr(cfg, 'group_size', 1)
        self.num_group = self.num_envs // self.group_size
        force_print(f"[TestEnv] auto_reset={self.auto_reset}, group_size={self.group_size}")
        
        self._is_start = True
        self._generator = np.random.default_rng(seed=self.seed)
        self._torch_generator = torch.Generator()
        self._torch_generator.manual_seed(self.seed)
        
        # 为每个环境创建RobotEnv实例，传递完整配置
        logger.info(f"[TestEnv] Using robot_env backend")
        force_print(f"[TestEnv] About to create RobotEnv, RobotEnv class = {RobotEnv}")
        force_print(f"[TestEnv] cfg type={type(cfg)}, has 'env' attr={hasattr(cfg, 'env')}")
        self.robot_backends = []
        for i in range(self.num_envs):
            force_print(f"[TestEnv] Creating RobotEnv instance {i}/{self.num_envs}...")
            robot_backend = RobotEnv(cfg, rank, world_size)
            force_print(f"[TestEnv] RobotEnv instance {i} created successfully!")
            self.robot_backends.append(robot_backend)
        force_print(f"[TestEnv] All {self.num_envs} RobotEnv instances created!")
        
        # Metrics
        self.prev_step_reward = np.zeros(self.num_envs)
        self.use_rel_reward = False
        self._init_metrics()
        self._elapsed_steps = np.zeros(self.num_envs, dtype=np.int32)
        
        # Task descriptions
        self.task_descriptions = ["" for _ in range(self.num_envs)]
        
        # Video recording
        self.video_cfg = getattr(cfg, 'video_cfg', None)
        self.video_cnt = 0
        self.render_images = []
        
        # Episode tracking: 记录每个环境当前使用的episode索引
        # -1 表示需要选择新的episode
        self.current_episode_ids = np.full(self.num_envs, -1, dtype=np.int32)
        
        # Build state ID mapping
        self._build_state_mapping()
        
        # Initialize reset state IDs for distributed inference
        self._init_reset_state_ids()
        
        logger.info(f"TestEnv initialized: rank={rank}, num_envs={self.num_envs}, "
                   f"total_states={self.total_num_group_envs}, use_real_robot={self.use_real_robot}")

    def _detect_real_robot_mode(self, cfg) -> bool:
        """检测是否使用真实机器人模式。
        
        真实机器人模式的特征：
        1. 配置中有 robot_config_path 且对应的 JSON 配置 name != "gym_testenv"
        2. 或者 cfg.env.name 不是 "gym_testenv"
        
        Returns:
            True 如果是真实机器人模式，False 如果是数据回放模式
        """
        try:
            # 检查是否有 robot_config_path
            robot_config_path = getattr(cfg, 'robot_config_path', None)
            force_print(f"[TestEnv] _detect_real_robot_mode: robot_config_path={robot_config_path}")
            
            if robot_config_path is not None:
                # 检查文件是否存在
                if not os.path.exists(robot_config_path):
                    force_print(f"[TestEnv] WARNING: robot_config_path does not exist: {robot_config_path}")
                    force_print(f"[TestEnv] This may cause issues if running on a different node!")
                    # 文件不存在时，尝试从 cfg.env.name 判断
                else:
                    # 尝试加载配置文件检查 name
                    try:
                        import json
                        with open(robot_config_path, 'r') as f:
                            robot_cfg = json.load(f)
                        env_name = robot_cfg.get('name', '')
                        force_print(f"[TestEnv] Loaded robot config from {robot_config_path}, name={env_name}")
                        if env_name != "gym_testenv":
                            force_print(f"[TestEnv] Detected REAL ROBOT mode (config name={env_name})")
                            return True
                        else:
                            force_print(f"[TestEnv] Config is gym_testenv, using DATA REPLAY mode")
                            return False
                    except Exception as e:
                        force_print(f"[TestEnv] Failed to load robot config: {e}")
                        import traceback
                        force_print(f"[TestEnv] Traceback: {traceback.format_exc()}")
            
            # 检查 cfg.env.name
            if hasattr(cfg, 'env') and hasattr(cfg.env, 'name'):
                env_name = cfg.env.name
                force_print(f"[TestEnv] cfg.env.name={env_name}")
                if env_name != "gym_testenv":
                    force_print(f"[TestEnv] Detected REAL ROBOT mode (cfg.env.name={env_name})")
                    return True
            
            force_print(f"[TestEnv] Detected DATA REPLAY mode (gym_testenv)")
            return False
        except Exception as e:
            import traceback
            force_print(f"[TestEnv] Error detecting robot mode: {e}")
            force_print(f"[TestEnv] Traceback: {traceback.format_exc()}")
            force_print(f"[TestEnv] Defaulting to data replay mode")
            return False

    def _build_state_mapping(self):
        """Build mapping from state IDs to (episode_id, frame_id) pairs.
        
        For robot_env mode with gym_testenv, we build mapping based on available episodes.
        Each episode starts from frame 0.
        
        For real robot mode, we use a single state (state_id=0) since there's no dataset.
        """
        force_print(f"[TestEnv] _build_state_mapping: use_real_robot={self.use_real_robot}")
        
        # 真实机器人模式：不依赖 hdf5_loader，使用单一状态
        if self.use_real_robot:
            force_print(f"[TestEnv] Real robot mode: using single state (no hdf5_loader needed)")
            self.total_num_group_envs = 1
            self.trial_id_bins = [1]
            self.cumsum_trial_id_bins = np.array([1])
            return
        
        # 数据回放模式：尝试从 robot_backends 获取 episode 数量
        try:
            force_print(f"[TestEnv] _build_state_mapping: robot_backends={self.robot_backends}")
            force_print(f"[TestEnv] _build_state_mapping: robot_backends[0]={self.robot_backends[0]}")
            
            if hasattr(self.robot_backends[0], 'env'):
                force_print(f"[TestEnv] robot_backends[0].env = {self.robot_backends[0].env}")
                if hasattr(self.robot_backends[0].env, 'hdf5_loader') and self.robot_backends[0].env.hdf5_loader is not None:
                    force_print(f"[TestEnv] robot_backends[0].env.hdf5_loader = {self.robot_backends[0].env.hdf5_loader}")
                    num_episodes = self.robot_backends[0].env.hdf5_loader.num_episodes
                    logger.info(f"[TestEnv] Found {num_episodes} episodes in dataset")
                    force_print(f"[TestEnv] Found {num_episodes} episodes in dataset")
                    self.total_num_group_envs = num_episodes
                    self.trial_id_bins = [1] * num_episodes  # 每个episode一个state
                    self.cumsum_trial_id_bins = np.arange(1, num_episodes + 1)
                    return
                else:
                    force_print(f"[TestEnv] robot_backends[0].env has no hdf5_loader attribute or it's None")
            else:
                force_print(f"[TestEnv] robot_backends[0] has no env attribute")
            
            # 如果无法获取episode数量，使用默认值
            logger.warning("[TestEnv] Cannot get episode count from robot_backends, using default (1 state)")
            force_print("[TestEnv] Cannot get episode count from robot_backends, using default (1 state)")
            self.total_num_group_envs = 1
            self.trial_id_bins = [1]
            self.cumsum_trial_id_bins = np.array([1])
        except Exception as e:
            import traceback
            logger.warning(f"[TestEnv] Error getting episode count: {e}, using default")
            force_print(f"[TestEnv] Error getting episode count: {e}")
            force_print(f"[TestEnv] Traceback: {traceback.format_exc()}")
            self.total_num_group_envs = 1
            self.trial_id_bins = [1]
            self.cumsum_trial_id_bins = np.array([1])

    @property
    def is_start(self):
        """Whether the environment has just started."""
        return self._is_start

    @is_start.setter
    def is_start(self, value):
        self._is_start = value

    @property
    def elapsed_steps(self):
        return self._elapsed_steps

    def get_all_state_ids(self) -> np.ndarray:
        """Returns all possible state IDs."""
        return np.arange(self.total_num_group_envs)

    def _init_metrics(self):
        self.success_once = np.zeros(self.num_envs, dtype=bool)
        self.fail_once = np.zeros(self.num_envs, dtype=bool)
        self.returns = np.zeros(self.num_envs)
        # Intervention related metrics
        self.intervened_once = np.zeros(self.num_envs, dtype=bool)
        self.intervened_steps = np.zeros(self.num_envs, dtype=int)

    def _reset_metrics(self, env_idx=None):
        if env_idx is not None:
            mask = np.zeros(self.num_envs, dtype=bool)
            mask[env_idx] = True
            self.prev_step_reward[mask] = 0.0
            self.success_once[mask] = False
            self.fail_once[mask] = False
            self.returns[mask] = 0
            self._elapsed_steps[env_idx] = 0
            self.intervened_once[mask] = False
            self.intervened_steps[mask] = 0
        else:
            self.prev_step_reward[:] = 0
            self.success_once[:] = False
            self.fail_once[:] = False
            self.returns[:] = 0.0
            self._elapsed_steps[:] = 0
            self.intervened_once[:] = False
            self.intervened_steps[:] = 0

    def _record_metrics(self, step_reward, terminations, infos):
        episode_info = {}
        self.returns += step_reward
        self.success_once = self.success_once | terminations
        
        # Handle intervention metrics if present
        if "intervene_action" in infos:
            for env_id in range(self.num_envs):
                if infos["intervene_action"][env_id] is not None:
                    self.intervened_once[env_id] = True
                    self.intervened_steps[env_id] += 1
        
        episode_info["success_once"] = self.success_once.copy()
        episode_info["return"] = self.returns.copy()
        episode_info["episode_len"] = self.elapsed_steps.copy()
        episode_info["reward"] = np.where(
            episode_info["episode_len"] > 0,
            episode_info["return"] / episode_info["episode_len"],
            0.0
        )
        episode_info["intervened_once"] = self.intervened_once.copy()
        episode_info["intervened_steps"] = self.intervened_steps.copy()
        infos["episode"] = to_tensor(episode_info)
        return infos

    def _get_state_from_id(self, state_id: int):
        """Convert state_id to (episode_idx, frame_ptr) tuple.
        
        For robot_env mode, this is not used since we can't reset to
        arbitrary states. Kept for compatibility.
        """
        # robot_env模式不支持state_id重置
        return 0, 0

    def _wrap_obs(self, obs_list: List[Dict]) -> Dict:
        """Wrap observations into the expected format.
        
        RobotEnv返回的观测格式: {"images": {...}, "state": ndarray, "task_description": str}
        
        注意：只返回模型需要的三张图像以减少数据传输量
        
        Returns:
            Dict with 'images_and_states' containing observations:
            {
                "images_and_states": {
                    "head_image": tensor,        # 头部相机
                    "left_wrist_image": tensor,  # 左腕相机
                    "right_wrist_image": tensor, # 右腕相机
                    "state": tensor,
                },
                "task_descriptions": List[str],
            }
        """
        images_list = []
        states_list = []
        
        for obs in obs_list:
            # robot_env返回的观测已经是正确格式
            images_list.append(obs["images"])
            states_list.append({"state": obs["state"]})
            
            # 提取任务描述
            task_desc = obs.get("task_description", "")
            if task_desc:
                idx = len(images_list) - 1
                if idx < len(self.task_descriptions):
                    self.task_descriptions[idx] = task_desc
        
        # 合并所有图像和状态数据到 images_and_states
        images_and_states = {}
        
        # 模型只需要三张图像: head_image, left_wrist_image, right_wrist_image
        # 不再传输 full_image 和 wrist_image 以减少数据传输量
        REQUIRED_IMAGE_KEYS = {"head_image", "left_wrist_image", "right_wrist_image"}
        
        # 添加图像数据
        # 注意：图像是 JPEG 编码的 tensor，不同图像长度可能不同，需要特殊处理
        images_dict_of_list = list_of_dict_to_dict_of_list(images_list)
        
        for img_key, img_tensor_list in images_dict_of_list.items():
            # 只处理需要的图像键
            if img_key not in REQUIRED_IMAGE_KEYS:
                continue
                
            # 处理 JPEG 编码的图像 tensor 列表
            # 每个 tensor 形状为 [1, encoded_len]，长度可能不同
            if len(img_tensor_list) > 0:
                if isinstance(img_tensor_list[0], torch.Tensor):
                    # 找到最大长度并填充
                    max_len = max(t.shape[-1] for t in img_tensor_list)
                    padded_tensors = []
                    for t in img_tensor_list:
                        if t.shape[-1] < max_len:
                            # 填充到最大长度
                            padding = torch.zeros((*t.shape[:-1], max_len - t.shape[-1]), dtype=t.dtype)
                            t = torch.cat([t, padding], dim=-1)
                        padded_tensors.append(t)
                    # 合并: [num_envs, max_len]
                    img_tensor = torch.cat(padded_tensors, dim=0)
                else:
                    # 非 tensor 类型，使用默认处理
                    img_tensor = to_tensor(img_tensor_list)
            else:
                img_tensor = torch.tensor([])
            
            # 保存图像（只保留原始键名，不再做映射）
            images_and_states[img_key] = img_tensor
        
        # 添加状态数据
        states_dict = to_tensor(list_of_dict_to_dict_of_list(states_list))
        images_and_states["state"] = states_dict["state"]
        
        obs = {
            "images_and_states": images_and_states,
            "task_descriptions": self.task_descriptions,
        }
        return obs

    def reset(
        self,
        env_idx: Optional[int | List[int] | np.ndarray] = None,
        reset_state_ids=None,
        options: Optional[dict] = None,
    ):
        """Reset environment(s) to initial state.
        
        For real robot mode:
        - Simply reset the robot to its initial pose
        - reset_state_ids is ignored (no episode concept)
        
        For gym_testenv mode (data replay):
        - If current_episode_id is -1 (first time or episode completed), select a new episode
        - Otherwise, reset to frame 0 of the current episode
        - reset_state_ids can be used to specify episode indices (will be mapped to episode range)
        """
        if env_idx is None:
            env_idx = np.arange(self.num_envs)
        elif isinstance(env_idx, int):
            env_idx = [env_idx]
        
        env_idx = np.asarray(env_idx)
        
        # 重置机器人环境
        obs_list = []
        
        if self.use_real_robot:
            # 真实机器人模式：直接调用 reset，不传递 episode_id
            force_print(f"[TestEnv] Real robot mode: resetting {len(env_idx)} envs")
            for i, eid in enumerate(env_idx):
                force_print(f"[TestEnv] Resetting real robot env {eid}...")
                try:
                    # 真实机器人的 reset 不需要 episode_id 参数
                    single_obs = self.robot_backends[eid].reset(episode_id=None)
                    obs_list.append(single_obs)
                    force_print(f"[TestEnv] Real robot env {eid} reset complete")
                except Exception as e:
                    import traceback
                    force_print(f"[TestEnv] ERROR: Failed to reset real robot env {eid}: {e}")
                    force_print(f"[TestEnv] Traceback: {traceback.format_exc()}")
                    raise
        else:
            # 数据回放模式：使用 episode_id 来选择数据
            # 获取实际的episode数量（用于映射state_id）
            num_episodes = self.total_num_group_envs
            if hasattr(self.robot_backends[0], 'env') and hasattr(self.robot_backends[0].env, 'hdf5_loader'):
                if self.robot_backends[0].env.hdf5_loader is not None:
                    num_episodes = self.robot_backends[0].env.hdf5_loader.num_episodes
            
            for i, eid in enumerate(env_idx):
                # 确定要使用的episode索引
                episode_id = None
                
                # 如果提供了reset_state_ids，使用它作为episode索引
                if reset_state_ids is not None:
                    if isinstance(reset_state_ids, (list, np.ndarray, torch.Tensor)):
                        if isinstance(reset_state_ids, torch.Tensor):
                            reset_state_ids = reset_state_ids.cpu().numpy()
                        raw_state_id = int(reset_state_ids[i])
                    else:
                        raw_state_id = int(reset_state_ids)
                    
                    # 将state_id映射到episode范围 [0, num_episodes)
                    # 使用取模操作确保在有效范围内
                    episode_id = raw_state_id % num_episodes
                    logger.info(f"[TestEnv] Reset env {eid}: state_id={raw_state_id} -> episode={episode_id} (total_episodes={num_episodes})")
                # 如果当前episode_id为-1，需要选择新episode
                elif self.current_episode_ids[eid] == -1:
                    # 随机选择新episode（None会让HDF5DataLoader随机选择）
                    episode_id = None
                    logger.info(f"[TestEnv] Reset env {eid} to new random episode")
                else:
                    # 使用当前episode，重置到第0帧
                    episode_id = int(self.current_episode_ids[eid])
                    logger.info(f"[TestEnv] Reset env {eid} to frame 0 of current episode {episode_id}")
                
                # 调用robot_backend的reset，传递episode_id
                single_obs = self.robot_backends[eid].reset(episode_id=episode_id)
                obs_list.append(single_obs)
                
                # 更新当前episode_id（如果是随机选择，需要从backend获取）
                if hasattr(self.robot_backends[eid], 'env') and hasattr(self.robot_backends[eid].env, 'hdf5_loader'):
                    if self.robot_backends[eid].env.hdf5_loader is not None:
                        self.current_episode_ids[eid] = self.robot_backends[eid].env.hdf5_loader.episode_index
                        logger.info(f"[TestEnv] Env {eid} now at episode {self.current_episode_ids[eid]}")
        
        obs = self._wrap_obs(obs_list)
        
        if env_idx is not None:
            self._reset_metrics(env_idx)
        else:
            self._reset_metrics()
        
        infos = {}
        return obs, infos

    def _handle_auto_reset(self, dones, _final_obs, infos):
        """Handle auto reset when episodes are done.
        
        For real robot mode:
        - Simply reset the robot to initial pose
        
        For data replay mode:
        - 标记当前episode为完成状态（设置为-1）
        - 下次reset时会随机选择新的episode
        
        Args:
            dones: Boolean array indicating which environments are done
            _final_obs: Final observations before reset
            infos: Info dict
            
        Returns:
            obs: New observations after reset
            infos: Updated info dict with final_observation and final_info
        """
        final_obs = copy.deepcopy(_final_obs)
        env_idx = np.arange(0, self.num_envs)[dones]
        final_info = copy.deepcopy(infos)
        
        if self.use_real_robot:
            # 真实机器人模式：直接重置
            force_print(f"[TestEnv] Real robot mode: auto-resetting {len(env_idx)} envs")
        else:
            # 数据回放模式：标记完成的环境需要选择新episode
            for eid in env_idx:
                logger.info(f"[TestEnv] Episode completed for env {eid}, marking for new episode selection")
                self.current_episode_ids[eid] = -1
        
        # Reset时会根据模式自动处理
        obs, infos = self.reset(
            env_idx=env_idx,
            reset_state_ids=None,  # 让系统自动选择新episode（数据回放模式）或直接重置（真实机器人模式）
        )
        
        # gymnasium calls it final observation but it really is just o_{t+1} or the true next observation
        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        infos["_final_info"] = dones
        infos["_final_observation"] = dones
        infos["_elapsed_steps"] = dones
        return obs, infos

    def _init_reset_state_ids(self):
        """Initialize reset state IDs generator."""
        self.update_reset_state_ids()

    def update_reset_state_ids(self):
        """Update reset state IDs for grouped environments."""
        reset_state_ids = torch.randint(
            low=0,
            high=self.total_num_group_envs,
            size=(self.num_group,),
            generator=self._torch_generator,
        )
        self.reset_state_ids = reset_state_ids.repeat_interleave(repeats=self.group_size)

    def step(self, actions=None, auto_reset=True):
        """Execute one step in the environment.
        
        Args:
            actions: Action tensor [num_envs, action_dim], None for reset
            auto_reset: Whether to auto reset when done (default True)
            
        Returns:
            obs: Observations dict
            rewards: Reward tensor
            terminations: Termination flags
            truncations: Truncation flags
            infos: Info dict with 'episode' metrics
        """
        logger.debug(f"[TestEnv] step called, actions shape: {actions.shape if actions is not None else None}")
        
        if actions is None:
            # Reset all environments
            obs, infos = self.reset()
            terminations = np.zeros(self.num_envs, dtype=bool)
            truncations = np.zeros(self.num_envs, dtype=bool)
            return obs, None, to_tensor(terminations), to_tensor(truncations), infos
        
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()
        
        self._elapsed_steps += 1
        
        obs_list = []
        terminations = np.zeros(self.num_envs, dtype=bool)
        truncations = np.zeros(self.num_envs, dtype=bool)
        raw_obs_list = []
        step_rewards = np.zeros(self.num_envs)
        intervene_actions = []
        intervene_flags = []
        
        # 执行机器人环境步骤
        for eid in range(self.num_envs):
            single_action = actions[eid] if actions.ndim > 1 else actions
            single_obs, reward, terminated, truncated, info = self.robot_backends[eid].step(single_action)
            
            obs_list.append(single_obs)
            terminations[eid] = terminated
            truncations[eid] = truncated
            step_rewards[eid] = reward
            
            # 记录介入信息
            intervene_actions.append(info.get('intervene_action', None))
            intervene_flags.append(info.get('intervene_flag', False))
            
            # raw_obs用于视频录制（如果需要）
            raw_obs_list.append({
                'images': single_obs['images'],
                'state': single_obs['state']
            })
        
        # Check truncation
        max_steps = getattr(self.cfg, 'max_episode_steps', 1000)
        truncations = self.elapsed_steps >= max_steps
        
        obs = self._wrap_obs(obs_list)
        
        # 使用机器人环境返回的奖励
        step_reward = step_rewards
        
        # Video recording
        if self.video_cfg is not None and getattr(self.video_cfg, 'save_video', False):
            plot_infos = {
                "rewards": step_reward,
                "terminations": terminations,
                "steps": self._elapsed_steps,
            }
            self.add_new_frames(raw_obs_list, plot_infos)
        
        infos = {}
        infos = self._record_metrics(step_reward, terminations, infos)
        
        # Handle ignore_terminations
        if self.ignore_terminations:
            infos["episode"]["success_at_end"] = to_tensor(terminations)
            terminations[:] = False
        
        # 添加介入信息
        if intervene_actions:
            intervene_action_array = np.array([ia if ia is not None else np.zeros_like(actions[0]) 
                                               for ia in intervene_actions])
            intervene_flag_array = np.array(intervene_flags, dtype=bool)
        else:
            # 如果没有介入数据，使用placeholder
            intervene_action_array = np.zeros_like(actions)
            intervene_flag_array = np.zeros((self.num_envs,), dtype=bool)
        
        infos["intervene_action"] = to_tensor(intervene_action_array)
        infos["intervene_flag"] = to_tensor(intervene_flag_array)
        
        # Handle auto reset
        dones = terminations | truncations
        _auto_reset = auto_reset and self.auto_reset
        if dones.any() and _auto_reset:
            obs, infos = self._handle_auto_reset(dones, obs, infos)
        
        return (
            obs,
            to_tensor(step_reward),
            to_tensor(terminations),
            to_tensor(truncations),
            infos,
        )

    def chunk_step(self, chunk_actions):
        """Execute a chunk of actions.
        
        Args:
            chunk_actions: [num_envs, chunk_size, action_dim]
            
        Returns:
            obs: Final observations
            chunk_rewards: [num_envs, chunk_steps]
            chunk_terminations: [num_envs, chunk_steps]
            chunk_truncations: [num_envs, chunk_steps]
            infos: Info dict
        """
        chunk_size = chunk_actions.shape[1]
        
        chunk_rewards = []
        raw_chunk_terminations = []
        raw_chunk_truncations = []
        raw_chunk_intervene_actions = []
        raw_chunk_intervene_flag = []
        
        for i in range(chunk_size):
            actions = chunk_actions[:, i]
            extracted_obs, step_reward, terminations, truncations, infos = self.step(
                actions, auto_reset=False
            )
            
            if "intervene_action" in infos:
                raw_chunk_intervene_actions.append(infos["intervene_action"])
                raw_chunk_intervene_flag.append(infos["intervene_flag"])
            
            chunk_rewards.append(step_reward)
            raw_chunk_terminations.append(terminations)
            raw_chunk_truncations.append(truncations)
        
        chunk_rewards = torch.stack(chunk_rewards, dim=1)  # [num_envs, chunk_steps]
        raw_chunk_terminations = torch.stack(raw_chunk_terminations, dim=1)  # [num_envs, chunk_steps]
        raw_chunk_truncations = torch.stack(raw_chunk_truncations, dim=1)  # [num_envs, chunk_steps]
        
        past_terminations = raw_chunk_terminations.any(dim=1)
        past_truncations = raw_chunk_truncations.any(dim=1)
        past_dones = torch.logical_or(past_terminations, past_truncations)
        
        # Stack intervention info
        if raw_chunk_intervene_actions:
            infos["intervene_action"] = torch.stack(
                raw_chunk_intervene_actions, dim=1
            ).reshape(self.num_envs, -1)
            infos["intervene_flag"] = torch.stack(raw_chunk_intervene_flag, dim=1)
        
        # Handle auto reset
        if past_dones.any() and self.auto_reset:
            extracted_obs, infos = self._handle_auto_reset(
                past_dones.cpu().numpy(), extracted_obs, infos
            )
        
        # Handle termination logic based on auto_reset and ignore_terminations
        if self.auto_reset or self.ignore_terminations:
            chunk_terminations = torch.zeros_like(raw_chunk_terminations)
            chunk_terminations[:, -1] = past_terminations
            
            chunk_truncations = torch.zeros_like(raw_chunk_truncations)
            chunk_truncations[:, -1] = past_truncations
        else:
            chunk_terminations = raw_chunk_terminations.clone()
            chunk_truncations = raw_chunk_truncations.clone()
        
        return (
            extracted_obs,
            chunk_rewards,
            chunk_terminations,
            chunk_truncations,
            infos,
        )

    def add_new_frames(self, raw_obs_list, plot_infos):
        """Add frames for video recording."""
        images = []
        for env_id, raw_single_obs in enumerate(raw_obs_list):
            info_item = {k: v if np.size(v) == 1 else v[env_id] for k, v in plot_infos.items()}
            # Get first camera image
            img = None
            for cam_name, cam_img in raw_single_obs['images'].items():
                img = cam_img
                break
            if img is not None:
                img = put_info_on_image(img, info_item)
                images.append(img)
        
        if images:
            full_image = tile_images(images, nrows=int(np.sqrt(self.num_envs)))
            self.render_images.append(full_image)

    def flush_video(self, video_sub_dir: Optional[str] = None):
        """Save recorded video."""
        if not self.render_images:
            return
            
        if self.video_cfg is None:
            return
            
        video_base_dir = getattr(self.video_cfg, 'video_base_dir', './videos')
        output_dir = os.path.join(video_base_dir, f"rank_{self.rank}")
        if video_sub_dir is not None:
            output_dir = os.path.join(output_dir, f"{video_sub_dir}")
        
        save_rollout_video(
            self.render_images,
            output_dir=output_dir,
            video_name=f"{self.video_cnt}",
        )
        self.video_cnt += 1
        self.render_images = []

    def reset_envs_to_state_ids(self, state_ids_list, task_ids_list):
        """Reset environments to specified state IDs.
        
        For real robot mode:
        - state_ids_list is ignored (no episode concept)
        - Simply reset the robot to initial pose
        
        For data replay mode:
        - state_ids_list specifies which episodes to load
        
        Args:
            state_ids_list: List of state IDs to reset environments to
            task_ids_list: List of task IDs (unused in test env)
            
        Returns:
            List of tuples where each tuple is (dict, dict):
                - First dict contains "images_and_states" with tensors
                - Second dict contains "task_descriptions" for that environment
        """
        force_print(f"[TestEnv] reset_envs_to_state_ids ENTER, num_envs={len(state_ids_list)}, use_real_robot={self.use_real_robot}")
        force_print(f"[TestEnv] state_ids_list: {state_ids_list}")
        force_print(f"[TestEnv] task_ids_list: {task_ids_list}")
        
        try:
            env_idx = np.arange(len(state_ids_list))
            
            if self.use_real_robot:
                # 真实机器人模式：忽略 state_ids，直接重置
                force_print(f"[TestEnv] Real robot mode: ignoring state_ids, resetting to initial pose")
                obs, infos = self.reset(env_idx=env_idx, reset_state_ids=None)
            else:
                # 数据回放模式：使用 state_ids 选择 episode
                obs, infos = self.reset(env_idx=env_idx, reset_state_ids=state_ids_list)
            
            force_print(f"[TestEnv] Reset complete, obs keys: {obs.keys()}")
            force_print(f"[TestEnv] Task descriptions: {self.task_descriptions}")
            
            # obs 已经是正确格式: {"images_and_states": {...}, "task_descriptions": list}
            # env_worker 期望: [dict, dict] 其中:
            #   - 第一个dict同时包含 "images_and_states" 和 "task_descriptions" 键
            #   - 第二个dict也包含 "task_descriptions" 键（用于兼容性）
            
            num_envs = len(state_ids_list)
            
            # 直接使用 obs 中的数据
            images_and_states = obs["images_and_states"]
            task_descriptions = obs["task_descriptions"]
            
            # 打印图像 tensor 的形状信息
            force_print(f"[TestEnv] images_and_states keys: {images_and_states.keys()}")
            for k, v in images_and_states.items():
                if isinstance(v, torch.Tensor):
                    force_print(f"[TestEnv]   {k}: shape={v.shape}, dtype={v.dtype}")
                else:
                    force_print(f"[TestEnv]   {k}: type={type(v)}")
            
            # 返回格式: [dict, dict]
            # 第一个dict必须同时包含 "images_and_states" 和 "task_descriptions"
            result = [
                {
                    "images_and_states": images_and_states,
                    "task_descriptions": task_descriptions
                },
                {"task_descriptions": task_descriptions}
            ]
            
            force_print(f"[TestEnv] reset_envs_to_state_ids SUCCESS, returning result")
            
        except Exception as e:
            import traceback
            force_print(f"[TestEnv] reset_envs_to_state_ids FAILED: {e}")
            force_print(f"[TestEnv] Traceback: {traceback.format_exc()}")
            raise
        
        logger.info(f"[TestEnv] Returning result for {num_envs} envs")
        logger.info(f"[TestEnv] images_and_states keys: {images_and_states.keys()}")
        logger.info(f"[TestEnv] task_descriptions count: {len(task_descriptions)}")
        
        return result

    def close(self):
        """Close environment and release resources."""
        self.flush_video()
        
        # 关闭机器人后端
        for backend in self.robot_backends:
            backend.close()

    def get_state(self) -> bytes:
        """Serialize environment state for pausing."""
        return b""

    def load_state(self, state_buffer: bytes):
        """Restore from serialized state."""
        pass
