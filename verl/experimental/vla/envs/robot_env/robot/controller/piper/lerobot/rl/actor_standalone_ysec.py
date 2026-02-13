#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Standalone Actor with PI0 Policy: run policy locally without Learner.
- Loads pretrained PI0 policy from cfg.policy.pretrained_path
- Steps the robot env with policy actions
- Logs episodic rewards & simple FPS stats
"""
import os
import time
import logging
import math
import torch
from torch import nn
import numpy as np

from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.model.kinematics import RobotKinematics
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import init_logging, get_safe_torch_device, TimerManager
from lerobot.processor import TransitionKey
from .gym_manipulator_bipiper import (
    make_robot_env,
    make_processors,
    create_transition,
    step_env_and_process_transition,
)
from lerobot.processor.factory import make_default_robot_observation_processor

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

@parser.wrap()
def actor_standalone_cli(cfg: TrainRLServerPipelineConfig):
    cfg.validate()

    # 日志
    log_dir = os.path.join(cfg.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"actor_standalone_{cfg.job_name}.log")
    init_logging(log_file=log_file, display_pid=False)
    logging.info("[ACTOR-SA] Standalone mode with PI0 policy: NO learner, NO param updates.")

    # 必须指定预训练权重目录
    pretrained = getattr(cfg.policy, "pretrained_path", None)
    if not pretrained:
        raise ValueError(
            "Standalone 运行需要 cfg.policy.pretrained_path 指向你的预训练目录"
        )
    logging.info(f"[ACTOR-SA] Using pretrained_path={pretrained}")

    # 设备/加速
    set_seed(cfg.seed)
    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # 环境与预处理器
    logging.info("[ACTOR-SA] Creating env and processors")
    env, teleop_device = make_robot_env(cfg=cfg.env)
    env_proc, act_proc = make_processors(env, teleop_device, cfg.env, cfg.policy.device)

    # ============ PI0策略加载 ============
    logging.info("[ACTOR-SA] Loading PI0 policy from pretrained")
    
    policy_class = get_policy_class(cfg.policy.type)
    
    from lerobot.configs.policies import PreTrainedConfig
    config = PreTrainedConfig.from_pretrained(cfg.policy.pretrained_path)
    
    policy = policy_class.from_pretrained(cfg.policy.pretrained_path, config=config)
    policy = policy.to(device)
    policy = policy.eval()
    
    logging.info(f"[ACTOR-SA] Loaded {cfg.policy.type} policy successfully")

    # ============ 加载预处理器和后处理器 ============
    logging.info(f"[ACTOR-SA] Loading preprocessor/postprocessor from {cfg.policy.pretrained_path}")
    
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        dataset_stats=None,
        preprocessor_overrides={
            "device_processor": {"device": cfg.policy.device},
        },
    )
    robot_observation_processor = make_default_robot_observation_processor()
    
    logging.info("[ACTOR-SA] Preprocessor/postprocessor loaded successfully")

    # ============ 准备数据集特征映射 ============
    robot_obs_features = env.robot.observation_features
    dataset_features = hw_to_dataset_features(robot_obs_features, "observation")
    
    logging.info(f"[ACTOR-SA] Dataset features: {list(dataset_features.keys())}")

    # ============ 相机键名映射 ============
    camera_mapping = {
        "observation.images.front": "observation.images.cam_high",
        "observation.images.left": "observation.images.cam_left_wrist", 
        "observation.images.right": "observation.images.cam_right_wrist"
    }

    # 重置环境
    obs, info = env.reset()
    env_proc.reset()
    act_proc.reset()

    transition = create_transition(observation=obs, info=info)
    transition = env_proc(transition)

    # 统计
    episode_idx = 0
    ep_reward = 0.0
    ep_steps = 0
    timer = TimerManager("Policy inference", log=False)

    # 任务描述
    task_description = getattr(cfg.env, "task", "push bowl")
    
    # logging.info(f"[ACTOR-SA] Starting execution loop for {cfg.policy.online_steps} steps")

    #机器人运动学 yag
    left_kin = RobotKinematics(
    urdf_path="local_assets/piper.urdf",
    target_frame_name="joint6",
    joint_names=[f"joint{i+1}" for i in range(6)],
    )
    right_kin = RobotKinematics(
    urdf_path="local_assets/piper.urdf",
    target_frame_name="joint6",
    joint_names=[f"joint{i+1}" for i in range(6)],
    )
    left_joint_idx = [0, 1, 2, 3, 4, 5]
    left_grip_idx = 6
    right_joint_idx = [7, 8, 9, 10, 11, 12]
    right_grip_idx = 13

    # 主循环
    for interaction_step in range(10000):
        t0 = time.perf_counter()

        # ============ 直接从机器人获取观测============
        raw_obs = env.robot.get_observation(mode = "Present_Position")
        #TODO: 这里的get_observation的方法他控制获取endpose或者是joint，是通过self.control_mode来区分,是不是给这个函数传一个control_mode参数去控制更好一点，这样更灵活
        #目前这里获得的是endpose，需要获取joint

        obs_processed = robot_observation_processor(raw_obs)
        
        # 构建数据集格式的观测
        obs_with_policy_features = build_dataset_frame(
            dataset_features, obs_processed, prefix="observation"
        )
        
        # 转换为tensor并预处理
        for name in obs_with_policy_features:
            obs_with_policy_features[name] = torch.from_numpy(obs_with_policy_features[name])
            
            # 图像需要归一化和调整维度
            if "image" in name:
                obs_with_policy_features[name] = (
                    obs_with_policy_features[name].type(torch.float32) / 255
                )
                # HWC -> CHW
                obs_with_policy_features[name] = (
                    obs_with_policy_features[name].permute(2, 0, 1).contiguous()
                )
            
            # 添加batch维度并移到设备
            obs_with_policy_features[name] = obs_with_policy_features[name].unsqueeze(0)
            obs_with_policy_features[name] = obs_with_policy_features[name].to(device)
        
        # 应用相机名称映射
        for old_key, new_key in camera_mapping.items():
            if old_key in obs_with_policy_features:
                obs_with_policy_features[new_key] = obs_with_policy_features.pop(old_key)
        
        # 添加任务描述
        obs_with_policy_features["task"] = [task_description] if task_description else [""]
        
        # 添加机器人类型
        robot_name = getattr(env.robot, "name", "")
        if robot_name:
            obs_with_policy_features["robot_type"] = robot_name

        # ============ 预处理观测 ============
        preprocessed_obs = preprocessor(obs_with_policy_features)

        # ============ 策略推理（PI0方式）============
        with timer:
            # 使用predict_action_chunk
            action_chunk = policy.predict_action_chunk(
                preprocessed_obs,
                # inference_delay=0,
                # prev_chunk_left_over=None
            )
            
            # 后处理动作
            action = postprocessor(action_chunk)
            
            # 取第一个动作
            if action.dim() > 1:
                action = action.squeeze(0)  # [batch, chunk_size, action_dim]
                if action.dim() > 1:
                    action = action[0]  # 取第一个动作 [action_dim]
            else:
                action = action.squeeze(0)
            # 取出关节（rad）-> numpy
            ql_rad = action[left_joint_idx].detach().cpu().numpy().astype(float)
            qr_rad = action[right_joint_idx].detach().cpu().numpy().astype(float)

            # RobotKinematics.forward_kinematics 接口吃的是“度”
            ql_deg = np.rad2deg(ql_rad)
            qr_deg = np.rad2deg(qr_rad)

            # 做 FK
            T_l = left_kin.forward_kinematics(ql_deg)   # 4x4
            T_r = right_kin.forward_kinematics(qr_deg)  # 4x4

            # 位置：米
            lx, ly, lz = T_l[:3, 3].tolist()
            rx, ry, rz = T_r[:3, 3].tolist()

            # 姿态：把旋转矩阵 -> RPY（弧度）
            lrx, lry, lrz = rotmat_to_rpy_zyx(T_l[:3, :3])
            rrx, rry, rrz = rotmat_to_rpy_zyx(T_r[:3, :3])

            # 夹爪：这里直接用 action 里的值（你也可以改成从 obs 里拿）
            l_grip = float(action[left_grip_idx].item())
            r_grip = float(action[right_grip_idx].item())

            # 如果你需要的是“张量形式的 endpose 动作”，比如 14 维：
            #   [L_x, L_y, L_z, L_rx, L_ry, L_rz, L_grip,
            #    R_x, R_y, R_z, R_rx, R_ry, R_rz, R_grip]
            endpose_action = torch.tensor(
                [
                    lx, ly, lz, lrx, lry, lrz, l_grip,
                    rx, ry, rz, rrx, rry, rrz, r_grip,
                ],
                device=device,
            )

            #TODO:目前这里得到的action是joint，要加一个FK去转换成endpose


        # ============ 环境一步 + 处理 ============
        new_transition = step_env_and_process_transition(
            env=env,
            transition=transition,
            action=endpose_action,
            env_processor=env_proc,
            action_processor=act_proc,
        )

        reward = float(new_transition[TransitionKey.REWARD])
        done = new_transition.get(TransitionKey.DONE, False)
        truncated = new_transition.get(TransitionKey.TRUNCATED, False)

        ep_reward += reward
        ep_steps += 1
        transition = new_transition

        # 回合结束就重置并打印
        if done or truncated:
            fps_mean = timer.fps_avg if timer.count > 1 else 0.0
            logging.info(
                f"[ACTOR-SA] Episode {episode_idx}: reward={ep_reward:.3f}, "
                f"steps={ep_steps}, policy_fps≈{fps_mean:.1f}"
            )
            episode_idx += 1
            ep_reward = 0.0
            ep_steps = 0
            timer.reset()

            obs, info = env.reset()
            env_proc.reset()
            act_proc.reset()
            transition = create_transition(observation=obs, info=info)
            transition = env_proc(transition)

        # 控制节拍
        if cfg.env.fps is not None:
            dt = time.perf_counter() - t0
            from lerobot.utils.robot_utils import busy_wait
            busy_wait(1 / cfg.env.fps - dt)

    logging.info("[ACTOR-SA] Execution completed")

if __name__ == "__main__":
    actor_standalone_cli()