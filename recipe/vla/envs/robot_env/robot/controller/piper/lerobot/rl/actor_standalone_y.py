#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Standalone Actor: run policy locally without Learner (no gRPC, no parameter updates).
- Loads pretrained policy from cfg.policy.pretrained_path
- Steps the robot env with policy actions
- Logs episodic rewards & simple FPS stats
"""
import pdb
import os
import time
import logging
import torch
from torch import nn

from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.policies.factory import make_policy
from lerobot.policies.sac.modeling_sac import SACPolicy

from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import init_logging, get_safe_torch_device, TimerManager
from lerobot.processor import TransitionKey
from .gym_manipulator import (
    make_robot_env,
    make_processors,
    create_transition,
    step_env_and_process_transition,
)

@parser.wrap()
def actor_standalone_cli(cfg: TrainRLServerPipelineConfig):
    cfg.validate()

    # 日志
    log_dir = os.path.join(cfg.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"actor_standalone_{cfg.job_name}.log")
    init_logging(log_file=log_file, display_pid=False)
    logging.info("[ACTOR-SA] Standalone mode: NO learner, NO param updates.")

    # 必须指定预训练权重目录（含 config.json + model.safetensors）
    pretrained = getattr(cfg.policy, "pretrained_path", None)
    if not pretrained:
        raise ValueError(
            "Standalone 运行需要 cfg.policy.pretrained_path 指向你的预训练目录，如 "
            ".../checkpoints/050000/pretrained_model"
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

    # 加载策略（from_pretrained）
    logging.info("[ACTOR-SA] Loading policy from pretrained")
    policy: SACPolicy = make_policy(cfg=cfg.policy, env_cfg=cfg.env)
    policy = policy.eval()
    assert isinstance(policy, nn.Module)

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

    # 用 cfg.policy.online_steps 作为最大交互步数（也可自行改成 max_episodes）
    for interaction_step in range(cfg.policy.online_steps):
        t0 = time.perf_counter()

        # 取出策略需要的观测
        observation = {
            k: v for k, v in transition[TransitionKey.OBSERVATION].items()
            if k in cfg.policy.input_features
        }
        # pdb.set_trace()
        # 前向推理
        with timer:
            action = policy.select_action(batch=observation)

        # 环境一步 + 处理
        new_transition = step_env_and_process_transition(
            env=env,
            transition=transition,
            action=action,
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

if __name__ == "__main__":
    actor_standalone_cli()
