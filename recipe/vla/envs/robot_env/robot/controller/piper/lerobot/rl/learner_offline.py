#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Offline-only Learner for LeRobot SAC (no actor, no gRPC)
# Copyright 2025

import logging
import os
import shutil
import time
from pathlib import Path
from pprint import pformat
import pdb
import torch
from torch import nn
from torch.optim import Optimizer
from termcolor import colored

from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.transforms import JointToEEDeltaConfig, JointToEEDeltaDataset
from lerobot.policies.factory import make_policy
from lerobot.policies.sac.modeling_sac import SACPolicy
from lerobot.rl.buffer import ReplayBuffer
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.utils.constants import (
    ACTION,
    CHECKPOINTS_DIR,
    LAST_CHECKPOINT_LINK,
    PRETRAINED_MODEL_DIR,
    TRAINING_STATE_DIR,
)
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    init_logging,
)

# -----------------------------
# CLI
# -----------------------------

def save_batch_state_images(state_dict: dict, base_dir: str = "img_output/learner_batch") -> None:
    """
    从 batch['state'] 可视化并保存各相机的图像：
    - 仅保存每个相机最近 5 张；
    - 支持 key: 'observation.images.<camera>' 或 'observation.image'；
    - 输入期望为 (B, C, H, W) 的 float32 张量，范围 [0,1]。
    """
    import os, glob, time
    import torch
    import cv2

    os.makedirs(base_dir, exist_ok=True)

    def _cam_name_from_key(k: str) -> str:
        if ".images." in k:
            return k.split(".images.", 1)[1]
        if k.endswith(".image"):
            return "image"
        return k.replace(".", "_")

    for key, val in state_dict.items():
        if "image" not in key:
            continue
        if not isinstance(val, torch.Tensor):
            continue

        t = val.detach()
        if t.device.type != "cpu":
            t = t.cpu()

        # 允许 (C,H,W)，统一成 (B,C,H,W)
        if t.ndim == 3:
            t = t.unsqueeze(0)
        if t.ndim != 4:
            continue  # 非图像跳过

        # 规范到 [0,1]
        t = t.clamp(0.0, 1.0)

        # 只取前 5 张
        b = min(5, t.shape[0])
        t = t[:b]  # (b, C, H, W)

        # to uint8, BCHW->BHWC
        t_u8 = torch.round(t * 255.0).to(torch.uint8)        # (b, C, H, W)
        t_hwc = t_u8.permute(0, 2, 3, 1).contiguous().numpy()  # (b, H, W, C)

        cam = _cam_name_from_key(key)
        save_dir = os.path.join(base_dir, cam)
        os.makedirs(save_dir, exist_ok=True)

        now_ms = int(time.time() * 1000)
        for i in range(b):
            # RGB -> BGR
            bgr = t_hwc[i][..., ::-1]
            fname = os.path.join(save_dir, f"{now_ms}_{i}.jpg")
            try:
                cv2.imwrite(fname, bgr)
            except Exception:
                pass

        # 只保留最新 5 张
        files = sorted(glob.glob(os.path.join(save_dir, "*.jpg")), key=os.path.getmtime)
        if len(files) > 5:
            for old in files[:-5]:
                try:
                    os.remove(old)
                except Exception:
                    pass


@parser.wrap()
def train_cli(cfg: TrainRLServerPipelineConfig):
    """
    Offline-only entrypoint: no actor/gRPC, train purely from an offline dataset.
    """
    train_offline(cfg)


def train_offline(cfg: TrainRLServerPipelineConfig):
    """
    Main function for 100% offline training.
    """
    # ---- basic setup
    cfg.validate()
    os.makedirs(os.path.join(cfg.output_dir, "logs"), exist_ok=True)
    log_file = os.path.join(cfg.output_dir, "logs", "learner_offline.log")
    init_logging(log_file=log_file, display_pid=False)
    logging.info("Offline learner logging initialized → %s", log_file)
    logging.info(pformat(cfg.to_dict()))

    # ---- wandb (optional)
    if cfg.wandb.enable and cfg.wandb.project:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        logging.info(colored("Logs will be saved locally (W&B disabled).", "yellow", attrs=["bold"]))

    # ---- resume guard (no actor, but we keep the same convention)
    cfg = _handle_resume_logic(cfg)

    # ---- seeds, backends
    set_seed(seed=cfg.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # ---- devices
    device = get_safe_torch_device(try_device=cfg.policy.device, log=True)
    storage_device = get_safe_torch_device(try_device=cfg.policy.storage_device)

    # ---- policy
    logging.info("Initializing SAC policy")
    policy: SACPolicy = make_policy(cfg=cfg.policy, env_cfg=cfg.env)
    assert isinstance(policy, nn.Module)
    policy.train()

    # ---- optimizers
    optimizers, lr_scheduler = _make_optimizers_and_scheduler(cfg=cfg, policy=policy)

    # ---- load training state (resume)
    optimization_step = _load_training_state(cfg=cfg, optimizers=optimizers) or 0

    # ---- buffers (offline only)
    offline_replay_buffer = _initialize_offline_replay_buffer(cfg=cfg, device=device, storage_device=storage_device)

    # ---- info
    _log_training_info(cfg=cfg, policy=policy, offline_buffer=offline_replay_buffer)

    # ---- iterator
    async_prefetch = cfg.policy.async_prefetch
    batch_size = cfg.batch_size  # 100% offline → 全部 batch 来自离线
    offline_iterator = offline_replay_buffer.get_iterator(
        batch_size=batch_size, async_prefetch=async_prefetch, queue_size=2
    )

    # ---- hyperparams
    clip_grad_norm_value = cfg.policy.grad_clip_norm
    utd_ratio = cfg.policy.utd_ratio
    log_freq = cfg.log_freq
    save_freq = cfg.save_freq
    policy_update_freq = cfg.policy.policy_update_freq
    online_steps = cfg.policy.online_steps  # 用作总优化步数上限

    # ---- training loop
    logging.info(colored("Start 100% OFFLINE training", "green", attrs=["bold"]))
    t_last_log = time.time()

    while optimization_step < online_steps:
        # ---------- (k-1) times: critic(-only) updates ----------
        for _ in range(max(0, utd_ratio - 1)):
            batch = next(offline_iterator)
            _check_nan_in_transition(batch)

            obs_feats, next_obs_feats = _get_observation_features(policy=policy,
                                                                  observations=batch["state"],
                                                                  next_observations=batch["next_state"])
            # save_batch_state_images(batch["state"], base_dir="img_output/learner_batch") # 可视化 batch 图像 debug
            forward_batch = {
                ACTION: batch[ACTION],
                "reward": batch["reward"],
                "state": batch["state"],
                "next_state": batch["next_state"],
                "done": batch["done"],
                "observation_feature": obs_feats,
                "next_observation_feature": next_obs_feats,
                "complementary_info": batch.get("complementary_info"),
            }

            # critic
            loss_critic = policy.forward(forward_batch, model="critic")["loss_critic"]
            optimizers["critic"].zero_grad()
            loss_critic.backward()
            torch.nn.utils.clip_grad_norm_(policy.critic_ensemble.parameters(), max_norm=clip_grad_norm_value)
            optimizers["critic"].step()

            # discrete critic (optional)
            if policy.config.num_discrete_actions is not None:
                loss_disc = policy.forward(forward_batch, model="discrete_critic")["loss_discrete_critic"]
                optimizers["discrete_critic"].zero_grad()
                loss_disc.backward()
                torch.nn.utils.clip_grad_norm_(policy.discrete_critic.parameters(), max_norm=clip_grad_norm_value)
                optimizers["discrete_critic"].step()

            # target EMA
            policy.update_target_networks()

        # ---------- last update in the UTD block ----------
        batch = next(offline_iterator)
        _check_nan_in_transition(batch)

        obs_feats, next_obs_feats = _get_observation_features(policy=policy,
                                                              observations=batch["state"],
                                                              next_observations=batch["next_state"])

        forward_batch = {
            ACTION: batch[ACTION],
            "reward": batch["reward"],
            "state": batch["state"],
            "next_state": batch["next_state"],
            "done": batch["done"],
            "observation_feature": obs_feats,
            "next_observation_feature": next_obs_feats,
            "complementary_info": batch.get("complementary_info"),
        }
    
        # critic
        loss_critic = policy.forward(forward_batch, model="critic")["loss_critic"]
        optimizers["critic"].zero_grad()
        loss_critic.backward()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.critic_ensemble.parameters(), max_norm=clip_grad_norm_value
        ).item()
        optimizers["critic"].step()
        
        # discrete critic (optional)
        metrics = {
            "loss_critic": loss_critic.item(),
            "critic_grad_norm": critic_grad_norm,
        }
        if policy.config.num_discrete_actions is not None:
            loss_disc = policy.forward(forward_batch, model="discrete_critic")["loss_discrete_critic"]
            optimizers["discrete_critic"].zero_grad()
            loss_disc.backward()
            disc_gn = torch.nn.utils.clip_grad_norm_(
                policy.discrete_critic.parameters(), max_norm=clip_grad_norm_value
            ).item()
            optimizers["discrete_critic"].step()
            metrics["loss_discrete_critic"] = loss_disc.item()
            metrics["discrete_critic_grad_norm"] = disc_gn

        # actor & temperature (at frequency)
        if optimization_step % policy_update_freq == 0:
            # actor
            loss_actor = policy.forward(forward_batch, model="actor")["loss_actor"]
            optimizers["actor"].zero_grad()
            loss_actor.backward()
            actor_gn = torch.nn.utils.clip_grad_norm_(policy.actor.parameters(), max_norm=clip_grad_norm_value).item()
            optimizers["actor"].step()
            metrics["loss_actor"] = loss_actor.item()
            metrics["actor_grad_norm"] = actor_gn

            # temperature
            loss_temp = policy.forward(forward_batch, model="temperature")["loss_temperature"]
            optimizers["temperature"].zero_grad()
            loss_temp.backward()
            temp_gn = torch.nn.utils.clip_grad_norm_([policy.log_alpha], max_norm=clip_grad_norm_value).item()
            optimizers["temperature"].step()
            policy.update_temperature()
            metrics["loss_temperature"] = loss_temp.item()
            metrics["temperature_grad_norm"] = temp_gn
            metrics["temperature"] = policy.temperature

        # target EMA
        policy.update_target_networks()

        # logs
        if (optimization_step % log_freq == 0) or (time.time() - t_last_log > 10):
            t_last_log = time.time()
            metrics["Optimization step"] = optimization_step
            metrics["offline_replay_buffer_size"] = len(offline_replay_buffer)
            logging.info(
                f"[OFFLINE] step={optimization_step} | "
                + " ".join([f"{k}={v:.4f}" for k, v in metrics.items() if isinstance(v, (int, float))])
            )
            if wandb_logger:
                wandb_logger.log_dict(d=metrics, mode="train", custom_step_key="Optimization step")

        # save
        if (optimization_step % save_freq == 0 and optimization_step > 0) or optimization_step + 1 == online_steps:
            _save_training_checkpoint_offline(
                cfg=cfg,
                optimization_step=optimization_step,
                online_steps=online_steps,
                policy=policy,
                optimizers=optimizers,
                offline_replay_buffer=offline_replay_buffer,
                fps=cfg.env.fps if cfg.env and hasattr(cfg.env, "fps") else 30,
            )

        optimization_step += 1

    logging.info(colored("Offline training finished", "green", attrs=["bold"]))


# -----------------------------
# Helpers (offline-only)
# -----------------------------
def _make_optimizers_and_scheduler(cfg: TrainRLServerPipelineConfig, policy: nn.Module):
    """
    Create Adam optimizers for actor/critic/(discrete_critic)/temperature.
    """
    optimizer_actor = torch.optim.Adam(
        params=[
            p
            for n, p in policy.actor.named_parameters()
            if not policy.config.shared_encoder or not n.startswith("encoder")
        ],
        lr=cfg.policy.actor_lr,
    )
    optimizer_critic = torch.optim.Adam(params=policy.critic_ensemble.parameters(), lr=cfg.policy.critic_lr)
    optimizers = {"actor": optimizer_actor, "critic": optimizer_critic}

    if getattr(policy.config, "num_discrete_actions", None) is not None:
        optimizer_discrete_critic = torch.optim.Adam(params=policy.discrete_critic.parameters(),
                                                     lr=cfg.policy.critic_lr)
        optimizers["discrete_critic"] = optimizer_discrete_critic
    
    optimizer_temperature = torch.optim.Adam(params=[policy.log_alpha], lr=cfg.policy.critic_lr)
    optimizers["temperature"] = optimizer_temperature
    lr_scheduler = None
    return optimizers, lr_scheduler


def _handle_resume_logic(cfg: TrainRLServerPipelineConfig) -> TrainRLServerPipelineConfig:
    """
    Mirror the original resume guard: forbid accidental overwrite, or load config from last checkpoint.
    """
    out_dir = cfg.output_dir
    if not cfg.resume:
        checkpoint_dir = os.path.join(out_dir, CHECKPOINTS_DIR, LAST_CHECKPOINT_LINK)
        if os.path.exists(checkpoint_dir):
            raise RuntimeError(f"Output directory {checkpoint_dir} already exists. Use resume=true to resume training.")
        return cfg

    checkpoint_dir = os.path.join(out_dir, CHECKPOINTS_DIR, LAST_CHECKPOINT_LINK)
    if not os.path.exists(checkpoint_dir):
        raise RuntimeError(f"No model checkpoint found in {checkpoint_dir} for resume=True")

    logging.info(colored("Valid checkpoint found → resume", "yellow", attrs=["bold"]))
    checkpoint_cfg_path = os.path.join(checkpoint_dir, PRETRAINED_MODEL_DIR, "train_config.json")
    checkpoint_cfg = TrainRLServerPipelineConfig.from_pretrained(checkpoint_cfg_path)
    checkpoint_cfg.resume = True
    return checkpoint_cfg


def _load_training_state(cfg: TrainRLServerPipelineConfig, optimizers: Optimizer | dict[str, Optimizer]):
    """
    Load optimizer states and return the previous optimization step, if resume.
    """
    if not cfg.resume:
        return None
    from lerobot.utils.train_utils import load_training_state as utils_load_training_state

    checkpoint_dir = os.path.join(cfg.output_dir, CHECKPOINTS_DIR, LAST_CHECKPOINT_LINK)
    logging.info(f"Loading training state from {checkpoint_dir}")
    try:
        step, _, _ = utils_load_training_state(Path(checkpoint_dir), optimizers, None)
        # offline-only 没有 interaction 步的概念，但保留 step
        training_state_path = os.path.join(checkpoint_dir, TRAINING_STATE_DIR, "training_state.pt")
        if os.path.exists(training_state_path):
            _ = torch.load(training_state_path, weights_only=False)  # noqa: S301
        logging.info(f"Resuming from step {step}")
        return step
    except Exception as e:
        logging.error(f"Failed to load training state: {e}")
        return None


def _initialize_offline_replay_buffer(cfg: TrainRLServerPipelineConfig, device: str, storage_device: str) -> ReplayBuffer:
    """
    Build an offline replay buffer from cfg.dataset.
    """
    if not cfg.dataset:
        raise RuntimeError("Offline-only learner requires cfg.dataset to be set (with a valid dataset).")

    if not cfg.resume:
        logging.info("Loading offline dataset via make_dataset(cfg)")
        offline_dataset = make_dataset(cfg)
        # Optional: convert joint actions to EE-delta 4D if IK info and action_dim<=4
        try:
            ik_cfg = cfg.env.processor.inverse_kinematics if (cfg.env and hasattr(cfg.env, "processor")) else None
            
            # Assume first 6 joints + 1 gripper
            left_joint_indices = list(range(0, 6))
            left_gripper_index = 6
            right_joint_indices = list(range(7, 13))
            right_gripper_index = 13    
            step_sizes = 1
            jt_cfg = JointToEEDeltaConfig(
                urdf_path=ik_cfg.urdf_path,
                target_frame_name=ik_cfg.target_frame_name,
                left_joint_indices=left_joint_indices,
                left_gripper_index=left_gripper_index,
                right_joint_indices=right_joint_indices,
                right_gripper_index=right_gripper_index,
                step_sizes=step_sizes,
            )

            offline_dataset = JointToEEDeltaDataset(offline_dataset, jt_cfg)
            logging.info("Wrapped offline dataset with JointToEEDeltaDataset (4D EE delta actions)")
        except Exception as e:
            logging.warning(f"Could not wrap offline dataset with JointToEEDeltaDataset: {e}")
    else:
        # resume: load snapshot exported previously
        logging.info("Resume: loading offline dataset snapshot from outputs/dataset_offline")
        dataset_offline_path = os.path.join(cfg.output_dir, "dataset_offline")
        offline_dataset = LeRobotDataset(repo_id=cfg.dataset.repo_id, root=dataset_offline_path)

    logging.info("Converting offline dataset → offline replay buffer")
    offline_replay_buffer = ReplayBuffer.from_lerobot_dataset(
        offline_dataset,
        device=device,
        state_keys=cfg.policy.input_features.keys(),
        storage_device=storage_device,
        optimize_memory=True,
        capacity=cfg.policy.offline_buffer_capacity,
    )
    return offline_replay_buffer


def _get_observation_features(policy: SACPolicy, observations: torch.Tensor, next_observations: torch.Tensor):
    """
    Cache vision features only if using a pretrained & frozen vision encoder.
    """
    if policy.config.vision_encoder_name is None or not policy.config.freeze_vision_encoder:
        return None, None
    with torch.no_grad():
        obs_feat = policy.actor.encoder.get_cached_image_features(observations)
        next_feat = policy.actor.encoder.get_cached_image_features(next_observations)
    return obs_feat, next_feat


def _save_training_checkpoint_offline(
    cfg: TrainRLServerPipelineConfig,
    optimization_step: int,
    online_steps: int,
    policy: nn.Module,
    optimizers: dict[str, Optimizer],
    offline_replay_buffer: ReplayBuffer,
    fps: int = 30,
):
    """
    Save model/optimizers + export offline buffer snapshot. No online buffer or interaction step.
    """
    logging.info(f"Checkpoint policy after step {optimization_step}")
    checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, online_steps, optimization_step)

    save_checkpoint(
        checkpoint_dir=checkpoint_dir,
        step=optimization_step,
        cfg=cfg,
        policy=policy,
        optimizer=optimizers,
        scheduler=None,
    )

    # Save a simple training state (no interaction step offline)
    training_state_dir = os.path.join(checkpoint_dir, TRAINING_STATE_DIR)
    os.makedirs(training_state_dir, exist_ok=True)
    training_state = {"step": optimization_step}
    torch.save(training_state, os.path.join(training_state_dir, "training_state.pt"))

    # Update "last" symlink
    update_last_checkpoint(checkpoint_dir)

    # Export offline buffer snapshot (overwrite a stable folder)
    dataset_offline_dir = os.path.join(cfg.output_dir, "dataset_offline")
    if os.path.exists(dataset_offline_dir) and os.path.isdir(dataset_offline_dir):
        shutil.rmtree(dataset_offline_dir)
    offline_replay_buffer.to_lerobot_dataset(
        cfg.dataset.repo_id,
        fps=fps,
        root=dataset_offline_dir,
    )
    logging.info("Checkpoint saved and offline buffer snapshot exported.")


def _log_training_info(cfg: TrainRLServerPipelineConfig, policy: nn.Module, offline_buffer: ReplayBuffer) -> None:
    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    if cfg.env is not None and hasattr(cfg.env, "task"):
        logging.info(f"{cfg.env.task=}")
    logging.info(f"{cfg.policy.online_steps=}")
    logging.info(f"offline_replay_buffer_size={len(offline_buffer)}")
    logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
    logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")


def _check_nan_in_transition(batch: dict) -> None:
    """
    Quick NaN guard in sampled batch (optional).
    """
    def _has_nan(x: torch.Tensor) -> bool:
        return isinstance(x, torch.Tensor) and torch.isnan(x).any().item()

    for k, v in batch.get("state", {}).items():
        if _has_nan(v):
            logging.warning(f"[OFFLINE] NaN detected in state[{k}]")
    for k, v in batch.get("next_state", {}).items():
        if _has_nan(v):
            logging.warning(f"[OFFLINE] NaN detected in next_state[{k}]")
    if _has_nan(batch.get(ACTION)):
        logging.warning("[OFFLINE] NaN detected in actions")


if __name__ == "__main__":
    train_cli()
