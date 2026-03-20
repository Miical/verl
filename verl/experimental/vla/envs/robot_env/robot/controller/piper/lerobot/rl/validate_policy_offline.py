#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Offline validation: print one sampled action (TARGET / PRE / RAND) per PRINT_EVERY batches

import os
import time
import random
import logging
from copy import deepcopy
from pprint import pformat
from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.transforms import JointToEEDeltaConfig, JointToEEDeltaDataset
from lerobot.policies.factory import make_policy
from lerobot.policies.sac.modeling_sac import SACPolicy
from lerobot.rl.buffer import ReplayBuffer
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import get_safe_torch_device
from lerobot.utils.constants import ACTION

# ----- 打印控制 -----
PRINT_EVERY = 10     # 每隔多少个 batch 打印一次
PRINT_DIMS  = 8      # 打印动作的前多少维（避免太长）
SAMPLE_RANDOM = True # True: 每次从 batch 随机抽一个样本；False: 一直取第一个样本


def _maybe_wrap_joint_to_ee_delta(cfg, dataset):
    """若策略输出为 4D EE delta，则把原始数据集包装为 EE-delta 版本，保持动作空间对齐。"""
    try:
        ik_cfg = cfg.env.processor.inverse_kinematics if (cfg.env and hasattr(cfg.env, "processor")) else None
        if ik_cfg is not None and cfg.policy.output_features[ACTION].shape[0] <= 4:
            joint_indices = list(range(0, 6))
            gripper_index = 6
            step_sizes = {
                "x": ik_cfg.end_effector_step_sizes["x"],
                "y": ik_cfg.end_effector_step_sizes["y"],
                "z": ik_cfg.end_effector_step_sizes["z"],
            }
            jt_cfg = JointToEEDeltaConfig(
                urdf_path=ik_cfg.urdf_path,
                target_frame_name=ik_cfg.target_frame_name,
                joint_indices=joint_indices,
                gripper_index=gripper_index,
                step_sizes=step_sizes,
                gripper_speed_factor=20.0,
            )
            dataset = JointToEEDeltaDataset(dataset, jt_cfg)
            logging.info("Dataset wrapped with JointToEEDeltaDataset (4D EE delta actions).")
    except Exception as e:
        logging.warning(f"[VAL] Could not wrap dataset with JointToEEDeltaDataset: {e}")
    return dataset


def _build_offline_iterator_like_train(cfg: TrainRLServerPipelineConfig,
                                       device: str,
                                       storage_device: str,
                                       batch_size: int,
                                       resume: bool = False):
    if not cfg.dataset:
        raise RuntimeError("[VAL] cfg.dataset 未设置，无法加载离线数据。")

    if not resume:
        logging.info("[VAL] Loading offline dataset via make_dataset(cfg)")
        dataset = make_dataset(cfg)
        dataset = _maybe_wrap_joint_to_ee_delta(cfg, dataset)
    else:
        logging.info("[VAL] Resume mode: loading offline dataset snapshot from outputs/dataset_offline")
        dataset_offline_path = os.path.join(cfg.output_dir, "dataset_offline")
        dataset = LeRobotDataset(repo_id=cfg.dataset.repo_id, root=dataset_offline_path)

    logging.info("[VAL] Converting dataset → offline replay buffer")
    offline_replay_buffer = ReplayBuffer.from_lerobot_dataset(
        dataset,
        device=device,
        state_keys=cfg.policy.input_features.keys(),
        storage_device=storage_device,
        optimize_memory=True,
        capacity=cfg.policy.offline_buffer_capacity,
    )
    iterator = offline_replay_buffer.get_iterator(
        batch_size=batch_size,
        async_prefetch=cfg.policy.async_prefetch,
        queue_size=2,
    )
    return offline_replay_buffer, iterator


def _compute_batch_metrics(pred: Tensor, target: Tensor) -> Dict[str, float]:
    if pred.dtype != target.dtype:
        target = target.to(pred.dtype)
    mse = F.mse_loss(pred, target, reduction="mean").item()
    mae = F.l1_loss(pred, target, reduction="mean").item()
    pred_n = F.normalize(pred, dim=-1)
    tgt_n  = F.normalize(target, dim=-1)
    cos = (pred_n * tgt_n).sum(dim=-1).mean().item()
    return {"mse": mse, "mae": mae, "cosine": cos}


def _accumulate_metrics(sum_dict: Dict[str, float], add_dict: Dict[str, float], n: int):
    for k, v in add_dict.items():
        sum_dict[k] = sum_dict.get(k, 0.0) + float(v) * n


def _select_action_deterministic(policy: SACPolicy, observation: Dict[str, Tensor]) -> Tensor:
    """
    确定性推理（跨模型/运行可复现，便于对比）：
      - 连续维：actor 的 mean
      - 离散维：discrete_critic 的 argmax（若有）
    """
    with torch.no_grad():
        obs_feats = None
        if policy.config.shared_encoder and policy.actor.encoder.has_images:
            obs_feats = policy.actor.encoder.get_cached_image_features(observation)

        _, _, means = policy.actor(observation, obs_feats)

        if policy.config.num_discrete_actions is not None:
            q_dis = policy.discrete_critic(observation, obs_feats)  # (B, K)
            disc = torch.argmax(q_dis, dim=-1, keepdim=True)        # (B, 1)
            actions = torch.cat([means, disc], dim=-1)
        else:
            actions = means
        return actions


def _fmt_row(vec: Tensor, max_dims: int) -> str:
    vec = vec.detach().cpu()
    d = vec.numel()
    d_show = min(d, max_dims)
    vals = [f"{float(vec[i]): .4f}" for i in range(d_show)]
    if d_show < d:
        vals.append("...")
    return "[" + ", ".join(vals) + "]"


@parser.wrap()
def validate_cli(cfg: TrainRLServerPipelineConfig):
    # 终端日志
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.info("[VAL] Offline policy validation (print single sampled action) started")
    logging.info(pformat(cfg.to_dict()))

    pretrained = getattr(cfg.policy, "pretrained_path", None)
    if not pretrained:
        logging.warning("[VAL] WARNING: cfg.policy.pretrained_path 未设置，PRETRAINED 与 RANDOM 将等价（均随机）。")
    else:
        logging.info(f"[VAL] Using pretrained_path={pretrained}")

    # 设备/加速
    set_seed(cfg.seed)
    device = get_safe_torch_device(cfg.policy.device, log=True)
    storage_device = get_safe_torch_device(cfg.policy.storage_device)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # 模型 1：PRETRAINED
    logging.info("[VAL] Loading PRETRAINED policy")
    policy_pre: SACPolicy = make_policy(cfg=cfg.policy, env_cfg=cfg.env)
    policy_pre.eval()
    assert isinstance(policy_pre, nn.Module)

    # 模型 2：RANDOM（不加载权重）
    logging.info("[VAL] Building RANDOM policy (same arch, no pretrained)")
    cfg_rand = deepcopy(cfg)
    cfg_rand.policy.pretrained_path = None
    policy_rand: SACPolicy = make_policy(cfg=cfg_rand.policy, env_cfg=cfg_rand.env)
    policy_rand.eval()
    assert isinstance(policy_rand, nn.Module)

    # 数据
    batch_size = cfg.batch_size
    offline_buffer, iterator = _build_offline_iterator_like_train(
        cfg=cfg, device=device, storage_device=storage_device, batch_size=batch_size, resume=cfg.resume
    )

    total = 0
    sums_pre, sums_rand = {}, {}
    t0 = time.time()
    batch_idx = 0

    logging.info(f"[VAL] Dataset size≈{len(offline_buffer)}, batch_size={batch_size}")
    with torch.no_grad():
        for batch in iterator:
            obs: Dict[str, Tensor] = batch["state"]
            observation = {k: v for k, v in obs.items() if k in cfg.policy.input_features}
            target_action: Tensor = batch[ACTION]
            if target_action.ndim == 1:
                target_action = target_action.unsqueeze(0)

            act_pre  = _select_action_deterministic(policy_pre,  observation)
            act_rand = _select_action_deterministic(policy_rand, observation)
            if act_pre.ndim == 1:  act_pre = act_pre.unsqueeze(0)
            if act_rand.ndim == 1: act_rand = act_rand.unsqueeze(0)

            n = act_pre.shape[0]
            total += n
            m_pre  = _compute_batch_metrics(act_pre,  target_action)
            m_rand = _compute_batch_metrics(act_rand, target_action)
            _accumulate_metrics(sums_pre,  m_pre,  n)
            _accumulate_metrics(sums_rand, m_rand, n)

            if (batch_idx % PRINT_EVERY) == 0:
                # 选样本索引
                idx = random.randrange(n) if SAMPLE_RANDOM else 0
                # 打印指标
                print(
                    f"[VAL] b{batch_idx:05d} | "
                    f"PRE  mse={m_pre['mse']:.6f} mae={m_pre['mae']:.6f} cos={m_pre['cosine']:.4f}  ||  "
                    f"RAND mse={m_rand['mse']:.6f} mae={m_rand['mae']:.6f} cos={m_rand['cosine']:.4f}"
                    f"差值mse={m_rand['mse'] - m_pre['mse']:.6f},mae={m_rand['mae'] - m_pre['mae']:.6f},cos={m_rand['cosine'] - m_pre['cosine']:.4f}"
                )
                # 打印一条动作（裁剪维度）
                pre_row = _fmt_row(act_pre[idx], PRINT_DIMS)
                rnd_row = _fmt_row(act_rand[idx], PRINT_DIMS)
                tgt_row = _fmt_row(target_action[idx], PRINT_DIMS)
                print(f"    sample idx={idx} | PRE={pre_row}")
                print(f"                     RAND={rnd_row}")
                print(f"                     TARGET={tgt_row}")

            batch_idx += 1

    if total == 0:
        logging.warning("[VAL] No samples were validated (total=0). Check dataset / filters.")
        return

    mean_pre  = {k: v / float(total) for k, v in sums_pre.items()}
    mean_rand = {k: v / float(total) for k, v in sums_rand.items()}
    elapsed = time.time() - t0

    print("\n[VAL] ================== SUMMARY ==================")
    print(f"[VAL] Total samples: {total}   Time: {elapsed:.2f}s  | {(total/elapsed):.1f} samples/s")
    print("[VAL] --- PRETRAINED vs TARGET ---")
    print(f"[VAL] MSE={mean_pre['mse']:.8f}   MAE={mean_pre['mae']:.8f}   COS={mean_pre['cosine']:.6f}")
    print("[VAL] --- RANDOM vs TARGET ---")
    print(f"[VAL] MSE={mean_rand['mse']:.8f}  MAE={mean_rand['mae']:.8f}  COS={mean_rand['cosine']:.6f}")
    print("[VAL] ============================================\n")


if __name__ == "__main__":
    validate_cli()
