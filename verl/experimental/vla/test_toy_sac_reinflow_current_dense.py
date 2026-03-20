#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Toy SAC + PI0/flow exact-path-logprob validation script
Dense-reward version for current ReinFlow-lite interfaces.
"""
import argparse
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import nn
from tensordict import TensorDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from omegaconf import OmegaConf

from verl.protocol import DataProto
from verl.utils.device import get_device_name, get_device_id
from verl.experimental.vla.models.pi0_torch.configuration_pi0_torch import PI0TorchConfig
from verl.experimental.vla.models.pi0_torch.modeling_pi0_torch import PI0ForActionPrediction
from verl.experimental.vla.sac.sac_actor import RobDataParallelSACActor


# ----------------------------
# Tiny fake PI0 backbone
# ----------------------------
class FakePI0Model(nn.Module):
    def __init__(
        self,
        state_dim: int = 32,
        n_action_steps: int = 50,
        max_action_dim: int = 32,
        exec_steps: int = 10,
        exec_dim: int = 7,
        num_steps: int = 8,
        prefix_len: int = 1,
        prefix_dim: int = 2048,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.n_action_steps = n_action_steps
        self.max_action_dim = max_action_dim
        self.exec_steps = exec_steps
        self.exec_dim = exec_dim
        self.num_steps = num_steps
        self.use_cache = False
        self.prefix_len = prefix_len
        self.prefix_dim = prefix_dim

        self.state_proj = nn.Linear(state_dim, exec_dim)
        self.v_mlp = nn.Sequential(
            nn.Linear(exec_dim + exec_dim + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, exec_dim),
        )

    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks):
        B = images[0].shape[0]
        device = images[0].device
        prefix_embs = torch.zeros((B, self.prefix_len, self.prefix_dim), device=device, dtype=torch.float32)
        prefix_pad_masks = torch.ones((B, self.prefix_len), device=device, dtype=torch.bool)
        prefix_att_masks = torch.ones((B, self.prefix_len), device=device, dtype=torch.bool)
        return (prefix_embs, prefix_pad_masks, prefix_att_masks)

    def sample_noise(self, actions_shape, device):
        x = torch.zeros(actions_shape, device=device, dtype=torch.float32)
        x[:, : self.exec_steps, : self.exec_dim] = torch.randn(
            (actions_shape[0], self.exec_steps, self.exec_dim), device=device, dtype=torch.float32
        )
        return x

    def denoise_step(self, states, prefix_pad_masks, past_key_values, x_t, t):
        B = states.shape[0]
        device = states.device

        x_exec = x_t[:, : self.exec_steps, : self.exec_dim]
        s_cond = self.state_proj(states)
        s_cond = s_cond[:, None, :].expand(B, self.exec_steps, self.exec_dim)
        t_cond = t[:, None, None].expand(B, self.exec_steps, 1)

        inp = torch.cat([x_exec, s_cond, t_cond], dim=-1)
        v_exec = self.v_mlp(inp.reshape(B * self.exec_steps, -1)).reshape(B, self.exec_steps, self.exec_dim)

        v = torch.zeros((B, self.n_action_steps, self.max_action_dim), device=device, dtype=torch.float32)
        v[:, : self.exec_steps, : self.exec_dim] = v_exec
        return v


# ----------------------------
# Dense-reward origin-reaching env
# ----------------------------
class ReachOriginToyEnv:
    def __init__(
        self,
        exec_steps: int = 10,
        exec_dim: int = 7,
        success_radius: float = 0.15,
        action_cost: float = 0.01,
        success_reward: float = 1.0,
        noise_std: float = 0.01,
        done_on_success: bool = True,
        state_scale: float = 1.0,
        reward_mode: str = "dense",
    ):
        self.exec_steps = exec_steps
        self.exec_dim = exec_dim
        self.success_radius = success_radius
        self.action_cost = action_cost
        self.success_reward = success_reward
        self.noise_std = noise_std
        self.done_on_success = done_on_success
        self.state_scale = state_scale
        assert reward_mode in ("dense", "sparse")
        self.reward_mode = reward_mode

    def reset_raw(self, B: int, device: torch.device, start_range: float = 1.0) -> torch.Tensor:
        s7 = torch.empty((B, self.exec_dim), device=device).uniform_(-start_range, start_range)
        s32 = torch.zeros((B, 32), device=device, dtype=torch.float32)
        s32[:, : self.exec_dim] = s7
        return s32

    def _reward_dense(self, s7: torch.Tensor, a: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return -self.state_scale * (s7.pow(2).sum(dim=-1)) - self.action_cost * (a.pow(2).sum(dim=-1))

    def _reward_sparse(self, s7: torch.Tensor, a: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        dist = torch.norm(s7, dim=-1)
        success = dist <= self.success_radius
        return torch.where(
            success,
            torch.full((s7.shape[0],), self.success_reward, device=s7.device, dtype=dtype),
            torch.zeros((s7.shape[0],), device=s7.device, dtype=dtype),
        ) - self.action_cost * (a.pow(2).sum(dim=-1))

    def step_chunk_raw(
        self, s32: torch.Tensor, a_seq_10x7: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B = s32.shape[0]
        device = s32.device
        s7 = s32[:, : self.exec_dim].clone()

        rewards = []
        masks = []
        done = torch.zeros((B,), device=device, dtype=torch.bool)

        for k in range(self.exec_steps):
            a = a_seq_10x7[:, k, :]
            not_done = ~done
            if not_done.any():
                if self.noise_std > 0:
                    noise = self.noise_std * torch.randn_like(s7[not_done])
                else:
                    noise = torch.zeros_like(s7[not_done])
                s7_next = s7[not_done] + a[not_done] + noise
                s7 = s7.clone()
                s7[not_done] = s7_next

            if self.reward_mode == "dense":
                r = self._reward_dense(s7, a, s32.dtype)
            else:
                r = self._reward_sparse(s7, a, s32.dtype)

            step_mask = (~done).to(torch.bool)
            masks.append(step_mask)
            rewards.append(torch.where(step_mask, r, torch.zeros_like(r)))

            if self.done_on_success:
                dist = torch.norm(s7, dim=-1)
                success = dist <= self.success_radius
                done = done | success

        s_next = torch.zeros_like(s32)
        s_next[:, : self.exec_dim] = s7
        rewards_seq = torch.stack(rewards, dim=-1)      # [B, T]
        response_mask = torch.stack(masks, dim=-1)      # [B, T]
        chunk_done = done.clone()                       # [B]
        return s_next, rewards_seq, response_mask, chunk_done


def make_dummy_obs(states32: torch.Tensor, device: torch.device) -> Dict[str, torch.Tensor]:
    B = states32.shape[0]
    images = torch.zeros((B, 1, 3, 224, 224), device=device, dtype=torch.float32)
    image_masks = torch.ones((B, 1), device=device, dtype=torch.bool)
    lang_tokens = torch.zeros((B, 1), device=device, dtype=torch.long)
    lang_masks = torch.ones((B, 1), device=device, dtype=torch.bool)
    return {
        "images": images,
        "image_masks": image_masks,
        "lang_tokens": lang_tokens,
        "lang_masks": lang_masks,
        "states": states32,
    }


def _call_sac_forward_actor(policy, state_features):
    out = policy.sac_forward_actor(state_features)
    if not isinstance(out, tuple):
        raise TypeError(f"policy.sac_forward_actor must return tuple, got {type(out)}")
    if len(out) == 2:
        actions, logp = out
        metrics = {}
    elif len(out) == 3:
        actions, logp, metrics = out
        if metrics is None:
            metrics = {}
    else:
        raise ValueError(f"Unexpected sac_forward_actor return length: {len(out)}")
    return actions, logp, metrics


@torch.no_grad()
def collect_rollout_batch(policy, env, B, exec_steps, exec_dim, gamma_micro, device, start_range=1.0):
    s0 = env.reset_raw(B, device=device, start_range=start_range)
    s0_obs = make_dummy_obs(s0, device)

    sf0 = policy.sac_forward_state_features(s0_obs)
    a0_full, logp0, _actor_metrics = _call_sac_forward_actor(policy, sf0)

    a_exec = a0_full[:, :exec_steps, :exec_dim].contiguous()
    s1, rewards_seq, response_mask, chunk_done = env.step_chunk_raw(s0, a_exec)

    discounts = (gamma_micro ** torch.arange(exec_steps, device=device, dtype=rewards_seq.dtype))[None, :]
    mask_f = response_mask.to(rewards_seq.dtype)
    denom = mask_f.sum(dim=-1).clamp_min(1.0)
    reward_disc_sum = (rewards_seq * discounts * mask_f).sum(dim=-1)
    reward_scalar = reward_disc_sum / denom   # [B]

    rewards_for_algo = reward_scalar.to(torch.float32)                 # [B]
    dones_for_algo = chunk_done.to(torch.float32)                     # [B]
    valids_for_algo = torch.ones((B,), device=device, dtype=torch.float32)
    positive_for_algo = chunk_done.to(torch.float32)                  # [B]

    batch = TensorDict(
        {
            "task_ids": torch.zeros((B,), device=device, dtype=torch.long),
            "a0.full_action": a0_full.to(torch.float32),
            "a1.full_action": torch.zeros_like(a0_full, dtype=torch.float32),
            "s0.states": s0.to(torch.float32),
            "s1.states": s1.to(torch.float32),
            "s0.images": s0_obs["images"],
            "s1.images": make_dummy_obs(s1, device)["images"],
            "s0.image_masks": s0_obs["image_masks"],
            "s1.image_masks": make_dummy_obs(s1, device)["image_masks"],
            "s0.lang_tokens": s0_obs["lang_tokens"],
            "s1.lang_tokens": make_dummy_obs(s1, device)["lang_tokens"],
            "s0.lang_masks": s0_obs["lang_masks"],
            "s1.lang_masks": make_dummy_obs(s1, device)["lang_masks"],
            "rewards": rewards_for_algo,
            "dones": dones_for_algo,
            "valids": valids_for_algo,
            "positive_sample_mask": positive_for_algo,
            "response_mask": response_mask,
        },
        batch_size=[B],
        device=device,
    )

    dist0 = torch.norm(s0[:, :exec_dim], dim=-1)
    dist1 = torch.norm(s1[:, :exec_dim], dim=-1)

    dbg = {
        "reward_scalar_mean": reward_scalar.mean().item(),
        "dist0": dist0.mean().item(),
        "dist1": dist1.mean().item(),
        "succ1": (dist1 <= env.success_radius).float().mean().item(),
        "chunk_done_ratio": chunk_done.float().mean().item(),
        "avg_valid_steps": response_mask.float().sum(dim=-1).mean().item(),
        "logp0_mean": logp0.mean().item() if logp0 is not None else 0.0,
    }
    return batch, dbg


def moving_average(x, w=10):
    if len(x) < 2:
        return x
    out = []
    for i in range(len(x)):
        j0 = max(0, i - w + 1)
        out.append(sum(x[j0:i + 1]) / (i - j0 + 1))
    return out


def plot_series(series, out_path, smooth=1):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    x = list(range(1, len(series) + 1))
    y = moving_average(series, max(1, int(smooth))) if smooth > 1 else series
    plt.figure()
    plt.plot(x, y)
    plt.xlabel("update step")
    plt.ylabel(Path(out_path).stem)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--micro_bs", type=int, default=64)
    parser.add_argument("--exec_steps", type=int, default=10)
    parser.add_argument("--exec_dim", type=int, default=7)
    parser.add_argument("--gamma_micro", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smooth", type=int, default=10)
    parser.add_argument("--out_dir", type=str, default="toy_reinflow_dense")
    parser.add_argument("--success_radius", type=float, default=0.15)
    parser.add_argument("--success_reward", type=float, default=1.0)
    parser.add_argument("--action_cost", type=float, default=0.01)
    parser.add_argument("--state_scale", type=float, default=1.0)
    parser.add_argument("--start_range", type=float, default=1.0)
    parser.add_argument("--noise_std", type=float, default=0.01)
    parser.add_argument("--reward_mode", type=str, default="dense", choices=["dense", "sparse"])

    parser.add_argument("--auto_entropy", action="store_true", default=False)
    parser.add_argument("--initial_alpha", type=float, default=0.0)
    parser.add_argument("--alpha_lr", type=float, default=3e-4)
    parser.add_argument("--target_entropy", type=float, default=None)

    parser.add_argument("--flow_logprob_mode", type=str, default="path_exact", choices=["path_exact", "surrogate"])
    parser.add_argument("--flow_sigma_min", type=float, default=1e-3)
    parser.add_argument("--flow_sigma_max", type=float, default=5e-1)
    parser.add_argument("--flow_sigma_init", type=float, default=5e-2)
    parser.add_argument("--flow_sigma_head_hidden_dim", type=int, default=128)
    parser.add_argument("--flow_sigma_use_latent_stats", action="store_true", default=True)

    parser.add_argument("--critic_warmup_steps", type=int, default=20)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    import torch.distributed as dist

    def _init_dist_if_needed():
        if not dist.is_available() or dist.is_initialized():
            return
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            dist.init_process_group(backend=backend, init_method="env://")
        else:
            init_file = f"file://{tempfile.gettempdir()}/pg_{os.getpid()}"
            dist.init_process_group(backend=backend, init_method=init_file, rank=0, world_size=1)

    _init_dist_if_needed()

    device_type = get_device_name()
    device = torch.device(f"cuda:{get_device_id()}") if device_type == "cuda" else torch.device(device_type)

    if args.target_entropy is None:
        args.target_entropy = -float(args.exec_dim)

    env = ReachOriginToyEnv(
        exec_steps=args.exec_steps,
        exec_dim=args.exec_dim,
        success_radius=args.success_radius,
        success_reward=args.success_reward,
        action_cost=args.action_cost,
        noise_std=args.noise_std,
        done_on_success=True,
        state_scale=args.state_scale,
        reward_mode=args.reward_mode,
    )

    state_norm_stats = {"mean": [0.0] * 32, "std": [1.0] * 32}
    action_norm_stats = {"mean": [0.0] * 32, "std": [1.0] * 32}
    pi0_cfg = PI0TorchConfig(
        state_norm_stats=state_norm_stats,
        action_norm_stats=action_norm_stats,
        pi05_enabled=False,
        sac_enable=True,
        double_q=True,
        flow_logprob_mode=args.flow_logprob_mode,
        flow_sigma_min=args.flow_sigma_min,
        flow_sigma_max=args.flow_sigma_max,
        flow_sigma_init=args.flow_sigma_init,
        flow_sigma_head_hidden_dim=args.flow_sigma_head_hidden_dim,
        flow_sigma_use_latent_stats=args.flow_sigma_use_latent_stats,
    )

    policy = PI0ForActionPrediction(pi0_cfg).to(device)
    policy.model = FakePI0Model(
        state_dim=32,
        n_action_steps=50,
        max_action_dim=32,
        exec_steps=args.exec_steps,
        exec_dim=args.exec_dim,
        num_steps=8,
        prefix_len=1,
        prefix_dim=2048,
    ).to(device)

    policy.freeze_vision_tower = lambda: None
    policy._build_kv_cache_from_prefix = lambda prefix_features: None

    gamma_eff = float(args.gamma_micro) ** int(args.exec_steps)
    cfg = OmegaConf.create(
        {
            "sac": {
                "gamma": gamma_eff,
                "tau": 0.005,
                "auto_entropy": bool(args.auto_entropy),
                "initial_alpha": float(args.initial_alpha),
                "alpha_type": "softplus",
                "alpha_lr": float(args.alpha_lr),
                "target_entropy": float(args.target_entropy),
                "critic_replay_positive_sample_ratio": 0.5,
                "actor_replay_positive_sample_ratio": 0.5,
            },
            "replay_pool_single_size": 4000,
            "replay_pool_save_dir": "/tmp/toy_replay_pool_current",
            "replay_pool_save_interval": 10_000_000,
            "critic_lr": 1e-4,
            "critic_weight_decay": 0.0,
            "ppo_mini_batch_size": int(args.batch_size),
            "ppo_micro_batch_size_per_gpu": int(args.micro_bs),
            "grad_clip": 1.0,
            "critic_warmup_steps": int(args.critic_warmup_steps),
            "actor_update_interval": 1,
        }
    )

    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    actor = RobDataParallelSACActor(cfg, policy, optimizer)

    metrics_hist = []
    for step in range(1, args.updates + 1):
        batch, dbg = collect_rollout_batch(
            policy=policy,
            env=env,
            B=args.batch_size,
            exec_steps=args.exec_steps,
            exec_dim=args.exec_dim,
            gamma_micro=args.gamma_micro,
            device=device,
            start_range=args.start_range,
        )

        data = DataProto(batch=batch, meta_info={"global_steps": step})
        metrics = actor.update_policy(data)

        with torch.no_grad():
            s0 = batch["s0.states"]
            s0_obs = make_dummy_obs(s0, device)
            sf = policy.sac_forward_state_features(s0_obs)
            a_full, logp, _actor_metrics_eval = _call_sac_forward_actor(policy, sf)
            q_pi = policy.sac_forward_critic(
                {"full_action": a_full},
                sf,
                use_target_network=False,
                method="min",
                requires_grad=False,
            ).mean().item()

        metrics["debug/Q_pi"] = q_pi
        metrics["debug/logp_mean_eval"] = logp.mean().item() if logp is not None else 0.0
        for k, v in dbg.items():
            metrics[f"debug/{k}"] = float(v)
        metrics_hist.append(metrics)

        if step % 10 == 0 or step == 1:
            critic_loss = float(metrics.get("critic/loss", float("nan")))
            actor_loss = float(metrics.get("actor/loss", float("nan")))
            logp_mean = float(metrics.get("actor/logprob_mean", float("nan")))
            q_pi_val = float(metrics.get("debug/Q_pi", float("nan")))
            succ1 = float(metrics.get("debug/succ1", float("nan")))
            done_ratio = float(metrics.get("debug/chunk_done_ratio", float("nan")))
            sigma_mean = float(metrics.get("actor/flow_sigma_mean", float("nan")))
            path_logp = float(metrics.get("actor/flow_path_logprob_mean", float("nan")))
            path_entropy = float(metrics.get("actor/flow_path_entropy_mean", float("nan")))

            print(
                f"[step {step:04d}] "
                f"critic_loss={critic_loss:.4f} "
                f"actor_loss={actor_loss:.4f} "
                f"logp={logp_mean:.3f} "
                f"Q_pi={q_pi_val:.3f} "
                f"succ1={succ1:.3f} "
                f"done={done_ratio:.3f} "
                f"sigma={sigma_mean:.5f} "
                f"path_logp={path_logp:.3f} "
                f"path_entropy={path_entropy:.3f}"
            )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = out_dir / "metrics.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for m in metrics_hist:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f"Saved metrics to: {jsonl_path}")

    def get_series(key):
        return [float(m.get(key, 0.0)) for m in metrics_hist]

    for key, name in [
        ("critic/loss", "critic_loss.png"),
        ("actor/loss", "actor_loss.png"),
        ("debug/Q_pi", "Q_pi.png"),
        ("debug/succ1", "succ1.png"),
        ("debug/chunk_done_ratio", "chunk_done_ratio.png"),
        ("debug/reward_scalar_mean", "reward_scalar_mean.png"),
        ("debug/dist0", "dist0.png"),
        ("debug/dist1", "dist1.png"),
        ("actor/flow_sigma_mean", "flow_sigma_mean.png"),
        ("actor/flow_sigma_std", "flow_sigma_std.png"),
        ("actor/flow_path_logprob_mean", "flow_path_logprob_mean.png"),
        ("actor/flow_path_entropy_mean", "flow_path_entropy_mean.png"),
        ("debug/logp_mean_eval", "eval_logp_mean.png"),
    ]:
        plot_series(get_series(key), out_dir / name, smooth=args.smooth)


if __name__ == "__main__":
    main()