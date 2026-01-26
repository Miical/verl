# recipe/vla/sac/tests/test_toy_sac_with_your_interfaces_original.py
# -*- coding: utf-8 -*-
"""
Original-style SAC toy (intentionally misaligned/noisy):
- reward: keep per-micro-step rewards (B,10), algorithm uses rewards.max(dim=-1)
- discount: only once at chunk-level (sac.gamma = gamma_micro)
- prefix: random (2048-dim noise)
- actions: full 50x32 noise (unused steps/dims are NOT zeroed)
"""

import os
import math
import json
import random
import argparse
from pathlib import Path

import torch
from torch import nn
from tensordict import TensorDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from omegaconf import OmegaConf

from verl.utils.device import get_device_name, get_device_id
from verl.protocol import DataProto

from recipe.vla.models.pi0_torch.configuration_pi0_torch import PI0TorchConfig
from recipe.vla.models.pi0_torch.modeling_pi0_torch import PI0ForActionPrediction
from recipe.vla.sac.sac_actor import PI0RobDataParallelPPOActor


# ----------------------------
# Fake PI0Model stub (ORIGINAL)
# ----------------------------
class FakePI0Model(nn.Module):
    """
    Same API, but:
      - prefix_embs is random
      - sample_noise is full randn (B,50,32)
      - denoise_step only models executed slice (10x7); rest v=0
        -> non-executed dims remain noisy in x trajectory
    """

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

        # keep conditioning from state
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
        # ORIGINAL: random prefix noise
        prefix_embs = torch.randn((B, self.prefix_len, self.prefix_dim), device=device, dtype=torch.float32)
        prefix_pad_masks = torch.ones((B, self.prefix_len), device=device, dtype=torch.bool)
        prefix_att_masks = torch.ones((B, self.prefix_len), device=device, dtype=torch.bool)
        return (prefix_embs, prefix_pad_masks, prefix_att_masks)

    def sample_noise(self, actions_shape, device):
        # ORIGINAL: full noise everywhere
        return torch.randn(actions_shape, device=device, dtype=torch.float32)

    def denoise_step(self, states, prefix_pad_masks, past_key_values, x_t, t):
        B = states.shape[0]
        device = states.device

        x_exec = x_t[:, : self.exec_steps, : self.exec_dim]  # (B,10,7)
        s_cond = self.state_proj(states)                      # (B,7)
        s_cond = s_cond[:, None, :].expand(B, self.exec_steps, self.exec_dim)
        t_cond = t[:, None, None].expand(B, self.exec_steps, 1)

        inp = torch.cat([x_exec, s_cond, t_cond], dim=-1)  # (B,10,15)
        v_exec = self.v_mlp(inp.reshape(B * self.exec_steps, -1)).reshape(B, self.exec_steps, self.exec_dim)

        v = torch.zeros((B, self.n_action_steps, self.max_action_dim), device=device, dtype=torch.float32)
        v[:, : self.exec_steps, : self.exec_dim] = v_exec
        return v


# ----------------------------
# Toy env: "push ball to origin"
# ----------------------------
class PushBallToyEnv:
    def __init__(self, exec_steps=10, exec_dim=7, noise_std=0.01, action_cost=0.01):
        self.exec_steps = exec_steps
        self.exec_dim = exec_dim
        self.noise_std = noise_std
        self.action_cost = action_cost

    def reset(self, B, device):
        s7 = torch.empty((B, self.exec_dim), device=device).uniform_(-1.0, 1.0)
        s32 = torch.zeros((B, 32), device=device, dtype=torch.float32)
        s32[:, : self.exec_dim] = s7
        return s32

    def step_chunk(self, s32, a_seq_10x7):
        B = s32.shape[0]
        s7 = s32[:, : self.exec_dim]
        rewards = []
        for k in range(self.exec_steps):
            a = a_seq_10x7[:, k, :]
            noise = self.noise_std * torch.randn_like(s7)
            s7 = s7 + a + noise
            r = -(s7.pow(2).sum(dim=-1)) - self.action_cost * (a.pow(2).sum(dim=-1))
            rewards.append(r)
        rewards_seq = torch.stack(rewards, dim=-1)  # (B,10)
        s32_next = torch.zeros_like(s32)
        s32_next[:, : self.exec_dim] = s7
        return s32_next, rewards_seq


def make_dummy_obs(states32, device):
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


@torch.no_grad()
def collect_batch_original(policy, env, B, exec_steps, exec_dim, device):
    """
    ORIGINAL:
      - rewards stored as per-step (B,10)
      - algo uses rewards.max(dim=-1) internally
      - no micro-step discounting here
    """
    s0 = env.reset(B, device=device)
    s0_obs = make_dummy_obs(s0, device)

    sf = policy.sac_forward_state_features(s0_obs)
    a0_full, _logp0 = policy.sac_forward_actor(sf)  # (B,50,32)

    a_exec = a0_full[:, :exec_steps, :exec_dim].contiguous()
    s1, rewards_seq = env.step_chunk(s0, a_exec)  # (B,10)

    response_mask = torch.ones((B, exec_steps), device=device, dtype=torch.bool)

    batch = TensorDict(
        {
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

            "rewards": rewards_seq.to(torch.float32),   # (B,10) ORIGINAL
            "response_mask": response_mask,
        },
        batch_size=[B],
        device=device,
    )

    dbg = {
        "reward_max_mean": rewards_seq.max(dim=-1).values.mean().item(),
        "reward_seq_mean": rewards_seq.mean().item(),
        "s0_norm": s0[:, :exec_dim].pow(2).sum(dim=-1).sqrt().mean().item(),
        "s1_norm": s1[:, :exec_dim].pow(2).sum(dim=-1).sqrt().mean().item(),
    }
    return batch, dbg


def moving_average(x, w=10):
    if len(x) < 2:
        return x
    w = max(1, int(w))
    out = []
    for i in range(len(x)):
        j0 = max(0, i - w + 1)
        out.append(sum(x[j0:i+1]) / (i - j0 + 1))
    return out


def plot_series(series, out_path, smooth=1):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    x = list(range(1, len(series) + 1))
    y = series
    if smooth and smooth > 1:
        y = moving_average(y, w=smooth)
    plt.figure()
    plt.plot(x, y)
    plt.xlabel("update step")
    plt.ylabel(Path(out_path).stem)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


@torch.no_grad()
def rollout_and_render(policy, exec_steps, exec_dim, out_dir,
                       episode_chunks=30,
                       target_mode="fixed", target_range=1.0,
                       stop_eps=0.05,
                       noise_std=0.01,
                       make_gif=True):
    device = next(policy.parameters()).device
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if target_mode == "fixed":
        g = torch.zeros((1, 2), device=device)
    elif target_mode == "random":
        g = torch.empty((1, 2), device=device).uniform_(-target_range, target_range)
    else:
        raise ValueError("target_mode must be 'fixed' or 'random'")

    s32 = torch.zeros((1, 32), device=device, dtype=torch.float32)
    s32[:, :2] = torch.empty((1, 2), device=device).uniform_(-1.0, 1.0)

    positions = [s32[0, :2].detach().cpu().clone()]
    chunk_start_indices = []

    for chunk in range(episode_chunks):
        chunk_start_indices.append(len(positions) - 1)

        obs = {
            "images": torch.zeros((1, 1, 3, 224, 224), device=device, dtype=torch.float32),
            "image_masks": torch.ones((1, 1), device=device, dtype=torch.bool),
            "lang_tokens": torch.zeros((1, 1), device=device, dtype=torch.long),
            "lang_masks": torch.ones((1, 1), device=device, dtype=torch.bool),
            "states": s32,
        }

        sf = policy.sac_forward_state_features(obs)
        a_full, _logp = policy.sac_forward_actor(sf)  # (1,50,32)
        a_exec = a_full[:, :exec_steps, :2]          # (1,10,2)

        p = s32[:, :2]
        for k in range(exec_steps):
            p = p + a_exec[:, k, :] + noise_std * torch.randn_like(p)
            s32[:, :2] = p
            positions.append(p[0].detach().cpu().clone())

        if torch.norm(p - g, dim=-1).item() < stop_eps:
            break

    pos = torch.stack(positions, dim=0).numpy()
    gx, gy = float(g[0, 0].cpu()), float(g[0, 1].cpu())

    png_path = out_dir / "trajectory.png"
    plt.figure()
    plt.plot(pos[:, 0], pos[:, 1], linewidth=1.0, alpha=0.35)

    total_micro_steps = len(pos) - 1
    chunks_drawn = 0
    for cs in chunk_start_indices:
        ce = min(cs + exec_steps, total_micro_steps)
        if ce <= cs:
            continue

        if cs + 1 <= ce:
            x = pos[cs:cs+2, 0]
            y = pos[cs:cs+2, 1]
            plt.plot(x, y, linewidth=2.5, marker="o", markersize=3)
            plt.gca().lines[-1].set_color("orange")

        if cs + 1 < ce:
            x = pos[cs+1:ce+1, 0]
            y = pos[cs+1:ce+1, 1]
            plt.plot(x, y, linewidth=1.8, marker="o", markersize=2)

        chunks_drawn += 1

    plt.scatter([pos[0, 0]], [pos[0, 1]], marker="s", s=60)
    plt.scatter([gx], [gy], marker="*", s=160)
    plt.title(f"Trajectory (micro-steps={len(pos)-1}, chunks={chunks_drawn}, exec_steps={exec_steps})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(png_path, dpi=160)
    plt.close()

    if make_gif:
        gif_path = out_dir / "trajectory.gif"
        frames = []
        stride = max(1, (len(pos) // 80))

        for end_i in range(1, len(pos) + 1, stride):
            plt.figure()
            plt.plot(pos[:, 0], pos[:, 1], linewidth=1.0, alpha=0.15)
            end_step = end_i - 1

            for cs in chunk_start_indices:
                if cs >= end_step:
                    break
                ce = min(cs + exec_steps, end_step)
                if ce <= cs:
                    continue
                if cs + 1 <= ce:
                    x = pos[cs:cs+2, 0]
                    y = pos[cs:cs+2, 1]
                    plt.plot(x, y, linewidth=2.5, marker="o", markersize=3)
                    plt.gca().lines[-1].set_color("orange")
                if cs + 1 < ce:
                    x = pos[cs+1:ce+1, 0]
                    y = pos[cs+1:ce+1, 1]
                    plt.plot(x, y, linewidth=1.8, marker="o", markersize=2)

            plt.scatter([pos[0, 0]], [pos[0, 1]], marker="s", s=60)
            plt.scatter([gx], [gy], marker="*", s=160)
            plt.title(f"t={end_step}")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.axis("equal")
            plt.grid(True)
            plt.tight_layout()

            frame_path = out_dir / f"_frame_{end_i:04d}.png"
            plt.savefig(frame_path, dpi=120)
            plt.close()
            frames.append(Image.open(frame_path))

        if frames:
            frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=80, loop=0)

        for pth in out_dir.glob("_frame_*.png"):
            try:
                pth.unlink()
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--micro_bs", type=int, default=128)
    parser.add_argument("--exec_steps", type=int, default=10)
    parser.add_argument("--exec_dim", type=int, default=7)
    parser.add_argument("--gamma_micro", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smooth", type=int, default=10)
    parser.add_argument("--out_dir", type=str, default="toy_plots_original")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device_type = get_device_name()
    if device_type == "cuda":
        device = torch.device(f"cuda:{get_device_id()}")
    else:
        device = torch.device(device_type)

    state_norm_stats = {"mean": [0.0] * 32, "std": [1.0] * 32}
    action_norm_stats = {"mean": [0.0] * 32, "std": [1.0] * 32}
    pi0_cfg = PI0TorchConfig(
        state_norm_stats=state_norm_stats,
        action_norm_stats=action_norm_stats,
        pi05_enabled=False,
        sac_enable=True,
        double_q=True,
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

    # ORIGINAL: only discount once at chunk-level
    gamma_eff = float(args.gamma_micro)

    cfg = OmegaConf.create(
        {
            "sac": {
                "gamma": gamma_eff,
                "tau": 0.005,
                "auto_entropy": False,
                "initial_alpha": 0.2,
            },
            "replay_pool_capacity": 2000,
            "replay_pool_save_dir": f"/tmp/toy_replay_pool_original_seed{args.seed}",
            "replay_pool_save_interval": 10_000_000,
            "ppo_micro_batch_size_per_gpu": int(args.micro_bs),
            "grad_clip": 1.0,
            "critic_warmup_steps": 0,
        }
    )

    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    actor = PI0RobDataParallelPPOActor(cfg, policy, optimizer)

    env = PushBallToyEnv(exec_steps=args.exec_steps, exec_dim=args.exec_dim)

    metrics_hist = []

    for step in range(1, args.updates + 1):
        batch, dbg = collect_batch_original(
            policy=policy,
            env=env,
            B=args.batch_size,
            exec_steps=args.exec_steps,
            exec_dim=args.exec_dim,
            device=device,
        )

        data = DataProto(batch=batch, meta_info={"global_steps": step})
        metrics = actor.update_policy(data)

        # debug (same as aligned)
        with torch.no_grad():
            s0 = batch["s0.states"]
            s0_obs = make_dummy_obs(s0, device)
            sf = policy.sac_forward_state_features(s0_obs)
            a_full, logp = policy.sac_forward_actor(sf)
            a_exec = a_full[:, : args.exec_steps, : args.exec_dim]
            target = (-s0[:, : args.exec_dim])[:, None, :].expand_as(a_exec)
            dist2 = (a_exec - target).pow(2).mean().item()

            q_pi = policy.sac_forward_critic(
                {"full_action": a_full},
                sf,
                use_target_network=False,
                method="min",
                requires_grad=False,
            ).mean().item()

        metrics["debug/dist2_to_target"] = dist2
        metrics["debug/Q_pi"] = q_pi
        metrics["debug/logp_mean_eval"] = logp.mean().item()
        metrics["debug/reward_max_mean"] = dbg["reward_max_mean"]
        metrics["debug/s0_norm"] = dbg["s0_norm"]
        metrics["debug/s1_norm"] = dbg["s1_norm"]

        metrics_hist.append(metrics)

        if step % 10 == 0 or step == 1:
            print(
                f"[step {step:04d}] "
                f"critic_loss={metrics['critic/loss']:.4f} "
                f"actor_loss={metrics['actor/loss']:.4f} "
                f"logp={metrics['actor/logprob_mean']:.3f} "
                f"Q_pi={metrics['debug/Q_pi']:.3f} "
                f"dist2={metrics['debug/dist2_to_target']:.3f} "
                f"Rmax={metrics['debug/reward_max_mean']:.3f}"
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

    plot_series(get_series("critic/loss"), out_dir / "critic_loss.png", smooth=args.smooth)
    plot_series(get_series("actor/loss"), out_dir / "actor_loss.png", smooth=args.smooth)
    plot_series(get_series("actor/logprob_mean"), out_dir / "logp_mean.png", smooth=args.smooth)
    plot_series(get_series("debug/Q_pi"), out_dir / "Q_pi.png", smooth=args.smooth)
    plot_series(get_series("debug/dist2_to_target"), out_dir / "dist2_to_target.png", smooth=args.smooth)
    plot_series(get_series("reward/mean"), out_dir / "reward_mean.png", smooth=args.smooth)
    plot_series(get_series("critic/grad_norm"), out_dir / "critic_grad_norm.png", smooth=args.smooth)
    plot_series(get_series("actor/grad_norm"), out_dir / "actor_grad_norm.png", smooth=args.smooth)
    plot_series(get_series("debug/s0_norm"), out_dir / "s0_norm.png", smooth=args.smooth)
    plot_series(get_series("debug/s1_norm"), out_dir / "s1_norm.png", smooth=args.smooth)

    base_out = out_dir / "renderings"
    for i in range(8):
        rollout_and_render(
            policy=policy,
            exec_steps=args.exec_steps,
            exec_dim=args.exec_dim,
            out_dir=base_out / f"traj_{i:02d}",
            episode_chunks=30,
            target_mode="fixed",
            target_range=1.0,
            stop_eps=0.05,
            noise_std=0.01,
            make_gif=True,
        )

    print(f"Saved plots to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
