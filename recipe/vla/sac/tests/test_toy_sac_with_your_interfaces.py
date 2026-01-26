# recipe/vla/sac/tests/test_toy_sac_with_your_interfaces.py
# -*- coding: utf-8 -*-
"""
Toy SAC test reusing YOUR interfaces:
- Reuse PI0ForActionPrediction.sac_forward_state_features / sac_forward_actor / sac_forward_critic
- Reuse your actor_loss / critic_loss / logprob approximation (flow one-step gaussian)
- Build a simple MDP ("push ball to origin") with micro-step discounting

Key alignments:
1) policy outputs (B, 50, 32), env executes ONLY (first 10 steps, first 7 dims)
   -> we force unused steps/dims to zeros in sampling and denoise_step
2) prefix features are constant zeros (no vision/lang noise)
3) reward: discounted sum/mean over 10 micro-steps (gamma_micro applied per micro-step)
4) bootstrap: since your SAC uses single gamma, we set sac.gamma = gamma_micro ** chunk_len (gamma_eff)
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

# headless plot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from omegaconf import OmegaConf

from verl.utils.device import get_device_name, get_device_id
from verl.protocol import DataProto

from recipe.vla.models.pi0_torch.configuration_pi0_torch import PI0TorchConfig
from recipe.vla.models.pi0_torch.modeling_pi0_torch import PI0ForActionPrediction
from recipe.vla.sac.sac_actor import PI0RobDataParallelPPOActor


# ----------------------------
# Fake PI0Model stub
# ----------------------------
class FakePI0Model(nn.Module):
    """
    Minimal stub that satisfies PI0ForActionPrediction's SAC path:
      - embed_prefix(...) -> (prefix_embs, prefix_pad_masks, prefix_att_masks)
      - sample_noise(shape, device) -> x0
      - denoise_step(states, prefix_pad_masks, past_key_values, x_t, t) -> v_t
    """

    def __init__(
        self,
        state_dim: int = 32,
        n_action_steps: int = 50,
        max_action_dim: int = 32,
        exec_steps: int = 10,
        exec_dim: int = 7,
        num_steps: int = 8,   # flow integration steps (K)
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
        self.use_cache = False  # not used in our stub

        self.prefix_len = prefix_len
        self.prefix_dim = prefix_dim

        # Make conditioning "obviously from state":
        # v depends on (x_t[:10,:7], Linear(state)[:7], t)
        self.state_proj = nn.Linear(state_dim, exec_dim)
        self.v_mlp = nn.Sequential(
            nn.Linear(exec_dim + exec_dim + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, exec_dim),
        )

    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks):
        # images is a list of tensors after unbind(dim=1) in your code.
        # We ignore all content and return constant zeros.
        # Need shapes:
        #   prefix_embs: (B, prefix_len, 2048)
        #   prefix_pad_masks: (B, prefix_len) bool
        #   prefix_att_masks: (B, prefix_len) bool
        B = images[0].shape[0]
        device = images[0].device
        prefix_embs = torch.zeros((B, self.prefix_len, self.prefix_dim), device=device, dtype=torch.float32)
        prefix_pad_masks = torch.ones((B, self.prefix_len), device=device, dtype=torch.bool)
        prefix_att_masks = torch.ones((B, self.prefix_len), device=device, dtype=torch.bool)
        return (prefix_embs, prefix_pad_masks, prefix_att_masks)

    def sample_noise(self, actions_shape, device):
        # actions_shape is (B, 50, 32)
        # We ONLY add noise to executed region (first 10 steps, first 7 dims).
        x = torch.zeros(actions_shape, device=device, dtype=torch.float32)
        x[:, : self.exec_steps, : self.exec_dim] = torch.randn(
            (actions_shape[0], self.exec_steps, self.exec_dim), device=device, dtype=torch.float32
        )
        return x

    def denoise_step(self, states, prefix_pad_masks, past_key_values, x_t, t):
        # states: (B, 32)
        # x_t: (B, 50, 32)
        # t: (B,) float
        B = states.shape[0]
        device = states.device

        # only operate on executed slice
        x_exec = x_t[:, : self.exec_steps, : self.exec_dim]  # (B,10,7)
        s_cond = self.state_proj(states)                      # (B,7)
        s_cond = s_cond[:, None, :].expand(B, self.exec_steps, self.exec_dim)  # (B,10,7)
        t_cond = t[:, None, None].expand(B, self.exec_steps, 1)                # (B,10,1)

        inp = torch.cat([x_exec, s_cond, t_cond], dim=-1)  # (B,10,15)
        v_exec = self.v_mlp(inp.reshape(B * self.exec_steps, -1)).reshape(B, self.exec_steps, self.exec_dim)

        # fill full v with zeros outside executed region
        v = torch.zeros((B, self.n_action_steps, self.max_action_dim), device=device, dtype=torch.float32)
        v[:, : self.exec_steps, : self.exec_dim] = v_exec
        return v


# ----------------------------
# Toy env: "push ball to origin"
# ----------------------------
class PushBallToyEnv:
    """
    State is 7-dim (embedded into 32 with padding zeros).
    Each micro-step: s <- s + a + noise
    Reward: -||s||^2 - action_cost*||a||^2
    """

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
        """
        s32: (B,32)
        a_seq_10x7: (B,10,7) executed micro-actions
        returns:
          s32_next: (B,32)
          rewards_seq: (B,10)
        """
        B = s32.shape[0]
        device = s32.device
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
    """
    Build s dict keys required by your sac_forward_state_features():
      images: (B, 1, 3, 224, 224)
      image_masks: (B, 1)
      lang_tokens: (B, 1)
      lang_masks: (B, 1)
      states: (B, 32)
    """
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
def collect_batch(policy: PI0ForActionPrediction, env: PushBallToyEnv, B: int, exec_steps: int, exec_dim: int,
                 gamma_micro: float, device):
    """
    Create one batch of transitions in YOUR expected TensorDict format.
    We keep rewards as (B,10) to match your code, but we fill it with reward_scalar repeated,
    because your current algorithm uses rewards.max(dim=-1).
    """
    s0 = env.reset(B, device=device)
    s0_obs = make_dummy_obs(s0, device)

    # use your own interface to sample actions (B,50,32)
    state_features = policy.sac_forward_state_features(s0_obs)
    a0_full, _logp0 = policy.sac_forward_actor(state_features)  # (B,50,32), (B,)

    # executed part only
    a_exec = a0_full[:, :exec_steps, :exec_dim].contiguous()  # (B,10,7)
    s1 = None
    rewards_seq = None
    s1, rewards_seq = env.step_chunk(s0, a_exec)              # rewards_seq (B,10)

    # discounted reward per micro-step (you can choose sum or mean)
    discounts = (gamma_micro ** torch.arange(exec_steps, device=device, dtype=rewards_seq.dtype))[None, :]  # (1,10)
    reward_disc_sum = (rewards_seq * discounts).sum(dim=-1)   # (B,)
    reward_scalar = reward_disc_sum / float(exec_steps)       # mean; if you want sum, remove /exec_steps

    # IMPORTANT: your algorithm currently uses rewards.max(dim=-1)
    # To make it consistent WITHOUT changing your main code, we store the scalar repeated in all steps.
    rewards_for_algo = reward_scalar[:, None].expand(B, exec_steps).contiguous()  # (B,10)

    response_mask = torch.ones((B, exec_steps), device=device, dtype=torch.bool)

    # build tensordict keys used by update_policy.select([...])
    batch = TensorDict(
        {
            "a0.full_action": a0_full.to(torch.float32),                      # (B,50,32)
            "a1.full_action": torch.zeros_like(a0_full, dtype=torch.float32), # not used, but required by select
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

            "rewards": rewards_for_algo.to(torch.float32),        # (B,10)
            "response_mask": response_mask,                       # (B,10)
        },
        batch_size=[B],
        device=device,
    )
    # also return some debug info for plotting
    debug = {
        "reward_scalar_mean": reward_scalar.mean().item(),
        "rewards_seq_mean": rewards_seq.mean().item(),
        "s0_norm": s0[:, :exec_dim].pow(2).sum(dim=-1).sqrt().mean().item(),
        "s1_norm": s1[:, :exec_dim].pow(2).sum(dim=-1).sqrt().mean().item(),
    }
    return batch, debug


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

from pathlib import Path
import torch
import matplotlib.pyplot as plt
from PIL import Image

@torch.no_grad()
def rollout_and_render(policy, exec_steps, exec_dim, out_dir,
                       episode_chunks=30,
                       target_mode="fixed", target_range=1.0,
                       stop_eps=0.05,
                       noise_std=0.01,
                       make_gif=True):
    """
    Visualize a single episode.
    - state[:2] is 2D position (x,y)
    - action[:2] is push delta in plane
    Highlight:
      - the 1st micro-step of EACH chunk in orange
      - the remaining (exec_steps-1) micro-steps in default color
    """
    device = next(policy.parameters()).device
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- target ----
    if target_mode == "fixed":
        g = torch.zeros((1, 2), device=device)
    elif target_mode == "random":
        g = torch.empty((1, 2), device=device).uniform_(-target_range, target_range)
    else:
        raise ValueError("target_mode must be 'fixed' or 'random'")

    # ---- reset ----
    s32 = torch.zeros((1, 32), device=device, dtype=torch.float32)
    s32[:, :2] = torch.empty((1, 2), device=device).uniform_(-1.0, 1.0)

    # For plotting: store positions and also chunk boundaries
    # positions: list of (x,y) at each micro-step (including start)
    positions = [s32[0, :2].detach().cpu().clone()]
    # chunk_start_indices: index in positions list where each chunk starts (start point index)
    chunk_start_indices = []

    for chunk in range(episode_chunks):
        chunk_start_indices.append(len(positions) - 1)  # the index of start pos of this chunk

        obs = {
            "images": torch.zeros((1, 1, 3, 224, 224), device=device, dtype=torch.float32),
            "image_masks": torch.ones((1, 1), device=device, dtype=torch.bool),
            "lang_tokens": torch.zeros((1, 1), device=device, dtype=torch.long),
            "lang_masks": torch.ones((1, 1), device=device, dtype=torch.bool),
            "states": s32,
        }

        sf = policy.sac_forward_state_features(obs)
        a_full, _logp = policy.sac_forward_actor(sf)  # (1,50,32)

        # execute only first exec_steps, first 2 dims for 2D push
        a_exec = a_full[:, :exec_steps, :2]  # (1,10,2)

        p = s32[:, :2]
        for k in range(exec_steps):
            p = p + a_exec[:, k, :] + noise_std * torch.randn_like(p)
            s32[:, :2] = p
            positions.append(p[0].detach().cpu().clone())

        if torch.norm(p - g, dim=-1).item() < stop_eps:
            break

    # ---- Plot with highlighting ----
    pos = torch.stack(positions, dim=0).numpy()  # (T,2)
    gx, gy = float(g[0, 0].cpu()), float(g[0, 1].cpu())

    png_path = out_dir / "trajectory.png"
    plt.figure()

    # Draw full trajectory faintly as background (optional but helpful)
    plt.plot(pos[:, 0], pos[:, 1], linewidth=1.0, alpha=0.35)

    # Now draw per-chunk segments with highlight
    total_micro_steps = len(pos) - 1
    chunks_drawn = 0

    for cs in chunk_start_indices:
        # positions indices:
        # cs is chunk start point
        # the chunk ends at cs + exec_steps (but clip to end)
        ce = min(cs + exec_steps, total_micro_steps)
        # Need points from cs to ce (inclusive end point index = ce)
        # segment points index range: [cs, ce]
        if ce <= cs:
            continue

        # First micro-step segment: [cs -> cs+1] in orange if exists
        if cs + 1 <= ce:
            x = pos[cs:cs+2, 0]
            y = pos[cs:cs+2, 1]
            plt.plot(x, y, linewidth=2.5, marker="o", markersize=3)  # default color for line+marker
            # recolor that just-plotted line to orange:
            plt.gca().lines[-1].set_color("orange")

        # Remaining micro-steps: [cs+1 -> ce]
        if cs + 1 < ce:
            x = pos[cs+1:ce+1, 0]
            y = pos[cs+1:ce+1, 1]
            plt.plot(x, y, linewidth=1.8, marker="o", markersize=2)

        chunks_drawn += 1

    # Start/target markers
    plt.scatter([pos[0, 0]], [pos[0, 1]], marker="s", s=60)  # start
    plt.scatter([gx], [gy], marker="*", s=160)               # target

    plt.title(f"Toy PushBall Trajectory (micro-steps={len(pos)-1}, chunks={chunks_drawn}, exec_steps={exec_steps})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(png_path, dpi=160)
    plt.close()

    print(f"[render] saved: {png_path}")

    # ---- Optional GIF ----
    if make_gif:
        gif_path = out_dir / "trajectory.gif"
        frames = []
        stride = max(1, (len(pos) // 80))

        for end_i in range(1, len(pos)+1, stride):
            plt.figure()
            plt.plot(pos[:, 0], pos[:, 1], linewidth=1.0, alpha=0.15)

            # draw chunks up to end_i-1 micro-step
            # convert end_i to micro-step count
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
            plt.title(f"t={end_step} (chunks={chunks_drawn})")
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
            print(f"[render] saved: {gif_path}")

        # cleanup frames
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
    parser.add_argument("--out_dir", type=str, default="toy_plots")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device_type = get_device_name()
    if device_type == "cuda":
        device = torch.device(f"cuda:{get_device_id()}")
    else:
        device = torch.device(device_type)

    # -------- build your PI0 policy with fake model --------
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

    # IMPORTANT: avoid freeze_vision_tower() touching real backbone
    policy.freeze_vision_tower = lambda: None

    # IMPORTANT: avoid KV-cache build (we don't have paligemma in toy)
    policy._build_kv_cache_from_prefix = lambda prefix_features: None

    # -------- build your SAC actor (algorithm layer) --------
    # Your current SAC loss uses single gamma; for micro-step discounting over exec_steps,
    # use gamma_eff = gamma_micro ** exec_steps
    gamma_eff = float(args.gamma_micro) ** int(args.exec_steps)

    cfg = OmegaConf.create(
        {
            "sac": {
                "gamma": gamma_eff,
                "tau": 0.005,
                "auto_entropy": False,
                "initial_alpha": 0.2,
            },
            "replay_pool_capacity": 2000,
            "replay_pool_save_dir": "/tmp/toy_replay_pool",
            "replay_pool_save_interval": 10_000_000,
            "ppo_micro_batch_size_per_gpu": int(args.micro_bs),
            "grad_clip": 1.0,
            "critic_warmup_steps": 0,
        }
    )

    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    actor = PI0RobDataParallelPPOActor(cfg, policy, optimizer)

    env = PushBallToyEnv(exec_steps=args.exec_steps, exec_dim=args.exec_dim)

    # -------- training loop --------
    metrics_hist = []
    debug_hist = []

    for step in range(1, args.updates + 1):
        batch, dbg = collect_batch(
            policy=policy,
            env=env,
            B=args.batch_size,
            exec_steps=args.exec_steps,
            exec_dim=args.exec_dim,
            gamma_micro=args.gamma_micro,
            device=device,
        )

        data = DataProto(batch=batch, meta_info={"global_steps": step})
        metrics = actor.update_policy(data)  # your code returns dict

        # extra debug: evaluate policy action quality vs "target mu = -state"
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
            )
            q_pi = q_pi.mean().item()

        metrics["debug/dist2_to_target"] = dist2
        metrics["debug/Q_pi"] = q_pi
        metrics["debug/logp_mean_eval"] = logp.mean().item()
        metrics["debug/reward_scalar_mean"] = dbg["reward_scalar_mean"]
        metrics["debug/s0_norm"] = dbg["s0_norm"]
        metrics["debug/s1_norm"] = dbg["s1_norm"]

        metrics_hist.append(metrics)
        debug_hist.append(dbg)

        if step % 10 == 0 or step == 1:
            print(
                f"[step {step:04d}] "
                f"critic_loss={metrics['critic/loss']:.4f} "
                f"actor_loss={metrics['actor/loss']:.4f} "
                f"logp={metrics['actor/logprob_mean']:.3f} "
                f"Q_pi={metrics['debug/Q_pi']:.3f} "
                f"dist2={metrics['debug/dist2_to_target']:.3f} "
                f"R={metrics['debug/reward_scalar_mean']:.3f}"
            )

    # -------- save & plot --------
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # save jsonl
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

    print(f"Saved plots to: {out_dir.resolve()}")
    # -------- render a test episode --------
    base_out = out_dir / "renderings"

    for i in range(8):
        rollout_and_render(
            policy=policy,
            exec_steps=args.exec_steps,
            exec_dim=args.exec_dim,
            out_dir=base_out / f"traj_{i:02d}",   # ✅ 每次不同目录
            episode_chunks=30,
            target_mode="fixed",
            target_range=1.0,
            stop_eps=0.05,
            noise_std=0.01,
            make_gif=True,
        )



if __name__ == "__main__":
    main()
