# recipe/vla/sac/tests/test_toy_sac_with_your_interfaces_earlydone.py
# -*- coding: utf-8 -*-
"""
Toy SAC test reusing YOUR interfaces (with early-done inside a chunk):

What changed vs your aligned version:
- env.step_chunk_earlydone() supports per-sample termination within the 10 micro-steps.
- response_mask becomes like:
    [T,T,T,T,T,T,F,F,F,F]  (if done at step 6)
- rewards after done are padded as 0 but masked out in our reward_scalar computation.
- we still store rewards as (B,10) by repeating reward_scalar, so your current code's
  `rewards.max(dim=-1)` behaves as intended (does NOT accidentally take padded zeros).

NOTE:
- Your current SAC actor code computes `valid = response_mask.any(-1)`.
  So earlydone samples remain valid as long as there is at least one True in response_mask.
"""

import json
import math
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
from PIL import Image

from omegaconf import OmegaConf

from verl.utils.device import get_device_name, get_device_id
from verl.protocol import DataProto


# ---------------------------------------------------------
# OPTIONAL: patch missing `onnx_ir` dependency if needed
# (your PI0 code imports: `from onnx_ir import Tensor`)
# ---------------------------------------------------------
try:
    import onnx_ir  # noqa: F401
except Exception:
    import sys
    import types
    m = types.ModuleType("onnx_ir")
    class _Tensor:  # minimal stub
        pass
    m.Tensor = _Tensor
    sys.modules["onnx_ir"] = m


from recipe.vla.models.pi0_torch.configuration_pi0_torch import PI0TorchConfig
from recipe.vla.models.pi0_torch.modeling_pi0_torch import PI0ForActionPrediction
from recipe.vla.sac.sac_actor import PI0RobDataParallelPPOActor


# ----------------------------
# Fake PI0Model stub (aligned)
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
        self.use_cache = False

        self.prefix_len = prefix_len
        self.prefix_dim = prefix_dim

        # Make conditioning clearly depend on state
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
        # aligned: constant zero prefix (no random vision/lang noise)
        prefix_embs = torch.zeros((B, self.prefix_len, self.prefix_dim), device=device, dtype=torch.float32)
        prefix_pad_masks = torch.ones((B, self.prefix_len), device=device, dtype=torch.bool)
        prefix_att_masks = torch.ones((B, self.prefix_len), device=device, dtype=torch.bool)
        return (prefix_embs, prefix_pad_masks, prefix_att_masks)

    def sample_noise(self, actions_shape, device):
        # aligned: only add noise to executed region (first 10 steps, first 7 dims)
        x = torch.zeros(actions_shape, device=device, dtype=torch.float32)
        x[:, : self.exec_steps, : self.exec_dim] = torch.randn(
            (actions_shape[0], self.exec_steps, self.exec_dim), device=device, dtype=torch.float32
        )
        return x

    def denoise_step(self, states, prefix_pad_masks, past_key_values, x_t, t):
        B = states.shape[0]
        device = states.device

        x_exec = x_t[:, : self.exec_steps, : self.exec_dim]  # (B,10,7)
        s_cond = self.state_proj(states)                      # (B,7)
        s_cond = s_cond[:, None, :].expand(B, self.exec_steps, self.exec_dim)  # (B,10,7)
        t_cond = t[:, None, None].expand(B, self.exec_steps, 1)                # (B,10,1)

        inp = torch.cat([x_exec, s_cond, t_cond], dim=-1)  # (B,10,15)
        v_exec = self.v_mlp(inp.reshape(B * self.exec_steps, -1)).reshape(B, self.exec_steps, self.exec_dim)

        v = torch.zeros((B, self.n_action_steps, self.max_action_dim), device=device, dtype=torch.float32)
        v[:, : self.exec_steps, : self.exec_dim] = v_exec
        return v


# ----------------------------
# Toy env: push state to origin (with early-done)
# ----------------------------
class PushBallToyEnv:
    """
    State: exec_dim-dim (embedded into 32 with padding zeros).
    micro-step dynamics:
      s <- s + a + noise    (only while alive)
      after done: absorbing (s stays, reward=0)

    Reward (alive steps only):
      r = -||s||^2 - action_cost*||a||^2
    Done:
      ||s|| < stop_eps
    """

    def __init__(self, exec_steps=10, exec_dim=7, noise_std=0.01, action_cost=0.01, stop_eps=0.05):
        self.exec_steps = exec_steps
        self.exec_dim = exec_dim
        self.noise_std = noise_std
        self.action_cost = action_cost
        self.stop_eps = stop_eps

    def reset(self, B, device):
        s = torch.empty((B, self.exec_dim), device=device).uniform_(-1.0, 1.0)
        s32 = torch.zeros((B, 32), device=device, dtype=torch.float32)
        s32[:, : self.exec_dim] = s
        return s32

    def step_chunk_earlydone(self, s32, a_seq_10x7):
        """
        s32: (B,32)
        a_seq_10x7: (B,10,7)

        returns:
          s32_next: (B,32)
          rewards_seq: (B,10)  (padded zeros after done)
          response_mask: (B,10) bool (True where step was actually executed)
          done_step: (B,) int, first done index (0..9) or 10 if never done in this chunk
        """
        B = s32.shape[0]
        device = s32.device
        s = s32[:, : self.exec_dim]

        alive = torch.ones((B,), device=device, dtype=torch.bool)
        rewards = []
        masks = []
        done_step = torch.full((B,), fill_value=self.exec_steps, device=device, dtype=torch.long)

        for k in range(self.exec_steps):
            masks.append(alive.clone())  # executed at this step?

            a = a_seq_10x7[:, k, :]  # (B,7)

            # only update alive samples
            noise = self.noise_std * torch.randn_like(s)
            delta = a + noise

            s_next = s + delta
            s = torch.where(alive[:, None], s_next, s)  # absorbing for non-alive

            # reward only for alive steps; padded 0 after done
            r = -(s.pow(2).sum(dim=-1)) - self.action_cost * (a.pow(2).sum(dim=-1))
            r = torch.where(alive, r, torch.zeros_like(r))
            rewards.append(r)

            # check done for those still alive
            is_done_now = alive & (torch.norm(s, dim=-1) < self.stop_eps)
            # record first done step
            done_step = torch.where(is_done_now & (done_step == self.exec_steps), torch.tensor(k, device=device), done_step)
            # update alive
            alive = alive & (~is_done_now)

        rewards_seq = torch.stack(rewards, dim=-1)                 # (B,10)
        response_mask = torch.stack(masks, dim=-1).to(torch.bool)  # (B,10)

        s32_next = torch.zeros_like(s32)
        s32_next[:, : self.exec_dim] = s
        return s32_next, rewards_seq, response_mask, done_step


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
def collect_batch_earlydone(policy: PI0ForActionPrediction,
                           env: PushBallToyEnv,
                           B: int,
                           exec_steps: int,
                           exec_dim: int,
                           gamma_micro: float,
                           device):
    """
    Build one batch in your expected TensorDict format.

    Key:
    - response_mask is per micro-step and can look like [T..T,F..F].
    - reward_scalar is computed only over mask=True steps, with per-step discount.
    - store rewards as (B,10) by repeating reward_scalar to keep your current
      `rewards.max(dim=-1)` behavior consistent.
    """
    s0 = env.reset(B, device=device)
    s0_obs = make_dummy_obs(s0, device)

    sf = policy.sac_forward_state_features(s0_obs)
    a0_full, _logp0 = policy.sac_forward_actor(sf)  # (B,50,32)

    a_exec = a0_full[:, :exec_steps, :exec_dim].contiguous()  # (B,10,7)

    s1, rewards_seq, response_mask, done_step = env.step_chunk_earlydone(s0, a_exec)

    # discounted reward only for valid micro-steps
    discounts = (gamma_micro ** torch.arange(exec_steps, device=device, dtype=rewards_seq.dtype))[None, :]  # (1,10)
    w = response_mask.to(rewards_seq.dtype) * discounts  # (B,10)
    R = (rewards_seq * w).sum(dim=-1)  # (B,)

    # mean over executed steps (optional, keep scale stable)
    denom = response_mask.to(rewards_seq.dtype).sum(dim=-1).clamp_min(1.0)  # (B,)
    reward_scalar = R / denom  # (B,)

    # IMPORTANT: match your current algo expectation (rewards is (B,10) and then max over dim=-1)
    rewards_for_algo = reward_scalar[:, None].expand(B, exec_steps).contiguous()

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

            "rewards": rewards_for_algo.to(torch.float32),   # (B,10), repeated scalar
            "response_mask": response_mask,                  # (B,10) with earlydone pattern
        },
        batch_size=[B],
        device=device,
    )

    dbg = {
        "reward_scalar_mean": reward_scalar.mean().item(),
        "reward_seq_mean_alive": (rewards_seq * response_mask.to(rewards_seq.dtype)).sum().item() / (response_mask.sum().clamp_min(1).item()),
        "done_ratio": (done_step < exec_steps).float().mean().item(),
        "eff_steps_mean": response_mask.float().sum(dim=-1).mean().item(),
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
                       stop_eps=0.05,
                       noise_std=0.01,
                       make_gif=True):
    """
    Render one episode.
    Highlight:
      - the 1st micro-step of EACH chunk in orange
      - remaining micro-steps in default color
    Early stop:
      - stop if ||state[:exec_dim]|| < stop_eps
    """
    device = next(policy.parameters()).device
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # reset
    s32 = torch.zeros((1, 32), device=device, dtype=torch.float32)
    s32[:, :exec_dim] = torch.empty((1, exec_dim), device=device).uniform_(-1.0, 1.0)

    positions = [s32[0, :2].detach().cpu().clone()]
    chunk_start_indices = []

    for chunk in range(episode_chunks):
        chunk_start_indices.append(len(positions) - 1)

        obs = make_dummy_obs(s32, device)
        sf = policy.sac_forward_state_features(obs)
        a_full, _logp = policy.sac_forward_actor(sf)  # (1,50,32)

        # execute only first exec_steps; plot only first 2 dims
        a_exec_2d = a_full[:, :exec_steps, :2]

        p2 = s32[:, :2]
        for k in range(exec_steps):
            p2 = p2 + a_exec_2d[:, k, :] + noise_std * torch.randn_like(p2)
            s32[:, :2] = p2
            positions.append(p2[0].detach().cpu().clone())

            # early stop uses full exec_dim norm (but only 2D is shown)
            if torch.norm(s32[:, :exec_dim], dim=-1).item() < stop_eps:
                break

        if torch.norm(s32[:, :exec_dim], dim=-1).item() < stop_eps:
            break

    pos = torch.stack(positions, dim=0).numpy()

    png_path = out_dir / "trajectory.png"
    plt.figure()
    plt.plot(pos[:, 0], pos[:, 1], linewidth=1.0, alpha=0.35)

    total_micro_steps = len(pos) - 1
    for cs in chunk_start_indices:
        ce = min(cs + exec_steps, total_micro_steps)
        if ce <= cs:
            continue

        # first step orange
        if cs + 1 <= ce:
            x = pos[cs:cs+2, 0]
            y = pos[cs:cs+2, 1]
            plt.plot(x, y, linewidth=2.5, marker="o", markersize=3)
            plt.gca().lines[-1].set_color("orange")

        # remaining
        if cs + 1 < ce:
            x = pos[cs+1:ce+1, 0]
            y = pos[cs+1:ce+1, 1]
            plt.plot(x, y, linewidth=1.8, marker="o", markersize=2)

    plt.scatter([pos[0, 0]], [pos[0, 1]], marker="s", s=60)   # start
    plt.scatter([0.0], [0.0], marker="*", s=160)              # target (origin)

    plt.title(f"Trajectory (micro-steps={len(pos)-1}, exec_steps={exec_steps})")
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
            plt.scatter([0.0], [0.0], marker="*", s=160)
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
    parser.add_argument("--updates", type=int, default=200)       # 训练轮数在这里
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--micro_bs", type=int, default=128)
    parser.add_argument("--exec_steps", type=int, default=10)
    parser.add_argument("--exec_dim", type=int, default=7)
    parser.add_argument("--gamma_micro", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smooth", type=int, default=10)
    parser.add_argument("--out_dir", type=str, default="toy_plots_earlydone")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device_type = get_device_name()
    if device_type == "cuda":
        device = torch.device(f"cuda:{get_device_id()}")
    else:
        device = torch.device(device_type)

    # ----- build policy -----
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

    # avoid touching real backbone / kv-cache
    policy.freeze_vision_tower = lambda: None
    policy._build_kv_cache_from_prefix = lambda prefix_features: None

    # ----- build SAC actor -----
    # still use chunk-level gamma_eff (same as your aligned version)
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
            "replay_pool_save_dir": f"/tmp/toy_replay_pool_earlydone_seed{args.seed}",
            "replay_pool_save_interval": 10_000_000,
            "ppo_micro_batch_size_per_gpu": int(args.micro_bs),
            "grad_clip": 1.0,
            "critic_warmup_steps": 0,
        }
    )

    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    actor = PI0RobDataParallelPPOActor(cfg, policy, optimizer)

    env = PushBallToyEnv(exec_steps=args.exec_steps, exec_dim=args.exec_dim, stop_eps=0.05)

    metrics_hist = []

    for step in range(1, args.updates + 1):
        batch, dbg = collect_batch_earlydone(
            policy=policy,
            env=env,
            B=args.batch_size,
            exec_steps=args.exec_steps,
            exec_dim=args.exec_dim,
            gamma_micro=args.gamma_micro,
            device=device,
        )

        data = DataProto(batch=batch, meta_info={"global_steps": step})
        metrics = actor.update_policy(data)

        # debug: dist2_to_target and Q_pi
        with torch.no_grad():
            s0 = batch["s0.states"]
            sf = policy.sac_forward_state_features(make_dummy_obs(s0, device))
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
        metrics["debug/reward_scalar_mean"] = dbg["reward_scalar_mean"]
        metrics["debug/done_ratio"] = dbg["done_ratio"]
        metrics["debug/eff_steps_mean"] = dbg["eff_steps_mean"]
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
                f"R={metrics['debug/reward_scalar_mean']:.3f} "
                f"done_ratio={metrics['debug/done_ratio']:.3f} "
                f"eff_steps={metrics['debug/eff_steps_mean']:.2f}"
            )

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
    plot_series(get_series("debug/done_ratio"), out_dir / "done_ratio.png", smooth=args.smooth)
    plot_series(get_series("debug/eff_steps_mean"), out_dir / "eff_steps_mean.png", smooth=args.smooth)
    plot_series(get_series("debug/s0_norm"), out_dir / "s0_norm.png", smooth=args.smooth)
    plot_series(get_series("debug/s1_norm"), out_dir / "s1_norm.png", smooth=args.smooth)

    # render 8 trajectories
    base_out = out_dir / "renderings"
    for i in range(8):
        rollout_and_render(
            policy=policy,
            exec_steps=args.exec_steps,
            exec_dim=args.exec_dim,
            out_dir=base_out / f"traj_{i:02d}",
            episode_chunks=30,
            stop_eps=0.05,
            noise_std=0.01,
            make_gif=True,
        )

    print(f"Saved plots to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
