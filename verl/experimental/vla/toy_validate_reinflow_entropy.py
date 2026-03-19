#!/usr/bin/env python3
"""Minimal toy validation for ReinFlow-lite path likelihood / entropy.

This script validates two things:
1) Single-step Gaussian policy: Monte Carlo -E[log p(a|s)] matches analytic entropy.
2) Multi-step Gaussian Markov process: Monte Carlo -E[log p(path|s)] matches
   the sum of per-step conditional entropies.

It is intentionally self-contained and does not require loading the full robot model.
"""
from __future__ import annotations

import sys
import math
from dataclasses import dataclass

import torch
import torch.nn as nn


def diag_gaussian_log_prob_sum(sample: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    std_safe = std.clamp_min(1e-6)
    log_prob = -0.5 * (((sample - mean) / std_safe) ** 2 + 2.0 * torch.log(std_safe) + math.log(2.0 * math.pi))
    return log_prob.sum(dim=-1)


def diag_gaussian_entropy_sum(std: torch.Tensor) -> torch.Tensor:
    std_safe = std.clamp_min(1e-6)
    ent = 0.5 * (1.0 + math.log(2.0 * math.pi)) + torch.log(std_safe)
    return ent.sum(dim=-1)


class TinySigmaHead(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, sigma_min: float, sigma_max: float, sigma_init: float):
        super().__init__()
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.net = nn.Sequential(
            nn.Linear(state_dim + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # bias init to target sigma_init in log-space
        init_log_sigma = math.log(max(sigma_init, sigma_min))
        log_sigma_min = math.log(sigma_min)
        log_sigma_max = math.log(sigma_max)
        ratio = (init_log_sigma - log_sigma_min) / max(log_sigma_max - log_sigma_min, 1e-6)
        ratio = min(max(ratio, 1e-4), 1.0 - 1e-4)
        init_bias = math.log(ratio / (1.0 - ratio))
        nn.init.zeros_(self.net[0].weight)
        nn.init.zeros_(self.net[0].bias)
        nn.init.zeros_(self.net[2].weight)
        self.net[2].bias.data.fill_(init_bias)

    def forward(self, state: torch.Tensor, latent: torch.Tensor, t_scalar: torch.Tensor) -> torch.Tensor:
        # state: [B, Ds], latent: [B, D], t_scalar: [B, 1]
        latent_std = latent.std(dim=-1, keepdim=True, unbiased=False)
        x = torch.cat([state, latent_std, t_scalar], dim=-1)
        gate = torch.sigmoid(self.net(x))
        log_sigma_min = math.log(self.sigma_min)
        log_sigma_max = math.log(self.sigma_max)
        log_sigma = log_sigma_min + (log_sigma_max - log_sigma_min) * gate
        return torch.exp(log_sigma)


@dataclass
class ToyStats:
    mc_nll: float
    analytic_entropy: float
    abs_error: float


class TinyStochasticFlow(nn.Module):
    def __init__(self, state_dim: int = 5, action_dim: int = 4, sigma_min: float = 1e-3, sigma_max: float = 0.4):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.mean_proj = nn.Linear(state_dim + action_dim + 1, action_dim)
        self.sigma_head = TinySigmaHead(state_dim, 32, sigma_min, sigma_max, sigma_init=0.07)
        nn.init.normal_(self.mean_proj.weight, std=0.1)
        nn.init.normal_(self.mean_proj.bias, std=0.05)

    def mean_fn(self, state: torch.Tensor, x_t: torch.Tensor, t_scalar: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([state, x_t, t_scalar], dim=-1)
        return self.mean_proj(inp)

    def sample_one_step(self, state: torch.Tensor, t_scalar: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = state.shape[0]
        x0 = torch.zeros(batch, self.action_dim, device=state.device, dtype=state.dtype)
        mean = self.mean_fn(state, x0, t_scalar)
        sigma = self.sigma_head(state, x0, t_scalar).expand_as(mean)
        sample = mean + sigma * torch.randn_like(mean)
        return sample, mean, sigma

    def sample_path(self, state: torch.Tensor, num_steps: int = 3) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        batch = state.shape[0]
        x_t = torch.randn(batch, self.action_dim, device=state.device, dtype=state.dtype)
        samples, means, sigmas = [], [], []
        timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=state.device, dtype=state.dtype)
        for k in range(num_steps):
            t_cur = timesteps[k].expand(batch, 1)
            delta = (timesteps[k] - timesteps[k + 1]).clamp_min(1e-6)
            mean = self.mean_fn(state, x_t, t_cur)
            sigma = self.sigma_head(state, x_t, t_cur).expand_as(mean) * torch.sqrt(delta)
            x_next = mean + sigma * torch.randn_like(mean)
            samples.append(x_next)
            means.append(mean)
            sigmas.append(sigma)
            x_t = x_next
        return samples, means, sigmas


def validate_single_step(device: str = "cpu") -> ToyStats:
    torch.manual_seed(0)
    model = TinyStochasticFlow().to(device)
    batch = 12000
    state = torch.randn(batch, model.state_dim, device=device)
    t = torch.full((batch, 1), 0.6, device=device)
    sample, mean, sigma = model.sample_one_step(state, t)
    logp = diag_gaussian_log_prob_sum(sample, mean, sigma)
    analytic_ent = diag_gaussian_entropy_sum(sigma).mean()
    mc_nll = (-logp).mean()
    return ToyStats(float(mc_nll.detach()), float(analytic_ent.detach()), float((mc_nll - analytic_ent).abs().detach()))


def validate_multi_step(device: str = "cpu") -> ToyStats:
    torch.manual_seed(0)
    model = TinyStochasticFlow().to(device)
    batch = 12000
    state = torch.randn(batch, model.state_dim, device=device)
    samples, means, sigmas = model.sample_path(state, num_steps=3)
    logp = torch.zeros(batch, device=device)
    ent = torch.zeros(batch, device=device)
    for sample, mean, sigma in zip(samples, means, sigmas, strict=False):
        logp = logp + diag_gaussian_log_prob_sum(sample, mean, sigma)
        ent = ent + diag_gaussian_entropy_sum(sigma)
    mc_nll = (-logp).mean()
    analytic_ent = ent.mean()
    return ToyStats(float(mc_nll.detach()), float(analytic_ent.detach()), float((mc_nll - analytic_ent).abs().detach()))


def validate_monotonicity(device: str = "cpu") -> list[tuple[float, float]]:
    torch.manual_seed(1)
    model = TinyStochasticFlow().to(device)
    batch = 4000
    state = torch.randn(batch, model.state_dim, device=device)
    x0 = torch.zeros(batch, model.action_dim, device=device)
    t = torch.full((batch, 1), 0.5, device=device)
    mean = model.mean_fn(state, x0, t)
    base_sigma = model.sigma_head(state, x0, t).expand_as(mean)
    scales = [0.5, 0.75, 1.0, 1.25, 1.5]
    results = []
    for scale in scales:
        sigma = base_sigma * scale
        sample = mean + sigma * torch.randn_like(mean)
        neg_logp = (-diag_gaussian_log_prob_sum(sample, mean, sigma)).mean().item()
        results.append((scale, neg_logp))
    return results


def main() -> None:
    device = "cpu"
    single = validate_single_step(device)
    multi = validate_multi_step(device)
    monotonic = validate_monotonicity(device)

    lines = [
        "=== ReinFlow-lite toy validation ===",
        f"Device: {device}",
        "Single-step Gaussian:",
        f"  Monte Carlo -E[log p] : {single.mc_nll:.6f}",
        f"  Analytic entropy      : {single.analytic_entropy:.6f}",
        f"  Abs error             : {single.abs_error:.6f}",
        "Multi-step path process:",
        f"  Monte Carlo -E[log p(path)] : {multi.mc_nll:.6f}",
        f"  Analytic path entropy      : {multi.analytic_entropy:.6f}",
        f"  Abs error                  : {multi.abs_error:.6f}",
        "Monotonicity sweep (scale, -E[log p]):",
    ]
    lines.extend([f"  {scale:.2f} -> {value:.6f}" for scale, value in monotonic])

    assert single.abs_error < 0.05, f"single-step entropy mismatch too large: {single.abs_error}"
    assert multi.abs_error < 0.08, f"multi-step entropy mismatch too large: {multi.abs_error}"
    vals = [v for _, v in monotonic]
    assert vals == sorted(vals), "neg log-likelihood should increase monotonically with sigma scale"
    lines.append("All toy validations passed.")
    sys.stdout.write("\n".join(lines) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
