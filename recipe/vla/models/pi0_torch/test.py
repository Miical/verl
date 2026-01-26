import torch
from torch.distributions import Normal

def logp_like_your_code(x_t, x_next, v_t, t, step_idx, K=4, noise_level=0.5):
    # shapes:
    # x_t, x_next, v_t: (B, T, A)
    # t, step_idx: (B,)
    B = x_t.shape[0]
    device = x_t.device

    dt_pos = 1.0 / float(K)
    t_b = t[:, None, None]                          # (B,1,1)
    dt_b = torch.full_like(t_b, dt_pos)             # (B,1,1)

    # ----- same sigma schedule -----
    t_grid_full = torch.arange(1.0, -dt_pos / 2, -dt_pos, dtype=torch.float32, device=device)  # len=K+1
    t_for_sigma = torch.where(t_grid_full == 1.0, t_grid_full[1], t_grid_full)
    sigmas = noise_level * torch.sqrt(t_grid_full / (1.0 - t_for_sigma).clamp_min(1e-6))
    sigmas = sigmas[:-1]                            # len=K

    sigma_i = sigmas[step_idx][:, None, None].clamp_min(1e-6)  # (B,1,1)

    # ----- same mean/std construction -----
    x0_pred = x_t - v_t * t_b
    x1_pred = x_t + v_t * (1.0 - t_b)

    x0_weight = torch.ones_like(t_b) - (t_b - dt_b)
    x1_weight = t_b - dt_b - (sigma_i ** 2) * dt_b / (2.0 * t_b.clamp_min(1e-6))

    mean = x0_pred * x0_weight + x1_pred * x1_weight
    std  = (dt_b.sqrt() * sigma_i).clamp_min(1e-6)

    dist = Normal(mean.float(), std.float())
    logp = dist.log_prob(x_next.float()).sum(dim=(1, 2))       # (B,)
    return mean, std, logp, sigmas

# ----- fake data (B=2,T=1,A=1) -----
K = 4
B = 2
x_t = torch.tensor([[[0.2]], [[-0.5]]], dtype=torch.float32)
v_t = torch.tensor([[[1.0]], [[0.3]]], dtype=torch.float32)
t = torch.full((B,), 1.0 / K, dtype=torch.float32)            # 0.25
step_idx = torch.full((B,), K - 1, dtype=torch.long)          # 3

# choose x_next (any value is ok; here mimic sampler update x_next = x_t - (1/K)*v_t)
x_next = x_t - (1.0 / K) * v_t

mean, std, logp, sigmas = logp_like_your_code(x_t, x_next, v_t, t, step_idx, K=K, noise_level=0.5)

print("sigmas:", sigmas.tolist())
print("mean :", mean.squeeze().tolist())
print("std  :", std.squeeze().tolist())
print("x_next:", x_next.squeeze().tolist())
print("logp :", logp.tolist())
