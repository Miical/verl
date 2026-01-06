import torch


def compute_discounted_returns(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    """
    Compute discounted returns for a batch of reward sequences.

    Args:
        rewards: Tensor of shape (batch_size, num_steps) or
            (batch_size, num_steps, chunk_size) containing rewards.
        gamma: Discount factor.
    """

    if rewards.ndim == 3:
        batch_size, num_steps, chunk_size = rewards.shape
        flat_rewards = rewards.reshape(batch_size, num_steps * chunk_size)
        flat_returns = torch.zeros_like(flat_rewards)
        running = torch.zeros(batch_size, device=rewards.device, dtype=rewards.dtype)
        for t in range(flat_rewards.shape[1] - 1, -1, -1):
            running = flat_rewards[:, t] + gamma * running
            flat_returns[:, t] = running
        return flat_returns.reshape(batch_size, num_steps, chunk_size)
    if rewards.ndim == 2:
        batch_size, num_steps = rewards.shape
        returns = torch.zeros_like(rewards)
        running = torch.zeros(batch_size, device=rewards.device, dtype=rewards.dtype)
        for t in range(num_steps - 1, -1, -1):
            running = rewards[:, t] + gamma * running
            returns[:, t] = running
        return returns
    return rewards
