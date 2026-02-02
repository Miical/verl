import torch

def pad_last_dim_to(x: torch.Tensor, target_size: int, pad_value: float = 0.0) -> torch.Tensor:
    """
    Pads the last dimension of the tensor `x` to `target_size` with `pad_value`.
    If the last dimension is already greater than or equal to `target_size`, returns `x` unchanged.
    
    Args:
        x (torch.Tensor): Input tensor.
        target_size (int): Target size for the last dimension.
        pad_value (float): Value to use for padding.
        
    Returns:
        torch.Tensor: Padded tensor.
    """

    current_size = x.shape[-1]
    if current_size >= target_size:
        return x
    pad_size = target_size - current_size
    pad_shape = list(x.shape[:-1]) + [pad_size]
    pad_tensor = torch.full(pad_shape, pad_value, device=x.device, dtype=x.dtype)
    return torch.cat([x, pad_tensor], dim=-1)