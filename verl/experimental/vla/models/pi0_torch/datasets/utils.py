import torch

def pad_dim_to(x: torch.Tensor, dim: int, target_size: int, pad_value: float = 0.0):
    """
    Pad (or truncate) tensor x along dimension `dim` to `target_size`.

    Args:
        x: input tensor
        dim: which dimension to pad
        target_size: target size along dim
        pad_value: value to pad with

    Returns:
        Tensor with same ndim as x, but size[dim] == target_size
    """
    cur = x.size(dim)

    # truncate if too long
    if cur >= target_size:
        idx = [slice(None)] * x.dim()
        idx[dim] = slice(0, target_size)
        return x[tuple(idx)]

    # pad if too short
    pad_shape = list(x.shape)
    pad_shape[dim] = target_size - cur

    pad = x.new_full(pad_shape, pad_value)

    return torch.cat([x, pad], dim=dim)


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

    return pad_dim_to(x, dim=-1, target_size=target_size, pad_value=pad_value)