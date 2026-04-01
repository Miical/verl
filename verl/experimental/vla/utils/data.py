import numpy as np
import torch
from verl.protocol import DataProto

def get_dataproto_from_prefix(data: DataProto, prefix: str, separator: str = "") -> DataProto:
    """Extract a sub-DataProto from a DataProto based on a given prefix.

    Args:
        data: The input DataProto containing various keys.
        prefix: The prefix string to filter keys.
        separator: Optional separator appended after prefix when matching keys.
    Returns:
        A DataProto containing tensor and non-tensor entries whose keys start
        with the specified prefix. The prefix is removed from the result keys.
    """

    match_prefix = prefix if not separator or prefix.endswith(separator) else f"{prefix}{separator}"
    prefix_length = len(match_prefix)
    tensor_batch = {}
    non_tensor_batch = {}
    for key in data.batch.keys():
        if key.startswith(match_prefix):
            new_key = key[prefix_length:]
            tensor_batch[new_key] = data.batch[key]
    for key in data.non_tensor_batch.keys():
        if key.startswith(match_prefix):
            new_key = key[prefix_length:]
            non_tensor_batch[new_key] = data.non_tensor_batch[key]
    return DataProto.from_dict(tensors=tensor_batch, non_tensors=non_tensor_batch, meta_info=data.meta_info)


def slice_dataproto_batch(data: DataProto, start: int, end: int) -> DataProto:
    return DataProto.from_dict(
        tensors={key: value[:, start:end] for key, value in data.batch.items()},
        meta_info=data.meta_info,
    )

def merge_nested_dicts_or_tuples(a: dict | tuple, b: dict | tuple) -> dict | tuple:
    """Merge two nested structures (dictionaries or tuples) by concatenating tensors
    along the first dimension.
    """

    if isinstance(a, dict) and isinstance(b, dict):
        merged = {}
        for key in a.keys():
            merged[key] = merge_nested_dicts_or_tuples(a[key], b[key])
        return merged
    elif isinstance(a, tuple) and isinstance(b, tuple):
        merged = []
        for item_a, item_b in zip(a, b, strict=False):
            merged.append(merge_nested_dicts_or_tuples(item_a, item_b))
        return tuple(merged)
    else:
        return torch.cat([a, b], dim=0)


def split_nested_dicts_or_tuples(data: dict | tuple, split_num: int) -> list[dict | tuple]:
    """Split a nested structure (dictionary or tuple) into smaller chunks along the first dimension."""

    if isinstance(data, torch.Tensor):
        split_tensors = torch.chunk(data, split_num, dim=0)
        return list(split_tensors)
    elif isinstance(data, dict):
        split_dicts = [dict() for _ in range(split_num)]
        for key, value in data.items():
            split_values = split_nested_dicts_or_tuples(value, split_num)
            for i in range(split_num):
                split_dicts[i][key] = split_values[i]
        return split_dicts
    elif isinstance(data, tuple):
        split_tuples = [list() for _ in range(split_num)]
        for item in data:
            split_items = split_nested_dicts_or_tuples(item, split_num)
            for i in range(split_num):
                split_tuples[i].append(split_items[i])
        return [tuple(split_tuple) for split_tuple in split_tuples]
    else:
        raise TypeError("Input data must be a torch.Tensor, dict, or tuple.")


def valid_mean(x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Compute the mean of tensor `x` over valid entries indicated by `valid` mask.

    Args:
        x: Tensor of shape (B, ...) containing values to average.
        valid: Tensor of shape (B,) indicating valid entries (1 for valid, 0 for invalid).

    Returns:
        Scalar tensor (mean over valid samples only)
    """
    x = x.squeeze(-1)
    valid_f = valid.float().to(x.device)
    denom = valid_f.sum().clamp_min(1.0)
    return (x * valid_f).sum() / denom


def stack_dataproto_with_padding(data_protos: list[DataProto], prefix: str) -> dict[str, torch.Tensor | np.ndarray]:
    """Stack a list of DataProto objects along the time dimension.

    If a field is missing from some steps, fill that step with zeros matching
    the first available value for the same field.
    """

    merged = {}

    tensor_keys = sorted({key for data in data_protos for key in (data.batch.keys() if data.batch is not None else [])})
    for key in tensor_keys:
        template = next(data.batch[key] for data in data_protos if data.batch is not None and key in data.batch.keys())
        per_step_values = []
        for data in data_protos:
            if data.batch is not None and key in data.batch.keys():
                per_step_values.append(data.batch[key])
            else:
                per_step_values.append(torch.zeros_like(template))
        merged[f"{prefix}.{key}"] = torch.stack(per_step_values, dim=1)

    non_tensor_keys = sorted({key for data in data_protos for key in data.non_tensor_batch.keys()})
    for key in non_tensor_keys:
        template = next(data.non_tensor_batch[key] for data in data_protos if key in data.non_tensor_batch)
        per_step_values = []
        for data in data_protos:
            if key in data.non_tensor_batch:
                per_step_values.append(data.non_tensor_batch[key])
            else:
                per_step_values.append(np.zeros_like(template))
        merged[f"{prefix}.{key}"] = np.stack(per_step_values, axis=1)

    return merged
