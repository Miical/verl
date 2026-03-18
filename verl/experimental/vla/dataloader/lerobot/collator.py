from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch


def collate_and_split_t0_t1(batch: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Input:
        batch: List[B][2][Dict[str, value]]
            - value can be torch.Tensor or non-tensor (str, int, bool, etc.)

    Output:
        Dict with keys:
            - 't0.<key>' and 't1.<key>'

        Rules:
            - Tensor fields -> Tensor with shape (B, ...)
            - Non-tensor fields -> np.ndarray with shape (B,)
    """
    B = len(batch)
    assert B > 0, "Empty batch"
    assert len(batch[0]) == 2, "Second dimension must be 2 (t0, t1)"

    keys = batch[0][0].keys()
    out: Dict[str, Any] = {}

    for key in keys:
        sample_val = batch[0][0][key]

        if isinstance(sample_val, torch.Tensor):
            stacked = torch.stack(
                [
                    torch.stack([batch[b][i][key] for i in range(2)], dim=0)
                    for b in range(B)
                ],
                dim=0,
            )  # shape: (B, 2, ...)
            out[f"t0.{key}"] = stacked[:, 0, ...]
            out[f"t1.{key}"] = stacked[:, 1, ...]
        else:
            t0_list = [batch[b][0][key] for b in range(B)]
            t1_list = [batch[b][1][key] for b in range(B)]

            out[f"t0.{key}"] = np.asarray(t0_list, dtype=object)
            out[f"t1.{key}"] = np.asarray(t1_list, dtype=object)

    return out


collate_fn = collate_and_split_t0_t1
collate_func = collate_and_split_t0_t1