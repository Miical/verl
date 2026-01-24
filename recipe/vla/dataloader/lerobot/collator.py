import torch
import numpy as np
from typing import List, Dict, Any

def collate_and_split_t0_t1(
    batch: List[List[Dict[str, Any]]]
) -> Dict[str, Any]:
    """
    Input:
        batch: List[B][2][Dict[str, value]]
            - value can be torch.Tensor or non-tensor (str, int, bool, etc.)

    Output:
        Dict with keys:
            - 't0.<key>' and 't1.<key>'
        Rules:
            - Tensor fields -> Tensor with shape (B, ...)
            - Non-tensor fields -> List[B]
    """
    B = len(batch)
    assert B > 0, "Empty batch"
    assert len(batch[0]) == 2, "Second dimension must be 2 (t0, t1)"

    keys = batch[0][0].keys()
    out: Dict[str, Any] = {}

    for key in keys:
        sample_val = batch[0][0][key]

        # ---------- Tensor fields ----------
        if isinstance(sample_val, torch.Tensor):
            # collect (B, 2, ...)
            stacked = torch.stack(
                [
                    torch.stack([batch[b][i][key] for i in range(2)], dim=0)
                    for b in range(B)
                ],
                dim=0,  # (B, 2, ...)
            )
            out[f"t0.{key}"] = stacked[:, 0, ...]
            out[f"t1.{key}"] = stacked[:, 1, ...]

        # ---------- Non-tensor fields (str, etc.) ----------
        else:
            t0_list = np.array([])
            t1_list = np.array([])
            for b in range(B):
                v0 = batch[b][0][key]
                v1 = batch[b][1][key]
                t0_list = np.append(t0_list, v0)
                t1_list = np.append(t1_list, v1)

            out[f"t0.{key}"] = t0_list
            out[f"t1.{key}"] = t1_list

    return out

collate_func = collate_and_split_t0_t1