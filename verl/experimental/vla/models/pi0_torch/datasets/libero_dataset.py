import os
import torch
from typing_extensions import override
from verl.protocol import DataProto
from .base import Pi0DatasetInput
from .utils import pad_last_dim_to, pad_dim_to

def parse_task(p: str) -> str:
    name = os.path.basename(p)
    if name.endswith("_demo.hdf5"):
        name = name[:-len("_demo.hdf5")]
    name = name.replace("_", " ")
    return name

class LiberoPi0DatasetInput(Pi0DatasetInput):

    @override
    @classmethod
    def from_dataset_batch(cls, batch: DataProto) -> "LiberoPi0DatasetInput":
        input = cls()
        batch_size = batch.batch["t0.actions"].shape[0]

        state = {}
        for prefix in ["t0", "t1"]:
            state[prefix] = {
                "images": {
                    "observation.images.cam_high": batch.batch[f"{prefix}.obs.agentview_rgb"].permute(0, 3, 1, 2).to(torch.bfloat16),
                    "observation.images.cam_left_wrist": batch.batch[f"{prefix}.obs.eye_in_hand_rgb"].permute(0, 3, 1, 2).to(torch.bfloat16),
                    "observation.images.cam_right_wrist": batch.batch[f"{prefix}.obs.eye_in_hand_rgb"].permute(0, 3, 1, 2).to(torch.bfloat16),
                },
                "img_masks": [torch.ones((batch_size,), dtype=torch.bool),
                              torch.ones((batch_size,), dtype=torch.bool),
                              torch.zeros((batch_size,), dtype=torch.bool)],
                "task": [parse_task(p) for p in batch.non_tensor_batch[f"{prefix}.hdf5_path"]],
                "state": pad_last_dim_to(torch.cat([batch.batch[f"{prefix}.obs.ee_pos"], batch.batch[f"{prefix}.obs.ee_ori"], batch.batch[f"{prefix}.obs.gripper_states"]], dim=-1), 32),
            }
        
        input.s0 = state["t0"]
        input.s1 = state["t1"]
        input.a0 = {"action": pad_dim_to(pad_last_dim_to(batch.batch["t0.actions"], 32), dim=1, target_size=50)}
        input.a1 = {"action": pad_dim_to(pad_last_dim_to(batch.batch["t1.actions"], 32), dim=1, target_size=50)}
        input.reward = batch.batch["t1.chunk_dones"].unsqueeze(-1).expand(batch_size, 10)  # (B, 10)
        input.valid = torch.ones((batch_size, 70), dtype=torch.bool, device=batch.batch["t0.actions"].device)

        return input