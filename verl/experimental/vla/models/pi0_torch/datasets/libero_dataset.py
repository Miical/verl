import os
import re

import torch
import torch.nn.functional as F
from typing_extensions import override

from verl.protocol import DataProto

from .base import Pi0DatasetInput
from .utils import pad_dim_to, pad_last_dim_to

LIBERO_SIM_IMAGE_SIZE = 256


def parse_task(p: str) -> str:
    name = os.path.basename(p)
    if name.endswith("_demo.hdf5"):
        name = name[: -len("_demo.hdf5")]

    # Strip leading scene prefix (e.g., KITCHEN_SCENE3_*) to align with
    # benchmark task.language style used by online RL rollout data.
    name = re.sub(r"^[A-Z_]+_SCENE\d+_", "", name)
    name = name.replace("_", " ").lower()
    return name


class LiberoPi0DatasetInput(Pi0DatasetInput):
    @override
    @classmethod
    def from_dataset_batch(cls, batch: DataProto) -> "LiberoPi0DatasetInput":
        input = cls()
        device = batch.batch["t0.actions"].device
        batch_size = batch.batch["t0.actions"].shape[0]

        state = {}
        for prefix in ["t0", "t1"]:
            cam_high = torch.flip(batch.batch[f"{prefix}.obs.agentview_rgb"], dims=(1, 2)).permute(0, 3, 1, 2).float()
            left_wrist = (
                torch.flip(batch.batch[f"{prefix}.obs.eye_in_hand_rgb"], dims=(1, 2)).permute(0, 3, 1, 2).float()
            )

            if cam_high.shape[-2:] != (LIBERO_SIM_IMAGE_SIZE, LIBERO_SIM_IMAGE_SIZE):
                cam_high = F.interpolate(
                    cam_high,
                    size=(LIBERO_SIM_IMAGE_SIZE, LIBERO_SIM_IMAGE_SIZE),
                    mode="bilinear",
                    align_corners=False,
                )
                left_wrist = F.interpolate(
                    left_wrist,
                    size=(LIBERO_SIM_IMAGE_SIZE, LIBERO_SIM_IMAGE_SIZE),
                    mode="bilinear",
                    align_corners=False,
                )

            cam_high = cam_high.to(torch.bfloat16)
            left_wrist = left_wrist.to(torch.bfloat16)
            empty_right = torch.zeros_like(left_wrist)

            state[prefix] = {
                "images": {
                    "observation.images.cam_high": cam_high,
                    "observation.images.cam_left_wrist": left_wrist,
                    "observation.images.cam_right_wrist": empty_right,
                },
                "img_masks": [
                    torch.ones((batch_size,), dtype=torch.bool),
                    torch.ones((batch_size,), dtype=torch.bool),
                    torch.zeros((batch_size,), dtype=torch.bool),
                ],
                "task": [parse_task(p) for p in batch.non_tensor_batch[f"{prefix}.hdf5_path"]],
                "state": pad_last_dim_to(
                    torch.cat(
                        [
                            batch.batch[f"{prefix}.obs.ee_pos"],
                            batch.batch[f"{prefix}.obs.ee_ori"],
                            batch.batch[f"{prefix}.obs.gripper_states"],
                        ],
                        dim=-1,
                    ),
                    32,
                ),
            }

        input.s0 = state["t0"]
        input.s1 = state["t1"]
        input.a0 = {"action": pad_dim_to(pad_last_dim_to(batch.batch["t0.actions"], 32), dim=1, target_size=50)}
        input.a1 = {"action": pad_dim_to(pad_last_dim_to(batch.batch["t1.actions"], 32), dim=1, target_size=50)}
        input.rewards = batch.batch["t1.chunk_dones"].float()
        input.valids = torch.ones((batch_size,), dtype=torch.bool, device=device)
        input.dones = batch.batch["t1.chunk_dones"].bool()
        input.positive_sample_mask = torch.ones((batch_size,), dtype=torch.bool, device=device)

        return input
