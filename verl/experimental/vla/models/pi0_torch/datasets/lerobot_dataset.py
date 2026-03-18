import torch
from typing_extensions import override

from verl.protocol import DataProto

from .base import Pi0DatasetInput
from .utils import pad_last_dim_to


def _get_required_image(batch: DataProto, key: str) -> torch.Tensor:
    if key not in batch.batch:
        raise KeyError(f"Missing required image key: {key}")
    return batch.batch[key]


def _get_optional_image(batch: DataProto, key: str, fallback: torch.Tensor) -> torch.Tensor:
    if key in batch.batch:
        return batch.batch[key]
    return torch.zeros_like(fallback)


class LeRobotPi0DatasetInput(Pi0DatasetInput):
    @override
    @classmethod
    def from_dataset_batch(cls, batch: DataProto) -> "LeRobotPi0DatasetInput":
        input = cls()

        cam_high_t0 = _get_required_image(batch, "t0.observation.images.cam_high")
        left_wrist_t0 = _get_required_image(batch, "t0.observation.images.cam_left_wrist")
        right_wrist_t0 = _get_optional_image(batch, "t0.observation.images.cam_right_wrist", left_wrist_t0)

        cam_high_t1 = _get_required_image(batch, "t1.observation.images.cam_high")
        left_wrist_t1 = _get_required_image(batch, "t1.observation.images.cam_left_wrist")
        right_wrist_t1 = _get_optional_image(batch, "t1.observation.images.cam_right_wrist", left_wrist_t1)

        device = batch.batch["t0.observation.state"].device
        batch_size = batch.batch["t0.observation.state"].shape[0]

        input.s0 = {
            "images": {
                "observation.images.cam_high": cam_high_t0,
                "observation.images.cam_left_wrist": left_wrist_t0,
                "observation.images.cam_right_wrist": right_wrist_t0,
            },
            "img_masks": [
                torch.ones((batch_size,), dtype=torch.bool, device=device),
                torch.ones((batch_size,), dtype=torch.bool, device=device),
                torch.zeros((batch_size,), dtype=torch.bool, device=device),
            ],
            "task": list(batch.non_tensor_batch["t0.task"]),
            "state": pad_last_dim_to(batch.batch["t0.observation.state"], 32),
        }

        input.s1 = {
            "images": {
                "observation.images.cam_high": cam_high_t1,
                "observation.images.cam_left_wrist": left_wrist_t1,
                "observation.images.cam_right_wrist": right_wrist_t1,
            },
            "img_masks": [
                torch.ones((batch_size,), dtype=torch.bool, device=device),
                torch.ones((batch_size,), dtype=torch.bool, device=device),
                torch.zeros((batch_size,), dtype=torch.bool, device=device),
            ],
            "task": list(batch.non_tensor_batch["t1.task"]),
            "state": pad_last_dim_to(batch.batch["t1.observation.state"], 32),
        }

        input.a0 = {
            "action": pad_last_dim_to(batch.batch["t0.action"], 32),
        }
        input.a1 = {
            "action": pad_last_dim_to(batch.batch["t1.action"], 32),
        }

        done_key = "t1.next.done" if "t1.next.done" in batch.batch else "t0.next.done"
        done = batch.batch[done_key].bool()

        input.rewards = done.float()
        input.valids = torch.ones((batch_size,), dtype=torch.bool, device=device)
        input.dones = done
        input.positive_sample_mask = done

        return input
