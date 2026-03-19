import torch
from typing_extensions import override

from verl.protocol import DataProto

from .base import Pi0DatasetInput
from .utils import pad_last_dim_to, pad_dim_to


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
                torch.ones((batch_size,), dtype=torch.bool, device=device),
            ],
            "task": ["catch_bowl"] * batch_size,
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
                torch.ones((batch_size,), dtype=torch.bool, device=device),
            ],
            "task": ["catch_bowl"] * batch_size,
            "state": pad_last_dim_to(batch.batch["t1.observation.state"], 32),
        }

        a0 = batch.batch["t0.action"]
        a1 = batch.batch["t1.action"]

        if a0.ndim != 3:
            raise ValueError(f"Expected t0.action to be chunk action [B, T, A], got shape={tuple(a0.shape)}")
        if a1.ndim != 3:
            raise ValueError(f"Expected t1.action to be chunk action [B, T, A], got shape={tuple(a1.shape)}")

        a0 = pad_last_dim_to(a0, 32)
        a1 = pad_last_dim_to(a1, 32)
        a0 = pad_dim_to(a0, dim=1, target_size=50)
        a1 = pad_dim_to(a1, dim=1, target_size=50)

        input.a0 = {"action": a0}
        input.a1 = {"action": a1}

        done = None
        for done_key in ["t1.next.done", "t0.next.done", "next.done"]:
            if done_key in batch.batch:
                done = batch.batch[done_key].bool()
                break

        if done is None:
            done = torch.zeros((batch_size,), dtype=torch.bool, device=device)

        done = done.view(batch_size)

        # Chunk-horizon semantics:
        # - reward/done stay sparse and mark whether the chunk reaches terminal
        # - positive_sample_mask is aligned with online rollout semantics for
        #   successful demonstrations: all transitions from successful demos are positive
        input.rewards = done.float()
        input.valids = torch.ones((batch_size,), dtype=torch.float32, device=device)
        input.dones = done.float()
        input.positive_sample_mask = torch.ones((batch_size,), dtype=torch.float32, device=device)

        return input
