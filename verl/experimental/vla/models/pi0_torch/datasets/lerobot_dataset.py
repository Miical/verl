import torch
from typing_extensions import override

from verl.protocol import DataProto

from .base import Pi0DatasetInput
from .utils import pad_dim_to


def _get_required_image(batch: DataProto, key: str) -> torch.Tensor:
    if key not in batch.batch:
        raise KeyError(f"Missing required image key: {key}")
    return batch.batch[key]


def _get_optional_image(batch: DataProto, key: str, fallback: torch.Tensor) -> torch.Tensor:
    if key in batch.batch:
        return batch.batch[key]
    return torch.zeros_like(fallback)


def _infer_task_list(batch: DataProto, batch_size: int) -> list[str]:
    non_tensor_batch = getattr(batch, "non_tensor_batch", None)
    if non_tensor_batch is not None:
        for key in [
            "task",
            "tasks",
            "language_instruction",
            "instruction",
            "t0.task",
            "t0.language_instruction",
        ]:
            if key in non_tensor_batch:
                values = non_tensor_batch[key]
                if isinstance(values, str):
                    return [values] * batch_size
                return [str(v) for v in values]
    return ["catch_bowl"] * batch_size


def _get_action_loss_mask(batch: DataProto, key: str, batch_size: int, time_steps: int, device: torch.device) -> torch.Tensor:
    if key in batch.batch:
        mask = ~batch.batch[key].to(device=device, dtype=torch.bool)
        if mask.ndim != 2:
            raise ValueError(f"Expected {key} to have shape [B, T], got {tuple(mask.shape)}")
        return mask
    return torch.ones((batch_size, time_steps), dtype=torch.bool, device=device)


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

        tasks = _infer_task_list(batch, batch_size)

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
            "task": tasks,
            "state": batch.batch["t0.observation.state"],
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
            "task": tasks,
            "state": batch.batch["t1.observation.state"],
        }

        a0 = batch.batch["t0.action"]
        a1 = batch.batch["t1.action"]

        if a0.ndim != 3:
            raise ValueError(f"Expected t0.action to be chunk action [B, T, A], got shape={tuple(a0.shape)}")
        if a1.ndim != 3:
            raise ValueError(f"Expected t1.action to be chunk action [B, T, A], got shape={tuple(a1.shape)}")

        a0_time_steps = a0.shape[1]
        a1_time_steps = a1.shape[1]
        a0_action_loss_mask = _get_action_loss_mask(batch, "t0.action_is_pad", batch_size, a0_time_steps, device)
        a1_action_loss_mask = _get_action_loss_mask(batch, "t1.action_is_pad", batch_size, a1_time_steps, device)

        if a0_time_steps < 50:
            a0 = pad_dim_to(a0, dim=1, target_size=50)
            a0_action_loss_mask = pad_dim_to(a0_action_loss_mask, dim=1, target_size=50)
        elif a0_time_steps > 50:
            a0 = a0[:, :50]
            a0_action_loss_mask = a0_action_loss_mask[:, :50]

        if a1_time_steps < 50:
            a1 = pad_dim_to(a1, dim=1, target_size=50)
            a1_action_loss_mask = pad_dim_to(a1_action_loss_mask, dim=1, target_size=50)
        elif a1_time_steps > 50:
            a1 = a1[:, :50]
            a1_action_loss_mask = a1_action_loss_mask[:, :50]

        input.a0 = {"action": a0, "action_loss_mask": a0_action_loss_mask}
        input.a1 = {"action": a1, "action_loss_mask": a1_action_loss_mask}

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
