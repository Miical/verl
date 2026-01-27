import torch
from typing_extensions import override
from verl.protocol import DataProto
from .base import Pi0DatasetInput
from .utils import pad_last_dim_to


class LeRobotPi0DatasetInput(Pi0DatasetInput):

    @override
    @classmethod
    def from_dataset_batch(cls, batch: DataProto) -> "LeRobotPi0DatasetInput":
        input = cls()

        batch_size = batch.batch["t0.observation.images.cam_high"].shape[0]

        input.s0 = {
            'images' : {
                "observation.images.cam_high": batch.batch["t0.observation.images.cam_high"],
                "observation.images.cam_left_wrist": batch.batch["t0.observation.images.cam_left_wrist"],
                "observation.images.cam_right_wrist": batch.batch["t0.observation.images.cam_right_wrist"],
            }, 
            'img_masks' : [torch.ones((batch_size,), dtype=torch.bool),
                           torch.ones((batch_size,), dtype=torch.bool),
                           torch.zeros((batch_size,), dtype=torch.bool)],
            'task' : list(batch.non_tensor_batch["t0.task"]),
            'state' : pad_last_dim_to(batch.batch["t0.observation.state"], 32),
        }

        input.s1 = {
            'images' : {
                "observation.images.cam_high": batch.batch["t1.observation.images.cam_high"],
                "observation.images.cam_left_wrist": batch.batch["t1.observation.images.cam_left_wrist"],
                "observation.images.cam_right_wrist": batch.batch["t1.observation.images.cam_right_wrist"],
            }, 
            'img_masks' : [torch.ones((batch_size,), dtype=torch.bool),
                           torch.ones((batch_size,), dtype=torch.bool),
                           torch.zeros((batch_size,), dtype=torch.bool)],
            'task' : list(batch.non_tensor_batch["t1.task"]),
            'state' : pad_last_dim_to(batch.batch["t1.observation.state"], 32),
        }

        input.a0 = {
            'action' : pad_last_dim_to(batch.batch["t0.action"], 32),
        }

        input.a1 = {
            'action' : pad_last_dim_to(batch.batch["t1.action"], 32),
        }

        input.reward = batch.batch['t0.next.done'].float().unsqueeze(-1).expand(*batch.batch['t0.next.done'].shape, 10)
        input.valid = torch.ones((batch_size, 70), dtype=torch.bool)

        return input



