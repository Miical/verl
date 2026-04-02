# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing_extensions import override

import torch
import torch.nn.functional as F
from verl.protocol import DataProto

from .base import Pi0Input, Pi0Output

PI0_MAX_STATE_DIM = 32
PI0_ACTION_CHUNK_SIZE = 10
LEROBOT_ACTION_DIM = 6
LEROBOT_IMAGE_CROP_SIZE = 480


class LerobotPi0Input(Pi0Input):
    @staticmethod
    def _center_crop(image: torch.Tensor, crop_size: int) -> torch.Tensor:
        image = image.detach().cpu().float().squeeze()
        _, height, width = image.shape
        crop_size = min(crop_size, height, width)
        top = max((height - crop_size) // 2, 0)
        left = max((width - crop_size) // 2, 0)
        return image[:, top : top + crop_size, left : left + crop_size]

    @staticmethod
    def _resize_image(image: torch.Tensor, resize_size: tuple[int, int]) -> torch.Tensor:
        return F.interpolate(
            image.unsqueeze(0),
            size=resize_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    @override
    @classmethod
    def from_env_obs(cls, env_obs: DataProto) -> "LerobotPi0Input":
        input = cls()

        # Process images
        images = env_obs.batch["observation.images.top"]
        wrist_images = env_obs.batch["observation.images.wrist"]
        device = images.device
        resize_size = (224, 224)

        images = torch.stack(
            [cls._resize_image(cls._center_crop(image, LEROBOT_IMAGE_CROP_SIZE), resize_size) for image in images],
            dim=0,
        ).to(device=device)
        wrist_images = torch.stack(
            [cls._resize_image(cls._center_crop(image, LEROBOT_IMAGE_CROP_SIZE), resize_size) for image in wrist_images],
            dim=0,
        ).to(device=device)

        batch_size = images.shape[0]
        cam_high = images
        left_wrist = wrist_images
        empty_images = torch.zeros(
            (batch_size, 3, cam_high.shape[2], cam_high.shape[3]),
            device=device,
            dtype=torch.bfloat16,
        )

        input.images = {
            "observation.images.cam_high": cam_high.to(torch.bfloat16),
            "observation.images.cam_left_wrist": left_wrist.to(torch.bfloat16),
            "observation.images.cam_right_wrist": empty_images,
        }
        input.img_masks = [
            torch.ones((batch_size,), device=device, dtype=torch.bool),
            torch.ones((batch_size,), device=device, dtype=torch.bool),
            torch.zeros((batch_size,), device=device, dtype=torch.bool),
        ]

        # Process other data
        input.task = list(env_obs.non_tensor_batch["task"])

        state = env_obs.batch["observation.state"]
        input.state = torch.nn.functional.pad(
            state, (0, max(0, PI0_MAX_STATE_DIM - state.shape[-1])), "constant", 0
        ).to(device=device, dtype=torch.float32)

        return input


class LerobotPi0Output(Pi0Output):
    @override
    @classmethod
    def from_model_output(cls, model_output: dict) -> "LerobotPi0Output":
        output = cls()
        output.action = model_output["full_action"][:, :PI0_ACTION_CHUNK_SIZE, :LEROBOT_ACTION_DIM]
        output.log_prob = model_output["log_probs"]
        return output
