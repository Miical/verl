# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

import torch
from typing_extensions import override

from verl.protocol import DataProto

from .base import Pi0Input, Pi0Output

PI0_MAX_STATE_DIM = 32
PI0_ACTION_CHUNK_SIZE = 10
PIPER_ACTION_DIM = 14


class PiperPi0Input(Pi0Input):
    @override
    @classmethod
    def from_env_obs(cls, env_obs: DataProto) -> "PiperPi0Input":
        input = cls()

        head_image = env_obs.batch["head_image"]
        left_wrist_image = env_obs.batch["left_wrist_image"]
        right_wrist_image = env_obs.batch["right_wrist_image"]
        batch_size = head_image.shape[0]
        device = head_image.device

        input.images = {
            "observation.images.cam_high": head_image.permute(0, 3, 1, 2).to(torch.bfloat16),
            "observation.images.cam_left_wrist": left_wrist_image.permute(0, 3, 1, 2).to(torch.bfloat16),
            "observation.images.cam_right_wrist": right_wrist_image.permute(0, 3, 1, 2).to(torch.bfloat16),
        }
        input.img_masks = [
            torch.ones((batch_size,), device=device, dtype=torch.bool),
            torch.ones((batch_size,), device=device, dtype=torch.bool),
            torch.ones((batch_size,), device=device, dtype=torch.bool),
        ]

        input.task = list(env_obs.non_tensor_batch["task_descriptions"])

        state = env_obs.batch["state"]
        if state.ndim == 3:
            state = state[:, -1, :]
        elif state.ndim == 1:
            state = state.unsqueeze(0)
        elif state.ndim != 2:
            raise ValueError(f"Unexpected robot state shape: {tuple(state.shape)}")

        input.state = torch.nn.functional.pad(
            state, (0, max(0, PI0_MAX_STATE_DIM - state.shape[-1])), "constant", 0
        ).to(device=device, dtype=torch.float32)

        return input


class PiperPi0Output(Pi0Output):
    @override
    @classmethod
    def from_model_output(cls, model_output: dict) -> "PiperPi0Output":
        output = cls()
        full_action = model_output["full_action"]
        output.action = full_action[:, :PI0_ACTION_CHUNK_SIZE, : min(PIPER_ACTION_DIM, full_action.shape[-1])]
        return output
