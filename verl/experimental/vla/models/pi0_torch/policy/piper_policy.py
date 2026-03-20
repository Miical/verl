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
PIPER_ACTION_DIM = 14  # Piper 双臂机器人的动作维度（每个臂7维）


class PiperPi0Input(Pi0Input):
    @override
    @classmethod
    def from_env_obs(cls, env_obs: DataProto) -> "PiperPi0Input":
        input = cls()

        # 检测输入类型：真机 vs 仿真
        is_real_robot = all(key in env_obs.batch for key in ["head_image", "left_wrist_image", "right_wrist_image"])
        
        if is_real_robot:
            # 真机模式：处理三个实际的相机输入
            head_image = env_obs.batch["head_image"]
            left_wrist_image = env_obs.batch["left_wrist_image"]
            right_wrist_image = env_obs.batch["right_wrist_image"]
            
            batch_size = head_image.shape[0]
            # 真机输入已经是 [B,H,W,C] 格式，需要转换为 [B,C,H,W]
            cam_high = head_image.permute(0, 3, 1, 2)
            left_wrist = left_wrist_image.permute(0, 3, 1, 2)
            right_wrist = right_wrist_image.permute(0, 3, 1, 2)
            
            input.images = {
                "observation.images.cam_high": cam_high.to(torch.bfloat16),
                "observation.images.cam_left_wrist": left_wrist.to(torch.bfloat16),
                "observation.images.cam_right_wrist": right_wrist.to(torch.bfloat16),  # 真机有实际的右腕图像
            }
            input.img_masks = [
                torch.ones((batch_size,), device=env_obs.batch.device, dtype=torch.bool),  # cam_high 有效
                torch.ones((batch_size,), device=env_obs.batch.device, dtype=torch.bool),  # left_wrist 有效
                torch.ones((batch_size,), device=env_obs.batch.device, dtype=torch.bool),  # right_wrist 有效（真机）
            ]
        else:
            # 仿真模式：处理两个相机输入 + 一个空图像
            images = env_obs.batch["full_image"]
            wrist_images = env_obs.batch["wrist_image"]

            batch_size = images.shape[0]
            cam_high = images.permute(0, 3, 1, 2)
            left_wrist = wrist_images.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
            empty_images = torch.zeros(
                (batch_size, 3, cam_high.shape[2], cam_high.shape[3]),
                device=env_obs.batch.device,
                dtype=torch.bfloat16,
            )

            input.images = {
                "observation.images.cam_high": cam_high.to(torch.bfloat16),
                "observation.images.cam_left_wrist": left_wrist.to(torch.bfloat16),
                "observation.images.cam_right_wrist": empty_images,  # 仿真使用空图像
            }
            input.img_masks = [
                torch.ones((batch_size,), device=env_obs.batch.device, dtype=torch.bool),  # cam_high 有效
                torch.ones((batch_size,), device=env_obs.batch.device, dtype=torch.bool),  # left_wrist 有效
                torch.zeros((batch_size,), device=env_obs.batch.device, dtype=torch.bool),  # right_wrist 无效（仿真）
            ]

        # Process other data
        input.task = list(env_obs.non_tensor_batch["task_descriptions"])

        state = env_obs.batch["state"]
        input.state = torch.nn.functional.pad(
            state, (0, max(0, PI0_MAX_STATE_DIM - state.shape[-1])), "constant", 0
        ).to(env_obs.batch.device, dtype=torch.float32)

        return input


class PiperPi0Output(Pi0Output):
    @override
    @classmethod
    def from_model_output(cls, model_output: dict) -> "PiperPi0Output":
        output = cls()
        full_action = model_output["full_action"]
        
        # 动态获取实际的 action_dim，默认使用 PIPER_ACTION_DIM
        actual_action_dim = full_action.shape[-1]
        action_dim = min(actual_action_dim, PIPER_ACTION_DIM)
        
        # 提取前 PI0_ACTION_CHUNK_SIZE 个时间步和前 action_dim 个动作维度
        output.action = full_action[:, :PI0_ACTION_CHUNK_SIZE, :action_dim]
        return output
