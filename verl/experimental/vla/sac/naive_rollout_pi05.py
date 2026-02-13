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
"""
In single GPU rollout, the sequences are generated directly by sampling from the model.
The output will contain
1. output_ids
2. attention_masks (left padding)
3. eos_masks
4. log_probs
"""

import inspect
import logging
from typing import Any

import cv2
import numpy as np
import torch

from verl import DataProto
from verl.experimental.vla.naive_rollout_rob import NaiveRolloutRob
from verl.utils.device import get_device_id, get_device_name

logger = logging.getLogger(__name__)

__all__ = ["PI0RolloutRob"]


class PI0RolloutRob(NaiveRolloutRob):
    def __init__(
        self,
        model_config: dict,
        module: torch.nn.Module,
        tokenizer: Any,
    ):
        self.model_config = model_config
        self.module = module
        self.tokenizer = tokenizer

        # 注册FSDP forward方法（对分布式训练至关重要）
        from torch.distributed.fsdp import register_fsdp_forward_method

        register_fsdp_forward_method(self.module, "sample_actions")

    def _decode_jpeg_images(self, encoded_tensor: torch.Tensor) -> torch.Tensor:
        """解码 JPEG 编码的图像数据。"""
        batch_size = encoded_tensor.shape[0]
        decoded_images = []

        for i in range(batch_size):
            img_bytes = encoded_tensor[i].cpu().numpy().tobytes()
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                decoded_images.append(torch.from_numpy(img_rgb))
            else:
                decoded_images.append(torch.zeros((224, 224, 3), dtype=torch.uint8))

        return torch.stack(decoded_images)

    def _is_real_robot_input(self, prompts: DataProto) -> bool:
        """检测是否为真机输入（包含 head_image, left_wrist_image, right_wrist_image, state）"""
        required_keys = ["head_image", "left_wrist_image", "right_wrist_image", "state"]
        return all(key in prompts.batch for key in required_keys)

    def _generate_sequences_real_robot(self, prompts: DataProto) -> DataProto:
        """真机模式：处理三摄像头压缩输入，解码后重新打包为 DataProto 调用 PI0ForActionPrediction.sample_actions"""
        head_image = prompts.batch["head_image"]
        left_wrist_image = prompts.batch["left_wrist_image"]
        right_wrist_image = prompts.batch["right_wrist_image"]
        state = prompts.batch["state"]
        task_descriptions = prompts.non_tensor_batch["task_descriptions"]

        # 1) 统一 state shape
        if state.ndim == 3:
            state = state[:, -1, :]
        elif state.ndim == 2:
            pass
        elif state.ndim == 1:
            state = state.unsqueeze(0)
        else:
            raise ValueError(f"[PI0RolloutRob] Unexpected state shape: {state.shape}")

        device = next(self.module.parameters()).device
        state = state.to(device=device, dtype=torch.float32)
        
        with torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
            # 2) JPEG 解码（如果需要）
            if head_image.ndim == 2:
                head_image = self._decode_jpeg_images(head_image)
            if left_wrist_image.ndim == 2:
                left_wrist_image = self._decode_jpeg_images(left_wrist_image)
            if right_wrist_image.ndim == 2:
                right_wrist_image = self._decode_jpeg_images(right_wrist_image)

            # 3) 确保所有图像都在正确的设备上，保持 [B,H,W,C] 格式
            head_image = head_image.to(device)
            left_wrist_image = left_wrist_image.to(device)
            right_wrist_image = right_wrist_image.to(device)

            # 4) 重新打包成 DataProto 对象
            # 真机有三个相机输入，需要用特殊的键名来区分
            # 注意：这里使用真机特有的键名，与仿真的 full_image/wrist_image 不同
            repackaged_env_obs = DataProto.from_dict(
                {
                    "head_image": head_image,           # [B,H,W,C] - 头部相机
                    "left_wrist_image": left_wrist_image,   # [B,H,W,C] - 左腕相机
                    "right_wrist_image": right_wrist_image, # [B,H,W,C] - 右腕相机
                    "state": state,
                },
                non_tensors={
                    "task_descriptions": task_descriptions.tolist() if hasattr(task_descriptions, "tolist") else list(task_descriptions)
                }
            )

            # 5) 调用 PI0ForActionPrediction.sample_actions (标准接口)
            # 注意：当前 PI0ForActionPrediction.sample_actions 内部的 LiberoPi0Input.from_env_obs
            # 只处理仿真的 full_image/wrist_image，需要扩展以支持真机的三相机输入
            output, s, a = self.module.sample_actions(repackaged_env_obs, tokenizer=self.tokenizer, env_name="piper")

        # 6) 读取 chunk_len 和 action_dim
        cfg = getattr(self.module, "config", None)
        T = getattr(cfg, "num_action_chunks", 10)
        A = getattr(cfg, "action_dim", output.action.shape[-1])
        T = min(int(T), int(output.action.shape[1]))
        A = min(int(A), int(output.action.shape[2]))

        ret = DataProto.from_dict(
            {
                "action": output.action[:, :T, :A],
                "full_action": a["full_action"],
                "images": s["images"],
                "image_masks": s["image_masks"],
                "lang_tokens": s["lang_tokens"],
                "lang_masks": s["lang_masks"],
                "states": s["states"],
            }
        )
        return ret

    def _generate_sequences_simulation(self, prompts: DataProto) -> DataProto:
        """仿真模式：简单输入处理（代码A逻辑）"""
        with torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
            prompts.to(get_device_id())
            output, s, a = self.module.sample_actions(prompts, tokenizer=self.tokenizer, env_name="libero")

        ret = DataProto.from_dict(
            {
                "action": output.action,
                "full_action": a["full_action"],
                "images": s["images"],
                "image_masks": s["image_masks"],
                "lang_tokens": s["lang_tokens"],
                "lang_masks": s["lang_masks"],
                "states": s["states"],
            }
        )

        return ret

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """
        智能推理入口：自动检测输入类型并选择对应的处理逻辑
        
        模式1（真机）：包含 head_image, left_wrist_image, right_wrist_image, state
        模式2（仿真）：其他输入格式
        """
        if self._is_real_robot_input(prompts):
            return self._generate_sequences_real_robot(prompts)
        else:
            return self._generate_sequences_simulation(prompts)
