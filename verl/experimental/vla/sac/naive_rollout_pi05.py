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
        """真机模式：处理复杂的压缩输入（代码B逻辑）"""
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
        raw_state_dim = int(state.shape[-1])

        # 2) pad state 到 32（模型forward需要）
        state_pad32 = torch.nn.functional.pad(state, (0, max(0, 32 - raw_state_dim)), "constant", 0.0)

        # 3) 检查模型是否支持 state_dim 参数
        sample_sig = inspect.signature(self.module.sample_actions)
        supports_state_dim = "state_dim" in sample_sig.parameters

        with torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
            # 4) JPEG 解码（如果需要）
            if head_image.ndim == 2:
                head_image = self._decode_jpeg_images(head_image)
            if left_wrist_image.ndim == 2:
                left_wrist_image = self._decode_jpeg_images(left_wrist_image)
            if right_wrist_image.ndim == 2:
                right_wrist_image = self._decode_jpeg_images(right_wrist_image)

            # 5) 转换图像格式：[B,H,W,C] -> [B,C,H,W]
            batch_size = head_image.shape[0]
            cam_high = head_image.permute(0, 3, 1, 2).to(device)
            left_wrist = left_wrist_image.permute(0, 3, 1, 2).to(device)
            right_wrist = right_wrist_image.permute(0, 3, 1, 2).to(device)

            # 6) 构造详细的 kwargs
            kwargs = dict(
                images={
                    "observation.images.cam_high": cam_high,
                    "observation.images.cam_left_wrist": left_wrist,
                    "observation.images.cam_right_wrist": right_wrist,
                },
                img_masks=[
                    torch.ones((batch_size,), device=device, dtype=torch.bool),
                    torch.ones((batch_size,), device=device, dtype=torch.bool),
                    torch.ones((batch_size,), device=device, dtype=torch.bool),
                ],
                task=task_descriptions.tolist() if hasattr(task_descriptions, "tolist") else list(task_descriptions),
                state=state_pad32,
                tokenizer=self.tokenizer,
            )

            # 7) 传递 state_dim（如果支持）
            if supports_state_dim:
                kwargs["state_dim"] = raw_state_dim

            # 8) 从模型config读取 use_endpose 和 no_state 配置并传递
            cfg = getattr(self.module, "config", None)
            if cfg is not None:
                use_endpose = getattr(cfg, "use_endpose", False)
                no_state = getattr(cfg, "no_state", False)
                kwargs["use_endpose"] = use_endpose
                kwargs["no_state"] = no_state

            # 9) 调用模型
            (
                action,
                images_out,
                img_masks,
                lang_tokens,
                lang_masks,
                state_out,
            ) = self.module.sample_actions(**kwargs)

        # 10) 读取 chunk_len 和 action_dim
        cfg = getattr(self.module, "config", None)
        T = getattr(cfg, "num_action_chunks", 30)
        A = getattr(cfg, "action_dim", action.shape[-1])
        T = min(int(T), int(action.shape[1]))
        A = min(int(A), int(action.shape[2]))

        ret = DataProto.from_dict(
            {
                "action": action[:, :T, :A],
                "full_action": action,
                "images": torch.stack(images_out, dim=1) if isinstance(images_out, (list, tuple)) else images_out,
                "image_masks": torch.stack(img_masks, dim=1) if isinstance(img_masks, (list, tuple)) else img_masks,
                "lang_tokens": lang_tokens,
                "lang_masks": lang_masks,
                "states": state_out,
            }
        )
        return ret

    def _generate_sequences_simulation(self, prompts: DataProto) -> DataProto:
        """仿真模式：简单输入处理（代码A逻辑）"""
        with torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
            prompts.to(get_device_id())
            output, s, a = self.module.sample_actions(prompts, tokenizer=self.tokenizer)

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
