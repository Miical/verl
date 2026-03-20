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

        from torch.distributed.fsdp import register_fsdp_forward_method

        register_fsdp_forward_method(self.module, "sample_actions")
        register_fsdp_forward_method(self.module, "sac_forward_state_features")
        register_fsdp_forward_method(self.module, "sac_forward_critic")

    def _is_real_robot_input(self, prompts: DataProto) -> bool:
        required_keys = ["head_image", "left_wrist_image", "right_wrist_image", "state"]
        return all(key in prompts.batch.keys() for key in required_keys)

    def _decode_jpeg_images(self, encoded_tensor: torch.Tensor) -> torch.Tensor:
        batch_size = encoded_tensor.shape[0]
        decoded_images = []
        for i in range(batch_size):
            img_bytes = encoded_tensor[i].detach().cpu().numpy().tobytes()
            nparr = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                img_rgb = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            decoded_images.append(torch.from_numpy(img_rgb))
        return torch.stack(decoded_images, dim=0)

    def _prepare_robot_prompts(self, prompts: DataProto) -> DataProto:
        head_image = prompts.batch["head_image"]
        left_wrist_image = prompts.batch["left_wrist_image"]
        right_wrist_image = prompts.batch["right_wrist_image"]
        state = prompts.batch["state"]
        task_descriptions = prompts.non_tensor_batch["task_descriptions"]

        if state.ndim == 3:
            state = state[:, -1, :]
        elif state.ndim == 1:
            state = state.unsqueeze(0)
        elif state.ndim != 2:
            raise ValueError(f"Unexpected robot state shape: {tuple(state.shape)}")

        if head_image.ndim == 2:
            head_image = self._decode_jpeg_images(head_image)
        if left_wrist_image.ndim == 2:
            left_wrist_image = self._decode_jpeg_images(left_wrist_image)
        if right_wrist_image.ndim == 2:
            right_wrist_image = self._decode_jpeg_images(right_wrist_image)

        device = get_device_id()
        return DataProto.from_dict(
            tensors={
                "head_image": head_image.to(device),
                "left_wrist_image": left_wrist_image.to(device),
                "right_wrist_image": right_wrist_image.to(device),
                "state": state.to(device=device, dtype=torch.float32),
            },
            non_tensors={
                "task_descriptions": task_descriptions.tolist()
                if hasattr(task_descriptions, "tolist")
                else list(task_descriptions)
            },
        )

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Generate one rollout chunk while keeping current SAC critic-value outputs intact."""

        with torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
            if self._is_real_robot_input(prompts):
                prompts = self._prepare_robot_prompts(prompts)
            else:
                prompts.to(get_device_id())

            output, s, a = self.module.sample_actions(prompts, tokenizer=self.tokenizer)
            state_features = self.module.sac_forward_state_features(s)
            critic_value = self.module.sac_forward_critic(
                {"full_action": a["full_action"]},
                state_features,
                use_target_network=False,
                method="min",
                requires_grad=False,
            ).detach().float().reshape(-1)

        tensor_batch = {
            "action": output.action,
            "full_action": a["full_action"],
            "images": s["images"],
            "image_masks": s["image_masks"],
            "lang_tokens": s["lang_tokens"],
            "lang_masks": s["lang_masks"],
            "states": s["states"],
            "critic_value": critic_value,
        }

        ret = DataProto.from_dict(tensor_batch)
        return ret
