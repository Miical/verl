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

import torch
import logging
from typing_extensions import override

from verl import DataProto
from .base import BaseSACRollout, SupportSACTraining
from verl.utils.device import get_device_name

CRITIC_VALUE_KEY = "critic_value"
logger = logging.getLogger(__name__)

class RobSACRollout(BaseSACRollout):
    def __init__(
        self,
        model_config: dict,
        model: SupportSACTraining,
        tokenizer,
    ):
        self.model = model
        self.model_config = model_config
        self.tokenizer = tokenizer
        self.device = get_device_name()
    
    @torch.no_grad()
    @override
    def generate_sequences(self, obs: DataProto) -> DataProto:
        obs.to(self.device)

        with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
            output = self.model.sac_sample_actions(obs, self.tokenizer, obs.meta_info.get("validate", False))
            critic_value = self.model.sac_get_critic_value(obs, output, self.tokenizer)

        output = output.to_data_proto()
        output.batch[CRITIC_VALUE_KEY] = critic_value

        return output
