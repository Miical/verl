#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("piper_leader")
@dataclass
class PiperLeaderConfig(TeleoperatorConfig):
    # Port to connect to the arm
    can_name_left: str = "can_left"        # 左臂CAN接口名称
    can_name_right: str = "can_right"       # 右臂CAN接口名称
    baud_rate: int = 1000000          # CAN波特率 1Mbps