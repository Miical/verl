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


@TeleoperatorConfig.register_subclass("keyboard")
@dataclass
class KeyboardTeleopConfig(TeleoperatorConfig):
    """
    单臂关节空间键盘遥操作配置（配合 KeyboardTeleop 使用）
    键位：
      q w e a s d  -> 6 个关节 +
      u i o j k l  -> 6 个关节 -
      r(+), p(-)   -> 夹爪
    """
    mock: bool = False
    use_gripper: bool = True
    # 电机命名前缀（你当前写死左臂，这里默认 'left'）
    motor_prefix: str = "left"

    # 6 关节 + 1 夹爪 的名字（不带前缀）
    # 不用 field / List[str]，先给 None，再在 __post_init__ 里补默认值
    motor_names: list = None

    # 步长与边界
    step: float = 0.02           # 关节每次按键变化量
    gripper_step: float = 0.02   # 夹爪每次按键变化量
    joint_min: float = -1.0
    joint_max: float = 1.0
    gripper_min: float = 0.0
    gripper_max: float = 1.0

    def __post_init__(self):
        if self.motor_names is None:
            self.motor_names = [
                "joint1",
                "joint2",
                "joint3",
                "joint4",
                "joint5",
                "joint6",
                "gripper",
            ]



@TeleoperatorConfig.register_subclass("keyboard_ee")
@dataclass
class KeyboardEndEffectorTeleopConfig(KeyboardTeleopConfig):
    use_gripper: bool = True
