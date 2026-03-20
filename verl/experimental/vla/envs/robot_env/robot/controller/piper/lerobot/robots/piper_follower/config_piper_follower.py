#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass, field

# TODO
from lerobot import cameras
from lerobot.cameras import CameraConfig
from lerobot.cameras.dabai import OrbbecDabaiCameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("piper_follower")
@dataclass
class PiperFollowerConfig(RobotConfig):
    """Piper双臂机器人配置"""

    # 双臂CAN接口配置
    can_name_left: str = "can_left"        # 左臂CAN接口名称
    can_name_right: str = "can_right"       # 右臂CAN接口名称
    baud_rate: int = 1000000          # CAN波特率 1Mbps
    urdf_path: str = "local_assets/piper.urdf" # 机器人运动学模型
    target_frame_name: str = "gripper"  # urdf中末端执行器frame name TODO: check
    # 安全配置
    reset_pos_on_disconnect: bool = False  # 断开连接时是否复位
    disable_torque_on_disconnect: bool = False  # 断开连接时是否失能
    max_relative_target: dict[str, float] | None = None  # 最大相对位移限制
    # 见src/lerobot/robots/utils.py line 76
    """
    max_relative_target = {
    "joint_1": 0.5,    # joint_1单次最大移动 0.5rad
    "joint_2": 0.2,    # joint_2单次最大移动 0.2rad
    ...
    "gripper": 0.05,    # gripper单次最大移动 0.05m
    }
    """

    # V2版本特有配置
    start_sdk_joint_limit: bool = True      # 启用软件关节限位
    start_sdk_gripper_limit: bool = True    # 启用软件夹爪限位
    move_spd_rate_ctrl: int = 50            # 运动速度百分比

    # 相机配置 TODO
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
