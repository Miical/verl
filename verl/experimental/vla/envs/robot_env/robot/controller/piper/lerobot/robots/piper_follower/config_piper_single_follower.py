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
from lerobot.cameras import CameraConfig

from ..config import RobotConfig
from .config_piper_follower import PiperFollowerConfig


@RobotConfig.register_subclass("piper_single_follower")
@dataclass
class PiperSingleFollowerConfig(RobotConfig):
    """
    Piper 单臂机器人配置

    与双臂版的差异：
    - 仅一条 CAN 总线；
    - 通过 side 指定使用 left 或 right；
    - max_relative_target 的键不带左右前缀（单臂环境下推荐使用 'joint_1'…'joint_6'、'gripper'）。

    注意：
    - 如果你后续的驱动/总线实现仍旧要求带前缀（如 'left_joint_1'），
      可以在加载 config 时根据 side 动态加前缀再下发。
    """

    # 选择使用哪一侧的手臂
    side="left"

    # 单臂 CAN 接口配置
    can_name: str = "can_left"         # 可根据 side 自动派生（见 __post_init__）
    baud_rate: int = 1_000_000         # 1 Mbps

    # 运动学与末端
    urdf_path: str = "local_assets/piper.urdf"
    target_frame_name: str = "link6"  # urdf 中末端执行器的 frame 名

    # 安全配置
    reset_pos_on_disconnect: bool = False
    disable_torque_on_disconnect: bool = False

    # 单次相对位移限制（单臂无前缀写法：'joint_1'…'joint_6'、'gripper'）
    max_relative_target: dict[str, float] | None = None
    """
    示例：
    max_relative_target = {
        "joint_1": 0.5,   # rad
        "joint_2": 0.2,
        "joint_3": 0.2,
        "joint_4": 0.5,
        "joint_5": 0.5,
        "joint_6": 0.5,
        "gripper": 0.05,  # m
    }
    """

    # V2 版本特有
    start_sdk_joint_limit: bool = True
    start_sdk_gripper_limit: bool = True
    move_spd_rate_ctrl: int = 50  # 速度百分比（0–100）

    # 相机（按需填写）
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # 可选：根据 side 自动规范化 can_name
    def __post_init__(self):
        # 若用户未显式修改 can_name，则按 side 设置默认值
        if self.can_name == "can_left" and self.side == "right":
            self.can_name = "can_right"