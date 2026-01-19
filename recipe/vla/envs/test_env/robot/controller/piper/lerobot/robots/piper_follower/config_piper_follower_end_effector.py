#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataclasses import dataclass, field
from lerobot.cameras import CameraConfig

from ..config import RobotConfig
from .config_piper_follower import PiperFollowerConfig


@RobotConfig.register_subclass("piper_follower_end_effector")
@dataclass
class PiperFollowerEndEffectorConfig(PiperFollowerConfig):
    """
    Piper 双臂末端空间控制配置（在 PiperFollowerConfig 基础上扩展）
    """

    # URDF 与目标末端 frame（与 PiperFollowerConfig 中保持一致或覆盖）
    # 若左右臂末端 frame 名字相同，可共用一个；也可以写成 left/right 两个字段
    urdf_path: str | None = "local_assets/piper.urdf"
    # target_frame_name: str = "gripper"   # 你的 URDF 中末端执行器 link 名称
    target_frame_name: str = "link6"   # 你的 URDF 中末端执行器 link 名称

    # 末端工作空间边界（单位: 米）——左右臂可分开配置
    end_effector_bounds_left: dict[str, list[float]] = field(
        default_factory=lambda: {"min": [-0.6, -0.6, -0.1], "max": [0.6, 0.6, 0.8]}
    )
    end_effector_bounds_right: dict[str, list[float]] = field(
        default_factory=lambda: {"min": [-0.6, -0.6, -0.1], "max": [0.6, 0.6, 0.8]}
    )

    # 每次键盘/teleop 的步长（单位: 米）
    end_effector_step_sizes: dict[str, float] = field(
        default_factory=lambda: {"x": 0.0001, "y": 0.0001, "z": 0.0001}
    )

    # 夹爪最大开合（单位: 米，Piper 小夹爪约 0.07m）
    max_gripper_pos: float = 0.07

    # 相机（沿用父类）
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
