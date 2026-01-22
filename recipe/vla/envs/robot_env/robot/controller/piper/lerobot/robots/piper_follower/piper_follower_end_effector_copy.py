# !/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import time
from typing import Any
import numpy as np
import pdb
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.utils.errors import DeviceNotConnectedError
from lerobot.model.kinematics import RobotKinematics

from .piper_follower import PiperFollower
from .config_piper_follower_end_effector import PiperFollowerEndEffectorConfig

logger = logging.getLogger(__name__)



class PiperFollowerEndEffector(Robot):
    """Piper 双臂末端 6-DoF 控制：仅平移时 IK 只约束位置；显式转动时才启用姿态权重。"""

    config_class = PiperFollowerEndEffectorConfig
    name = "piper_follower_end_effector"

    def __init__(self, config: PiperFollowerEndEffectorConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config
        self.ctrl_mode = "End_Pose"
        self.piper_left = C_PiperInterface("can_left")
        self.piper_right = C_PiperInterface("can_right")

        self.cameras = make_cameras_from_configs(config.cameras)



    @property
    def action_features(self) -> dict[str, Any]:
        return {
            "dtype": "float32",
            "shape": (14,),
            "names": {
                "left_delta_x": 0, "left_delta_y": 1, "left_delta_z": 2,
                "left_delta_rx": 3, "left_delta_ry": 4, "left_delta_rz": 5,
                "left_gripper": 6,
                "right_delta_x": 7, "right_delta_y": 8, "right_delta_z": 9,
                "right_delta_rx": 10, "right_delta_ry": 11, "right_delta_rz": 12,
                "right_gripper": 13,
            },
        }


    def send_action(self, action: dict[str, Any] | np.ndarray) -> dict[str, Any]:
        



    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Read arm position from both arms
        start = time.perf_counter()
        if self.ctrl_mode == "Present_Position":
            # 左臂位置
            left_positions = self.bus_left.sync_read(self.ctrl_mode)
            obs_dict = {f"{motor}.pos": val for motor, val in left_positions.items()}
            
            # 右臂位置
            right_positions = self.bus_right.sync_read(self.ctrl_mode)
            obs_dict.update({f"{motor}.pos": val for motor, val in right_positions.items()})
        elif self.ctrl_mode == "End_Pose":
            # pdb.set_trace()
            left_end_pos = self.bus_left.sync_read(self.ctrl_mode)
            right_end_pos = self.bus_right.sync_read(self.ctrl_mode)
            obs_dict = ({f"{left_end_pos}.value": val for left_end_pos, val in left_end_pos.items()})
            obs_dict.update({f"{right_end_pos}.value": val for right_end_pos, val in right_end_pos.items()})

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read dual-arm state: {dt_ms:.1f}ms")

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            image = cam.async_read()
            if image is not None:
                obs_dict[cam_key] = image
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict
    @property
    def is_connected(self) -> bool:
        
    def connect(self, calibrate: bool = True) -> None:
        self.piper_left.ConnectPort()
        self.piper_right.ConnectPort()
        piper_left.MotionCtrl_2(0x01, 0x00, 100, 0x00)
        piper_right.MotionCtrl_2(0x01, 0x00, 100, 0x00)
    def disconnect(self) -> None:
        self.piper_left.DisconnectPort()
        self.piper_right.DisconnectPort()
    def calibrate(self) -> None:
        def configure(self) -> None:
        # Piper 的基础电机配置已在 bus.connect() 内完成
        pass
    def reset(self):