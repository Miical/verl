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
import pdb
import logging
import time
from functools import cached_property
from pathlib import Path
from typing import Dict, Any
from traceback import format_exc

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.motors.piper import (
    PiperMotorsBus,
)
from lerobot.cameras.opencv import OpenCVCamera  # TODO

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_piper_follower import PiperFollowerConfig

logger = logging.getLogger(__name__)


class PiperFollower(Robot):
    """Piper双臂跟随机器人实现 - 继承Robot抽象基类"""

    # 必须的类属性
    config_class = PiperFollowerConfig
    name = "piper_follower"

    def __init__(self, config: PiperFollowerConfig, **kwargs):
        super().__init__(config, **kwargs)

        self.config = config

        # 初始化双臂电机总线
        self.bus_left = PiperMotorsBus(
            can_name=config.can_name_left,
            baud_rate=config.baud_rate,
            motor_prefix='left'
        )
        self.bus_right = PiperMotorsBus(
            can_name=config.can_name_right,
            baud_rate=config.baud_rate,
            motor_prefix='right'
        )

        # 初始化相机系统 # TODO
        self.cameras = make_cameras_from_configs(config.cameras)

        # 运动学模型 (如果有URDF)
        self.kinematics = None
        # if hasattr(config, 'urdf_path') and config.urdf_path and Path(config.urdf_path).exists():
        #     try:
        #         from lerobot.model.kinematics import RobotKinematics
        #         self.kinematics = RobotKinematics(
        #             urdf_path=config.urdf_path,
        #             target_frame_name=getattr(config, 'target_frame_name', 'end_effector')  # TODO: check
        #         )
        #     except Exception as e:
        #         logging.warning(f"Failed to load kinematics: {format_exc()}")

        logging.info("PiperFollower (dual-arm) initialized")

    # 必须实现的抽象属性
    @property
    def _motors_ft(self) -> dict[str, type]:
        """双臂电机特征字典"""
        left_motors = {f"{motor}.pos": float for motor in self.bus_left.motors}
        right_motors = {f"{motor}.pos": float for motor in self.bus_right.motors}
        # left_motors = {f"left_{motor}.pos": float for motor in self.bus_left.motors}
        # right_motors = {f"right_{motor}.pos": float for motor in self.bus_right.motors}
        return {**left_motors, **right_motors}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        """相机特征字典""" # TODO
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
            for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        """观测特征描述 - Robot抽象属性"""
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        """动作特征描述 - Robot抽象属性"""
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        """检查连接状态 - Robot抽象属性"""
        return (self.bus_left.is_connected and self.bus_right.is_connected and
                all(cam.is_connected for cam in self.cameras.values()))

    @property
    def is_calibrated(self) -> bool:
        """检查校准状态 - Robot抽象属性"""
        # Piper 机械臂通常不需要传统意义的校准，检查是否已正确初始化
        return self.is_connected

    # 必须实现的抽象方法
    def connect(self, calibrate: bool = True) -> None:
        """连接到双臂机器人 - Robot抽象方法"""
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")
        
        # 连接双臂
        left_success = self.bus_left.connect(
            start_sdk_joint_limit=self.config.start_sdk_joint_limit,
            start_sdk_gripper_limit=self.config.start_sdk_gripper_limit,
            move_spd_rate_ctrl=self.config.move_spd_rate_ctrl
        )
        right_success = self.bus_right.connect(
            start_sdk_joint_limit=self.config.start_sdk_joint_limit,
            start_sdk_gripper_limit=self.config.start_sdk_gripper_limit,
            move_spd_rate_ctrl=self.config.move_spd_rate_ctrl
        )

        if not (left_success and right_success):
            raise ConnectionError(f"Failed to connect to {self}")

        # 连接相机 # TODO
        for cam in self.cameras.values():
            cam.connect()

        # 如果需要校准
        if not self.is_calibrated and calibrate:
            self.calibrate()

        self.configure()
        logging.info(f"{self} connected.")

    def calibrate(self) -> None:
        """校准机器人 - Robot抽象方法"""
        logging.info(f"Calibrating {self}")
        # Piper 机械臂的校准逻辑：重置到安全位置
        try:
            self.reset()
            logging.info("Piper dual-arm robot calibration completed")
        except Exception as e:
            logging.error(f"Calibration failed: {format_exc()}")

    def configure(self) -> None:
        """配置机器人 - Robot抽象方法"""
        logging.info(f"Configuring {self}")
        # Piper 机械臂的配置在 bus.connect() 中的 _initialize_robot() 已完成
        # 这里可以添加额外的双臂协调配置
        pass

    def get_observation(self) -> Dict[str, Any]:
        """获取观测数据 - Robot抽象方法"""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        try:
            observation = {}
            """
            observation = {
                "left_joint1.pos": ...,
                "left_joint2.pos": ...,
                ...
                "left_gripper.pos": ...,

                "right_joint1.pos": ...,
                "right_joint2.pos": ...,
                ...
                "right_gripper.pos": ...,

                "camera_name": image,
                ...
            }
            """

            # 获取双臂关节位置
            start = time.perf_counter()

            # 左臂位置
            left_positions = self.bus_left.sync_read("Present_Position")
            for motor_name in self.bus_left.motors:
                observation[f"{motor_name}.pos"] = left_positions.get(motor_name, 0.0)

            # 右臂位置
            right_positions = self.bus_right.sync_read("Present_Position")
            for motor_name in self.bus_right.motors:
                observation[f"{motor_name}.pos"] = right_positions.get(motor_name, 0.0)

            dt_ms = (time.perf_counter() - start) * 1e3
            logging.debug(f"{self} read dual-arm state: {dt_ms:.1f}ms")

            # 获取图像 # TODO
            for camera_name, camera in self.cameras.items():
                start = time.perf_counter()
                image = camera.async_read()
                if image is not None:
                    observation[camera_name] = image
                dt_ms = (time.perf_counter() - start) * 1e3
                logging.debug(f"{self} read {camera_name}: {dt_ms:.1f}ms")

            return observation
        except Exception as e:
            logging.error(f"Failed to get leader action: {format_exc()}")
            return {f"{motor}.pos": 0.0 for motor in self.bus_left.motors} | \
                   {f"{motor}.pos": 0.0 for motor in self.bus_right.motors}

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """发送动作指令 - Robot抽象方法
        action = {
            "left_joint1.pos": ...,
            "left_joint2.pos": ...,
            ...
            "left_gripper.pos": ...,

            "right_joint1.pos": ...,
            "right_joint2.pos": ...,
            ...
            "right_gripper.pos": ...,
        }
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # 分离左臂和右臂的动作
        left_goal_pos = {}
        right_goal_pos = {}

        # 解析动作指令
        for key, value in action.items():
            if key.startswith("left_") and key.endswith(".pos"):
                motor = key[:-len(".pos")]
                left_goal_pos[motor] = value
            elif key.startswith("right_") and key.endswith(".pos"):
                motor = key[:-len(".pos")]
                right_goal_pos[motor] = value

        # 安全限制 (如果配置了)
        if self.config.max_relative_target is not None:
            # 左臂安全检查
            if left_goal_pos:
                left_present_pos = self.bus_left.sync_read("Present_Position")
                left_goal_present_pos = {motor: (g_pos, left_present_pos[motor]) for motor, g_pos in left_goal_pos.items()}
                left_goal_pos = ensure_safe_goal_position(left_goal_present_pos, self.config.max_relative_target)

            # 右臂安全检查
            if right_goal_pos:
                right_present_pos = self.bus_right.sync_read("Present_Position")
                right_goal_present_pos = {motor: (g_pos, right_present_pos[motor]) for motor, g_pos in right_goal_pos.items()}
                right_goal_pos = ensure_safe_goal_position(right_goal_present_pos, self.config.max_relative_target)

        try:
            # 发送双臂指令
            if left_goal_pos:
                self.bus_left.sync_write("Goal_Position", left_goal_pos)
            if right_goal_pos:
                self.bus_right.sync_write("Goal_Position", right_goal_pos)

            # 返回实际发送的动作
            sent_action = {
                "left_arm": left_goal_pos,
                "right_arm": right_goal_pos
            }

            return sent_action

        except Exception as e:
            logging.error(f"Failed to send dual-arm action: {format_exc()}")
            return {"left_arm": {}, "right_arm": {}}

    def disconnect(self) -> None:
        """断开连接 - Robot抽象方法"""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # 断开双臂连接
        self.bus_left.disconnect(self.config.reset_pos_on_disconnect, self.config.disable_torque_on_disconnect)
        self.bus_right.disconnect(self.config.reset_pos_on_disconnect, self.config.disable_torque_on_disconnect)

        # 断开相机连接 # TODO
        for camera in self.cameras.values():
            camera.disconnect()

        logging.info(f"{self} disconnected.")

    def reset(self):
        """重置双臂机器人到初始位置"""
        try:
            # 双臂复位
            self.bus_left.reset_pos()
            self.bus_right.reset_pos()

            # 等待到达位置
            time.sleep(5.0)

            logging.info("Dual-arm robot reset to initial position")

        except Exception as e:
            logging.error(f"Failed to reset dual-arm robot: {format_exc()}")