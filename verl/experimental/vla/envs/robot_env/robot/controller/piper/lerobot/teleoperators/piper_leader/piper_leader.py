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

import logging
import time
from traceback import format_exc
from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.motors.piper import (
    PiperMotorsBus,
)

from ..teleoperator import Teleoperator
from .config_piper_leader import PiperLeaderConfig

logger = logging.getLogger(__name__)


class PiperLeader(Teleoperator):
    """Piper双臂领导机器人实现 - 继承Teleoperator抽象基类"""

    # 必须的类属性
    config_class = PiperLeaderConfig
    name = "piper_leader"

    def __init__(self, config: PiperLeaderConfig, **kwargs):
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

        logging.info("PiperLeader (dual-arm) initialized")

    # 必须实现的抽象属性
    @property
    def action_features(self) -> dict[str, type]:
        """动作特征描述 - Teleoperator抽象属性"""
        left_motors = {f"{motor}.pos": float for motor in self.bus_left.motors}
        right_motors = {f"{motor}.pos": float for motor in self.bus_right.motors}
        return {**left_motors, **right_motors}

    @property
    def feedback_features(self) -> dict[str, type]:
        """反馈特征描述 - Teleoperator抽象属性"""
        return {}  # Piper暂不支持力反馈

    @property
    def is_connected(self) -> bool:
        """检查连接状态 - Teleoperator抽象属性"""
        return self.bus_left.is_connected and self.bus_right.is_connected

    @property
    def is_calibrated(self) -> bool:
        """检查校准状态"""
        return self.is_connected

    # 必须实现的抽象方法
    def connect(self, calibrate: bool = True) -> None:
        """连接到双臂领导机器人 - Teleoperator抽象方法"""
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        # 连接双臂
        left_success = self.bus_left.connect()
        right_success = self.bus_right.connect()

        if not (left_success and right_success):
            raise ConnectionError(f"Failed to connect to {self}")

        if not self.is_calibrated and calibrate:
            self.calibrate()

        self.configure()
        logging.info(f"{self} connected.")

    def calibrate(self) -> None:
        """校准领导机器人 - Teleoperator抽象方法"""
        logging.info(f"Calibrating {self}")
        try:
            pass  # Piper 主臂双臂暂不需要额外校准
            # # 设置为主从模式 TODO: 是否需要配置？
            # if self.config.enable_master_slave:
            #     self.bus_left.interface.MasterSlaveConfig(
            #         linkage_config=self.config.linkage_config,
            #         feedback_offset=0x00,
            #         ctrl_offset=0x00,
            #         linkage_offset=0x00
            #     )
            #     self.bus_right.interface.MasterSlaveConfig(
            #         linkage_config=self.config.linkage_config,
            #         feedback_offset=0x00,
            #         ctrl_offset=0x00,
            #         linkage_offset=0x00
            #     )

            logging.info("Dual-arm leader calibration completed")
        except Exception as e:
            logging.error(f"Leader calibration failed: {format_exc()}")

    def configure(self) -> None:
        """配置领导机器人 - Teleoperator抽象方法"""
        logging.info(f"Configuring {self}")
        # Piper 机械臂的配置在 bus.connect() 中的 _initialize_robot() 已完成
        # 这里可以添加额外的双臂协调配置
        pass

    def get_action(self) -> dict[str, float]:
        """获取遥操控动作 - Teleoperator抽象方法"""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        try:
            start = time.perf_counter()

            # 读取双臂位置作为动作
            left_action = self.bus_left.sync_read("Present_Position")
            right_action = self.bus_right.sync_read("Present_Position")

            # 格式化动作
            action = {}
            for motor_name in self.bus_left.motors:
                action[f"{motor_name}.pos"] = left_action.get(motor_name, 0.0)
            for motor_name in self.bus_right.motors:
                action[f"{motor_name}.pos"] = right_action.get(motor_name, 0.0)

            dt_ms = (time.perf_counter() - start) * 1e3
            logging.debug(f"{self} read dual-arm action: {dt_ms:.1f}ms")

            return action

        except Exception as e:
            logging.error(f"Failed to get leader action: {format_exc()}")
            return {f"{motor}.pos": 0.0 for motor in self.bus_left.motors} | \
                   {f"{motor}.pos": 0.0 for motor in self.bus_right.motors}

    def send_feedback(self, feedback: dict[str, float]) -> None:
        """发送反馈 - Teleoperator抽象方法"""
        # Piper机械臂暂不支持力反馈
        raise NotImplementedError("Piper does not support force feedback yet")

    def disconnect(self) -> None:
        """断开连接 - Teleoperator抽象方法"""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.bus_left.disconnect()
        self.bus_right.disconnect()

        logging.info(f"{self} disconnected.")

