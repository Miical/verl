#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2024 The HuggingFace Inc.
# Licensed under the Apache License, Version 2.0
import pdb
import logging
import time
from functools import cached_property
from typing import Any


from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.motors.piper import PiperMotorsBus

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_piper_single_follower import PiperSingleFollowerConfig

logger = logging.getLogger(__name__)


class PiperSingleFollower(Robot):
    """
    Piper 单臂 Follower 实现（与 SO100Follower 的 I/O 格式完全对齐）
    - 观测：{"{motor}.pos": float, ... , "<camera_name>": image}
    - 动作：send_action 接收 & 返回 {"{motor}.pos": float, ...}
    - 电机名：PiperMotorsBus.motors 内部已含侧前缀（如 'left_joint_1' 或 'right_joint_1'）
    """

    config_class = PiperSingleFollowerConfig
    name = "piper_single_follower"

    def __init__(self, config: PiperSingleFollowerConfig):
        super().__init__(config)
        self.config = config

        # 单臂总线（motor_prefix 使用 'left' 或 'right'）
        self.bus = PiperMotorsBus(
            can_name=config.can_name,
            baud_rate=config.baud_rate,
            motor_prefix=config.side
        )

        # 相机
        self.cameras = make_cameras_from_configs(config.cameras)

    # ---------------- Feature schema（对齐 SO100Follower） ----------------
    @property
    def _motors_ft(self) -> dict[str, type]:
        # { "{motor}.pos": float, ... }
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
            for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    # ---------------- 连接 / 校准 / 配置 ----------------
    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected and all(cam.is_connected for cam in self.cameras.values())

    @property
    def is_calibrated(self) -> bool:
        # Piper 无强制零位校准，连通即视为可用（与 SO100Follower 接口一致保留该属性）
        return self.is_connected

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        ok = self.bus.connect(
            start_sdk_joint_limit=self.config.start_sdk_joint_limit,
            start_sdk_gripper_limit=self.config.start_sdk_gripper_limit,
            move_spd_rate_ctrl=self.config.move_spd_rate_ctrl
        )
        if not ok:
            raise ConnectionError(f"Failed to connect to {self}")

        if not self.is_calibrated and calibrate:
            logger.info("No explicit calibration required for Piper; running reset() as a soft calibration.")
            self.calibrate()

        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        logger.info(f"{self} connected.")

    def calibrate(self) -> None:
        # 轻量“校准”= 复位到安全位姿
        try:
            self.reset()
        except Exception:
            logger.error("Calibration (soft reset) failed", exc_info=True)

    def configure(self) -> None:
        # Piper 的基础电机配置已在 bus.connect() 内完成
        pass

    # ---------------- 观测 / 动作（与 SO100Follower 完全对齐） ----------------
    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # 读取电机位置
        start = time.perf_counter()
        obs_dict = self.bus.sync_read("Present_Position")            # {motor: val, ...}
        obs_dict = {f"{m}.pos": v for m, v in obs_dict.items()}      # → { "motor.pos": val, ... }
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # 读取相机
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")
            #可视化
        # pdb.set_trace()

        import os
        import glob
        import cv2
        base_dir = "img_output/robot"
        os.makedirs(base_dir, exist_ok=True)
        
        image_keys = list(self.cameras.keys())

        for key in image_keys:
            img = obs_dict[key]
            save_dir = os.path.join(base_dir, key)
            os.makedirs(save_dir, exist_ok=True)

            # 保存当前帧
            filename = os.path.join(save_dir, f"{int(time.time() * 1000)}.jpg")
            cv2.imwrite(filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            # 只保留最新 5 张
            files = sorted(glob.glob(os.path.join(save_dir, "*.jpg")), key=os.path.getmtime)
            if len(files) > 5:
                for old_file in files[:-5]:
                    os.remove(old_file)

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """
        与 SO100Follower 一致：
        - 接收扁平 {"{motor}.pos": value}
        - 返回扁平 {"{motor}.pos": value}
        - 兼容：若传入键未带侧前缀（如 "joint_1.pos"），自动按 side 补前缀。
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        prefix = f"{self.config.side}_"

        # 解析输入为目标位置（去掉 ".pos" 后缀），并做侧前缀兼容
        goal_pos: dict[str, float] = {}
        for key, val in action.items():
            if not isinstance(key, str) or not key.endswith(".pos"):
                continue
            motor = key.removesuffix(".pos")
            # 自动补侧前缀：若未带前缀且存在对应电机
            if not motor.startswith(prefix) and f"{prefix}{motor}" in self.bus.motors:
                motor = f"{prefix}{motor}"
            if motor in self.bus.motors:
                goal_pos[motor] = float(val)

        # 相对位移裁剪（如配置了）
        if self.config.max_relative_target is not None and goal_pos:
            present_pos = self.bus.sync_read("Present_Position")
            goal_present_pos = {m: (g, present_pos[m]) for m, g in goal_pos.items() if m in present_pos}
            goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

        # 下发
        if goal_pos:
            self.bus.sync_write("Goal_Position", goal_pos)

        # 返回与 SO100Follower 对齐的扁平格式
        return {f"{m}.pos": v for m, v in goal_pos.items()}

    # ---------------- 断开 / 复位 ----------------
    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Piper 的断开需要两个标志位
        self.bus.disconnect(
            self.config.reset_pos_on_disconnect,
            self.config.disable_torque_on_disconnect
        )
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")

    def reset(self) -> None:
        try:
            self.bus.reset_pos()
            time.sleep(5.0)
            logger.info("Single-arm robot reset to initial position")
        except Exception:
            logger.error("Failed to reset single-arm robot", exc_info=True)
