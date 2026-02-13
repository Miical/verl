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
import math
from .piper_follower import PiperFollower
from .config_piper_follower_end_effector import PiperFollowerEndEffectorConfig

logger = logging.getLogger(__name__)

POS_LIMIT = 0.3                    # 米
ROT_LIMIT = math.pi / 6.0          # 弧度，30度

_POS_KEYS = {"end_x", "end_y", "end_z"}
_ROT_KEYS = {"end_rx", "end_ry", "end_rz"}

def _skew(v: np.ndarray) -> np.ndarray:
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    return np.array([[0.0, -z, y],
                     [z, 0.0, -x],
                     [-y, x, 0.0]], dtype=np.float64)


def _exp_so3(w: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(w))
    if theta < 1e-9:
        return np.eye(3, dtype=np.float64) + _skew(w)
    K = _skew(w / theta)
    return np.eye(3, dtype=np.float64) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


class PiperFollowerEndEffector(PiperFollower):
    """Piper 双臂末端 6-DoF 控制：仅平移时 IK 只约束位置；显式转动时才启用姿态权重。"""

    config_class = PiperFollowerEndEffectorConfig
    name = "piper_follower_end_effector"

    def __init__(self, config: PiperFollowerEndEffectorConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config
        self.ctrl_mode = "End_Pose"
        if self.config.urdf_path is None:
            raise ValueError("必须提供 urdf_path 才能进行末端控制。")

        # 可被 config 覆盖的默认参数
        if not hasattr(self.config, "end_effector_rot_step_sizes"):
            self.config.end_effector_rot_step_sizes = {"rx": 0.003, "ry": 0.003, "rz": 0.003}
        if not hasattr(self.config, "ik_orientation_weight"):
            self.config.ik_orientation_weight = 1.0
        if not hasattr(self.config, "ik_position_weight"):
            self.config.ik_position_weight = 1.0
        if not hasattr(self.config, "orientation_activation_eps"):
            self.config.orientation_activation_eps = 1e-9

        # 🔧 使用绝对路径，基于当前文件位置计算
        import os
        from pathlib import Path
        
        # 基于当前文件位置计算 local_assets 目录的绝对路径
        # piper_follower_end_effector.py 位于 .../lerobot/robots/piper_follower/
        # local_assets 位于 .../local_assets/
        current_file = Path(__file__).resolve()
        piper_follower_dir = current_file.parent  # .../lerobot/robots/piper_follower/
        lerobot_dir = piper_follower_dir.parent.parent  # .../lerobot/
        piper_dir = lerobot_dir.parent  # .../piper/
        local_assets_dir = piper_dir / "local_assets"
        
        logger.info(f"[PiperFollowerEndEffector] Current file: {current_file}")
        logger.info(f"[PiperFollowerEndEffector] local_assets dir (absolute): {local_assets_dir}")
        logger.info(f"[PiperFollowerEndEffector] Directory exists: {local_assets_dir.exists()}")
        logger.info(f"[PiperFollowerEndEffector] robot.urdf exists: {(local_assets_dir / 'robot.urdf').exists()}")
        logger.info(f"[PiperFollowerEndEffector] meshes/ exists: {(local_assets_dir / 'meshes').exists()}")
        
        # 🔧 RobotKinematics 内部会自动处理工作目录切换
        # 只需传入 local_assets 目录的绝对路径
        kinL = RobotKinematics(urdf_path=str(local_assets_dir), target_frame_name=self.config.target_frame_name)
        kinR = RobotKinematics(urdf_path=str(local_assets_dir), target_frame_name=self.config.target_frame_name)
        
        logger.info(f"[PiperFollowerEndEffector] RobotKinematics initialized successfully!")

        model_joint_order = ["joint1","joint2","joint3","joint4","joint5","joint6","joint7","joint8"]
        for kin in (kinL, kinR):
            if hasattr(kin, "joint_names"):
                try:
                    kin.joint_names = model_joint_order
                except Exception:
                    pass
        
        self.cameras = make_cameras_from_configs(config.cameras)

        self.sides = {
            "left": {
                "bus": self.bus_left, "kin": kinL,
                "arm_keys": [f"left_joint_{i}" for i in range(1, 7)],
                "grip_key": "left_gripper",
                "q6": None, "width": None, "T": None,
            },
            "right": {
                "bus": self.bus_right, "kin": kinR,
                "arm_keys": [f"right_joint_{i}" for i in range(1, 7)],
                "grip_key": "right_gripper",
                "q6": None, "width": None, "T": None,
            },
        }
        # pdb.set_trace()
        self._urdf_finger_max = 0.04

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

    # ---------- 工具函数 ----------



    def _grip_next_width(self, cur_w: float, cmd: float) -> float:
        return float(np.clip(cur_w + (float(cmd) - 1.0) * self.config.max_gripper_pos,
                             0.0, self.config.max_gripper_pos))

    # ---------- 主流程 ----------

    def send_action(self, action: dict[str, Any] | np.ndarray) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConectedError(f"{self} is not connected.")

        #分离左右臂姿态
        left_goal_end_pose = {}
        right_goal_end_pose = {}
        # pdb.set_trace()
        #解析动作指令
        for key, value in action.items():
            if key.startswith("left_") and key.endswith(".value"):
                end_pose_name = key[:-len(".value")]
                left_goal_end_pose[end_pose_name] = value
            elif key.startswith("right_") and key.endswith(".value"):
                end_pose_name = key[:-len(".value")]
                right_goal_end_pose[end_pose_name] = value

        #print(f"piper send action: left_goal_end_pose: {left_goal_end_pose}, right_goal_end_pose: {right_goal_end_pose}")
        #发送左右臂指令
        try:
            # 发送双臂指令
            # pdb.set_trace()
            if left_goal_end_pose:
                #print("left_goal_end_pose:", left_goal_end_pose)
                self.bus_left.sync_write("End_Pose", left_goal_end_pose)
            if right_goal_end_pose:
                # print("right_goal_end_pose:", right_goal_end_pose)
                self.bus_right.sync_write("End_Pose", right_goal_end_pose)

            # 返回实际发送的动作
            sent_action = {
                "left_end_pose": left_goal_end_pose,
                "right_end_pose": right_goal_end_pose
            }

            return sent_action

        except Exception as e:
            logging.error(f"Failed to send dual-arm action: {format_exc()}")
            return {"left_arm": {}, "right_arm": {}}
            

    def reset(self):
        print("机器人reset不走这")
        # pass
        # super().reset()


    # def get_observation(self) -> dict[str, Any]:
    #     if not self.is_connected:
    #         raise DeviceNotConnectedError(f"{self} is not connected.")

    #     # Read arm position from both arms
    #     start = time.perf_counter()
    #     if self.ctrl_mode == "Present_Position":
    #         # 左臂位置
    #         left_positions = self.bus_left.sync_read(self.ctrl_mode)
    #         obs_dict = {f"{motor}.pos": val for motor, val in left_positions.items()}
            
    #         # 右臂位置
    #         right_positions = self.bus_right.sync_read(self.ctrl_mode)
    #         obs_dict.update({f"{motor}.pos": val for motor, val in right_positions.items()})
    #     elif self.ctrl_mode == "End_Pose":
    #         # pdb.set_trace()
    #         left_end_pos = self.bus_left.sync_read(self.ctrl_mode)
    #         right_end_pos = self.bus_right.sync_read(self.ctrl_mode)
    #         obs_dict = ({f"{left_end_pos}.value": val for left_end_pos, val in left_end_pos.items()})
    #         obs_dict.update({f"{right_end_pos}.value": val for right_end_pos, val in right_end_pos.items()})

    #     dt_ms = (time.perf_counter() - start) * 1e3
    #     logger.debug(f"{self} read dual-arm state: {dt_ms:.1f}ms")

    #     # Capture images from cameras
    #     for cam_key, cam in self.cameras.items():
    #         start = time.perf_counter()
    #         image = cam.async_read()
    #         if image is not None:
    #             obs_dict[cam_key] = image
    #         dt_ms = (time.perf_counter() - start) * 1e3
    #         logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

    #     return obs_dict
    def get_observation(self, mode = None) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        
        use_mode = mode if mode is not None else self.ctrl_mode
        # Read arm position from both arms
        start = time.perf_counter()

        if use_mode == "Present_Position":
            # 左臂位置
            left_positions = self.bus_left.sync_read(use_mode)
            obs_dict = {f"{motor}.pos": val for motor, val in left_positions.items()}
            
            # 右臂位置
            right_positions = self.bus_right.sync_read(use_mode)
            obs_dict.update({f"{motor}.pos": val for motor, val in right_positions.items()})
        elif use_mode == "End_Pose":
            tmp_use_mode = "Present_Position"

            left_joint_pos = self.bus_left.sync_read(tmp_use_mode)
            right_joint_pos = self.bus_right.sync_read(tmp_use_mode)

            print("left_joint_pos:", left_joint_pos)
            print("right_joint_pos:", right_joint_pos)

            obs_dict = {}

            # ===== left arm: joint -> fake end pose =====
            if left_joint_pos is not None:
                obs_dict.update({
                    "left_end_x.value":  left_joint_pos.get("left_joint_1", 0.0),
                    "left_end_y.value":  left_joint_pos.get("left_joint_2", 0.0),
                    "left_end_z.value":  left_joint_pos.get("left_joint_3", 0.0),
                    "left_end_rx.value": left_joint_pos.get("left_joint_4", 0.0),
                    "left_end_ry.value": left_joint_pos.get("left_joint_5", 0.0),
                    "left_end_rz.value": left_joint_pos.get("left_joint_6", 0.0),
                    "left_end_gripper.value": left_joint_pos.get("left_gripper", 0.0),
                })

            # ===== right arm: joint -> fake end pose =====
            if right_joint_pos is not None:
                obs_dict.update({
                    "right_end_x.value":  right_joint_pos.get("right_joint_1", 0.0),
                    "right_end_y.value":  right_joint_pos.get("right_joint_2", 0.0),
                    "right_end_z.value":  right_joint_pos.get("right_joint_3", 0.0),
                    "right_end_rx.value": right_joint_pos.get("right_joint_4", 0.0),
                    "right_end_ry.value": right_joint_pos.get("right_joint_5", 0.0),
                    "right_end_rz.value": right_joint_pos.get("right_joint_6", 0.0),
                    "right_end_gripper.value": right_joint_pos.get("right_gripper", 0.0),
                })

            # ===== optional debug flag =====
            obs_dict["_end_pose_is_fake"] = True

        # elif use_mode == "End_Pose":
        #     tmp_use_mode = "Present_Position"
        #     left_end_pos = self.bus_left.sync_read(tmp_use_mode)
        #     right_end_pos = self.bus_right.sync_read(tmp_use_mode)
        #     # pdb.set_trace()
        #     # left_end_pos = self.bus_left.sync_read(use_mode)
        #     # right_end_pos = self.bus_right.sync_read(use_mode)
        #     obs_dict = ({f"{left_end_pos}.value": val for left_end_pos, val in left_end_pos.items()})
        #     obs_dict.update({f"{right_end_pos}.value": val for right_end_pos, val in right_end_pos.items()})

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