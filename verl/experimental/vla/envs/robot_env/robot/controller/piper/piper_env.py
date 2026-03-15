# !/usr/bin/env python

import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Any

import cv2
import gymnasium as gym
import numpy as np
import torch

logging.getLogger("can.interfaces.socketcan").setLevel(logging.ERROR)

# 仅把 piper 本地目录加入 path，避免依赖外部安装的整套 lerobot 包
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if _CURRENT_DIR not in sys.path:
    sys.path.insert(0, _CURRENT_DIR)

# ===== 仅依赖三部分：camera / motor / teleop =====
from lerobot.motors.piper.piper import PiperMotorsBus

try:
    from lerobot.cameras.dabai.camera_dabai import OrbbecDabaiCamera
    from lerobot.cameras.dabai.configuration_dabai import OrbbecDabaiCameraConfig
except Exception:  # pragma: no cover - 运行环境可能没装 dabai SDK
    OrbbecDabaiCamera = None
    OrbbecDabaiCameraConfig = None

try:
    from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
    from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
except Exception:  # pragma: no cover - 运行环境可能没装 realsense SDK
    RealSenseCamera = None
    RealSenseCameraConfig = None

try:
    from lerobot.teleoperators.pico_vr.teleop_pico_vr import PicoVrTeleop
    from lerobot.teleoperators.pico_vr.config_pico_vr import PicoVrTeleopConfig
except Exception:  # pragma: no cover - 运行环境可能没装 zmq 或 teleop 依赖
    PicoVrTeleop = None
    PicoVrTeleopConfig = None


def force_print(*args, **kwargs):
    print(*args, **kwargs, flush=True)
    sys.stdout.flush()
    sys.stderr.flush()


def images_encoding(imgs: list[np.ndarray]):
    encode_data = []
    max_len = 0
    for img in imgs:
        success, encoded_image = cv2.imencode(".jpg", img)
        if not success:
            encoded_image = np.zeros((1,), dtype=np.uint8)
        jpeg_data = encoded_image.tobytes()
        encode_data.append(jpeg_data)
        max_len = max(max_len, len(jpeg_data))
    return encode_data, max_len


@dataclass
class _DefaultCfg:
    can_name_left: str = "can0"
    can_name_right: str = "can1"
    baud_rate: int = 1_000_000
    device: str = "cpu"
    num_envs: int = 1
    task_description: str = "catch_bowl"
    camera_mapping: dict[str, str] | None = None


class PiperJointRobot:
    """最小双臂 Piper 机器人封装：joint 控制 + 3 相机 + pico_vr 干预信号。"""

    def __init__(self, cfg: Any):
        self.cfg = cfg
        self.bus_left = PiperMotorsBus(cfg.can_name_left, cfg.baud_rate, motor_prefix="left")
        self.bus_right = PiperMotorsBus(cfg.can_name_right, cfg.baud_rate, motor_prefix="right")

        self.cam_front = self._build_dabai_camera(cfg)
        self.cam_left = self._build_realsense_camera(cfg, side="left")
        self.cam_right = self._build_realsense_camera(cfg, side="right")

        self.teleop = self._build_teleop(cfg)
        self.is_connected = False

    def _build_dabai_camera(self, cfg: Any):
        if OrbbecDabaiCamera is None or OrbbecDabaiCameraConfig is None:
            return None
        cam_cfg = getattr(cfg, "dabai_camera", None)
        if cam_cfg is None:
            return None
        return OrbbecDabaiCamera(OrbbecDabaiCameraConfig(**cam_cfg))

    def _build_realsense_camera(self, cfg: Any, side: str):
        if RealSenseCamera is None or RealSenseCameraConfig is None:
            return None
        key = f"realsense_{side}_camera"
        cam_cfg = getattr(cfg, key, None)
        if cam_cfg is None:
            return None
        return RealSenseCamera(RealSenseCameraConfig(**cam_cfg))

    def _build_teleop(self, cfg: Any):
        if PicoVrTeleop is None or PicoVrTeleopConfig is None:
            return None
        teleop_cfg = getattr(cfg, "pico_vr", None)
        if teleop_cfg is None:
            return None
        return PicoVrTeleop(PicoVrTeleopConfig(**teleop_cfg))

    def connect(self):
        if self.is_connected:
            return
        self.bus_left.connect()
        self.bus_right.connect()
        for cam in [self.cam_front, self.cam_left, self.cam_right]:
            if cam is not None:
                try:
                    cam.connect()
                except Exception as exc:
                    force_print(f"[PiperJointRobot] camera connect failed: {exc}")
        if self.teleop is not None:
            try:
                self.teleop.connect()
            except Exception as exc:
                force_print(f"[PiperJointRobot] teleop connect failed: {exc}")
        self.is_connected = True

    def disconnect(self):
        for cam in [self.cam_front, self.cam_left, self.cam_right]:
            if cam is not None and getattr(cam, "is_connected", False):
                try:
                    cam.disconnect()
                except Exception:
                    pass
        if self.teleop is not None and getattr(self.teleop, "is_connected", False):
            try:
                self.teleop.disconnect()
            except Exception:
                pass
        self.bus_left.disconnect()
        self.bus_right.disconnect()
        self.is_connected = False

    @staticmethod
    def _safe_read_image(cam, default_shape=(224, 224, 3)):
        if cam is None:
            return np.zeros(default_shape, dtype=np.uint8)
        try:
            img = cam.async_read()
            if img is None:
                return np.zeros(default_shape, dtype=np.uint8)
            return img
        except Exception:
            return np.zeros(default_shape, dtype=np.uint8)

    def get_observation(self) -> dict[str, Any]:
        left_pos = self.bus_left.sync_read("Present_Position")
        right_pos = self.bus_right.sync_read("Present_Position")

        left_state = [left_pos.get(k, 0.0) for k in self.bus_left.motors]
        right_state = [right_pos.get(k, 0.0) for k in self.bus_right.motors]
        state = np.asarray(left_state + right_state, dtype=np.float32)

        return {
            "front": self._safe_read_image(self.cam_front),
            "left": self._safe_read_image(self.cam_left),
            "right": self._safe_read_image(self.cam_right),
            "state": state,
        }

    def apply_joint_action(self, action_14: np.ndarray):
        action_14 = np.asarray(action_14, dtype=np.float32).reshape(-1)
        if action_14.shape[0] < 14:
            action_14 = np.pad(action_14, (0, 14 - action_14.shape[0]), mode="constant")
        elif action_14.shape[0] > 14:
            action_14 = action_14[:14]

        left_targets = dict(zip(self.bus_left.motors, action_14[:7], strict=False))
        right_targets = dict(zip(self.bus_right.motors, action_14[7:14], strict=False))
        self.bus_left.sync_write("Goal_Position", left_targets)
        self.bus_right.sync_write("Goal_Position", right_targets)

    def get_intervention_action(self) -> tuple[bool, np.ndarray | None]:
        if self.teleop is None or not getattr(self.teleop, "is_connected", False):
            return False, None

        try:
            events = self.teleop.get_teleop_events()
            is_intervention = bool(events.get("is_intervention", False))
        except Exception:
            is_intervention = False

        if not is_intervention:
            return False, None

        try:
            teleop_action = self.teleop.get_action()
        except Exception:
            return True, None

        if isinstance(teleop_action, np.ndarray):
            return True, teleop_action.astype(np.float32)
        if isinstance(teleop_action, torch.Tensor):
            return True, teleop_action.detach().cpu().numpy().astype(np.float32)
        if isinstance(teleop_action, dict):
            # 兼容 dict 格式，按已有 14 维 joint 顺序抽取
            ordered_keys = [
                "left_joint_1", "left_joint_2", "left_joint_3", "left_joint_4", "left_joint_5", "left_joint_6", "left_gripper",
                "right_joint_1", "right_joint_2", "right_joint_3", "right_joint_4", "right_joint_5", "right_joint_6", "right_gripper",
            ]
            arr = np.asarray([float(teleop_action.get(k, 0.0)) for k in ordered_keys], dtype=np.float32)
            return True, arr

        return True, None


class PiperJointEnv(gym.Env):
    """极简 gym 环境：action 为 14 维双臂 joint，obs 为 state + 三路图像。"""

    def __init__(self, robot: PiperJointRobot):
        super().__init__()
        self.robot = robot
        self.current_step = 0
        self.action_space = gym.spaces.Box(low=-np.ones(14), high=np.ones(14), shape=(14,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Box(low=-10, high=10, shape=(14,), dtype=np.float32),
                "front": gym.spaces.Box(low=0, high=255, shape=(224, 224, 3), dtype=np.uint8),
                "left": gym.spaces.Box(low=0, high=255, shape=(224, 224, 3), dtype=np.uint8),
                "right": gym.spaces.Box(low=0, high=255, shape=(224, 224, 3), dtype=np.uint8),
            }
        )

    def reset(self, *, seed=None, options=None):
        del seed, options
        self.current_step = 0
        if not self.robot.is_connected:
            self.robot.connect()
        obs = self.robot.get_observation()
        return obs, {"is_intervention": False}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)

        intervene_flag, intervene_action = self.robot.get_intervention_action()
        final_action = intervene_action if (intervene_flag and intervene_action is not None) else action

        self.robot.apply_joint_action(final_action)
        obs = self.robot.get_observation()
        self.current_step += 1

        info = {
            "is_intervention": intervene_flag,
            "intervene_action": intervene_action,
            "applied_action": final_action,
        }
        return obs, 0.0, False, False, info

    def close(self):
        self.robot.disconnect()


class RealRobotEnvWrapper:
    """对外兼容封装：reset/step 输出 test_env 侧预期字段。"""

    def __init__(self, cfg, rank: int = 0, world_size: int = 1):
        force_print(f"[RealRobotEnvWrapper] init rank={rank}, world_size={world_size}")
        self.rank = rank
        self.world_size = world_size

        if cfg is None:
            cfg = _DefaultCfg()
        if not hasattr(cfg, "can_name_left"):
            cfg = _DefaultCfg(**getattr(cfg, "__dict__", {}))

        self.device = getattr(cfg, "device", "cpu")
        self.num_envs = getattr(cfg, "num_envs", 1)
        self.task_description = getattr(cfg, "task_description", "catch_bowl")
        self.camera_mapping = getattr(
            cfg,
            "camera_mapping",
            {"front": "head_image", "left": "left_wrist_image", "right": "right_wrist_image"},
        )

        self.robot = PiperJointRobot(cfg)
        self.env = PiperJointEnv(self.robot)

    def _convert_obs_to_test_env_format(self, obs: dict[str, Any]) -> dict[str, Any]:
        images_list = []
        camera_names = []
        for robot_cam_name, test_cam_name in self.camera_mapping.items():
            if robot_cam_name in obs:
                images_list.append(obs[robot_cam_name])
                camera_names.append(test_cam_name)

        default_image = np.zeros((224, 224, 3), dtype=np.uint8)
        for test_cam_name in ["head_image", "left_wrist_image", "right_wrist_image"]:
            if test_cam_name not in camera_names:
                images_list.append(default_image)
                camera_names.append(test_cam_name)

        encoded_data, max_len = images_encoding(images_list)
        images_dict = {}
        for i, cam_name in enumerate(camera_names):
            padded_bytes = encoded_data[i].ljust(max_len, b"\0")
            img_array = np.array(np.frombuffer(padded_bytes, dtype=np.uint8))
            images_dict[cam_name] = torch.from_numpy(img_array).unsqueeze(0)

        return {
            "images": images_dict,
            "state": obs["state"].astype(np.float32),
            "task_description": self.task_description,
        }

    @staticmethod
    def _normalize_action(action: Any) -> np.ndarray:
        if isinstance(action, torch.Tensor):
            action_np = action.detach().cpu().numpy().astype(np.float32)
        else:
            action_np = np.asarray(action, dtype=np.float32)

        action_np = action_np.reshape(-1)
        if action_np.shape[0] == 7:
            action_np = np.concatenate([action_np, np.zeros(7, dtype=np.float32)])
        elif action_np.shape[0] > 14:
            action_np = action_np[:14]
        elif action_np.shape[0] < 7:
            action_np = np.concatenate([action_np, np.zeros(14 - action_np.shape[0], dtype=np.float32)])
        return action_np

    def reset(self):
        obs, _ = self.env.reset()
        return self._convert_obs_to_test_env_format(obs)

    def step(self, action):
        action_np = self._normalize_action(action)
        obs, reward, terminated, truncated, info = self.env.step(action_np)

        obs_converted = self._convert_obs_to_test_env_format(obs)
        info_dict = {
            "intervene_action": info.get("intervene_action", None),
            "intervene_flag": info.get("is_intervention", False),
        }
        return obs_converted, float(reward), bool(terminated), bool(truncated), info_dict

    def close(self):
        self.env.close()


def make_robot_env(cfg) -> tuple[gym.Env, Any]:
    robot = PiperJointRobot(cfg)
    env = PiperJointEnv(robot)
    return env, robot.teleop


def make_processors(env, teleop_device, cfg, device: str = "cpu"):
    del env, teleop_device, cfg, device
    # 新实现不再依赖 lerobot processor pipeline，保留函数仅为了兼容调用
    return None, None


def step_env_and_process_transition(env, transition, action, env_processor=None, action_processor=None, save_images=False):
    del transition, env_processor, action_processor, save_images
    return env.step(action)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Piper joint env loader test")
    parser.add_argument("--can-left", default="can0", help="Left arm CAN interface")
    parser.add_argument("--can-right", default="can1", help="Right arm CAN interface")
    parser.add_argument("--baud-rate", type=int, default=1_000_000, help="CAN baud rate")
    parser.add_argument(
        "--run-reset-step",
        action="store_true",
        help="Run reset()+one zero-action step (requires real hardware)",
    )
    args = parser.parse_args()

    cfg = _DefaultCfg(
        can_name_left=args.can_left,
        can_name_right=args.can_right,
        baud_rate=args.baud_rate,
    )

    force_print("=" * 80)
    force_print("[PiperEnv Test] Start loading environment")
    force_print(f"[PiperEnv Test] cfg={{left:{cfg.can_name_left}, right:{cfg.can_name_right}, baud:{cfg.baud_rate}}}")

    wrapper = None
    try:
        wrapper = RealRobotEnvWrapper(cfg=cfg, rank=0, world_size=1)
        force_print("[PiperEnv Test] ✅ RealRobotEnvWrapper initialized successfully")

        if args.run_reset_step:
            force_print("[PiperEnv Test] Running reset() and one step() ...")
            obs = wrapper.reset()
            force_print(
                f"[PiperEnv Test] reset ok: keys={list(obs.keys())}, image_keys={list(obs['images'].keys())}"
            )
            zero_action = np.zeros((14,), dtype=np.float32)
            obs, reward, terminated, truncated, info = wrapper.step(zero_action)
            force_print(
                "[PiperEnv Test] step ok: "
                f"reward={reward}, terminated={terminated}, truncated={truncated}, intervene={info.get('intervene_flag')}"
            )
        else:
            force_print("[PiperEnv Test] Skip reset/step. Use --run-reset-step to test real hardware path.")

        force_print("[PiperEnv Test] ✅ Environment load test passed")
    except Exception as exc:
        force_print(f"[PiperEnv Test] ❌ Failed: {exc}")
        raise
    finally:
        if wrapper is not None:
            try:
                wrapper.close()
                force_print("[PiperEnv Test] wrapper closed")
            except Exception as close_exc:
                force_print(f"[PiperEnv Test] close warning: {close_exc}")
        force_print("=" * 80)
