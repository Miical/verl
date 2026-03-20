# !/usr/bin/env python

import logging
import os
import sys
import time
from types import SimpleNamespace
from typing import Any

import cv2
import gymnasium as gym
import numpy as np
import torch

logging.getLogger("can.interfaces.socketcan").setLevel(logging.ERROR)

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if _CURRENT_DIR not in sys.path:
    sys.path.insert(0, _CURRENT_DIR)

from lerobot.motors.piper.piper import PiperMotorsBus

try:
    from lerobot.cameras.dabai.camera_dabai import OrbbecDabaiCamera
    from lerobot.cameras.dabai.configuration_dabai import OrbbecDabaiCameraConfig
except Exception:
    OrbbecDabaiCamera = None
    OrbbecDabaiCameraConfig = None

try:
    from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
    from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
except Exception:
    RealSenseCamera = None
    RealSenseCameraConfig = None

try:
    from lerobot.teleoperators.pico_vr.teleop_pico_vr import PicoVrTeleop
    from lerobot.teleoperators.pico_vr.config_pico_vr import PicoVrTeleopConfig
except Exception:
    PicoVrTeleop = None
    PicoVrTeleopConfig = None


DEFAULT_CAMERA_MAPPING = {
    "front": "head_image",
    "left": "left_wrist_image",
    "right": "right_wrist_image",
}


def force_print(*args, **kwargs):
    print(*args, **kwargs, flush=True)
    sys.stdout.flush()
    sys.stderr.flush()


def images_encoding(imgs: list[np.ndarray]):
    encoded = []
    max_len = 0
    for img in imgs:
        ok, enc = cv2.imencode(".jpg", img)
        if not ok:
            enc = np.zeros((1,), dtype=np.uint8)
        b = enc.tobytes()
        encoded.append(b)
        max_len = max(max_len, len(b))
    return encoded, max_len


def _to_dict(cfg_obj: Any) -> dict[str, Any]:
    if cfg_obj is None:
        return {}
    if isinstance(cfg_obj, dict):
        d = dict(cfg_obj)
    else:
        d = dict(getattr(cfg_obj, "__dict__", {}))
    d.pop("type", None)
    return d


def build_fixed_test_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        env=SimpleNamespace(
            name="real_robot",
            fps=50,
            robot=SimpleNamespace(
                can_name_left="can_left",
                can_name_right="can_right",
                baud_rate=1000000,
                cameras={
                    "front": {
                        "serial_number_or_name": "CC1B25100EM",
                        "width": 640,
                        "height": 480,
                        "fps": 30,
                    },
                    "left": {
                        "serial_number_or_name": "242522071187",
                        "width": 640,
                        "height": 480,
                        "fps": 30,
                    },
                    "right": {
                        "serial_number_or_name": "327122075682",
                        "width": 640,
                        "height": 480,
                        "fps": 30,
                    },
                },
            ),
            teleop=SimpleNamespace(
                zmq_host="127.0.0.1",
                zmq_port=5555,
                left_pose_source="left_controller",
                right_pose_source="right_controller",
            ),
        ),
        device="cpu",
        num_envs=1,
        task_description="real_robot_hardware_smoke_test",
        camera_mapping=DEFAULT_CAMERA_MAPPING.copy(),
    )


class PiperJointRobot:
    def __init__(self, robot_cfg: Any, teleop_cfg: Any):
        self.bus_left = PiperMotorsBus(robot_cfg.can_name_left, robot_cfg.baud_rate, motor_prefix="left")
        self.bus_right = PiperMotorsBus(robot_cfg.can_name_right, robot_cfg.baud_rate, motor_prefix="right")

        cams = getattr(robot_cfg, "cameras", {}) or {}
        self.cam_front = self._build_dabai_camera(cams.get("front"))
        self.cam_left = self._build_realsense_camera(cams.get("left"))
        self.cam_right = self._build_realsense_camera(cams.get("right"))
        self.teleop = self._build_teleop(teleop_cfg)

        self.is_connected = False

    def _build_dabai_camera(self, cfg: Any):
        if OrbbecDabaiCamera is None or OrbbecDabaiCameraConfig is None:
            return None
        d = _to_dict(cfg)
        if not d:
            return None
        return OrbbecDabaiCamera(OrbbecDabaiCameraConfig(**d))

    def _build_realsense_camera(self, cfg: Any):
        if RealSenseCamera is None or RealSenseCameraConfig is None:
            return None
        d = _to_dict(cfg)
        if not d:
            return None
        return RealSenseCamera(RealSenseCameraConfig(**d))

    def _build_teleop(self, cfg: Any):
        if PicoVrTeleop is None or PicoVrTeleopConfig is None:
            return None
        d = _to_dict(cfg)
        if not d:
            return None
        return PicoVrTeleop(PicoVrTeleopConfig(**d))

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

    def reset_to_home(self, wait_s: float = 2.0):
        if not self.is_connected:
            return
        self.bus_left.reset_pos(wait_s=wait_s)
        self.bus_right.reset_pos(wait_s=wait_s)

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
        self.bus_left.disconnect(disable_torque=False)
        self.bus_right.disconnect(disable_torque=False)
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

        return {
            "front": self._safe_read_image(self.cam_front),
            "left": self._safe_read_image(self.cam_left),
            "right": self._safe_read_image(self.cam_right),
            "state": np.asarray(left_state + right_state, dtype=np.float32),
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
            return False, None
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
        return True, None


class PiperJointEnv(gym.Env):
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
        self.robot.reset_to_home(wait_s=2.0)
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
    def __init__(self, cfg=None, rank: int = 0, world_size: int = 1):
        self.rank = rank
        self.world_size = world_size
        if cfg is None:
            cfg = build_fixed_test_cfg()

        self.device = getattr(cfg, "device", "cpu")
        self.num_envs = getattr(cfg, "num_envs", 1)
        self.task_description = getattr(cfg, "task_description", "catch_bowl")
        self.camera_mapping = getattr(cfg, "camera_mapping", DEFAULT_CAMERA_MAPPING.copy())

        robot_cfg = cfg.env.robot
        teleop_cfg = cfg.env.teleop
        self.robot = PiperJointRobot(robot_cfg=robot_cfg, teleop_cfg=teleop_cfg)
        self.env = PiperJointEnv(self.robot)

    def _convert_obs_to_test_env_format(self, obs: dict[str, Any]) -> dict[str, Any]:
        images_list, camera_names = [], []
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
            images_dict[cam_name] = torch.from_numpy(np.array(np.frombuffer(padded_bytes, dtype=np.uint8))).unsqueeze(0)

        return {
            "images": images_dict,
            "state": obs["state"].astype(np.float32),
            "task_description": self.task_description,
        }

    @staticmethod
    def _normalize_action(action: Any) -> np.ndarray:
        action_np = action.detach().cpu().numpy().astype(np.float32) if isinstance(action, torch.Tensor) else np.asarray(action, dtype=np.float32)
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
        return self._convert_obs_to_test_env_format(obs), float(reward), bool(terminated), bool(truncated), {
            "intervene_action": info.get("intervene_action"),
            "intervene_flag": info.get("is_intervention", False),
        }

    def close(self):
        self.env.close()


# def make_robot_env(cfg=None) -> tuple[gym.Env, Any]:
#     wrapper = RealRobotEnvWrapper(cfg=cfg)
#     return wrapper.env, wrapper.robot.teleop


# def make_processors(env, teleop_device, cfg, device: str = "cpu"):
#     del env, teleop_device, cfg, device
#     return None, None


# def step_env_and_process_transition(env, transition, action, env_processor=None, action_processor=None, save_images=False):
#     del transition, env_processor, action_processor, save_images
#     return env.step(action)


if __name__ == "__main__":
    cfg = build_fixed_test_cfg()

    RUN_RESET_AND_MOTION = True
    MOTION_STEPS = 40
    MOTION_HZ = 10.0
    MOTION_DELTA_RAD = 0.03

    force_print("=" * 80)
    force_print("[PiperEnv Test] Start loading environment")
    force_print(
        f"[PiperEnv Test] cfg={{left:{cfg.env.robot.can_name_left}, right:{cfg.env.robot.can_name_right}, baud:{cfg.env.robot.baud_rate}}}"
    )

    wrapper = None
    try:
        wrapper = RealRobotEnvWrapper(cfg=cfg, rank=0, world_size=1)
        force_print("[PiperEnv Test] ✅ RealRobotEnvWrapper initialized successfully")

        if RUN_RESET_AND_MOTION:
            force_print("[PiperEnv Test] Running REAL hardware reset() + motion test ...")
            obs = wrapper.reset()
            state = obs["state"]
            base = np.asarray(state, dtype=np.float32).copy()
            dt = 1.0 / max(MOTION_HZ, 1e-3)

            for i in range(MOTION_STEPS):
                t = i * dt
                action = base.copy()
                d = float(MOTION_DELTA_RAD * np.sin(2.0 * np.pi * 0.2 * t))
                action[0] = base[0] + d
                action[3] = base[3] - d
                action[7] = base[7] + d
                action[10] = base[10] - d
                _, reward, terminated, truncated, info = wrapper.step(action)
                if i % max(1, MOTION_STEPS // 10) == 0:
                    force_print(
                        f"[PiperEnv Test] step={i:03d}/{MOTION_STEPS}, d={d:.4f}, "
                        f"intervene={info.get('intervene_flag')}, reward={reward}, term={terminated}, trunc={truncated}"
                    )
                time.sleep(dt)
            wrapper.step(base)
            force_print("[PiperEnv Test] Motion test finished and returned to base pose")

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