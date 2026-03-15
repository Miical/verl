# !/usr/bin/env python

import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
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
    camera_mapping: dict[str, str] = field(
        default_factory=lambda: {
            "front": "head_image",
            "left": "left_wrist_image",
            "right": "right_wrist_image",
        }
    )


def _default_camera_mapping() -> dict[str, str]:
    return {
        "front": "head_image",
        "left": "left_wrist_image",
        "right": "right_wrist_image",
    }


def _get_attr(obj: Any, key: str, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _sanitize_component_cfg(cfg_obj: Any) -> dict[str, Any]:
    """把 dict/namespace 配置转为可用于 dataclass(**kwargs) 的参数字典。

    会移除上层路由字段（例如 `type`），避免传入具体 config dataclass 时触发
    `unexpected keyword argument 'type'`。
    """
    cfg_dict = cfg_obj if isinstance(cfg_obj, dict) else getattr(cfg_obj, "__dict__", {})
    if cfg_dict is None:
        return {}
    cfg_dict = dict(cfg_dict)
    cfg_dict.pop("type", None)
    return cfg_dict


def _normalize_runtime_cfg(cfg: Any) -> Any:
    """兼容旧配置入口：
    1) cfg.robot_config_path -> json(包含 env)
    2) cfg.env.robot/teleop (Hydra/namespace)
    3) 扁平字段 can_name_left/can_name_right
    """
    if cfg is None:
        cfg = _DefaultCfg()

    # 若传入 config 文件路径，优先读取
    robot_config_path = _get_attr(cfg, "robot_config_path", None)
    if robot_config_path:
        with open(robot_config_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        cfg_env = loaded.get("env", loaded)
    else:
        cfg_env = _get_attr(cfg, "env", None)

    robot_cfg = _get_attr(cfg_env, "robot", None)
    teleop_cfg = _get_attr(cfg_env, "teleop", None)
    cameras_cfg = _get_attr(robot_cfg, "cameras", {}) or {}

    can_name_left = _get_attr(cfg, "can_name_left", None) or _get_attr(robot_cfg, "can_name_left", "can_left")
    can_name_right = _get_attr(cfg, "can_name_right", None) or _get_attr(robot_cfg, "can_name_right", "can_right")
    baud_rate = _get_attr(cfg, "baud_rate", None) or _get_attr(robot_cfg, "baud_rate", 1_000_000)

    task_description = _get_attr(cfg, "task_description", None) or _get_attr(cfg_env, "task", "catch_bowl")
    device = _get_attr(cfg, "device", "cpu")
    num_envs = _get_attr(cfg, "num_envs", 1)

    camera_mapping = _get_attr(cfg, "camera_mapping", None)
    if camera_mapping is None:
        camera_mapping = _default_camera_mapping()

    # 转换 camera 配置到当前实现字段
    dabai_camera = _get_attr(cameras_cfg, "front", None)
    realsense_left_camera = _get_attr(cameras_cfg, "left", None)
    realsense_right_camera = _get_attr(cameras_cfg, "right", None)

    return SimpleNamespace(
        can_name_left=can_name_left,
        can_name_right=can_name_right,
        baud_rate=baud_rate,
        task_description=task_description,
        device=device,
        num_envs=num_envs,
        camera_mapping=camera_mapping,
        dabai_camera=dabai_camera,
        realsense_left_camera=realsense_left_camera,
        realsense_right_camera=realsense_right_camera,
        pico_vr=teleop_cfg,
    )


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
        cam_cfg_dict = _sanitize_component_cfg(cam_cfg)
        return OrbbecDabaiCamera(OrbbecDabaiCameraConfig(**cam_cfg_dict))

    def _build_realsense_camera(self, cfg: Any, side: str):
        if RealSenseCamera is None or RealSenseCameraConfig is None:
            return None
        key = f"realsense_{side}_camera"
        cam_cfg = getattr(cfg, key, None)
        if cam_cfg is None:
            return None
        cam_cfg_dict = _sanitize_component_cfg(cam_cfg)
        return RealSenseCamera(RealSenseCameraConfig(**cam_cfg_dict))

    def _build_teleop(self, cfg: Any):
        if PicoVrTeleop is None or PicoVrTeleopConfig is None:
            return None
        teleop_cfg = getattr(cfg, "pico_vr", None)
        if teleop_cfg is None:
            return None
        teleop_cfg_dict = _sanitize_component_cfg(teleop_cfg)
        return PicoVrTeleop(PicoVrTeleopConfig(**teleop_cfg_dict))

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
        """复位到电机层定义的 home 位姿。"""
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
        # ⚠️ 安全策略：默认不下发 DisableArm，避免停进程时机械臂掉电下坠
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
        # 每次 reset 时先回到安全初始位
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
    """对外兼容封装：reset/step 输出 test_env 侧预期字段。"""

    def __init__(self, cfg, rank: int = 0, world_size: int = 1):
        force_print(f"[RealRobotEnvWrapper] init rank={rank}, world_size={world_size}")
        self.rank = rank
        self.world_size = world_size

        runtime_cfg = _normalize_runtime_cfg(cfg)

        self.device = getattr(runtime_cfg, "device", "cpu")
        self.num_envs = getattr(runtime_cfg, "num_envs", 1)
        self.task_description = getattr(runtime_cfg, "task_description", "catch_bowl")
        camera_mapping = getattr(runtime_cfg, "camera_mapping", None)
        if camera_mapping is None:
            camera_mapping = _default_camera_mapping()
        self.camera_mapping = camera_mapping

        self.robot = PiperJointRobot(runtime_cfg)
        self.env = PiperJointEnv(self.robot)

    def _convert_obs_to_test_env_format(self, obs: dict[str, Any]) -> dict[str, Any]:
        images_list = []
        camera_names = []
        if not isinstance(self.camera_mapping, dict):
            self.camera_mapping = _default_camera_mapping()
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
    runtime_cfg = _normalize_runtime_cfg(cfg)
    robot = PiperJointRobot(runtime_cfg)
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
    # 直接在代码中写死一份可运行的配置（不走外部命令行传参）
    # 如需调整，请直接修改下面这份配置。
    cfg = SimpleNamespace(
        env=SimpleNamespace(
            name="real_robot",
            fps=50,
            robot=SimpleNamespace(
                type="piper_follower_end_effector",
                can_name_left="can_left",
                can_name_right="can_right",
                baud_rate=1000000,
                cameras={
                    "front": {
                        "type": "orbbec_dabai",
                        "serial_number_or_name": "CC1B25100EM",
                        "width": 640,
                        "height": 480,
                        "fps": 30,
                    },
                    "left": {
                        "type": "intelrealsense",
                        "serial_number_or_name": "242522071187",
                        "width": 640,
                        "height": 480,
                        "fps": 30,
                    },
                    "right": {
                        "type": "intelrealsense",
                        "serial_number_or_name": "327122075682",
                        "width": 640,
                        "height": 480,
                        "fps": 30,
                    },
                },
            ),
            teleop=SimpleNamespace(
                type="pico_vr",
                left_pose_source="left_controller",
                right_pose_source="right_controller",
                translation_scale=1,
                rotation_scale=1,
                rotation_yaw_gain=0.1,
                translation_clip=1.0,
                rotation_clip=2.0,
                translation_deadband=1e-4,
                rotation_deadband=5e-3,
                gripper_open=0.07,
                gripper_close=0.0,
                gripper_delta_gain=1.0,
                gripper_deadband=1e-3,
            ),
        ),
        device="cpu",
        num_envs=1,
        task_description="real_robot_hardware_smoke_test",
        camera_mapping={
            "front": "head_image",
            "left": "left_wrist_image",
            "right": "right_wrist_image",
        },
    )

    # 测试开关（按需在代码里改）
    RUN_RESET_AND_MOTION = True
    MOTION_STEPS = 40
    MOTION_HZ = 10.0
    MOTION_DELTA_RAD = 0.03

    runtime_cfg_preview = _normalize_runtime_cfg(cfg)
    force_print("=" * 80)
    force_print("[PiperEnv Test] Start loading environment")
    force_print(
        f"[PiperEnv Test] cfg={{left:{runtime_cfg_preview.can_name_left}, right:{runtime_cfg_preview.can_name_right}, baud:{runtime_cfg_preview.baud_rate}}}"
    )

    wrapper = None
    try:
        wrapper = RealRobotEnvWrapper(cfg=cfg, rank=0, world_size=1)
        force_print("[PiperEnv Test] ✅ RealRobotEnvWrapper initialized successfully")

        if not RUN_RESET_AND_MOTION:
            force_print("[PiperEnv Test] RUN_RESET_AND_MOTION=False, skip hardware reset/step")
        else:
            force_print("[PiperEnv Test] Running REAL hardware reset() + motion test ...")
            obs = wrapper.reset()
            state = obs["state"]
            force_print(
                f"[PiperEnv Test] reset ok: keys={list(obs.keys())}, image_keys={list(obs['images'].keys())}, state_shape={state.shape}"
            )

            base = np.asarray(state, dtype=np.float32).copy()
            if base.shape[0] != 14:
                raise RuntimeError(f"Unexpected state dim {base.shape[0]}, expected 14")

            dt = 1.0 / max(MOTION_HZ, 1e-3)
            for i in range(MOTION_STEPS):
                t = i * dt
                action = base.copy()
                s = float(np.sin(2.0 * np.pi * 0.2 * t))
                d = float(MOTION_DELTA_RAD * s)

                # 双臂 joint_1 / joint_4 对称小幅运动（更容易观察，且不易触发 joint_2/joint_3 边界裁剪）
                action[0] = base[0] + d
                action[3] = base[3] - d
                action[7] = base[7] + d
                action[10] = base[10] - d

                obs, reward, terminated, truncated, info = wrapper.step(action)
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
