# !/usr/bin/env python

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

import logging
logging.getLogger("can.interfaces.socketcan").setLevel(logging.ERROR)
import time
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from lerobot.cameras import opencv  # noqa: F401
from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.envs.configs import HILSerlRobotEnvConfig
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    AddTeleopActionAsComplimentaryDataStep,
    AddTeleopEventsAsInfoStep,
    DataProcessorPipeline,
    DeviceProcessorStep,
    EnvTransition,
    GripperPenaltyProcessorStep,
    ImageCropResizeProcessorStep,
    InterventionActionProcessorStep,
    RenameObservationsProcessorStep,
    JointVelocityProcessorStep,
    MapDeltaActionToRobotActionStep,
    MapTensorToDeltaActionDictStep,
    MotorCurrentProcessorStep,
    Numpy2TorchActionProcessorStep,
    RewardClassifierProcessorStep,
    RobotActionToPolicyActionProcessorStep,
    TimeLimitProcessorStep,
    Torch2NumpyActionProcessorStep,
    TransitionKey,
    VanillaObservationProcessorStep,
    create_transition,
)
from lerobot.processor.converters import identity_transition
from lerobot.robots import (  # noqa: F401
    RobotConfig,
    make_robot_from_config,
    so100_follower,
    piper_follower,
)
from lerobot.robots.robot import Robot
from lerobot.robots.so100_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    EEReferenceAndDelta,
    ForwardKinematicsJointsToEEObservation,
    GripperVelocityToJoint,
    InverseKinematicsRLStep,
)
from lerobot.teleoperators import (
    gamepad,  # noqa: F401
    keyboard,  # noqa: F401
    make_teleoperator_from_config,
    so101_leader,  # noqa: F401
)

from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.utils.constants import ACTION, DONE, OBS_IMAGES, OBS_STATE, REWARD
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import log_say
from lerobot.cameras import (  # noqa: F401 - 导入相机以触发注册
    opencv,
    realsense,
    dabai,
)
import pdb
logging.basicConfig(level=logging.INFO)

def save_cropped_images_from_processed_obs(processed_obs: dict, base_dir: str = "img_output/img_crop") -> None:
    """
    从经过 VanillaObservationProcessorStep (+ ImageCropResizeProcessorStep) 处理后的观测中，
    提取裁剪/缩放后的图像 (B,C,H,W, float32, [0,1]) 并保存到 img_output/img_crop/<camera_name>/，
    每个相机只保留最新5张。

    参数:
        processed_obs: env_processor 输出的观测字典（transition[TransitionKey.OBSERVATION]）
        base_dir: 保存根目录
    """
    import os, glob, time
    import torch
    import cv2

    os.makedirs(base_dir, exist_ok=True)

    def _key_to_camname(k: str) -> str:
        # 兼容两类键：
        # 1) 多相机: "observation.images.<camera>"
        # 2) 单图像: "observation.image"
        if ".images." in k:
            return k.split(".images.", 1)[1]
        if k.endswith(".image"):
            return "image"
        return k.replace(".", "_")  # 兜底

    for key, tensor_img in list(processed_obs.items()):
        if "image" not in key:
            continue
        if not isinstance(tensor_img, torch.Tensor):
            continue

        # 移到 CPU、去梯度
        img = tensor_img.detach()
        if img.device.type != "cpu":
            img = img.cpu()
        # pdb.set_trace()
        # 允许 (C,H,W) 或 (B,C,H,W)
        if img.ndim == 3:
            img = img.unsqueeze(0)  # -> (1,C,H,W)
        assert img.ndim == 4, f"Expect (B,C,H,W), got {img.shape}"

        # 规范到 [0,1] 再转 [0,255] uint8
        img = img.clamp(0.0, 1.0)
        img_u8 = torch.round(img * 255.0).to(torch.uint8)  # (B,C,H,W)

        cam_name = _key_to_camname(key)
        save_dir = os.path.join(base_dir, cam_name)
        os.makedirs(save_dir, exist_ok=True)

        # 逐张保存（通常 B=1）
        b, c, h, w = img_u8.shape
        for i in range(b):
            # (C,H,W) -> (H,W,C)
            hwc = img_u8[i].permute(1, 2, 0).numpy()
            # RGB -> BGR (cv2.imwrite 期望BGR)
            bgr = hwc[..., ::-1]

            filename = os.path.join(save_dir, f"{int(time.time() * 1000)}.jpg")
            cv2.imwrite(filename, bgr)

            # 只保留最近 5 张
            files = sorted(glob.glob(os.path.join(save_dir, "*.jpg")), key=os.path.getmtime)
            if len(files) > 5:
                for old_file in files[:-5]:
                    try:
                        os.remove(old_file)
                    except Exception:
                        pass

@dataclass
class DatasetConfig:
    """Configuration for dataset creation and management."""

    repo_id: str
    task: str
    root: str | None = None
    num_episodes_to_record: int = 5
    replay_episode: int | None = None
    push_to_hub: bool = False


@dataclass
class GymManipulatorConfig:
    """Main configuration for gym manipulator environment."""

    env: HILSerlRobotEnvConfig
    dataset: DatasetConfig
    mode: str | None = None  # Either "record", "replay", None
    device: str = "cpu"


def reset_follower_position(robot_arm: Robot, target_position: np.ndarray,end_pose_name) -> None:
    """Reset robot arm to target position using smooth trajectory."""
    robot_arm.bus_left.interface.ModeCtrl(ctrl_mode=0x01, move_mode=0x01,move_spd_rate_ctrl=50, is_mit_mode=0x00)
    robot_arm.bus_right.interface.ModeCtrl(ctrl_mode=0x01, move_mode=0x01,move_spd_rate_ctrl=50, is_mit_mode=0x00)
    current_position_dict_left = robot_arm.bus_left.sync_read("Present_Position")
    current_position_dict_right = robot_arm.bus_right.sync_read("Present_Position")
    current_position_left = np.array(
        [current_position_dict_left[name] for name in current_position_dict_left], dtype=np.float32
    )
    current_position_right = np.array(
        [current_position_dict_right[name] for name in current_position_dict_right], dtype=np.float32
    )
    # pdb.set_trace()
    target_position_left=target_position[:7]
    target_position_right=target_position[7:]
    trajectory_left = torch.from_numpy(
        np.linspace(current_position_left, target_position_left, 50)
    )  # NOTE: 30 is just an arbitrary number
    trajectory_right = torch.from_numpy(
        np.linspace(current_position_right, target_position_right, 50)
    )  # NOTE: 30 is just an arbitrary number
    # pdb.set_trace()
    # busy_wait(0.1)
    for pose_left,pose_right in zip(trajectory_left,trajectory_right):
        action_dict_left = dict(zip(current_position_dict_left, pose_left, strict=False))
        action_dict_right = dict(zip(current_position_dict_right, pose_right, strict=False))
        # pdb.set_trace()
        robot_arm.bus_left.sync_write("Goal_Position", action_dict_left)
        robot_arm.bus_right.sync_write("Goal_Position", action_dict_right)
        busy_wait(0.015)
    

    robot_arm.bus_left.interface.ModeCtrl(ctrl_mode=0x01, move_mode=0x00,move_spd_rate_ctrl=50, is_mit_mode=0x00)
    robot_arm.bus_right.interface.ModeCtrl(ctrl_mode=0x01, move_mode=0x00,move_spd_rate_ctrl=50, is_mit_mode=0x00)
    action_zero=[0.056127, 0.0, 0.213266,0.0, 1.48351241090266,0.0,0.05957,0.056127, 0.0, 0.213266,0.0, 1.48351241090266,0.0,0.05957]
    action_target=[0.056127, 0.0, 0.213266,0.0, 1.48351241090266,0.0,0.05957,0.056127, 0.0, 0.213266,0.0, 1.48351241090266,0.0,0.05957]
    
    #action_target=[0.252452, -0.034618, 0.272318, 3.07350736, 0.4325624, 2.90283161, 0.06916,0.252452, -0.034618, 0.272318, 3.07350736, 0.4325624, 2.90283161, 0.06916]
    trajectory_endpose=np.linspace(action_zero, action_target, 12)
    for action in trajectory_endpose:
        end_pose_targets_dict = {f"{key}.value": action[i] for i, key in enumerate(end_pose_name)}
        robot_arm.send_action(end_pose_targets_dict)
        busy_wait(0.12)
    busy_wait(0.1)
    
    # pdb.set_trace()    


class RobotEnv(gym.Env):
    """Gym environment for robotic control with human intervention support."""

    def __init__(
        self,
        robot,
        use_gripper: bool = False,
        display_cameras: bool = False,
        reset_pose: list[float] | None = None,
        reset_time_s: float = 5.0,
    ) -> None:
        """Initialize robot environment with configuration options.

        Args:
            robot: Robot interface for hardware communication.
            use_gripper: Whether to include gripper in action space.
            display_cameras: Whether to show camera feeds during execution.
            reset_pose: Joint positions for environment reset.
            reset_time_s: Time to wait during reset.
        """
        super().__init__()
        
        self.robot = robot
        self.display_cameras = display_cameras

        # Connect to the robot if not already connected.
        if not self.robot.is_connected:
            self.robot.connect()

        # Episode tracking.
        self.current_step = 0
        self.episode_data = None

        # self._joint_names = [f"{key}.pos" for key in self.robot.bus.motors]
        self._image_keys = self.robot.cameras.keys()

        self.reset_pose = reset_pose
        self.reset_time_s = reset_time_s

        self.use_gripper = use_gripper

        # self._joint_names = list(self.robot.bus.motors.keys())
        self._left_joint_names = self.robot.bus_left.motors
        self._right_joint_names = self.robot.bus_right.motors
        self._joint_names= self._left_joint_names + self._right_joint_names
        # self._raw_joint_positions = None
        self._raw_end_pose_value = None
        self._left_end_pose_name = self.robot.bus_left.end_pose_keys
        self._right_end_pose_name = self.robot.bus_right.end_pose_keys
        self._end_pose_name= self._left_end_pose_name + self._right_end_pose_name
        
        self._setup_spaces()

    def _get_observation(self) -> dict[str, Any]:
        """Get current robot observation including joint positions and camera images."""
        obs_dict = self.robot.get_observation()
        for k in ("front", "left", "right"):
            img = obs_dict[k]
            obs_dict[k] = img[:, :, ::-1].copy()   
        # pdb.set_trace()
        # raw_joint_joint_position = {f"{name}.pos": obs_dict[f"{name}.pos"] for name in self._joint_names}
        # joint_positions = np.array([raw_joint_joint_position[f"{name}.pos"] for name in self._joint_names])
        raw_left_end_pose_values = {f"{name}.value": obs_dict[f"{name}.value"] for name in self._left_end_pose_name}
        raw_right_end_pose_values = {f"{name}.value": obs_dict[f"{name}.value"] for name in self._right_end_pose_name}
        left_end_pose_value=np.array([raw_left_end_pose_values[f"{name}.value"] for name in self._left_end_pose_name])
        right_end_pose_value=np.array([raw_right_end_pose_values[f"{name}.value"] for name in self._right_end_pose_name])

        agent_end_pose_value = np.concatenate([left_end_pose_value, right_end_pose_value], axis=0)
        images = {key: obs_dict[key] for key in self._image_keys}
        # print("agent_end_pose_value:",agent_end_pose_value)
        # pdb.set_trace()
        return {"agent_end_pose_value":agent_end_pose_value , "pixels": images, **raw_left_end_pose_values,**raw_right_end_pose_values}

    def _setup_spaces(self) -> None:
        """Configure observation and action spaces based on robot capabilities."""
        current_observation = self._get_observation()

        observation_spaces = {}

        # Define observation spaces for images and other states.
        if current_observation is not None and "pixels" in current_observation:
            prefix = OBS_IMAGES
            observation_spaces = {
                f"{prefix}.{key}": gym.spaces.Box(
                    low=0, high=255, shape=current_observation["pixels"][key].shape, dtype=np.uint8
                )
                for key in current_observation["pixels"]
            }
        # pdb.set_trace()
        if current_observation is not None:
            agent_end_pose_value = current_observation["agent_end_pose_value"]
            observation_spaces[OBS_STATE] = gym.spaces.Box(
                low=-10,
                high=10,
                shape=agent_end_pose_value.shape,
                dtype=np.float32,
            )
        
        self.observation_space = gym.spaces.Dict(observation_spaces)

        # Define the action space for joint positions along with setting an intervention flag.
        action_dim = 14
        bounds = {}
        bounds["min"] = -np.ones(action_dim)
        bounds["max"] = np.ones(action_dim)

        # if self.use_gripper:
        #     action_dim += 1
        #     bounds["min"] = np.concatenate([bounds["min"], [0]])
        #     bounds["max"] = np.concatenate([bounds["max"], [2]])

        self.action_space = gym.spaces.Box(
            low=bounds["min"],
            high=bounds["max"],
            shape=(action_dim,),
            dtype=np.float32,
        )
        # pdb.set_trace()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset environment to initial state.

        Args:
            seed: Random seed for reproducibility.
            options: Additional reset options.

        Returns:
            Tuple of (observation, info) dictionaries.
        """
        # Reset the robot
        # self.robot.reset()

        start_time = time.perf_counter()
        if self.reset_pose is not None:
            # print("here")
            print('reset the environment')
            #log_say("Reset the environment.", play_sounds=True)
            # pdb.set_trace()
            reset_follower_position(self.robot, np.array(self.reset_pose),self._end_pose_name)
            # action=[0.056127, 0.0, 0.213266,0.0, 1.48351241090266,0.0,0.05957,0.056127, 0.0, 0.213266,0.0, 1.48351241090266,0.0,0.05957]

            #log_say("Reset the environment done.", play_sounds=True)
        busy_wait(self.reset_time_s - (time.perf_counter() - start_time))

        # super().reset(seed=seed, options=options)

        # Reset episode tracking variables.
        self.current_step = 0
        self.episode_data = None
        obs = self._get_observation()
        # pdb.set_trace() 
        # self._raw_joint_positions = {f"{key}.pos": obs[f"{key}.pos"] for key in self._joint_names}
        self._raw_end_pose_value = {f"{key}.value": obs[f"{key}.value"] for key in self._end_pose_name}

        return obs, {TeleopEvents.IS_INTERVENTION: False}

    def step(self, action, transition_info) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Execute one environment step with given action."""
        # joint_targets_dict = {f"{key}.pos": action[i] for i, key in enumerate(self.robot.bus.motors)}
        #盲改
        end_pose_targets_dict = {f"{key}.value": action[i] for i, key in enumerate(self._end_pose_name)}
        is_intervention = False

        if TeleopEvents.IS_INTERVENTION in transition_info:
            is_intervention = transition_info[TeleopEvents.IS_INTERVENTION]
        #print(f"is_intervention: {is_intervention}")
        #当不进行干预时，发送动作
        if not is_intervention:
            self.robot.bus_left.interface.ModeCtrl(ctrl_mode=0x01, move_mode=0x00,move_spd_rate_ctrl=50, is_mit_mode=0x00)
            self.robot.bus_right.interface.ModeCtrl(ctrl_mode=0x01, move_mode=0x00,move_spd_rate_ctrl=50, is_mit_mode=0x00)
            print("不在干预")
            self.robot.send_action(end_pose_targets_dict)

        # print("在干预")
        obs = self._get_observation()
        self._raw_end_pose_value = {f"{key}.value": obs[f"{key}.value"] for key in self._end_pose_name}
        # print("self._raw_end_pose_value:",self._raw_end_pose_value)
        # self._raw_joint_positions = {f"{key}.pos": obs[f"{key}.pos"] for key in self._joint_names}

        if self.display_cameras:
            self.render()

        self.current_step += 1

        reward = 0.0
        terminated = False
        truncated = False

        return (
            obs,
            reward,
            terminated,
            truncated,
            {TeleopEvents.IS_INTERVENTION: False},
        )

    def render(self) -> None:
        """Save latest 5 frames from each camera to img_output/gym/<camera_name>/."""
        import cv2, os, glob

        current_observation = self._get_observation()
        if current_observation is None:
            return

        base_dir = "img_output/robotenv"
        os.makedirs(base_dir, exist_ok=True)

        image_keys = list(current_observation["pixels"].keys())
        for key in image_keys:
            img = current_observation["pixels"][key]
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

    # def render(self) -> None:
    #     """Display robot camera feeds."""
    #     import cv2
    #     # pdb.set_trace()
    #     current_observation = self._get_observation()
    #     if current_observation is not None:
    #         image_keys = [key for key in current_observation["pixels"]]
    #         # pdb.set_trace()
    #         for key in image_keys:
    #             cv2.imshow(key, cv2.cvtColor(current_observation["pixels"][key], cv2.COLOR_RGB2BGR))
    #             cv2.waitKey(1)

    def close(self) -> None:
        """Close environment and disconnect robot."""
        if self.robot.is_connected:
            self.robot.disconnect()

    # def get_raw_joint_positions(self) -> dict[str, float]:
    #     """Get raw joint positions."""
    #     return self._raw_joint_positions
    def get_raw_end_pose_value(self) -> dict[str, float]:
        return self._raw_end_pose_value
    def get_end_pose_name(self) -> list[str]:
        return self._end_pose_name

def make_robot_env(cfg: HILSerlRobotEnvConfig) -> tuple[gym.Env, Any]:
    """Create robot environment from configuration.

    Args:
        cfg: Environment configuration.

    Returns:
        Tuple of (gym environment, teleoperator device).
    """
    # Check if this is a GymHIL simulation environment
    if cfg.name == "gym_hil":
        assert cfg.robot is None and cfg.teleop is None, "GymHIL environment does not support robot or teleop"
        import gym_hil  # noqa: F401

        # Extract gripper settings with defaults
        use_gripper = cfg.processor.gripper.use_gripper if cfg.processor.gripper is not None else True
        gripper_penalty = cfg.processor.gripper.gripper_penalty if cfg.processor.gripper is not None else 0.0

        env = gym.make(
            f"gym_hil/{cfg.task}",
            image_obs=True,
            render_mode="human",
            use_gripper=use_gripper,
            gripper_penalty=gripper_penalty,
        )

        return env, None

    # Real robot environment
    assert cfg.robot is not None, "Robot config must be provided for real robot environment"
    assert cfg.teleop is not None, "Teleop config must be provided for real robot environment"

    robot = make_robot_from_config(cfg.robot)
    teleop_device = make_teleoperator_from_config(cfg.teleop)
    teleop_device.connect()

    # Create base environment with safe defaults
    use_gripper = cfg.processor.gripper.use_gripper if cfg.processor.gripper is not None else True
    display_cameras = (
        cfg.processor.observation.display_cameras if cfg.processor.observation is not None else False
    )
    reset_pose = cfg.processor.reset.fixed_reset_joint_positions if cfg.processor.reset is not None else None

    env = RobotEnv(
        robot=robot,
        use_gripper=use_gripper,
        display_cameras=display_cameras,
        reset_pose=reset_pose,
    )

    return env, teleop_device


def make_processors(
    env: gym.Env, teleop_device: Teleoperator | None, cfg: HILSerlRobotEnvConfig, device: str = "cpu"
) -> tuple[
    DataProcessorPipeline[EnvTransition, EnvTransition], DataProcessorPipeline[EnvTransition, EnvTransition]
]:
    """Create environment and action processors.

    Args:
        env: Robot environment instance.
        teleop_device: Teleoperator device for intervention.
        cfg: Processor configuration.
        device: Target device for computations.

    Returns:
        Tuple of (environment processor, action processor).
    """
    terminate_on_success = (
        cfg.processor.reset.terminate_on_success if cfg.processor.reset is not None else True
    )

    if cfg.name == "gym_hil":
        action_pipeline_steps = [
            InterventionActionProcessorStep(terminate_on_success=terminate_on_success),
            Torch2NumpyActionProcessorStep(),
        ]

        env_pipeline_steps = [
            Numpy2TorchActionProcessorStep(),
            VanillaObservationProcessorStep(),
            AddBatchDimensionProcessorStep(),
            DeviceProcessorStep(device=device),
        ]

        return DataProcessorPipeline(
            steps=env_pipeline_steps, to_transition=identity_transition, to_output=identity_transition
        ), DataProcessorPipeline(
            steps=action_pipeline_steps, to_transition=identity_transition, to_output=identity_transition
        )

    # Full processor pipeline for real robot environment
    # Get robot and motor information for kinematics
    

    # 映射：bus 电机名 -> URDF 关节名（顺序必须与 motor_names 一致）
    # 1) bus 层电机名
    end_pose_name= env.get_end_pose_name()

    # 2) 拆分：臂的6关节 vs 夹爪
    # arm_motor_names = [m for m in motor_names if "gripper" not in m]  # 或者直接 motor_names[:6]
    # gripper_motor_names = [m for m in motor_names if "gripper" in m]  # 可为空或1个

    # 3) 只为“臂的6关节”做 URDF 名映射（按你的 URDF 实际名字改）
    # name_map = {
    #     "left_joint_1": "joint1",
    #     "left_joint_2": "joint2",
    #     "left_joint_3": "joint3",
    #     "left_joint_4": "joint4",
    #     "left_joint_5": "joint5",
    #     "left_joint_6": "joint6",
    # }
    # ik_joint_names = [name_map.get(n, n) for n in arm_motor_names]
    # pdb.set_trace()
    # Set up kinematics solver if inverse kinematics is configured
    kinematics_solver = None
    if cfg.processor.inverse_kinematics is not None:
        kinematics_solver = RobotKinematics(
            urdf_path=cfg.processor.inverse_kinematics.urdf_path,
            target_frame_name=cfg.processor.inverse_kinematics.target_frame_name,
            joint_names=motor_names[:-1],
            # joint_names=ik_joint_names,
        )

    env_pipeline_steps = [VanillaObservationProcessorStep()]
    
    # If environment feature keys differ from policy keys, insert a rename step
    if cfg.features_map:
        rename_map = {src: dst for src, dst in cfg.features_map.items() if src != dst}
        if rename_map:
            env_pipeline_steps.append(RenameObservationsProcessorStep(rename_map=rename_map))

    if cfg.processor.observation is not None:
        if cfg.processor.observation.add_joint_velocity_to_observation:#false
            env_pipeline_steps.append(JointVelocityProcessorStep(dt=1.0 / cfg.fps))
        if cfg.processor.observation.add_current_to_observation:#false
            env_pipeline_steps.append(MotorCurrentProcessorStep(robot=env.robot))

    if kinematics_solver is not None: #none
        env_pipeline_steps.append(
            ForwardKinematicsJointsToEEObservation(
                kinematics=kinematics_solver,
                motor_names=motor_names,
                # motor_names=arm_motor_names
                # motor_names=ik_joint_names,
            )
        )

    if cfg.processor.image_preprocessing is not None:
        env_pipeline_steps.append(
            ImageCropResizeProcessorStep(
                crop_params_dict=cfg.processor.image_preprocessing.crop_params_dict,
                resize_size=cfg.processor.image_preprocessing.resize_size,
            )
        )

    # Add time limit processor if reset config exists
    if cfg.processor.reset is not None:
        env_pipeline_steps.append(
            TimeLimitProcessorStep(max_episode_steps=int(cfg.processor.reset.control_time_s * cfg.fps))
        )

    # Add gripper penalty processor if gripper config exists and enabled
    if cfg.processor.gripper is not None and cfg.processor.gripper.use_gripper:
        env_pipeline_steps.append(
            GripperPenaltyProcessorStep( #需要等到action走通
                penalty=cfg.processor.gripper.gripper_penalty,
                max_gripper_pos=cfg.processor.max_gripper_pos,
            )
        )

    if (
        cfg.processor.reward_classifier is not None
        and cfg.processor.reward_classifier.pretrained_path is not None
    ):
        env_pipeline_steps.append(
            RewardClassifierProcessorStep(
                pretrained_path=cfg.processor.reward_classifier.pretrained_path,
                device=device,
                success_threshold=cfg.processor.reward_classifier.success_threshold,
                success_reward=cfg.processor.reward_classifier.success_reward,
                terminate_on_success=terminate_on_success,
            )
        )

    env_pipeline_steps.append(AddBatchDimensionProcessorStep())
    env_pipeline_steps.append(DeviceProcessorStep(device=device))

    action_pipeline_steps = [
        AddTeleopActionAsComplimentaryDataStep(teleop_device=teleop_device),
        AddTeleopEventsAsInfoStep(teleop_device=teleop_device),
        InterventionActionProcessorStep(
            use_gripper=cfg.processor.gripper.use_gripper if cfg.processor.gripper is not None else False,
            terminate_on_success=terminate_on_success,
        ),
    ]
    action_pipeline_steps.append(RobotActionToPolicyActionProcessorStep(end_pose_name=end_pose_name))
    # Replace InverseKinematicsProcessor with new kinematic processors
    # if cfg.processor.inverse_kinematics is not None and kinematics_solver is not None:
    #     # Add EE bounds and safety processor
    #     inverse_kinematics_steps = [
    #         MapTensorToDeltaActionDictStep(
    #             use_gripper=cfg.processor.gripper.use_gripper if cfg.processor.gripper is not None else False
    #         ),
    #         MapDeltaActionToRobotActionStep(),
    #         EEReferenceAndDelta(
    #             kinematics=kinematics_solver,
    #             end_effector_step_sizes=cfg.processor.inverse_kinematics.end_effector_step_sizes,
    #             motor_names=motor_names,
    #             # motor_names=arm_motor_names,
    #             # motor_names=ik_joint_names,
    #             use_latched_reference=False,#测试一下True
    #             use_ik_solution=True,
    #         ),
    #         EEBoundsAndSafety(
    #             end_effector_bounds=cfg.processor.inverse_kinematics.end_effector_bounds,
    #         ),
    #         GripperVelocityToJoint(
    #             clip_max=cfg.processor.max_gripper_pos,
    #             speed_factor=1.0,
    #             discrete_gripper=False,
    #         ),
    #         InverseKinematicsRLStep(
    #             kinematics=kinematics_solver, 
    #             motor_names=motor_names, 
    #             # motor_names=arm_motor_names,
    #             # motor_names=ik_joint_names
    #             initial_guess_current_joints=False
    #         ),
    #     ]
    #     action_pipeline_steps.extend(inverse_kinematics_steps)
    #     action_pipeline_steps.append(RobotActionToPolicyActionProcessorStep(motor_names=motor_names))

    return DataProcessorPipeline(
        steps=env_pipeline_steps, to_transition=identity_transition, to_output=identity_transition
    ), DataProcessorPipeline(
        steps=action_pipeline_steps, to_transition=identity_transition, to_output=identity_transition
    )


def step_env_and_process_transition(
    env: gym.Env,
    transition: EnvTransition,
    action: torch.Tensor,
    env_processor: DataProcessorPipeline[EnvTransition, EnvTransition],
    action_processor: DataProcessorPipeline[EnvTransition, EnvTransition],
) -> EnvTransition:
    """
    Execute one step with processor pipeline.

    Args:
        env: The robot environment
        transition: Current transition state
        action: Action to execute
        env_processor: Environment processor
        action_processor: Action processor

    Returns:
        Processed transition with updated state.
    """
    # pdb.set_trace()
    # Create action transition
    transition[TransitionKey.ACTION] = action
    transition[TransitionKey.OBSERVATION] = (
        env.get_raw_end_pose_value() if hasattr(env, "get_raw_end_pose_value") else {}
    )
    processed_action_transition = action_processor(transition)
    processed_action = processed_action_transition[TransitionKey.ACTION]


    # Debug: 打印当前一步最终送入 env 的 action（14 维）
    try:
        pa_np = (
            processed_action.detach().cpu().numpy()
            if hasattr(processed_action, "detach")
            else np.asarray(processed_action)
        )
        #print("[step_env_and_process_transition] processed_action:", pa_np)
    except Exception:
        print("[step_env_and_process_transition] processed_action (non-numpy printable):", processed_action)

    transition_info = processed_action_transition[TransitionKey.INFO]
    obs, reward, terminated, truncated, info = env.step(processed_action, transition_info)

    reward = reward + processed_action_transition[TransitionKey.REWARD]
    terminated = terminated or processed_action_transition[TransitionKey.DONE]
    truncated = truncated or processed_action_transition[TransitionKey.TRUNCATED]
    complementary_data = processed_action_transition[TransitionKey.COMPLEMENTARY_DATA].copy()
    new_info = processed_action_transition[TransitionKey.INFO].copy()
    new_info.update(info)

    new_transition = create_transition(
        observation=obs,
        action=processed_action,
        reward=reward,
        done=terminated,
        truncated=truncated,
        info=new_info,
        complementary_data=complementary_data,
    )
    new_transition = env_processor(new_transition)
    
    # 可视化裁剪/缩放后的图像
    save_cropped_images_from_processed_obs(new_transition[TransitionKey.OBSERVATION])

    return new_transition


def control_loop(
    env: gym.Env,
    env_processor: DataProcessorPipeline[EnvTransition, EnvTransition],
    action_processor: DataProcessorPipeline[EnvTransition, EnvTransition],
    teleop_device: Teleoperator,
    cfg: GymManipulatorConfig,
) -> None:
    """Main control loop for robot environment interaction.
    if cfg.mode == "record": then a dataset will be created and recorded

    Args:
     env: The robot environment
     env_processor: Environment processor
     action_processor: Action processor
     teleop_device: Teleoperator device
     cfg: gym_manipulator configuration
    """
    dt = 1.0 / cfg.env.fps

    print(f"Starting control loop at {cfg.env.fps} FPS")
    print("Controls:")
    print("- Use gamepad/teleop device for intervention")
    print("- When not intervening, robot will stay still")
    print("- Press Ctrl+C to exit")

    # Reset environment and processors
    obs, info = env.reset()
    # pdb.set_trace()
    complementary_data = (
        {"raw_end_pose_value": info.pop("raw_end_pose_value")} if "raw_end_pose_value" in info else {}
    )
    env_processor.reset()
    action_processor.reset()

    # Process initial observation
    # pdb.set_trace()
    transition = create_transition(observation=obs, info=info, complementary_data=complementary_data)
    transition = env_processor(data=transition)

    # Determine if gripper is used
    use_gripper = cfg.env.processor.gripper.use_gripper if cfg.env.processor.gripper is not None else True

    dataset = None
    if cfg.mode == "record":
        action_features = teleop_device.action_features
        features = {
            ACTION: action_features,
            REWARD: {"dtype": "float32", "shape": (1,), "names": None},
            DONE: {"dtype": "bool", "shape": (1,), "names": None},
        }
        if use_gripper:
            features["complementary_info.discrete_penalty"] = {
                "dtype": "float32",
                "shape": (1,),
                "names": ["discrete_penalty"],
            }

        for key, value in transition[TransitionKey.OBSERVATION].items():
            if key == OBS_STATE:
                features[key] = {
                    "dtype": "float32",
                    "shape": value.squeeze(0).shape,
                    "names": None,
                }
            if "image" in key:
                features[key] = {
                    "dtype": "video",
                    "shape": value.squeeze(0).shape,
                    "names": ["channels", "height", "width"],
                }

        # Create dataset
        dataset = LeRobotDataset.create(
            cfg.dataset.repo_id,
            cfg.env.fps,
            root=cfg.dataset.root,
            use_videos=True,
            image_writer_threads=4,
            image_writer_processes=0,
            features=features,
        )

    episode_idx = 0
    episode_step = 0
    episode_start_time = time.perf_counter()

    while episode_idx < cfg.dataset.num_episodes_to_record:
        step_start_time = time.perf_counter()

        # Create a neutral action (no movement)
        neutral_action = torch.tensor([0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype=torch.float32)
        # if use_gripper:
            # neutral_action = torch.cat([neutral_action, torch.tensor([0.0])])  # Gripper stay

        # Use the new step function
        transition = step_env_and_process_transition(
            env=env,
            transition=transition,
            action=neutral_action,
            env_processor=env_processor,
            action_processor=action_processor,
        )
        terminated = transition.get(TransitionKey.DONE, False)
        truncated = transition.get(TransitionKey.TRUNCATED, False)

        if cfg.mode == "record":
            observations = {
                k: v.squeeze(0).cpu()
                for k, v in transition[TransitionKey.OBSERVATION].items()
                if isinstance(v, torch.Tensor)
            }
            # Use teleop_action if available, otherwise use the action from the transition
            action_to_record = transition[TransitionKey.COMPLEMENTARY_DATA].get(
                "teleop_action", transition[TransitionKey.ACTION]
            )
            frame = {
                **observations,
                ACTION: action_to_record.cpu(),
                REWARD: np.array([transition[TransitionKey.REWARD]], dtype=np.float32),
                DONE: np.array([terminated or truncated], dtype=bool),
            }
            if use_gripper:
                discrete_penalty = transition[TransitionKey.COMPLEMENTARY_DATA].get("discrete_penalty", 0.0)
                frame["complementary_info.discrete_penalty"] = np.array([discrete_penalty], dtype=np.float32)

            if dataset is not None:
                frame["task"] = cfg.dataset.task
                dataset.add_frame(frame)

        episode_step += 1

        # Handle episode termination
        if terminated or truncated:
            episode_time = time.perf_counter() - episode_start_time
            logging.info(
                f"Episode ended after {episode_step} steps in {episode_time:.1f}s with reward {transition[TransitionKey.REWARD]}"
            )
            episode_step = 0
            episode_idx += 1

            if dataset is not None:
                if transition[TransitionKey.INFO].get(TeleopEvents.RERECORD_EPISODE, False):
                    logging.info(f"Re-recording episode {episode_idx}")
                    dataset.clear_episode_buffer()
                    episode_idx -= 1
                else:
                    logging.info(f"Saving episode {episode_idx}")
                    dataset.save_episode()

            # Reset for new episode
            obs, info = env.reset()
            # pdb.set_trace()
            env_processor.reset()
            action_processor.reset()

            transition = create_transition(observation=obs, info=info)
            transition = env_processor(transition)

        # Maintain fps timing
        busy_wait(dt - (time.perf_counter() - step_start_time))

    if dataset is not None and cfg.dataset.push_to_hub:
        logging.info("Pushing dataset to hub")
        dataset.push_to_hub()


def replay_trajectory(
    env: gym.Env, action_processor: DataProcessorPipeline, cfg: GymManipulatorConfig
) -> None:
    """Replay recorded trajectory on robot environment."""
    assert cfg.dataset.replay_episode is not None, "Replay episode must be provided for replay"

    dataset = LeRobotDataset(
        cfg.dataset.repo_id,
        root=cfg.dataset.root,
        episodes=[cfg.dataset.replay_episode],
        download_videos=False,
    )
    episode_frames = dataset.hf_dataset.filter(lambda x: x["episode_index"] == cfg.dataset.replay_episode)
    actions = episode_frames.select_columns(ACTION)

    _, info = env.reset()

    for action_data in actions:
        start_time = time.perf_counter()
        transition = create_transition(
            observation=env.get_raw_joint_positions() if hasattr(env, "get_raw_joint_positions") else {},
            action=action_data[ACTION],
        )
        transition = action_processor(transition)
        env.step(transition[TransitionKey.ACTION])
        busy_wait(1 / cfg.env.fps - (time.perf_counter() - start_time))


@parser.wrap()
def main(cfg: GymManipulatorConfig) -> None:
    """Main entry point for gym manipulator script."""
    env, teleop_device = make_robot_env(cfg.env)
    env_processor, action_processor = make_processors(env, teleop_device, cfg.env, cfg.device)
    
    print("Environment observation space:", env.observation_space)
    print("Environment action space:", env.action_space)
    print("Environment processor:", env_processor)
    print("Action processor:", action_processor)

    if cfg.mode == "replay":
        replay_trajectory(env, action_processor, cfg)
        exit()

    control_loop(env, env_processor, action_processor, teleop_device, cfg)


if __name__ == "__main__":
    main()
