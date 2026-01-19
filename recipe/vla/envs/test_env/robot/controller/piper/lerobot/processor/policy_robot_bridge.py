#!/usr/bin/env python

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
import math
from dataclasses import asdict, dataclass
from tkinter.constants import TRUE
from typing import Any

import torch
import numpy as np
import meshcat.transformations as tf
import pdb
from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor import ActionProcessorStep, PolicyAction, ProcessorStepRegistry, RobotAction
from lerobot.utils.constants import ACTION
from .core import TransitionKey
from lerobot.teleoperators.utils import TeleopEvents

@dataclass
@ProcessorStepRegistry.register("robot_action_to_policy_action_processor")
class RobotActionToPolicyActionProcessorStep(ActionProcessorStep):
    """Processor step to map a dictionary to a tensor action."""

    end_pose_name: list[str]
    prev_is_intervention: bool = False
    prev_obs_tensor: torch.Tensor = None

    def _apply_rotation_delta(self, euler_angles: torch.Tensor, axis_angle_delta: torch.Tensor) -> torch.Tensor:
        """
        Apply axis-angle rotation delta to Euler angles.
        
        Args:
            euler_angles: Euler angles (rx, ry, rz) in radians, shape (3,)
            axis_angle_delta: Axis-angle rotation delta, shape (3,)
        
        Returns:
            New Euler angles after applying the rotation delta
        """
        # Convert Euler angles (rx, ry, rz) to quaternion
        # Use XYZ intrinsic convention (rxyz) - rotations applied in X, Y, Z order
        # This assumes rx, ry, rz are rotations around X, Y, Z axes respectively
        euler_np = euler_angles.detach().cpu().numpy()
        quat = tf.quaternion_from_euler(euler_np[0], euler_np[1], euler_np[2], axes='sxyz')
        
        # Convert axis-angle delta to quaternion
        axis_angle_np = axis_angle_delta.detach().cpu().numpy()
        
        # Fix rotation direction issue: controller rotation -> opposite robot rotation
        # Configuration flags to test different combinations:
        # Set to True to negate that axis (reverse rotation direction)
        NEGATE_X_AXIS = False  # Roll
        NEGATE_Y_AXIS = False  # Pitch  
        NEGATE_Z_AXIS = False  # Yaw
        
        # Based on testing feedback:
        # - Negating all axes made it worse
        # - Need to find the right combination
        # Try different combinations:
        # Option 1: Negate X and Y only
        # Option 2: Negate X only
        # Option 3: Negate Y and Z only
        # Option 4: Negate Z only
        # Option 5: No negation (standard)
        
        if NEGATE_X_AXIS:
            axis_angle_np[0] = -axis_angle_np[0]
        if NEGATE_Y_AXIS:
            axis_angle_np[1] = -axis_angle_np[1]
        if NEGATE_Z_AXIS:
            axis_angle_np[2] = -axis_angle_np[2]
        
        angle = np.linalg.norm(axis_angle_np)
        if angle > 1e-6:
            axis = axis_angle_np / angle
            delta_quat = tf.quaternion_about_axis(angle, axis)
        else:
            delta_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Apply rotation: new_quat = delta_quat * quat (standard left multiply)
        new_quat = tf.quaternion_multiply(delta_quat, quat)
        
        # Convert back to Euler angles using same convention
        euler_new = tf.euler_from_quaternion(new_quat, axes='sxyz')
        
        return torch.tensor(euler_new, dtype=euler_angles.dtype, device=euler_angles.device)

    def action(self, action: RobotAction) -> PolicyAction:


        if len(self.end_pose_name) != len(action):
            raise ValueError(f"Action must have {len(self.end_pose_name)} elements, got {len(action)}")
        obs_dict = self.transition[TransitionKey.OBSERVATION]
        obs_keys = [f"{name}.value" for name in self.end_pose_name]
        obs_vals = [float(obs_dict[k]) for k in obs_keys]
        obs_tensor = torch.tensor(obs_vals, dtype=action.dtype, device=action.device)
        
        process_action = obs_tensor

        '''
        if self.prev_obs_tensor is None:
            self.prev_obs_tensor = obs_tensor
        
        transition_info = self.transition[TransitionKey.INFO]
        if TeleopEvents.IS_INTERVENTION in transition_info:
            is_intervention = transition_info[TeleopEvents.IS_INTERVENTION]
            
            if is_intervention:
                #process_action = action + obs_tensor
                print(f"[RobotActionToPolicyActionProcessorStep]: intervention")
                # Start from previous observation pose
                process_action = self.prev_obs_tensor.clone()
                
                # Apply translation deltas (x, y, z) - these can be added directly
                # Left arm: indices 0, 1, 2
                process_action[0] = self.prev_obs_tensor[0] + action[0]
                process_action[1] = self.prev_obs_tensor[1] + action[1]
                process_action[2] = self.prev_obs_tensor[2] + action[2]
                # Right arm: indices 7, 8, 9
                process_action[7] = self.prev_obs_tensor[7] + action[7]
                process_action[8] = self.prev_obs_tensor[8] + action[8]
                process_action[9] = self.prev_obs_tensor[9] + action[9]
                
                # Apply rotation deltas (rx, ry, rz) - need proper rotation composition
                # Left arm rotation: indices 3, 4, 5
                left_euler = self.prev_obs_tensor[3:6]
                left_delta = action[3:6]
                process_action[3:6] = self._apply_rotation_delta(left_euler, left_delta)
                
                # Right arm rotation: indices 10, 11, 12
                right_euler = self.prev_obs_tensor[10:13]
                right_delta = action[10:13]
                process_action[10:13] = self._apply_rotation_delta(right_euler, right_delta)
                
                # Gripper deltas are absolute, not relative
                process_action[6] = action[6]  # Left gripper
                process_action[13] = action[13]  # Right gripper
            if is_intervention and not self.prev_is_intervention:
                self.prev_obs_tensor = obs_tensor
            self.prev_is_intervention = is_intervention

            # if not is_intervention:
            #     """
            #     限位:
            #     - 左臂: 0–5 → dx, dy, dz, drx, dry, drz
            #     - 右臂: 7–12 → dx, dy, dz, drx, dry, drz
            #     - 夹爪: 6, 13 不限位
            #     位移 ±0.3 米, 姿态 ±30 度(π/6 弧度)
            #     """
            #     # === 限位 ===
            #     pos_limit = 0.01
            #     rot_limit = math.pi / 180

            #     # 位移索引（左右臂）
            #     pos_idx = [0, 1, 2, 7, 8, 9]
            #     # 姿态索引（左右臂）
            #     rot_idx = [3, 4, 5, 10, 11, 12]

            #     # clamp 限幅
            #     action[pos_idx] = torch.clamp(action[pos_idx], -pos_limit, pos_limit)
            #     action[rot_idx] = torch.clamp(action[rot_idx], -rot_limit, rot_limit)
            #     process_action = action + obs_tensor
                # print("触发限位")
        '''
        # origin_obs_tensor_action = torch.tensor([5.6127e-02, 0.0000e+00, 2.1327e-01, 0.0000e+00, 1.4835e+00, 0.0000e+00,
        # 1.4700e-03, 5.6127e-02, 0.0000e+00, 2.1327e-01, 0.0000e+00, 1.4835e+00,
        # 0.0000e+00, 2.3100e-03], dtype=action.dtype, device=action.device)
        # process_action = action + origin_obs_tensor_action
        
        # 在 policy_robot_bridge.py 的 action() 方法中添加
        # print(f"[单位检查] delta_rot 范围: [{action[3]:.6f}, {action[4]:.6f}, {action[5]:.6f}] rad")
        # print(f"[单位检查] obs_rot 范围: [{obs_tensor[3]:.6f}, {obs_tensor[4]:.6f}, {obs_tensor[5]:.6f}] rad")
        # print(f"[单位检查] 结果范围: [{process_action[3]:.6f}, {process_action[4]:.6f}, {process_action[5]:.6f}] rad")
        #print(f"[RobotActionToPolicyActionProcessorStep]: delta action: {action}, total action: {process_action}")
        return process_action

    def get_config(self) -> dict[str, Any]:
        return asdict(self)

    def transform_features(self, features):
        features[PipelineFeatureType.ACTION][ACTION] = PolicyFeature(
            type=FeatureType.ACTION, shape=(len(self.motor_names),)
        )
        return features

# @dataclass
# @ProcessorStepRegistry.register("robot_action_to_policy_action_processor")
# class RobotActionToPolicyActionProcessorStep(ActionProcessorStep):
#     """Processor step to map a dictionary to a tensor action."""

#     motor_names: list[str]

#     def action(self, action: RobotAction) -> PolicyAction:
#         pdb.set_trace()
#         if len(self.motor_names) != len(action):
#             raise ValueError(f"Action must have {len(self.motor_names)} elements, got {len(action)}")
#         return torch.tensor([action[f"{name}.pos"] for name in self.motor_names])

#     def get_config(self) -> dict[str, Any]:
#         return asdict(self)

#     def transform_features(self, features):
#         features[PipelineFeatureType.ACTION][ACTION] = PolicyFeature(
#             type=FeatureType.ACTION, shape=(len(self.motor_names),)
#         )
#         return features


@dataclass
@ProcessorStepRegistry.register("policy_action_to_robot_action_processor")
class PolicyActionToRobotActionProcessorStep(ActionProcessorStep):
    """Processor step to map a policy action to a robot action."""

    motor_names: list[str]

    def action(self, action: PolicyAction) -> RobotAction:
        if len(self.motor_names) != len(action):
            raise ValueError(f"Action must have {len(self.motor_names)} elements, got {len(action)}")
        return {f"{name}.pos": action[i] for i, name in enumerate(self.motor_names)}

    def get_config(self) -> dict[str, Any]:
        return asdict(self)

    def transform_features(self, features):
        for name in self.motor_names:
            features[PipelineFeatureType.ACTION][f"{name}.pos"] = PolicyFeature(
                type=FeatureType.ACTION, shape=(1,)
            )
        return features
