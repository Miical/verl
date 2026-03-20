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
        
        # process_action = obs_tensor
        process_action = action
        print(f"process action:{process_action},action:{action}")
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
