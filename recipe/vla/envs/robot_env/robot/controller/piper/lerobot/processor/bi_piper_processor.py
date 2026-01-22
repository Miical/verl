import math
import time
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar, runtime_checkable

import numpy as np
import torch
import torchvision.transforms.functional as F  # noqa: N812

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.teleoperators.utils import TeleopEvents

from .core import EnvTransition, PolicyAction, TransitionKey
from .pipeline import (
    ComplementaryDataProcessorStep,
    InfoProcessorStep,
    ObservationProcessorStep,
    ProcessorStep,
    ProcessorStepRegistry,
    TruncatedProcessorStep,
)

GRIPPER_KEY = "gripper"
DISCRETE_PENALTY_KEY = "discrete_penalty"
TELEOP_ACTION_KEY = "teleop_action"
@runtime_checkable
class HasTeleopEvents(Protocol):
    """
    Minimal protocol for objects that provide teleoperation events.

    This protocol defines the `get_teleop_events()` method, allowing processor
    steps to interact with teleoperators that support event-based controls
    (like episode termination or success flagging) without needing to know the
    teleoperator's specific class.
    """

    def get_teleop_events(self) -> dict[str, Any]:
        """
        Get extra control events from the teleoperator.

        Returns:
            A dictionary containing control events such as:
            - `is_intervention`: bool - Whether the human is currently intervening.
            - `terminate_episode`: bool - Whether to terminate the current episode.
            - `success`: bool - Whether the episode was successful.
            - `rerecord_episode`: bool - Whether to rerecord the episode.
        """
        ...


# Type variable constrained to Teleoperator subclasses that also implement events
TeleopWithEvents = TypeVar("TeleopWithEvents", bound=Teleoperator)


def _check_teleop_with_events(teleop: Teleoperator) -> None:
    """
    Runtime check that a teleoperator implements the `HasTeleopEvents` protocol.

    Args:
        teleop: The teleoperator instance to check.

    Raises:
        TypeError: If the teleoperator does not have a `get_teleop_events` method.
    """
    if not isinstance(teleop, HasTeleopEvents):
        raise TypeError(
            f"Teleoperator {type(teleop).__name__} must implement get_teleop_events() method. "
            f"Compatible teleoperators: GamepadTeleop, KeyboardEndEffectorTeleop"
        )

@ProcessorStepRegistry.register("add_teleop_action_as_complementary_data_bipiper")
@dataclass
class AddTeleopActionAsComplimentaryDataStepBipiper(ComplementaryDataProcessorStep):
    """
    Adds the raw action from a teleoperator to the transition's complementary data.

    This is useful for human-in-the-loop scenarios where the human's input needs to
    be available to downstream processors, for example, to override a policy's action
    during an intervention.

    Attributes:
        teleop_device: The teleoperator instance to get the action from.
    """

    teleop_device: Teleoperator

    def complementary_data(self, complementary_data: dict) -> dict:
        """
        Retrieves the teleoperator's action and adds it to the complementary data.

        Args:
            complementary_data: The incoming complementary data dictionary.

        Returns:
            A new dictionary with the teleoperator action added under the
            `teleop_action` key.
        """
        new_complementary_data = dict(complementary_data)
        new_complementary_data[TELEOP_ACTION_KEY] = self.teleop_device.get_action()
        # pdb.set_trace()
        return new_complementary_data

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features

@ProcessorStepRegistry.register("add_teleop_action_as_info_bipiper")
@dataclass
class AddTeleopEventsAsInfoStepBipiper(InfoProcessorStep):
    """
    Adds teleoperator control events (e.g., terminate, success) to the transition's info.

    This step extracts control events from teleoperators that support event-based
    interaction, making these signals available to other parts of the system.

    Attributes:
        teleop_device: An instance of a teleoperator that implements the
                       `HasTeleopEvents` protocol.
    """

    teleop_device: TeleopWithEvents

    def __post_init__(self):
        """Validates that the provided teleoperator supports events after initialization."""
        _check_teleop_with_events(self.teleop_device)

    def info(self, info: dict) -> dict:
        """
        Retrieves teleoperator events and updates the info dictionary.

        Args:
            info: The incoming info dictionary.

        Returns:
            A new dictionary including the teleoperator events.
        """
        new_info = dict(info)
        
        teleop_events = self.teleop_device.get_teleop_events()
        new_info.update(teleop_events)
        # if (teleop_events[TeleopEvents.IS_INTERVENTION]):
        #     print(new_info)
        return new_info

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@dataclass
@ProcessorStepRegistry.register("intervention_action_processor_bipiper")
class InterventionActionProcessorStepBipiper(ProcessorStep):
    """
    Handles human intervention, overriding policy actions and managing episode termination.

    When an intervention is detected (via teleoperator events in the `info` dict),
    this step replaces the policy's action with the human's teleoperated action.
    It also processes signals to terminate the episode or flag success.

    Attributes:
        use_gripper: Whether to include the gripper in the teleoperated action.
        terminate_on_success: If True, automatically sets the `done` flag when a
                              `success` event is received.
    """

    use_gripper: bool = False
    terminate_on_success: bool = True

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """
        Processes the transition to handle interventions.

        Args:
            transition: The incoming environment transition.

        Returns:
            The modified transition, potentially with an overridden action, updated
            reward, and termination status.
        """
        # pdb.set_trace()
        action = transition.get(TransitionKey.ACTION)
        if not isinstance(action, PolicyAction):
            raise ValueError(f"Action should be a PolicyAction type got {type(action)}")

        # Get intervention signals from complementary data
        info = transition.get(TransitionKey.INFO, {})
        complementary_data = transition.get(TransitionKey.COMPLEMENTARY_DATA, {})
        teleop_action = complementary_data.get(TELEOP_ACTION_KEY, {})
        # print(teleop_action)
        is_intervention = info.get(TeleopEvents.IS_INTERVENTION, False)
        terminate_episode = info.get(TeleopEvents.TERMINATE_EPISODE, False)
        success = info.get(TeleopEvents.SUCCESS, False)
        rerecord_episode = info.get(TeleopEvents.RERECORD_EPISODE, False)

        new_transition = transition.copy()
        # print("is_intervention:",is_intervention)
        # Override action if intervention is active
        if is_intervention and teleop_action is not None:
            if isinstance(teleop_action, dict):
                # Convert teleop_action dict to tensor format
                action_list = [
                    teleop_action.get("delta_x", 0.0),
                    teleop_action.get("delta_y", 0.0),
                    teleop_action.get("delta_z", 0.0),
                ]
                print(action_list)
                if self.use_gripper:
                    action_list.append(teleop_action.get(GRIPPER_KEY, 1.0))
            elif isinstance(teleop_action, np.ndarray):
                action_list = teleop_action.tolist()
            else:
                action_list = teleop_action

            teleop_action_tensor = torch.tensor(action_list, dtype=action.dtype, device=action.device)
            new_transition[TransitionKey.ACTION] = teleop_action_tensor

        # Handle episode termination
        new_transition[TransitionKey.DONE] = bool(terminate_episode) or (
            self.terminate_on_success and success
        )
        new_transition[TransitionKey.REWARD] = float(success)

        # Update info with intervention metadata
        info = new_transition.get(TransitionKey.INFO, {})
        info[TeleopEvents.IS_INTERVENTION] = is_intervention
        info[TeleopEvents.RERECORD_EPISODE] = rerecord_episode
        info[TeleopEvents.SUCCESS] = success
        new_transition[TransitionKey.INFO] = info

        # Update complementary data with teleop action
        complementary_data = new_transition.get(TransitionKey.COMPLEMENTARY_DATA, {})
        complementary_data[TELEOP_ACTION_KEY] = new_transition.get(TransitionKey.ACTION)
        new_transition[TransitionKey.COMPLEMENTARY_DATA] = complementary_data

        return new_transition

    def get_config(self) -> dict[str, Any]:
        """
        Returns the configuration of the step for serialization.

        Returns:
            A dictionary containing the step's configuration attributes.
        """
        return {
            "use_gripper": self.use_gripper,
            "terminate_on_success": self.terminate_on_success,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features



#########------------------##########

from dataclasses import dataclass

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature

from .core import PolicyAction, RobotAction
from .pipeline import ActionProcessorStep, ProcessorStepRegistry, RobotActionProcessorStep


@ProcessorStepRegistry.register("map_tensor_to_delta_action_dict_bipiper")
@dataclass
class MapTensorToDeltaActionDictStepBipiper(ActionProcessorStep):
    """
    Maps a flat action tensor from a policy to a structured delta action dictionary.

    This step is typically used after a policy outputs a continuous action vector.
    It decomposes the vector into named components for delta movements of the
    end-effector (x, y, z) and optionally the gripper.

    Attributes:
        use_gripper: If True, assumes the 4th element of the tensor is the
                     gripper action.
    """

    use_gripper: bool = True

    def action(self, action: PolicyAction) -> RobotAction:
        # import pdb
        # pdb.set_trace()
        if not isinstance(action, PolicyAction):
            raise ValueError("Only PolicyAction is supported for this processor")

        if action.dim() > 1:
            action = action.squeeze(0)

        # TODO (maractingi): add rotation
        delta_action = {
            "delta_x": action[0].item(),
            "delta_y": action[1].item(),
            "delta_z": action[2].item(),
        }
        if self.use_gripper:
            delta_action["gripper"] = action[3].item()
        return delta_action

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        for axis in ["x", "y", "z"]:
            features[PipelineFeatureType.ACTION][f"delta_{axis}"] = PolicyFeature(
                type=FeatureType.ACTION, shape=(1,)
            )

        if self.use_gripper:
            features[PipelineFeatureType.ACTION]["gripper"] = PolicyFeature(
                type=FeatureType.ACTION, shape=(1,)
            )
        return features


@ProcessorStepRegistry.register("map_delta_action_to_robot_action_bipiper")
@dataclass
class MapDeltaActionToRobotActionStepBipiper(RobotActionProcessorStep):
    """
    Maps delta actions from teleoperators to robot target actions for inverse kinematics.

    This step converts a dictionary of delta movements (e.g., from a gamepad)
    into a target action format that includes an "enabled" flag and target
    end-effector positions. It also handles scaling and noise filtering.

    Attributes:
        position_scale: A factor to scale the delta position inputs.
        noise_threshold: The magnitude below which delta inputs are considered noise
                         and do not trigger an "enabled" state.
    """

    # Scale factors for delta movements
    position_scale: float = 1.0
    noise_threshold: float = 1e-3  # 1 mm threshold to filter out noise

    def action(self, action: RobotAction) -> RobotAction:
        # NOTE (maractingi): Action can be a dict from the teleop_devices or a tensor from the policy
        # TODO (maractingi): changing this target_xyz naming convention from the teleop_devices
        delta_x = action.pop("delta_x")
        delta_y = action.pop("delta_y")
        delta_z = action.pop("delta_z")
        gripper = action.pop("gripper")

        # Determine if the teleoperator is actively providing input
        # Consider enabled if any significant movement delta is detected
        position_magnitude = (delta_x**2 + delta_y**2 + delta_z**2) ** 0.5  # Use Euclidean norm for position
        enabled = position_magnitude > self.noise_threshold  # Small threshold to avoid noise

        # Scale the deltas appropriately
        scaled_delta_x = delta_x * self.position_scale
        scaled_delta_y = delta_y * self.position_scale
        scaled_delta_z = delta_z * self.position_scale

        # For gamepad/keyboard, we don't have rotation input, so set to 0
        # These could be extended in the future for more sophisticated teleoperators
        target_wx = 0.0
        target_wy = 0.0
        target_wz = 0.0

        # Update action with robot target format
        action = {
            "enabled": enabled,
            "target_x": scaled_delta_x,
            "target_y": scaled_delta_y,
            "target_z": scaled_delta_z,
            "target_wx": target_wx,
            "target_wy": target_wy,
            "target_wz": target_wz,
            "gripper_vel": float(gripper),
        }

        return action

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        for axis in ["x", "y", "z", "gripper"]:
            features[PipelineFeatureType.ACTION].pop(f"delta_{axis}", None)

        for feat in ["enabled", "target_x", "target_y", "target_z", "target_wx", "target_wy", "target_wz"]:
            features[PipelineFeatureType.ACTION][f"{feat}"] = PolicyFeature(
                type=FeatureType.ACTION, shape=(1,)
            )

        return features