from __future__ import annotations

import logging
from typing import Any

import numpy as np

from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from .config_pico_vr import PicoVrTeleopConfig
from .pose_delta import GripperDeltaTracker, PoseDeltaCalculator
from .geometry import R_HEADSET_TO_WORLD
from .xr_client import XrClient
from time import time

class PicoVrTeleop(Teleoperator):
    """
    Dual-arm teleoperation driven by Pico VR controllers.
    """

    config_class = PicoVrTeleopConfig
    name = "pico_vr"

    def __init__(self, config: PicoVrTeleopConfig):
        super().__init__(config)
        self.config = config
        self.client: XrClient | None = None

        self._left_pose_calc = PoseDeltaCalculator(
            R_headset_world=R_HEADSET_TO_WORLD,
            scale_factor=config.translation_scale,
        )
        self._right_pose_calc = PoseDeltaCalculator(
            R_headset_world=R_HEADSET_TO_WORLD,
            scale_factor=config.translation_scale,
        )
        self._left_gripper_tracker = GripperDeltaTracker(
            trigger_name=config.left_gripper_input,
            open_pos=config.gripper_open,
            close_pos=config.gripper_close,
        )
        self._right_gripper_tracker = GripperDeltaTracker(
            trigger_name=config.right_gripper_input,
            open_pos=config.gripper_open,
            close_pos=config.gripper_close,
        )

        self._connected = False
        self._intervention_latched = False
        self._last_action = self._zero_action()
        self._button_states: dict[str, bool] | None = None
        self._reset_button_prev = False
        self._intervention_reset_prev = False
        self._printInterval = 1 # 
        self._last_print_time = time()

    # ------------------------------------------------------------------ #
    # Teleoperator interface
    # ------------------------------------------------------------------ #
    @property
    def action_features(self) -> dict:
        return {
            "dtype": "float32",
            "shape": (14,),
            "names": {
                "left_delta_x": 0,
                "left_delta_y": 1,
                "left_delta_z": 2,
                "left_delta_rx": 3,
                "left_delta_ry": 4,
                "left_delta_rz": 5,
                "left_delta_gripper": 6,
                "right_delta_x": 7,
                "right_delta_y": 8,
                "right_delta_z": 9,
                "right_delta_rx": 10,
                "right_delta_ry": 11,
                "right_delta_rz": 12,
                "right_delta_gripper": 13,
            },
        }

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError("Pico VR teleop already connected.")
        self.client = XrClient()
        self._connected = True
        self._reset_pose_references()

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def get_action(self) -> dict[str, Any]:
        if not self.is_connected or self.client is None:
            raise DeviceNotConnectedError(
                "Pico VR teleop is not connected. Call `connect()` before `get_action()`."
            )
        self._button_states = self._fetch_button_states()
        self._maybe_handle_reset_buttons()

        left_xyz, left_rot = self._process_pose(
            self._left_pose_calc, self.config.left_pose_source, self._intervention_latched
        )
        right_xyz, right_rot = self._process_pose(
            self._right_pose_calc, self.config.right_pose_source, self._intervention_latched
        )
        ##
        #帮我加载这一块，左臂右臂的left_rot和right_rot数据，变换到基坐标系
        ##

        action = {
            "left_delta_x": left_xyz[0],
            "left_delta_y": left_xyz[1],
            "left_delta_z": left_xyz[2],
            "left_delta_rx": left_rot[0],
            "left_delta_ry": left_rot[1],
            "left_delta_rz": left_rot[2],
            "left_delta_gripper": self._get_gripper_delta(self._left_gripper_tracker),
            "right_delta_x": right_xyz[0],
            "right_delta_y": right_xyz[1],
            "right_delta_z": right_xyz[2],
            "right_delta_rx": right_rot[0],
            "right_delta_ry": right_rot[1],
            "right_delta_rz": right_rot[2],
            "right_delta_gripper": self._get_gripper_delta(self._right_gripper_tracker),
        }

        self._apply_output_limits(action)
        self._last_action = action
        # 不再自动检测手柄移动来设置干预标志，改为手动按钮切换
        # self._update_intervention_latch(action)
        
        # Debug: 周期性打印当前左右手控制器的增量，便于排查映射问题
        now = time()
        if now - self._last_print_time >= self._printInterval:
            left_mag = float(np.linalg.norm([left_xyz[0], left_xyz[1], left_xyz[2]]))
            right_mag = float(np.linalg.norm([right_xyz[0], right_xyz[1], right_xyz[2]]))
            print(
                "[PicoVrTeleop] action:",
                #f"L_xyz={left_xyz}, L_rot={left_rot},",
                #f"R_xyz={right_xyz}, R_rot={right_rot},",
                #f"|L_xyz|={left_mag:.4f}, |R_xyz|={right_mag:.4f},",
                f"intervention_latched={self._intervention_latched}",
            )
            self._last_print_time = now
        
        #当前遥操在另外一个单独的程序中实现，这里仅获取遥操的信号，而不发送遥操动作
        action = {
            "left_delta_x": 0,
            "left_delta_y": 0,
            "left_delta_z": 0,
            "left_delta_rx": 0,
            "left_delta_ry": 0,
            "left_delta_rz": 0,
            "left_delta_gripper": 0,
            "right_delta_x": 0,
            "right_delta_y": 0,
            "right_delta_z": 0,
            "right_delta_rx": 0,
            "right_delta_ry": 0,
            "right_delta_rz": 0,
            "right_delta_gripper": 0,
        }

        return action

    def get_teleop_events(self) -> dict[str, Any]:
        if not self.is_connected:
            return {
                TeleopEvents.IS_INTERVENTION: False,
                TeleopEvents.TERMINATE_EPISODE: False,
                TeleopEvents.SUCCESS: False,
                TeleopEvents.RERECORD_EPISODE: False,
            }

        if self._button_states is None:
            self._button_states = self._fetch_button_states()

        terminate = self._get_button_state(self.config.terminate_button)
        success = self._get_button_state(self.config.success_button)
        rerecord = self._get_button_state(self.config.rerecord_button)
        if bool(success):
            logging.info("Pico VR teleop: SUCCESS button pressed.")
        events = {
            TeleopEvents.IS_INTERVENTION: self._intervention_latched,
            TeleopEvents.TERMINATE_EPISODE: bool(terminate),
            TeleopEvents.SUCCESS: bool(success),
            TeleopEvents.RERECORD_EPISODE: bool(rerecord),
        }

        #print("events:", events)
        # Clear cache to avoid stale values on next tick
        self._button_states = None
        return events

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        del feedback  # Not supported

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError("Pico VR teleop is not connected.")
        if self.client is not None:
            try:
                self.client.close()
            except Exception as exc:  # pragma: no cover - SDK specific
                logging.warning("Failed to close XR client cleanly: %s", exc)
        self.client = None
        self._connected = False

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _zero_action(self) -> dict[str, float]:
        return {
            "left_delta_x": 0.0,
            "left_delta_y": 0.0,
            "left_delta_z": 0.0,
            "left_delta_rx": 0.0,
            "left_delta_ry": 0.0,
            "left_delta_rz": 0.0,
            "left_delta_gripper": 0.0,
            "right_delta_x": 0.0,
            "right_delta_y": 0.0,
            "right_delta_z": 0.0,
            "right_delta_rx": 0.0,
            "right_delta_ry": 0.0,
            "right_delta_rz": 0.0,
            "right_delta_gripper": 0.0,
        }

    def _fetch_button_states(self) -> dict[str, bool]:
        assert self.client is not None
        buttons = {
            self.config.reset_reference_button,
            self.config.intervention_reset_button,
            self.config.terminate_button,
            self.config.success_button,
            self.config.rerecord_button,
        }
        states: dict[str, bool] = {}
        for name in filter(None, buttons):
            try:
                states[name] = bool(self.client.get_button_state_by_name(name))
            except ValueError as exc:
                logging.warning("Unknown Pico VR button '%s': %s", name, exc)
        return states

    def _maybe_handle_reset_buttons(self) -> None:
        reset_pressed = self._get_button_state(self.config.reset_reference_button)
        if reset_pressed and not self._reset_button_prev:
            self._reset_pose_references()
        self._reset_button_prev = reset_pressed

        # 手动切换干预状态：按下按钮时切换 intervention_latched
        intervention_toggle = self._get_button_state(self.config.intervention_reset_button)
        if intervention_toggle and not self._intervention_reset_prev:
            # 按钮刚按下时，切换干预状态
            self._intervention_latched = not self._intervention_latched
            logging.info(f"Intervention toggled: {self._intervention_latched}")
        self._intervention_reset_prev = intervention_toggle

    def _process_pose(
        self,
        calculator: PoseDeltaCalculator,
        source_name: str,
        is_intervention: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        assert self.client is not None
        pose = self.client.get_pose_by_name(source_name)
        delta_xyz, delta_rot = calculator.update(pose, is_intervention)
        delta_xyz = self._apply_deadband(delta_xyz, self.config.translation_deadband)
        delta_rot = self._apply_deadband(delta_rot, self.config.rotation_deadband)

        delta_xyz = np.clip(delta_xyz, -self.config.translation_clip, self.config.translation_clip)
        # Apply global rotation scale first
        delta_rot = delta_rot * self.config.rotation_scale
        # Then reduce yaw (z-axis) sensitivity if requested.
        # Note: delta_rot is in axis-angle form; for small rotations, the z component
        # roughly corresponds to yaw around world Z.
        # if hasattr(self.config, "rotation_yaw_gain") and self.config.rotation_yaw_gain != 1.0:
        #     delta_rot[2] *= self.config.rotation_yaw_gain
        delta_rot = np.clip(delta_rot, -self.config.rotation_clip, self.config.rotation_clip)
        #print(f"[PicoVrTeleop] _process_pose: delta_xyz: {delta_xyz}, delta_rot: {delta_rot}, is_intervention: {is_intervention}")
        return delta_xyz, delta_rot

    def _apply_deadband(self, values: np.ndarray, deadband: float) -> np.ndarray:
        mask = np.abs(values) < deadband
        values = values.copy()
        values[mask] = 0.0
        return values

    def _get_gripper_delta(self, tracker: GripperDeltaTracker) -> float:
        assert self.client is not None
        grip = tracker.update(self.client)
        delta = grip["opening_m"] * self.config.gripper_delta_gain
        return delta
        # if abs(delta) < self.config.gripper_deadband:
        #     delta = 0.0
        # return float(np.clip(delta, -1.0, 1.0))
        

    def _apply_output_limits(self, action: dict[str, float]) -> None:
        for key, value in action.items():
            if key.endswith("_gripper"):
                action[key] = float(np.clip(value, -1.0, 1.0))
            else:
                action[key] = float(value)

    def _update_intervention_latch(self, action: dict[str, float]) -> None:
        values = np.array(
            [v for k, v in action.items() if not k.endswith("_gripper")], dtype=np.float64
        )
        print("_update_intervention_latch values:", values)
        magnitude = np.max(np.abs(values)) if values.size else 0.0
        if magnitude > self.config.intervention_threshold:
            print("intervention_latched set to true: magnitude:", magnitude)
            self._intervention_latched = True

    def _get_button_state(self, button_name: str | None) -> bool:
        if button_name is None or self._button_states is None:
            return False
        return self._button_states.get(button_name, False)

    def _reset_pose_references(self) -> None:
        self._left_pose_calc.reset_reference()
        self._right_pose_calc.reset_reference()
        self._left_gripper_tracker.ref_value = None
        self._right_gripper_tracker.ref_value = None
        # 不再自动重置干预状态，干预状态由用户手动按钮控制
        # self._intervention_latched = False

