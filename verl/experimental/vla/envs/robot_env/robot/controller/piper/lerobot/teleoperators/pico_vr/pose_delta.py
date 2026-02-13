from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import meshcat.transformations as tf
import numpy as np

from .geometry import R_HEADSET_TO_WORLD, quat_diff_as_angle_axis


def _normalize_quat(quat: np.ndarray) -> np.ndarray:
    quat = np.array(quat, dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return quat / norm


@dataclass
class PoseDeltaCalculator:
    """
    Helper that mirrors BaseTeleopController._process_xr_pose.
    """

    R_headset_world: np.ndarray | None = None
    scale_factor: float = 1.0

    def __post_init__(self):
        self.R_headset_world = (
            self.R_headset_world if self.R_headset_world is not None else np.eye(3)
        )
        # self.prev_xyz: np.ndarray | None = None
        # self.prev_quat: np.ndarray | None = None
        self.ref_xyz: np.ndarray | None = None
        self.ref_quat: np.ndarray | None = None
        self._prev_is_intervention: bool = False

    def reset_reference(self):
        self.ref_xyz = None
        self.ref_quat = None
        self._prev_is_intervention = False

    def update(self, xr_pose: Iterable[float], is_intervention: bool):
        """
        Args:
            xr_pose: Iterable of length 7 [tx, ty, tz, qx, qy, qz, qw] in headset frame.
        Returns:
            delta_xyz (np.ndarray): 3D translation delta in meters.
            delta_rot (np.ndarray): Axis-angle rotation delta.
        """
        xr_pose = list(xr_pose)
        controller_xyz = np.array(xr_pose[:3])
        controller_quat = np.array([xr_pose[6], xr_pose[3], xr_pose[4], xr_pose[5]])  # w, x, y, z
        controller_quat = _normalize_quat(controller_quat)

        controller_xyz = self.R_headset_world @ controller_xyz
        R_transform = np.eye(4)
        R_transform[:3, :3] = self.R_headset_world
        R_quat = tf.quaternion_from_matrix(R_transform)
        controller_quat = tf.quaternion_multiply(
            tf.quaternion_multiply(R_quat, controller_quat),
            tf.quaternion_conjugate(R_quat),
        )
        controller_quat = _normalize_quat(controller_quat)

        '''
        if self.ref_xyz is None:
            self.ref_xyz = controller_xyz.copy()
            self.ref_quat = controller_quat.copy()
            delta_xyz = np.zeros(3)
            delta_rot = np.zeros(3)
        else:
            delta_xyz = (controller_xyz - self.ref_xyz) * self.scale_factor
            delta_rot = quat_diff_as_angle_axis(self.ref_quat, controller_quat)
        '''
        # When intervention turns on, capture the current pose as the reference so deltas
        # start from zero even if the controller drifted while intervention was off.
        if is_intervention and not self._prev_is_intervention:
            self.ref_xyz = controller_xyz.copy()
            self.ref_quat = controller_quat.copy()

        if self.ref_xyz is None:
            self.ref_xyz = controller_xyz.copy()
            self.ref_quat = controller_quat.copy()

        if is_intervention:
            delta_xyz = (controller_xyz - self.ref_xyz) * self.scale_factor
            delta_rot = quat_diff_as_angle_axis(self.ref_quat, controller_quat)
        else:
            delta_xyz = np.zeros(3)
            delta_rot = np.zeros(3)

        self._prev_is_intervention = is_intervention
        
        return delta_xyz, delta_rot


@dataclass
class GripperDeltaTracker:
    """
    Tracks delta of a gripper analog input (trigger/grip) and maps it to opening.
    """

    trigger_name: str
    open_pos: float
    close_pos: float

    def __post_init__(self):
        self.ref_value: float | None = None

    def update(self, xr_client):
        value = xr_client.get_key_value_by_name(self.trigger_name)
        if self.ref_value is None:
            self.ref_value = value
        delta = value - self.ref_value
        # Map current trigger value (0-1) to opening distance
        opening = self.open_pos + (self.close_pos - self.open_pos) * np.clip(value, 0.0, 1.0)
        return {
            "raw": value,
            "delta": delta,
            "opening_m": opening,
        }

