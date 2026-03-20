#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.

#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, Dict

import numpy as np
import torch

from lerobot.model.kinematics import RobotKinematics
from lerobot.utils.constants import ACTION


#!/usr/bin/env python
# -*- coding: utf-8 -*-




@dataclass
class JointToEEDeltaConfig:
    urdf_path: str
    target_frame_name: str
    left_joint_indices: Sequence[int]   # indices in observation.state for arm joints (len=6)
    left_gripper_index: int             # index in observation.state for gripper position
    right_joint_indices: Sequence[int]  # indices in observation.state for arm joints (len=6)
    right_gripper_index: int            # index in observation.state for gripper position
    step_sizes: int                     # 把相邻两帧的位姿差均匀拆成多少小步
    clamp_abs: float = 1.0              # clamp ee deltas after normalization


def _wrap_to_pi(x: np.ndarray) -> np.ndarray:
    """
    把角度差值 wrap 到 [-pi, pi] 区间，避免从 +179° 跳到 -179° 这类大跳变。
    """
    # 先缩放到 [-2pi, 2pi]，再映射到 [-pi, pi]
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def _rotmat_to_rpy_zyx(R: np.ndarray) -> np.ndarray:
    """
    和测试脚本一致的欧拉角转换：
        - 输入: 3x3 旋转矩阵 R
        - 输出: [rx, ry, rz] = [roll, pitch, yaw] (弧度)
      约定: Z-Y-X (yaw-pitch-roll)，返回顺序 rx, ry, rz
    """
    # 避免数值误差
    r20 = -R[2, 0]
    r20_clamped = float(np.clip(r20, -1.0, 1.0))
    pitch = np.arcsin(r20_clamped)

    cos_pitch = np.cos(pitch)
    if abs(cos_pitch) < 1e-6:
        # 退化情况（接近 ±90°）
        roll = 0.0
        yaw = np.arctan2(-R[0, 1], R[1, 1])
    else:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])

    return np.array([float(roll), float(pitch), float(yaw)], dtype=float)


class JointToEEDeltaDataset:
    """
    Dataset wrapper:
      - 从相邻两帧的关节状态 (observation.state) 计算左右臂末端位姿增量 + 夹爪速度
      - 使用 URDF 的 FK 得到末端位姿
      - 输出动作：14 维
        [l_dx, l_dy, l_dz, l_drx, l_dry, l_drz, l_g_vel,
         r_dx, r_dy, r_dz, r_drx, r_dry, r_drz, r_g_vel]

    约定：
      - observation.state 中的关节值单位为 *弧度* (rad)
      - RobotKinematics.forward_kinematics() 接口接受 *度* (deg)，内部再转弧度
      - 这里的姿态增量采用 **欧拉角 (roll, pitch, yaw) 的差值**（弧度）
    """

    def __init__(self, base_dataset, cfg: JointToEEDeltaConfig):
        self.dataset = base_dataset
        self.cfg = cfg

        # 这里简单起见，左右臂共用同一个 URDF / target_frame_name
        self.left_kin = RobotKinematics(
            urdf_path=cfg.urdf_path,
            target_frame_name=cfg.target_frame_name,
            joint_names=[f"joint{i+1}" for i in range(len(cfg.left_joint_indices))],
        )
        self.right_kin = RobotKinematics(
            urdf_path=cfg.urdf_path,
            target_frame_name=cfg.target_frame_name,
            joint_names=[f"joint{i+1}" for i in range(len(cfg.right_joint_indices))],
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        cur = dict(self.dataset[idx])

        # 默认都设为 0（例如最后一帧 / episode 边界）
        l_dx = l_dy = l_dz = l_drx = l_dry = l_drz = l_g_vel = 0.0
        r_dx = r_dy = r_dz = r_drx = r_dry = r_drz = r_g_vel = 0.0

        if idx < len(self.dataset) - 1:
            nxt = self.dataset[idx + 1]
            if nxt["episode_index"] == cur["episode_index"]:
                # 当前帧 & 下一帧的状态向量（observation.state，单位：弧度）
                cur_state = (
                    cur["observation.state"].clone()
                    if isinstance(cur["observation.state"], torch.Tensor)
                    else torch.tensor(cur["observation.state"])
                )
                nxt_state = (
                    nxt["observation.state"].clone()
                    if isinstance(nxt["observation.state"], torch.Tensor)
                    else torch.tensor(nxt["observation.state"])
                )

                # ======================
                # 左臂：末端位置 + 姿态 + 夹爪
                # ======================
                ql_cur_rad = (
                    cur_state[self.cfg.left_joint_indices]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(float)
                )
                ql_nxt_rad = (
                    nxt_state[self.cfg.left_joint_indices]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(float)
                )

                # 弧度 -> 度（RobotKinematics 接口仍旧吃 deg）
                ql_cur_deg = np.rad2deg(ql_cur_rad)
                ql_nxt_deg = np.rad2deg(ql_nxt_rad)

                t_l_cur = self.left_kin.forward_kinematics(ql_cur_deg)
                t_l_nxt = self.left_kin.forward_kinematics(ql_nxt_deg)

                # 位置（米）
                p_l_cur = t_l_cur[:3, 3]
                p_l_nxt = t_l_nxt[:3, 3]
                dp_l = p_l_nxt - p_l_cur

                # 位置增量（拆成 step_sizes 小步）
                l_dx = dp_l[0] / self.cfg.step_sizes
                l_dy = dp_l[1] / self.cfg.step_sizes
                l_dz = dp_l[2] / self.cfg.step_sizes

                # 姿态增量：用欧拉角 (RPY) 的差值（弧度）
                R_l_cur = t_l_cur[:3, :3]
                R_l_nxt = t_l_nxt[:3, :3]

                rpy_l_cur = _rotmat_to_rpy_zyx(R_l_cur)  # [roll, pitch, yaw] rad
                rpy_l_nxt = _rotmat_to_rpy_zyx(R_l_nxt)

                # 直接做欧拉角差值，然后 wrap 到 [-pi, pi]
                drpy_l = rpy_l_nxt - rpy_l_cur
                drpy_l = _wrap_to_pi(drpy_l)

                l_drx = drpy_l[0] / self.cfg.step_sizes
                l_dry = drpy_l[1] / self.cfg.step_sizes
                l_drz = drpy_l[2] / self.cfg.step_sizes

                # 夹爪速度：用位置差近似（这里仍然是“关节空间”的 scalar 差分）
                g_l_cur = float(cur_state[self.cfg.left_gripper_index].item())
                g_l_nxt = float(nxt_state[self.cfg.left_gripper_index].item())
                l_g_vel = (g_l_nxt - g_l_cur) / self.cfg.step_sizes

                # ======================
                # 右臂：末端位置 + 姿态 + 夹爪
                # ======================
                qr_cur_rad = (
                    cur_state[self.cfg.right_joint_indices]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(float)
                )
                qr_nxt_rad = (
                    nxt_state[self.cfg.right_joint_indices]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(float)
                )

                qr_cur_deg = np.rad2deg(qr_cur_rad)
                qr_nxt_deg = np.rad2deg(qr_nxt_rad)

                t_r_cur = self.right_kin.forward_kinematics(qr_cur_deg)
                t_r_nxt = self.right_kin.forward_kinematics(qr_nxt_deg)

                p_r_cur = t_r_cur[:3, 3]
                p_r_nxt = t_r_nxt[:3, 3]
                dp_r = p_r_nxt - p_r_cur

                r_dx = dp_r[0] / self.cfg.step_sizes
                r_dy = dp_r[1] / self.cfg.step_sizes
                r_dz = dp_r[2] / self.cfg.step_sizes

                R_r_cur = t_r_cur[:3, :3]
                R_r_nxt = t_r_nxt[:3, :3]

                rpy_r_cur = _rotmat_to_rpy_zyx(R_r_cur)
                rpy_r_nxt = _rotmat_to_rpy_zyx(R_r_nxt)

                drpy_r = rpy_r_nxt - rpy_r_cur
                drpy_r = _wrap_to_pi(drpy_r)

                r_drx = drpy_r[0] / self.cfg.step_sizes
                r_dry = drpy_r[1] / self.cfg.step_sizes
                r_drz = drpy_r[2] / self.cfg.step_sizes

                g_r_cur = float(cur_state[self.cfg.right_gripper_index].item())
                g_r_nxt = float(nxt_state[self.cfg.right_gripper_index].item())
                r_g_vel = (g_r_nxt - g_r_cur) / self.cfg.step_sizes

        # ================
        # Clamp 一下范围
        # ================
        l_dx = float(np.clip(l_dx, -self.cfg.clamp_abs, self.cfg.clamp_abs))
        l_dy = float(np.clip(l_dy, -self.cfg.clamp_abs, self.cfg.clamp_abs))
        l_dz = float(np.clip(l_dz, -self.cfg.clamp_abs, self.cfg.clamp_abs))
        l_drx = float(np.clip(l_drx, -self.cfg.clamp_abs, self.cfg.clamp_abs))
        l_dry = float(np.clip(l_dry, -self.cfg.clamp_abs, self.cfg.clamp_abs))
        l_drz = float(np.clip(l_drz, -self.cfg.clamp_abs, self.cfg.clamp_abs))
        l_g_vel = float(np.clip(l_g_vel, -1.0, 1.0))

        r_dx = float(np.clip(r_dx, -self.cfg.clamp_abs, self.cfg.clamp_abs))
        r_dy = float(np.clip(r_dy, -self.cfg.clamp_abs, self.cfg.clamp_abs))
        r_dz = float(np.clip(r_dz, -self.cfg.clamp_abs, self.cfg.clamp_abs))
        r_drx = float(np.clip(r_drx, -self.cfg.clamp_abs, self.cfg.clamp_abs))
        r_dry = float(np.clip(r_dry, -self.cfg.clamp_abs, self.cfg.clamp_abs))
        r_drz = float(np.clip(r_drz, -self.cfg.clamp_abs, self.cfg.clamp_abs))
        r_g_vel = float(np.clip(r_g_vel, -1.0, 1.0))

        # 按 [左7维, 右7维] 拼成 14 维动作
        cur[ACTION] = torch.tensor(
            [
                l_dx, l_dy, l_dz, l_drx, l_dry, l_drz, l_g_vel,
                r_dx, r_dy, r_dz, r_drx, r_dry, r_drz, r_g_vel,
            ],
            dtype=torch.float32,
        )
        return cur


#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import collections
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torchvision.transforms import v2
from torchvision.transforms.v2 import (
    Transform,
    functional as F,  # noqa: N812
)


class RandomSubsetApply(Transform):
    """Apply a random subset of N transformations from a list of transformations.

    Args:
        transforms: list of transformations.
        p: represents the multinomial probabilities (with no replacement) used for sampling the transform.
            If the sum of the weights is not 1, they will be normalized. If ``None`` (default), all transforms
            have the same probability.
        n_subset: number of transformations to apply. If ``None``, all transforms are applied.
            Must be in [1, len(transforms)].
        random_order: apply transformations in a random order.
    """

    def __init__(
        self,
        transforms: Sequence[Callable],
        p: list[float] | None = None,
        n_subset: int | None = None,
        random_order: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(transforms, Sequence):
            raise TypeError("Argument transforms should be a sequence of callables")
        if p is None:
            p = [1] * len(transforms)
        elif len(p) != len(transforms):
            raise ValueError(
                f"Length of p doesn't match the number of transforms: {len(p)} != {len(transforms)}"
            )

        if n_subset is None:
            n_subset = len(transforms)
        elif not isinstance(n_subset, int):
            raise TypeError("n_subset should be an int or None")
        elif not (1 <= n_subset <= len(transforms)):
            raise ValueError(f"n_subset should be in the interval [1, {len(transforms)}]")

        self.transforms = transforms
        total = sum(p)
        self.p = [prob / total for prob in p]
        self.n_subset = n_subset
        self.random_order = random_order

        self.selected_transforms = None

    def forward(self, *inputs: Any) -> Any:
        needs_unpacking = len(inputs) > 1

        selected_indices = torch.multinomial(torch.tensor(self.p), self.n_subset)
        if not self.random_order:
            selected_indices = selected_indices.sort().values

        self.selected_transforms = [self.transforms[i] for i in selected_indices]

        for transform in self.selected_transforms:
            outputs = transform(*inputs)
            inputs = outputs if needs_unpacking else (outputs,)

        return outputs

    def extra_repr(self) -> str:
        return (
            f"transforms={self.transforms}, "
            f"p={self.p}, "
            f"n_subset={self.n_subset}, "
            f"random_order={self.random_order}"
        )


class SharpnessJitter(Transform):
    """Randomly change the sharpness of an image or video.

    Similar to a v2.RandomAdjustSharpness with p=1 and a sharpness_factor sampled randomly.
    While v2.RandomAdjustSharpness applies — with a given probability — a fixed sharpness_factor to an image,
    SharpnessJitter applies a random sharpness_factor each time. This is to have a more diverse set of
    augmentations as a result.

    A sharpness_factor of 0 gives a blurred image, 1 gives the original image while 2 increases the sharpness
    by a factor of 2.

    If the input is a :class:`torch.Tensor`,
    it is expected to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        sharpness: How much to jitter sharpness. sharpness_factor is chosen uniformly from
            [max(0, 1 - sharpness), 1 + sharpness] or the given
            [min, max]. Should be non negative numbers.
    """

    def __init__(self, sharpness: float | Sequence[float]) -> None:
        super().__init__()
        self.sharpness = self._check_input(sharpness)

    def _check_input(self, sharpness):
        if isinstance(sharpness, (int | float)):
            if sharpness < 0:
                raise ValueError("If sharpness is a single number, it must be non negative.")
            sharpness = [1.0 - sharpness, 1.0 + sharpness]
            sharpness[0] = max(sharpness[0], 0.0)
        elif isinstance(sharpness, collections.abc.Sequence) and len(sharpness) == 2:
            sharpness = [float(v) for v in sharpness]
        else:
            raise TypeError(f"{sharpness=} should be a single number or a sequence with length 2.")

        if not 0.0 <= sharpness[0] <= sharpness[1]:
            raise ValueError(f"sharpness values should be between (0., inf), but got {sharpness}.")

        return float(sharpness[0]), float(sharpness[1])

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        sharpness_factor = torch.empty(1).uniform_(self.sharpness[0], self.sharpness[1]).item()
        return {"sharpness_factor": sharpness_factor}

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        sharpness_factor = params["sharpness_factor"]
        return self._call_kernel(F.adjust_sharpness, inpt, sharpness_factor=sharpness_factor)


@dataclass
class ImageTransformConfig:
    """
    For each transform, the following parameters are available:
      weight: This represents the multinomial probability (with no replacement)
            used for sampling the transform. If the sum of the weights is not 1,
            they will be normalized.
      type: The name of the class used. This is either a class available under torchvision.transforms.v2 or a
            custom transform defined here.
      kwargs: Lower & upper bound respectively used for sampling the transform's parameter
            (following uniform distribution) when it's applied.
    """

    weight: float = 1.0
    type: str = "Identity"
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class ImageTransformsConfig:
    """
    These transforms are all using standard torchvision.transforms.v2
    You can find out how these transformations affect images here:
    https://pytorch.org/vision/0.18/auto_examples/transforms/plot_transforms_illustrations.html
    We use a custom RandomSubsetApply container to sample them.
    """

    # Set this flag to `true` to enable transforms during training
    enable: bool = False
    # This is the maximum number of transforms (sampled from these below) that will be applied to each frame.
    # It's an integer in the interval [1, number_of_available_transforms].
    max_num_transforms: int = 3
    # By default, transforms are applied in Torchvision's suggested order (shown below).
    # Set this to True to apply them in a random order.
    random_order: bool = False
    tfs: dict[str, ImageTransformConfig] = field(
        default_factory=lambda: {
            "brightness": ImageTransformConfig(
                weight=1.0,
                type="ColorJitter",
                kwargs={"brightness": (0.8, 1.2)},
            ),
            "contrast": ImageTransformConfig(
                weight=1.0,
                type="ColorJitter",
                kwargs={"contrast": (0.8, 1.2)},
            ),
            "saturation": ImageTransformConfig(
                weight=1.0,
                type="ColorJitter",
                kwargs={"saturation": (0.5, 1.5)},
            ),
            "hue": ImageTransformConfig(
                weight=1.0,
                type="ColorJitter",
                kwargs={"hue": (-0.05, 0.05)},
            ),
            "sharpness": ImageTransformConfig(
                weight=1.0,
                type="SharpnessJitter",
                kwargs={"sharpness": (0.5, 1.5)},
            ),
            "affine": ImageTransformConfig(
                weight=1.0,
                type="RandomAffine",
                kwargs={"degrees": (-5.0, 5.0), "translate": (0.05, 0.05)},
            ),
        }
    )


def make_transform_from_config(cfg: ImageTransformConfig):
    if cfg.type == "Identity":
        return v2.Identity(**cfg.kwargs)
    elif cfg.type == "ColorJitter":
        return v2.ColorJitter(**cfg.kwargs)
    elif cfg.type == "SharpnessJitter":
        return SharpnessJitter(**cfg.kwargs)
    elif cfg.type == "RandomAffine":
        return v2.RandomAffine(**cfg.kwargs)
    else:
        raise ValueError(f"Transform '{cfg.type}' is not valid.")


class ImageTransforms(Transform):
    """A class to compose image transforms based on configuration."""

    def __init__(self, cfg: ImageTransformsConfig) -> None:
        super().__init__()
        self._cfg = cfg

        self.weights = []
        self.transforms = {}
        for tf_name, tf_cfg in cfg.tfs.items():
            if tf_cfg.weight <= 0.0:
                continue

            self.transforms[tf_name] = make_transform_from_config(tf_cfg)
            self.weights.append(tf_cfg.weight)

        n_subset = min(len(self.transforms), cfg.max_num_transforms)
        if n_subset == 0 or not cfg.enable:
            self.tf = v2.Identity()
        else:
            self.tf = RandomSubsetApply(
                transforms=list(self.transforms.values()),
                p=self.weights,
                n_subset=n_subset,
                random_order=cfg.random_order,
            )

    def forward(self, *inputs: Any) -> Any:
        return self.tf(*inputs)
