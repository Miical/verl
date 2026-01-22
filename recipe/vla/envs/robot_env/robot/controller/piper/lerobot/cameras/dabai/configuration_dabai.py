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

from dataclasses import dataclass

from ..configs import CameraConfig, ColorMode, Cv2Rotation


@CameraConfig.register_subclass("orbbec_dabai")
@dataclass
class OrbbecDabaiCameraConfig(CameraConfig):
    """
    Configuration class for Orbbec Dabai cameras.

    This class provides specialized configuration options for Orbbec Dabai series RGB-D cameras
    (e.g., Dabai, Dabai DCW, Dabai Dual Pro). It supports device selection via serial number,
    resolution, frame rate, depth sensing, and image post-processing.

    Example configurations:
    ```python
    # Basic usage with serial number
    OrbbecDabaiCameraConfig(serial_number_or_name="ABC123XYZ", fps=30, width=640, height=480)

    # With depth enabled
    OrbbecDabaiCameraConfig(
        serial_number_or_name="ABC123XYZ",
        fps=30,
        width=640,
        height=480,
        use_depth=True
    )

    # With BGR output and 90° clockwise rotation (useful for upright mounting)
    OrbbecDabaiCameraConfig(
        serial_number_or_name="ABC123XYZ",
        fps=30,
        width=640,
        height=480,
        color_mode=ColorMode.BGR,
        rotation=Cv2Rotation.ROTATE_90,
        use_depth=True
    )
    ```

    Attributes:
        serial_number_or_name: Unique serial number or device name to identify the camera.
            Using serial number is recommended for stability.
        fps: Requested frames per second. Must be supported by the camera at the given resolution.
        width: Requested image width in pixels.
        height: Requested image height in pixels.
        color_mode: Output color format: RGB (default) or BGR. Choose BGR if integrating with OpenCV
            pipelines that expect BGR input.
        use_depth: Whether to enable the depth stream. If True, `read_depth()` will be available.
            Depth data is aligned to the color stream when possible.
        rotation: Optional 90°/180°/270° rotation applied via OpenCV (software rotation).
        warmup_s: Time (in seconds) to discard initial frames during `connect()` to allow
            sensor and auto-exposure to stabilize.

    Note:
        - You must specify either `serial_number_or_name`, and it should uniquely identify one camera.
        - If enabling depth, the depth stream will use the same FPS as the color stream.
        - The actual resolution and FPS may be adjusted by the camera to the nearest supported mode.
        - For `fps`, `width`, and `height`, either all three must be set, or none — partial specification
          is not allowed.
        - Rotation is applied *after* capture and is performed in software using OpenCV.
    """

    serial_number_or_name: str
    color_mode: ColorMode = ColorMode.RGB
    use_depth: bool = False
    rotation: Cv2Rotation = Cv2Rotation.NO_ROTATION
    warmup_s: int = 2  # Slightly longer warmup for Orbbec devices

    def __post_init__(self):
        # Validate color mode
        if self.color_mode not in (ColorMode.RGB, ColorMode.BGR):
            raise ValueError(
                f"`color_mode` must be '{ColorMode.RGB.value}' or '{ColorMode.BGR.value}', "
                f"but got '{self.color_mode}'."
            )

        # Validate rotation
        valid_rotations = (
            Cv2Rotation.NO_ROTATION,
            Cv2Rotation.ROTATE_90,
            Cv2Rotation.ROTATE_180,
            Cv2Rotation.ROTATE_270,
        )
        if self.rotation not in valid_rotations:
            raise ValueError(
                f"`rotation` must be one of {valid_rotations}, but got '{self.rotation}'."
            )

        # Ensure fps, width, height are all set or all None
        values = (self.fps, self.width, self.height)
        if any(v is not None for v in values) and any(v is None for v in values):
            raise ValueError(
                "For `fps`, `width`, and `height`, either all must be specified, or none."
            )

        # Optional: Validate serial number non-empty
        if not self.serial_number_or_name or not self.serial_number_or_name.strip():
            raise ValueError("`serial_number_or_name` must be a non-empty string.")