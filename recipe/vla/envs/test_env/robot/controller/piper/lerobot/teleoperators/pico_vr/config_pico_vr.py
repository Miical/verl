from dataclasses import dataclass

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("pico_vr")
@dataclass
class PicoVrTeleopConfig(TeleoperatorConfig):
    """
    Configuration for Pico VR dual-controller teleoperation.
    """

    # Pose sources
    left_pose_source: str = "left_controller"
    right_pose_source: str = "right_controller"

    # Translation/rotation scaling
    translation_scale: float = 1
    rotation_scale: float = 1.0
    # Optional per-axis scaling for rotation deltas (in axis-angle, x/y/z).
    # Used mainly to reduce yaw (z-axis) sensitivity during teleop.
    rotation_yaw_gain: float = 1.0
    translation_clip: float = 0.2
    rotation_clip: float = 0.5
    translation_deadband: float = 1e-4
    rotation_deadband: float = 5e-3

    # Gripper inputs
    left_gripper_input: str = "left_trigger"
    right_gripper_input: str = "right_trigger"
    gripper_open: float = 0.07
    gripper_close: float = 0.0
    gripper_delta_gain: float = 1.0
    gripper_deadband: float = 1e-3

    # Buttons for events
    reset_reference_button: str = "A"  # 重置参考位姿
    intervention_reset_button: str = "X"  # 切换干预状态（按下开启/关闭干预）
    terminate_button: str = "right_axis_click"  # 手柄摇杆按钮，终止回合
    success_button: str = "Y"  # 标记成功
    rerecord_button: str = "B"  # 重录回合

    # Intervention logic (已废弃：现在使用手动按钮切换，不再自动检测)
    intervention_threshold: float = 1e-3

    # ZeroMQ configuration for subscribing to teleop signals
    zmq_port: int = 5555  # ZeroMQ publisher port
    zmq_host: str = "localhost"  # ZeroMQ publisher host

