import logging

# 关闭 python-can 的 DEBUG 日志
logging.getLogger("can.interfaces.socketcan").setLevel(logging.WARNING)
import numpy as np
from typing import Dict, Literal
from traceback import format_exc
# Piper SDK导入 - 使用V2版本接口
from piper_sdk import C_PiperInterface_V2
from ..motors_bus import Motor, MotorCalibration, MotorsBus, NameOrID, Value, get_address


class PiperMotorsBus:
    """
    Piper机械臂电机总线独立实现 - 使用V2接口

    提供类似MotorsBus的接口，但直接使用Piper SDK V2的CAN总线通信
    适配固件版本 V1.5-2 及以上（V1.6-5完全支持）
    """

    def __init__(
        self, can_name: str,
        baud_rate: int = 1000000,
        motor_prefix: Literal['left', 'right'] = 'left'
    ):
        """
        初始化Piper电机总线

        Args:
            can_name: CAN接口名称 (如 "can0", "can_left", "can_right")
            baud_rate: CAN波特率 (通常为1000000)
            motor_prefix: 用于内部区分左右臂，但对外接口隐藏前缀
        """
        self.can_name = can_name
        self.baud_rate = baud_rate
        self.motor_prefix = motor_prefix  # !!!! 保存 prefix 供后续使用
        self.interface = None
        self.is_connected = False

        # !!!! 保持对外接口干净：不加前缀
        # self.motors = [
        #     "joint_1",  # 底座旋转
        #     "joint_2",  # 肩部俯仰
        #     "joint_3",  # 肘部俯仰
        #     "joint_4",  # 腕部旋转
        #     "joint_5",  # 腕部俯仰
        #     "joint_6",  # 腕部滚转
        #     "gripper"   # 夹爪
        # ]
        self.motors = [
            "joint1",  # 底座旋转
            "joint2",  # 肩部俯仰
            "joint3",  # 肘部俯仰
            "joint4",  # 腕部旋转
            "joint5",  # 腕部俯仰
            "joint6",  # 腕部滚转
            "gripper"   # 夹爪
        ]

        # !!!! 新增：逻辑名 -> 物理名（带前缀）的映射
        self._motor_name_map = {
            motor: f"{self.motor_prefix}_{motor}" for motor in self.motors
        }

        # 关节限位 (弧度) - 基于Piper官方文档
        raw_limits = {
            "joint1": (-2.6179, 2.6179),      # [-150°, 150°]
            "joint2": (0, 3.14),              # [0°, 180°]
            "joint3": (-2.967, 0),            # [-170°, 0°]
            "joint4": (-1.745, 1.745),        # [-100°, 100°]
            "joint5": (-1.22, 1.22),          # [-70°, 70°]
            "joint6": (-2.09439, 2.09439),    # [-120°, 120°]
            "gripper": (0.0, 0.07)             # 夹爪开合度 [0-70mm]
        }
        # !!!! 限位使用带前缀的物理名
        self.joint_limits = {
            f"{self.motor_prefix}_{k}": v for k, v in raw_limits.items()
        }

        # 复位位置
        if 'left' in self.can_name:
            self.arm_reset_rad_pos = [
                -0.00133514404296875, 0.00209808349609375, 0.01583099365234375,
                -0.032616615295410156, -0.00286102294921875, 0.00095367431640625, 0.06
            ]
        elif 'right' in self.can_name:
            self.arm_reset_rad_pos = [
                -0.00133514404296875, 0.00438690185546875, 0.034523963928222656,
                -0.053597450256347656, -0.00476837158203125, -0.00209808349609375, 0.06
            ]

        # !!!! 状态缓存使用带前缀的物理名
        self._current_positions = {self._motor_name_map[m]: 0.0 for m in self.motors}
        self._current_velocities = {self._motor_name_map[m]: 0.0 for m in self.motors}
        self._current_currents = {self._motor_name_map[m]: 0.0 for m in self.motors}

        logging.info(f"Initialized PiperMotorsBus V2 with {len(self.motors)} motors")
        logging.info(f"CAN interface: {can_name}, Baud rate: {baud_rate}, Prefix: {motor_prefix}")

    def connect(
        self,
        start_sdk_joint_limit: bool = True,
        start_sdk_gripper_limit: bool = True,
        move_spd_rate_ctrl: int = 50
    ) -> bool:
        """连接到Piper机械臂"""
        if C_PiperInterface_V2 is None:
            logging.error("Piper SDK V2 not available")
            return False

        try:
            self.interface = C_PiperInterface_V2(
                can_name=self.can_name,
                judge_flag=True,
                can_auto_init=True,
                dh_is_offset=0x01,
                start_sdk_joint_limit=start_sdk_joint_limit,
                start_sdk_gripper_limit=start_sdk_gripper_limit,
                logger_level=logging.DEBUG,
                log_to_file=False,
                log_file_path=None
            )

            result = self.interface.ConnectPort(
                can_init=True,
                piper_init=True,
                start_thread=True
            )

            if self.interface.get_connect_status():
                self.is_connected = True
                logging.info(f"Successfully connected to Piper robot on {self.can_name}")
                self._initialize_robot(move_spd_rate_ctrl=move_spd_rate_ctrl)
                return True
            else:
                logging.error(f"Failed to connect to Piper robot on {self.can_name}")
                return False

        except Exception as e:
            logging.error(f"Exception during Piper connection: {format_exc()}")
            return False

    def disconnect(self, reset_pos: bool = False, disable_torque: bool = True):
        """断开连接"""
        if self.interface and self.is_connected:
            try:
                if reset_pos:
                    self.reset_pos()
                if disable_torque:
                    self.sync_write("Torque_Enable", 0)
                self.interface.DisconnectPort()
                self.is_connected = False
                logging.info("Disconnected from Piper robot")
            except Exception as e:
                logging.error(f"Error disconnecting from Piper robot: {format_exc()}")

    def _initialize_robot(self, move_spd_rate_ctrl: int = 50):
        """初始化机器人状态"""
        try:
            self.interface.EnableArm(motor_num=7, enable_flag=0x02)
            self.interface.ModeCtrl(
                ctrl_mode=0x01,
                move_mode=0x01,
                move_spd_rate_ctrl=move_spd_rate_ctrl,
                is_mit_mode=0x00
            )
            self.interface.GripperTeachingPendantParamConfig(
                teaching_range_per=100,
                max_range_config=70,
                teaching_friction=1
            )
            self.interface.GripperCtrl(
                gripper_angle=0,
                gripper_effort=1000,
                gripper_code=0x01,
                set_zero=0x00
            )
            self.reset_pos()
            self._update_positions()
            logging.info("Piper robot V2 initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize Piper robot: {format_exc()}")

    def reset_pos(self):
        if self.arm_reset_rad_pos:
            # !!!! 对外使用逻辑名，内部自动映射
            self.sync_write("Goal_Position", {
                motor: pos for motor, pos in zip(self.motors, self.arm_reset_rad_pos)
            })
            logging.info(f"Robot {self.can_name} reset position")
        else:
            logging.warning(f"No reset position defined for this arm {self.can_name}")

    def _update_positions(self):
        """更新关节位置（内部使用带前缀的物理名）"""
        try:
            joint_msgs = self.interface.GetArmJointMsgs()
            if joint_msgs is not None:
                joint_values = [
                    joint_msgs.joint_state.joint_1,
                    joint_msgs.joint_state.joint_2,
                    joint_msgs.joint_state.joint_3,
                    joint_msgs.joint_state.joint_4,
                    joint_msgs.joint_state.joint_5,
                    joint_msgs.joint_state.joint_6,
                ]
                # !!!! 使用逻辑名遍历，但存入物理名
                for i, motor in enumerate(self.motors[:-1]):
                    phys_name = self._motor_name_map[motor]
                    angle_deg = joint_values[i] / 1000.0
                    angle_rad = np.deg2rad(angle_deg)
                    self._current_positions[phys_name] = angle_rad

            gripper_msgs = self.interface.GetArmGripperMsgs()
            if gripper_msgs is not None:
                gripper_pos_m = gripper_msgs.gripper_state.grippers_angle / 1000000.0
                phys_gripper = self._motor_name_map["gripper"]
                self._current_positions[phys_gripper] = gripper_pos_m

        except Exception as e:
            logging.error(f"Failed to update positions: {format_exc()}")

    def sync_read(self, parameter: str) -> Dict[str, float]:
        """
        同步读取参数，返回逻辑名（无前缀）
        """
        try:
            if parameter == "Present_Position":
                self._update_positions()
                # !!!! 将内部物理名映射回用户友好的逻辑名
                return {
                    motor: self._current_positions[self._motor_name_map[motor]]
                    for motor in self.motors
                }
            else:
                logging.warning(f"Unsupported parameter: {parameter}")
                return {}
        except Exception as e:
            logging.error(f"Failed to sync_read {parameter}: {format_exc()}")
            return {}

    def sync_write(self, parameter: str, values: Dict[str, float] | float):
        """
        同步写入参数，接收逻辑名（无前缀），内部转为物理名
        """
        try:
            if parameter == "Goal_Position":
                if isinstance(values, dict):
                    # !!!! 将用户输入的逻辑名映射为内部物理名
                    internal_values = {
                        self._motor_name_map[motor]: val
                        for motor, val in values.items()
                        if motor in self._motor_name_map
                    }

                    joint_angles = []
                    # !!!! 遍历逻辑名，但使用物理名查限位和状态
                    for motor in self.motors[:-1]:
                        phys_name = self._motor_name_map[motor]
                        if phys_name in internal_values:
                            angle_rad = np.clip(
                                internal_values[phys_name],
                                self.joint_limits[phys_name][0],
                                self.joint_limits[phys_name][1]
                            )
                            angle_millideg = int(np.rad2deg(angle_rad) * 1000)
                            joint_angles.append(angle_millideg)
                        else:
                            current_rad = self._current_positions[phys_name]
                            joint_angles.append(int(np.rad2deg(current_rad) * 1000))

                    self.interface.JointCtrl(*joint_angles)

                    # 夹爪处理
                    gripper_logical = "gripper"
                    gripper_phys = self._motor_name_map[gripper_logical]
                    if gripper_phys in internal_values:
                        gripper_pos_m = np.clip(
                            internal_values[gripper_phys],
                            self.joint_limits[gripper_phys][0],
                            self.joint_limits[gripper_phys][1]
                        )
                        gripper_angle_micro = int(gripper_pos_m * 1e6)
                        self.interface.GripperCtrl(
                            gripper_angle=gripper_angle_micro,
                            gripper_effort=1000,
                            gripper_code=0x01,
                            set_zero=0x00
                        )

            elif parameter == "Torque_Enable":
                enable = int(values) if not isinstance(values, dict) else 1
                if enable:
                    self.interface.EnableArm(motor_num=7, enable_flag=0x02)
                else:
                    self.interface.DisableArm(motor_num=7, enable_flag=0x01)

            else:
                logging.warning(f"Unsupported write parameter: {parameter}")

        except Exception as e:
            logging.error(f"Failed to sync_write {parameter}: {format_exc()}")

    def write(self, parameter: str, motor: str, value: float):
        """
        写入单个电机参数，接收逻辑名（如 "joint_1"）
        """
        self.sync_write(parameter, {motor: value})