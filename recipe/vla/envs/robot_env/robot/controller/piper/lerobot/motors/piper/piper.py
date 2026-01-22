import logging
import numpy as np
from typing import Dict, Literal, Optional
from traceback import format_exc
# Piper SDK导入 - 使用V2版本接口
from piper_sdk import C_PiperInterface_V2
import pdb
import time

class PiperMotorsBus:
    """
    Piper机械臂电机总线独立实现 - 使用V2接口
    提供关节空间控制 + 末端位姿控制（含夹爪），单位统一：
      - 关节角度: rad
      - 末端位置: m
      - 末端姿态: rad
      - 夹爪开合: m
    """

    def __init__(
        self, can_name: str,
        baud_rate: int = 1000000,
        motor_prefix: Literal['left', 'right'] = 'left'
    ):
        self.can_name = can_name
        self.baud_rate = baud_rate
        self.interface = None
        self.is_connected = False
        
        # 关节命名（6关节+夹爪）
        self.motors = [f"{motor_prefix}_{m}" for m in [
            "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"
        ]]

        # 关节限位（rad / m）
        self.joint_limits = {
            "joint_1": (-2.6179, 2.6179),      # [-150°, 150°]
            "joint_2": (0, 3.14),              # [0°, 180°]
            "joint_3": (-2.967, 0),            # [-170°, 0°]
            "joint_4": (-1.745, 1.745),        # [-100°, 100°]
            "joint_5": (-1.22, 1.22),          # [-70°, 70°]
            "joint_6": (-2.09439, 2.09439),    # [-120°, 120°]
            "gripper": (0.0, 0.07)             # 0~70mm -> m
        }
        self.joint_limits = {f"{motor_prefix}_{k}": v for k, v in self.joint_limits.items()}

        # 复位位姿（6关节rad + gripper m）
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
        else:
            self.arm_reset_rad_pos = [0, 0, 0, 0, 0, 0, 0.06]

        # 状态缓存
        self._current_positions = {motor: 0.0 for motor in self.motors}  # rad/m
        self._current_velocities = {motor: 0.0 for motor in self.motors}
        self._current_currents = {motor: 0.0 for motor in self.motors}

        # 末端位姿缓存（含夹爪）
        self.end_pose_keys = [
            f"{motor_prefix}_end_x", f"{motor_prefix}_end_y", f"{motor_prefix}_end_z",
            f"{motor_prefix}_end_rx", f"{motor_prefix}_end_ry", f"{motor_prefix}_end_rz",
            f"{motor_prefix}_end_gripper"  # 新增：末端模式中的夹爪
        ]
        self._current_end_pose = {k: 0.0 for k in self.end_pose_keys}

        logging.info(f"Initialized PiperMotorsBus V2 with {len(self.motors)} motors")
        logging.info(f"CAN interface: {can_name}, Baud rate: {baud_rate}")

    # ======== 工具换算函数 ========
    @staticmethod
    def _m_to_sdk_len_units(x_m: float) -> int:
        """m -> 0.001 mm（int）"""
        return int(round(x_m * 1_000_000))

    @staticmethod
    def _sdk_len_units_to_m(x_units: int) -> float:
        """0.001 mm（int）-> m"""
        return x_units / 1_000_000.0

    @staticmethod
    def _rad_to_sdk_angle_units(rad: float) -> int:
        """rad -> 0.001 degree（int）"""
        return int(round(np.rad2deg(rad) * 1000.0))

    @staticmethod
    def _sdk_angle_units_to_rad(units: int) -> float:
        """0.001 degree（int）-> rad"""
        return np.deg2rad(units / 1000.0)

    # ======== 连接/初始化 ========
    def connect(
        self,
        start_sdk_joint_limit: bool = True,
        start_sdk_gripper_limit: bool = True,
        move_spd_rate_ctrl: int = 50
    ) -> bool:
        if C_PiperInterface_V2 is None:
            logging.error("Piper SDK V2 not available")
            return False
        
        # ===== 诊断信息：检查 CAN 接口状态 =====
        import os
        import subprocess
        logging.info(f"[CAN Diagnostics] Attempting to connect to CAN: {self.can_name}")
        logging.info(f"[CAN Diagnostics] Current user: {os.getenv('USER', 'unknown')}, UID: {os.getuid()}, GID: {os.getgid()}")
        logging.info(f"[CAN Diagnostics] Current working directory: {os.getcwd()}")
        logging.info(f"[CAN Diagnostics] Process PID: {os.getpid()}, PPID: {os.getppid()}")
        
        # 检查网络接口
        try:
            result = subprocess.run(['ip', 'link', 'show'], capture_output=True, text=True, timeout=5)
            can_interfaces = [line for line in result.stdout.split('\n') if 'can' in line.lower()]
            logging.info(f"[CAN Diagnostics] Available CAN interfaces: {can_interfaces}")
            if not can_interfaces:
                logging.warning(f"[CAN Diagnostics] No CAN interfaces found! Full output:\n{result.stdout}")
        except Exception as e:
            logging.error(f"[CAN Diagnostics] Failed to check network interfaces: {e}")
        
        # 检查 /sys/class/net/ 下的设备
        try:
            net_devices = os.listdir('/sys/class/net/')
            can_devices = [d for d in net_devices if 'can' in d.lower()]
            logging.info(f"[CAN Diagnostics] Devices in /sys/class/net/: {net_devices}")
            logging.info(f"[CAN Diagnostics] CAN devices: {can_devices}")
        except Exception as e:
            logging.error(f"[CAN Diagnostics] Failed to list /sys/class/net/: {e}")
        
        # 检查 CAN socket 文件是否存在
        can_socket_path = f"/sys/class/net/{self.can_name}"
        logging.info(f"[CAN Diagnostics] Checking CAN socket path: {can_socket_path}, exists: {os.path.exists(can_socket_path)}")
        # ===== 诊断信息结束 =====
        
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
            # pdb.set_trace()
            _ = self.interface.ConnectPort(can_init=True, piper_init=True, start_thread=True)
            if self.interface.get_connect_status():
                self.is_connected = True
                logging.info(f"Connected on {self.can_name}")
                # pdb.set_trace()
                self._initialize_robot(move_spd_rate_ctrl=move_spd_rate_ctrl)
                return True
            logging.error(f"Failed to connect on {self.can_name}")
            return False
        except Exception:
            logging.error(f"Exception during connect: {format_exc()}")
            return False

    def disconnect(self, reset_pos: bool = False, disable_torque: bool = True):
        if self.interface and self.is_connected:
            try:
                if reset_pos:
                    self.reset_pos()
                if disable_torque:
                    self.sync_write("Torque_Enable", 0)
                self.interface.DisconnectPort()
                self.is_connected = False
                logging.info("Disconnected from Piper robot")
            except Exception:
                logging.error(f"Error disconnecting: {format_exc()}")

    def _initialize_robot(self, move_spd_rate_ctrl: int = 50):
        try:
            self.interface.EnableArm(motor_num=7, enable_flag=0x02)
            self.interface.ModeCtrl(
                ctrl_mode=0x01,     # CAN command
                move_mode=0x00,     # MOVE P
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
            # self.reset_pos()
            self._update_positions()
            self._update_end_pose()
            logging.info("Robot initialized")
        except Exception:
            logging.error(f"Failed to initialize: {format_exc()}")

    def _set_move_mode(self, move_mode: int = 0x00, move_spd_rate_ctrl: Optional[int] = None):
        """0x00:P, 0x01:J, 0x02:L, 0x03:C"""
        try:
            # ctrl_mode=0x01: 使用 CAN 指令控制（和 _initialize_robot 保持一致）
            kwargs = dict(ctrl_mode=0x01, move_mode=move_mode, is_mit_mode=0x00)
            if move_spd_rate_ctrl is not None:
                kwargs["move_spd_rate_ctrl"] = move_spd_rate_ctrl
            self.interface.ModeCtrl(**kwargs)
        except Exception:
            logging.error(f"Failed to set move mode: {format_exc()}")


    # ======== 状态更新 ========
    def _update_positions(self):
        """更新各关节(rad)与夹爪(m)"""
        try:
            jm = self.interface.GetArmJointMsgs()
            if jm is not None:
                vals = [
                    jm.joint_state.joint_1,
                    jm.joint_state.joint_2,
                    jm.joint_state.joint_3,
                    jm.joint_state.joint_4,
                    jm.joint_state.joint_5,
                    jm.joint_state.joint_6,
                ]
                for i, v in enumerate(vals):
                    self._current_positions[self.motors[i]] = np.deg2rad(v / 1000.0)
            gm = self.interface.GetArmGripperMsgs()
            if gm is not None:
                self._current_positions[self.motors[-1]] = gm.gripper_state.grippers_angle / 1_000_000.0
        except Exception:
            logging.error(f"Failed to update positions: {format_exc()}")

    def _update_end_pose(self):
        """更新末端位姿(m/rad) + 夹爪(m)"""
        try:
            epm = self.interface.GetArmEndPoseMsgs()
            if epm is not None:
                ep = epm.end_pose
                x_m = self._sdk_len_units_to_m(ep.X_axis)
                y_m = self._sdk_len_units_to_m(ep.Y_axis)
                z_m = self._sdk_len_units_to_m(ep.Z_axis)
                rx = self._sdk_angle_units_to_rad(ep.RX_axis)
                ry = self._sdk_angle_units_to_rad(ep.RY_axis)
                rz = self._sdk_angle_units_to_rad(ep.RZ_axis)
                kx, ky, kz, krx, kry, krz, kgr = self.end_pose_keys
                self._current_end_pose[kx] = x_m
                self._current_end_pose[ky] = y_m
                self._current_end_pose[kz] = z_m
                self._current_end_pose[krx] = rx
                self._current_end_pose[kry] = ry
                self._current_end_pose[krz] = rz
            # 同步夹爪
            gm = self.interface.GetArmGripperMsgs()
            if gm is not None:
                kgr = self.end_pose_keys[-1]
                self._current_end_pose[kgr] = gm.gripper_state.grippers_angle / 1_000_000.0
            else:
                # fallback: 用关节缓存中的夹爪
                self._current_end_pose[self.end_pose_keys[-1]] = self._current_positions[self.motors[-1]]
        except Exception:
            logging.error(f"Failed to update end pose: {format_exc()}")

    # ======== 公共读取接口 ========
    def sync_read(self, parameter: str) -> Dict[str, float]:
        """
        - "Present_Position": 返回各关节(rad)和夹爪(m)
        - "End_Pose": 返回末端位姿(m, rad) + 夹爪(m)
        """
        try:
            if parameter == "Present_Position":
                self._update_positions()
                return self._current_positions.copy()
            elif parameter == "End_Pose":
                self._update_end_pose()
                return self._current_end_pose.copy()
            else:
                logging.warning(f"Unsupported parameter: {parameter}")
                return {}
        except Exception:
            logging.error(f"Failed to sync_read {parameter}: {format_exc()}")
            return {}

    def get_end_pose(self) -> Dict[str, float]:
        """同 sync_read('End_Pose')"""
        if not (self.interface and self.is_connected):
            logging.warning("get_end_pose called while not connected.")
            return self._current_end_pose.copy()
        self._update_end_pose()
        return self._current_end_pose.copy()

    def reset_pos(self, wait_s: float = 2.0, move_spd_rate_ctrl: int = 50):
        """
        复位位姿（6 关节 + 夹爪）：
          1. 临时切到关节控制模式 (MOVE J, 0x01)
          2. 下发关节空间绝对目标位姿 self.arm_reset_rad_pos
          3. 等待一小段时间
          4. 切回末端位姿控制模式 (MOVE P, 0x00)
        """
        if not (self.interface and self.is_connected):
            logging.warning(f"reset_pos called while not connected on {self.can_name}.")
            return

        if not self.arm_reset_rad_pos:
            logging.warning(f"No reset position defined for {self.can_name}")
            return

        try:
            
            # 1. 切到关节模式 MOVE J（CAN 控制）
            logging.info(f"[{self.can_name}] Switch to joint mode (MOVE J) for reset.")
            self._set_move_mode(move_mode=0x01, move_spd_rate_ctrl=move_spd_rate_ctrl)

            # 2. 下发关节复位目标（绝对位置，单位 rad/m）
            reset_values = {
                motor: pos for motor, pos in zip(self.motors, self.arm_reset_rad_pos)
            }
            logging.info(f"[{self.can_name}] Sending joint reset position: {reset_values}")
            self.sync_write("Goal_Position", reset_values)

            # 3. 粗略等待机械臂到位
            if wait_s > 0:
                time.sleep(wait_s)
            
        except Exception:
            logging.error(f"Failed to reset position on {self.can_name}: {format_exc()}")
        finally:
            # 4. 切回末端位姿控制模式 MOVE P（CAN 控制）
            try:
                logging.info(f"[{self.can_name}] Switch back to end-pose mode (MOVE P).")
                # pdb.set_trace()
                self._set_move_mode(move_mode=0x00, move_spd_rate_ctrl=move_spd_rate_ctrl)
            except Exception:
                logging.error(f"Failed to switch back to end-pose mode on {self.can_name}: {format_exc()}")


    def end_pose_ctrl(
        self,
        x_m: float, y_m: float, z_m: float,
        rx_rad: float, ry_rad: float, rz_rad: float,
        gripper_m: Optional[float] = None,
        move_mode_linear: bool = False,
        move_spd_rate_ctrl: Optional[int] = None
    ):
        """
        末端位姿控制（EndPoseCtrl + 可选夹爪 GripperCtrl）
        - 位姿：m / rad
        - gripper_m：若不为 None，则一并控制夹爪（m）
        """
        # print(f"[{self.can_name}] EndPoseCtrl: X:{x_m}, Y:{y_m}, Z:{z_m}, RX:{rx_rad}, RY:{ry_rad}, RZ:{rz_rad}, gripper:{gripper_m}")

        # pdb.set_trace()
        if not (self.interface and self.is_connected):
            logging.warning("end_pose_ctrl called while not connected.")
            return
        try:
            if move_mode_linear:
                self._set_move_mode(0x02, move_spd_rate_ctrl)

            X = self._m_to_sdk_len_units(x_m)
            Y = self._m_to_sdk_len_units(y_m)
            Z = self._m_to_sdk_len_units(z_m)
            RX = self._rad_to_sdk_angle_units(rx_rad)
            RY = self._rad_to_sdk_angle_units(ry_rad)
            RZ = self._rad_to_sdk_angle_units(rz_rad)
            self.interface.EndPoseCtrl(X=X, Y=Y, Z=Z, RX=RX, RY=RY, RZ=RZ)

            # 夹爪（可选）
            if gripper_m is not None:
                low, high = self.joint_limits[self.motors[-1]]
                g = float(np.clip(gripper_m, low, high))
                self.interface.GripperCtrl(
                    gripper_angle=self._m_to_sdk_len_units(g),
                    gripper_effort=1000,
                    gripper_code=0x01,
                    set_zero=0x00
                )
            # print(f"[{self.can_name}] [Arm status]: ")
            # print(
            #     f"|{'commuciation_err':<16}|"
            #     f"{str(self.interface.GetArmStatus().arm_status.err_status.communication_status_joint_1):^15}"
            #     f"{str(self.interface.GetArmStatus().arm_status.err_status.communication_status_joint_2):^15}"
            #     f"{str(self.interface.GetArmStatus().arm_status.err_status.communication_status_joint_3):^15}"
            #     f"{str(self.interface.GetArmStatus().arm_status.err_status.communication_status_joint_4):^15}"
            #     f"{str(self.interface.GetArmStatus().arm_status.err_status.communication_status_joint_5):^15}"
            #     f"{str(self.interface.GetArmStatus().arm_status.err_status.communication_status_joint_6):^15}|\n"
            #     f"|{'over_angle':<16}|"
            #     f"{str(self.interface.GetArmStatus().arm_status.err_status.joint_1_angle_limit):^15}"
            #     f"{str(self.interface.GetArmStatus().arm_status.err_status.joint_2_angle_limit):^15}"
            #     f"{str(self.interface.GetArmStatus().arm_status.err_status.joint_3_angle_limit):^15}"
            #     f"{str(self.interface.GetArmStatus().arm_status.err_status.joint_4_angle_limit):^15}"
            #     f"{str(self.interface.GetArmStatus().arm_status.err_status.joint_5_angle_limit):^15}"
            #     f"{str(self.interface.GetArmStatus().arm_status.err_status.joint_6_angle_limit):^15}|\n"
            # )
        except Exception:
            logging.error(f"Failed to control end pose: {format_exc()}")

    def sync_write(self, parameter: str, values: Dict[str, float] | float):
        """
        - "Goal_Position": 关节空间控制（rad/m）
        - "Torque_Enable": 使能
        - "End_Pose": 末端位姿控制（m/rad/m）
            可包含 keys:
              '<prefix>_end_x','<prefix>_end_y','<prefix>_end_z',
              '<prefix>_end_rx','<prefix>_end_ry','<prefix>_end_rz',
              '<prefix>_end_gripper'
        """
        try:
            #print(f"[{self.can_name}] sync_write ")
            if parameter == "Goal_Position":
                if not isinstance(values, dict):
                    logging.warning("Goal_Position expects a dict.")
                    return
                joint_angles = []
                # 6关节
                for motor in self.motors[:-1]:
                    if motor in values:
                        ang = np.clip(values[motor], *self.joint_limits[motor])
                        joint_angles.append(int(np.rad2deg(ang) * 1000))
                    else:
                        cur = self._current_positions[motor]
                        joint_angles.append(int(np.rad2deg(cur) * 1000))
                self.interface.JointCtrl(
                    joint_1=joint_angles[0],
                    joint_2=joint_angles[1],
                    joint_3=joint_angles[2],
                    joint_4=joint_angles[3],
                    joint_5=joint_angles[4],
                    joint_6=joint_angles[5]
                )
                # 夹爪
                if self.motors[-1] in values:
                    g = np.clip(values[self.motors[-1]], *self.joint_limits[self.motors[-1]])
                    self.interface.GripperCtrl(
                        gripper_angle=self._m_to_sdk_len_units(float(g)),
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

            elif parameter == "End_Pose":
                if not isinstance(values, dict):
                    logging.warning("End_Pose expects a dict.")
                    return

                # 获取当前末端位姿（含夹爪）作为默认
                # self._update_end_pose()
                # pdb.set_trace()
                # pose = self._current_end_pose.copy()
                # allow = set(self.end_pose_keys)
                # pose.update({k: v for k, v in values.items() if k in allow})
                # time.sleep(3.0)
                #print("values:", values )
                pose=values
                kx, ky, kz, krx, kry, krz, kgr = self.end_pose_keys

                # 限幅夹爪
                g_val = float(pose[kgr])
                g_low, g_high = self.joint_limits[self.motors[-1]]
                g_val = float(np.clip(g_val, g_low, g_high))

                # 下发：EndPoseCtrl + 可选夹爪（总是一起下，便于原子化）
                self.end_pose_ctrl(
                    x_m=float(pose[kx]),
                    y_m=float(pose[ky]),
                    z_m=float(pose[kz]),
                    rx_rad=float(pose[krx]),
                    ry_rad=float(pose[kry]),
                    rz_rad=float(pose[krz]),
                    gripper_m=g_val,
                    move_mode_linear=False,
                    move_spd_rate_ctrl=None
                )

            else:
                logging.warning(f"Unsupported write parameter: {parameter}")

        except Exception:
            logging.error(f"Failed to sync_write {parameter}: {format_exc()}")

    def write(self, parameter: str, motor: str, value: float):
        self.sync_write(parameter, {motor: value})
