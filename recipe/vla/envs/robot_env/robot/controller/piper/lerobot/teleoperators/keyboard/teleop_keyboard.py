#!/usr/bin/env python

# Copyright 2024 ...
import logging
import os
import sys
import time
from queue import Queue
from typing import Any
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..teleoperator import Teleoperator
from .configuration_keyboard import KeyboardEndEffectorTeleopConfig, KeyboardTeleopConfig

PYNPUT_AVAILABLE = True
try:
    if ("DISPLAY" not in os.environ) and ("linux" in sys.platform):
        logging.info("No DISPLAY set. Skipping pynput import.")
        raise ImportError("pynput blocked intentionally due to no display.")
    from pynput import keyboard
except ImportError:
    keyboard = None
    PYNPUT_AVAILABLE = False
except Exception as e:
    keyboard = None
    PYNPUT_AVAILABLE = False
    logging.info(f"Could not import pynput: {e}")


class KeyboardTeleop(Teleoperator):
    """
    单臂关节空间键盘遥操作器（只控一侧，默认 left）
    键位:
      增大: q w e a s d  -> 6 个关节 +
      减小: u i o j k l  -> 6 个关节 -
      夹爪: r(+), p(-)
    输出:
      扁平 { "{motor}.pos": float }，与 SO100Leader/SO100Follower 的 I/O 结构一致
      例如 "left_joint_1.pos": 0.12
    """

    config_class = KeyboardTeleopConfig
    name = "keyboard"

    def __init__(self, config: KeyboardTeleopConfig):
        super().__init__(config)
        self.config = config

        # 组合电机完整名（如 left_joint_1 … left_gripper）
        self.motors: list[str] = [m for m in config.motor_names]
        assert len(self.motors) >= 7, "KeyboardTeleop 期望 motor_names 至少包含 6 关节 + 1 夹爪"

        self.joint_motors = self.motors[:6]
        self.gripper_motor = self.motors[6]

        # 内部“软 leader”位置缓存
        self._pos: dict[str, float] = {m: 0.0 for m in self.motors}

        # 键盘监听
        self.event_queue: Queue = Queue()
        self.current_pressed: dict[Any, bool] = {}
        self.listener = None
        self.logs = {}

        # 键位映射：增大/减小
        self._inc_keys = ["q", "w", "e", "a", "s", "d"]  # 1..6 +
        self._dec_keys = ["u", "i", "o", "j", "k", "l"]  # 1..6 -

        # —— 新增：人为介入锁存开关 —— 
        self._intervention_latched: bool = False

    # -------- Teleoperator 必需接口 --------
    @property
    def action_features(self) -> dict:
        # 返回与 SO100Leader 对齐的 schema：每个关节/夹爪一条 { "{motor}.pos": float }
        return {f"{m}.pos": float for m in self.motors}

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return PYNPUT_AVAILABLE and isinstance(self.listener, keyboard.Listener) and self.listener.is_alive()

    @property
    def is_calibrated(self) -> bool:
        # 键盘无需校准
        return True

    def connect(self) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError("Keyboard is already connected. Do not run `robot.connect()` twice.")

        if PYNPUT_AVAILABLE:
            self.listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
            self.listener.start()
        else:
            self.listener = None

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    # -------- 键盘事件 --------
    def _on_press(self, key):
        k = key.char if hasattr(key, "char") else key
        self.event_queue.put((k, True))

    def _on_release(self, key):
        k = key.char if hasattr(key, "char") else key
        self.event_queue.put((k, False))
        if PYNPUT_AVAILABLE and keyboard is not None and hasattr(keyboard, "Key") and k == keyboard.Key.esc:
            logging.info("ESC pressed, disconnecting.")
            self.disconnect()

    def _drain_pressed_keys(self):
        while not self.event_queue.empty():
            key_char, is_pressed = self.event_queue.get_nowait()
            self.current_pressed[key_char] = is_pressed

    def _is_pressed(self, k: Any) -> bool:
        return self.current_pressed.get(k, False) is True

    # -------- 生成动作 --------
    def get_action(self) -> dict[str, float]:
        if not self.is_connected:
            raise DeviceNotConnectedError("KeyboardTeleop is not connected. You need to run `connect()` before `get_action()`.")

        self._drain_pressed_keys()

        # 6 个关节：按住按键则每个 tick 递增/递减（不清空 pressed 状态，长按会持续生效）
        for idx in range(6):
            inc_key = self._inc_keys[idx]
            dec_key = self._dec_keys[idx]
            m = self.joint_motors[idx]
            if self._is_pressed(inc_key):
                print(f"当前按压{inc_key}")
                self._pos[m] = min(self._pos[m] + self.config.step, self.config.joint_max)
            if self._is_pressed(dec_key):
                self._pos[m] = max(self._pos[m] - self.config.step, self.config.joint_min)

        # 夹爪：r 增大，p 减小
        if self._is_pressed("r"):
            self._pos[self.gripper_motor] = min(self._pos[self.gripper_motor] + self.config.gripper_step,
                                                self.config.gripper_max)
        if self._is_pressed("p"):
            self._pos[self.gripper_motor] = max(self._pos[self.gripper_motor] - self.config.gripper_step,
                                                self.config.gripper_min)

        # 返回扁平 { "{motor}.pos": value }
        return {f"{m}.pos": self._pos[m] for m in self.motors}

    def get_teleop_events(self) -> dict[str, Any]:
        """
        供 AddTeleopEventsAsInfoStep 调用。返回以下键：
        - TeleopEvents.IS_INTERVENTION  : 是否人为介入（被“锁存”）
        逻辑：任意关节增减键或夹爪键（不含 'p'）被按下 -> 置 True 并保持
            按下 'p' -> 清为 False
        - TeleopEvents.TERMINATE_EPISODE: 是否终止本回合（按 'y'）
        - TeleopEvents.SUCCESS          : 是否标记成功（按 'u'）
        - TeleopEvents.RERECORD_EPISODE : 是否重录（按 'i'）
        """
        # 刷新键盘队列，更新 current_pressed
        self._drain_pressed_keys()

        # —— 复位逻辑优先：按下 'p' 清除锁存 —— 
        if self._is_pressed("p"):
            self._intervention_latched = False
        else:
            # —— 置真逻辑：任意关节增减或夹爪“增大”键（'r'）触发锁存 —— 
            # 注意：为避免同一 tick 中 'p' 又清又置，这里显式排除 'p'
            trigger_keys = self._inc_keys + self._dec_keys + ["r"]
            if any(self._is_pressed(k) for k in trigger_keys):
                self._intervention_latched = True

        # 终止/成功/重录：热键
        terminate_episode = self._is_pressed("r")
        success = self._is_pressed("t")
        rerecord = self._is_pressed("y")

        return {
            TeleopEvents.IS_INTERVENTION: self._intervention_latched,
            TeleopEvents.TERMINATE_EPISODE: bool(terminate_episode),
            TeleopEvents.SUCCESS: bool(success),
            TeleopEvents.RERECORD_EPISODE: bool(rerecord),
        }

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        # 无力反馈
        pass

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError("KeyboardTeleop is not connected. You need to run `robot.connect()` before `disconnect()`.")
        if self.listener is not None:
            self.listener.stop()
    



class KeyboardEndEffectorTeleop(KeyboardTeleop):
    # """
    # Teleop class to use keyboard inputs for end effector control.
    # Designed to be used with PiperFollowerEndEffector robot (6-DoF per arm).
    # """

    # """
    # Simplified keyboard teleoperation for single-arm end-effector control.
    
    # Outputs a dictionary with keys: "delta_x", "delta_y", "delta_z", "gripper".
    # Compatible with standard action processing pipeline steps such as:
    #   - MapTensorToDeltaActionDictStep
    #   - InterventionActionProcessorStep
    
    # Key bindings:
    #   - Translation:
    #       A/D → delta_x (-/+) 
    #       W/S → delta_y (+/-)
    #       Q/E → delta_z (+/-)  [Q: up, E: down]
    #   - Gripper:
    #       C   → close (gripper = 0.0)
    #       V   → open  (gripper = 2.0)
    #   - Exit:
    #       ESC → disconnect
    # """

    # name = "keyboard_ee"

    # def __init__(self, config=None):
    #     """
    #     Initialize the teleop controller.
    #     Config is optional since this is a minimal implementation.
    #     """
    #     super().__init__(config)
    #     self._intervention_latched: bool = False
    #     # Optional: keep a queue for misc keys if needed later
    #     # self.misc_keys_queue = None

    # @property
    # def action_features(self) -> dict:
    #     """
    #     Declare the structure of the action output for logging/schema purposes.
    #     """
    #     return {
    #         "dtype": "float32",
    #         "shape": (4,),
    #         "names": {
    #             "delta_x": 0,
    #             "delta_y": 1,
    #             "delta_z": 2,
    #             "gripper": 3,
    #         },
    #     }

    # def _is_pressed(self, key: Any) -> bool:
    #     """
    #     Helper to check if a key is currently pressed.
    #     Handles both pynput.Key and character keys.
    #     """
    #     return self.current_pressed.get(key, False) is True

    # def get_action(self) -> dict[str, float]:
    #     """
    #     Generate the current action based on pressed keys.
        
    #     Returns:
    #         dict: {"delta_x": float, "delta_y": float, "delta_z": float, "gripper": float}
    #     """
    #     if not self.is_connected:
    #         raise DeviceNotConnectedError(
    #             "Keyboard teleop is not connected. Call `connect()` before `get_action()`."
    #         )

    #     # Process key events from pynput
    #     self._drain_pressed_keys()

    #     # Default: no motion, gripper stays closed (1.0 = hold)
    #     dx = dy = dz = 0.0
    #     gripper = 1.0

    #     # --- Translation Control (WASD + Q/E) ---
    #     if self._is_pressed("a") or self._is_pressed("A"):
    #         dx = -1.0
    #         # print("当前get_action按下a")
    #     if self._is_pressed("d") or self._is_pressed("D"):
    #         dx = +1.0

    #     if self._is_pressed("w") or self._is_pressed("W"):
    #         dy = +1.0
    #     if self._is_pressed("s") or self._is_pressed("S"):
    #         dy = -1.0

    #     if self._is_pressed("q") or self._is_pressed("Q"):
    #         dz = +1.0  # move up
    #     if self._is_pressed("e") or self._is_pressed("E"):
    #         dz = -1.0  # move down

    #     # --- Gripper Control ---
    #     if self._is_pressed("c") or self._is_pressed("C"):
    #         gripper = +1.0  # close
    #     if self._is_pressed("v") or self._is_pressed("V"):
    #         gripper = -1.0  # open

    #     # --- Exit on ESC ---
    #     if self._is_pressed(keyboard.Key.esc):
    #         logging.info("ESC pressed. Disconnecting teleop.")
    #         self.disconnect()

    #     # Clear current pressed keys to avoid sticky actions
    #     # self.current_pressed.clear()

    #     return {
    #         "delta_x": dx,
    #         "delta_y": dy,
    #         "delta_z": dz,
    #         "gripper": gripper,
    #     }

    # def get_teleop_events(self) -> dict[str, Any]:
    #     """
    #     供 AddTeleopEventsAsInfoStep 调用。返回以下键：
    #     - TeleopEvents.IS_INTERVENTION  : 锁存的人为介入标志（按任一 EE 控制键置 True，按 'p' 清零）
    #     - TeleopEvents.TERMINATE_EPISODE: 是否终止本回合（按 'x'）
    #     - TeleopEvents.SUCCESS          : 是否标记成功（按 'y'）
    #     - TeleopEvents.RERECORD_EPISODE : 是否重录（按 't'）
    #     """
    #     # 刷新键盘队列，确保 current_pressed 是最新的
    #     self._drain_pressed_keys()

    #     # --- 触发集合：任一 EE 控制键（平移或夹爪）被按住则置位锁存标志 ---
    #     control_keys = ["a", "A", "d", "D", "w", "W", "s", "S", "q", "Q", "e", "E", "c", "C", "v", "V"]
    #     if any(self._is_pressed(k) for k in control_keys):
    #         self._intervention_latched = True

    #     # --- 复位键：按下 'p'（大小写均可）则清除锁存 ---
    #     if self._is_pressed("p") or self._is_pressed("P"):
    #         self._intervention_latched = False

    #     # 事件热键（与父类一致，便于统一 pipeline）
    #     terminate_episode = self._is_pressed("x")
    #     success = self._is_pressed("y")
    #     rerecord = self._is_pressed("t")

    #     return {
    #         TeleopEvents.IS_INTERVENTION: self._intervention_latched,
    #         TeleopEvents.TERMINATE_EPISODE: bool(terminate_episode),
    #         TeleopEvents.SUCCESS: bool(success),
    #         TeleopEvents.RERECORD_EPISODE: bool(rerecord),
    #     }
    config_class = KeyboardEndEffectorTeleopConfig
    name = "keyboard_ee"

    def __init__(self, config: KeyboardEndEffectorTeleopConfig):
        super().__init__(config)
        self.config = config
        self.misc_keys_queue = Queue()

    # === 升级：输出 Piper EE 期望的 14 个字段（双臂 6-DoF + 夹爪） ===
    @property
    def action_features(self) -> dict:
        return {
            "dtype": "float32",
            "shape": (14,),
            "names": {
                # 左臂
                "left_delta_x": 0, "left_delta_y": 1, "left_delta_z": 2,
                "left_delta_rx": 3, "left_delta_ry": 4, "left_delta_rz": 5,
                "left_delta_gripper": 6,
                # 右臂
                "right_delta_x": 7, "right_delta_y": 8, "right_delta_z": 9,
                "right_delta_rx": 10, "right_delta_ry": 11, "right_delta_rz": 12,
                "right_delta_gripper": 13,
            },
        }

    def _on_press(self, key):
        if hasattr(key, "char"):
            key = key.char
        self.event_queue.put((key, True))

    def _on_release(self, key):
        if hasattr(key, "char"):
            key = key.char
        self.event_queue.put((key, False))
        if key == keyboard.Key.esc:
            logging.info("ESC pressed, disconnecting.")
            self.disconnect()

    def _is_pressed(self, k: Any) -> bool:
        return self.current_pressed.get(k, False) is True

    # === 键位说明 ===
    # 左臂（平移）：←/→: ±x，↑/↓: ∓y，Left-Shift: z-，Right-Shift: z+
    # 左臂（姿态）：R/Y: ±roll，T/G: ±pitch，F/H: ±yaw
    # 左臂（夹爪）：Ctrl-L 收拢，Ctrl-R 张开；默认 1.0 保持
    #
    # 右臂（平移）：A/D: ±x，W/S: ∓y，Z/X: ∓/+ z
    # 右臂（姿态）：U/O: ±roll，I/K: ±pitch，J/L: ±yaw
    # 右臂（夹爪）：N 收拢，M 张开；默认 1.0 保持
    #
    # 说明：实际步长由机器人端 config 的 end_effector_step_sizes / end_effector_rot_step_sizes 决定。

    def get_action(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(
                "KeyboardTeleop is not connected. You need to run `connect()` before `get_action()`."
            )

        self._drain_pressed_keys()

        # 默认：不动（Δ=0），夹爪保持（=1.0）
        l_dx = l_dy = l_dz = 0.0
        l_rx = l_ry = l_rz = 0.0
        l_gr = 0.0

        r_dx = r_dy = r_dz = 0.0
        r_rx = r_ry = r_rz = 0.0
        r_gr = 0.0

        # ---- 左臂 平移（箭头 + Shift 键）----
        if self._is_pressed(keyboard.Key.up):
            l_dx = +1.0
        if self._is_pressed(keyboard.Key.down):
            l_dx = -1.0
        if self._is_pressed(keyboard.Key.right):
            l_dy = -1.0
        if self._is_pressed(keyboard.Key.left):
            l_dy = +1.0
        if self._is_pressed(keyboard.Key.ctrl):
            l_dz = -1.0
        if self._is_pressed(keyboard.Key.shift):
            l_dz = +1.0

        # ---- 左臂 姿态（R/Y: roll，T/G: pitch，F/H: yaw）----
        for k in ("r", "R"):
            if self._is_pressed(k):
                l_rx = -1.0
        for k in ("y", "Y"):
            if self._is_pressed(k):
                l_rx = +1.0
        for k in ("t", "T"):
            if self._is_pressed(k):
                l_ry = +1.0
        for k in ("g", "G"):
            if self._is_pressed(k):
                l_ry = -1.0
        for k in ("f", "F"):
            if self._is_pressed(k):
                l_rz = -1.0
        for k in ("h", "H"):
            if self._is_pressed(k):
                l_rz = +1.0

        # ---- 左臂 夹爪 ----
        if self._is_pressed(keyboard.Key.shift_r):
            l_gr = -1.0  # 收拢
        if self._is_pressed(keyboard.Key.ctrl_r):
            l_gr = 0.0  # 张开

        # ---- 右臂 平移（WASD + ZX）----
        for k in ("a", "A"):
            if self._is_pressed(k):
                r_dx = +1.0
        for k in ("d", "D"):
            if self._is_pressed(k):
                r_dx = -1.0
        for k in ("w", "W"):
            if self._is_pressed(k):
                r_dy = -1.0
        for k in ("s", "S"):
            if self._is_pressed(k):
                r_dy = +1.0
        for k in ("z", "Z"):
            if self._is_pressed(k):
                r_dz = -1.0
        for k in ("x", "X"):
            if self._is_pressed(k):
                r_dz = +1.0

        # ---- 右臂 姿态（U/O: roll，I/K: pitch，J/L: yaw）----
        for k in ("u", "U"):
            if self._is_pressed(k):
                r_rx = -1.0
        for k in ("o", "O"):
            if self._is_pressed(k):
                r_rx = +1.0
        for k in ("i", "I"):
            if self._is_pressed(k):
                r_ry = +1.0
        for k in ("k", "K"):
            if self._is_pressed(k):
                r_ry = -1.0
        for k in ("j", "J"):
            if self._is_pressed(k):
                r_rz = -1.0
        for k in ("l", "L"):
            if self._is_pressed(k):
                r_rz = +1.0

        # ---- 右臂 夹爪 ----
        for k in ("n", "N"):
            if self._is_pressed(k):
                r_gr = -1.0
        for k in ("m", "M"):
            if self._is_pressed(k):
                r_gr = 1.0

        # 记录其它键（可用于打标签/事件）
        for key, pressed in list(self.current_pressed.items()):
            if pressed:
                known = {
                    keyboard.Key.left, keyboard.Key.right, keyboard.Key.up, keyboard.Key.down,
                    keyboard.Key.shift, keyboard.Key.shift_r, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r,
                    "r","R","y","Y","t","T","g","G","f","F","h","H",
                    "a","A","d","D","w","W","s","S","z","Z","x","X",
                    "u","U","o","O","i","I","k","K","j","J","l","L",
                    "n","N","m","M"
                }
                if key not in known:
                    self.misc_keys_queue.put(key)

        # 清空已处理的键表（避免持续粘连；需要长按时，pynput 会继续上报）
        # self.current_pressed.clear()

        # 输出 Piper EE 期望的动作字典（与机器人端 14 维一一对应）
        actions_dict = {
            # 左臂：平移 + 姿态 + 夹爪
            "left_delta_x": l_dx, "left_delta_y": l_dy, "left_delta_z": l_dz,
            "left_delta_rx": l_rx, "left_delta_ry": l_ry, "left_delta_rz": l_rz,
            "left_delta_gripper": l_gr,
            # 右臂：平移 + 姿态 + 夹爪
            "right_delta_x": r_dx, "right_delta_y": r_dy, "right_delta_z": r_dz,
            "right_delta_rx": r_rx, "right_delta_ry": r_ry, "right_delta_rz": r_rz,
            "right_delta_gripper": r_gr,
        }
        for k in actions_dict:
            if k.endswith("_gripper"):
                continue
            actions_dict[k] /= 100.0
        return actions_dict

    def get_teleop_events(self) -> dict[str, Any]:
        """
        供 AddTeleopEventsAsInfoStep 调用。返回以下键：
        - TeleopEvents.IS_INTERVENTION  : 锁存的人为介入标志（按任一 EE 控制键置 True，按 'p' 清零）
        - TeleopEvents.TERMINATE_EPISODE: 是否终止本回合（按 'x'）
        - TeleopEvents.SUCCESS          : 是否标记成功（按 'y'）
        - TeleopEvents.RERECORD_EPISODE : 是否重录（按 't'）
        """
        # 刷新键盘队列，确保 current_pressed 是最新的
        self._drain_pressed_keys()

        # --- 触发集合：任一 EE 控制键（平移或夹爪）被按住则置位锁存标志 ---
        control_keys = ["a", "A", "d", "D", "w", "W", "s", "S", "q", "Q", "e", "E", "c", "C", "v", "V",
                        keyboard.Key.left, keyboard.Key.right, keyboard.Key.up, keyboard.Key.down,
                        keyboard.Key.shift, keyboard.Key.shift_r, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r]
        if any(self._is_pressed(k) for k in control_keys):
            self._intervention_latched = True

        # --- 复位键：按下 'p'（大小写均可）则清除锁存 ---
        if self._is_pressed("p") or self._is_pressed("P"):
            self._intervention_latched = False

        # 事件热键（与父类一致，便于统一 pipeline）
        terminate_episode = self._is_pressed("x")
        success = self._is_pressed("y")
        rerecord = self._is_pressed("t")
        # self._intervention_latched = True#debug
        return {
            TeleopEvents.IS_INTERVENTION: self._intervention_latched,
            TeleopEvents.TERMINATE_EPISODE: bool(terminate_episode),
            TeleopEvents.SUCCESS: bool(success),
            TeleopEvents.RERECORD_EPISODE: bool(rerecord),
        }