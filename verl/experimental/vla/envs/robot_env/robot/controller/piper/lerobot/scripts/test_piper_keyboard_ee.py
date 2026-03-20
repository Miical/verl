#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
logging.getLogger("can.interfaces.socketcan").setLevel(logging.ERROR)

from dataclasses import dataclass
import time

import draccus
from lerobot.utils.errors import DeviceNotConnectedError

# 关键：导入以触发“注册”（不要删除）
from lerobot.robots import (  # noqa: F401
    piper_follower,
    RobotConfig,
    make_robot_from_config,
)
from lerobot.teleoperators import (  # noqa: F401
    keyboard,
    TeleoperatorConfig,
    make_teleoperator_from_config,
)
from lerobot.utils.robot_utils import busy_wait


@dataclass
class KeyboardEEDemoConfig:
    # 必填：从命令行传入类型
    teleop: TeleoperatorConfig   # --teleop.type=keyboard_ee
    robot: RobotConfig           # --robot.type=piper_follower_end_effector
    rate_hz: float = 100.0       # 发送频率
    print_every: int = 20        # 每多少帧打印一次动作（0 表示不打印）


BANNER = r"""
[INFO] Keyboard EE teleop started (6-DoF per arm + gripper).

左臂（平移）：←/→ = ±x，↑/↓ = ∓/＋y，Left-Shift = z-，Right-Shift = z+
左臂（姿态）：R/Y = roll ∓/+，T/G = pitch +/−，F/H = yaw ∓/+
左臂（夹爪）：Ctrl-L = 闭合(0)，Ctrl-R = 张开(2)

右臂（平移）：A/D = ±x，W/S = ∓/＋y，Z/X = z ∓/+
右臂（姿态）：U/O = roll ∓/+，I/K = pitch +/−，J/L = yaw ∓/+
右臂（夹爪）：N = 闭合(0)，M = 张开(2)

按 ESC 退出。
"""


@draccus.wrap()
def main(cfg: KeyboardEEDemoConfig):
    teleop = make_teleoperator_from_config(cfg.teleop)
    robot = make_robot_from_config(cfg.robot)

    teleop.connect()
    robot.connect()

    print(BANNER)

    dt = 1.0 / max(1.0, float(cfg.rate_hz))
    step = 0

    try:
        while True:
            try:
                # 从键盘读取本帧动作（14维字典：左右臂位置+姿态+夹爪）
                action = teleop.get_action()
            except DeviceNotConnectedError:
                print("[INFO] Teleop disconnected (ESC). Exiting loop.")
                break

            # 下发到机器人（机器人端已兼容 14维 或旧 8维）
            robot.send_action(action)

            if cfg.print_every > 0 and (step % cfg.print_every == 0):
                # 只展示关键字段，避免刷屏
                l = {k: action[k] for k in ("left_delta_x","left_delta_y","left_delta_z",
                                            "left_delta_rx","left_delta_ry","left_delta_rz","left_gripper")}
                r = {k: action[k] for k in ("right_delta_x","right_delta_y","right_delta_z",
                                            "right_delta_rx","right_delta_ry","right_delta_rz","right_gripper")}
                print(f"[tick={step}] L:{l} | R:{r}")

            step += 1
            busy_wait(dt)

    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt, stopping.")

    finally:
        # 清理
        try:
            teleop.disconnect()
        except Exception:
            pass
        try:
            robot.disconnect()
        except Exception:
            pass
        print("[INFO] Clean exit.\n")


if __name__ == "__main__":
    main()

"""
用法示例（推荐 -m 方式）：
python -m lerobot.scripts.test_piper_keyboard_ee \
  --teleop.type=keyboard_ee \
  --robot.type=piper_follower_end_effector \
  --rate_hz=50
"""
