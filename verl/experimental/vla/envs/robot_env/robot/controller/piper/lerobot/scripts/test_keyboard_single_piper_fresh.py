#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
最小可用：单臂关节空间键盘遥操作 → Piper 单臂 Follower
- 键位：q w e a s d (6关节+) / u i o j k l (6关节-) / r(夹爪+) / p(夹爪-)
- Teleop 输出：{"<prefix>_joint_i.pos": float, ..., "<prefix>_gripper.pos": float}
- Robot 期望：与 SO100Follower 的 action schema 一致
"""
import pdb
import time
from dataclasses import dataclass

import draccus

from lerobot.utils.errors import DeviceNotConnectedError

# 触发注册：很重要（根据你的工程结构，确保这些 import 不要删）
from lerobot.robots import (  # noqa: F401
    piper_follower,           # 注册 PiperSingleFollower / PiperFollower / EE 版本
    RobotConfig,
    make_robot_from_config,
)
from lerobot.teleoperators import (  # noqa: F401
    keyboard,                 # 注册 KeyboardTeleop / KeyboardEndEffectorTeleop
    TeleoperatorConfig,
    make_teleoperator_from_config,
)

from lerobot.utils.robot_utils import busy_wait


BANNER = r"""
[Keyboard Joint Teleop → Piper Single Follower]
关节 +: q w e a s d
关节 -: u i o j k l
夹爪  +: r
夹爪  -: p
按 ESC 退出
"""


@dataclass
class RunConfig:
    teleop: TeleoperatorConfig     # --teleop.type=keyboard
    robot: RobotConfig             # --robot.type=piper_single_follower
    rate_hz: float = 50.0          # 循环频率
    print_every: int = 10          # 每多少 tick 打印一次（0=不打印）
    duration_s: float = 0.0        # 运行时长（<=0 表示无限循环）


@draccus.wrap()
def main(cfg: RunConfig):
    teleop = make_teleoperator_from_config(cfg.teleop)
    robot = make_robot_from_config(cfg.robot)

    # 连接
    teleop.connect()
    robot.connect()

    print(BANNER)

    dt = 1.0 / max(1.0, float(cfg.rate_hz))
    step = 0
    t0 = time.time()

    try:
        while True:
            # 可选停止条件
            if cfg.duration_s and (time.time() - t0) >= cfg.duration_s:
                print("[INFO] Reached duration limit, exiting.")
                break

            # 取键盘动作
            try:
                action = teleop.get_action()
            except DeviceNotConnectedError:
                print("[INFO] Teleop disconnected (ESC). Exiting loop.")
                break

            # 下发到机器人（PiperSingleFollower 的 send_action 与 SO100 对齐）
            robot.send_action(action)
            # pdb.set_trace()
            if cfg.print_every > 0 and (step % cfg.print_every == 0):
                # 打印前 6 个关节与夹爪（按命名过滤，避免刷屏）
                keys = [k for k in action.keys() if k.endswith(".pos")]
                keys = sorted(keys)  # 稳定输出
                show = {k: action[k] for k in keys[:7]}  # 6 关节 + 夹爪
                print(f"[tick={step}] {show}")

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
        print("[INFO] Clean exit.")


if __name__ == "__main__":
    main()

"""
python -m lerobot.scripts.test_keyboard_single_piper_fresh   --teleop.type=keyboard   --robot.type=piper_single_follower   --rate_hz=50   --print_every=10

"""