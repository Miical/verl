#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
logging.getLogger("can.interfaces.socketcan").setLevel(logging.ERROR)

import math
from dataclasses import dataclass
import draccus

# 关键：导入以触发注册（无需使用变量）
from lerobot.robots import (  # noqa: F401
    piper_follower,
    RobotConfig,
    make_robot_from_config,
)
from lerobot.utils.robot_utils import busy_wait


@dataclass
class TestPiperEEConfig:
    # 机器人配置，仅需要 type；其余走你在 Config 里写死/默认的值
    robot: RobotConfig

    # 测试参数
    arm: str = "left"            # "left" 或 "right"
    axis: str = "x"              # "x" / "y" / "z"
    distance_m: float = 0.05     # 期望位移（米），默认 5cm
    rate_hz: float = 50.0        # 发送频率（Hz）
    pause_s: float = 0.5         # 抵达后原地停留（秒）


@draccus.wrap()
def main(cfg: TestPiperEEConfig):
    robot = make_robot_from_config(cfg.robot)  # 读取你写死/默认的 PiperFollowerEndEffectorConfig
    robot.connect()
    try:
        assert cfg.arm in ("left", "right"), "arm 只能为 'left' 或 'right'"
        assert cfg.axis in ("x", "y", "z"), "axis 只能为 'x'/'y'/'z'"

        # 从机器人配置读取末端步长（单位米）
        step_sizes = robot.config.end_effector_step_sizes
        step_m = float(step_sizes[cfg.axis])
        assert step_m > 0, "end_effector_step_sizes 中的步长必须为正数"

        n_steps = max(1, int(math.ceil(cfg.distance_m / step_m)))
        dt = 1.0 / cfg.rate_hz

        def send_delta(arm: str, axis: str, sign: float):
            """在对应臂/轴给 +1/-1，其它置 0；夹爪维持 1.0（不动作）"""
            a = {
                "left_delta_x": 0.0, "left_delta_y": 0.0, "left_delta_z": 0.0, "left_gripper": 1.0,
                "right_delta_x": 0.0, "right_delta_y": 0.0, "right_delta_z": 0.0, "right_gripper": 1.0,
            }
            a[f"{arm}_delta_{axis}"] = sign
            robot.send_action(a)

        print(f"[INFO] Start EE test: arm={cfg.arm}, axis={cfg.axis}, "
              f"distance≈{n_steps*step_m:.3f} m ({n_steps} steps, step={step_m:.3f} m/step)")

        # 正向
        for _ in range(n_steps):
            send_delta(cfg.arm, cfg.axis, +1.0)
            busy_wait(dt)

        busy_wait(cfg.pause_s)

        # 反向回到起点
        for _ in range(n_steps):
            send_delta(cfg.arm, cfg.axis, -1.0)
            busy_wait(dt)

        busy_wait(cfg.pause_s)
        print("[INFO] Done. EE forward & back completed.")
    finally:
        robot.disconnect()
        print("[INFO] Robot disconnected.")


if __name__ == "__main__":
    main()


"""
python -m lerobot.scripts.test_piper_ee_walk \
  --robot.type=piper_follower_end_effector \
  --arm=left --axis=x --distance_m=0.05 --rate_hz=50
"""