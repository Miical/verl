#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""
Replay recorded teleoperation actions using only the Piper follower robot.

Example:
python -m lerobot.scripts.replay_action \
    --robot.type=piper_follower \
    --action_file=/home/agilex/lerobot_hil-serl/logs/action_log_20251013_152955.csv
"""

import logging
logging.getLogger("can.interfaces.socketcan").setLevel(logging.ERROR)

import time
from dataclasses import dataclass
import draccus
import numpy as np
from lerobot.robots import (
    piper_follower,
    RobotConfig,
    make_robot_from_config,
)
from lerobot.utils.robot_utils import busy_wait
import csv, os
import pdb


@dataclass
class ReplayJointActionsConfig:
    robot: RobotConfig
    # 已记录的动作 CSV 文件路径
    action_file: str
    # 控制时长倍率（1.0 = 原速, >1 = 加快, <1 = 慢放）
    playback_speed: float = 1.0


@draccus.wrap()
def replay_joint_actions(cfg: ReplayJointActionsConfig):
    ### === 修改开始：只连接 robot，不使用 teleop ===
    robot = make_robot_from_config(cfg.robot)
    robot.connect()
    print(f"[INFO] Connected to {robot}")
    ### === 修改结束 ===

    if not os.path.exists(cfg.action_file):
        raise FileNotFoundError(f"动作文件不存在: {cfg.action_file}")

    print(f"[INFO] 正在加载动作文件: {cfg.action_file}")

    # 读取 CSV 文件
    actions = []
    with open(cfg.action_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = float(row["time_s"])
            # 将每一行动作转为字典
            act = {k: float(v) for k, v in row.items() if k != "time_s"}
            actions.append((t, act))

    print(f"[INFO] 已加载 {len(actions)} 帧动作，总时长 {actions[-1][0]:.2f}s")

    start_t = time.perf_counter()

    # 循环发送动作
    for i in range(len(actions)):
        cur_time, action = actions[i]

        # 计算下一帧时间间隔
        if i < len(actions) - 1:
            next_time = actions[i + 1][0]
            dt = (next_time - cur_time) / cfg.playback_speed
        else:
            dt = 0.01

        # 发送动作给机械臂
        robot.send_action(action)

        # 控制播放速率
        busy_wait(dt)

        if i % 100 == 0:  # 每100帧提示一次
            print(f"[INFO] 已回放 {i+1}/{len(actions)} 帧")

    print("[INFO] 动作回放结束。")
    robot.disconnect()
    print("[INFO] 已断开 Piper follower 连接。")


if __name__ == "__main__":
    # pdb.set_trace()
    replay_joint_actions()
