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
Simple script to control a robot from teleoperation.

Example:

```shell
python -m lerobot.scripts.server.find_joint_limits \
    --robot.type=piper_follower \
    --robot.port_left=can0 \
    --robot.port_right=can1 \
    --teleop.type=piper_leader \
    --teleop.port_left=can2 \
    --teleop.port_right=can3 \
```
"""
import logging
logging.getLogger("can.interfaces.socketcan").setLevel(logging.ERROR)

import time
from dataclasses import dataclass

import draccus
import numpy as np

from lerobot.model.kinematics import RobotKinematics
from lerobot.robots import (  # noqa: F401
    piper_follower,
    RobotConfig,
    make_robot_from_config,
)
from lerobot.teleoperators import (  # noqa: F401
    piper_leader,
    TeleoperatorConfig,
    make_teleoperator_from_config,
)
from lerobot.utils.robot_utils import busy_wait
import pdb

@dataclass
class FindJointLimitsConfig:
    teleop: TeleoperatorConfig
    robot: RobotConfig
    # Limit the maximum frames per second. By default, no limit.
    teleop_time_s: float = 30
    # Display all cameras on screen
    display_data: bool = False


@draccus.wrap()
def find_joint_and_ee_bounds(cfg: FindJointLimitsConfig):
    teleop = make_teleoperator_from_config(cfg.teleop)
    robot = make_robot_from_config(cfg.robot)

    teleop.connect()
    robot.connect()

    start_episode_t = time.perf_counter()
    robot_type = getattr(robot.config, "robot_type", "piper")
    # ### === 修改开始：创建日志文件 ===
    import os, csv
    from datetime import datetime

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"action_log_{timestamp}.csv")
    # 写入表头：时间戳 + 所有动作键
    # log_path="/home/agilex/lerobot_hil-serl/src/lerobot/scripts/test1.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["time_s"] + list(teleop.action_features.keys())
        writer.writerow(header)

    print(f"[INFO] Action log initialized at: {log_path}")
    ### === 修改结束 ===
    # kinematics = RobotKinematics(cfg.robot.urdf_path, cfg.robot.target_frame_name)

    # Initialize min/max values
    observation = robot.get_observation()
    left_joint_positions = np.array([observation[f"{motor}.pos"] for motor in robot.bus_left.motors if 'gripper' not in motor])
    right_joint_positions = np.array([observation[f"{motor}.pos"] for motor in robot.bus_right.motors if 'gripper' not in motor])
    # 末端执行器位置
    # left_ee_pos = kinematics.forward_kinematics(left_joint_positions)[:3, 3]
    # right_ee_pos = kinematics.forward_kinematics(right_joint_positions)[:3, 3]

    max_pos_left = left_joint_positions.copy()
    min_pos_left = left_joint_positions.copy()
    # max_ee_left = left_ee_pos.copy()
    # min_ee_left = left_ee_pos.copy()
    max_pos_right = right_joint_positions.copy()
    min_pos_right = right_joint_positions.copy()
    # max_ee_right = right_ee_pos.copy()
    # min_ee_right = right_ee_pos.copy()

    while True:
        action = teleop.get_action()
        robot.send_action(action)

        observation = robot.get_observation()
        left_joint_positions = np.array([observation[f"{motor}.pos"] for motor in robot.bus_left.motors if 'gripper' not in motor])
        right_joint_positions = np.array([observation[f"{motor}.pos"] for motor in robot.bus_right.motors if 'gripper' not in motor])
        # left_ee_pos = kinematics.forward_kinematics(left_joint_positions)[:3, 3]
        # right_ee_pos = kinematics.forward_kinematics(right_joint_positions)[:3, 3]

        # Skip initial warmup period
        if (time.perf_counter() - start_episode_t) < 5:
            continue
            ### === 修改开始：记录每帧 action ===
        elapsed = time.perf_counter() - start_episode_t
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            row = [round(elapsed, 3)] + [round(action.get(k, 0.0), 6) for k in teleop.action_features.keys()]
            writer.writerow(row)
    ### === 修改结束 ===
        # Update min/max values
        # max_ee_left = np.maximum(max_ee_left, left_ee_pos)
        # max_ee_right = np.maximum(max_ee_right, right_ee_pos)
        # min_ee_left = np.minimum(min_ee_left, left_ee_pos)
        # min_ee_right = np.minimum(min_ee_right, right_ee_pos)
        max_pos_left = np.maximum(max_pos_left, left_joint_positions)
        max_pos_right = np.maximum(max_pos_right, right_joint_positions)
        min_pos_left = np.minimum(min_pos_left, left_joint_positions)
        min_pos_right = np.minimum(min_pos_right, right_joint_positions)
        # pdb.set_trace()
        if time.perf_counter() - start_episode_t > cfg.teleop_time_s:
            
            # print(f"Max left ee position {np.round(max_ee_left, 4).tolist()}")
            # print(f"Min left ee position {np.round(min_ee_left, 4).tolist()}")
            print(f"Max left joint pos position {np.round(max_pos_left, 4).tolist()}")
            print(f"Min left joint pos position {np.round(min_pos_left, 4).tolist()}")
            # print(f"Max right ee position {np.round(max_ee_right, 4).tolist()}")
            # print(f"Min right ee position {np.round(min_ee_right, 4).tolist()}")
            print(f"Max right joint pos position {np.round(max_pos_right, 4).tolist()}")
            print(f"Min right joint pos position {np.round(min_pos_right, 4).tolist()}")
            break

busy_wait(0.01)


if __name__ == "__main__":
    # pdb.set_trace()
    find_joint_and_ee_bounds()
    """
    # cd to the lerobot directory and run:
    python -m lerobot.scripts.find_joint_limits_piper \
    --robot.type=piper_follower \
    --teleop.type=piper_leader
    """