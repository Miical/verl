import logging
logging.getLogger("can.interfaces.socketcan").setLevel(logging.ERROR)
# #!/usr/bin/env python
# # -*- coding: utf-8 -*-

# """
# 回放末端轨迹到 Piper 双臂（PiperFollowerEndEffector）
# - 读取两个 txt 文件：
#     left_ee_poses.txt:  每行 [x, y, z, roll, pitch, yaw]
#     right_ee_poses.txt: 每行 [x, y, z, roll, pitch, yaw]
# - 默认假设位置单位: mm，姿态单位: deg（和你原始 demo 一致）
# - 会转换为:
#     位置: m
#     姿态: rad
# - 通过 PiperFollowerEndEffector.send_action() 下发：
#     {
#         "left_end_x.value": x_m,
#         "left_end_y.value": y_m,
#         "left_end_z.value": z_m,
#         "left_end_rx.value": rx_rad,
#         "left_end_ry.value": ry_rad,
#         "left_end_rz.value": rz_rad,
#         "right_end_x.value": ...,
#         ...
#     }

# 同时记录：
# - 每一步发送给机器人的目标末端位姿（左右臂）
# - robot.get_observation() 返回的实际末端位姿
# - 观测值与发送值的误差（obs - cmd）
# 到 CSV 文件：piper_end_effector_replay_log.csv
# """

# import time
# from dataclasses import dataclass
# import csv

# import draccus
# import numpy as np

# from lerobot.utils.errors import DeviceNotConnectedError

# # 触发注册 PiperFollowerEndEffector （和你原来的 keyboard 脚本一样写就行）
# from lerobot.robots import (  # noqa: F401
#     piper_follower,           # 注册 PiperSingleFollower / PiperFollower / EndEffector 版本
#     RobotConfig,
#     make_robot_from_config,
# )


# BANNER = r"""
# [PiperFollowerEndEffector 末端轨迹回放]
# - 从 txt 中读取左右臂末端位姿 (x,y,z,roll,pitch,yaw)
# - 单位: 默认 mm / deg，会自动转换到 m / rad
# - 按 txt 顺序依次发送给机器人
# - 同时记录命令值 / 观测值 / 误差到 CSV
# """


# @dataclass
# class RunConfig:
#     robot: RobotConfig
#     left_path: str                   # --left_path=/path/to/left_ee_poses.txt
#     right_path: str                  # --right_path=/path/to/right_ee_poses.txt
#     rate_hz: float = 50.0            # 发送频率（类似你原来 freq）
#     print_every: int = 10            # 每多少步打印一次末端信息（0=不打印）
#     pos_in_mm: bool = True           # txt 里的位置是否为 mm
#     rot_in_deg: bool = True          # txt 里的姿态是否为 deg


# @draccus.wrap()
# def main(cfg: RunConfig):
#     robot = make_robot_from_config(cfg.robot)

#     # 连接机器人（内部会连左右臂 + 相机等）
#     robot.connect()
#     robot.reset()
#     print(BANNER)

#     # 1. 读取左右臂末端位姿
#     left_ee_poses = np.loadtxt(cfg.left_path, delimiter=",")
#     right_ee_poses = np.loadtxt(cfg.right_path, delimiter=",")

#     # 处理只有一行的情况，保证是 2D
#     if left_ee_poses.ndim == 1:
#         left_ee_poses = left_ee_poses.reshape(1, -1)
#     if right_ee_poses.ndim == 1:
#         right_ee_poses = right_ee_poses.reshape(1, -1)

#     if left_ee_poses.shape != right_ee_poses.shape:
#         raise ValueError(f"左右臂轨迹维度不一致: left={left_ee_poses.shape}, right={right_ee_poses.shape}")

#     if left_ee_poses.shape[1] < 6:
#         raise ValueError(f"期望每行至少 6 个值 [x,y,z,roll,pitch,yaw]，实际 {left_ee_poses.shape[1]}")

#     N = left_ee_poses.shape[0]
#     print(f"[INFO] Loaded {N} poses for each arm.")

#     dt = 1.0 / max(1.0, float(cfg.rate_hz))

#     # 日志：记录每一步的发送值、观测值和误差
#     logs = []

#     try:
#         for i in range(N):
#             # ------- 左臂 -------
#             lx, ly, lz, lroll, lpitch, lyaw = left_ee_poses[i, :6]
#             # ------- 右臂 -------
#             rx, ry, rz, rroll, rpitch, ryaw = right_ee_poses[i, :6]

#             # 单位转换：mm -> m
#             if cfg.pos_in_mm:
#                 lx_m, ly_m, lz_m = lx / 1000.0, ly / 1000.0, lz / 1000.0
#                 rx_m, ry_m, rz_m = rx / 1000.0, ry / 1000.0, rz / 1000.0
#             else:
#                 lx_m, ly_m, lz_m = float(lx), float(ly), float(lz)
#                 rx_m, ry_m, rz_m = float(rx), float(ry), float(rz)

#             # 单位转换：deg -> rad
#             if cfg.rot_in_deg:
#                 lrx, lry, lrz = np.deg2rad([lroll, lpitch, lyaw])
#                 rrx, rry, rrz = np.deg2rad([rroll, rpitch, ryaw])
#             else:
#                 lrx, lry, lrz = float(lroll), float(lpitch), float(lyaw)
#                 rrx, rry, rrz = float(rroll), float(rpitch), float(ryaw)

#             # 构造 PiperFollowerEndEffector 期望的 action dict
#             action = {
#                 # 左臂
#                 "left_end_x.value": float(lx_m),
#                 "left_end_y.value": float(ly_m),
#                 "left_end_z.value": float(lz_m),
#                 "left_end_rx.value": float(lrx),
#                 "left_end_ry.value": float(lry),
#                 "left_end_rz.value": float(lrz),
#                 "left_end_gripper.value": 0.5,
#                 # 右臂
#                 "right_end_x.value": float(rx_m),
#                 "right_end_y.value": float(ry_m),
#                 "right_end_z.value": float(rz_m),
#                 "right_end_rx.value": float(rrx),
#                 "right_end_ry.value": float(rry),
#                 "right_end_rz.value": float(rrz),
#                 "right_end_gripper.value": 0.5,
#             }

#             # 下发命令
#             try:
#                 robot.send_action(action)
#             except DeviceNotConnectedError:
#                 print("[ERR] Robot device disconnected, stop replay.")
#                 break

#             # 打印发送的目标
#             if cfg.print_every > 0 and (i % cfg.print_every == 0):
#                 print(
#                     f"[{i:05d}/{N:05d}] "
#                     f"L xyz=({lx_m:.3f},{ly_m:.3f},{lz_m:.3f}) "
#                     f"R xyz=({rx_m:.3f},{ry_m:.3f},{rz_m:.3f})"
#                 )

#             # 等待机器人执行一小段时间
#             time.sleep(dt)

#             # 读取观测并计算误差
#             obs = robot.get_observation()
#             try:
#                 # 左臂观测
#                 lox = float(obs["left_end_x.value"])
#                 loy = float(obs["left_end_y.value"])
#                 loz = float(obs["left_end_z.value"])
#                 lorx = float(obs["left_end_rx.value"])
#                 lory = float(obs["left_end_ry.value"])
#                 lorz = float(obs["left_end_rz.value"])

#                 # 右臂观测
#                 rox = float(obs["right_end_x.value"])
#                 roy = float(obs["right_end_y.value"])
#                 roz = float(obs["right_end_z.value"])
#                 rorx = float(obs["right_end_rx.value"])
#                 rory = float(obs["right_end_ry.value"])
#                 rorz = float(obs["right_end_rz.value"])
#             except KeyError as e:
#                 print(f"[WARN] get_observation() 缺少键 {e}，可用键有: {list(obs.keys())}")
#                 continue

#             # 误差 = 观测 - 发送（同单位：m / rad）
#             left_err = [
#                 lox - lx_m,
#                 loy - ly_m,
#                 loz - lz_m,
#                 lorx - lrx,
#                 lory - lry,
#                 lorz - lrz,
#             ]
#             right_err = [
#                 rox - rx_m,
#                 roy - ry_m,
#                 roz - rz_m,
#                 rorx - rrx,
#                 rory - rry,
#                 rorz - rrz,
#             ]

#             # 每 print_every 步打印一次误差
#             if cfg.print_every > 0 and (i % cfg.print_every == 0):
#                 print(
#                     f"    L err=({left_err[0]:.4e},{left_err[1]:.4e},{left_err[2]:.4e},"
#                     f"{left_err[3]:.4e},{left_err[4]:.4e},{left_err[5]:.4e})"
#                 )
#                 print(
#                     f"    R err=({right_err[0]:.4e},{right_err[1]:.4e},{right_err[2]:.4e},"
#                     f"{right_err[3]:.4e},{right_err[4]:.4e},{right_err[5]:.4e})"
#                 )

#             # 记录到 logs（1 行）：step + cmd(12) + obs(12) + err(12) = 37 列
#             logs.append(
#                 [
#                     float(i),
#                     # 左臂命令
#                     lx_m, ly_m, lz_m, lrx, lry, lrz,
#                     # 右臂命令
#                     rx_m, ry_m, rz_m, rrx, rry, rrz,
#                     # 左臂观测
#                     lox, loy, loz, lorx, lory, lorz,
#                     # 右臂观测
#                     rox, roy, roz, rorx, rory, rorz,
#                     # 左臂误差
#                     *left_err,
#                     # 右臂误差
#                     *right_err,
#                 ]
#             )

#         print("[INFO] Replay finished.")

#     except KeyboardInterrupt:
#         print("\n[INFO] KeyboardInterrupt, stopping replay.")
#     finally:
#         # 先断开机器人
#         try:
#             robot.disconnect()
#         except Exception:
#             pass

#         # 再把日志写入 CSV 文件
#         if len(logs) > 0:
#             header_cols = [
#                 "step",
#                 # 命令值（commanded）
#                 "lx_cmd", "ly_cmd", "lz_cmd", "lrx_cmd", "lry_cmd", "lrz_cmd",
#                 "rx_cmd", "ry_cmd", "rz_cmd", "rrx_cmd", "rry_cmd", "rrz_cmd",
#                 # 观测值（observed）
#                 "lx_obs", "ly_obs", "lz_obs", "lrx_obs", "lry_obs", "lrz_obs",
#                 "rx_obs", "ry_obs", "rz_obs", "rrx_obs", "rry_obs", "rrz_obs",
#                 # 误差（error = obs - cmd）
#                 "lx_err", "ly_err", "lz_err", "lrx_err", "lry_err", "lrz_err",
#                 "rx_err", "ry_err", "rz_err", "rrx_err", "rry_err", "rrz_err",
#             ]
#             log_path = "piper_end_effector_replay_log.csv"

#             with open(log_path, "w", newline="", encoding="utf-8") as f:
#                 writer = csv.writer(f)
#                 # 写表头
#                 writer.writerow(header_cols)
#                 # 写每一行数据
#                 for row in logs:
#                     writer.writerow(row)

#             print(f"[INFO] Saved observation log to {log_path}")

#         print("[INFO] Clean exit.")


# if __name__ == "__main__":
#     main()

# """
# 示例运行：

# python -m lerobot.scripts.test_piper_end_effector_replay \
#   --robot.type=piper_follower_end_effector \
#   --left_path=/home/agilex-home/agilex/lerobot_hil-serl/src/lerobot/motors/piper/left_ee_poses.txt \
#   --right_path=/home/agilex-home/agilex/lerobot_hil-serl/src/lerobot/motors/piper/left_ee_poses.txt \
#   --rate_hz=50 \
#   --print_every=10
# """


# !/usr/bin/env python
# -*- coding: utf-8 -*-

# """
# [PiperFollowerEndEffector 观测回显测试]
# 循环执行：
#   1. obs = robot.get_observation()
#   2. robot.send_action(obs 对应的末端位姿)
# 用途：测试 get_observation() 返回值是否稳定。
# """

# import time
# from dataclasses import dataclass
# import draccus

# from lerobot.utils.errors import DeviceNotConnectedError
# from lerobot.robots import (  # noqa: F401
#     piper_follower,
#     RobotConfig,
#     make_robot_from_config,
# )

# BANNER = r"""
# [PiperFollowerEndEffector 末端观测回显测试]
# - 连续读取 get_observation() 并将观测值原样下发
# - 不记录误差、不保存日志
# - 用于测试观测信号稳定性
# """


# @dataclass
# class RunConfig:
#     robot: RobotConfig
#     rate_hz: float = 50.0     # 发送频率
#     print_every: int = 10     # 每多少步打印一次
#     max_steps: int = 0        # 0 表示无限循环


# @draccus.wrap()
# def main(cfg: RunConfig):
#     robot = make_robot_from_config(cfg.robot)
#     robot.connect()
#     # robot.reset()
#     print(BANNER)

#     dt = 1.0 / max(1.0, float(cfg.rate_hz))
#     step = 0

#     try:
#         while True:
#             if cfg.max_steps > 0 and step >= cfg.max_steps:
#                 break

#             obs = robot.get_observation()
#             print(obs)
#             if obs is None:
#                 print("[WARN] get_observation() 返回 None，跳过该步。")
#                 continue

#             try:
#                 # action = {
#                 #     # 左臂
#                 #     "left_end_x.value": float(obs["left_end_x.value"]),
#                 #     "left_end_y.value": float(obs["left_end_y.value"]),
#                 #     "left_end_z.value": float(obs["left_end_z.value"]),
#                 #     "left_end_rx.value": float(obs["left_end_rx.value"]),
#                 #     "left_end_ry.value": float(obs["left_end_ry.value"]),
#                 #     "left_end_rz.value": float(obs["left_end_rz.value"]),
#                 #     "left_end_gripper.value": float(obs.get("left_end_gripper.value", 0.5)),
#                 #     # 右臂
#                 #     "right_end_x.value": float(obs["right_end_x.value"]),
#                 #     "right_end_y.value": float(obs["right_end_y.value"]),
#                 #     "right_end_z.value": float(obs["right_end_z.value"]),
#                 #     "right_end_rx.value": float(obs["right_end_rx.value"]),
#                 #     "right_end_ry.value": float(obs["right_end_ry.value"]),
#                 #     "right_end_rz.value": float(obs["right_end_rz.value"]),
#                 #     "right_end_gripper.value": float(obs.get("right_end_gripper.value", 0.5)),
#                 # }
#                 action_R=[0.0850,  0.0214, 0.2027, 0.00397, 1.48957, 0.00325]
#                 action = {
#                     # 左臂
#                     "left_end_x.value": float(obs["left_end_x.value"]),
#                     "left_end_y.value": float(obs["left_end_y.value"]),
#                     "left_end_z.value": float(obs["left_end_z.value"]),
#                     "left_end_rx.value": float(obs["left_end_rx.value"]),
#                     "left_end_ry.value": float(obs["left_end_ry.value"]),
#                     "left_end_rz.value": float(obs["left_end_rz.value"]),
#                     "left_end_gripper.value": float(obs.get("left_end_gripper.value", 0.5)),
#                     # 右臂
#                     "right_end_x.value": float(action_R[0]),
#                     "right_end_y.value": float(action_R[1]),
#                     "right_end_z.value": float(action_R[2]),
#                     "right_end_rx.value": float(action_R[3]),
#                     "right_end_ry.value": float(action_R[4]),
#                     "right_end_rz.value": float(action_R[5]),
#                     "right_end_gripper.value": float(obs.get("right_end_gripper.value", 0.5)),
#                 }
#             except KeyError as e:
#                 print(f"[WARN] get_observation() 缺少键 {e}，可用键有: {list(obs.keys())}")
#                 continue

#             try:
#                 print( "[DEBUG] Sending")
#                 robot.send_action(action)
#             except DeviceNotConnectedError:
#                 print("[ERR] Robot device disconnected.")
#                 break

#             if cfg.print_every > 0 and (step % cfg.print_every == 0):
#                 print(
#                     f"[{step:05d}] "
#                     f"L=({action['left_end_x.value']:.4f},{action['left_end_y.value']:.4f},{action['left_end_z.value']:.4f}) "
#                     f"R=({action['right_end_x.value']:.4f},{action['right_end_y.value']:.4f},{action['right_end_z.value']:.4f})"
#                 )

#             step += 1
#             time.sleep(dt)

#     except KeyboardInterrupt:
#         print("\n[INFO] Stopped by user.")
#     finally:
#         try:
#             robot.disconnect()
#         except Exception:
#             pass
#         print("[INFO] Clean exit.")


# if __name__ == "__main__":
#     main()

# """
# python -m lerobot.scripts.test_piper_end_effector_replay \
#   --robot.type=piper_follower_end_effector \
#   --rate_hz=50 \
#   --print_every=10
# python -m lerobot.scripts.test_piper_end_effector_replay \
#   --robot.type=piper_follower \
#   --rate_hz=50 \
#   --print_every=10
# """


#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
[PiperFollowerEndEffector endpose 数据集回放]

适用于“action 已经是末端绝对位姿”的 LeRobot 数据集：
  action 维度 = 14:
    [l_x, l_y, l_z, l_rx, l_ry, l_rz, l_grip,
     r_x, r_y, r_z, r_rx, r_ry, r_rz, r_grip]

功能：
  - 加载 LeRobotDataset
  - 按顺序或指定区间 index 读取样本
  - 直接将 action 映射为机器人末端控制命令并下发
"""

import logging
logging.getLogger("can.interfaces.socketcan").setLevel(logging.ERROR)

import time
from dataclasses import dataclass

import draccus
import numpy as np
import torch

from lerobot.utils.errors import DeviceNotConnectedError
from lerobot.robots import (  # noqa: F401
    piper_follower,
    RobotConfig,
    make_robot_from_config,
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset


BANNER = r"""
[PiperFollowerEndEffector endpose 数据集回放]
- 从 LeRobot 数据集读取 14 维末端绝对位姿 action
- 直接通过 robot.send_action() 下发到机器人
"""


@dataclass
class RunConfig:
    # 机器人配置（照抄 hil-serl 的写法）
    robot: RobotConfig

    # ===== 数据集相关 =====
    dataset_root: str = "/home/agilex-home/agilex/lerobot_hil-serl/dataset/your_dataset_root"
    dataset_repo_id: str = "local/your_repo_id"

    # ===== 回放区间（按全局 index）=====
    start_index: int = 0    # 从第几个样本开始
    max_steps: int = 0      # 回放多少步；0 表示一直到数据集结束

    # ===== 控制参数 =====
    rate_hz: float = 30.0   # 发送频率
    print_every: int = 10   # 每多少步打印一次日志


@draccus.wrap()
def main(cfg: RunConfig):
    # 1. 连机器人
    robot = make_robot_from_config(cfg.robot)
    robot.connect()
    print(BANNER)

    dt = 1.0 / max(1.0, float(cfg.rate_hz))

    # 2. 加载数据集
    dataset = LeRobotDataset(repo_id=cfg.dataset_repo_id, root=cfg.dataset_root)
    n_samples = len(dataset)
    print(f"[INFO] Loaded LeRobotDataset: len={n_samples}")

    # 3. 计算回放范围
    start = max(0, int(cfg.start_index))
    if cfg.max_steps > 0:
        end = min(n_samples, start + int(cfg.max_steps))
    else:
        end = n_samples

    if start >= end:
        print(f"[ERR] 回放范围无效: start_index={start}, end={end}, len={n_samples}")
        robot.disconnect()
        return

    print(f"[INFO] Replaying samples from index [{start}, {end}) at {cfg.rate_hz} Hz")

    step = 0
    try:
        for idx in range(start, end):
            sample = dataset[idx]

            # 取出 action（14 维末端绝对位姿）
            action = sample["action"]
            if isinstance(action, torch.Tensor):
                act = action.detach().cpu().numpy().astype(float)
            else:
                act = np.array(action, dtype=float)

            if act.shape[0] != 14:
                raise ValueError(f"期望 action 维度为 14，实际为 {act.shape}")

            (
                lx, ly, lz,
                lrx, lry, lrz,
                lgrip,
                rx, ry, rz,
                rrx, rry, rrz,
                rgrip,
            ) = act.tolist()

            action_dict = {
                # 左臂末端
                "left_end_x.value": float(lx),
                "left_end_y.value": float(ly),
                "left_end_z.value": float(lz),
                "left_end_rx.value": float(lrx),
                "left_end_ry.value": float(lry),
                "left_end_rz.value": float(lrz),
                "left_end_gripper.value": float(lgrip),
                # 右臂末端
                "right_end_x.value": float(rx),
                "right_end_y.value": float(ry),
                "right_end_z.value": float(rz),
                "right_end_rx.value": float(rrx),
                "right_end_ry.value": float(rry),
                "right_end_rz.value": float(rrz),
                "right_end_gripper.value": float(rgrip),
            }

            try:
                robot.send_action(action_dict)
            except DeviceNotConnectedError:
                print("[ERR] Robot device disconnected.")
                break

            if cfg.print_every > 0 and (step % cfg.print_every == 0):
                print(
                    f"[{step:05d}] idx={idx} "
                    f"L=({lx:.4f},{ly:.4f},{lz:.4f}) "
                    f"R=({rx:.4f},{ry:.4f},{rz:.4f})"
                )

            step += 1
            time.sleep(dt)

        print("[INFO] Endpose dataset replay finished.")

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
    finally:
        try:
            robot.disconnect()
        except Exception:
            pass
        print("[INFO] Clean exit.")


if __name__ == "__main__":
    main()
"""
python -m lerobot.scripts.test_piper_end_effector_replay \
  --robot.type=piper_follower_end_effector \
  --dataset_root=/home/agilex-home/agilex/lerobot_hil-serl/dataset/20251125T005_lsz001_01_lerobot_endpose \
  --dataset_repo_id=local/test_install_belt_v0.3_resized128_fk_endpose \
  --start_index=0 \
  --max_steps=0 \
  --rate_hz=30
"""