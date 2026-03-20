#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
[PiperFollowerEndEffector FK 数据集回放测试]

三种用法：
1）ctrl_mode="position"：
    - 用关节空间回放（直接发关节 Goal_Position）
    - 同时记录每步 get_observation() 的末端位姿到文件（obs 轨迹）
    - 同时记录同一帧关节通过 RobotKinematics 做 FK 得到的末端位姿（FK 轨迹）

2）ctrl_mode="endpose"：
    - replay_source="dataset"：用数据集 + 第三方 FK 的末端位姿（绝对值）回放
    - replay_source="obs"：用上一阶段保存的 obs 末端位姿回放（原样复现）

3）ctrl_mode="endpose_delta"：
    - 仅支持 replay_source="dataset"
    - 第 0 帧：用 FK 的绝对末端位姿对齐机器人
    - 后续帧：用相邻两帧 FK 结果计算 Δx,Δy,Δz,Δroll,Δpitch,Δyaw
      作为“增量式控制”，在上一帧目标基础上累加得到新目标再下发
    - 同时记录每一帧的左右臂增量到 .npz 文件（共 12 维：左6 + 右6）
"""

import logging
logging.getLogger("can.interfaces.socketcan").setLevel(logging.ERROR)

import time
import math
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
from lerobot.model.kinematics import RobotKinematics


BANNER = r"""
[PiperFollowerEndEffector FK 数据集回放测试]
- ctrl_mode="position"：按数据集关节值驱动机器人，同时记录 obs & FK 末端位姿
- ctrl_mode="endpose"：按 FK 末端 / 记录的 obs 末端，做绝对末端位姿回放
- ctrl_mode="endpose_delta"：第 0 帧绝对定位，后续用 FK 末端增量做增量式末端控制，并记录增量 .npz
"""


def rotmat_to_rpy_zyx(R: np.ndarray) -> tuple[float, float, float]:
    """
    将旋转矩阵 R 转成欧拉角 (rx, ry, rz)，单位 rad
    使用 Z-Y-X (yaw-pitch-roll) 约定：
      - rz: yaw (绕 Z)
      - ry: pitch (绕 Y)
      - rx: roll (绕 X)
    返回顺序按 EndPoseCtrl 习惯：rx, ry, rz
    """
    # 避免数值误差
    r20 = -R[2, 0]
    r20_clamped = max(min(r20, 1.0), -1.0)
    pitch = math.asin(r20_clamped)

    cos_pitch = math.cos(pitch)
    if abs(cos_pitch) < 1e-6:
        # 退化情况（接近 ±90°），简单处理
        roll = 0.0
        yaw = math.atan2(-R[0, 1], R[1, 1])
    else:
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(R[1, 0], R[0, 0])

    return float(roll), float(pitch), float(yaw)


def wrap_to_pi(x: np.ndarray) -> np.ndarray:
    """
    将角度（rad）wrap 到 (-pi, pi] 区间，支持标量或数组。
    """
    return (x + np.pi) % (2.0 * np.pi) - np.pi


@dataclass
class RunConfig:
    # 机器人配置
    robot: RobotConfig

    # ===== 数据集相关 =====
    dataset_root: str = "/home/agilex-home/agilex/lerobot_hil-serl/dataset/20251125T005_lsz001_01_lerobot"
    dataset_repo_id: str = "local/test_hil_serl_v0.5_cropped_v2"

    # ===== FK / URDF 相关（和 env.processor.inverse_kinematics 保持一致） =====
    urdf_path: str = "local_assets/piper.urdf"
    target_frame_name: str = "joint6"   # 对应 config 里的 joint6

    # observation.state 中关节索引（和 JointToEEDeltaDataset 一样）
    left_joint_indices: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
    right_joint_indices: tuple[int, ...] = (7, 8, 9, 10, 11, 12)

    # endpose 模式下是否用 FK 的姿态（否则只用 FK 位置，姿态用 obs）
    use_fk_orientation: bool = True

    # ===== 回放控制 =====
    episode_index: int = 2     # 回放哪个 episode（你现在手动用 642~967 这段）
    start_step: int = 0        # 从 episode 中第几个 step 开始（对 dataset 源有用）
    max_steps: int = 0         # 0 表示一直到该 episode 结束 / 或 obs 日志结束

    rate_hz: float = 30.0      # 发送频率
    print_every: int = 10      # 每多少步打印一次

    # 控制模式：
    #   "position"：关节空间回放 + 记录 obs & FK 末端
    #   "endpose"：末端空间回放（绝对位姿）
    #   "endpose_delta"：末端空间增量回放（第 0 帧绝对 + 后续增量）
    ctrl_mode: str = "position"

    # 末端回放的数据来源：
    #   "dataset"：用数据集关节 + FK 得到的末端
    #   "obs"：用之前保存的 obs 末端（原样复现）
    replay_source: str = "dataset"

    # 保存 / 读取 obs 末端日志的路径（里面有 {episode} 占位）
    obs_log_path: str = "/home/agilex-home/agilex/lerobot_hil-serl/src/lerobot/scripts/piper_obs_endpose_ep{episode}.npz"

    # 保存 FK 末端日志的路径（position 模式）
    fk_log_path: str = "/home/agilex-home/agilex/lerobot_hil-serl/src/lerobot/scripts/piper_fk_endpose_ep{episode}.npz"

    # 保存 FK 增量日志的路径（endpose_delta 模式）
    fk_delta_log_path: str = "/home/agilex-home/agilex/lerobot_hil-serl/src/lerobot/scripts/piper_fk_delta_ep{episode}.npz"


@draccus.wrap()
def main(cfg: RunConfig):
    # 1. 连机器人
    robot = make_robot_from_config(cfg.robot)
    robot.connect()
    print(BANNER)

    dt = 1.0 / max(1.0, float(cfg.rate_hz))

    # 2. 加载数据集
    dataset = LeRobotDataset(repo_id=cfg.dataset_repo_id, root=cfg.dataset_root)
    print(f"[INFO] Loaded LeRobotDataset: len={len(dataset)}")

    # 3. 准备 FK 模型（与 transforms.py 完全一致的配置）
    left_kin = RobotKinematics(
        urdf_path=cfg.urdf_path,
        target_frame_name=cfg.target_frame_name,
        joint_names=[f"joint{i+1}" for i in range(len(cfg.left_joint_indices))],
    )
    right_kin = RobotKinematics(
        urdf_path=cfg.urdf_path,
        target_frame_name=cfg.target_frame_name,
        joint_names=[f"joint{i+1}" for i in range(len(cfg.right_joint_indices))],
    )

    # ===== episode 索引：当前手动使用 [642, 968) 这一段 =====
    # 如果之后想按 episode_index 来切片，可以把下面改回按 episode_index 过滤
    episode_indices = [i for i in range(642, 968)]
    print(f"[INFO] Episode {cfg.episode_index}: {len(episode_indices)} frames total (using indices 642~967).")

    # 应用 start_step / max_steps
    if cfg.start_step > 0:
        episode_indices = episode_indices[cfg.start_step:]
    if cfg.max_steps > 0:
        episode_indices = episode_indices[: cfg.max_steps]

    print(f"[INFO] Replaying {len(episode_indices)} steps from episode {cfg.episode_index}.")

    # ===== 如果 endpose+obs 模式，需要预加载 obs 日志 =====
    obs_log_path = cfg.obs_log_path.format(episode=cfg.episode_index)
    obs_log_data = None
    if cfg.ctrl_mode == "endpose" and cfg.replay_source == "obs":
        try:
            loaded = np.load(obs_log_path)
            obs_log_data = loaded["data"]  # shape: [N, 12]
            print(f"[INFO] Loaded obs endpose log from {obs_log_path}, shape={obs_log_data.shape}")
        except FileNotFoundError:
            print(f"[ERR] Obs log file not found: {obs_log_path}")
            robot.disconnect()
            return

    # ===== position 模式下要记录 obs & FK 末端 =====
    # 每一行: [lx,ly,lz,lrx,lry,lrz,  rx,ry,rz, rrx,rry,rrz]
    recorded_obs_endpose: list[list[float]] = []
    recorded_fk_endpose: list[list[float]] = []

    # ===== endpose_delta 模式下要记录增量 =====
    # 每一行: [l_dx,l_dy,l_dz,l_droll,l_dpitch,l_dyaw,  r_dx,...,r_dyaw]
    recorded_fk_delta: list[list[float]] = []

    # endpose_delta 模式下，用来保存上一帧“目标末端位姿”（左右各 6 维）
    last_l_pose: np.ndarray | None = None  # [x,y,z, roll,pitch,yaw]
    last_r_pose: np.ndarray | None = None

    step = 0
    try:
        for local_step, idx in enumerate(episode_indices):
            sample = dataset[idx]
            state = sample["observation.state"]
            if not isinstance(state, torch.Tensor):
                state = torch.tensor(state, dtype=torch.float32)

            # ===== 左右臂关节（弧度） =====
            ql_rad = (
                state[list(cfg.left_joint_indices)]
                .detach()
                .cpu()
                .numpy()
                .astype(float)
            )
            qr_rad = (
                state[list(cfg.right_joint_indices)]
                .detach()
                .cpu()
                .numpy()
                .astype(float)
            )

            # ========= 关节空间控制模式 =========
            if cfg.ctrl_mode == "position":
                # 设置关节模式（你之前的写法）
                robot.bus_left.interface.ModeCtrl(
                    ctrl_mode=0x01, move_mode=0x01, move_spd_rate_ctrl=50, is_mit_mode=0x00
                )
                robot.bus_right.interface.ModeCtrl(
                    ctrl_mode=0x01, move_mode=0x01, move_spd_rate_ctrl=50, is_mit_mode=0x00
                )
                left_q = {
                    "left_joint_1": float(ql_rad[0]), "left_joint_2": float(ql_rad[1]),
                    "left_joint_3": float(ql_rad[2]), "left_joint_4": float(ql_rad[3]),
                    "left_joint_5": float(ql_rad[4]), "left_joint_6": float(ql_rad[5]),
                    "left_gripper": 0.0035,
                }
                right_q = {
                    "right_joint_1": float(qr_rad[0]), "right_joint_2": float(qr_rad[1]),
                    "right_joint_3": float(qr_rad[2]), "right_joint_4": float(qr_rad[3]),
                    "right_joint_5": float(qr_rad[4]), "right_joint_6": float(qr_rad[5]),
                    "right_gripper": 0.0035,
                }
                robot.bus_left.sync_write("Goal_Position", left_q)
                robot.bus_right.sync_write("Goal_Position", right_q)

                # === 记录这一帧的 第三方 FK 末端位姿 ===
                ql_deg = np.rad2deg(ql_rad)
                qr_deg = np.rad2deg(qr_rad)
                T_l_fk = left_kin.forward_kinematics(ql_deg)
                T_r_fk = right_kin.forward_kinematics(qr_deg)

                lx_fk, ly_fk, lz_fk = T_l_fk[:3, 3].tolist()
                rx_fk, ry_fk, rz_fk = T_r_fk[:3, 3].tolist()

                lrx_fk, lry_fk, lrz_fk = rotmat_to_rpy_zyx(T_l_fk[:3, :3])
                rrx_fk, rry_fk, rrz_fk = rotmat_to_rpy_zyx(T_r_fk[:3, :3])

                recorded_fk_endpose.append(
                    [
                        lx_fk, ly_fk, lz_fk, lrx_fk, lry_fk, lrz_fk,
                        rx_fk, ry_fk, rz_fk, rrx_fk, rry_fk, rrz_fk,
                    ]
                )

                # === 记录这一帧的 obs 末端位姿 ===
                obs = robot.get_observation()
                if obs is None:
                    print("[WARN] get_observation() returned None, skip logging.")
                else:
                    try:
                        lx = float(obs["left_end_x.value"])
                        ly = float(obs["left_end_y.value"])
                        lz = float(obs["left_end_z.value"])
                        lrx = float(obs["left_end_rx.value"])
                        lry = float(obs["left_end_ry.value"])
                        lrz = float(obs["left_end_rz.value"])

                        rx = float(obs["right_end_x.value"])
                        ry = float(obs["right_end_y.value"])
                        rz = float(obs["right_end_z.value"])
                        rrx = float(obs["right_end_rx.value"])
                        rry = float(obs["right_end_ry.value"])
                        rrz = float(obs["right_end_rz.value"])

                        recorded_obs_endpose.append(
                            [
                                lx, ly, lz, lrx, lry, lrz,
                                rx, ry, rz, rrx, rry, rrz,
                            ]
                        )
                    except KeyError as e:
                        print(f"[WARN] get_observation() 缺少键 {e}，不记录该步。")

            # ========= 绝对末端位姿控制模式 =========
            elif cfg.ctrl_mode == "endpose":
                if cfg.replay_source == "dataset":
                    # 用 FK 计算当前帧的绝对末端位姿
                    ql_deg = np.rad2deg(ql_rad)
                    qr_deg = np.rad2deg(qr_rad)
                    T_l = left_kin.forward_kinematics(ql_deg)
                    T_r = right_kin.forward_kinematics(qr_deg)

                    lx, ly, lz = T_l[:3, 3].tolist()
                    rx, ry, rz = T_r[:3, 3].tolist()

                    obs = robot.get_observation()
                    if obs is None:
                        print("[WARN] get_observation() returned None, skip this step.")
                        continue

                    if cfg.use_fk_orientation:
                        lrx, lry, lrz = rotmat_to_rpy_zyx(T_l[:3, :3])
                        rrx, rry, rrz = rotmat_to_rpy_zyx(T_r[:3, :3])
                    else:
                        lrx = float(obs["left_end_rx.value"])
                        lry = float(obs["left_end_ry.value"])
                        lrz = float(obs["left_end_rz.value"])
                        rrx = float(obs["right_end_rx.value"])
                        rry = float(obs["right_end_ry.value"])
                        rrz = float(obs["right_end_rz.value"])

                    lgrip = float(obs.get("left_end_gripper.value", 0.5))
                    rgrip = float(obs.get("right_end_gripper.value", 0.5))

                else:
                    # 用 obs 轨迹做原样回放
                    if obs_log_data is None:
                        print("[ERR] obs_log_data is None in endpose+obs mode.")
                        break
                    if local_step >= len(obs_log_data):
                        print("[INFO] Reached end of obs log, stop replay.")
                        break

                    row = obs_log_data[local_step]
                    (
                        lx, ly, lz, lrx, lry, lrz,
                        rx, ry, rz, rrx, rry, rrz,
                    ) = row.tolist()

                    obs = robot.get_observation()
                    if obs is None:
                        lgrip = 0.5
                        rgrip = 0.5
                    else:
                        lgrip = float(obs.get("left_end_gripper.value", 0.5))
                        rgrip = float(obs.get("right_end_gripper.value", 0.5))

                action = {
                    # 左臂
                    "left_end_x.value": float(lx),
                    "left_end_y.value": float(ly),
                    "left_end_z.value": float(lz),
                    "left_end_rx.value": float(lrx),
                    "left_end_ry.value": float(lry),
                    "left_end_rz.value": float(lrz),
                    "left_end_gripper.value": lgrip,
                    # 右臂
                    "right_end_x.value": float(rx),
                    "right_end_y.value": float(ry),
                    "right_end_z.value": float(rz),
                    "right_end_rx.value": float(rrx),
                    "right_end_ry.value": float(rry),
                    "right_end_rz.value": float(rrz),
                    "right_end_gripper.value": rgrip,
                }

                try:
                    robot.send_action(action)
                except DeviceNotConnectedError:
                    print("[ERR] Robot device disconnected.")
                    break

            # ========= 增量式末端位姿控制模式 =========
            elif cfg.ctrl_mode == "endpose_delta":
                if cfg.replay_source != "dataset":
                    print("[WARN] endpose_delta 目前只支持 replay_source='dataset'，已强制使用 dataset。")

                # 1) 用 FK 算当前帧的绝对末端位姿（RPY）
                ql_deg = np.rad2deg(ql_rad)
                qr_deg = np.rad2deg(qr_rad)
                T_l = left_kin.forward_kinematics(ql_deg)
                T_r = right_kin.forward_kinematics(qr_deg)

                lx_cur, ly_cur, lz_cur = T_l[:3, 3].tolist()
                rx_cur, ry_cur, rz_cur = T_r[:3, 3].tolist()
                lrx_cur, lry_cur, lrz_cur = rotmat_to_rpy_zyx(T_l[:3, :3])
                rrx_cur, rry_cur, rrz_cur = rotmat_to_rpy_zyx(T_r[:3, :3])

                cur_l_pose = np.array(
                    [lx_cur, ly_cur, lz_cur, lrx_cur, lry_cur, lrz_cur],
                    dtype=float,
                )
                cur_r_pose = np.array(
                    [rx_cur, ry_cur, rz_cur, rrx_cur, rry_cur, rrz_cur],
                    dtype=float,
                )

                # 2) 计算增量 & 更新目标
                if last_l_pose is None:
                    # 第 0 帧：直接用绝对位姿对齐，不做增量
                    delta_l = np.zeros(6, dtype=float)
                    delta_r = np.zeros(6, dtype=float)
                    target_l_pose = cur_l_pose.copy()
                    target_r_pose = cur_r_pose.copy()
                else:
                    # 增量 = 当前 FK - 上一帧 FK （位置和欧拉角的差值）
                    delta_l = cur_l_pose - last_l_pose
                    delta_r = cur_r_pose - last_r_pose

                    # 对角度部分 wrap 一下，避免跨越 pi 带来的大跳变
                    delta_l[3:] = wrap_to_pi(delta_l[3:])
                    delta_r[3:] = wrap_to_pi(delta_r[3:])

                    # 目标 = 上一帧目标 + 增量（理想情况下等于当前 FK）
                    target_l_pose = last_l_pose + delta_l
                    target_r_pose = last_r_pose + delta_r

                # 更新“上一帧目标”
                last_l_pose = target_l_pose
                last_r_pose = target_r_pose

                # 记录增量（左右共 12 维）
                recorded_fk_delta.append(
                    np.concatenate([delta_l, delta_r]).tolist()
                )

                # 3) 下发目标末端位姿
                lx, ly, lz, lrx, lry, lrz = target_l_pose.tolist()
                rx, ry, rz, rrx, rry, rrz = target_r_pose.tolist()

                obs = robot.get_observation()
                if obs is None:
                    lgrip = 0.5
                    rgrip = 0.5
                else:
                    lgrip = float(obs.get("left_end_gripper.value", 0.5))
                    rgrip = float(obs.get("right_end_gripper.value", 0.5))

                action = {
                    # 左臂
                    "left_end_x.value": float(lx),
                    "left_end_y.value": float(ly),
                    "left_end_z.value": float(lz),
                    "left_end_rx.value": float(lrx),
                    "left_end_ry.value": float(lry),
                    "left_end_rz.value": float(lrz),
                    "left_end_gripper.value": lgrip,
                    # 右臂
                    "right_end_x.value": float(rx),
                    "right_end_y.value": float(ry),
                    "right_end_z.value": float(rz),
                    "right_end_rx.value": float(rrx),
                    "right_end_ry.value": float(rry),
                    "right_end_rz.value": float(rrz),
                    "right_end_gripper.value": rgrip,
                }

                try:
                    robot.send_action(action)
                except DeviceNotConnectedError:
                    print("[ERR] Robot device disconnected.")
                    break

            else:
                raise ValueError(f"Unknown ctrl_mode: {cfg.ctrl_mode}")

            # ===== 打印 & sleep =====
            if cfg.print_every > 0 and (step % cfg.print_every == 0):
                if cfg.ctrl_mode == "position":
                    lx_fk, ly_fk, lz_fk = recorded_fk_endpose[-1][:3]
                    print(
                        f"[{step:05d}] idx={idx} "
                        f"(JointCtrl) FK_L=({lx_fk:.4f},{ly_fk:.4f},{lz_fk:.4f})"
                    )
                elif cfg.ctrl_mode == "endpose":
                    print(
                        f"[{step:05d}] idx={idx} "
                        f"(EndPose {cfg.replay_source}) "
                        f"L=({lx:.4f},{ly:.4f},{lz:.4f}) "
                        f"R=({rx:.4f},{ry:.4f},{rz:.4f})"
                    )
                elif cfg.ctrl_mode == "endpose_delta":
                    dl = recorded_fk_delta[-1][:6]
                    dr = recorded_fk_delta[-1][6:]
                    print(
                        f"[{step:05d}] idx={idx} (EndPoseDelta) "
                        f"dL=({dl[0]:.4e},{dl[1]:.4e},{dl[2]:.4e},{dl[3]:.4e},{dl[4]:.4e},{dl[5]:.4e}) "
                        f"dR=({dr[0]:.4e},{dr[1]:.4e},{dr[2]:.4e},{dr[3]:.4e},{dr[4]:.4e},{dr[5]:.4e})"
                    )

            step += 1
            time.sleep(dt)

        print("[INFO] FK dataset replay finished.")

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
    finally:
        # ===== 如果是 position 模式，把 obs 和 FK 两个 endpose 轨迹都存盘 =====
        if cfg.ctrl_mode == "position":
            if len(recorded_obs_endpose) > 0:
                obs_arr = np.array(recorded_obs_endpose, dtype=np.float32)  # [T, 12]
                obs_save_path = cfg.obs_log_path.format(episode=cfg.episode_index)
                np.savez(obs_save_path, data=obs_arr)
                print(f"[INFO] Saved obs endpose log to {obs_save_path}, shape={obs_arr.shape}")

            if len(recorded_fk_endpose) > 0:
                fk_arr = np.array(recorded_fk_endpose, dtype=np.float32)   # [T, 12]
                fk_save_path = cfg.fk_log_path.format(episode=cfg.episode_index)
                np.savez(fk_save_path, data=fk_arr)
                print(f"[INFO] Saved FK endpose log to {fk_save_path}, shape={fk_arr.shape}")

        # ===== 如果是 endpose_delta 模式，把 FK 增量轨迹存盘 =====
        if cfg.ctrl_mode == "endpose_delta" and len(recorded_fk_delta) > 0:
            delta_arr = np.array(recorded_fk_delta, dtype=np.float32)  # [T, 12]
            delta_save_path = cfg.fk_delta_log_path.format(episode=cfg.episode_index)
            np.savez(delta_save_path, data=delta_arr)
            print(f"[INFO] Saved FK delta log to {delta_save_path}, shape={delta_arr.shape}")

        try:
            robot.disconnect()
        except Exception:
            pass
        print("[INFO] Clean exit.")


if __name__ == "__main__":
    main()


"""
示例：

1）先关节空间回放 + 记录 obs & FK：
python -m lerobot.scripts.test_piper_fk_dataset_replay \
  --robot.type=piper_follower_end_effector \
  --ctrl_mode=position \
  --episode_index=2

2）再末端空间回放（用刚才记录的 obs 末端，绝对值）：
python -m lerobot.scripts.test_piper_fk_dataset_replay \
  --robot.type=piper_follower_end_effector \
  --ctrl_mode=endpose \
  --replay_source=obs \
  --episode_index=2

3）末端空间回放（用数据集 + FK，绝对值）：
python -m lerobot.scripts.test_piper_fk_dataset_replay \
  --robot.type=piper_follower_end_effector \
  --ctrl_mode=endpose \
  --replay_source=dataset \
  --episode_index=2

4）末端空间“增量式”回放（第 0 帧绝对 + 后续增量）：
python -m lerobot.scripts.test_piper_fk_dataset_replay \
  --robot.type=piper_follower_end_effector \
  --ctrl_mode=endpose_delta \
  --replay_source=dataset \
  --episode_index=2
"""
