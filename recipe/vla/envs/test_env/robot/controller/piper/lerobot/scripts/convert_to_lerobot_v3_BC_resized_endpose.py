#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 直接将 HDF5 转成用于 RL/BC 训练的 LeRobot 数据集。
# 本版本的特点：
#   1. 使用 /observations/qpos 通过 URDF + FK 计算左右臂的“绝对末端位姿”作为 observation.state
#   2. 使用 /action 通过 URDF + FK 计算左右臂的“绝对末端位姿”作为 action
#      两者的维度均为 14:
#        [l_x, l_y, l_z, l_rx, l_ry, l_rz, l_grip,
#         r_x, r_y, r_z, r_rx, r_ry, r_rz, r_grip]
#   3. 图像在转换时 resize，并以 video 形式保存（use_videos=True）
#   4. 奖励规则：
#        - 如果 HDF5 中存在 /success 数据集，则按 success 数组逐帧作为 reward（0/1）
#        - 否则，默认最后 reward_last_seconds 秒内 reward=1，其余为 0

import h5py
import numpy as np
from PIL import Image
import glob
import os
import argparse
from tqdm import tqdm
import sys
from pathlib import Path
import shutil
import math

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.model.kinematics import RobotKinematics
except ImportError as e:
    print(f"[ERROR] 无法导入 LeRobot 库: {e}")
    sys.exit(1)


# ======================
#   FK / 欧拉角工具函数
# ======================

def _rotmat_to_rpy_zyx(R: np.ndarray) -> np.ndarray:
    """
    与测试脚本一致的欧拉角转换：
        - 输入: 3x3 旋转矩阵 R
        - 输出: [rx, ry, rz] = [roll, pitch, yaw] (弧度)
      约定: Z-Y-X (yaw-pitch-roll)，返回顺序 rx, ry, rz
    """
    r20 = -R[2, 0]
    r20_clamped = float(np.clip(r20, -1.0, 1.0))
    pitch = math.asin(r20_clamped)

    cos_pitch = math.cos(pitch)
    if abs(cos_pitch) < 1e-6:
        # 退化情况（接近 ±90°）
        roll = 0.0
        yaw = math.atan2(-R[0, 1], R[1, 1])
    else:
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(R[1, 0], R[0, 0])

    return np.array([roll, pitch, yaw], dtype=float)


# ======================
#   features 定义
# ======================

def get_features_dict(image_height: int = 128, image_width: int = 128):
    """
    定义 RL/BC 训练需要的特征（用于 hil-serl）

    - observation.state: 14 维末端绝对位姿（由 /observations/qpos FK 得到）
    - action: 14 维末端绝对位姿（由 /action FK 得到）
    """
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (14,),
            "names": None,
        },
        "action": {
            "dtype": "float32",
            "shape": (14,),
            "names": None,
        },
        "next.reward": {
            "dtype": "float32",
            "shape": (1,),
            "names": None,
        },
        "next.done": {
            "dtype": "bool",
            "shape": (1,),
            "names": None,
        },
        "complementary_info.discrete_penalty": {
            "dtype": "float32",
            "shape": (1,),
            "names": ["discrete_penalty"],
        },
    }

    # 图像特征（高位相机 + 左腕 + 右腕），以 video 方式保存
    img_shape = (3, image_height, image_width)
    img_names = ["channels", "height", "width"]

    features["observation.images.cam_high"] = {
        "dtype": "video",
        "shape": img_shape,
        "names": img_names,
    }
    features["observation.images.cam_left_wrist"] = {
        "dtype": "video",
        "shape": img_shape,
        "names": img_names,
    }
    features["observation.images.cam_right_wrist"] = {
        "dtype": "video",
        "shape": img_shape,
        "names": img_names,
    }

    return features


def _bgr_to_rgb_uint8(img: np.ndarray) -> np.ndarray:
    """HDF5 里是 OpenCV BGR，这里转成 RGB 再给 PIL"""
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    if img.ndim == 3 and img.shape[-1] == 3:
        img = img[..., ::-1]  # BGR -> RGB
    return img


def _build_rewards_with_fallback(
    f: h5py.File,
    num_steps: int,
    fps: int,
    reward_last_seconds: float,
) -> np.ndarray:
    """
    优先使用 HDF5 中的 /success 数据集：
      - 如果存在 /success 且长度和 num_steps 一致，则逐帧使用 success 作为 reward
      - 否则，退回到“最后 reward_last_seconds 秒 为 1”的规则
    """
    # 默认规则：最后 reward_last_seconds 秒为 1，其余为 0
    def _last_seconds_rule() -> np.ndarray:
        rewards = np.zeros(num_steps, dtype=np.float32)
        n_last = int(reward_last_seconds * fps)
        n_last = min(num_steps, n_last)
        if n_last > 0:
            rewards[-n_last:] = 1.0
        return rewards

    # 如果存在 /success，优先尝试使用它
    if "/success" in f:
        try:
            success = f["/success"][:]
            success = np.array(success).astype(np.float32)

            # 常见情况：success 为逐帧 0/1，shape = (T,) 或 (T,1)
            if success.ndim == 1 and success.shape[0] == num_steps:
                rewards = success
            elif success.ndim == 2 and success.shape[0] == num_steps and success.shape[1] == 1:
                rewards = success[:, 0]
            else:
                # 形状对不上，回退到时间规则
                print(
                    f"[WARN] /success shape={success.shape} 与 num_steps={num_steps} 不匹配，"
                    f"使用默认最后 {reward_last_seconds} 秒 reward=1 的规则。"
                )
                rewards = _last_seconds_rule()

            # clip 到 [0,1]，防止奇怪值
            rewards = np.clip(rewards, 0.0, 1.0).astype(np.float32)
            return rewards

        except Exception as e:
            print(
                f"[WARN] 读取 /success 失败（{e}），"
                f"使用默认最后 {reward_last_seconds} 秒 reward=1 的规则。"
            )
            return _last_seconds_rule()

    # 如果根本没有 /success，就用最后 N 秒规则
    return _last_seconds_rule()


def _fk_endpose_from_joints(
    q_all: np.ndarray,
    left_kin: RobotKinematics,
    right_kin: RobotKinematics,
    left_joint_indices,
    right_joint_indices,
    left_gripper_index: int,
    right_gripper_index: int,
) -> np.ndarray:
    """
    给定一帧 14 维关节角向量 q_all（单位：rad），通过 FK 计算 14 维末端绝对位姿：
        [l_x, l_y, l_z, l_rx, l_ry, l_rz, l_grip,
         r_x, r_y, r_z, r_rx, r_ry, r_rz, r_grip]
    """
    # 左臂关节（弧度 -> 度）
    ql_rad = q_all[left_joint_indices].astype(float)
    ql_deg = np.rad2deg(ql_rad)
    T_l = left_kin.forward_kinematics(ql_deg)
    p_l = T_l[:3, 3]
    rpy_l = _rotmat_to_rpy_zyx(T_l[:3, :3])  # (rx, ry, rz), rad
    l_grip = float(q_all[left_gripper_index])

    # 右臂关节（弧度 -> 度）
    qr_rad = q_all[right_joint_indices].astype(float)
    qr_deg = np.rad2deg(qr_rad)
    T_r = right_kin.forward_kinematics(qr_deg)
    p_r = T_r[:3, 3]
    rpy_r = _rotmat_to_rpy_zyx(T_r[:3, :3])
    r_grip = float(q_all[right_gripper_index])

    endpose = np.array(
        [
            p_l[0], p_l[1], p_l[2],
            rpy_l[0], rpy_l[1], rpy_l[2],
            l_grip,
            p_r[0], p_r[1], p_r[2],
            rpy_r[0], rpy_r[1], rpy_r[2],
            r_grip,
        ],
        dtype=np.float32,
    )
    return endpose


def process_episode(
    hdf5_file_path: str,
    dataset: LeRobotDataset,
    left_kin: RobotKinematics,
    right_kin: RobotKinematics,
    left_joint_indices,
    right_joint_indices,
    left_gripper_index: int,
    right_gripper_index: int,
    task_name: str = "default_task",
    fps: int = 30,
    reward_last_seconds: float = 3.0,
    image_height: int = 128,
    image_width: int = 128,
):
    """
    处理单个 episode 的 HDF5 文件

    - observation.state：由 /observations/qpos 取关节 → FK → endpose(14D)
    - action：           由 /action 取关节 → FK → endpose(14D)

    奖励：
    - 如果存在 /success，则优先使用；
    - 否则默认最后 reward_last_seconds 秒 reward=1，其余为 0
    """
    try:
        with h5py.File(hdf5_file_path, "r") as f:
            # 加载图像数据集
            cam_high_ds = f["/observations/images/cam_high"]
            cam_left_wrist_ds = f["/observations/images/cam_left_wrist"]
            cam_right_wrist_ds = f["/observations/images/cam_right_wrist"]

            # 加载关节（单位：弧度）
            qpos = f["/observations/qpos"][:].astype(np.float32)   # shape: (T, 14)
            actions_joints = f["/action"][:].astype(np.float32)    # shape: (T, 14)

            num_steps = qpos.shape[0]
            if num_steps <= 0:
                print(f"警告: {hdf5_file_path} 为空 (0 步)。跳过。")
                return False

            if actions_joints.shape[0] != num_steps:
                print(
                    f"[WARN] /action 长度 {actions_joints.shape[0]} 与 qpos 长度 {num_steps} 不一致，"
                    f"将按 min(T_qpos, T_action) 截断。"
                )
                num_steps = min(num_steps, actions_joints.shape[0])

            # ========= 奖励构造：先看是否有 /success =========
            rewards = _build_rewards_with_fallback(
                f=f,
                num_steps=num_steps,
                fps=fps,
                reward_last_seconds=reward_last_seconds,
            )
            # =================================================

            # 每一帧：图像 + obs_endpose + act_endpose + reward/done
            for i in range(num_steps):
                # 从 HDF5 里读出图像 (H, W, C)，当前是 BGR
                img_high = cam_high_ds[i]
                img_left = cam_left_wrist_ds[i]
                img_right = cam_right_wrist_ds[i]

                # BGR -> RGB，并确保 uint8
                img_high = _bgr_to_rgb_uint8(img_high)
                img_left = _bgr_to_rgb_uint8(img_left)
                img_right = _bgr_to_rgb_uint8(img_right)

                # 转成 PIL 图像并 resize 到 (image_width, image_height)
                # 注意：PIL 的 size 是 (width, height)
                pil_high = Image.fromarray(img_high).resize(
                    (image_width, image_height), resample=Image.BILINEAR
                )
                pil_left = Image.fromarray(img_left).resize(
                    (image_width, image_height), resample=Image.BILINEAR
                )
                pil_right = Image.fromarray(img_right).resize(
                    (image_width, image_height), resample=Image.BILINEAR
                )

                # ========= 通过 FK 构造 observation.state & action =========
                q_state = qpos[i]           # 来自 /observations/qpos
                q_action = actions_joints[i]  # 来自 /action

                obs_endpose = _fk_endpose_from_joints(
                    q_state,
                    left_kin,
                    right_kin,
                    left_joint_indices,
                    right_joint_indices,
                    left_gripper_index,
                    right_gripper_index,
                )
                act_endpose = _fk_endpose_from_joints(
                    q_action,
                    left_kin,
                    right_kin,
                    left_joint_indices,
                    right_joint_indices,
                    left_gripper_index,
                    right_gripper_index,
                )
                # ======================================================

                frame_data = {
                    "observation.images.cam_high": pil_high,
                    "observation.images.cam_left_wrist": pil_left,
                    "observation.images.cam_right_wrist": pil_right,

                    "observation.state": obs_endpose,
                    "action": act_endpose,

                    "next.reward": np.array([rewards[i]], dtype=np.float32),

                    # done：只有最后一个时间步是 True
                    "next.done": np.array([False], dtype=bool),
                    "complementary_info.discrete_penalty": np.array([0.0], dtype=np.float32),
                    "task": task_name,
                }

                if i == num_steps - 1:
                    frame_data["next.done"] = np.array([True], dtype=bool)

                dataset.add_frame(frame_data)

        # 一个 episode 完成，写入 meta/episodes
        dataset.save_episode()
        return True

    except Exception as e:
        print(f"\n[ERROR] 处理文件 {hdf5_file_path} 时发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False


# ==============
#   主入口
# ==============

def main():
    parser = argparse.ArgumentParser(
        description="将 HDF5 转换为 LeRobot RL/BC 数据集（图像 resize & /success 奖励 & FK 末端绝对位姿 obs/action）"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="输入 HDF5 文件所在目录（包含 episode_*.hdf5）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="输出 LeRobot 数据集保存路径（例如 /mnt/.../install_belt_v0.3）",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="数据集仓库 ID（例如 local/test_install_belt_v0.3）",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="数据集帧率（采集是 30Hz，这里也默认按 30Hz）",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="default_task",
        help="任务名称（写入 frame['task']）",
    )
    parser.add_argument(
        "--reward_last_seconds",
        type=float,
        default=3.0,
        help="当没有 /success 时：最后多少秒的 reward 置为 1（默认 3 秒）",
    )
    parser.add_argument(
        "--img_height",
        type=int,
        default=128,
        help="输出图像高度（默认 128）",
    )
    parser.add_argument(
        "--img_width",
        type=int,
        default=128,
        help="输出图像宽度（默认 128）",
    )

    args = parser.parse_args()

    output_path = Path(args.output_dir)
    if output_path.exists():
        print(f"警告：目标文件夹 {output_path} 已存在，将删除。")
        shutil.rmtree(output_path)

    # ========= 这里开始：FK 相关配置全部写死 =========
    urdf_path = "local_assets/piper.urdf"
    target_frame_name = "joint6"

    left_joint_indices = [0, 1, 2, 3, 4, 5]
    left_gripper_index = 6
    right_joint_indices = [7, 8, 9, 10, 11, 12]
    right_gripper_index = 13

    print(f"URDF 路径: {urdf_path}, 末端连杆: {target_frame_name}")
    print(f"左臂关节索引: {left_joint_indices}, 左夹爪索引: {left_gripper_index}")
    print(f"右臂关节索引: {right_joint_indices}, 右夹爪索引: {right_gripper_index}")
    # =================================================

    # 创建数据集（features 里图像 shape 直接就是 (3, img_height, img_width)）
    features = get_features_dict(image_height=args.img_height, image_width=args.img_width)
    print(f"正在创建 RL/BC 训练数据集...")
    print(f"图像分辨率将被统一成: ({args.img_height}, {args.img_width})")

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=output_path,
        features=features,
        fps=args.fps,
        use_videos=True,
        image_writer_processes=0,
        image_writer_threads=4,
    )

    # 确保启动 image writer
    if not dataset.image_writer:
        dataset.start_image_writer(num_processes=0, num_threads=4)

    # 准备 FK 模型（左右臂各一份）
    left_kin = RobotKinematics(
        urdf_path=urdf_path,
        target_frame_name=target_frame_name,
        joint_names=[f"joint{i+1}" for i in range(len(left_joint_indices))],
    )
    right_kin = RobotKinematics(
        urdf_path=urdf_path,
        target_frame_name=target_frame_name,
        joint_names=[f"joint{i+1}" for i in range(len(right_joint_indices))],
    )

    # 处理 HDF5 文件
    source_files = sorted(glob.glob(os.path.join(args.input_dir, "episode_*.hdf5")))
    if not source_files:
        print(f"未找到 episode_*.hdf5 文件。")
        dataset.finalize()
        return

    print(f"找到 {len(source_files)} 个 episode 文件")

    success_count = 0
    for hdf5_file in tqdm(source_files, desc="转换 Episodes"):
        ok = process_episode(
            hdf5_file,
            dataset,
            left_kin=left_kin,
            right_kin=right_kin,
            left_joint_indices=left_joint_indices,
            right_joint_indices=right_joint_indices,
            left_gripper_index=left_gripper_index,
            right_gripper_index=right_gripper_index,
            task_name=args.task_name,
            fps=args.fps,
            reward_last_seconds=args.reward_last_seconds,
            image_height=args.img_height,
            image_width=args.img_width,
        )
        if ok:
            success_count += 1

    if success_count == 0:
        print("无有效数据。")
        dataset.finalize()
        return

    print(f"\n成功转换 {success_count} 个 episode。")

    # finalize 会关闭 image writer / metadata writer，写入 meta/episodes 等信息
    dataset.finalize()
    print(f"\n✅ 数据集已保存至: {output_path}")
    print(f"   repo_id = {args.repo_id}")
    print(f"   图像分辨率 = (3, {args.img_height}, {args.img_width})")


if __name__ == "__main__":
    main()


"""
python src/lerobot/scripts/convert_to_lerobot_v3_BC_resized_endpose.py \
  --input_dir /home/agilex-home/agilex/lerobot_hil-serl/dataset/20251125T005_lsz001_01 \
  --output_dir /home/agilex-home/agilex/lerobot_hil-serl/dataset/20251125T005_lsz001_01_lerobot_endpose \
  --repo_id local/test_install_belt_v0.3_resized128_fk_endpose \
  --task_name install_belt \
  --fps 30 \
  --reward_last_seconds 3 \
  --img_height 128 \
  --img_width 128
"""