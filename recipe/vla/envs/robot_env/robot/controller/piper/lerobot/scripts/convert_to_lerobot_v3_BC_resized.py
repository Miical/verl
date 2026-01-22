#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 直接将 HDF5 转成用于 RL/BC 训练的 LeRobot 数据集，
# 并在转换过程中把相机图像 resize 到固定分辨率（默认 128x128）。
# 奖励规则：
#   - 如果 HDF5 中存在 /success 数据集，则按 success 数组逐帧作为 reward（0/1）
#   - 否则，默认最后 reward_last_seconds 秒内 reward=1，其余为 0

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

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError as e:
    print(f"[ERROR] 无法导入 LeRobot 库: {e}")
    sys.exit(1)


def get_features_dict(image_height: int = 128, image_width: int = 128, has_gripper: bool = True):
    """
    定义 RL/BC 训练需要的特征（用于 hil-serl）
    基于 record_dataset.py 中的特征定义

    image_height, image_width 用于设定图像特征的 shape，
    例如 (3, 128, 128)
    """
    features = {
        # observation.state: 从 qpos 和 qvel 合并
        "observation.state": {
            "dtype": "float32",
            "shape": (28,),  # qpos(14) + qvel(14)
            "names": None,
        },
        # action: 从原始 action 数据获取
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

    # 添加图像特征（高位相机 + 左腕 + 右腕）
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


def process_episode(
    hdf5_file_path: str,
    dataset: LeRobotDataset,
    task_name: str = "default_task",
    fps: int = 30,
    reward_last_seconds: float = 3.0,
    image_height: int = 128,
    image_width: int = 128,
):
    """
    处理单个 episode 的 HDF5 文件

    奖励规则：
    - 如果存在 /success，则优先：
        rewards[i] = success[i]  (0/1)
      （要求 /success 的长度和时间步数一致）
    - 否则：
        最后 reward_last_seconds 秒内的帧 reward = 1，其他时间步 reward = 0

    同时在这里把图像 resize 到 (image_height, image_width)
    """
    try:
        with h5py.File(hdf5_file_path, "r") as f:
            # 加载图像数据集
            cam_high_ds = f["/observations/images/cam_high"]
            cam_left_wrist_ds = f["/observations/images/cam_left_wrist"]
            cam_right_wrist_ds = f["/observations/images/cam_right_wrist"]

            # 加载关节与动作
            qpos = f["/observations/qpos"][:].astype(np.float32)  # shape: (T, 14)
            qvel = f["/observations/qvel"][:].astype(np.float32)  # shape: (T, 14)
            actions = f["/action"][:].astype(np.float32)          # shape: (T, 14)

            num_steps = qpos.shape[0]
            if num_steps <= 0:
                print(f"警告: {hdf5_file_path} 为空 (0 步)。跳过。")
                return False

            # 合并 qpos 和 qvel 为 observation.state
            obs_state = np.concatenate([qpos, qvel], axis=1)  # shape: (T, 28)

            # ========= 奖励构造：先看是否有 /success =========
            rewards = _build_rewards_with_fallback(
                f=f,
                num_steps=num_steps,
                fps=fps,
                reward_last_seconds=reward_last_seconds,
            )
            # =================================================

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

                # 构建帧数据
                frame_data = {
                    "observation.images.cam_high": pil_high,
                    "observation.images.cam_left_wrist": pil_left,
                    "observation.images.cam_right_wrist": pil_right,

                    "observation.state": obs_state[i].astype(np.float32),
                    "action": actions[i].astype(np.float32),

                    # 使用上面构造好的 rewards（优先 success）
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


def main():
    parser = argparse.ArgumentParser(
        description="将 HDF5 转换为 LeRobot RL/BC 训练数据集（转换过程中直接 resize 图像 & 处理 /success 奖励）"
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
使用示例（直接生成 128x128 的 LeRobot 数据集，自动利用 /success）：

python src/lerobot/scripts/convert_to_lerobot_v3_BC_resized.py \
  --input_dir /home/agilex-home/agilex/lerobot_hil-serl/dataset/one_test \
  --output_dir /home/agilex-home/agilex/lerobot_hil-serl/dataset/one_test_lerobot \
  --repo_id local/test_install_belt_v0.3_resized128 \
  --task_name install_belt \
  --fps 30 \
  --reward_last_seconds 3 \
  --img_height 128 \
  --img_width 128


- 如果 HDF5 中有 /success 且长度为 T，则 reward[i] = success[i]
- 如果没有 /success，则最后 3 秒 reward=1，其余为 0
"""
