#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
将绝对位姿 txt 转换为增量位姿 txt

输入文件格式：
每一行 6 个浮点数，用逗号分隔：
x, y, z, rx, ry, rz

输出文件：
同样每一行 6 个浮点数（增量），用逗号分隔。
第一行增量设为 0 0 0 0 0 0。
"""

import numpy as np
from pathlib import Path

# 你可以改成自己的路径
INPUT_PATH = Path("/home/agilex-home/agilex/lerobot_hil-serl/src/lerobot/motors/piper/left_ee_poses.txt")
OUTPUT_PATH = Path("/home/agilex-home/agilex/lerobot_hil-serl/src/lerobot/motors/piper/left_ee_delta_poses.txt")


def load_poses(path: Path) -> np.ndarray:
    """从 txt 文件读取位姿，返回 (N, 6) 的 numpy 数组。"""
    poses = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 按逗号拆分并转成 float
            nums = [float(x) for x in line.split(",")]
            if len(nums) != 6:
                raise ValueError(f"行格式错误（不是 6 个数）: {line}")
            poses.append(nums)
    if not poses:
        raise ValueError("输入文件为空")
    return np.array(poses, dtype=float)


def compute_deltas(poses: np.ndarray) -> np.ndarray:
    """
    计算增量：
    delta[0] = 0
    delta[i] = poses[i] - poses[i-1]
    """
    deltas = np.zeros_like(poses)
    deltas[1:] = poses[1:] - poses[:-1]
    return deltas


def save_poses(path: Path, poses: np.ndarray) -> None:
    """把 (N, 6) 的数组保存到 txt，每行 6 个数，用逗号分隔。"""
    with path.open("w", encoding="utf-8") as f:
        for row in poses:
            line = ",".join(f"{v:.9f}" for v in row)
            f.write(line + "\n")


def main():
    print(f"读取绝对位姿：{INPUT_PATH}")
    poses = load_poses(INPUT_PATH)
    print(f"共有 {poses.shape[0]} 行位姿")

    deltas = compute_deltas(poses)

    print(f"保存增量位姿到：{OUTPUT_PATH}")
    save_poses(OUTPUT_PATH, deltas)
    print("完成 ✅")


if __name__ == "__main__":
    main()
