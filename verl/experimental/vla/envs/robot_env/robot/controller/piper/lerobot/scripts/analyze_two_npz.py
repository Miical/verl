#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
简单分析两个/三个 npz 文件：
- npz1: 比如 obs 末端位姿，data 形状 [T, 12]
- npz2: 比如 FK 末端位姿，data 形状 [T, 12]
- delta_npz(可选): 比如 FK 增量（endpose_delta 模式保存），data 形状 [T, 12]

功能：
1. 打印 npz1 和 npz2 的数据值，以及差值 diff_abs = data1 - data2
2. 对每个自由度画一张“综合图”（一个文件）：
   - 上半图：npz1 / npz2 / （可选）recon 的绝对值曲线
   - 下半图：diff_abs / （可选）diff_delta 的差值曲线
3. 画左右臂的三维轨迹（同一个坐标系下 npz1、npz2、recon），保存到同一文件夹
"""

import argparse
import os

import numpy as np

# 强制使用非交互后端，避免在没有 GUI 的环境卡住
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def main():
    print("=== analyze_two_npz.py START ===", flush=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--npz1", type=str, required=True, help="第一个 npz 路径（例如 obs）")
    parser.add_argument("--npz2", type=str, required=True, help="第二个 npz 路径（例如 fk）")
    parser.add_argument(
        "--delta_npz",
        type=str,
        default=None,
        help="第三个 npz 路径（可选，增量 FK 末端，例如 fk_delta）",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="diff_plots",
        help="保存图像的文件夹",
    )
    args = parser.parse_args()

    print(f"[INFO] npz1 path: {args.npz1}", flush=True)
    print(f"[INFO] npz2 path: {args.npz2}", flush=True)
    print(f"[INFO] delta_npz path: {args.delta_npz}", flush=True)
    print(f"[INFO] out_dir: {args.out_dir}", flush=True)

    # 1. 读取 npz1 / npz2
    print("[INFO] Loading npz1...", flush=True)
    npz1 = np.load(args.npz1)
    print(f"[DEBUG] npz1 keys: {list(npz1.keys())}", flush=True)
    arr1 = npz1["data"]

    print("[INFO] Loading npz2...", flush=True)
    npz2 = np.load(args.npz2)
    print(f"[DEBUG] npz2 keys: {list(npz2.keys())}", flush=True)
    arr2 = npz2["data"]

    print(f"[INFO] npz1 shape = {arr1.shape}, npz2 shape = {arr2.shape}", flush=True)

    # 2. 对齐长度（绝对值比较）
    T_abs = min(arr1.shape[0], arr2.shape[0])
    arr1 = arr1[:T_abs]
    arr2 = arr2[:T_abs]
    print(f"[INFO] Using first {T_abs} frames from each npz (absolute comparison).", flush=True)

    # 3. 计算绝对差值
    print("[INFO] Computing diff_abs = npz1 - npz2 ...", flush=True)
    diff_abs = arr1 - arr2

    # 打印数据
    print("\n===== npz1 数据（对齐后的前 T_abs 帧）=====", flush=True)
    print(arr1, flush=True)

    print("\n===== npz2 数据（对齐后的前 T_abs 帧）=====", flush=True)
    print(arr2, flush=True)

    print("\n===== 差值 diff_abs = npz1 - npz2 （前 T_abs 帧）=====", flush=True)
    print(diff_abs, flush=True)

    # 4. 创建输出文件夹
    print("[INFO] Creating output directory if needed...", flush=True)
    os.makedirs(args.out_dir, exist_ok=True)

    num_dims = diff_abs.shape[1]
    if num_dims == 12:
        dof_names = [
            "L_x", "L_y", "L_z",
            "L_rx", "L_ry", "L_rz",
            "R_x", "R_y", "R_z",
            "R_rx", "R_ry", "R_rz",
        ]
    else:
        dof_names = [f"dof_{i}" for i in range(num_dims)]

    x_abs = np.arange(T_abs)

    # ============ 下面是增量式重建部分（可选） ============
    recon = None         # 用增量重建的绝对 FK 轨迹
    diff_delta = None    # recon - arr2 的差值
    x_delta = None

    if args.delta_npz is not None:
        print("\n[INFO] ==== Delta-based reconstruction START ====", flush=True)
        print("[INFO] Loading delta_npz...", flush=True)
        delta_npz = np.load(args.delta_npz)
        print(f"[DEBUG] delta_npz keys: {list(delta_npz.keys())}", flush=True)
        delta = delta_npz["data"]  # [T_delta, D]
        print(f"[INFO] delta shape = {delta.shape}", flush=True)

        T_delta, D_delta = delta.shape
        if D_delta != num_dims:
            print(
                f"[WARN] delta dims ({D_delta}) != npz dims ({num_dims})，"
                f"仅按前 {min(D_delta, num_dims)} 维累加。",
                flush=True,
            )
        D_use = min(D_delta, num_dims)

        # 使用 npz2 的第 0 帧绝对末端位姿作为对齐基准
        T_recon = min(T_delta, arr2.shape[0])
        print(f"[INFO] Reconstructing absolute FK from delta, T_recon = {T_recon}", flush=True)

        recon = np.zeros((T_recon, num_dims), dtype=float)
        recon[0, :] = arr2[0, :]   # 初始对齐

        for t in range(1, T_recon):
            recon[t, :D_use] = recon[t - 1, :D_use] + delta[t, :D_use]
            if D_use < num_dims:
                recon[t, D_use:] = arr2[t, D_use:]

        print("\n===== Reconstructed absolute FK from delta (recon) =====", flush=True)
        print(recon, flush=True)

        arr2_trunc = arr2[:T_recon]
        diff_delta = recon - arr2_trunc
        x_delta = np.arange(T_recon)

        print("\n===== diff_delta = recon - npz2 （前 T_recon 帧）=====", flush=True)
        print(diff_delta, flush=True)
        print("[INFO] ==== Delta-based reconstruction END ====", flush=True)

    # ============ 针对每个自由度，画“综合图” ============
    print("[INFO] Plotting combined curves (absolute + diff) for each DoF...", flush=True)
    for i in range(num_dims):
        fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

        # ---------- 上半图：绝对值 ----------
        axs[0].plot(x_abs, arr1[:, i], label="npz1", linewidth=1.0)
        axs[0].plot(x_abs, arr2[:, i], label="npz2", linewidth=1.0)

        if recon is not None:
            # recon 可能长度和 T_abs 不一样，用 x_delta
            axs[0].plot(x_delta, recon[:, i], label="recon_from_delta", linestyle="--", linewidth=1.0)

        axs[0].set_ylabel("value")
        axs[0].set_title(f"{dof_names[i]}: absolute values")
        axs[0].legend()
        axs[0].grid(True, linestyle=":")

        # ---------- 下半图：差值 ----------
        axs[1].plot(x_abs, diff_abs[:, i], label="diff_abs (npz1 - npz2)", color="r", linewidth=1.0)

        if diff_delta is not None:
            axs[1].plot(
                x_delta,
                diff_delta[:, i],
                label="diff_delta (recon - npz2)",
                color="m",
                linestyle="--",
                linewidth=1.0,
            )

        axs[1].axhline(0.0, linestyle="--", color="k", linewidth=0.8)
        axs[1].set_xlabel("frame index")
        axs[1].set_ylabel("diff")
        axs[1].set_title(f"{dof_names[i]}: differences")
        axs[1].legend()
        axs[1].grid(True, linestyle=":")

        plt.tight_layout()
        fname = f"combined_{i:02d}_{dof_names[i]}.png"
        fpath = os.path.join(args.out_dir, fname)
        plt.savefig(fpath)
        plt.close()

        print(f"[INFO] Saved combined plot for {dof_names[i]} to {fpath}", flush=True)

    # ============ 3D 轨迹图 ============
    if num_dims >= 9:
        print("[INFO] Plotting 3D trajectories (npz1 / npz2 / recon)...", flush=True)

        # 左臂: X,Y,Z = 0,1,2
        L1 = arr1[:, 0:3]  # npz1
        L2 = arr2[:, 0:3]  # npz2

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(L1[:, 0], L1[:, 1], L1[:, 2], label="npz1 Left")
        ax.plot(L2[:, 0], L2[:, 1], L2[:, 2], label="npz2 Left")

        if recon is not None:
            L_recon = recon[:, 0:3]
            ax.plot(L_recon[:, 0], L_recon[:, 1], L_recon[:, 2], label="recon Left", linestyle="--")

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title("Left arm 3D trajectory (npz1 / npz2 / recon)")
        ax.legend()
        plt.tight_layout()
        left_traj_path = os.path.join(args.out_dir, "traj_left_3d.png")
        plt.savefig(left_traj_path)
        plt.close()
        print(f"[INFO] Saved left arm 3D trajectory to {left_traj_path}", flush=True)

        # 右臂: X,Y,Z = 6,7,8
        R1 = arr1[:, 6:9]
        R2 = arr2[:, 6:9]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(R1[:, 0], R1[:, 1], R1[:, 2], label="npz1 Right")
        ax.plot(R2[:, 0], R2[:, 1], R2[:, 2], label="npz2 Right")

        if recon is not None:
            R_recon = recon[:, 6:9]
            ax.plot(R_recon[:, 0], R_recon[:, 1], R_recon[:, 2], label="recon Right", linestyle="--")

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title("Right arm 3D trajectory (npz1 / npz2 / recon)")
        ax.legend()
        plt.tight_layout()
        right_traj_path = os.path.join(args.out_dir, "traj_right_3d.png")
        plt.savefig(right_traj_path)
        plt.close()
        print(f"[INFO] Saved right arm 3D trajectory to {right_traj_path}", flush=True)
    else:
        print("[WARN] num_dims < 9，无法解析出左右臂 3D 位置，不画 3D 轨迹。", flush=True)

    print("[INFO] All done.", flush=True)
    print("=== analyze_two_npz.py END ===", flush=True)


if __name__ == "__main__":
    main()

"""
示例：

1）只分析 obs vs fk 的绝对差异：
python analyze_two_npz.py \
  --npz1 /home/.../piper_obs_endpose_ep2.npz \
  --npz2 /home/.../piper_fk_endpose_ep2.npz \
  --out_dir /home/.../diff_plots_ep2

2）再加上增量式分析（基于 fk_delta 重建轨迹）：
python analyze_two_npz.py \
  --npz1 /home/.../piper_obs_endpose_ep2.npz \
  --npz2 /home/.../piper_fk_endpose_ep2.npz \
  --delta_npz /home/.../piper_fk_delta_ep2.npz \
  --out_dir /home/.../diff_plots_ep2
"""


"""
示例：

1）只分析 obs vs fk 的绝对差异：
python analyze_two_npz.py \
  --npz1 /home/.../piper_obs_endpose_ep2.npz \
  --npz2 /home/.../piper_fk_endpose_ep2.npz \
  --out_dir /home/.../diff_plots_ep2

2）再加上增量式分析（基于 fk_delta 重建轨迹）：
python analyze_two_npz.py \
  --npz1 piper_obs_endpose_ep2.npz \
  --npz2 piper_fk_endpose_ep2.npz \
  --delta_npz piper_fk_delta_ep2.npz \
  --out_dir diff_plots_ep2
"""
