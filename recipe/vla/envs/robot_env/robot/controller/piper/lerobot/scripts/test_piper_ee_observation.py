#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
logging.getLogger("can.interfaces.socketcan").setLevel(logging.ERROR)

from dataclasses import dataclass
from pathlib import Path
import time

import cv2
import numpy as np
import draccus
from lerobot.utils.errors import DeviceNotConnectedError
# 关键：导入以触发"注册"（不要删除）
from lerobot.robots import (  # noqa: F401
    piper_follower,
    RobotConfig,
    make_robot_from_config,
)
from lerobot.cameras import (  # noqa: F401 - 导入相机以触发注册
    opencv,
    realsense,
    dabai,
)
from lerobot.utils.robot_utils import busy_wait


@dataclass
class PiperEEObservationTestConfig:
    # 必填：从命令行传入类型
    robot: RobotConfig  # --robot.type=piper_follower_end_effector
    rate_hz: float = 10.0     # 读取观测的频率（通常 10Hz 已足够）
    print_every: int = 10     # 每多少帧打印一次（0 表示不打印）
    duration_s: float = 10.0  # 测试持续时间（秒）
    save_images: bool = True  # 是否保存相机图像
    image_output_dir: str = "outputs/camera_images"  # 图像保存目录


BANNER = r"""
[INFO] Piper End Effector Observation Test started.

将每秒读取并打印机器人的观测数据（关节位置、夹爪状态、相机图像等）。

按 Ctrl+C 退出测试。
"""


@draccus.wrap()
def main(cfg: PiperEEObservationTestConfig):
    robot = make_robot_from_config(cfg.robot)

    robot.connect()

    print(BANNER)

    # 创建图像保存目录
    output_dir = Path(cfg.image_output_dir)
    if cfg.save_images:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] 图像将保存到: {output_dir.absolute()}")

    dt = 1.0 / max(1.0, float(cfg.rate_hz))
    step = 0
    start_time = time.time()

    try:
        while time.time() - start_time < cfg.duration_s:
            try:
                # 读取观测数据
                obs = robot.get_observation()
            except DeviceNotConnectedError:
                print("[ERROR] Robot disconnected. Exiting loop.")
                break

            # 保存相机图像
            if cfg.save_images:
                for key, val in obs.items():
                    # 检查是否是图像数据 (numpy array 且有 shape)
                    if isinstance(val, np.ndarray) and hasattr(val, 'shape') and len(val.shape) == 3:
                        # RGB 转 BGR（如果是RGB格式）
                        if val.shape[2] == 3:
                            timestamp = time.time()
                            timestamp_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(timestamp))
                            image_path = output_dir / f"{timestamp_str}_{key}_{step:05d}.png"
                            # cv2.imwrite 需要 BGR 格式
                            image_bgr = cv2.cvtColor(val, cv2.COLOR_RGB2BGR)
                            cv2.imwrite(str(image_path), image_bgr)
                            if cfg.print_every > 0 and (step % cfg.print_every == 0):
                                print(f"    保存图像: {image_path}")

            if cfg.print_every > 0 and (step % cfg.print_every == 0):
                # 打印观测数据的关键信息
                print(f"[tick={step}, time={time.time()-start_time:.2f}s]")
                
                # 列出所有观测的键
                keys = list(obs.keys())
                print(f"  观测键数量: {len(keys)}")
                print(f"  观测键: {keys}")
                
                # 打印一些关键数值观测（跳过图像数据）
                for key, val in obs.items():
                    if isinstance(val, (int, float)):
                        print(f"    {key}: {val}")
                    elif hasattr(val, 'shape'):  # numpy array
                        print(f"    {key}: {val.shape}")
                    else:
                        print(f"    {key}: {type(val)}")

            step += 1
            busy_wait(dt)

        print(f"\n[INFO] 测试完成，共读取了 {step} 次观测数据。")

    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt, stopping.")

    finally:
        # 清理
        try:
            robot.disconnect()
        except Exception:
            pass
        print("[INFO] Clean exit.\n")


if __name__ == "__main__":
    main()

"""

用法示例：
conda activate lerobot_hil-serl
#export PYTHONPATH=/home/agilex-home/agilex/dengqiuping/code/lerobot/src:$PYTHONPATH

# 1. 基本用法（不保存图像）：
python src/lerobot/scripts/test_piper_ee_observation.py \
  --robot.type=piper_follower_end_effector \
  --rate_hz=10 --duration_s=5 --print_every=10 \
  --save_images=false

# 2. 单个相机并保存图像：
python src/lerobot/scripts/test_piper_ee_observation.py \
  --robot.type=piper_follower_end_effector \
  --robot.cameras='{"front": {"type": "orbbec_dabai", "serial_number_or_name": "CC1B25100EM", "width": 640, "height": 480, "fps": 30}}' \
  --rate_hz=10 --duration_s=5 --print_every=5 \
  --save_images=true --image_output_dir=outputs/my_test

# 3. 多个相机并保存图像（注意：所有相机都在一个字典里）：

python src/lerobot/scripts/test_piper_ee_observation.py \
  --robot.type=piper_follower_end_effector \
  --robot.cameras='{"front": {"type": "orbbec_dabai", "serial_number_or_name": "CC1B25100EM", "width": 640, "height": 480, "fps": 30}, "left": {"type": "intelrealsense", "serial_number_or_name": "242522071187", "width": 640, "height": 480, "fps": 30}, "right": {"type": "intelrealsense", "serial_number_or_name": "327122075682", "width": 640, "height": 480, "fps": 30}}' \
  --rate_hz=10 --duration_s=5 --print_every=5 \
  --save_images=true

python src/lerobot/scripts/test_piper_ee_observation.py \
  --robot.type=piper_single_follower \
  --robot.cameras='{"front": {"type": "orbbec_dabai", "serial_number_or_name": "CC1B25100EM", "width": 640, "height": 480, "fps": 30}, "left": {"type": "intelrealsense", "serial_number_or_name": "242522071187", "width": 640, "height": 480, "fps": 30}, "right": {"type": "intelrealsense", "serial_number_or_name": "327122075682", "width": 640, "height": 480, "fps": 30}}' \
  --rate_hz=10 --duration_s=5 --print_every=5 \
  --save_images=true
  

参数说明：
  --save_images: 是否保存相机图像（默认: true）
  --image_output_dir: 图像保存目录（默认: "outputs/camera_images"）
  --rate_hz: 读取频率（默认: 10）
  --duration_s: 测试持续时间（默认: 10秒）
  --print_every: 每多少帧打印一次（默认: 10）
"""
