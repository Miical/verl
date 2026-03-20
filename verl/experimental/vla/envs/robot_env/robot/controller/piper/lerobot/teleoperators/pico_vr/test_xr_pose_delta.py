"""
Standalone test script for the `_process_xr_pose` logic.

Supports two modes:
  1. demo (default): feeds synthetic poses to show how deltas behave.
  2. live: connects to XRoboToolkit SDK, reads VR controller poses, and
     prints delta translations / rotations in real time.

Usage examples:
    python src/lerobot/teleoperators/pico_vr/test_xr_pose_delta.py --mode demo
    python src/lerobot/teleoperators/pico_vr/test_xr_pose_delta.py --mode live --source right_controller
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path
import numpy as np

# Allow running the script directly (python scripts/tests/test_xr_pose_delta.py)
# by ensuring the project root is available on sys.path.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from xr_client import XrClient
from geometry import R_HEADSET_TO_WORLD
from pose_delta import GripperDeltaTracker, PoseDeltaCalculator


def run_demo(calculator: PoseDeltaCalculator):
    xr_samples = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # reference pose
        [0.05, 0.02, -0.01, 0.0, 0.0, 0.130526, 0.991444],  # ~15 deg yaw
        [0.08, 0.03, 0.00, 0.0, 0.130526, 0.0, 0.991444],   # ~15 deg pitch
    ]

    for idx, pose in enumerate(xr_samples):
        delta_xyz, delta_rot = calculator.update(pose)
        print(f"[demo] Sample {idx}:")
        print(f"  delta_xyz = {delta_xyz}")
        print(f"  delta_rot (axis-angle) = {delta_rot}")


def run_live(
    calculator: PoseDeltaCalculator,
    source: str,
    sample_rate_hz: float,
    gripper_tracker: GripperDeltaTracker | None = None,
):
    client = XrClient()
    period = 1.0 / sample_rate_hz
    running = True

    def handle_sigint(signum, frame):
        nonlocal running
        running = False
        print("\nStopping live capture...")

    signal.signal(signal.SIGINT, handle_sigint)
    print(f"Live capture started. Reading {source} at {sample_rate_hz} Hz. Press Ctrl+C to exit.")
    print("Tip: Move the controller to see delta updates. Restart script to reset the reference pose.")

    while running:
        pose = client.get_pose_by_name(source)
        delta_xyz, delta_rot = calculator.update(pose)
        log_msg = (
            f"[live][{source}] delta_xyz={np.array2string(delta_xyz, precision=4)}, "
            f"delta_rot={np.array2string(delta_rot, precision=4)}"
        )
        if gripper_tracker is not None:
            grip = gripper_tracker.update(client)
            log_msg += (
                f", gripper_trigger={grip['raw']:.3f}, "
                f"gripper_delta={grip['delta']:.3f}, "
                f"opening={grip['opening_m']*1000:.1f}mm"
            )
        print(log_msg)
        time.sleep(period)


def parse_args():
    parser = argparse.ArgumentParser(description="Test XR pose delta computation.")
    parser.add_argument(
        "--mode",
        choices=["demo", "live"],
        default="demo",
        help="demo: use synthetic data; live: read from actual XR device",
    )
    parser.add_argument(
        "--source",
        default="right_controller",
        choices=["left_controller", "right_controller", "headset"],
        help="Pose source name when running in live mode.",
    )
    parser.add_argument(
        "--scale-factor",
        type=float,
        default=1.0,
        help="Scaling applied to translation deltas.",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=20.0,
        help="Sampling frequency (Hz) when running in live mode.",
    )
    parser.add_argument(
        "--gripper-trigger",
        type=str,
        default='right_trigger',
        help="XR input name to treat as gripper trigger (e.g., right_trigger).",
    )
    parser.add_argument(
        "--gripper-open",
        type=float,
        default=0.07,
        help="Gripper opening (meters) when trigger value is 0.0.",
    )
    parser.add_argument(
        "--gripper-close",
        type=float,
        default=0.0,
        help="Gripper opening (meters) when trigger value is 1.0.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    calculator = PoseDeltaCalculator(
        R_headset_world=R_HEADSET_TO_WORLD,
        scale_factor=args.scale_factor,
    )

    if args.mode == "demo":
        run_demo(calculator)
    else:
        gripper_tracker = (
            GripperDeltaTracker(
                trigger_name=args.gripper_trigger,
                open_pos=args.gripper_open,
                close_pos=args.gripper_close,
            )
            if args.gripper_trigger
            else None
        )
        run_live(
            calculator,
            args.source,
            args.sample_rate,
            gripper_tracker=gripper_tracker,
        )


if __name__ == "__main__":
    main()

