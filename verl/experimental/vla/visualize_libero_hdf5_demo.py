#!/usr/bin/env python3
"""Visualize a single demo from a LIBERO HDF5 file and export as MP4."""

from __future__ import annotations

import argparse
import os

import h5py
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


def to_uint8_hwc(img: np.ndarray) -> np.ndarray:
    if img.ndim != 3:
        raise ValueError(f"Expected 3D image, got {img.shape}")
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def add_text(image: np.ndarray, lines: list[str]) -> np.ndarray:
    canvas = Image.fromarray(image)
    draw = ImageDraw.Draw(canvas)
    y = 8
    for line in lines:
        draw.text((8, y), line, fill=(255, 255, 255))
        y += 14
    return np.asarray(canvas)


def tile_lr(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    h = max(left.shape[0], right.shape[0])
    lw, rw = left.shape[1], right.shape[1]
    out = np.zeros((h, lw + rw, 3), dtype=np.uint8)
    out[: left.shape[0], :lw] = left
    out[: right.shape[0], lw : lw + rw] = right
    return out


def resolve_demo_key(data_group: h5py.Group, demo: str | int) -> str:
    if isinstance(demo, int):
        return f"demo_{demo}"
    if demo.startswith("demo_"):
        return demo
    if demo.isdigit():
        return f"demo_{demo}"
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize one demo from LIBERO hdf5 and save mp4")
    parser.add_argument("--hdf5", required=True, help="Path to *_demo.hdf5")
    parser.add_argument("--demo", default="demo_0", help="Demo key or index, e.g. demo_7 or 7")
    parser.add_argument("--output", required=True, help="Output mp4 path")
    parser.add_argument("--fps", type=int, default=20, help="Video FPS")
    parser.add_argument("--max-frames", type=int, default=-1, help="Optional frame cap for quick check")
    args = parser.parse_args()

    with h5py.File(args.hdf5, "r") as f:
        if "data" not in f:
            raise KeyError(f"Invalid hdf5: missing 'data' group in {args.hdf5}")

        data = f["data"]
        demo_key = resolve_demo_key(data, str(args.demo))
        if demo_key not in data:
            raise KeyError(f"Demo '{demo_key}' not found. Available: {list(data.keys())[:10]}")

        demo = data[demo_key]
        obs = demo["obs"]
        if "agentview_rgb" not in obs or "eye_in_hand_rgb" not in obs:
            raise KeyError("Expected both obs/agentview_rgb and obs/eye_in_hand_rgb")

        agent = obs["agentview_rgb"]
        wrist = obs["eye_in_hand_rgb"]
        T = min(agent.shape[0], wrist.shape[0])
        if args.max_frames > 0:
            T = min(T, args.max_frames)

        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        writer = imageio.get_writer(args.output, fps=args.fps, codec="libx264")
        try:
            for t in range(T):
                a = to_uint8_hwc(agent[t])
                w = to_uint8_hwc(wrist[t])
                a = add_text(a, [f"{demo_key}", f"frame={t}/{T-1}", "cam=agentview"])
                w = add_text(w, [f"{demo_key}", f"frame={t}/{T-1}", "cam=eye_in_hand"])
                frame = tile_lr(a, w)
                writer.append_data(frame)
        finally:
            writer.close()

    print(f"Saved video: {args.output} (frames={T}, fps={args.fps}, demo={demo_key})")


if __name__ == "__main__":
    main()
