# Copyright 2025 Bytedance Ltd. and/or its affiliates
"""Render a quick GIF preview directly from Libero RLPD dataset samples.

Debug utility to verify dataset loading/format alignment with training.
"""

from __future__ import annotations

import argparse
import os
import random

import numpy as np
from PIL import Image, ImageDraw

from verl.experimental.vla.dataloader import build_dataloader_components


def _to_hwc_uint8(img: np.ndarray) -> np.ndarray:
    if img.ndim != 3:
        raise ValueError(f"Expected image with 3 dims, got shape={img.shape}")

    if img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):
        img = np.transpose(img, (1, 2, 0))

    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    # Match env rendering convention
    return img[::-1, ::-1]


def _overlay_text(image: np.ndarray, lines: list[str]) -> np.ndarray:
    pil = Image.fromarray(image)
    draw = ImageDraw.Draw(pil)
    y = 8
    for line in lines:
        draw.text((8, y), line, fill=(0, 0, 0))
        y += 14
    return np.asarray(pil)


def _tile_lr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    h = max(a.shape[0], b.shape[0])
    w = max(a.shape[1], b.shape[1])
    out_a = np.zeros((h, w, 3), dtype=np.uint8)
    out_b = np.zeros((h, w, 3), dtype=np.uint8)
    out_a[: a.shape[0], : a.shape[1]] = a
    out_b[: b.shape[0], : b.shape[1]] = b
    return np.concatenate([out_a, out_b], axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render preview GIF from Libero dataset")
    parser.add_argument("--root", type=str, required=True, help="Path to Libero dataset root (same as data.rlpd_files)")
    parser.add_argument("--output", type=str, required=True, help="Output .gif path")
    parser.add_argument("--num-samples", type=int, default=64, help="Number of dataset items to render")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=6)
    args = parser.parse_args()

    if not args.output.lower().endswith(".gif"):
        raise ValueError("Please set --output to a .gif file path")

    dataset, _, _ = build_dataloader_components(dataset_type="libero", repo_id="parquet", root=args.root)

    random.seed(args.seed)
    np.random.seed(args.seed)

    total = len(dataset)
    if total == 0:
        raise ValueError("Dataset is empty")

    indices = np.linspace(0, total - 1, num=min(args.num_samples, total), dtype=int).tolist()
    random.shuffle(indices)

    frames: list[Image.Image] = []
    for i, idx in enumerate(indices):
        item = dataset[idx]

        t0_img = _to_hwc_uint8(item["t0.obs"]["agentview_rgb"])
        t1_img = _to_hwc_uint8(item["t1.obs"]["agentview_rgb"])

        shared_lines = [
            f"sample={i} idx={idx}",
            f"demo={item['t0.demo_key']}",
            f"file={os.path.basename(item['t0.hdf5_path'])}",
            f"t0_chunk_done={int(item['t0.chunk_dones'])} t1_chunk_done={int(item['t1.chunk_dones'])}",
            f"a0[0,:3]={np.asarray(item['t0.actions'])[0, :3].round(3).tolist()}",
            f"a1[0,:3]={np.asarray(item['t1.actions'])[0, :3].round(3).tolist()}",
        ]

        t0_overlay = _overlay_text(t0_img, ["phase=t0"] + shared_lines)
        t1_overlay = _overlay_text(t1_img, ["phase=t1"] + shared_lines)
        frame = _tile_lr(t0_overlay, t1_overlay)
        frames.append(Image.fromarray(frame))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    duration_ms = int(1000 / max(args.fps, 1))
    frames[0].save(args.output, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0)
    print(f"Saved preview GIF to {args.output}. total={total}, rendered={len(frames)}")


if __name__ == "__main__":
    main()
