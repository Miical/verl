#!/usr/bin/env python3
"""Re-render LIBERO offline demos to regenerate sharp RGB observations.

This script replays each demo action sequence in simulator from the stored initial
state, then rewrites only image observations (`agentview_rgb`, `eye_in_hand_rgb`).
All non-image fields are copied unchanged.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv

from verl.experimental.vla.envs.libero_env.utils import get_libero_image, get_libero_wrist_image


def parse_task_name_from_file(hdf5_path: str) -> str:
    name = os.path.basename(hdf5_path)
    if name.endswith("_demo.hdf5"):
        name = name[: -len("_demo.hdf5")]
    name = re.sub(r"^[A-Z_]+_SCENE\d+_", "", name)
    return name.replace("_", " ").lower()


def resize_rgb(image: np.ndarray, size: int) -> np.ndarray:
    if image.shape[:2] == (size, size):
        return image.astype(np.uint8, copy=False)
    pil = Image.fromarray(image.astype(np.uint8, copy=False))
    return np.asarray(pil.resize((size, size), Image.Resampling.BICUBIC), dtype=np.uint8)


def make_env_for_task(task, image_size: int) -> OffScreenRenderEnv:
    bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        camera_depths=False,
        camera_heights=image_size,
        camera_widths=image_size,
        camera_names=["agentview", "robot0_eye_in_hand"],
    )
    env.seed(0)
    return env


def rerender_demo(env: OffScreenRenderEnv, init_state: np.ndarray, actions: np.ndarray, image_size: int) -> tuple[np.ndarray, np.ndarray]:
    env.reset()
    obs = env.set_init_state(init_state)

    zero = np.zeros(actions.shape[-1], dtype=np.float32)
    for _ in range(10):
        obs, _, _, _ = env.step(zero)

    T = actions.shape[0]
    agent_imgs = np.empty((T, image_size, image_size, 3), dtype=np.uint8)
    wrist_imgs = np.empty((T, image_size, image_size, 3), dtype=np.uint8)

    for t in range(T):
        agent_imgs[t] = resize_rgb(get_libero_image(obs), image_size)
        wrist_imgs[t] = resize_rgb(get_libero_wrist_image(obs), image_size)
        obs, _, _, _ = env.step(actions[t])

    return agent_imgs, wrist_imgs


def process_one_file(src_path: str, dst_path: str, task_lookup: dict[str, object], image_size: int) -> None:
    task_name = parse_task_name_from_file(src_path)
    if task_name not in task_lookup:
        raise KeyError(f"Cannot map file to LIBERO task: {src_path} (parsed task='{task_name}')")

    task = task_lookup[task_name]
    env = make_env_for_task(task, image_size)

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with h5py.File(src_path, "r") as f_src, h5py.File(dst_path, "w") as f_dst:
        for k, v in f_src.attrs.items():
            f_dst.attrs[k] = v

        f_src.copy("data", f_dst)
        data_src = f_src["data"]
        data_dst = f_dst["data"]

        for demo_key in data_src.keys():
            demo_src = data_src[demo_key]
            demo_dst = data_dst[demo_key]

            actions = demo_src["actions"][:]
            if "states" not in demo_src:
                raise KeyError(f"Demo '{demo_key}' in {src_path} has no 'states' for simulator reset")
            init_state = demo_src["states"][0]

            agent_imgs, wrist_imgs = rerender_demo(env, init_state, actions, image_size)

            obs_dst = demo_dst["obs"]
            if "agentview_rgb" in obs_dst:
                del obs_dst["agentview_rgb"]
            if "eye_in_hand_rgb" in obs_dst:
                del obs_dst["eye_in_hand_rgb"]

            obs_dst.create_dataset("agentview_rgb", data=agent_imgs, compression="gzip", compression_opts=4)
            obs_dst.create_dataset("eye_in_hand_rgb", data=wrist_imgs, compression="gzip", compression_opts=4)

    env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fix blurry LIBERO dataset images by simulator re-rendering")
    parser.add_argument(
        "--src",
        type=str,
        default="/shared_disk/users/angen.ye/code/hil-serl/datasets/LIBERO-dataset/libero_10",
        help="Source dataset dir containing *_demo.hdf5",
    )
    parser.add_argument(
        "--dst",
        type=str,
        default="/shared_disk/users/angen.ye/code/hil-serl/datasets/LIBERO-dataset/libero_10_fixed",
        help="Output dir for fixed dataset",
    )
    parser.add_argument("--task-suite", type=str, default="libero_10")
    parser.add_argument("--image-size", type=int, default=224, help="Output RGB resolution")
    parser.add_argument("--limit-files", type=int, default=-1, help="Optional debug limit")
    args = parser.parse_args()

    src_files = sorted(glob.glob(os.path.join(args.src, "*_demo.hdf5")))
    if not src_files:
        raise FileNotFoundError(f"No *_demo.hdf5 under: {args.src}")
    if args.limit_files > 0:
        src_files = src_files[: args.limit_files]

    benchmark = get_benchmark(args.task_suite)()
    task_lookup = {task.language.lower(): task for task in [benchmark.get_task(i) for i in range(benchmark.get_num_tasks())]}

    print(f"Found {len(src_files)} source files. Writing fixed dataset to: {args.dst}")
    Path(args.dst).mkdir(parents=True, exist_ok=True)

    for i, src in enumerate(src_files, start=1):
        dst = os.path.join(args.dst, os.path.basename(src))
        print(f"[{i}/{len(src_files)}] {os.path.basename(src)}")
        process_one_file(src, dst, task_lookup=task_lookup, image_size=args.image_size)

    print("Done. Dataset images regenerated.")


if __name__ == "__main__":
    main()
