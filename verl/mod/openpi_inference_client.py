# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
import json
import numpy as np
import os
import torch
import tqdm
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import sys
import tyro
from isaacsim import SimulationApp
from openpi_client import websocket_client_policy as _websocket_client_policy

# Ensure workspace root is on sys.path so "benchmarks" imports work even when
# launched from subdirectories.
_WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))

# Utilize the common utility functions from gr00t for OpenPI inference
from benchmarks.gr00t.closedloop_policy_inference import ClosedLoopArguments, ClosedLoopPolicyInference
from isaaclab.app import AppLauncher
from isaaclab.utils.datasets import HDF5DatasetFileHandler


@dataclass
class OpenpiClientArguments(ClosedLoopArguments):

    record_images: bool = False
    record_videos: bool = False
    num_envs: int = 1
    background_env_usd_path: str | None = None
    record_camera_output_path: str | None = None

    # Server connection parameters
    server_host: str = "localhost"
    server_port: int = 8000
    target_image_size: tuple[int, int, int] = (224, 224, 3)

    # Simulator specific parameters
    headless: bool = False
    seed: int = 11
    debug_mode: int = 0  # 0: no debug, 1: debug actions only, 2: debug images and actions
    debug_path: str = "./openpi_libero_debug"

    camera_names: tuple[str] = ("agentview_cam", "eye_in_hand_cam")
    num_steps_wait: int = 20  # Number of steps to wait for objects to stabilize i n sim
    replan_steps: int = 10  # For each action, will execute replan_steps times
    max_inference_steps: int = 80  # max number of inference steps to run
    num_success_steps: int = 8  # continuous success steps to consider the policy as successful
    num_total_experiments: int = 50  # total number of experiments to do policy evaluation

    policy_type: str = "task_space"  # task_space or joint_space
    task: str = "Isaac-Libero-Franka-IK-Abs-v0"

    # Task setup parameters
    task_suite: str = "libero_goal"
    task_id: int = 1
    task_config_path: Path = Path(__file__).parent.resolve() / "config"
    language_instruction: str = ""

    # HDF5 dataset parameters for initial state loading
    data_root: Path = Path("/home/weihuaz/projects/Robotics/RobotLearningLab")
    hdf5_folder_path: str = "benchmarks/datasets/libero/assembled_hdf5"
    hdf5_folder: Path = data_root / hdf5_folder_path


# Parse arguments first to get task_suite and task_id
args = tyro.cli(OpenpiClientArguments)

# Adjust max_inference_steps based on task suite name
if args.task_suite == "libero_spatial":
    args.max_inference_steps = 22  # longest training demo has 193 steps
elif args.task_suite == "libero_object":
    args.max_inference_steps = 28  # longest training demo has 254 steps
elif args.task_suite == "libero_goal":
    args.max_inference_steps = 30  # longest training demo has 270 steps
elif args.task_suite == "libero_10":
    args.max_inference_steps = 52  # longest training demo has 505 steps

# Launch the simulator
app_launcher = AppLauncher(headless=False, enable_cameras=True, num_envs=1)
simulation_app = app_launcher.app

# add configs for dataset generation for various task_suite and task_id,
# supported task_suites: [xhumanoid, libero, etc.]
if args.task_suite is not None:
    from isaaclab_playground.utils.task_configs import setup_task_objects

    setup_task_objects(args.task_suite, args.task_id)

import gymnasium as gym

from benchmarks.openpi.env import axisangle2quat, quat2axisangle, resize_frames_with_padding
from isaaclab.envs import ManagerBasedRLEnvCfg

from isaaclab_tasks.utils import import_packages

# The blacklist is used to prevent importing configs from sub-packages
_BLACKLIST_PKGS = ["utils", ".mdp", "pick_place"]
# Import all configs in this package
import_packages("isaaclab_tasks", _BLACKLIST_PKGS)


def get_episode_map(names):
    """Get a mapping of episode indices to their names.

    Args:
        names: List or dict of episode names

    Returns:
        dict: Mapping of episode indices to their names (e.g., {0: 'episode_0', 2: 'episode_2', 5: 'episode_5'})
    """
    import re

    def extract_episode_index(name):
        """Extract the episode index from the name."""
        match = re.search(r"(\d+)", name)
        if match:
            return int(match.group(1))
        return 0

    # Create a mapping of episode index to episode name
    episode_map = {}
    for name in names:
        idx = extract_episode_index(name)
        episode_map[idx] = name

    return episode_map


def find_hdf5_file(hdf5_folder: Path, task_suite: str, task_id: int) -> Path | None:
    """Find the HDF5 file for the given task_suite and task_id.

    Args:
        hdf5_folder: Path to the folder containing HDF5 files
        task_suite: Task suite name (e.g., "libero_10", "xhumanoid")
        task_id: Task ID number

    Returns:
        Path to the HDF5 file if found, None otherwise
    """
    if not hdf5_folder.exists():
        print(f"HDF5 folder does not exist: {hdf5_folder}")
        return None

    # Create pattern to match the HDF5 file
    pattern = f"{task_suite}_task{task_id}_*_demo.hdf5"

    # Find matching files
    matching_files = list(hdf5_folder.glob(pattern))

    if matching_files:
        hdf5_file = matching_files[0]
        print(f"Found HDF5 file: {hdf5_file}")
        return hdf5_file
    else:
        print(f"No HDF5 file found matching pattern: {pattern}")
        print(f"Searched in: {hdf5_folder}")
        # List available files for debugging
        available_files = list(hdf5_folder.glob("*.hdf5"))
        if available_files:
            print("Available HDF5 files:")
            for file in available_files:
                print(f"  - {file.name}")
        return None


def run_closed_loop_policy(
    args: OpenpiClientArguments,
    simulation_app: SimulationApp,
    env: gym.Env,
    env_cfg: ManagerBasedRLEnvCfg,
    success_term: Callable[[gym.Env], bool] | None,
):
    """Run the closed loop policy evaluation."""

    if args.debug_mode > 0:
        os.makedirs(args.debug_path, exist_ok=True)

    successful_experiments = 0

    # Find HDF5 file based on task_suite and task_id
    hdf5_file = find_hdf5_file(args.hdf5_folder, args.task_suite, args.task_id)

    # Load dataset and episode information if HDF5 file is found
    episode_indices_to_use = []
    episode_map = {}
    dataset_file_handler = None

    if hdf5_file and hdf5_file.exists():
        dataset_file_handler = HDF5DatasetFileHandler()
        dataset_file_handler.open(str(hdf5_file))
        episode_count = dataset_file_handler.get_num_episodes()
        episode_map = get_episode_map(dataset_file_handler.get_episode_names())
        episode_indices_to_use = list(range(episode_count))
        print(f"Loaded {episode_count} initial_states of episodes from dataset: {hdf5_file}")
    else:
        print(
            f"No valid HDF5 file found for {args.task_suite}_task{args.task_id}, will use default reset for all"
            " experiments"
        )

    # read language instruction from task_suite_config
    task_config_path = args.task_config_path / f"{args.task_suite}.json"
    if not task_config_path.exists():
        raise FileNotFoundError(f"Task config file not found: {task_config_path}")
    with open(task_config_path) as f:
        task_suite_config = json.load(f)

    for task in task_suite_config["tasks"]:
        task_id = task["task_id"]
        if task_id == args.task_id:
            args.language_instruction = task["language_instruction"]
            print(f"\nUsing language instruction: {args.language_instruction}")
            break

    client = _websocket_client_policy.WebsocketClientPolicy(args.server_host, args.server_port)
    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        for exp_idx in range(args.num_total_experiments):
            print(f"\nStarting experiment {exp_idx + 1}/{args.num_total_experiments}")
            success_step_count = 0
            experiment_success = False

            # reset environment with initial state from HDF5 if available
            if episode_indices_to_use:
                # Use episode index from the list (cycling through all episodes)
                episode_index = episode_indices_to_use[exp_idx % len(episode_indices_to_use)]
                episode_data = dataset_file_handler.load_episode(episode_map[episode_index], env.unwrapped.device)

                if "initial_state" in episode_data.data:
                    # reset environment
                    obs, info = env.reset()
                    # Set initial state for the environment
                    initial_state = episode_data.get_initial_state()
                    # print("---- initial_state: ", initial_state)
                    obs, info = env.reset_to(
                        initial_state, torch.arange(args.num_envs, device=env.unwrapped.device), is_relative=True
                    )

                    # # reset to idle action as state
                    # initial_action = initial_state["articulation"]["robot"]["joint_position"][:, :8]
                    # for action_idx in tqdm.tqdm(range(args.num_steps_wait)):
                    #     env.step(initial_action)
                    print(f"Reset environment to initial state from episode {episode_index}")
                else:
                    # Fallback to default reset if no initial state available
                    obs, info = env.reset()
                    print(f"No initial state found in episode {episode_index}, using default reset")
            else:
                # Fallback to default reset if no dataset file specified or doesn't exist
                obs, info = env.reset()

            frame_count = 0

            for action_idx in tqdm.tqdm(range(args.max_inference_steps)):
                # Get the cam rgb
                rgbs = []
                for cam_name in list(args.camera_names):
                    cam_id = cam_name.split("_")[0]
                    cam = env.unwrapped.scene[cam_name]
                    rgb = cam.data.output["rgb"]
                    # FIXME: Model trained with BGR ordering; convert RGB -> BGR for inference.
                    rgb = resize_frames_with_padding(rgb, args.target_image_size, bgr_conversion=True, pad_img=True)
                    rgbs.append(rgb)

                    if args.debug_mode > 1:
                        rgb_np = (rgb * 255).astype(np.uint8) if rgb.dtype == np.float32 else rgb.copy()
                        cv2.imwrite(
                            str(f"{args.debug_path}/frame_{frame_count:04d}_{cam_id}.png"),
                            cv2.cvtColor(rgb_np[0], cv2.COLOR_RGB2BGR),
                        )
                if args.policy_type == "task_space":
                    # get eef_pose in base frame
                    eef_pose = obs["policy"]["eef_pose"].cpu().numpy()
                    eef_pose = np.squeeze(eef_pose, axis=0)
                    eef_pose_axisangle = quat2axisangle(eef_pose[3:7])

                    gripper_pos = obs["policy"]["gripper_pos"].cpu().numpy()
                    gripper_state = gripper_pos[:, 0]

                    # print("eef_pose[:3]: ", eef_pose[:3].shape, "eef_pose_axisangle: ", eef_pose_axisangle.shape,
                    #       "gripper_state: ", gripper_state.shape)
                    eef_pose_states = np.concatenate((eef_pose[:3], eef_pose_axisangle, gripper_state), axis=0)

                    # Prepare observations dict
                    # print("input shape, rgbs[0]: ", rgbs[0].shape, "rgbs[1]: ", rgbs[1].shape,
                    #       "eef_pose_states: ", eef_pose_states.shape)
                    element = {
                        "observation/image": np.squeeze(rgbs[0], axis=0),
                        "observation/wrist_image": np.squeeze(rgbs[1], axis=0),
                        "observation/state": eef_pose_states,
                        "prompt": args.language_instruction,
                    }

                    # Query model to get action
                    action_chunk = client.infer(element)["actions"]
                    assert len(action_chunk) >= args.replan_steps, (
                        f"We want to replan every {args.replan_steps} steps, but policy only predicts"
                        f" {len(action_chunk)} steps."
                    )

                    # Translate the action into eef_pose
                    eef_pose = np.array([axisangle2quat(act[3:6]) for act in action_chunk])
                    eef_pose_quat = np.concatenate(
                        (action_chunk[:, :3], eef_pose, action_chunk[:, 6:7]), axis=1
                    )  # (N, 8)
                    action = torch.from_numpy(eef_pose_quat).float()

                    if args.debug_mode > 0:
                        np.save(str(f"{args.debug_path}/action_{frame_count:04d}.npy"), action)

                    action = action[: args.replan_steps, :]
                    assert action.shape[1] == env.action_space.shape[1], (
                        f"Action shape {action.shape} does not match environment action space shape"
                        f" {env.action_space.shape}"
                    )
                else:
                    raise ValueError(f"[ERROR] Invalid policy type: {args.policy_type}")

                for i in range(args.replan_steps):
                    obs, _, _, _, _ = env.step(action[i].reshape([1, -1]))

                    if success_term is not None:
                        if bool(success_term.func(env, **success_term.params)[0]):
                            success_step_count += 1
                            if success_step_count >= args.num_success_steps:
                                print(f"### Policy Evaluation Success, for {success_step_count} timestamp.")
                                experiment_success = True
                                break
                            print(f"## Success for {success_step_count} timestamp.")
                        else:
                            success_step_count = 0

                    if args.debug_mode > 2:
                        # get joint states
                        cam = env.unwrapped.scene["agentview_cam"]
                        rgb = cam.data.output["rgb"][0]
                        # get joint states
                        robot = env.unwrapped.scene["robot"]
                        states = robot.data.joint_pos
                        states = states.cpu().numpy()

                        np.save(str(f"{args.debug_path}/state_{frame_count:04d}_{i:02d}.npy"), states)
                        # Convert to numpy if it's a tensor
                        if isinstance(rgb, torch.Tensor):
                            rgb = rgb.cpu().numpy()
                        # Ensure correct format for saving
                        if rgb.dtype == np.float32:
                            rgb = (rgb * 255).astype(np.uint8)
                        # Save RGB image
                        cv2.imwrite(
                            str(f"{args.debug_path}/frame_{frame_count:04d}_{i:02d}.png"),
                            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                        )
                    frame_count += 1

                if experiment_success:
                    successful_experiments += 1
                    print(
                        f"## Experiment {exp_idx + 1} successful. Current SR: {successful_experiments} / {exp_idx + 1}."
                    )
                    break
                if action_idx >= args.max_inference_steps - 1:
                    print(f"## Experiment {exp_idx + 1} failed. Current SR: {successful_experiments} / {exp_idx + 1}.")

    success_rate = (successful_experiments / args.num_total_experiments) * 100
    print("\nEvaluation Results:")
    print(f"Total experiments: {args.num_total_experiments}")
    print(f"Successful experiments: {successful_experiments}")
    print(f"Success rate: {success_rate:.2f}%")


if __name__ == "__main__":
    print("args", args)

    if args.policy_type == "task_space":
        inferencer = ClosedLoopPolicyInference(args)
    else:
        raise ValueError(f"Invalid policy type: {args.policy_type}")

    # Initialize client policy inference
    env, env_cfg, success_term = inferencer.create_sim_environment()

    # Run the closed loop policy
    run_closed_loop_policy(
        args=args, simulation_app=simulation_app, env=env, env_cfg=env_cfg, success_term=success_term
    )

    # Close environment and simulation app after replay is complete
    env.close()
    simulation_app.close()