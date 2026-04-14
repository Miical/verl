import torch
import argparse
import logging
import os
import signal
import threading
import time

import draccus
import gymnasium as gym
import numpy as np

from lerobot.rl.gym_manipulator import (
    GymManipulatorConfig, 
    make_robot_env, 
    make_processors, 
    create_transition,
    step_env_and_process_transition
)
from lerobot.processor import TransitionKey
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.utils.transition import Transition
from lerobot.processor import (
    DataProcessorPipeline,
    EnvTransition,
)
from .ipc_channel import clear_ipc, recv_obj, reply_obj, setup_ipc


logger = logging.getLogger(__name__)
_STOP = False
_RUNTIME = None


def _handle_stop_signal(signum, frame):
    del frame
    global _STOP
    _STOP = True
    logger.info("Received signal %s, stopping lerobot runtime.", signum)

def step_env_and_process_transition(
    env: gym.Env,
    transition: EnvTransition,
    action: torch.Tensor,
    env_processor: DataProcessorPipeline[EnvTransition, EnvTransition],
    action_processor: DataProcessorPipeline[EnvTransition, EnvTransition],
) -> EnvTransition:
    """
    Execute one step with processor pipeline.

    Args:
        env: The robot environment
        transition: Current transition state
        action: Action to execute
        env_processor: Environment processor
        action_processor: Action processor

    Returns:
        Processed transition with updated state.
    """

    # Create action transition
    transition[TransitionKey.ACTION] = action
    transition[TransitionKey.OBSERVATION] = (
        env.get_raw_joint_positions() if hasattr(env, "get_raw_joint_positions") else {}
    )
    processed_action_transition = action_processor(transition)
    processed_action = processed_action_transition[TransitionKey.ACTION]

    obs, reward, terminated, truncated, info = env.step(processed_action)

    reward = reward + processed_action_transition[TransitionKey.REWARD]
    terminated = terminated or processed_action_transition[TransitionKey.DONE]
    truncated = truncated or processed_action_transition[TransitionKey.TRUNCATED]
    complementary_data = processed_action_transition[TransitionKey.COMPLEMENTARY_DATA].copy()
    # patch: Let action-processor info (teleop events) override env defaults when keys collide.
    new_info = info.copy()
    new_info.update(processed_action_transition[TransitionKey.INFO])

    new_transition = create_transition(
        observation=obs,
        action=processed_action,
        reward=reward,
        done=terminated,
        truncated=truncated,
        info=new_info,
        complementary_data=complementary_data,
    )
    new_transition = env_processor(new_transition)

    return new_transition

class LerobotRuntime:
    def __init__(self, config_path: str, rank: int, stage_id: int):
        self.rank = rank
        self.stage_id = stage_id
        
        self.lerobot_config = draccus.parse(
            config_class=GymManipulatorConfig,
            config_path=config_path,
            args=[],
        )
        self.interaction_step = 0
        self.online_env, self.teleop_device = make_robot_env(self.lerobot_config.env)
        self.env_processor, self.action_processor = make_processors(self.online_env, self.teleop_device, self.lerobot_config.env)

        self.obs = None
        self.info = None
        self.transition = None
        self.sum_reward_episode = 0
        self.list_transition_to_send_to_learner = []
        self.episode_intervention = False
        self.episode_intervention_steps = 0
        self.episode_total_steps = 0
    
    def reset(self, task_ids, state_ids):
        self.obs, self.info = self.online_env.reset()
        self.env_processor.reset()
        self.action_processor.reset()

        self.transition = create_transition(observation=self.obs, info=self.info)
        self.transition = self.env_processor(self.transition)
        self.sum_reward_episode = 0
        self.list_transition_to_send_to_learner = []
        self.episode_intervention = False
        self.episode_intervention_steps = 0
        self.episode_total_steps = 0

        obs = self.transition[TransitionKey.OBSERVATION]
        return obs
    
    def step(self, actions):
        actions = torch.as_tensor(actions, dtype=torch.float32)
        if actions.ndim > 1:
            actions = actions[0]

        new_transition = step_env_and_process_transition(
            env=self.online_env,
            transition=self.transition,
            action=actions,
            env_processor=self.env_processor,
            action_processor=self.action_processor,
        )

        next_obs = new_transition[TransitionKey.OBSERVATION]
        executed_action = new_transition[TransitionKey.COMPLEMENTARY_DATA]["teleop_action"]

        reward = new_transition[TransitionKey.REWARD]
        done = new_transition.get(TransitionKey.DONE, False)
        truncated = new_transition.get(TransitionKey.TRUNCATED, False)

        self.sum_reward_episode += float(reward)
        self.episode_total_steps += 1
        self.interaction_step += 1

        # Check for intervention from transition info
        intervention_info = new_transition[TransitionKey.INFO]
        if intervention_info.get(TeleopEvents.IS_INTERVENTION, False):
            self.episode_intervention = True
            self.episode_intervention_steps += 1

        # Create transition for learner (convert to old format)
        self.list_transition_to_send_to_learner.append(
            Transition(
                state=self.obs,
                action=executed_action,
                reward=reward,
                next_state=next_obs,
                done=done,
                truncated=truncated,
                complementary_info={},
            )
        )
        self.obs = next_obs
        self.transition = new_transition

        if done or truncated:
            intervention_rate = 0.0
            if self.episode_total_steps > 0:
                intervention_rate = self.episode_intervention_steps / self.episode_total_steps

            logger.info(f"Global step {self.interaction_step}: Episode reward: {self.sum_reward_episode} \
                        Episode reward: {self.sum_reward_episode}, Episode steps: {self.episode_total_steps}, \
                        Episode intervention: {self.episode_intervention}, Intervention rate: {intervention_rate:.2f}")

        return {
            "obs": next_obs,
            "reward": float(reward),
            "done": bool(done),
            "truncated": bool(truncated),
            "extra_info": {
                "is_intervention": intervention_info.get(TeleopEvents.IS_INTERVENTION, False),
                "executed_action": executed_action,
            }
        }
    
    def close(self):
        if self.teleop_device is not None and hasattr(self.teleop_device, "disconnect"):
            self.teleop_device.disconnect()
        if self.online_env is not None and hasattr(self.online_env, "close"):
            self.online_env.close()


def _terminate_runtime_process(rank: int, stage_id: int) -> None:
    global _RUNTIME
    try:
        if _RUNTIME is not None:
            _RUNTIME.close()
    except Exception:
        logger.exception("Failed to close LeRobot runtime during forced termination")

    try:
        clear_ipc(rank=rank, stage_id=stage_id)
    except Exception:
        logger.exception("Failed to clear IPC during forced termination")

    os._exit(0)


def _watch_owner_process(owner_pid: int, rank: int, stage_id: int) -> None:
    while True:
        if _STOP:
            return

        try:
            os.kill(owner_pid, 0)
        except ProcessLookupError:
            logger.warning(
                "Owner process %s is gone, shutting down LeRobot runtime for rank=%s stage=%s",
                owner_pid,
                rank,
                stage_id,
            )
            _terminate_runtime_process(rank=rank, stage_id=stage_id)
        except PermissionError:
            return

        time.sleep(1.0)


def start_lerobot_runtime(config_path: str, rank: int, stage_id: int, owner_pid: int | None = None) -> None:
    global _RUNTIME
    logger.info("LeRobot runtime started with config: %s", config_path)
    setup_ipc(rank=rank, stage_id=stage_id)
    runtime = LerobotRuntime(config_path=config_path, rank=rank, stage_id=stage_id)
    _RUNTIME = runtime

    if owner_pid is not None:
        threading.Thread(
            target=_watch_owner_process,
            args=(owner_pid, rank, stage_id),
            name=f"lerobot-runtime-owner-watchdog-{rank}-{stage_id}",
            daemon=True,
        ).start()

    while not _STOP:
        msg = recv_obj(rank=rank, stage_id=stage_id)
        reply = None
        if msg.get("type") == "reset":
            reply = runtime.reset(
                task_ids=msg.get("content", {}).get("task_ids"),
                state_ids=msg.get("content", {}).get("state_ids")
            )
        elif msg.get("type") == "step":
            reply = runtime.step(actions=msg.get("content", {}).get("actions"))
        else:
            logger.warning("Received unknown message type: %s", msg.get("type"))
            reply = {"status": "error", "message": f"Unknown message type: {msg.get('type')}"}

        reply_obj(reply, rank=rank, stage_id=stage_id)
    
    runtime.close()
    clear_ipc(rank=rank, stage_id=stage_id)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True, type=str)
    parser.add_argument("--rank", required=True, type=int)
    parser.add_argument("--stage_id", required=True, type=int)
    parser.add_argument("--owner_pid", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    signal.signal(signal.SIGTERM, _handle_stop_signal)
    signal.signal(signal.SIGINT, _handle_stop_signal)
    start_lerobot_runtime(args.config_path, args.rank, args.stage_id, owner_pid=args.owner_pid)


if __name__ == "__main__":
    main()
