import torch
import argparse
import logging
import signal
import time

import draccus

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
from .ipc_channel import clear_ipc, recv_obj, reply_obj, setup_ipc


logger = logging.getLogger(__name__)
_STOP = False


def _handle_stop_signal(signum, frame):
    del frame
    global _STOP
    _STOP = True
    logger.info("Received signal %s, stopping lerobot runtime.", signum)

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
        logger.info("Resetting LeRobot runtime.")
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

        # obs = self.transition[TransitionKey.OBSERVATION]
        # return obs

        return {"status": "ok"}
    
    def step(self, actions):
        raw = self.online_env.get_raw_joint_positions()
        neutral_action = torch.tensor(
            [raw[f"{k}.pos"] for k in self.online_env.robot.bus.motors], dtype=torch.float32
        )
        time.sleep(0.1)
        actions = neutral_action

        # ====

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
        self.transition = new_transition

        if done or truncated:
            intervention_rate = 0.0
            if self.episode_total_steps > 0:
                intervention_rate = self.episode_intervention_steps / self.episode_total_steps

            logger.info(f"Global step {self.interaction_step}: Episode reward: {self.sum_reward_episode} \
                        Episode reward: {self.sum_reward_episode}, Episode steps: {self.episode_total_steps}, \
                        Episode intervention: {self.episode_intervention}, Intervention rate: {intervention_rate:.2f}")

        return {"status": "ok"}
    
    def close(self):
        if self.teleop_device is not None and hasattr(self.teleop_device, "disconnect"):
            self.teleop_device.disconnect()
        if self.online_env is not None and hasattr(self.online_env, "close"):
            self.online_env.close()
        

def start_lerobot_runtime(config_path: str, rank: int, stage_id: int) -> None:
    logger.info("LeRobot runtime started with config: %s", config_path)
    setup_ipc(rank=rank, stage_id=stage_id)
    runtime = LerobotRuntime(config_path=config_path, rank=rank, stage_id=stage_id)

    while not _STOP:
        msg = recv_obj(rank=rank, stage_id=stage_id)
        reply = None
        if msg.get("type") == "reset":
            reply = runtime.reset(
                task_ids=msg.get("content", {}).get("task_ids"),
                state_ids=msg.get("content", {}).get("state_ids")
            )
        elif msg.get("type") == "step":
            reply = runtime.step(actions=None)
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
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    signal.signal(signal.SIGTERM, _handle_stop_signal)
    signal.signal(signal.SIGINT, _handle_stop_signal)
    start_lerobot_runtime(args.config_path, args.rank, args.stage_id)


if __name__ == "__main__":
    main()
