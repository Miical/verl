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


logger = logging.getLogger(__name__)
_STOP = False


def _handle_stop_signal(signum, frame):
    del frame
    global _STOP
    _STOP = True
    logger.info("Received signal %s, stopping lerobot runtime.", signum)


def start_lerobot_runtime(config_path: str) -> None:
    logger.info("LeRobot runtime started with config: %s", config_path)

    lerobot_config = draccus.parse(
        config_class=GymManipulatorConfig,
        config_path=config_path,
        args=[],
    )

    online_env, teleop_device = make_robot_env(lerobot_config.env)
    env_processor, action_processor = make_processors(online_env, teleop_device, lerobot_config.env)

    obs, info = online_env.reset()
    env_processor.reset()
    action_processor.reset()

    # Process initial observation
    transition = create_transition(observation=obs, info=info)
    transition = env_processor(transition)

    # NOTE: For the moment we will solely handle the case of a single environment
    sum_reward_episode = 0
    list_transition_to_send_to_learner = []
    episode_intervention = False
    # Add counters for intervention rate calculation
    episode_intervention_steps = 0
    episode_total_steps = 0


    for interaction_step in range(10000):
        obs = transition[TransitionKey.OBSERVATION]

        raw = online_env.get_raw_joint_positions()
        neutral_action = torch.tensor(
            [raw[f"{k}.pos"] for k in online_env.robot.bus.motors], dtype=torch.float32
        )
        def policy(obs):
            import time
            time.sleep(0.2)
            return neutral_action

        action = policy(obs)

        new_transition = step_env_and_process_transition(
            env=online_env,
            transition=transition,
            action=action,
            env_processor=env_processor,
            action_processor=action_processor,
        )

        next_obs = new_transition[TransitionKey.OBSERVATION]
        executed_action = new_transition[TransitionKey.COMPLEMENTARY_DATA]["teleop_action"]


        reward = new_transition[TransitionKey.REWARD]
        done = new_transition.get(TransitionKey.DONE, False)
        truncated = new_transition.get(TransitionKey.TRUNCATED, False)

        sum_reward_episode += float(reward)
        episode_total_steps += 1

        # Check for intervention from transition info
        intervention_info = new_transition[TransitionKey.INFO]
        if intervention_info.get(TeleopEvents.IS_INTERVENTION, False):
            episode_intervention = True
            episode_intervention_steps += 1
        

        complementary_info = {
            "discrete_penalty": torch.tensor(
                [new_transition[TransitionKey.COMPLEMENTARY_DATA].get("discrete_penalty", 0.0)]
            ),
        }
        # Create transition for learner (convert to old format)
        list_transition_to_send_to_learner.append(
            Transition(
                state=obs,
                action=executed_action,
                reward=reward,
                next_state=next_obs,
                done=done,
                truncated=truncated,
                complementary_info=complementary_info,
            )
        )

        transition = new_transition

        if done or truncated:
            logging.info(f"[ACTOR] Global step {interaction_step}: Episode reward: {sum_reward_episode}")

            if len(list_transition_to_send_to_learner) > 0:
                # TODO: send list_transition_to_send_to_learner to learner
                list_transition_to_send_to_learner = []
            
            # Calculate intervention rate
            intervention_rate = 0.0
            if episode_total_steps > 0:
                intervention_rate = episode_intervention_steps / episode_total_steps

            interactions_queue = []
            interactions_queue.append(
                {
                    "Episodic reward": sum_reward_episode,
                    "Interaction step": interaction_step,
                    "Episode intervention": int(episode_intervention),
                    "Intervention rate": intervention_rate,
                }
            )
            # TODO: send interactions_queue 

            # Reset intervention counters and environment
            sum_reward_episode = 0.0
            episode_intervention = False
            episode_intervention_steps = 0
            episode_total_steps = 0

            # Reset environment and processors
            obs, info = online_env.reset()
            env_processor.reset()
            action_processor.reset()

            # Process initial observation
            transition = create_transition(observation=obs, info=info)
            transition = env_processor(transition)



    while not _STOP:
        time.sleep(1.0)

    if teleop_device is not None and hasattr(teleop_device, "disconnect"):
        teleop_device.disconnect()
    if online_env is not None and hasattr(online_env, "close"):
        online_env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True, type=str)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    signal.signal(signal.SIGTERM, _handle_stop_signal)
    signal.signal(signal.SIGINT, _handle_stop_signal)
    start_lerobot_runtime(args.config_path)


if __name__ == "__main__":
    main()
