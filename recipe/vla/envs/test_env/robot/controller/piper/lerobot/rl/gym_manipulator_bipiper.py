#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import logging

from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.utils.utils import init_logging

# 复用你项目原生的 make_robot_env（不是 json->ns 那个）
from lerobot.rl.gym_manipulator import make_robot_env


@parser.wrap()
def main(cfg: TrainRLServerPipelineConfig):
    cfg.validate()

    init_logging(log_file=None, display_pid=False)
    logging.info("[BOOT] Creating robot env ...")

    env, teleop = make_robot_env(cfg=cfg.env)

    try:
        logging.info("[BOOT] Env created. Resetting ...")
        obs, info = env.reset()
        logging.info("[BOOT] Reset done. Env is UP. Idling... Ctrl+C to exit.")

        while True:
            time.sleep(5)

    except KeyboardInterrupt:
        logging.info("[BOOT] Interrupted by user.")
    finally:
        try:
            env.close()
        except Exception:
            pass
        if teleop is not None:
            try:
                teleop.disconnect()
            except Exception:
                pass
        logging.info("[BOOT] Clean exit.")


if __name__ == "__main__":
    main()
