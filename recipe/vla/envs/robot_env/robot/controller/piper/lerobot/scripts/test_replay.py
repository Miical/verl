#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import logging
import os
import tempfile
import time

import ray

logging.getLogger("can.interfaces.socketcan").setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO)


@ray.remote(resources={"node:B": 0.1})
class EnvKeeper:
    """跑在 node:B：只负责把真实机器人 env 拉起来并常驻。"""

    def __init__(self, env_json_text: str):
        # 0) actor 内触发注册
        from lerobot.robots import piper_follower  # noqa: F401
        from lerobot.teleoperators import keyboard  # noqa: F401
        from lerobot.teleoperators import gamepad  # noqa: F401

        # cameras 可选注册（缺依赖也别死）
        try:
            from lerobot.cameras import realsense  # noqa: F401
        except Exception as e:
            logging.info(f"[EnvKeeper] realsense import failed (ok): {e}")
        try:
            from lerobot.cameras import dabai  # noqa: F401
        except Exception as e:
            logging.info(f"[EnvKeeper] dabai import failed (ok): {e}")

        # 1) 反序列化 env dict，并移除 env 顶层的无关字段
        env_cfg_dict = json.loads(env_json_text)
        env_cfg_dict.pop("type", None)  # ✅ 关键：HILSerlRobotEnvConfig 不认这个字段

        # 2) node:B 写临时 env 配置（避免 B 读不到 /shared_disk）
        td = tempfile.mkdtemp(prefix="lerobot_envcfg_")
        self.env_cfg_path = os.path.join(td, "env_cfg.json")
        with open(self.env_cfg_path, "w") as f:
            json.dump(env_cfg_dict, f, ensure_ascii=False, indent=2)

        # 3) 只 parse env dataclass（绕开 policy）
        import draccus
        from lerobot.envs.configs import HILSerlRobotEnvConfig
        from lerobot.rl.gym_manipulator_bipiper import make_robot_env  # 你当前栈里在用的那个

        env_cfg: HILSerlRobotEnvConfig = draccus.parse(
            config_class=HILSerlRobotEnvConfig,
            config_path=self.env_cfg_path,
            args=[],
        )

        if hasattr(env_cfg, "validate"):
            env_cfg.validate()

        logging.info("[EnvKeeper] Creating robot env ...")
        self.env, self.teleop = make_robot_env(env_cfg)

        logging.info("[EnvKeeper] Reset env ...")
        self.env.reset()

        logging.info(f"[EnvKeeper] Env is UP on node:B. env_cfg_path={self.env_cfg_path}")

    def ping(self):
        return {"ok": True, "env_cfg_path_on_B": self.env_cfg_path}

    def hold(self, seconds: float = 3600.0):
        time.sleep(float(seconds))
        return True

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass
        try:
            if self.teleop is not None:
                self.teleop.disconnect()
        except Exception:
            pass
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_path", type=str, required=True)
    ap.add_argument("--ray_address", type=str, default="auto")
    ap.add_argument("--hold_s", type=float, default=999999)
    args = ap.parse_args()

    # Driver 端读 pipeline json
    with open(args.config_path, "r") as f:
        full_cfg = json.load(f)

    # 只取 env 段，并移除 env 顶层 type（双保险）
    env_cfg_dict = dict(full_cfg["env"])
    env_cfg_dict.pop("type", None)

    env_json_text = json.dumps(env_cfg_dict, ensure_ascii=False, indent=2)

    ray.init(address=args.ray_address, ignore_reinit_error=True)

    keeper = EnvKeeper.remote(env_json_text)
    logging.info("[Driver] " + str(ray.get(keeper.ping.remote())))

    try:
        ray.get(keeper.hold.remote(args.hold_s))
    except KeyboardInterrupt:
        pass
    finally:
        try:
            ray.get(keeper.close.remote())
        except Exception:
            pass


if __name__ == "__main__":
    main()
