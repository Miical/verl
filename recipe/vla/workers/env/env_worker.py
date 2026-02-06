# Copyright 2025 The RLinf Authors.
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import itertools

import torch
from omegaconf import DictConfig
from torch.distributed.device_mesh import init_device_mesh
import cv2
import numpy as np
import time
from recipe.vla.envs.action_utils import prepare_actions
from recipe.vla.workers.env.env_manager import EnvManager
from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.utils.device import (
    get_device_name,
)
from verl.utils.distributed import initialize_global_process_group_ray

def images_decoding(encoded_data, valid_len=None):
    """解码图像并记录耗时"""
    decode_start = time.perf_counter()
    imgs = []
    for data in encoded_data:
        if valid_len is not None:
            data = data[:valid_len]
        nparr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        imgs.append(img)
    decode_time = time.perf_counter() - decode_start
    return imgs, decode_time
def put_tensor_cpu(data_dict):
    for key, value in data_dict.items():
        if isinstance(value, dict):
            data_dict[key] = put_tensor_cpu(value)
        if isinstance(value, torch.Tensor):
            data_dict[key] = value.cpu().contiguous()
    return data_dict


def create_env_batch(obs, rews, dones, infos, meta=None):
    ret_dict = {"obs": obs, "rews": rews, "dones": dones, "infos": infos}
    if meta is not None:
        ret_dict.update(meta=meta)

    ret_dict = put_tensor_cpu(ret_dict)
    return ret_dict


def decode_images_if_needed(images_dict):
    """解码图像数据（如果是编码格式）
    
    Args:
        images_dict: 包含图像tensor的字典
        
    Returns:
        解码后的图像字典
    """
    decoded_dict = {}
    for key, img_tensor in images_dict.items():
        if isinstance(img_tensor, torch.Tensor) and img_tensor.dtype == torch.uint8 and img_tensor.ndim == 2:
            # 这是编码的图像数据 (batch, max_len)
            # 需要解码
            encoded_data = []
            for i in range(img_tensor.shape[0]):
                img_bytes = img_tensor[i].cpu().numpy().tobytes()
                encoded_data.append(img_bytes)
            
            # 解码图像
            decoded_imgs, decode_time = images_decoding(encoded_data)
            # 转换为tensor (batch, H, W, C)
            img_tensors = torch.stack([torch.from_numpy(img) for img in decoded_imgs])
            decoded_dict[key] = img_tensors
            print(f"[EnvWorker] 解码 {key}: {img_tensors.shape}, 耗时: {decode_time:.4f}s", flush=True)
        else:
            # 已经是解码后的图像或其他数据
            decoded_dict[key] = img_tensor
    return decoded_dict


def create_env_batch_dataproto(obs, rews, terminations, truncations, infos, meta=None):
    ret_dict = {"obs": obs, "rews": rews, "terminations": terminations, "truncations": truncations, "infos": infos}
    if meta is not None:
        ret_dict.update(meta=meta)

    ret_dict = put_tensor_cpu(ret_dict)
    
    # ⚠️ 保持 JPEG 编码格式传输，减少数据传输量
    # 图像解码将在 Rollout Worker (NodeA) 进行
    images_and_states = ret_dict["obs"]["images_and_states"]
    
    tensor_batch = {
        "head_image": images_and_states.get("head_image"),
        "left_wrist_image": images_and_states.get("left_wrist_image"),
        "right_wrist_image": images_and_states.get("right_wrist_image"),
        "state": images_and_states["state"],
        "rews": ret_dict["rews"],
        "terminations": ret_dict["terminations"],
        "truncations": ret_dict["truncations"],
    }
    
    # 打印传输数据信息
    for k, v in tensor_batch.items():
        if v is not None and hasattr(v, 'shape'):
            print(f"[create_env_batch_dataproto] {k}: shape={v.shape}, dtype={v.dtype}", flush=True)
    
    non_tensor_batch = {"task_descriptions": obs["task_descriptions"]}
    output = DataProto.from_dict(tensors=tensor_batch, non_tensors=non_tensor_batch)

    return output


class EnvWorker(Worker):
    def __init__(self, config: DictConfig):
        Worker.__init__(self)
        self.cfg = config
        self.train_video_cnt = 0
        self.eval_video_cnt = 0

        self.simulator_list = []
        self.last_obs_list = []
        self.last_dones_list = []
        self.eval_simulator_list = []

        self.stage_num = self.cfg.rollout.pipeline_stage_num
        initialize_global_process_group_ray(timeout_second=None)
        device_name = get_device_name()
        env_device_mesh = init_device_mesh(device_name, mesh_shape=(self.world_size, 1), mesh_dim_names=["dp", "tp"])
        self._register_dispatch_collect_info("env", dp_rank=env_device_mesh["dp"].get_local_rank(), is_collect=True)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_worker(self):
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"EnvWorker.init_worker: rank={self._rank}, simulator_type={self.cfg.train.simulator_type}, stage_num={self.stage_num}")
        
        if self.cfg.train.simulator_type == "libero":
            from recipe.vla.envs.libero_env.libero_env import LiberoEnv

            for i in range(self.stage_num):
                logger.info(f"  Creating LiberoEnv EnvManager {i+1}/{self.stage_num}...")
                self.simulator_list.append(
                    EnvManager(
                        self.cfg.train,
                        rank=self._rank,
                        world_size=self._world_size,
                        env_cls=LiberoEnv,
                    )
                )

        elif self.cfg.train.simulator_type == "isaac":
            from recipe.vla.envs.isaac_env.isaac_env import IsaacEnv

            for i in range(self.stage_num):
                logger.info(f"  Creating IsaacEnv EnvManager {i+1}/{self.stage_num}...")
                self.simulator_list.append(
                    EnvManager(
                        self.cfg.train,
                        rank=self._rank,
                        world_size=self._world_size,
                        env_cls=IsaacEnv,
                    )
                )
        
        elif self.cfg.train.simulator_type == "robot":
            from recipe.vla.envs.robot_env.robot_env import RealRobotEnv

            for i in range(self.stage_num):
                logger.info(f"  Creating TestEnv EnvManager {i+1}/{self.stage_num}...")
                self.simulator_list.append(
                    EnvManager(
                        self.cfg.train,
                        rank=self._rank,
                        world_size=self._world_size,
                        env_cls=RealRobotEnv,
                    )
                )
        
        else:
            raise NotImplementedError(f"Simulator type {self.cfg.train.simulator_type} not implemented")
        
        logger.info(f"EnvWorker.init_worker: Completed! Created {len(self.simulator_list)} EnvManagers")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_simulator(self):
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"EnvWorker.init_simulator: Starting {self.stage_num} simulators...")
        for i in range(self.stage_num):
            logger.info(f"  Starting simulator {i+1}/{self.stage_num}...")
            self.simulator_list[i].start_simulator()
            logger.info(f"  Simulator {i+1}/{self.stage_num} started successfully!")
        logger.info("EnvWorker.init_simulator: All simulators started!")
        return

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="env"), blocking=False)
    def env_interact_step(self, data: DataProto) -> dict:
        """
        This function is used to interact with the environment.
        """
        import time
        t_step_start = time.perf_counter()
        
        chunk_actions: torch.Tensor = data.non_tensor_batch["actions"]
        stage_id: int = data.meta_info["stage_id"]
        # Pi0.5 Libero is not required
        # chunk_actions = prepare_actions(
        #     simulator_type=self.cfg.train.simulator_type,
        #     raw_chunk_actions=chunk_actions,
        #     num_action_chunks=self.cfg.actor.model.num_action_chunks,
        #     action_dim=self.cfg.actor.model.action_dim,
        # )
        env_info_list = {}

        # 【性能监控】 环境step执行
        t_sim_start = time.perf_counter()
        chunk_obs, chunk_rewards, chunk_terminations, chunk_truncations, infos = self.simulator_list[
            stage_id
        ].chunk_step(chunk_actions)
        sim_time = time.perf_counter() - t_sim_start
        print(f"[EnvWorker] Stage {stage_id} 环境step耗时: {sim_time:.4f}s", flush=True)
        
        # 【调试信息】 打印 chunk_step 返回的观测格式
        print(f"[EnvWorker] Stage {stage_id} chunk_step 返回的观测格式:", flush=True)
        if "images_and_states" in chunk_obs:
            images_and_states = chunk_obs["images_and_states"]
            for key, value in images_and_states.items():
                if value is not None and hasattr(value, 'shape'):
                    print(f"[EnvWorker]   images_and_states[{key}]: shape={value.shape}, dtype={value.dtype}", flush=True)
                else:
                    print(f"[EnvWorker]   images_and_states[{key}]: None", flush=True)
        
        chunk_dones = torch.logical_or(chunk_terminations, chunk_truncations)

        if chunk_dones.any():
            if "final_info" in infos:
                final_info = infos["final_info"]
                for key in final_info["episode"]:
                    env_info_list[key] = final_info["episode"][key][chunk_dones[:, -1]].cpu()

        # 【性能监控】 数据打包
        t_pack_start = time.perf_counter()
        env_batch = create_env_batch_dataproto(
            obs=chunk_obs,
            rews=chunk_rewards,
            terminations=chunk_terminations,
            truncations=chunk_truncations,
            infos=infos,
            meta=env_info_list,
        )
        pack_time = time.perf_counter() - t_pack_start
        
        total_time = time.perf_counter() - t_step_start
        print(f"[EnvWorker] Stage {stage_id} 数据打包耗时: {pack_time:.4f}s, 总耗时: {total_time:.4f}s", flush=True)
        
        # 【调试信息】 打印返回数据的详细信息
        print(f"[EnvWorker] Stage {stage_id} 准备返回 DataProto:", flush=True)
        if env_batch.batch is not None:
            for k in env_batch.batch.keys():
                v = env_batch.batch[k]
                print(f"[EnvWorker]   {k}: shape={v.shape}, dtype={v.dtype}", flush=True)
        print(f"[EnvWorker] Stage {stage_id} 开始返回...", flush=True)
        
        return env_batch

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_all_state_ids(self):
        """Get all available state IDs from the environment."""
        state_ids = self.simulator_list[0].get_all_state_ids()
        return state_ids

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="env"))
    def reset_envs_to_state_ids(self, data: DataProto):
        """Reset environments to specified state IDs.

        Args:
            state_ids: State IDs to reset environments to
        """
        state_ids_list = list(data.non_tensor_batch["state_ids"])
        task_ids_list = list(data.non_tensor_batch["task_ids"])

        assert len(state_ids_list) == self.cfg.train.num_envs * self.stage_num, (
            f"state_ids_list length is {len(state_ids_list)}, but should be {self.cfg.train.num_envs * self.stage_num}"
        )
        result_list = []
        for stage_id in range(self.stage_num):
            if self.cfg.train.simulator_type in ["isaac", "test"]:
                assert (
                    len(
                        set(
                            state_ids_list[
                                stage_id * self.cfg.train.num_envs : (stage_id + 1) * self.cfg.train.num_envs
                            ]
                        )
                    )
                    == 1
                ), f"rollout.n should equal to num_envs for {self.cfg.train.simulator_type}"

            result = self.simulator_list[stage_id].reset_envs_to_state_ids(
                state_ids_list[stage_id * self.cfg.train.num_envs : (stage_id + 1) * self.cfg.train.num_envs],
                task_ids_list[stage_id * self.cfg.train.num_envs : (stage_id + 1) * self.cfg.train.num_envs],
            )
            result_list.append(result)
        output_tensor_dict = {}
        output_non_tensor_dict = {}

        # Handle nested 'images_and_states'
        images_and_states_list = [d[0]["images_and_states"] for d in result_list]
        
        # 模型只需要三张图像: head_image, left_wrist_image, right_wrist_image
        # 跳过 full_image 和 wrist_image 以减少数据传输量
        REQUIRED_IMAGE_KEYS = {"head_image", "left_wrist_image", "right_wrist_image", "state"}
        
        if images_and_states_list:
            # Assuming all dicts in the list have the same keys
            for k in images_and_states_list[0].keys():
                # 跳过不需要的图像键
                if k not in REQUIRED_IMAGE_KEYS and "image" in k.lower():
                    print(f"[EnvWorker] 跳过不需要的图像: {k}", flush=True)
                    continue
                    
                if isinstance(images_and_states_list[0][k], torch.Tensor):
                    # ⚠️ 关键修复：不要在这里解码图像！
                    # 保持 JPEG 编码格式传输，在 Rollout Worker 那边再解码
                    # 这样可以大大减少 Ray 对象存储的传输量（~50KB vs ~450KB）
                    if "image" in k.lower() and images_and_states_list[0][k].dtype == torch.uint8 and images_and_states_list[0][k].ndim == 2:
                        # 直接拼接编码数据，不解码
                        output_tensor_dict[k] = torch.cat([d[k] for d in images_and_states_list])
                        print(f"[EnvWorker] 保持 JPEG 编码传输 {k}: {output_tensor_dict[k].shape} (encoded)", flush=True)
                    else:
                        # For non-image tensors, just concatenate
                        output_tensor_dict[k] = torch.cat([d[k] for d in images_and_states_list])
        # Handle 'task_descriptions'
        task_descriptions_list = [d[0]["task_descriptions"] for d in result_list]
        output_non_tensor_dict["task_descriptions"] = list(itertools.chain.from_iterable(task_descriptions_list))

        output = DataProto.from_dict(tensors=output_tensor_dict, non_tensors=output_non_tensor_dict)
        
        # 调试信息：打印返回数据的大小
        print(f"[EnvWorker] reset_envs_to_state_ids 准备返回数据:", flush=True)
        print(f"[EnvWorker]   tensor keys: {list(output_tensor_dict.keys())}", flush=True)
        for k, v in output_tensor_dict.items():
            print(f"[EnvWorker]   {k}: shape={v.shape}, dtype={v.dtype}, device={v.device}", flush=True)
        print(f"[EnvWorker]   non_tensor keys: {list(output_non_tensor_dict.keys())}", flush=True)
        print(f"[EnvWorker]   task_descriptions: {output_non_tensor_dict['task_descriptions']}", flush=True)
        print(f"[EnvWorker] 开始返回 DataProto...", flush=True)
        
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def finish_rollout(self, mode="train"):
        # reset
        if mode == "train":
            if self.cfg.train.video_cfg.save_video:
                for i in range(self.stage_num):
                    self.simulator_list[i].flush_video(video_sub_dir=f"stage_{i}")
