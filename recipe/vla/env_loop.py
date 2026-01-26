# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import logging
import os
import time
from collections import defaultdict
from statistics import mean
import sys 
import numpy as np
import torch
from omegaconf import DictConfig

from verl import DataProto
from verl.single_controller.ray import RayWorkerGroup

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

def force_print(*args, **kwargs):
    """强制输出并刷新的 print 函数"""
    print(*args, **kwargs, flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
def estimate_dataproto_size(data_proto: DataProto) -> int:
    """估算 DataProto 的数据大小（字节）"""
    total_size = 0
    
    # 计算 tensor batch 大小
    if data_proto.batch is not None:
        for key in data_proto.batch.keys():
            tensor = data_proto.batch[key]
            if isinstance(tensor, torch.Tensor):
                total_size += tensor.numel() * tensor.element_size()
    
    # 计算 non_tensor_batch 大小（粗略估计）
    if data_proto.non_tensor_batch:
        for key, val in data_proto.non_tensor_batch.items():
            if isinstance(val, np.ndarray):
                total_size += val.nbytes
            elif isinstance(val, (list, tuple)):
                # 粗略估计字符串列表大小
                total_size += sum(len(str(v)) for v in val)
    
    return total_size

#监控数据传输延时和推理延迟
class PerformanceMonitor:
    """性能监控器，记录各个步骤的耗时"""
    def __init__(self):
        self.timings = defaultdict(list)
        self.data_sizes = defaultdict(list)  # 记录数据大小（字节）
        self.step_count = 0
    
    def record(self, name: str, duration: float):
        """记录一次耗时"""
        self.timings[name].append(duration)
    
    def record_with_size(self, name: str, duration: float, size_bytes: int):
        """记录耗时和数据大小"""
        self.timings[name].append(duration)
        self.data_sizes[name].append(size_bytes)
    
    def report(self):
        """打印性能报告"""
        force_print("\n" + "="*80)
        force_print("性能监控报告 (Performance Report)")
        force_print("="*80)
        
        for name, durations in sorted(self.timings.items()):
            if durations:
                force_print(f"\n【{name}】")
                force_print(f"  调用次数: {len(durations)}")
                force_print(f"  平均耗时: {mean(durations):.4f} s")
                force_print(f"  最小耗时: {min(durations):.4f} s")
                force_print(f"  最大耗时: {max(durations):.4f} s")
                force_print(f"  总耗时: {sum(durations):.4f} s")
                
                # 如果有数据大小记录，打印带宽信息
                if name in self.data_sizes and self.data_sizes[name]:
                    sizes = self.data_sizes[name]
                    avg_size_mb = mean(sizes) / (1024 * 1024)
                    total_size_mb = sum(sizes) / (1024 * 1024)
                    avg_duration = mean(durations)
                    if avg_duration > 0:
                        bandwidth_mbps = avg_size_mb / avg_duration
                        force_print(f"  平均数据大小: {avg_size_mb:.2f} MB")
                        force_print(f"  总数据传输量: {total_size_mb:.2f} MB")
                        force_print(f"  估算带宽: {bandwidth_mbps:.2f} MB/s")
        
        force_print("\n" + "="*80)


class EnvLoop:
    """An env loop manages interactions between models and vectorized environments. It's designed for computationally
    intensive environments, such as robotics simulators."""

    def __init__(self, env_wg: RayWorkerGroup, rollout_wg: RayWorkerGroup, config: DictConfig):
        """
        Initialize the EnvLoop.

        Args:
            env_wg (RayWorkerGroup): Environment worker group.
            rollout_wg (RayWorkerGroup): Rollout worker group for model inference.
            config (DictConfig): YAML config.
        """
        logger.info("EnvLoop.__init__: Starting initialization...")
        self.env_wg = env_wg
        self.rollout_wg = rollout_wg
        self.config = config
        # Extract relevant configuration
        logger.info("EnvLoop.__init__: Extracting configuration...")
        self.max_interactions = config.env.train.max_episode_steps // config.env.actor.model.num_action_chunks
        self.stage_num = config.env.rollout.pipeline_stage_num
        self.num_envs_per_worker = config.env.train.num_envs
        self.action_dim = config.env.actor.model.action_dim
        self.num_action_chunks = config.env.actor.model.num_action_chunks
        # Derived properties
        self.total_envs = self.env_wg.world_size * self.num_envs_per_worker
        logger.info(f"EnvLoop.__init__: total_envs={self.total_envs}, stage_num={self.stage_num}")
        if self.total_envs % self.stage_num != 0:
            raise ValueError(f"Total envs ({self.total_envs}) must be divisible by stage_num ({self.stage_num})")
        self.envs_per_stage = self.total_envs // self.stage_num

        logger.info(f"EnvLoop.__init__: Calling env_wg.init_worker() with simulator_type={config.env.train.simulator_type}...")
        self.env_wg.init_worker()
        logger.info("EnvLoop.__init__: init_worker() completed, calling init_simulator()...")
        self.env_wg.init_simulator()
        logger.info("EnvLoop.__init__: Initialization completed successfully!")
        
        # 初始化性能监控器
        self.perf_monitor = PerformanceMonitor()
        self.enable_perf_monitoring = config.get("enable_perf_monitoring", True)

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Split input batch and dispatch to env loop workers.

        Args:
            prompts (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
        """
        import time
        force_print(f"[EnvLoop.generate_sequences] Starting...")
        
        loop = asyncio.get_event_loop()
        
        force_print(f"[EnvLoop.generate_sequences] Calling switch_to_rollout()...")
        t_start = time.perf_counter()
        self.rollout_wg.switch_to_rollout()
        force_print(f"[EnvLoop.generate_sequences] switch_to_rollout() completed in {time.perf_counter() - t_start:.3f}s")
        
        force_print(f"[EnvLoop.generate_sequences] Calling run()...")
        t_start = time.perf_counter()
        output = loop.run_until_complete(self.run(prompts))
        force_print(f"[EnvLoop.generate_sequences] run() completed in {time.perf_counter() - t_start:.3f}s")
        
        force_print(f"[EnvLoop.generate_sequences] Calling switch_to_train()...")
        t_start = time.perf_counter()
        self.rollout_wg.switch_to_train()
        force_print(f"[EnvLoop.generate_sequences] switch_to_train() completed in {time.perf_counter() - t_start:.3f}s")
        
        force_print(f"[EnvLoop.generate_sequences] Completed!")
        return output

    async def run(self, prompts: DataProto) -> DataProto:
        """
        Run the environment interaction loop.
        This method orchestrates a pipelined process:
        1. Resets environments to specified initial states.
        2. In a loop, it gets actions from the rollout workers and applies them to the environments.
        3. Collects all trajectory data (observations, actions, rewards, dones).
        4. Formats and returns the collected trajectories as a single batch.
        Args:
            prompts (DataProto): Contains initial state IDs and other settings.
                                 - 'non_tensor_batch.state_ids': A numpy array of state IDs to reset envs.
        Returns:
            DataProto: A batch containing the complete trajectories.
        """
        import ray
        import os
        
        initial_state_ids = prompts.non_tensor_batch["state_ids"]
        task_ids = prompts.non_tensor_batch["task_ids"]

        # 【调试信息】 打印当前进程和节点信息
        force_print(f"[EnvLoop.run] Current process PID: {os.getpid()}")
        force_print(f"[EnvLoop.run] Current node IP: {ray.util.get_node_ip_address()}")
        force_print(f"[EnvLoop.run] env_wg world_size: {self.env_wg.world_size}")
        
        # 【性能监控】 环境重置 (NodeB -> NodeA)
        t_start = time.perf_counter()
        force_print(f"[EnvLoop.run] Creating reset_prompts with state_ids={initial_state_ids}, task_ids={task_ids}")
        
        # 使用 ray.put() 显式将数据放入对象存储，确保数据持久化
        # 这可以避免因为本地引用被回收导致的 OwnerDiedError
        reset_prompts = DataProto.from_dict(non_tensors={"state_ids": initial_state_ids, "task_ids": task_ids})
        force_print(f"[EnvLoop.run] reset_prompts created, type={type(reset_prompts)}")
        force_print(f"[EnvLoop.run] reset_prompts.non_tensor_batch keys: {list(reset_prompts.non_tensor_batch.keys())}")
        
        force_print(f"[EnvLoop.run] Calling reset_envs_to_state_ids...")
        force_print(f"[EnvLoop.run] env_wg type: {type(self.env_wg)}")
        force_print(f"[EnvLoop.run] env_wg workers: {len(self.env_wg.workers)}")
        
        import time as time_module
        call_start = time_module.perf_counter()
        try:
            reset_results = self.env_wg.reset_envs_to_state_ids(reset_prompts)
            call_duration = time_module.perf_counter() - call_start
            force_print(f"[EnvLoop.run] reset_envs_to_state_ids returned successfully in {call_duration:.3f}s")
            force_print(f"[EnvLoop.run] reset_results type: {type(reset_results)}")
        except Exception as e:
            force_print(f"[EnvLoop.run] reset_envs_to_state_ids FAILED with error: {type(e).__name__}: {e}")
            import traceback
            force_print(f"[EnvLoop.run] Traceback: {traceback.format_exc()}")
            # 打印更多调试信息
            force_print(f"[EnvLoop.run] Checking env_wg workers status...")
            try:
                for i, worker in enumerate(self.env_wg.workers):
                    try:
                        # 尝试 ping worker 检查是否存活
                        ray.get(worker.__ray_ready__.remote(), timeout=5)
                        force_print(f"[EnvLoop.run] Worker {i} is alive")
                    except Exception as worker_e:
                        force_print(f"[EnvLoop.run] Worker {i} check failed: {type(worker_e).__name__}: {worker_e}")
            except Exception as check_e:
                force_print(f"[EnvLoop.run] Failed to check workers: {check_e}")
            raise
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        reset_duration = time.perf_counter() - t_start
        if self.enable_perf_monitoring:
            # 计算回传数据大小
            reset_data_size = estimate_dataproto_size(reset_results)
            self.perf_monitor.record_with_size("1. Env Reset (NodeB->NodeA)", reset_duration, reset_data_size)
            force_print(f"[PerfMonitor] Env Reset: "
                       f"耗时={reset_duration:.4f}s, "
                       f"数据大小={reset_data_size/(1024*1024):.2f}MB, "
                       f"带宽={reset_data_size/(1024*1024)/reset_duration:.2f}MB/s")

        staged_obs = self._restructure_obs_data(reset_results)
        # --- Pipeline state ---
        trajectories = {i: [] for i in range(self.stage_num)}  # To store (obs, action, rew, done) tuples
        rollout_futures = {}
        env_step_futures = {}
        # is_complete = torch.zeros((self.total_envs,), dtype=torch.bool)

        # 【性能监控】 初始模型推理
        for stage_id in range(self.stage_num):
            trajectories[stage_id].append({})
            vla_input = staged_obs[stage_id]
            vla_input.meta_info = prompts.meta_info  # Pass along rollout config
            
            t_infer_start = time.perf_counter()
            rollout_futures[stage_id] = self.rollout_wg.generate_sequences(vla_input)
            # 注意：这里只是提交任务，实际执行在get()时
            
        for step in range(self.max_interactions):
            for stage_id in range(self.stage_num):
                # 【性能监控】 等待模型推理结果 (NodeA上的推理)
                t_infer_wait_start = time.perf_counter()
                action_result: DataProto = rollout_futures[stage_id].get()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                infer_duration = time.perf_counter() - t_infer_wait_start
                if self.enable_perf_monitoring:
                    self.perf_monitor.record("2. Model Inference (NodeA)", infer_duration)
                
                trajectories[stage_id][-1]["action"] = action_result
                action_data = DataProto.from_dict(
                    non_tensors={"actions": action_result.batch["action"].cpu().numpy()},
                    meta_info={"stage_id": stage_id},
                )
                
                # 【性能监控】 发送action到环境 (NodeA -> NodeB)
                t_action_send_start = time.perf_counter()
                env_step_futures[stage_id] = self.env_wg.env_interact_step(action_data)
                # 注意：这里只是提交任务

            staged_next_obs = {}
            for stage_id in range(self.stage_num):
                # 【性能监控】 等待环境step结果 (NodeB执行+回传到NodeA)
                force_print(f"[EnvLoop] Step {step}, Stage {stage_id}: 开始等待 env_step 结果...")
                force_print(f"[EnvLoop] env_step_futures[{stage_id}] type: {type(env_step_futures[stage_id])}")
                t_env_wait_start = time.perf_counter()
                env_result: DataProto = env_step_futures[stage_id].get()
                force_print(f"[EnvLoop] Step {step}, Stage {stage_id}: 收到 env_step 结果!")
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                env_step_duration = time.perf_counter() - t_env_wait_start
                
                # 【性能监控】 计算回传数据大小并记录
                if self.enable_perf_monitoring:
                    data_size = estimate_dataproto_size(env_result)
                    self.perf_monitor.record_with_size(
                        "3. Env Step+Transfer (NodeB->NodeA)", 
                        env_step_duration, 
                        data_size
                    )
                    # 打印每次传输的详细信息
                    force_print(f"[PerfMonitor] Step {step}, Stage {stage_id}: "
                               f"耗时={env_step_duration:.4f}s, "
                               f"数据大小={data_size/(1024*1024):.2f}MB, "
                               f"带宽={data_size/(1024*1024)/env_step_duration:.2f}MB/s")

                # Store rewards and terminations
                trajectories[stage_id][-1]["rew"] = env_result.batch["rews"]
                trajectories[stage_id][-1]["done"] = env_result.batch["terminations"]

                # Prepare next observation
                # 使用模型需要的三张图像: head_image, left_wrist_image, right_wrist_image
                next_obs = DataProto(
                    batch=env_result.batch.select("head_image", "left_wrist_image", "right_wrist_image", "state"),
                    non_tensor_batch={"task_descriptions": env_result.non_tensor_batch["task_descriptions"]},
                )
                staged_next_obs[stage_id] = next_obs

                if step < self.max_interactions - 1:
                    # Start next trajectory step
                    trajectories[stage_id].append({})

                    # Send next observation to model
                    vla_input = staged_next_obs[stage_id]
                    vla_input.meta_info = prompts.meta_info  # Pass along rollout config
                    rollout_futures[stage_id] = self.rollout_wg.generate_sequences(vla_input)
        self.env_wg.finish_rollout()
        
        # 【性能监控】 打印报告
        if self.enable_perf_monitoring:
            self.perf_monitor.report()

        return self._collate_trajectories(trajectories, initial_state_ids, meta_info=prompts.meta_info)

    def _restructure_obs_data(self, data_proto: DataProto) -> list[DataProto]:
        """Reshapes flat observation data from env_wg into a list of per-stage DataProto objects."""
        # env_wg returns a flat batch ordered by [worker0_stage0, worker0_stage1, ...,
        # worker1_stage0, worker1_stage1, ...]
        # First, un-flatten by worker, then by stage

        num_workers = self.env_wg.world_size

        staged_data = [[] for _ in range(self.stage_num)]
        chunks = data_proto.chunk(num_workers)
        for worker_chunk in chunks:
            stage_chunks = worker_chunk.chunk(self.stage_num)
            for stage_id, data in enumerate(stage_chunks):
                staged_data[stage_id].append(data)

        # Concatenate data from all workers for each stage
        return [DataProto.concat(data_list) for data_list in staged_data]

    def _collate_trajectories(self, trajectories: dict, initial_state_ids: np.ndarray, meta_info) -> DataProto:
        """
        Collates the collected trajectory data into the final batch format.
        """
        flat_trajs = [{} for _ in range(len(trajectories[0]))]
        for stage_id in range(self.stage_num):
            for step_idx, step_data in enumerate(trajectories[stage_id]):
                if not flat_trajs[step_idx]:  # if dict is empty
                    flat_trajs[step_idx] = step_data
                else:
                    # Concatenate DataProto objects
                    for key, value in step_data.items():
                        if isinstance(value, DataProto):
                            flat_trajs[step_idx][key] = DataProto.concat([flat_trajs[step_idx][key], value])
                        elif isinstance(value, torch.Tensor):
                            flat_trajs[step_idx][key] = torch.cat([flat_trajs[step_idx][key], value], dim=0)

        # iterate all action batch keys (e.g., action, images, pixel_values, input_ids, ...)
        batch_dict = {}
        action_batch_keys = list(flat_trajs[0]["action"].batch.keys())
        for key in action_batch_keys:
            per_step_values = [step["action"].batch[key] for step in flat_trajs]
            batch_dict[key] = torch.stack(per_step_values, dim=1)

        batch_dict["complete"] = torch.stack([step["done"] for step in flat_trajs], dim=1).squeeze(-1)
        batch_dict["env_state_id"] = torch.from_numpy(initial_state_ids.astype(int))

        return DataProto.from_single_dict(batch_dict, meta_info=meta_info)
