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
"""
In single GPU rollout, the sequences are generated directly by sampling from the model.
The output will contain
1. output_ids
2. attention_masks (left padding)
3. eos_masks
4. log_probs
"""

import json
import logging
import os
import inspect  # ===================== CHANGED =====================

import numpy as np
import torch
from PIL import Image
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.utils.rnn import pad_sequence
from typing import Any

from recipe.vla.envs.action_utils import center_crop_image, resize_image
from recipe.vla.models.openvla_oft.modeling_prismatic import OpenVLAForActionPrediction
from recipe.vla.models.openvla_oft.processing_prismatic import PrismaticProcessor
from verl import DataProto
from verl.utils.device import get_device_id, get_device_name, get_torch_device
from verl.utils.profiler import simple_timer
from verl.workers.rollout.base import BaseRollout
from recipe.vla.models.pi0_torch.pi0_utils import AlohaInputs, AlohaOutputs

import pdb
logger = logging.getLogger(__name__)

__all__ = ["NaiveRolloutRob", "PI0RolloutRob", "test_pi0_with_lerobot_dataset"]


def pad_sequence_to_length(tensors, max_seq_len, pad_token_id, left_pad=False):
    """
    pad a 2D tensors (e.g. responses, logprobs) in the last dim to max_seq_length.
    input shape: [bs, seq_length]
    output shape: [bs, max_seq_length]
    (0, max_seq_len - tensors.shape[-1]) means right pad to max_seq_length and no left pad
    """
    if tensors.shape[-1] >= max_seq_len:
        return tensors
    pad_tuple = (max_seq_len - tensors.shape[-1], 0) if left_pad else (0, max_seq_len - tensors.shape[-1])
    return torch.nn.functional.pad(tensors, pad_tuple, "constant", pad_token_id)


def process_input(task_descriptions, images_and_states, processor):
    batchdata = {"input_ids": [], "attention_mask": [], "pixel_values": []}

    for i in range(len(task_descriptions)):
        task_description = task_descriptions[i]
        image = resize_image(images_and_states["full_image"][i].cpu().numpy(), (224, 224))
        image = Image.fromarray(image).convert("RGB")
        image = center_crop_image(image)
        prompt = f"In: What action should the robot take to {task_description.lower()}?\nOut:"
        batch_feature = processor(prompt, image)

        input_ids = batch_feature["input_ids"]
        attention_mask = batch_feature["attention_mask"]
        pixel_values = batch_feature["pixel_values"]

        if not torch.all(input_ids[:, -1] == 29871):
            input_ids = torch.cat(
                (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
            )
            attention_mask = torch.cat(
                (attention_mask, torch.unsqueeze(torch.Tensor([True]).bool(), dim=0).to(attention_mask.device)), dim=1
            )

        batchdata["input_ids"].append(input_ids)
        batchdata["attention_mask"].append(attention_mask)
        batchdata["pixel_values"].append(pixel_values)

    device = get_device_id()

    batchdata["input_ids"] = [x.transpose(0, 1) for x in batchdata["input_ids"]]
    batchdata["attention_mask"] = [x.transpose(0, 1) for x in batchdata["attention_mask"]]
    batchdata["input_ids"] = (
        pad_sequence(batchdata["input_ids"], batch_first=True, padding_value=processor.tokenizer.pad_token_id)
        .squeeze(-1)
        .to(device)
    )
    batchdata["attention_mask"] = (
        pad_sequence(batchdata["attention_mask"], batch_first=True, padding_value=0).squeeze(-1).to(device)
    )

    padding_mask = batchdata["input_ids"].ne(processor.tokenizer.pad_token_id)
    assert torch.all(padding_mask == batchdata["attention_mask"].ne(0))
    padding_mask = ~padding_mask
    padding_mask = padding_mask.int()
    sorted_indices = torch.argsort(padding_mask, dim=1, descending=True, stable=True)
    batchdata["input_ids"] = torch.gather(batchdata["input_ids"], 1, sorted_indices)
    batchdata["attention_mask"] = torch.gather(batchdata["attention_mask"], 1, sorted_indices)

    batchdata["pixel_values"] = torch.cat(batchdata["pixel_values"], dim=0).to(device)
    assert torch.all(batchdata["attention_mask"].ne(0) == batchdata["input_ids"].ne(processor.tokenizer.pad_token_id))

    return batchdata


class NaiveRolloutRob(BaseRollout):
    def __init__(
        self,
        model_config: dict,
        module: torch.nn.Module = None,
    ):
        self.model_config = model_config
        if module is not None:
            self.module = module
        else:
            self.module = OpenVLAForActionPrediction.from_pretrained(model_config["path"], trust_remote_code=True)
        self.module.vision_backbone.set_num_images_in_input(1)
        self.processor = PrismaticProcessor.from_pretrained(model_config["path"], trust_remote_code=True)
        dataset_statistics_path = os.path.join(model_config["path"], "dataset_statistics.json")
        if os.path.isfile(dataset_statistics_path):
            with open(dataset_statistics_path) as f:
                norm_stats = json.load(f)
            if isinstance(self.module, FSDP):
                self.module.module.norm_stats = norm_stats
            else:
                self.module.norm_stats = norm_stats
        self.module.eval()

    @torch.no_grad()
    def _generate_one_step(self, prompts: dict, do_sample, temperature, max_prompt_length):
        idx = prompts["input_ids"]  # (bs, prompt_length)
        attention_mask = prompts["attention_mask"]  # left-padded attention_mask
        pixel_values = prompts["pixel_values"]

        with torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
            actions, response = self.module.generate_action_verl(
                input_ids=idx,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                padding_idx=self.processor.tokenizer.pad_token_id,
                do_sample=do_sample,
                unnorm_key="libero_10_no_noops",
                temperature=temperature,
            )

        assert self.processor.tokenizer.pad_token_id is not None

        assert idx.ndim == 2
        idx = pad_sequence_to_length(
            idx, max_seq_len=max_prompt_length, pad_token_id=self.processor.tokenizer.pad_token_id, left_pad=True
        )

        assert attention_mask.ndim == 2
        attention_mask = pad_sequence_to_length(
            attention_mask, max_seq_len=max_prompt_length, pad_token_id=0, left_pad=True
        )

        device_type = get_device_name()
        assert idx.device.type == device_type
        assert response.device.type == device_type
        assert attention_mask.device.type == device_type
        assert pixel_values.device.type == device_type
        batch = {
            "responses": response,
            "input_ids": idx,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "action": actions,
        }

        return batch



    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Generate sequences"""
        do_sample = prompts.meta_info["do_sample"]
        temperature = prompts.meta_info["temperature"]
        max_prompt_length = prompts.meta_info["prompt_length"]
        task_descriptions = prompts.non_tensor_batch["task_descriptions"]
        images_and_states = {"full_image": prompts.batch["full_image"]}
        vla_input = process_input(task_descriptions, images_and_states, self.processor)

        vla_output = self._generate_one_step(vla_input, do_sample, temperature, max_prompt_length)
        batch = DataProto.from_dict(tensors=vla_output)
        return batch

    async def update_weights(self, weights_iterator, **kwargs):
        prefix = "_fsdp_wrapped_module."
        target_state_dict = self.module.state_dict()
        loaded_tensors_count = 0
        for name, param in weights_iterator:
            cleaned_name = name.replace(prefix, "")
            if cleaned_name in target_state_dict:
                target_tensor = target_state_dict[cleaned_name]
                try:
                    target_tensor.copy_(param, non_blocking=True)
                    loaded_tensors_count += 1
                except Exception as e:
                    logger.warning(f"Warning: Failed to copy tensor '{cleaned_name}'. Error: {e}")
            else:
                logger.warning(f"Warning: Failed to copy tensor '{cleaned_name}'. Model has no such key.")
        logger.info(f"Rollout model weights updated. Loaded {loaded_tensors_count} tensors one by one.")

    async def release(self):
        if self.module.device.type == get_device_name():
            logger.info("Releasing rollout model to CPU.")
            self.module.cpu()
            self.device = torch.device("cpu")
            get_torch_device().empty_cache()

    async def resume(self, **kwargs):
        if self.module.device.type == "cpu":
            target_device = get_device_name()
            logger.info(f"Resuming rollout model to device: {target_device}.")
            self.module.to(target_device)
            self.device = torch.device(target_device)


class PI0RolloutRob(NaiveRolloutRob):
    def __init__(
        self,
        model_config: dict,
        module: torch.nn.Module,
        tokenizer: Any,
    ):
        self.model_config = model_config
        self.module = module
        self.tokenizer = tokenizer
        self.aloha_inputs = AlohaInputs(adapt_to_pi=False)
        self.aloha_outputs = AlohaOutputs(original_action_dim=14, adapt_to_pi=False)
        device = next(module.parameters()).device
        self.aloha_inputs.to(device)
        self.aloha_outputs.to(device)
        # 用于测试的数据集相关变量
        self.test_dataset = None

        # 用于保存输入数据的变量
        self.save_inputs_enabled = False
        self.save_inputs_base_path = None
        self.current_step_in_episode = 0
        self._episode_dir = None  # 当前episode文件夹路径

        # 硬编码保存路径，每次创建类时自动启用保存
        hardcoded_save_path = "/shared_disk/users/weijie.ke/verl/recipe/vla/obs"
        self.enable_input_saving(hardcoded_save_path)

        # ===== TEMP: hardcode load lerobot dataset in rollout worker =====
        try:
            from giga_datasets.datasets.lerobot_dataset import LeRobotDataset
            dataset_path = "/shared_disk/users/yejun.zeng/datasets/huggingface/lerobot/catch_bowl"
            logger.info(f"[TEMP] Auto-loading LeRobot dataset in rollout worker: {dataset_path}")
            self.test_dataset = LeRobotDataset(data_path=dataset_path)
            self.test_dataset.open()
            logger.info(f"[TEMP] Dataset loaded in rollout worker. len={len(self.test_dataset)}")
        except Exception as e:
            logger.exception(f"[TEMP] Failed to auto-load dataset: {e}")
            raise

    def enable_input_saving(self, base_path: str):
        """启用输入数据保存功能。只保存最新的episode数据，新episode会自动覆盖旧数据。"""
        self.save_inputs_enabled = True
        self.save_inputs_base_path = base_path
        self.current_step_in_episode = 0
        self._episode_dir = None
        os.makedirs(base_path, exist_ok=True)
        logger.info(f"已启用输入数据保存（仅保存最新episode），保存路径: {base_path}")

    def disable_input_saving(self):
        """禁用输入数据保存功能。"""
        self.save_inputs_enabled = False
        logger.info("已禁用输入数据保存")

    def _save_inputs(self, cam_high, left_wrist, right_wrist, state, step_idx=None):
        """保存输入图像和状态到文件。只保存最新episode的数据，新episode会自动覆盖旧数据。"""
        if not self.save_inputs_enabled or self.save_inputs_base_path is None:
            return

        if step_idx is None:
            step_idx = self.current_step_in_episode

        is_new_episode = (self._episode_dir is None) or (step_idx < self.current_step_in_episode)

        if is_new_episode:
            episode_dir = os.path.join(self.save_inputs_base_path, "episode_latest")
            if os.path.exists(episode_dir):
                import shutil
                shutil.rmtree(episode_dir)
            self._episode_dir = episode_dir
            self.current_step_in_episode = 0
            step_idx = 0

        episode_dir = self._episode_dir
        os.makedirs(episode_dir, exist_ok=True)

        image_dir = os.path.join(episode_dir, "image")
        cam_high_dir = os.path.join(image_dir, "cam_high")
        cam_left_wrist_dir = os.path.join(image_dir, "cam_left_wrist")
        cam_right_wrist_dir = os.path.join(image_dir, "cam_right_wrist")
        for dir_path in [cam_high_dir, cam_left_wrist_dir, cam_right_wrist_dir]:
            os.makedirs(dir_path, exist_ok=True)

        batch_size = cam_high.shape[0]

        for b in range(batch_size):
            cam_high_np = cam_high[b].permute(1, 2, 0).cpu().numpy()
            if cam_high_np.max() <= 1.0:
                cam_high_np = (cam_high_np * 255).astype(np.uint8)
            else:
                cam_high_np = cam_high_np.astype(np.uint8)
            Image.fromarray(cam_high_np).save(os.path.join(cam_high_dir, f"step_{step_idx:04d}_batch_{b:02d}.png"))

            left_wrist_np = left_wrist[b].permute(1, 2, 0).cpu().numpy()
            if left_wrist_np.max() <= 1.0:
                left_wrist_np = (left_wrist_np * 255).astype(np.uint8)
            else:
                left_wrist_np = left_wrist_np.astype(np.uint8)
            Image.fromarray(left_wrist_np).save(
                os.path.join(cam_left_wrist_dir, f"step_{step_idx:04d}_batch_{b:02d}.png")
            )

            right_wrist_np = right_wrist[b].permute(1, 2, 0).cpu().numpy()
            if right_wrist_np.max() <= 1.0:
                right_wrist_np = (right_wrist_np * 255).astype(np.uint8)
            else:
                right_wrist_np = right_wrist_np.astype(np.uint8)
            Image.fromarray(right_wrist_np).save(
                os.path.join(cam_right_wrist_dir, f"step_{step_idx:04d}_batch_{b:02d}.png")
            )

        state_file = os.path.join(episode_dir, "state.txt")
        mode = "w" if is_new_episode else "a"
        with open(state_file, mode) as f:
            for b in range(batch_size):
                state_vec = state[b].cpu().numpy()
                state_str = ", ".join([f"{x:.6f}" for x in state_vec])
                f.write(f"step_{step_idx:04d}_batch_{b:02d}: {state_str}\n")

        self.current_step_in_episode += 1

    def _decode_jpeg_images(self, encoded_tensor: torch.Tensor) -> torch.Tensor:
        """解码 JPEG 编码的图像数据。"""
        import cv2
        import numpy as np

        batch_size = encoded_tensor.shape[0]
        decoded_images = []

        for i in range(batch_size):
            img_bytes = encoded_tensor[i].cpu().numpy().tobytes()
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                decoded_images.append(torch.from_numpy(img_rgb))
            else:
                decoded_images.append(torch.zeros((224, 224, 3), dtype=torch.uint8))

        return torch.stack(decoded_images)

    def load_test_dataset(self, dataset_path: str):
        """加载测试数据集。"""
        from giga_datasets.datasets.lerobot_dataset import LeRobotDataset

        logger.info(f"加载 LeRobot 数据集: {dataset_path}")
        self.test_dataset = LeRobotDataset(data_path=dataset_path)
        self.test_dataset.open()
        logger.info(f"数据集加载完成，共 {len(self.test_dataset)} 个样本")

    # ===================== CHANGED: robust meta.episodes parser =====================
    def _episode_lengths_from_meta(self, meta) -> list[int]:
        eps = meta.episodes
        if isinstance(eps, list):
            lengths = []
            for ep in eps:
                if isinstance(ep, dict) and "length" in ep:
                    lengths.append(int(ep["length"]))
                else:
                    lengths.append(int(ep))
            return lengths

        if isinstance(eps, dict):
            # common cases:
            # 1) {"length": [..]}
            if "length" in eps and isinstance(eps["length"], (list, tuple)):
                return [int(x) for x in eps["length"]]

            # 2) {"0": {"length": ..}, "1": {"length": ..}, ...} OR other sortable keys
            items = list(eps.items())
            try:
                items = sorted(items, key=lambda kv: int(kv[0]))
            except Exception:
                items = sorted(items, key=lambda kv: str(kv[0]))
            lengths = []
            for _, v in items:
                if isinstance(v, dict) and "length" in v:
                    lengths.append(int(v["length"]))
                else:
                    lengths.append(int(v))
            return lengths

        raise TypeError(f"Unsupported meta.episodes type: {type(eps)}")

    def _episode_start_index(self, meta, episode_idx: int) -> tuple[int, int]:
        lengths = self._episode_lengths_from_meta(meta)
        if episode_idx < 0 or episode_idx >= len(lengths):
            raise IndexError(f"episode_idx out of range: {episode_idx} not in [0, {len(lengths)-1}]")
        start = int(sum(lengths[:episode_idx]))
        return start, int(lengths[episode_idx])
    # ===================== END CHANGED =====================

    # ===================== CHANGED: chunked episode fetch =====================
    def get_episode_chunk(
        self,
        episode_idx: int,
        start_step: int,
        chunk_len: int,
        *,
        action_dim: int,
    ):
        """
        从数据集中获取一个 chunk 输入（1帧 obs） + 后续 chunk_len 个 GT absolute actions。

        数据集约定（根据DEBUG分析确认）：
          - sample["observation.state"] 是当前帧 state[t]（绝对关节 + 绝对gripper）
          - sample["action"] 是下一步的绝对目标位置（不是delta！）
              * 验证：action[t] ≈ state[t+1]
          - GT = action（直接使用，不需要转换）
        
        模型输出约定：
          - 模型输出 = 绝对关节 + 绝对gripper
        """
        if self.test_dataset is None:
            raise ValueError("请先调用 load_test_dataset() 加载数据集")

        meta = self.test_dataset.dataset.meta
        episode_start_idx, episode_length = self._episode_start_index(meta, episode_idx)

        # 必须保证还有 chunk_len 个 action
        if start_step < 0 or start_step >= episode_length:
            raise ValueError(f"start_step out of range: {start_step} not in [0, {episode_length-1}]")
        if start_step + chunk_len > episode_length:
            # 不足 chunk_len：按你的要求直接停止（调用方 break）
            return None

        # input sample at t=start_step
        sample0 = self.test_dataset[episode_start_idx + start_step]

        head_img0 = sample0["observation.images.cam_high"].permute(1, 2, 0).contiguous()
        left_wrist_img0 = sample0["observation.images.cam_left_wrist"].permute(1, 2, 0).contiguous()
        right_wrist_img0 = sample0["observation.images.cam_right_wrist"].permute(1, 2, 0).contiguous()

        state0 = sample0["observation.state"].clone()[:action_dim]  # [A] absolute current state
        
        # 左臂：0-5 关节(delta)，6 夹爪(绝对)
        # 右臂：7-12 关节(delta)，13 夹爪(绝对)
        joint_dims = list(range(6)) + list(range(7, 13)) if action_dim >= 14 else list(range(6))
        gripper_dims = [6, 13] if action_dim >= 14 else [6]
        
        gt_abs = []
        for k in range(chunk_len):
            s = self.test_dataset[episode_start_idx + start_step + k]
            state_t = s["observation.state"].clone()[:action_dim]  # [A] 当前时间步的state（绝对关节+绝对gripper）
            raw_action = s["action"].clone()[:action_dim]  # 数据集原始action（delta joint + 绝对gripper）
            
            # 🔧 DEBUG: 打印前3个时间步的原始数据（chunk 0和chunk 1）
            if start_step <= 30 and k < 3:
                print(f"\n[DEBUG RAW DATA] k={k}:")
                print(f"  state_t[:7]:        {state_t[:7]}")
                print(f"  raw_action[:7]:     {raw_action[:7]}")
                print(f"  state0 + raw:       {state0[:7] + raw_action[:7]}")
                print(f"  state_t + raw:      {state_t[:7] + raw_action[:7]}")
            
            # ===================== GT转换逻辑 =====================
            # 根据DEBUG输出分析：
            #   - state[t]: 绝对关节 + 绝对gripper
            #   - action[t]: 就是下一步的绝对目标位置（不是delta！）
            #   - 验证：raw_action[k=0] ≈ state_t[k=1]
            # 
            # 模型输出格式：
            #   - 绝对关节 + 绝对gripper
            # 
            # 因此GT = raw_action（直接使用，不需要转换）
            abs_action = raw_action.clone()
            # 不需要任何转换，action本身就是绝对目标位置
            
            gt_abs.append(abs_action)
        gt_abs = torch.stack(gt_abs, dim=0)  # [T,A]

        task = sample0["task"]
        return {
            "head_image": head_img0,                   # [H,W,C]
            "left_wrist_image": left_wrist_img0,       # [H,W,C]
            "right_wrist_image": right_wrist_img0,     # [H,W,C]
            "state": state0,                           # [A]
            "action_abs_seq": gt_abs,                  # [T,A] - 已转换为绝对值
            "task": task,
            "episode_idx": episode_idx,
            "start_step": start_step,
            "episode_length": episode_length,
        }
    # ===================== END CHANGED =====================
    # @torch.no_grad()
    # def generate_sequences(self, prompts: DataProto) -> DataProto:
    #     """
    #     TEMP HACK (real-robot debug):
    #     Ignore real robot inputs; use LeRobot dataset obs instead.
    #     Dirty but stable.
    #     """
    #     # -----------------------------
    #     # hardcode episode control
    #     EPISODE_IDX = 0
    #     START_STEP = 0
    #     STRIDE = 10  # 每次调用推进多少步（你想 1 就改 1）

    #     # 记录全局 step（函数静态变量，不污染 self）
    #     if not hasattr(PI0RolloutRob.generate_sequences, "_ds_step"):
    #         PI0RolloutRob.generate_sequences._ds_step = START_STEP

    #     if self.test_dataset is None:
    #         raise RuntimeError("[TEMP] test_dataset is None. Your __init__ should auto-load it, or call load_test_dataset().")

    #     # 当前 episode 长度，用于 wrap
    #     meta = self.test_dataset.dataset.meta
    #     _, episode_length = self._episode_start_index(meta, EPISODE_IDX)

    #     # 用 prompts 的 batch size（真机可能 != 1）
    #     # 尽量从 head_image 推断；没有就 fallback 1
    #     try:
    #         B = int(prompts.batch["head_image"].shape[0])
    #     except Exception:
    #         B = 1

    #     ds_step0 = int(PI0RolloutRob.generate_sequences._ds_step)
    #     print(f"[TEMP] ds_step={ds_step0}, B={B}, STRIDE={STRIDE}, episode_len={episode_length}")

    #     # -----------------------------
    #     # 从 dataset 构造 B 个样本（每个样本取一帧 obs，task 可以相同）
    #     heads, lefts, rights, states = [], [], [], []
    #     tasks = []

    #     for b in range(B):
    #         st = ds_step0 + b  # 同一次调用内用相邻帧，避免全都一样
    #         # wrap
    #         st = st % max(1, episode_length)

    #         chunk = self.get_episode_chunk(
    #             episode_idx=EPISODE_IDX,
    #             start_step=st,
    #             chunk_len=1,          # 只要 1 帧 obs（GT action 我们只做打印用）
    #             action_dim=14,        # 你的 action_dim
    #         )
    #         if chunk is None:
    #             # 理论上不会发生（chunk_len=1, wrap 后肯定够）
    #             raise RuntimeError(f"[TEMP] get_episode_chunk returned None at step={st}")

    #         heads.append(chunk["head_image"])             # [H,W,C]
    #         lefts.append(chunk["left_wrist_image"])
    #         rights.append(chunk["right_wrist_image"])
    #         states.append(chunk["state"])                 # [A]
    #         tasks.append(chunk["task"])

    #     # 推进全局步数（一次调用推进 STRIDE）
    #     PI0RolloutRob.generate_sequences._ds_step = (ds_step0 + STRIDE) % max(1, episode_length)

    #     head_image = torch.stack(heads, dim=0)            # [B,H,W,C]
    #     left_wrist_image = torch.stack(lefts, dim=0)
    #     right_wrist_image = torch.stack(rights, dim=0)
    #     state = torch.stack(states, dim=0)                # [B,A]
    #     task_descriptions = np.array(tasks)               # len=B

    #     # -----------------------------
    #     # Below: keep your original logic as-is
    #     timing_generate = {}
    #     with simple_timer("rollout generate_sequences", timing_generate):

    #         # 1) 统一 state shape
    #         if state.ndim == 3:
    #             state = state[:, -1, :]
    #         elif state.ndim == 2:
    #             pass
    #         elif state.ndim == 1:
    #             state = state.unsqueeze(0)
    #         else:
    #             raise ValueError(f"[PI0RolloutRob] Unexpected state shape: {state.shape}")

    #         device = next(self.module.parameters()).device
    #         state = state.to(device=device, dtype=torch.float32)
    #         raw_state_dim = int(state.shape[-1])

    #         # pad 到 32
    #         state_pad32 = torch.nn.functional.pad(state, (0, max(0, 32 - raw_state_dim)), "constant", 0.0)

    #         sample_sig = inspect.signature(self.module.sample_actions)
    #         supports_state_dim = "state_dim" in sample_sig.parameters

    #         with torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
    #             # dataset 是 [B,H,W,C]，不会走 jpeg decode 分支；保留也无妨
    #             if head_image.ndim == 2:
    #                 head_image = self._decode_jpeg_images(head_image)
    #             if left_wrist_image.ndim == 2:
    #                 left_wrist_image = self._decode_jpeg_images(left_wrist_image)
    #             if right_wrist_image.ndim == 2:
    #                 right_wrist_image = self._decode_jpeg_images(right_wrist_image)

    #             batch_size = head_image.shape[0]
    #             cam_high = head_image.permute(0, 3, 1, 2).to(device)
    #             left_wrist = left_wrist_image.permute(0, 3, 1, 2).to(device)
    #             right_wrist = right_wrist_image.permute(0, 3, 1, 2).to(device)

    #             kwargs = dict(
    #                 images={
    #                     "observation.images.cam_high": cam_high,
    #                     "observation.images.cam_left_wrist": left_wrist,
    #                     "observation.images.cam_right_wrist": right_wrist,
    #                 },
    #                 img_masks=[
    #                     torch.ones((batch_size,), device=device, dtype=torch.bool),
    #                     torch.ones((batch_size,), device=device, dtype=torch.bool),
    #                     torch.ones((batch_size,), device=device, dtype=torch.bool),
    #                 ],
    #                 task=task_descriptions.tolist(),
    #                 state=state_pad32,
    #                 tokenizer=self.tokenizer,
    #             )
    #             if supports_state_dim:
    #                 kwargs["state_dim"] = raw_state_dim

    #             if getattr(self, "save_inputs_enabled", False):
    #                 self._save_inputs(cam_high, left_wrist, right_wrist, state_pad32)

    #             (
    #                 action,
    #                 images_out,
    #                 img_masks,
    #                 lang_tokens,
    #                 lang_masks,
    #                 state_out,
    #             ) = self.module.sample_actions(**kwargs)

    #     print("rollout generate_sequences time (s): %s" % timing_generate.get("rollout generate_sequences", 0.0))

    #     # chunk_len/action_dim
    #     cfg = getattr(self.module, "config", None)
    #     T = getattr(cfg, "num_action_chunks", 10)
    #     A = getattr(cfg, "action_dim", action.shape[-1])
    #     T = min(int(T), int(action.shape[1]))
    #     A = min(int(A), int(action.shape[2]))

    #     # -----------------------------
    #     # TEMP PRINT: pred[0,0] vs GT(step0)（用第一条样本做 quick sanity）
    #     try:
    #         # 重新取一下第一条的 GT：action_abs_seq[0] 是 gt_abs at step=st
    #         st0 = ds_step0 % max(1, episode_length)
    #         chunk0 = self.get_episode_chunk(EPISODE_IDX, st0, 1, action_dim=A)
    #         gt_abs0 = chunk0["action_abs_seq"][0].to(dtype=torch.float32)[:A]          # [A]
    #         pred0 = action[0, 0, :A].to(dtype=torch.float32).detach().cpu()            # [A]（注意：这里 pred 还是模型原始语义）
    #         diff = (pred0 - gt_abs0.cpu())
    #         print("\n[TEMP] pred0 vs gt_abs0 (first sample, first step)")
    #         print("  pred0 :", "[" + ", ".join([f"{x:.6f}" for x in pred0.tolist()]) + "]")
    #         print("  gt_abs:", "[" + ", ".join([f"{x:.6f}" for x in gt_abs0.cpu().tolist()]) + "]")
    #         print("  |diff| mean=%.6f max=%.6f\n" % (diff.abs().mean().item(), diff.abs().max().item()))
    #     except Exception as e:
    #         print(f"[TEMP] skip pred-vs-gt print due to: {e}")

    #     ret = DataProto.from_dict(
    #         {
    #             "action": action[:, :T, :A],
    #             "full_action": action,
    #             "images": torch.stack(images_out, dim=1) if isinstance(images_out, (list, tuple)) else images_out,
    #             "image_masks": torch.stack(img_masks, dim=1) if isinstance(img_masks, (list, tuple)) else img_masks,
    #             "lang_tokens": lang_tokens,
    #             "lang_masks": lang_masks,
    #             "states": state_out,
    #         }
    #     )
    #     return ret

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Generate sequences"""
        head_image = prompts.batch["head_image"]
        left_wrist_image = prompts.batch["left_wrist_image"]
        right_wrist_image = prompts.batch["right_wrist_image"]
        state = prompts.batch["state"]
        task_descriptions = prompts.non_tensor_batch["task_descriptions"]

        timing_generate = {}
        with simple_timer("rollout generate_sequences", timing_generate):

            # 1) 统一 state shape
            if state.ndim == 3:
                state = state[:, -1, :]
            elif state.ndim == 2:
                pass
            elif state.ndim == 1:
                state = state.unsqueeze(0)
            else:
                raise ValueError(f"[PI0RolloutRob] Unexpected state shape: {state.shape}")

            device = prompts.batch.device
            state = state.to(device=device, dtype=torch.float32)
            raw_state_dim = int(state.shape[-1])  # usually action_dim (e.g., 14)

            # prompt 用 raw_state_dim（不含 pad），模型 forward 用 pad32
            state_pad32 = torch.nn.functional.pad(state, (0, max(0, 32 - raw_state_dim)), "constant", 0.0)
            state_pad32 = state_pad32.to(device=device, dtype=torch.float32)
            sample_sig = inspect.signature(self.module.sample_actions)
            supports_state_dim = "state_dim" in sample_sig.parameters

            with torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
                if head_image.ndim == 2:
                    head_image = self._decode_jpeg_images(head_image)
                if left_wrist_image.ndim == 2:
                    left_wrist_image = self._decode_jpeg_images(left_wrist_image)
                if right_wrist_image.ndim == 2:
                    right_wrist_image = self._decode_jpeg_images(right_wrist_image)

                batch_size = head_image.shape[0]
                cam_high = head_image.permute(0, 3, 1, 2).to(device)
                left_wrist = left_wrist_image.permute(0, 3, 1, 2).to(device)
                right_wrist = right_wrist_image.permute(0, 3, 1, 2).to(device)

                kwargs = dict(
                    images={
                        "observation.images.cam_high": cam_high,
                        "observation.images.cam_left_wrist": left_wrist,
                        "observation.images.cam_right_wrist": right_wrist,
                    },
                    img_masks=[
                        torch.ones((batch_size,), device=device, dtype=torch.bool),
                        torch.ones((batch_size,), device=device, dtype=torch.bool),
                        torch.ones((batch_size,), device=device, dtype=torch.bool),
                    ],
                    task=task_descriptions.tolist() if hasattr(task_descriptions, "tolist") else list(task_descriptions),
                    state=state_pad32,
                    tokenizer=self.tokenizer,
                )

                if supports_state_dim:
                    kwargs["state_dim"] = raw_state_dim
                
                # ===================== CHANGED =====================
                # 从模型config中读取use_endpose和no_state配置并传递
                cfg = getattr(self.module, "config", None)
                if cfg is not None:
                    use_endpose = getattr(cfg, "use_endpose", False)
                    no_state = getattr(cfg, "no_state", False)
                    kwargs["use_endpose"] = use_endpose
                    kwargs["no_state"] = no_state
                # ===================== END CHANGED =====================

                if self.save_inputs_enabled:
                    self._save_inputs(cam_high, left_wrist, right_wrist, state_pad32)

                (
                    action,
                    images_out,
                    img_masks,
                    lang_tokens,
                    lang_masks,
                    state_out,
                ) = self.module.sample_actions(**kwargs)
        print("rollout generate_sequences time (s): %s" % timing_generate.get("rollout generate_sequences", 0.0))

        # ===================== CHANGED =====================
        # chunk_len 默认 30（按你的要求），取不到就 fallback=30
        cfg = getattr(self.module, "config", None)
        T = getattr(cfg, "num_action_chunks", 30)
        A = getattr(cfg, "action_dim", action.shape[-1])
        T = min(int(T), int(action.shape[1]))
        A = min(int(A), int(action.shape[2]))
        # ===================== END CHANGED =====================

        ret = DataProto.from_dict(
            {
                "action": action[:, :T, :A],
                "full_action": action,
                "images": torch.stack(images_out, dim=1) if isinstance(images_out, (list, tuple)) else images_out,
                "image_masks": torch.stack(img_masks, dim=1) if isinstance(img_masks, (list, tuple)) else img_masks,
                "lang_tokens": lang_tokens,
                "lang_masks": lang_masks,
                "states": state_out,
            }
        )
        return ret

    # ===================== CHANGED: episode evaluation (chunked, 30-step) =====================
    @torch.no_grad()
    def test_episode_chunked(
        self,
        episode_idx: int = 0,
        start_step: int = 0,
        *,
        max_chunks: int | None = None,
        verbose: bool = True,
        test_fk_conversion: bool = False,
    ):
        """
        对一段 episode 做 chunk 评估：
          - 每次用 1 张图片/1 个 state 输入模型
          - 模型输出一个 action chunk（默认 30 steps，已对齐为 ABS）
          - 用数据集对应的后续 30 个 absolute action 作为 GT
          - 对每个 chunk，分别统计：
              - 前10步 / 中10步 / 后10步 的误差
              - joints vs grippers 的误差（分开）
          - episode 末尾不足 30 步：直接停止
          - 最后输出 episode 总体 joints/grippers 误差
        
        Args:
            test_fk_conversion: 如果为True，将joint action通过FK转换为endpose进行对比测试
        """
        if self.test_dataset is None:
            raise ValueError("请先调用 load_test_dataset() 加载数据集")
        
        # ===================== FK转换初始化 =====================
        print(f"\n[FK Test] test_fk_conversion参数: {test_fk_conversion}")
        left_kin = None
        right_kin = None
        _rotmat_to_rpy_zyx = None
        
        if test_fk_conversion:
            print("[FK Test] 开始初始化运动学求解器...")
            try:
                import sys
                piper_path = "/shared_disk/users/weijie.ke/verl/recipe/vla/envs/robot_env/robot/controller/piper"
                if piper_path not in sys.path:
                    sys.path.insert(0, piper_path)
                
                # 导入本地的lerobot模块（不是pip安装的）
                from lerobot.model.kinematics import RobotKinematics
                
                # 直接定义旋转矩阵转RPY函数，避免导入问题
                def _rotmat_to_rpy_zyx(R: np.ndarray) -> np.ndarray:
                    """将旋转矩阵转换为RPY（Roll-Pitch-Yaw）欧拉角（ZYX顺序）"""
                    r20 = -R[2, 0]
                    r20_clamped = float(np.clip(r20, -1.0, 1.0))
                    pitch = np.arcsin(r20_clamped)
                    
                    cos_pitch = np.cos(pitch)
                    if abs(cos_pitch) < 1e-6:
                        # 退化情况（接近 ±90°）
                        roll = 0.0
                        yaw = np.arctan2(-R[0, 1], R[1, 1])
                    else:
                        roll = np.arctan2(R[2, 1], R[2, 2])
                        yaw = np.arctan2(R[1, 0], R[0, 0])
                    
                    return np.array([float(roll), float(pitch), float(yaw)], dtype=float)
                
                urdf_path = "/shared_disk/users/weijie.ke/verl/recipe/vla/envs/robot_env/robot/controller/piper/local_assets/robot.urdf"
                print(f"[FK Test] 加载URDF: {urdf_path}")
                print("[FK Test] 注意：URDF中关节名为 joint1-joint8（无left/right前缀）")
                print("[FK Test] 假设 joint1-joint6 为左臂，需要为右臂创建单独的URDF或镜像处理")
                
                # 左臂：使用 joint1-joint6
                left_kin = RobotKinematics(
                    urdf_path=urdf_path,
                    target_frame_name="link6",  # 末端执行器link
                    joint_names=[
                        "joint1",
                        "joint2",
                        "joint3",
                        "joint4",
                        "joint5",
                        "joint6",
                    ],
                )
                print("[FK Test] 左臂运动学求解器初始化成功 (joint1-joint6)")
                
                # 右臂：暂时使用相同的URDF（应该是镜像配置）
                # TODO: 如果有单独的右臂URDF，应该使用不同的文件
                right_kin = RobotKinematics(
                    urdf_path=urdf_path,
                    target_frame_name="link6",
                    joint_names=[
                        "joint1",  # 右臂也是6个关节，但在双臂系统中需要不同处理
                        "joint2",
                        "joint3",
                        "joint4",
                        "joint5",
                        "joint6",
                    ],
                )
                print("[FK Test] 右臂运动学求解器初始化成功")
                print("[FK Test] ✓ 运动学求解器初始化完成，FK转换已启用\n")
            except Exception as e:
                import traceback
                print(f"[FK Test] ✗ 运动学求解器初始化失败: {e}")
                print(f"[FK Test] Traceback:\n{traceback.format_exc()}")
                test_fk_conversion = False
        else:
            print("[FK Test] FK转换未启用（test_fk_conversion=False）\n")
        # ===================== END FK转换初始化 =====================

        cfg = getattr(self.module, "config", None)
        chunk_len = int(getattr(cfg, "num_action_chunks", 30))
        action_dim = int(getattr(cfg, "action_dim", 14))

        if chunk_len != 30:
            logger.warning(f"[chunk-eval] config.num_action_chunks={chunk_len} (expected 30). Still proceed.")

        # ===================== 只评估左臂（前7维）=====================
        # 左臂：0-5 关节，6 夹爪
        # 右臂：7-12 关节，13 夹爪（忽略）
        eval_dim = 7  # 只评估左臂
        gripper_idx = [6]  # 只评估左臂夹爪
        joint_idx = [i for i in range(eval_dim) if i not in gripper_idx]  # 0-5

        # segments: first/middle/last 10 (only meaningful for 30)
        def _segments(T: int):
            if T >= 30:
                return [
                    ("first10", slice(0, 10)),
                    ("mid10", slice(10, 20)),
                    ("last10", slice(20, 30)),
                ]
            # fallback: split into 3 parts
            k = T // 3
            return [
                ("first", slice(0, k)),
                ("mid", slice(k, 2 * k)),
                ("last", slice(2 * k, T)),
            ]

        # accumulate episode totals
        ep_joint_abs_sum = 0.0
        ep_joint_count = 0
        ep_grip_abs_sum = 0.0
        ep_grip_count = 0

        chunk_id = 0
        cur = int(start_step)

        # episode length for stopping (robust)
        meta = self.test_dataset.dataset.meta
        _, episode_length = self._episode_start_index(meta, episode_idx)

        print("\n" + "=" * 100)
        print(f"[EPISODE] idx={episode_idx}  start_step={start_step}  episode_length={episode_length}")
        print(f"[CONFIG] chunk_len={chunk_len}  eval_dim={eval_dim} (左臂)  gripper_idx={gripper_idx}  joint_dims={len(joint_idx)}")
        print("=" * 100)

        while True:
            if max_chunks is not None and chunk_id >= int(max_chunks):
                break

            chunk = self.get_episode_chunk(
                episode_idx=episode_idx,
                start_step=cur,
                chunk_len=chunk_len,
                action_dim=action_dim,
            )
            if chunk is None:
                print(f"\n[STOP] remaining steps < chunk_len ({chunk_len}). stop at step={cur}.")
                break

            # build prompts: ONLY ONE IMAGE for the chunk
            device = next(self.module.parameters()).device
            head_img = chunk["head_image"].unsqueeze(0).to(device)          # [1,H,W,C]
            left_img = chunk["left_wrist_image"].unsqueeze(0).to(device)
            right_img = chunk["right_wrist_image"].unsqueeze(0).to(device)
            state0 = chunk["state"].unsqueeze(0).to(device)                 # [1,A]
            task = chunk["task"]

            prompts = DataProto.from_dict(
                tensors={
                    "head_image": head_img,
                    "left_wrist_image": left_img,
                    "right_wrist_image": right_img,
                    "state": state0,
                },
                non_tensors={"task_descriptions": np.array([task])},
            )

            out = self.generate_sequences(prompts)
            pred_full = out.batch["action"].detach().float().cpu()          # [1,T,A] 模型输出
            gt_full_joint = chunk["action_abs_seq"].detach().float().cpu()  # [T,A] 数据集GT（绝对joint）

            T = min(int(pred_full.shape[1]), int(gt_full_joint.shape[0]), 30 if chunk_len >= 30 else chunk_len)
            
            # ===================== FK转换：将GT的joint转为endpose =====================
            if test_fk_conversion and left_kin is not None and right_kin is not None and _rotmat_to_rpy_zyx is not None:
                # 关键：将GT的joint action通过FK转换为endpose，然后与模型预测的endpose对比
                # 注意：endpose学习的是相对变化（delta），gripper学习的是绝对值
                print("\n[FK Test] 将GT的joint通过FK转换为endpose（相对变化）...")
                
                # 0. 首先计算state0的FK（作为基准）
                state0_joints = state0.cpu().squeeze()[:14].numpy()  # [14] 当前state的joint角度
                
                # 左臂state FK
                ql_state_rad = state0_joints[:6].astype(float)
                ql_state_deg = np.rad2deg(ql_state_rad)
                T_l_state = left_kin.forward_kinematics(ql_state_deg)
                p_l_state = T_l_state[:3, 3]
                rpy_l_state = _rotmat_to_rpy_zyx(T_l_state[:3, :3])
                
                # 右臂state FK
                qr_state_rad = state0_joints[7:13].astype(float)
                qr_state_deg = np.rad2deg(qr_state_rad)
                T_r_state = right_kin.forward_kinematics(qr_state_deg)
                p_r_state = T_r_state[:3, 3]
                rpy_r_state = _rotmat_to_rpy_zyx(T_r_state[:3, :3])
                
                print(f"[FK Test] State0 左臂endpose: pos={p_l_state}, rpy={rpy_l_state}")
                print(f"[FK Test] State0 右臂endpose: pos={p_r_state}, rpy={rpy_r_state}")
                
                # 1. 转换GT的joint action -> endpose delta（相对于state0）
                gt_endpose_list = []
                for t in range(T):
                    joint_action = gt_full_joint[t, :14].numpy()  # [14] 绝对joint角度(rad)
                    
                    # 左臂FK
                    ql_rad = joint_action[:6].astype(float)
                    ql_deg = np.rad2deg(ql_rad)
                    T_l = left_kin.forward_kinematics(ql_deg)
                    p_l = T_l[:3, 3]
                    rpy_l = _rotmat_to_rpy_zyx(T_l[:3, :3])
                    
                    # 计算相对变化（delta）
                    delta_p_l = p_l - p_l_state
                    delta_rpy_l = rpy_l - rpy_l_state
                    
                    # gripper使用绝对值（不是delta）
                    l_grip = float(joint_action[6])
                    
                    # 右臂FK
                    qr_rad = joint_action[7:13].astype(float)
                    qr_deg = np.rad2deg(qr_rad)
                    T_r = right_kin.forward_kinematics(qr_deg)
                    p_r = T_r[:3, 3]
                    rpy_r = _rotmat_to_rpy_zyx(T_r[:3, :3])
                    
                    # 计算相对变化（delta）
                    delta_p_r = p_r - p_r_state
                    delta_rpy_r = rpy_r - rpy_r_state
                    
                    # gripper使用绝对值（不是delta）
                    r_grip = float(joint_action[13])
                    
                    # 组合：endpose用delta，gripper用绝对值
                    endpose = np.array([
                        delta_p_l[0], delta_p_l[1], delta_p_l[2], 
                        delta_rpy_l[0], delta_rpy_l[1], delta_rpy_l[2], 
                        l_grip,  # 绝对值
                        delta_p_r[0], delta_p_r[1], delta_p_r[2], 
                        delta_rpy_r[0], delta_rpy_r[1], delta_rpy_r[2], 
                        r_grip,  # 绝对值
                    ], dtype=np.float32)
                    gt_endpose_list.append(endpose)
                
                gt_endpose = torch.from_numpy(np.stack(gt_endpose_list, axis=0))  # [T, 14]
                
                # 2. 模型预测已经是endpose（因为use_endpose=True）
                pred_endpose = pred_full[0, :T, :14]  # [T, 14]
                
                # 3. 对比endpose误差（只看左臂前7维）
                pred_endpose_left = pred_endpose[:, :7].numpy()
                gt_endpose_left = gt_endpose[:, :7].numpy()
                
                if chunk_id == 0:
                    print("\n[FK Test] Chunk 0 详细对比 (前3步):")
                    print("注意：GT是FK(action)-FK(state)的delta endpose + 绝对gripper")
                    print("      模型输出是delta endpose + 绝对gripper")
                    for t in range(min(3, T)):
                        print(f"\n  时间步 t={t}:")
                        print(f"    GT_joint_abs[{t}]:        {gt_full_joint[t, :7]}")  # 原始joint（绝对值）
                        print(f"    GT_endpose_delta[{t}]:    {gt_endpose_left[t]}")    # FK转换后的delta endpose
                        print(f"    Pred_endpose_delta[{t}]:  {pred_endpose_left[t]}")  # 模型输出的delta endpose
                        print(f"    diff:                     {np.abs(pred_endpose_left[t] - gt_endpose_left[t])}")
                        print(f"    位置delta误差(m):   {np.linalg.norm(pred_endpose_left[t, :3] - gt_endpose_left[t, :3]):.6f}")
                        print(f"    姿态delta误差(rad): {np.linalg.norm(pred_endpose_left[t, 3:6] - gt_endpose_left[t, 3:6]):.6f}")
                        print(f"    夹爪绝对值误差:     {np.abs(pred_endpose_left[t, 6] - gt_endpose_left[t, 6]):.6f}")
                
                # 计算整体endpose误差
                pos_error = np.linalg.norm(pred_endpose_left[:, :3] - gt_endpose_left[:, :3], axis=1)
                ori_error = np.linalg.norm(pred_endpose_left[:, 3:6] - gt_endpose_left[:, 3:6], axis=1)
                grip_error = np.abs(pred_endpose_left[:, 6] - gt_endpose_left[:, 6])
                
                print(f"\n[FK Test] Chunk {chunk_id} Endpose Delta + Gripper Absolute 整体误差:")
                print(f"  位置delta MAE: {pos_error.mean():.6f} m (max: {pos_error.max():.6f} m)")
                print(f"  姿态delta MAE: {ori_error.mean():.6f} rad (max: {ori_error.max():.6f} rad)")
                print(f"  夹爪绝对值 MAE: {grip_error.mean():.6f} (max: {grip_error.max():.6f})")
                
                # 使用转换后的endpose作为GT进行后续对比
                gt_full = gt_endpose
            else:
                # 不做FK转换，直接使用joint作为GT
                gt_full = gt_full_joint
            # ===================== END FK转换测试 =====================
            
            # 🔧 DEBUG: 打印第一个chunk的前3个时间步的详细信息（Joint空间）
            if chunk_id == 0:
                print("\n[DEBUG] Chunk 0 Joint空间详细对比:")
                print(f"state0: {state0.cpu().squeeze()[:7]}")
                for t in range(min(3, T)):
                    print(f"\n时间步 t={t}:")
                    print(f"  pred_action[{t}]: {pred_full[0, t, :7]}")
                    print(f"  gt_action[{t}]:   {gt_full[t, :7]}")
                    print(f"  diff[{t}]:        {(pred_full[0, t, :7] - gt_full[t, :7]).abs()}")
            
            # ===================== 只对比左臂（前7维）=====================
            pred = pred_full[0, :T, :eval_dim]                              # [T,7] 模型预测（绝对值）
            gt = gt_full[:T, :eval_dim]                                     # [T,7] 数据集GT（绝对值）
            

            # sanity
            if T == 0:
                print(f"\n[WARN] empty T at step={cur}. stop.")
                break

            segs = _segments(T)

            # compute per-chunk segment metrics
            print("\n" + "-" * 100)
            print(f"[CHUNK {chunk_id:03d}] step={cur:04d}  task={task}")
            print(f"         eval_T={T}  eval_dim={eval_dim} (左臂)")

            for name, sl in segs:
                p = pred[sl, :]
                g = gt[sl, :]
                if p.numel() == 0:
                    continue

                if len(joint_idx) > 0:
                    dj = (p[:, joint_idx] - g[:, joint_idx]).abs()
                    joint_mae = dj.mean().item()
                    joint_mse = (dj ** 2).mean().item()
                    # accumulate totals
                    ep_joint_abs_sum += dj.sum().item()
                    ep_joint_count += int(dj.numel())
                else:
                    joint_mae = float("nan")
                    joint_mse = float("nan")

                if len(gripper_idx) > 0:
                    dg = (p[:, gripper_idx] - g[:, gripper_idx]).abs()
                    grip_mae = dg.mean().item()
                    grip_mse = (dg ** 2).mean().item()
                    # accumulate totals
                    ep_grip_abs_sum += dg.sum().item()
                    ep_grip_count += int(dg.numel())
                else:
                    grip_mae = float("nan")
                    grip_mse = float("nan")

                print(
                    f"  [{name:>6}] "
                    f"joints: MAE={joint_mae:.6f} MSE={joint_mse:.6f}   "
                    f"gripper: MAE={grip_mae:.6f} MSE={grip_mse:.6f}"
                )


            # advance to next chunk (IMPORTANT: chunked dataset usage)
            chunk_id += 1
            cur += chunk_len

            if cur >= episode_length:
                print(f"\n[STOP] reached episode end: cur={cur} >= episode_length={episode_length}")
                break

        # episode summary
        ep_joint_mae = ep_joint_abs_sum / max(1, ep_joint_count)
        ep_grip_mae = ep_grip_abs_sum / max(1, ep_grip_count)

        print("\n" + "=" * 100)
        print(f"[SUMMARY] episode_idx={episode_idx}  tested_chunks={chunk_id}  tested_steps={chunk_id * chunk_len}")
        print(f"  joints  : overall MAE={ep_joint_mae:.6f}   (count={ep_joint_count})")
        print(f"  gripper : overall MAE={ep_grip_mae:.6f}   (count={ep_grip_count})")
        print("=" * 100 + "\n")
    # ===================== END CHANGED =====================


def test_pi0_with_lerobot_dataset(
    model_path: str,
    dataset_path: str,
    episode_idx: int = 0,
    start_step: int = 0,
    max_chunks: int | None = None,
    device: str = "cuda:7",
    test_fk_conversion: bool = False,
):
    """测试 PI0 模型在 lerobot 数据集上的表现（chunked evaluation）。
    
    Args:
        test_fk_conversion: 如果为True，将joint action通过FK转换为endpose进行对比测试
    """
    logger.info("初始化模型...")

    from transformers import AutoTokenizer
    from recipe.vla.models.pi0_torch.modeling_pi0_torch import PI0ForActionPrediction
    from recipe.vla.models.pi0_torch.configuration_pi0_torch import PI0TorchConfig

    logger.info(f"从 {model_path} 加载 PI0 模型...")

    config = PI0TorchConfig.from_pretrained(model_path)
    if hasattr(config, "attn_implementation"):
        config.attn_implementation = "eager"

    model = PI0ForActionPrediction.from_pretrained(model_path, config=config)
    model = model.to(device)
    model.eval()

    logger.info(f"模型配置: pi05_enabled={config.pi05_enabled}, use_endpose={getattr(config, 'use_endpose', False)}, no_state={getattr(config, 'no_state', False)}, dtype={model.dtype}")

    logger.info("加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)

    model_config = {"path": model_path}
    rollout = PI0RolloutRob(
        model_config=model_config,
        module=model,
        tokenizer=tokenizer,
    )
    rollout.enable_input_saving(base_path="/shared_disk/users/weijie.ke/verl/recipe/vla/obs")
    logger.info(f"加载数据集: {dataset_path}")
    rollout.load_test_dataset(dataset_path)

    rollout.test_episode_chunked(
        episode_idx=episode_idx,
        start_step=start_step,
        max_chunks=max_chunks,
        verbose=False,
        test_fk_conversion=test_fk_conversion,
    )

    logger.info("测试完成!")


if __name__ == "__main__":
    import argparse

    DEFAULT_MODEL_PATH = "/shared_disk/users/weijie.ke/weight/giga-openpi/pick_catch_bowl_eepose"
    DEFAULT_DATASET_PATH = "/shared_disk/users/yejun.zeng/datasets/huggingface/lerobot/catch_bowl"
    DEFAULT_EPISODE_IDX = 0
    DEFAULT_START_STEP = 0
    DEFAULT_DEVICE = "cuda:4"

    parser = argparse.ArgumentParser(
        description="测试 PI0 模型在 lerobot 数据集上的表现（chunked, 30-step）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, help="模型路径")
    parser.add_argument("--dataset_path", type=str, default=DEFAULT_DATASET_PATH, help="lerobot 数据集路径")
    parser.add_argument("--episode_idx", type=int, default=DEFAULT_EPISODE_IDX, help="episode 索引")
    parser.add_argument("--start_step", type=int, default=DEFAULT_START_STEP, help="起始步数（chunk起点）")
    parser.add_argument("--max_chunks", type=int, default=None, help="最多测试多少个chunk（None=跑到不够30步为止）")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE, help="设备")
    parser.add_argument("--test_fk", action="store_true", default=True, help="是否测试FK转换（将joint转为endpose对比），默认启用")

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("PI0 chunked 测试配置:")
    logger.info(f"  模型路径: {args.model_path}")
    logger.info(f"  数据集路径: {args.dataset_path}")
    logger.info(f"  Episode 索引: {args.episode_idx}")
    logger.info(f"  起始步数: {args.start_step}")
    logger.info(f"  max_chunks: {args.max_chunks}")
    logger.info(f"  设备: {args.device}")
    logger.info(f"  FK转换测试: {args.test_fk}")
    logger.info("=" * 80)

    test_pi0_with_lerobot_dataset(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        episode_idx=args.episode_idx,
        start_step=args.start_step,
        max_chunks=args.max_chunks,
        device=args.device,
        test_fk_conversion=args.test_fk,
    )
