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

import torch
from PIL import Image
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.utils.rnn import pad_sequence

from recipe.vla.envs.action_utils import center_crop_image, resize_image
from recipe.vla.models.openvla_oft.modeling_prismatic import OpenVLAForActionPrediction
from recipe.vla.models.openvla_oft.processing_prismatic import PrismaticProcessor
from recipe.vla.models.pi0_torch import Pi0Pipeline
from verl import DataProto
from verl.utils.device import get_device_id, get_device_name, get_torch_device
from verl.utils.profiler import simple_timer
from verl.workers.rollout.base import BaseRollout

logger = logging.getLogger(__name__)


__all__ = ["NaiveRolloutRob", "PI0RolloutRob"]


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

    # @conditional_profiler(name="generate_sequences", path="traces/rollout", max_steps=5)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Generate sequences"""
        # make sampling args can be overriden by inputs
        do_sample = prompts.meta_info["do_sample"]
        temperature = prompts.meta_info["temperature"]
        max_prompt_length = prompts.meta_info["prompt_length"]
        # TODO: split into micro-batches
        task_descriptions = prompts.non_tensor_batch["task_descriptions"]
        images_and_states = {"full_image": prompts.batch["full_image"]}
        vla_input = process_input(task_descriptions, images_and_states, self.processor)

        vla_output = self._generate_one_step(vla_input, do_sample, temperature, max_prompt_length)
        # batch = TensorDict(vla_output)
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
    ):
        self.model_config = model_config
        self.module = module
        self.module.eval()

        self.pipeline = Pi0Pipeline(
            self.module.policy,
            tokenizer_model_path='google/paligemma-3b-pt-224',
            # TODO(liujincheng): pass in using the configuration file
            state_norm_stats={
                'mean': [
                    -0.04363870248198509,
                    0.03525487706065178,
                    0.7637033462524414,
                    2.9673683643341064,
                    -0.2108035385608673,
                    -0.1297520250082016,
                    0.027788693085312843,
                    -0.028010232374072075,
                ] + [0.0] * (32 - 8),
                'std': [
                    0.10337679088115692,
                    0.15188011527061462,
                    0.38154250383377075,
                    0.3545231223106384,
                    0.929176390171051,
                    0.330748051404953,
                    0.014128931798040867,
                    0.013960899785161018,
                ] + [1.0] * (32 - 8),
                'q01': [
                    -0.3524468903720379,
                    -0.26824864755272865,
                    0.04083745917417109,
                    1.5317653684616088,
                    -2.7152330031871794,
                    -1.076538143157959,
                    0.001715825623134151,
                    -0.04003722561979666,
                ] + [-1.0] * (32 - 8),
                'q99': [
                    0.13891278689503672,
                    0.3251991607129573,
                    1.2568962905768304,
                    3.26276856803894,
                    2.4437233173847197,
                    0.5638469840288161,
                    0.04030780866963323,
                    -0.0017131616945378486,
                ] + [1.0] * (32 - 8),
            },
            action_norm_stats={
                'mean': [
                    0.026827840134501457,
                    0.08886060863733292,
                    -0.09983397275209427,
                    0.00024006747116800398,
                    0.0012838079128414392,
                    -0.0029443209059536457,
                    -0.1305243819952011,
                ] + [0.0] * (32 - 7),
                'std': [
                    0.3311910927295685,
                    0.37191954255104065,
                    0.45225635170936584,
                    0.03948824852705002,
                    0.06278067082166672,
                    0.07317619770765305,
                    0.9914451241493225,
                ] + [1.0] * (32 - 7),
                'q01': [
                    -0.747375,
                    -0.796125,
                    -0.9375,
                    -0.11580300460159779,
                    -0.16942972007393836,
                    -0.194502209174633,
                    -1.0,
                ] + [-1.0] * (32 - 7),
                'q99': [
                    0.937125,
                    0.8594999999999999,
                    0.937125,
                    0.1402260055720806,
                    0.18103543001413347,
                    0.3115457148551941,
                    0.9996,
                ] + [1.0] * (32 - 7),
            },
            original_action_dim=32,
        )
        self.pipeline.to(module.device)
        # self.pipeline.compile(fullgraph=True)

        # Perform a dummy forward in order to initialize fsdp
        with torch.no_grad(), torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
            _ = self.module(
                images=[torch.zeros((1, 3, 224, 224), device=self.pipeline.device, dtype=torch.float32)],
                img_masks=[torch.ones((1,), device=self.pipeline.device, dtype=torch.bool)],
                lang_tokens=torch.zeros((1, 1), device=self.pipeline.device, dtype=torch.long),
                lang_masks=torch.ones((1, 1), device=self.pipeline.device, dtype=torch.bool),
                state=torch.zeros((1, self.module.policy.max_state_dim), device=self.pipeline.device, dtype=torch.float32),
                x_t=self.module.policy.sample_noise(
                    (1, self.module.policy.n_action_steps, self.module.policy.max_action_dim),
                    device=self.pipeline.device,
                ),
                timestep=torch.full((1,), 0.5, device=self.pipeline.device, dtype=torch.float32),
            )

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Generate sequences

        Example prompts:
            DataProto(
            batch = TensorDict(
                fields = {
                full_image: Tensor(
                    shape = torch.Size([4, 512, 512, 3]),
                    device = cuda:0,
                    dtype = torch.uint8,
                    is_shared = True
                ),
                state: Tensor(
                    shape = torch.Size([4, 7]),
                    device = cuda:0,
                    dtype = torch.float32,
                    is_shared = True
                )
                },
                batch_size = torch.Size([4]),
                device = cuda:0,
                is_shared = True
            ),

            non_tensor_batch = {
                'task_descriptions': array(
                [
                    'put both moka pots on the stove',
                    'put both moka pots on the stove',
                    'put both moka pots on the stove',
                    'put both moka pots on the stove'
                ],
                dtype = object
                )
            },

            meta_info = {
                'global_steps': 1,
                'do_sample': True,
                'temperature': 1.6,
                'prompt_length': 512,
                'eos_token_id': None,
                'n_samples': 8,
                'pad_token_id': None
            }
            )
        """

        images = prompts.batch["full_image"]
        wrist_images = prompts.batch["wrist_image"]
        state = prompts.batch["state"]
        task_descriptions = prompts.non_tensor_batch["task_descriptions"]

        timing_generate = {}
        with simple_timer("rollout generate_sequences", timing_generate):
            with torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
                batch_size = images.shape[0]
                cam_high = images.permute(0, 3, 1, 2).to(prompts.batch.device)
                left_wrist = wrist_images.permute(0, 3, 1, 2).to(prompts.batch.device)
                zeros = torch.zeros(
                    (batch_size, 3, cam_high.shape[2], cam_high.shape[3]),
                    device=prompts.batch.device,
                    dtype=torch.uint8,
                )
                (
                    action,
                    images_out,
                    img_masks,
                    lang_tokens,
                    lang_masks,
                    state_out,
                ) = self.pipeline(
                    images={
                        "observation.images.cam_high": cam_high,
                        "observation.images.cam_left_wrist": left_wrist,
                        "observation.images.cam_right_wrist": zeros,
                    },
                    task=task_descriptions.tolist() if hasattr(task_descriptions, "tolist") else list(task_descriptions),
                    state=torch.nn.functional.pad(
                        state, (0, max(0, 32 - state.shape[-1])), "constant", 0,
                    ).to(prompts.batch.device, dtype=torch.float32),
                )

        print("rollout generate_sequences time (s): %s" % timing_generate.get("rollout generate_sequences", 0.0))

        ret = DataProto.from_dict(
            {
                "action": action[:, :10, :7],
                "full_action": action,
                "images": torch.stack(images_out, dim=1),
                "image_masks": torch.stack(img_masks, dim=1),
                "lang_tokens": lang_tokens,
                "lang_masks": lang_masks,
                "states": state_out,
            }
        )

        return ret
