# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
Single Process Actor
"""

import logging

import torch
from tensordict.base import TensorDictBase
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl.protocol import DataProto
from verl.trainer.ppo import core_algos
from verl.utils.device import get_device_id, get_device_name
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.profiler import simple_timer
from verl.workers.actor import BasePPOActor

logger = logging.getLogger(__name__)

__all__ = ["RobDataParallelPPOActor"]


import torch
import torch.nn as nn
import torch.nn.functional as F

class PI0Loss(nn.Module):
    """Diffusion-style training loss for PI0 actions."""

    def __init__(self) -> None:
        super().__init__()

    def sample_noise(self, shape: tuple[int, ...], device: torch.device | str) -> torch.Tensor:
        """Sample standard normal noise with the given shape on the device."""
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )
        return noise

    def _sample_beta(self, alpha: float, beta: float, bsize: int, device: torch.device | str) -> torch.Tensor:
        """Sample from Beta(alpha, beta) using the ratio of powered uniforms
        trick.

        Returns:
            A tensor of shape (bsize,) with samples in (0, 1).
        """
        gamma1 = torch.empty((bsize,), device=device).uniform_(0, 1).pow(1 / alpha)
        gamma2 = torch.empty((bsize,), device=device).uniform_(0, 1).pow(1 / beta)
        return gamma1 / (gamma1 + gamma2)

    def sample_time(self, bsize: int, device: torch.device | str) -> torch.Tensor:
        """Sample diffusion times in (0.001, 1.0) biased toward later times."""
        time_beta = self._sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def add_noise(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Create noisy actions x_t and compute the target u_t for training.

        Args:
            actions: Ground-truth actions of shape (B, T, D).

        Returns:
            A tuple (x_t, time) where x_t has shape (B, T, D) and time has shape (B,).
        """
        noise = self.sample_noise(actions.shape, actions.device)
        time = self.sample_time(actions.shape[0], actions.device)
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        self.x_t = x_t
        self.u_t = u_t
        self.time = time

        return x_t, time

    def forward(self, model_pred: torch.Tensor, loss_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Compute v-prediction MSE loss.

        Args:
            model_pred: Predicted v_t with shape (B, T, D).
            loss_mask: Optional mask (B, T) to ignore padded steps.

        Returns:
            Per-sample loss with shape (B, T) if mask provided then masked accordingly.
        Note:
            `add_noise` must be called before forward to set `self.u_t`.
        """
        target = self.u_t
        if target.dtype != model_pred.dtype:
            target = target.to(model_pred.dtype)
        loss = F.mse_loss(target, model_pred, reduction='none').mean(dim=-1)
        if loss_mask is not None:
            loss = loss * loss_mask.to(loss.dtype)

        return loss


class PI0RobDataParallelPPOActor(BasePPOActor):
    def __init__(
        self,
        config,
        actor_module: nn.Module,
        actor_optimizer: torch.optim.Optimizer = None,
    ):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.use_remove_padding = self.config.get("use_remove_padding", False)
        logger.info(f"Actor use_remove_padding={self.use_remove_padding}")
        logger.info(f"PRM use dynamic bsz={self.config.get('use_dynamic_bsz', False)}")
        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = False  # self.ulysses_sequence_parallel_size > 1
        self.compute_entropy_from_logits = torch.compile(verl_F.entropy_from_logits, dynamic=True)

        self.loss_func = PI0Loss()

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        self.actor_optimizer.step()
        return grad_norm

    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        pass

    def forward_step(self, batch_dict: dict[str, torch.Tensor | list[torch.Tensor]]) -> torch.Tensor:
        """Perform one training step and return the loss tensor.

        Args:
            batch_dict: Preprocessed batch containing images, masks, tokens, state and actions.

        Returns:
            Loss tensor (e.g., per-sample or aggregated depending on the Trainer reduction).
        """
        images = batch_dict['images']
        img_masks = batch_dict['image_masks']
        lang_tokens = batch_dict['lang_tokens']
        lang_masks = batch_dict['lang_masks']
        state = batch_dict['observation.state']
        actions = batch_dict['action']
        action_loss_mask = batch_dict['action_loss_mask']

        noisy_model_input, timesteps = self.loss_func.add_noise(actions)

        timing_generate = {}
        with simple_timer("training forward_step", timing_generate):
            with torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
                model_pred = self.actor_module(images, img_masks, lang_tokens, lang_masks, state, noisy_model_input, timesteps)

        print("training foward_step(s): %s" % timing_generate.get("training forward_step", 0.0))

        loss = self.loss_func(model_pred, loss_mask=action_loss_mask)
        return loss

    def update_policy(self, data: DataProto):
        assert self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size_per_gpu == 0
        self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid slient error

        select_keys = [
            "advantages",
            "full_action",
            "states",
            "images",
            "image_masks",
            "lang_tokens",
            "lang_masks",
            "reward_tensor",
        ]
        batch = data.select(batch_keys=select_keys).batch
        self.pad_token_id = data.meta_info["pad_token_id"]

        mini_batches = batch.split(self.config.ppo_mini_batch_size)
        metrics = {}
        for batch_idx, mini_batch in enumerate(mini_batches):
            if self.config.use_dynamic_bsz:
                max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
            else:
                self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

            self.actor_optimizer.zero_grad()

            for micro_batch in micro_batches:
                micro_batch = micro_batch.to(get_device_id())  # actor device is cpu when using offload
                bsz = micro_batch['full_action'].shape[0]
                print("micro_batch bsz", bsz)
                loss = self.forward_step({
                    'images': list(micro_batch['images'].unbind(1)),
                    'image_masks': list(micro_batch['image_masks'].unbind(1)),
                    'lang_tokens': micro_batch['lang_tokens'],
                    'lang_masks': micro_batch['lang_masks'],
                    'observation.state': micro_batch['states'],
                    'action': micro_batch['full_action'],
                    'action_loss_mask': torch.cat([
                        torch.ones((bsz, 10), dtype=torch.bool, device=micro_batch['full_action'].device),
                        torch.zeros((bsz, 40), dtype=torch.bool, device=micro_batch['full_action'].device),
                    ], dim=1)
                })

                advantages = micro_batch["advantages"].to(loss.device)
                loss = loss[:, 10]
                advantages = advantages.view(bsz, 10, 7).mean(dim=-1)
                loss = (loss * advantages).mean()

                loss.backward()

            grad_norm = self._optimizer_step()
            mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
            append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        return metrics
