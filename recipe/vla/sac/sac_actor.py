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

import torch
import logging
from typing_extensions import override
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.nn.functional as F

from verl.protocol import DataProto
from verl.utils.device import get_device_id, get_device_name
from verl.utils.py_functional import append_to_dict
from verl.utils.profiler import simple_timer
from .base import SupportSACTraining, BaseSACActor

logger = logging.getLogger(__name__)

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

class PI0RobDataParallelPPOActor(BaseSACActor):
    def __init__(
        self,
        config,
        actor_module: SupportSACTraining,
        actor_optimizer: torch.optim.Optimizer = None,
    ):
        """When optimizer is None, it is Reference Policy"""
        super().__init__()
        self.config = config
        self.device = get_device_name()
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer

        #TODO: remove it
        self.loss_func = PI0Loss()

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        logger.info(f"Actor use_remove_padding={self.use_remove_padding}")

    def _calculate_actor_loss(self, model_pred: torch.Tensor, action_loss_mask: torch.Tensor) -> torch.Tensor:
        # TODO: sac actor loss input args: logprobs(B, T, D), q_value(B, 1)
        """Calculate actor loss using the PI0 loss function.

        Args:
            model_pred: Predicted v_t with shape (B, T, D).
            action_loss_mask: Mask of shape (B, T) indicating valid action steps.

        Returns:
            Tensor of shape (1,) representing the actor loss.
        """

        loss = self.loss_func(model_pred, action_loss_mask)
        training_actions_per_chunk = self.config.get("training_actions_per_chunk", loss.shape[1])
        actor_loss = loss[:, :training_actions_per_chunk].mean()
        return actor_loss

    def _calculate_critic_loss(self, q_values: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
        """Calculate critic loss as MSE between Q-values and rewards.

        Args:
            q_values: Tensor of shape (B,) representing predicted Q-values.
            rewards: Tensor of shape (B,) representing observed rewards.

        Returns:
            Tensor of shape (1,) representing the critic loss.
        """

        # TODO: implement critic loss

        return None

    def _forward_step(self, micro_batch: DataProto) -> torch.Tensor:
        micro_batch = micro_batch.to(get_device_id())

        # make action loss mask
        actions = micro_batch["a0.full_action"]
        batch_size, action_chunk_size = actions.shape[0], actions.shape[1]
        training_actions_per_chunk = self.config.get("training_actions_per_chunk", action_chunk_size)
        action_loss_mask = torch.cat([
            torch.ones((batch_size, training_actions_per_chunk),
                       dtype=torch.bool,
                       device=self.device),
            torch.zeros((batch_size, action_chunk_size - training_actions_per_chunk),
                        dtype=torch.bool,
                        device=self.device)
        ], dim=1)

        # get model prediction
        noisy_model_input, timesteps = self.loss_func.add_noise(actions)
        timing_generate = {}
        with simple_timer("training forward_step", timing_generate):
            with torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
                # TODO: change to sac_forward
                model_pred = self.actor_module(
                    micro_batch["s0.images"].unbind(1),
                    micro_batch["s0.image_masks"].unbind(1),
                    micro_batch["s0.lang_tokens"],
                    micro_batch["s0.lang_masks"],
                    micro_batch["s0.states"],
                    noisy_model_input,
                    timesteps
                )

        print("training foward_step(s): %s" % timing_generate.get("training forward_step", 0.0))

        # compute losses
        actor_loss = self._calculate_actor_loss(model_pred, action_loss_mask)
        critic_loss = self._calculate_critic_loss(None, micro_batch["rewards"])

        return actor_loss, critic_loss

    @override
    def update_policy(self, data: DataProto):
        """
        Update the policy using the provided data batch.

        Args:
            data: DataProto containing the following entries in `data.batch`:
                - "a0.full_action": Tensor of shape (B, action_steps, action_dim),
                    representing the current action chunk for each sample.
                - "a1.full_action": Tensor of shape (B, action_steps, action_dim),
                    representing the next action chunk for each sample.
                - "s0.states": Tensor of shape (B, state_dim),
                    representing the current environment or agent state.
                - "s1.states": Tensor of shape (B, state_dim),
                    representing the next environment or agent state.
                - "s0.images": Tensor of shape (num_images, B, C, H, W),
                    containing current visual observations.
                - "s1.images": Tensor of shape (num_images, B, C, H, W),
                    containing next-step visual observations.
                - "s0.image_masks": Tensor of shape (num_images, B),
                    indicating valid images per sample.
                - "s1.image_masks": Tensor of shape (num_images, B),
                    indicating valid images per sample.
                - "s0.lang_tokens": Tensor of shape (B, max_seq_len),
                    tokenized language instructions.
                - "s1.lang_tokens": Tensor of shape (B, max_seq_len),
                    tokenized language instructions for the next step.
                - "s0.lang_masks": Tensor of shape (B, max_seq_len),
                    attention masks for language tokens.
                - "s1.lang_masks": Tensor of shape (B, max_seq_len),
                    attention masks for language tokens for the next step.
                - "rewards": Tensor of shape (B,),
                    chunk-level scalar rewards aligned to the next step.
                - "returns": Tensor of shape (B,),
                    chunk-level discounted returns aligned to the next step.
                - "response_mask": Tensor of shape (B,),
                    mask indicating whether each sample has a valid response.
        """

        batch = data.select([
            "a0.full_action",
            "a1.full_action",
            "s0.states",
            "s1.states",
            "s0.images",
            "s1.images",
            "s0.image_masks",
            "s1.image_masks",
            "s0.lang_tokens",
            "s1.lang_tokens",
            "s0.lang_masks",
            "s1.lang_masks",
            "rewards",
            "returns",
            "response_mask"
        ]).batch

        micro_batches = batch.split(self.config.ppo_micro_batch_size_per_gpu)

        metrics = {}
        for batch_idx, micro_batch in enumerate(micro_batches):
            print(f"[{batch_idx+1}/{len(micro_batches)}] actor micro batch ")
            self.actor_optimizer.zero_grad()

            actor_loss, critic_loss = self._forward_step(micro_batch)
            actor_loss.backward()
            # critic_loss.backward()

            grad_norm = self._optimizer_step()
            mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
            append_to_dict(metrics, mini_batch_metrics)

        return metrics

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        self.actor_optimizer.step()
        return grad_norm
