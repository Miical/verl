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
import math
import torch
import numpy as np
from typing_extensions import override
from tensordict import TensorDict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.nn.functional as F

from verl.protocol import DataProto
from verl.utils.device import get_device_id, get_device_name
from verl.utils.py_functional import append_to_dict
from verl.utils.profiler import simple_timer
from verl.utils.replay_pool import SACReplayPool
from .base import SupportSACTraining, BaseSACActor

logger = logging.getLogger(__name__)


def get_dict_from_prefix(tensordict: TensorDict, prefix: str) -> dict:
    """Extract a sub-dictionary from a TensorDict based on a given prefix.

    Args:
        tensordict: The input TensorDict containing various keys.
        prefix: The prefix string to filter keys.
    Returns:
        A dictionary containing key-value pairs from the TensorDict
        where the keys start with the specified prefix. The prefix is removed
        from the keys in the resulting dictionary.
    """

    result = {}
    prefix_length = len(prefix)
    for key in tensordict.keys():
        if key.startswith(prefix):
            new_key = key[prefix_length:]
            result[new_key] = tensordict[key]
    return result


class PI0RobDataParallelPPOActor(BaseSACActor):
    def __init__(
        self,
        config,
        actor_module: SupportSACTraining,
        actor_optimizer: torch.optim.Optimizer,
    ):
        super().__init__()
        self.config = config
        self.sac_config = config.sac
        self.device = get_device_name()

        self.actor_optimizer = actor_optimizer
        self.actor_module = actor_module
        self.actor_module.sac_init()

        self.replay_pool = SACReplayPool(capacity=self.config.replay_pool_capacity, sample_device=self.device)
        self.replay_pool.load(self.config.replay_pool_save_dir)

        self._init_alpha()

    def _init_alpha(self):
        """Initialize the alpha optimizer for automatic entropy tuning."""

        self.auto_entropy = self.sac_config.get("auto_entropy", False)

        if self.auto_entropy:
            self.target_entropy = torch.tensor(
                float(self.sac_config.get("target_entropy", -32.0)),
                device=self.device
            )

            # Initialize raw_alpha parameter
            self.alpha_type = self.sac_config.get("alpha_type", "softplus")
            if self.alpha_type == "exp":
                self.raw_alpha = torch.nn.Parameter(
                    np.log(np.exp(self.sac_config.get("initial_alpha", 1))) * torch.ones(1, device=self.device),
                    requires_grad=True,
                )
            elif self.alpha_type == "softplus":
                self.raw_alpha = torch.nn.Parameter(
                    np.log(np.exp(self.sac_config.get("initial_alpha", 0.01)) - 1) * torch.ones(1, device=self.device),
                    requires_grad=True,
                )
            else:
                return NotImplementedError(f"Unsupported alpha_type: {self.alpha_type}")

            # build alpha optimizer and scheduler
            self.alpha_optimizer = torch.optim.Adam([self.raw_alpha], lr=self.sac_config.get("alpha_lr", 3e-4))
            self.alpha_scheduler = torch.optim.lr_scheduler.ConstantLR(
                self.alpha_optimizer, factor=1.0
            )

    def _get_alpha(self) -> torch.Tensor:
        if self.auto_entropy:
            if self.alpha_type == "exp":
                return self.raw_alpha.exp()
            elif self.alpha_type == "softplus":
                return torch.nn.functional.softplus(self.raw_alpha)
            else:
                return NotImplementedError(f"Unsupported alpha_type: {self.alpha_type}")
        else:
            return torch.tensor(float(self.sac_config.get("initial_alpha", 0.2)), device=self.device)

    def _calculate_actor_loss(
        self,
        log_probs: torch.Tensor,
        q_values: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate actor loss using the SAC loss function.

        Args:
            log_probs: Tensor of shape (B,) representing the log probabilities of actions.
            q_values: Tensor of shape (B,) representing the Q-values for the actions.
            valid: Tensor of shape (B,) indicating valid samples (1 for valid, 0 for invalid).

        Returns:
            Tensor of shape (1,) representing the actor loss.
        """

        alpha = self._get_alpha()
        loss = (alpha * log_probs - q_values)
        actor_loss = (loss * valid).sum() / (valid.sum().clamp_min(1.0))

        return actor_loss

    def _calculate_alpha_loss(self, log_probs: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Calculate alpha loss for automatic entropy tuning.

        Args:
            log_probs: Tensor of shape (B,) representing the log probabilities of actions.
            valid: Tensor of shape (B,) indicating valid samples (1 for valid, 0 for invalid).

        Returns:
            Tensor of shape (1,) representing the alpha loss.
        """

        alpha_loss = -self._get_alpha() * (log_probs.detach() + self.target_entropy)
        alpha_loss = (alpha_loss * valid).sum() / (valid.sum().clamp_min(1.0))
        return alpha_loss

    def _calculate_critic_loss(
        self,
        q_predict: torch.Tensor,
        q_target: torch.Tensor,
        rewards: torch.Tensor,
        valid: torch.Tensor,
        next_log_prob: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate critic loss using the SAC loss function.

        Args:
            q_predict: Tensor of shape (B, critic_num) representing predicted Q-values.
            q_target: Tensor of shape (B,) representing target Q-values.
            rewards: Tensor of shape (B,) representing rewards.
            valid: Tensor of shape (B,) indicating valid samples (1 for valid, 0 for invalid).
            next_log_prob: Tensor of shape (B,) representing log probabilities of next actions.

        Returns:
            Tensor of shape (1,) representing the critic loss.
        """

        gamma = self.sac_config.gamma
        alpha = self._get_alpha()

        with torch.no_grad():
            y = rewards + valid * gamma * (q_target - alpha * next_log_prob)

        y = y.unsqueeze(1).expand_as(q_predict) # (B, critic_num)
        critic_loss = F.mse_loss(q_predict, y, reduction="none").mean(dim=0).sum()
        return critic_loss

    def _forward_critic(self, micro_batch: TensorDict) -> torch.Tensor:
        s0 = get_dict_from_prefix(micro_batch, "s0.")
        s1 = get_dict_from_prefix(micro_batch, "s1.")
        a0 = get_dict_from_prefix(micro_batch, "a0.")

        timing_generate = {}
        with simple_timer("_forward_critic", timing_generate):
            with torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
                q_values_0, q_values_1, log_probs_1 = self.actor_module.sac_forward_critic(s0, a0, s1)
                critic_loss = self._calculate_critic_loss(
                    q_predict=q_values_0,
                    q_target=q_values_1,
                    rewards=micro_batch["rewards"].max(dim=-1).values,
                    valid=micro_batch["valid"],
                    next_log_prob=log_probs_1
                )
        print("training forward_critic(s): %s" % timing_generate.get("_forward_critic", 0.0))
        return critic_loss

    def _forward_actor(self, micro_batch: TensorDict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        micro_batch = micro_batch.to(get_device_id())
        s0 = get_dict_from_prefix(micro_batch, "s0.")

        timing_generate = {}
        with simple_timer("_forward_actor", timing_generate):
            with torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
                log_probs, q_values_0 = self.actor_module.sac_forward_actor(s0)
                actor_loss = self._calculate_actor_loss(
                    log_probs=log_probs,
                    q_values=q_values_0,
                    valid=micro_batch["valid"],
                )
        print("training forward_actor(s): %s" % timing_generate.get("_forward_actor", 0.0))
        return actor_loss, log_probs

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
                - "s0.images": Tensor of shape (B, n_images, C, H, W),
                    containing current visual observations.
                - "s1.images": Tensor of shape (B, n_images, C, H, W),
                    containing next-step visual observations.
                - "s0.image_masks": Tensor of shape (B, n_images),
                    indicating valid images per sample.
                - "s1.image_masks": Tensor of shape (B, n_images),
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
                - "response_mask": Tensor of shape (B, action_steps),
                    mask indicating whether each sample has a valid response.
        """

        batch: TensorDict = data.select([
            "a0.full_action", "a1.full_action",
            "s0.states", "s1.states",
            "s0.images", "s1.images",
            "s0.image_masks", "s1.image_masks",
            "s0.lang_tokens", "s1.lang_tokens",
            "s0.lang_masks", "s1.lang_masks",
            "rewards",
            "response_mask"
        ]).batch
        global_steps = data.meta_info["global_steps"]

        batch = self.replay_pool.insert_and_resample(batch)
        batch["valid"] = batch["response_mask"].min(dim=-1).values # (B,)
        micro_batches = batch.split(self.config.ppo_micro_batch_size_per_gpu)
        actor_loss_list, critic_loss_list, alpha_loss_list = [], [], []

        # Training critic
        self.actor_optimizer.zero_grad()
        for batch_idx, micro_batch in enumerate(micro_batches):
            print(f"[{batch_idx+1}/{len(micro_batches)}] critic micro batch ")

            micro_batch = micro_batch.to(get_device_id())
            critic_loss = self._forward_critic(micro_batch)
            critic_loss.backward()
            critic_loss_list.append(critic_loss.detach().item())
        critic_grad_norm = self._optimizer_step(self.actor_optimizer)

        # Training actor
        self.actor_optimizer.zero_grad()
        actor_logprobs_list = []
        for batch_idx, micro_batch in enumerate(micro_batches):
            print(f"[{batch_idx+1}/{len(micro_batches)}] actor micro batch ")

            micro_batch = micro_batch.to(get_device_id())
            actor_loss, log_probs = self._forward_actor(micro_batch)
            actor_loss.backward()
            actor_loss_list.append(actor_loss.detach().item())
            actor_logprobs_list.append(log_probs.detach())
        actor_grad_norm = self._optimizer_step(self.actor_optimizer)

        # Training alpha
        # NOTE: We reuse the log-probabilities computed during the actor forward pass
        # to update the entropy temperature (alpha), instead of re-forwarding
        # the actor after the policy update (saving compute).
        if self.auto_entropy:
            self.alpha_optimizer.zero_grad()
            for micro_batch, log_probs in zip(micro_batches, actor_logprobs_list):
                micro_batch = micro_batch.to(get_device_id())
                alpha_loss = self._calculate_alpha_loss(log_probs, micro_batch["valid"])
                alpha_loss.backward()
                alpha_loss_list.append(alpha_loss.detach().item())
            torch.distributed.all_reduce(
                self.raw_alpha.grad, op=torch.distributed.ReduceOp.AVG
            )
            alpha_grad_norm = self._optimizer_step(self.alpha_optimizer)
            self.alpha_scheduler.step()

        # Update target networks
        self.actor_module.sac_update_target_network(self.sac_config.tau)

        # Save replay pool
        if global_steps % self.config.replay_pool_save_interval == 0:
            self.replay_pool.save(self.config.replay_pool_save_dir)

        # Log metrics
        metrics = {
            "reward/mean": batch["rewards"].mean().item(),
            "reward/std": batch["rewards"].std().item(),
            "valid_ratio": batch["valid"].float().mean().item(),

            "sac/alpha": self._get_alpha().detach().item(),
            "sac/alpha_lr": self.alpha_optimizer.param_groups[0]['lr'] if self.auto_entropy else 0.0,
            "sac/alpha_loss": sum(alpha_loss_list) / len(alpha_loss_list) if alpha_loss_list else 0.0,
            "sac/alpha_grad_norm": alpha_grad_norm.detach().item() if self.auto_entropy else 0.0,
            "sac/replay_pool_size": len(self.replay_pool),

            "actor/loss": sum(actor_loss_list) / len(actor_loss_list) if actor_loss_list else 0.0,
            "actor/lr": self.actor_optimizer.param_groups[0]['lr'],
            "actor/grad_norm": actor_grad_norm.detach().item(),
            "actor/logprob_mean": torch.cat(actor_logprobs_list).mean().detach().item() if actor_logprobs_list else 0.0,

            "critic/loss": sum(critic_loss_list) / len(critic_loss_list) if critic_loss_list else 0.0,
            "critic/grad_norm": critic_grad_norm.detach().item(),
        }

        return metrics

    def _optimizer_step(self, optimizer: torch.optim.Optimizer) -> torch.Tensor:
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        optimizer.step()
        return grad_norm
