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

        self.auto_entropy = getattr(self.sac_config, "auto_entropy", False)
        self.alpha_optimizer, self.log_alpha, self.target_entropy = self._init_alpha_optimizer()

    def _init_alpha_optimizer(self):
        """Initialize the alpha optimizer for automatic entropy tuning.

        Returns:
            alpha_optimizer: The optimizer for the log alpha parameter.
            log_alpha: The log alpha parameter tensor.
            target_entropy: The target entropy tensor.
        """

        if not self.auto_entropy:
            return None, None, None

        initial_alpha = float(self.sac_config.alpha)
        log_alpha = torch.tensor(
            math.log(initial_alpha),
            device=get_device_name(),
            requires_grad=True,
        )
        alpha_lr = float(getattr(self.sac_config, "alpha_lr", 3e-4))
        alpha_optimizer = torch.optim.Adam([log_alpha], lr=alpha_lr)
        target_entropy = getattr(self.sac_config, "target_entropy", None)
        if target_entropy is None:
            raise ValueError("sac.target_entropy must be set when sac.auto_entropy is enabled.")
        target_entropy = torch.tensor(float(target_entropy), device=get_device_name())

        return alpha_optimizer, log_alpha, target_entropy

    def _get_alpha(self) -> torch.Tensor:
        if self.auto_entropy:
            return self.log_alpha.exp().detach()
        return torch.tensor(float(self.sac_config.alpha), device=get_device_name())

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

        loss = -self.log_alpha * (log_probs.detach() + self.target_entropy)
        alpha_loss = (loss * valid).sum() / (valid.sum().clamp_min(1.0))
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
                    valid=micro_batch["response_mask"].max(dim=-1).values,
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
                    valid=micro_batch["response_mask"].max(dim=-1).values,
                )
        print("training forward_actor(s): %s" % timing_generate.get("_forward_actor", 0.0))
        valid = micro_batch["response_mask"].max(dim=-1).values
        return actor_loss, log_probs, valid

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
        micro_batches = batch.split(self.config.ppo_micro_batch_size_per_gpu)
        actor_loss_list, critic_loss_list = [], []

        # Training critic
        self.actor_optimizer.zero_grad()
        for batch_idx, micro_batch in enumerate(micro_batches):
            print(f"[{batch_idx+1}/{len(micro_batches)}] critic micro batch ")

            micro_batch = micro_batch.to(get_device_id())
            critic_loss = self._forward_critic(micro_batch)
            critic_loss.backward()
            critic_loss_list.append(critic_loss.detach().item())
        critic_grad_norm = self._optimizer_step()

        # Training actor
        self.actor_optimizer.zero_grad()
        if self.auto_entropy:
            self.alpha_optimizer.zero_grad()
        for batch_idx, micro_batch in enumerate(micro_batches):
            print(f"[{batch_idx+1}/{len(micro_batches)}] actor micro batch ")

            micro_batch = micro_batch.to(get_device_id())
            actor_loss, log_probs, valid = self._forward_actor(micro_batch)
            actor_loss.backward()
            actor_loss_list.append(actor_loss.detach().item())
            if self.auto_entropy:
                alpha_loss = self._calculate_alpha_loss(log_probs, valid)
                alpha_loss.backward()
        actor_grad_norm = self._optimizer_step()
        if self.auto_entropy:
            self.alpha_optimizer.step()

        # Update target networks
        self.actor_module.sac_update_target_network(self.sac_config.tau)

        # Save replay pool
        if global_steps % self.config.replay_pool_save_interval == 0:
            self.replay_pool.save(self.config.replay_pool_save_dir)

        # Log metrics
        metrics = {}
        metrics["reward/mean"] = batch["rewards"].mean().item()
        metrics["reward/std"] = batch["rewards"].std().item()
        metrics["valid_ratio"] = batch["response_mask"].max(dim=-1).values.float().mean().item()
        metrics["actor/replay_pool_size"] = len(self.replay_pool)
        metrics["actor/alpha"] = self._get_alpha().detach().item()
        metrics["actor/loss"] = sum(actor_loss_list) / len(actor_loss_list) if actor_loss_list else 0.0
        metrics["actor/grad_norm"] = actor_grad_norm.detach().item()
        metrics["critic/loss"] = sum(critic_loss_list) / len(critic_loss_list) if critic_loss_list else 0.0
        metrics["critic/grad_norm"] = critic_grad_norm.detach().item()

        return metrics

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        self.actor_optimizer.step()
        return grad_norm
