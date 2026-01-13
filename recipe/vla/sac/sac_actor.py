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
        actor_optimizer: torch.optim.Optimizer = None,
    ):
        """When optimizer is None, it is Reference Policy"""
        super().__init__()
        self.config = config
        self.device = get_device_name()

        self.actor_optimizer = actor_optimizer
        self.actor_module = actor_module
        self.actor_module.sac_init()

        self.replay_pool = SACReplayPool(capacity=self.config.replay_pool_capacity, sample_device=self.device)
        self.replay_pool.load(self.config.replay_pool_save_dir)


    def _calculate_actor_loss(
        self,
        log_probs: torch.Tensor,
        q_values: torch.Tensor,
        action_loss_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate actor loss using the PI0 loss function.

        Args:
            log_probs: Tensor of shape (B,) representing the log probabilities of actions.
            q_values: Tensor of shape (B,) representing the Q-values for the actions.
            action_loss_mask: Tensor of shape (B, action_steps) indicating valid actions for loss computation.

        Returns:
            Tensor of shape (1,) representing the actor loss.
        """

        alpha = 0.5  # config

        # valid sample mask: (B,)
        valid = action_loss_mask.any(dim=1).to(torch.float32)

        # L_pi = E[ alpha * logpi - Q ]
        loss = (alpha * log_probs - q_values)

        actor_loss = (loss * valid).sum() / (valid.sum().clamp_min(1.0))

        return actor_loss

    def _calculate_critic_loss(
            self,
            q_pred: torch.Tensor,        # (B,)
            q_targ: torch.Tensor,        # (B,)
            rewards: torch.Tensor,       # (B,)
            done: torch.Tensor,          # (B,)  1=terminal
            next_log_prob: torch.Tensor, # (B,)
        ) -> torch.Tensor:
            # gamma = self.config.discount
            # alpha = self.temperature  # 或 self.log_alpha.exp() / self.temperature

            gamma = 0.5 # config
            alpha = 0.5 # config

            with torch.no_grad():
                y = rewards + (1.0 - done.to(torch.float32)) * gamma * (q_targ - alpha * next_log_prob)  # (B,)

            critic_loss = F.mse_loss(q_pred, y, reduction="none").mean(dim=0).sum()
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
                    q_pred=q_values_0,
                    q_targ=q_values_1,
                    rewards=micro_batch["rewards"].max(dim=-1).values,
                    done=micro_batch["response_mask"].max(dim=-1).values,
                    next_log_prob=log_probs_1
                )
        print("training forward_critic(s): %s" % timing_generate.get("_forward_critic", 0.0))
        return critic_loss

    def _forward_actor(self, micro_batch: TensorDict) -> torch.Tensor:
        micro_batch = micro_batch.to(get_device_id())
        s0 = get_dict_from_prefix(micro_batch, "s0.")

        timing_generate = {}
        with simple_timer("_forward_actor", timing_generate):
            with torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
                log_probs, q_values_0 = self.actor_module.sac_forward_actor(s0)
                actor_loss = self._calculate_actor_loss(
                    log_probs=log_probs,
                    q_values=q_values_0,
                    action_loss_mask=micro_batch["a0.full_action"].bool()
                )
        print("training forward_actor(s): %s" % timing_generate.get("_forward_actor", 0.0))
        return actor_loss

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
                - "returns": Tensor of shape (B,),
                    chunk-level discounted returns aligned to the next step.
                - "response_mask": Tensor of shape (B,),
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
            "returns",
            "response_mask"
        ]).batch

        batch = self.replay_pool.insert_and_resample(batch)
        micro_batches = batch.split(self.config.ppo_micro_batch_size_per_gpu)

        # Training critic
        metrics = {}
        self.actor_optimizer.zero_grad()
        for batch_idx, micro_batch in enumerate(micro_batches):
            print(f"[{batch_idx+1}/{len(micro_batches)}] critic micro batch ")

            micro_batch = micro_batch.to(get_device_id())
            critic_loss = self._forward_critic(micro_batch)
            critic_loss.backward()
        grad_norm = self._optimizer_step()

        # Training actor
        self.actor_optimizer.zero_grad()
        for batch_idx, micro_batch in enumerate(micro_batches):
            print(f"[{batch_idx+1}/{len(micro_batches)}] actor micro batch ")

            micro_batch = micro_batch.to(get_device_id())
            actor_loss = self._forward_actor(micro_batch)
            actor_loss.backward()
        grad_norm = self._optimizer_step()


        # mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
        # append_to_dict(metrics, mini_batch_metrics)

        # TODO: save replay pool periodically
        self.replay_pool.save(self.config.replay_pool_save_dir)

        return metrics

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        self.actor_optimizer.step()
        return grad_norm
