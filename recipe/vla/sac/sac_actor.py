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
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.nn.functional as F

from verl.protocol import DataProto
from verl.utils.device import get_device_id, get_device_name
from verl.utils.py_functional import append_to_dict
from verl.utils.profiler import simple_timer
from verl.utils.replay_pool import SACReplayPool
from .base import SupportSACTraining, BaseSACActor

logger = logging.getLogger(__name__)

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
        timing_generate = {}
        self.actor_optimizer.zero_grad()
        with simple_timer("training forward_step", timing_generate):
            with torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):

                q_values_0, q_values_1, log_probs_1, shared_features = self.actor_module.sac_forward_critic(
                    s0 = {
                        "images": micro_batch["s0.images"],
                        "img_masks": micro_batch["s0.image_masks"],
                        "lang_tokens": micro_batch["s0.lang_tokens"],
                        "lang_masks": micro_batch["s0.lang_masks"],
                        "states": micro_batch["s0.states"]
                    },
                    a0 = {
                        "actions": micro_batch["a0.full_action"]
                    },
                    s1 = {
                        "images": micro_batch["s1.images"],
                        "img_masks": micro_batch["s1.image_masks"],
                        "lang_tokens": micro_batch["s1.lang_tokens"],
                        "lang_masks": micro_batch["s1.lang_masks"],
                        "states": micro_batch["s1.states"]
                    },
                    a1 = {
                        "actions": micro_batch["a1.full_action"]
                    }
                )
                critic_loss = self._calculate_critic_loss(
                    q_pred=q_values_0,
                    q_targ=q_values_1,
                    rewards=micro_batch["rewards"].max(dim=-1).values,
                    done=micro_batch["response_mask"].max(dim=-1).values,
                    next_log_prob=log_probs_1
                )
                critic_loss.backward()
                grad_norm = self._optimizer_step()

                self.actor_optimizer.zero_grad()
                log_probs, q_values_0 = self.actor_module.sac_forward_actor(
                    s0 = {
                        "images": micro_batch["s0.images"],
                        "img_masks": micro_batch["s0.image_masks"],
                        "lang_tokens": micro_batch["s0.lang_tokens"],
                        "lang_masks": micro_batch["s0.lang_masks"],
                        "states": micro_batch["s0.states"]
                    },
                    a0 = {
                        "actions": micro_batch["a0.full_action"]
                    },
                    shared_features = shared_features
                )

                # actor_loss = self._calculate_actor_loss(...)
                # actor_loss.backward()
                # optimizer.step()

                self.actor_module.sac_update_target_network(tau=0.01)

                print("log_probs shape:", log_probs.shape, "q_values_0 shape:", q_values_0.shape)

                exit(0)

        print("training foward_step(s): %s" % timing_generate.get("training forward_step", 0.0))

        return grad_norm

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

        batch = self.replay_pool.insert_and_resample(batch)
        micro_batches = batch.split(self.config.ppo_micro_batch_size_per_gpu)

        metrics = {}
        for batch_idx, micro_batch in enumerate(micro_batches):
            print(f"[{batch_idx+1}/{len(micro_batches)}] actor micro batch ")

            grad_norm = self._forward_step(micro_batch)

            mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
            append_to_dict(metrics, mini_batch_metrics)

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
