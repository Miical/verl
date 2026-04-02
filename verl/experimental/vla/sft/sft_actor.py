from __future__ import annotations

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from omegaconf import OmegaConf

from verl import DataProto
from verl.experimental.vla.sft.base import SupportSFTTraining
from verl.utils.device import get_device_id, get_device_name


class RobSFTActor:
    """Minimal VLA SFT actor scaffold built on top of the FSDP actor module."""

    def __init__(self, config, actor_module: SupportSFTTraining, actor_optimizer, tokenizer, processor=None):
        self.config = config
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.tokenizer = tokenizer
        self.processor = processor
        self.actor_module.sft_init()
        self.device = get_device_name()

    def _extract_obs(self, micro_batch: DataProto) -> DataProto:
        return micro_batch

    def _extract_actions(self, micro_batch: DataProto) -> dict[str, torch.Tensor]:
        batch = micro_batch.batch
        if batch is None:
            raise ValueError("micro_batch.batch must not be None")

        if "action.full_action" in batch:
            return {"full_action": batch["action.full_action"]}
        if "action.action" in batch:
            return {"full_action": batch["action.action"]}
        if "action" in batch:
            return {"full_action": batch["action"]}
        raise KeyError("No action tensor found. Expected one of: action.full_action, action.action, action")

    def _extract_valids(self, micro_batch: DataProto) -> torch.Tensor:
        batch = micro_batch.batch
        if batch is not None and "info.valids" in batch:
            return batch["info.valids"].float()
        return torch.ones(len(micro_batch), device=get_device_id(), dtype=torch.float32)

    def update_policy(self, data: DataProto) -> dict[str, float]:
        self.actor_module.train()

        actor_config = self.config.actor
        mini_batch_size = int(actor_config.ppo_mini_batch_size)
        micro_batch_size = actor_config.ppo_micro_batch_size_per_gpu
        if micro_batch_size is None:
            micro_batch_size = mini_batch_size
        micro_batch_size = int(micro_batch_size)

        mini_batches = data.split(mini_batch_size)
        grad_accum_steps = 0
        for mini_batch in mini_batches:
            grad_accum_steps += len(mini_batch.split(micro_batch_size))
        grad_accum_steps *= torch.distributed.get_world_size()

        self.actor_optimizer.zero_grad()

        loss_list = []
        for mini_batch in mini_batches:
            micro_batches = mini_batch.split(micro_batch_size)
            for micro_batch in micro_batches:
                micro_batch = micro_batch.to(get_device_id())
                obs = self._extract_obs(micro_batch)
                actions = self._extract_actions(micro_batch)
                valids = self._extract_valids(micro_batch)

                with torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
                    bc_loss = self.actor_module.bc_loss(
                        obs=obs,
                        tokenizer=self.tokenizer,
                        actions=actions,
                        valids=valids,
                    )

                (bc_loss / grad_accum_steps).backward()
                loss_list.append(bc_loss.detach())

        grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=actor_config.grad_clip)
        self.actor_optimizer.step()

        if isinstance(self.actor_module, FSDP):
            torch.cuda.empty_cache()

        mean_loss = torch.stack(loss_list).mean().item() if loss_list else 0.0
        grad_norm_value = grad_norm.item() if torch.is_tensor(grad_norm) else float(grad_norm)
        return {
            "loss": mean_loss,
            "grad_norm": grad_norm_value,
        }

    def summary(self) -> dict[str, Any]:
        module = self.actor_module.module if hasattr(self.actor_module, "module") else self.actor_module
        return {
            "policy_type": OmegaConf.select(self.config, "model.override_config.policy_type"),
            "module_class": module.__class__.__name__,
            "tokenizer_class": type(self.tokenizer).__name__ if self.tokenizer is not None else None,
            "processor_class": type(self.processor).__name__ if self.processor is not None else None,
        }
