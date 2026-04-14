from __future__ import annotations

from typing import Any

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
        actor_cfg = self.config.actor if hasattr(self.config, "actor") else self.config
        ema_enabled = actor_cfg.get("actor_ema_enabled", self.config.get("actor_ema_enabled", True))
        ema_decay = actor_cfg.get("actor_ema_decay", self.config.get("actor_ema_decay", 0.995))
        self.actor_ema_enabled = bool(ema_enabled)
        self.actor_ema_decay = float(ema_decay)
        self.actor_ema_shadow: dict[str, torch.Tensor] = {}
        self.actor_ema_initialized = False

    def _init_actor_ema(self):
        if self.actor_ema_initialized:
            return

        self.actor_ema_shadow = {}
        if not self.actor_ema_enabled:
            self.actor_ema_initialized = True
            return

        for name, param in self.actor_module.named_parameters():
            self.actor_ema_shadow[name] = param.detach().clone().to(dtype=torch.float32)
        self.actor_ema_initialized = True

    @torch.no_grad()
    def _update_actor_ema(self):
        if not self.actor_ema_enabled:
            return

        one_minus_decay = 1.0 - self.actor_ema_decay
        for name, param in self.actor_module.named_parameters():
            shadow = self.actor_ema_shadow[name]
            shadow.mul_(self.actor_ema_decay).add_(param.detach().to(dtype=torch.float32), alpha=one_minus_decay)

    @torch.no_grad()
    def _apply_actor_ema_to_actor_module(self):
        if not self.actor_ema_enabled:
            return

        for name, param in self.actor_module.named_parameters():
            shadow = self.actor_ema_shadow[name]
            param.copy_(shadow.to(device=param.device, dtype=param.dtype))

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
    
    
    def _force_set_lr(self, opt: torch.optim.Optimizer, lr: float):
        for pg in opt.param_groups:
            pg["lr"] = lr

    def update_policy(self, data: DataProto) -> dict[str, float]:
        if not self.actor_ema_initialized:
            self._init_actor_ema()

        self.actor_module.train()

        self._force_set_lr(self.actor_optimizer, 5e-5)

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
        self._update_actor_ema()
        self._apply_actor_ema_to_actor_module()

        if isinstance(self.actor_module, FSDP):
            torch.cuda.empty_cache()

        mean_loss = torch.stack(loss_list).mean().item() if loss_list else 0.0
        grad_norm_value = grad_norm.item() if torch.is_tensor(grad_norm) else float(grad_norm)
        return {
            "loss": mean_loss,
            "grad_norm": grad_norm_value,
            "sft/actor_ema_enabled": float(self.actor_ema_enabled),
            "sft/actor_ema_decay": self.actor_ema_decay,
        }

    def summary(self) -> dict[str, Any]:
        module = self.actor_module.module if hasattr(self.actor_module, "module") else self.actor_module
        return {
            "policy_type": OmegaConf.select(self.config, "model.override_config.policy_type"),
            "module_class": module.__class__.__name__,
            "tokenizer_class": type(self.tokenizer).__name__ if self.tokenizer is not None else None,
            "processor_class": type(self.processor).__name__ if self.processor is not None else None,
        }
