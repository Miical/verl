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

from __future__ import annotations

import math
from typing import Literal

import torch
from onnx_ir import Tensor
from torch import nn
from torch.distributed.fsdp import register_fsdp_forward_method
from transformers import PreTrainedModel
from typing_extensions import override

from verl.protocol import DataProto
from verl.utils.device import get_device_name

from ...sac.base import SupportSACTraining
from ..modules.mlp import MLP
from .configuration_pi0_torch import PI0TorchConfig
from .model.modeling_pi0 import PI0Model, make_att_2d_masks
from .pi0_utils import (
    ImageTransform,
    Normalize,
    PromptTokenizerTransform,
    Unnormalize,
)
from .policy.base import Pi0Output


def beta_schedule(step, beta0, beta_min, T):
    progress = min(step / T, 1.0)
    beta = beta_min + (beta0 - beta_min) * 0.5 * (1 + math.cos(math.pi * progress))
    return beta


def _resolve_rlpd_task_ids(tasks, config, device: torch.device) -> torch.Tensor:
    """Resolve offline RLPD task ids.

    Real-robot training is usually single-task, so we default to a fixed placeholder
    task id 0. For multi-task offline data, users can pass either:
      - config.rlpd_task_to_id: dict[str, int]
      - config.rlpd_single_task = False together with consistent task strings
    """
    tasks = [str(t) for t in tasks]

    single_task = getattr(config, "rlpd_single_task", True)
    single_task_id = int(getattr(config, "rlpd_single_task_id", 0))

    if single_task:
        return torch.full((len(tasks),), single_task_id, dtype=torch.long, device=device)

    task_to_id = getattr(config, "rlpd_task_to_id", None)
    if task_to_id is not None:
        return torch.tensor([int(task_to_id[str(t)]) for t in tasks], dtype=torch.long, device=device)

    unique_tasks = sorted(set(tasks))
    inferred = {task: idx for idx, task in enumerate(unique_tasks)}
    return torch.tensor([inferred[t] for t in tasks], dtype=torch.long, device=device)


def _infer_effective_action_dim_from_stats(action_norm_stats: dict, max_action_dim: int) -> int:
    """Infer the effective action dimension from normalization stats.

    PI0 pads actions/states to 32 dims, while many robot datasets only use the
    first N dims (e.g. 7 for single-arm, 14 for dual-arm). The padded trailing
    dims usually have zero std / zero quantile range. This helper recovers N so
    the SAC critic does not silently ignore or over-consume action dims.
    """
    if not action_norm_stats:
        return max_action_dim

    if "std" in action_norm_stats:
        values = list(action_norm_stats["std"])
        active = [i for i, v in enumerate(values[:max_action_dim]) if abs(float(v)) > 1e-8]
        if active:
            return active[-1] + 1

    if "q01" in action_norm_stats and "q99" in action_norm_stats:
        q01 = list(action_norm_stats["q01"])
        q99 = list(action_norm_stats["q99"])
        active = [
            i
            for i, (lo, hi) in enumerate(zip(q01[:max_action_dim], q99[:max_action_dim], strict=False))
            if abs(float(hi) - float(lo)) > 1e-8
        ]
        if active:
            return active[-1] + 1

    return max_action_dim


class PI0ForActionPrediction(PreTrainedModel, SupportSACTraining):
    config_class = PI0TorchConfig
    base_model_prefix = "pi0_torch"

    def __init__(self, config: PI0TorchConfig):
        super().__init__(config)
        self.model: PI0Model = None
        self.state_norm_stats = config.state_norm_stats
        self.action_norm_stats = config.action_norm_stats
        self.pi05_enabled = config.pi05_enabled

        assert self.state_norm_stats, "state_norm_stats must be provided in PI0TorchConfig"
        assert self.action_norm_stats, "action_norm_stats must be provided in PI0TorchConfig"
        assert isinstance(self.pi05_enabled, bool), "pi05_enabled must be provided in PI0TorchConfig"

        # Input transforms
        self.state_normalize_transform = Normalize(self.state_norm_stats, use_quantiles=self.pi05_enabled)
        self.action_normalize_transform = Normalize(self.action_norm_stats, use_quantiles=self.pi05_enabled)
        self.image_transform = ImageTransform(resize_imgs_with_padding=(224, 224), enable_image_aug=False)
        max_length = 200 if self.pi05_enabled else 48
        self.prompt_tokenizer_transform = PromptTokenizerTransform(max_length=max_length, discrete_state_input=False)

        # Output transforms
        self.state_unnormalize_transform = Unnormalize(self.state_norm_stats, use_quantiles=self.pi05_enabled)
        self.action_unnormalize_transform = Unnormalize(self.action_norm_stats, use_quantiles=self.pi05_enabled)

        # Stochastic flow parameters (ReinFlow-lite style)
        self._to(get_device_name())
        self.flow_sde_enable = bool(getattr(config, "flow_sde_enable", True))
        self.flow_sde_rollout_noise_scale = float(getattr(config, "flow_sde_rollout_noise_scale", 1.0))
        self.flow_sde_train_noise_scale = float(getattr(config, "flow_sde_train_noise_scale", 1.0))
        self.flow_logprob_mode = str(getattr(config, "flow_logprob_mode", "path_exact"))
        self.flow_sigma_head_hidden_dim = int(getattr(config, "flow_sigma_head_hidden_dim", 512))
        self.flow_sigma_min = float(getattr(config, "flow_sigma_min", 1e-3))
        self.flow_sigma_max = float(getattr(config, "flow_sigma_max", 5e-1))
        self.flow_sigma_init = float(getattr(config, "flow_sigma_init", 5e-2))
        self.flow_sigma_use_latent_stats = bool(getattr(config, "flow_sigma_use_latent_stats", True))
        self.register_buffer("flow_sde_step", torch.zeros((), dtype=torch.long))

        if self.flow_sde_enable:
            latent_stats_dim = 2 if self.flow_sigma_use_latent_stats else 0
            self.flow_sigma_feature_dim = 2048 + 32 + latent_stats_dim + 1
            self.flow_sigma_head = MLP(
                input_dim=self.flow_sigma_feature_dim,
                hidden_dims=[self.flow_sigma_head_hidden_dim, self.flow_sigma_head_hidden_dim // 2],
                output_dim=1,
                activation="relu",
                init_method="kaiming",
            )
            init_log_sigma = math.log(max(self.flow_sigma_init, self.flow_sigma_min))
            log_sigma_min = math.log(self.flow_sigma_min)
            log_sigma_max = math.log(self.flow_sigma_max)
            init_ratio = (init_log_sigma - log_sigma_min) / max(log_sigma_max - log_sigma_min, 1e-6)
            init_ratio = min(max(init_ratio, 1e-4), 1.0 - 1e-4)
            init_bias = math.log(init_ratio / (1.0 - init_ratio))
            self.flow_sigma_head.network[-1].bias.data.fill_(init_bias)

        ##### SAC Algorithm Support #####
        if getattr(self.config, "sac_enable", False):
            head_num = int(getattr(self.config, "critic_head_num", 10))
            attn_heads = int(getattr(self.config, "critic_prefix_attn_heads", 8))

            self.critic_action_steps = int(getattr(self.config, "critic_action_steps", getattr(self.config, "n_action_steps", 50)))
            inferred_action_dim = _infer_effective_action_dim_from_stats(
                self.action_norm_stats, int(getattr(self.config, "max_action_dim", 32))
            )
            self.critic_action_dim = int(getattr(self.config, "critic_action_dim", inferred_action_dim))
            self.critic_input_dim = 2048 + 32 + self.critic_action_steps * self.critic_action_dim

            self.critic_state_token = nn.Parameter(torch.zeros(1, 1, 2048))
            self.target_state_token = nn.Parameter(torch.zeros(1, 1, 2048))
            nn.init.normal_(self.critic_state_token, mean=0.0, std=0.02)
            self.target_state_token.data.copy_(self.critic_state_token.data)

            self.critic_prefix_cross_attn = nn.MultiheadAttention(
                embed_dim=2048,
                num_heads=attn_heads,
                batch_first=True,
            )
            self.target_prefix_cross_attn = nn.MultiheadAttention(
                embed_dim=2048,
                num_heads=attn_heads,
                batch_first=True,
            )

            self.critic_heads = nn.ModuleList(
                [
                    MLP(
                        input_dim=self.critic_input_dim,
                        hidden_dims=[2048, 1024, 256],
                        output_dim=1,
                        activation="relu",
                        init_method="kaiming",
                    )
                    for _ in range(head_num)
                ]
            )

            self.target_network_heads = nn.ModuleList(
                [
                    MLP(
                        input_dim=self.critic_input_dim,
                        hidden_dims=[2048, 1024, 256],
                        output_dim=1,
                        activation="relu",
                        init_method="kaiming",
                    )
                    for _ in range(head_num)
                ]
            )

            self.target_network_heads.load_state_dict(self.critic_heads.state_dict())
            self.target_prefix_cross_attn.load_state_dict(self.critic_prefix_cross_attn.state_dict())

    def _to(self, device: torch.device | str):
        self.state_normalize_transform.to(device)
        self.state_unnormalize_transform.to(device)
        self.action_normalize_transform.to(device)
        self.action_unnormalize_transform.to(device)
        return self

    def forward(
        self,
        images: list[torch.Tensor],
        img_masks: list[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        state: torch.Tensor,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
    ) -> Tensor:
        """Full forward pass for one diffusion denoising step.

        Args:
            images: List of image tensors, each shaped (B, C, H, W) after batching.
            img_masks: List of boolean masks corresponding to images, each (B,).
            lang_tokens: Language token ids (B, L).
            lang_masks: Language attention mask (B, L) with True for valid tokens.
            state: State tensor (B, state_dim) if pi05 is disabled else ignored.
            x_t: Noisy action tokens (B, n_action_steps, action_dim).
            timestep: Diffusion timestep as float tensor (B,).

        Returns:
            Predicted v_t with shape (B, n_action_steps, action_dim).
        """

        if self.model is None:
            raise RuntimeError("PI0ForActionPrediction.model is not initialized. Did from_pretrained() run?")

        return self.model(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            x_t,
            timestep,
        )

    @torch.no_grad()
    def sample_actions(
        self,
        env_obs: DataProto,
        tokenizer,
    ) -> tuple[Pi0Output, dict, dict]:
        """Run one forward pass from raw inputs to final action sequence.

        Args:
            env_obs: The environment observations as DataProto.
            tokenizer: The tokenizer used for prompt tokenization.

        Returns:
            A tuple of (pi0_output, s, a):
                - pi0_output: The Pi0Output containing the predicted actions.
                - s: Dictionary of tensors representing the states, with keys
                    - "images": torch.Tensor of shape (B, n_images, C, H, W)
                    - "image_masks": torch.Tensor of shape (B, n_images)
                    - "lang_tokens": torch.Tensor of shape (B, L)
                    - "lang_masks": torch.Tensor of shape (B, L)
                    - "states": torch.Tensor of shape (B, state_dim)
                - a: Dictionary of tensors representing actions, with key:
                    - "full_action": torch.Tensor of shape (B, action_steps, action_dim)
        """

        from .policy.libero_policy import LiberoPi0Input

        pi0_input = LiberoPi0Input.from_env_obs(env_obs)

        # Input transforms
        state = self.state_normalize_transform(pi0_input.state)
        images, _ = self.image_transform.call_batch(pi0_input.images)
        lang_tokens, lang_masks = self.prompt_tokenizer_transform.call_batch(
            {"task": pi0_input.task, "observation.state": state}, tokenizer
        )

        if self.flow_sde_enable:
            prefix_features = self.model.embed_prefix(
                images=images,
                img_masks=pi0_input.img_masks,
                lang_tokens=lang_tokens,
                lang_masks=lang_masks,
            )
            pred_action, rollout_log_probs, _ = self._sample_actions_flow_sde(
                state_features=(prefix_features, state),
                noise_scale=self.flow_sde_rollout_noise_scale,
                requires_grad=False,
                return_log_prob=True,
            )
        else:
            pred_action = self.model.sample_actions(images, pi0_input.img_masks, lang_tokens, lang_masks, state=state)
            rollout_log_probs = torch.zeros(pred_action.shape[0], device=pred_action.device, dtype=torch.float32)

        # Output transforms
        from .policy.libero_policy import LiberoPi0Output

        pi0_output = LiberoPi0Output.from_model_output({
            "full_action": self.action_unnormalize_transform(pred_action)
        })
        s = {
            "states": state,
            "images": torch.stack(images, dim=1),
            "image_masks": torch.stack(pi0_input.img_masks, dim=1),
            "lang_tokens": lang_tokens,
            "lang_masks": lang_masks,
        }
        a = {
            "full_action": pred_action,
            "log_probs": rollout_log_probs,
        }

        return pi0_output, s, a

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop("config", None)

        if config is None:
            config = PI0TorchConfig.from_pretrained(pretrained_model_name_or_path)

        policy = cls(config)
        policy.model = PI0Model.from_pretrained(pretrained_model_name_or_path)
        return policy

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        filtered_state_dict = {
            key: value
            for key, value in state_dict.items()
            if key.startswith("model.")
        }
        return super().load_state_dict(filtered_state_dict, strict=False, assign=assign)

    def freeze_vision_tower(self) -> None:
        """Freeze the vision tower parameters."""

        if self.model is None:
            raise RuntimeError("PI0ForActionPrediction.model is not initialized. Did from_pretrained() run?")
        vision_tower = self.model.paligemma_with_expert.vision_tower
        vision_tower.requires_grad_(False)
        vision_tower.eval()
    
    def bc_loss(
        self,
        state_features: tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        actions: dict[str, torch.Tensor],
        valids: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate the BC loss for the actor."""

        prefix_features, states = state_features
        _, prefix_pad_masks, _ = prefix_features
        action_tensor = actions["full_action"]

        batch_size = action_tensor.shape[0]
        device = action_tensor.device

        noise = self.model.sample_noise(action_tensor.shape, device=device)
        gamma1 = torch.empty((batch_size,), device=device).uniform_(0, 1).pow(1 / 1.5)
        gamma2 = torch.empty((batch_size,), device=device).uniform_(0, 1).pow(1 / 1.0)
        time = (gamma1 / (gamma1 + gamma2)) * 0.999 + 0.001
        time = time.to(dtype=torch.float32, device=device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1.0 - time_expanded) * action_tensor
        u_t = noise - action_tensor

        past_key_values = self._build_kv_cache_from_prefix(prefix_features)
        model_pred = self.model.denoise_step(
            states,
            prefix_pad_masks,
            past_key_values,
            x_t,
            time,
        )

        loss = torch.nn.functional.mse_loss(u_t, model_pred, reduction="none").mean(dim=-1).mean(dim=-1)
        valid_f = valids.float().to(loss.device)
        return (loss * valid_f).sum() / valid_f.sum().clamp_min(1.0)

    def process_dataset_batch(
        self, 
        dataset_batch: DataProto,
        tokenizer,
    ) -> dict[str, torch.Tensor]:

        if self.config.dataset_type == "lerobot":
            from .datasets.lerobot_dataset import LeRobotPi0DatasetInput
            batch = LeRobotPi0DatasetInput.from_dataset_batch(dataset_batch)
        elif self.config.dataset_type == "libero":
            from .datasets.libero_dataset import LiberoPi0DatasetInput
            batch = LiberoPi0DatasetInput.from_dataset_batch(dataset_batch)
        else:
            raise ValueError(f"Unknown dataset_type: {self.config.dataset_type}")

        out = {}

        # Process images
        # s0
        images, _ = self.image_transform.call_batch(batch.s0['images'])
        out['s0.images'] = torch.stack(images, dim=1)
        out['s0.image_masks'] = torch.stack(batch.s0['img_masks'], dim=1)
        # s1
        images, _ = self.image_transform.call_batch(batch.s1['images'])
        out['s1.images'] = torch.stack(images, dim=1)
        out['s1.image_masks'] = torch.stack(batch.s1['img_masks'], dim=1)

        # Process language
        # s0
        lang_tokens, lang_masks = self.prompt_tokenizer_transform.call_batch(
            {'task': batch.s0['task'], 'observation.state': batch.s0['state']},
            tokenizer=tokenizer
        )
        out['s0.lang_tokens'] = lang_tokens
        out['s0.lang_masks'] = lang_masks

        # s1
        lang_tokens, lang_masks = self.prompt_tokenizer_transform.call_batch(
            {'task': batch.s1['task'], 'observation.state': batch.s1['state']},
            tokenizer=tokenizer
        )
        out['s1.lang_tokens'] = lang_tokens
        out['s1.lang_masks'] = lang_masks

        # Process states
        out['s0.states'] = self.state_normalize_transform(batch.s0['state'])
        out['s1.states'] = self.state_normalize_transform(batch.s1['state'])

        # Process actions
        out['a0.full_action'] = self.action_normalize_transform(batch.a0['action'])
        out['a1.full_action'] = self.action_normalize_transform(batch.a1['action'])

        # Process other information
        out['rewards'] = batch.rewards.to(out['s0.states'].device)
        out['valids'] = batch.valids.to(out['s0.states'].device)
        out['dones'] = batch.dones.to(out['s0.states'].device)
        out['positive_sample_mask'] = batch.positive_sample_mask.to(out['s0.states'].device)
        out['task_ids'] = _resolve_rlpd_task_ids(
            tasks=batch.s0['task'],
            config=self.config,
            device=out['s0.states'].device,
        )

        return out

    # --- SAC Algorithm Support ---

    def _multi_heads_value(
        self, value_heads: nn.ModuleList, input_tensor: torch.Tensor, method: Literal["cat", "min"] = "cat"
    ) -> torch.Tensor:
        q_values = [head(input_tensor) for head in value_heads]
        if method == "cat":
            q_values = torch.cat(q_values, dim=-1)
        elif method == "min":
            q_values = torch.min(torch.cat(q_values, dim=-1), dim=-1).values
        else:
            raise ValueError(f"Unknown method: {method}")

        return q_values

    def _cross_attention_pool_prefix(
        self,
        prefix_embs: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        use_target_network: bool,
    ) -> torch.Tensor:
        cross_attn = self.target_prefix_cross_attn if use_target_network else self.critic_prefix_cross_attn
        state_token = self.target_state_token if use_target_network else self.critic_state_token

        batch_size = prefix_embs.shape[0]
        query = state_token.expand(batch_size, -1, -1)
        key_padding_mask = ~prefix_pad_masks.to(dtype=torch.bool)

        pooled, _ = cross_attn(
            query=query,
            key=prefix_embs,
            value=prefix_embs,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return pooled.squeeze(1)

    def _diag_gaussian_log_prob_sum(
        self,
        sample: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> torch.Tensor:
        std_safe = std.clamp_min(1e-6)
        log_prob = -0.5 * (((sample - mean) / std_safe) ** 2 + 2.0 * torch.log(std_safe) + math.log(2.0 * math.pi))
        return log_prob.sum(dim=(-1, -2))

    def _diag_gaussian_entropy_sum(self, std: torch.Tensor) -> torch.Tensor:
        std_safe = std.clamp_min(1e-6)
        entropy = 0.5 * (1.0 + math.log(2.0 * math.pi)) + torch.log(std_safe)
        return entropy.sum(dim=(-1, -2))

    def _masked_mean_prefix(self, prefix_embs: torch.Tensor, prefix_pad_masks: torch.Tensor) -> torch.Tensor:
        weights = prefix_pad_masks.to(dtype=prefix_embs.dtype).unsqueeze(-1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return (prefix_embs * weights).sum(dim=1) / denom

    def _predict_flow_sigma(
        self,
        prefix_context: torch.Tensor,
        states: torch.Tensor,
        x_t: torch.Tensor,
        t_cur: torch.Tensor,
        noise_scale: float,
    ) -> torch.Tensor:
        if noise_scale <= 0:
            return torch.zeros((x_t.shape[0], 1, 1), device=x_t.device, dtype=x_t.dtype)

        features = [prefix_context.to(dtype=x_t.dtype), states.to(dtype=x_t.dtype), t_cur.expand(x_t.shape[0], 1).to(dtype=x_t.dtype)]
        if self.flow_sigma_use_latent_stats:
            latent_mean = x_t.mean(dim=(-1, -2), keepdim=False).unsqueeze(-1)
            latent_std = x_t.std(dim=(-1, -2), keepdim=False, unbiased=False).unsqueeze(-1)
            features.extend([latent_mean.to(dtype=x_t.dtype), latent_std.to(dtype=x_t.dtype)])
        sigma_features = torch.cat(features, dim=-1)
        sigma_gate = torch.sigmoid(self.flow_sigma_head(sigma_features))
        log_sigma_min = math.log(self.flow_sigma_min)
        log_sigma_max = math.log(self.flow_sigma_max)
        log_sigma = log_sigma_min + (log_sigma_max - log_sigma_min) * sigma_gate
        sigma = torch.exp(log_sigma) * float(noise_scale)
        return sigma.unsqueeze(-1)

    def _sample_actions_flow_sde(
        self,
        state_features: tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        noise_scale: float,
        requires_grad: bool,
        return_log_prob: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, float]]:
        prefix_features, states = state_features
        prefix_embs, prefix_pad_masks, _ = prefix_features
        batch_size = prefix_embs.shape[0]
        device = prefix_embs.device

        past_key_values = self._build_kv_cache_from_prefix(prefix_features)
        prefix_context = self._masked_mean_prefix(prefix_embs, prefix_pad_masks)

        actions_shape = (batch_size, self.model.n_action_steps, self.model.max_action_dim)
        x_t = torch.randn(actions_shape, device=device, dtype=prefix_embs.dtype)

        timesteps = torch.linspace(1.0, 0.0, self.model.num_steps + 1, dtype=torch.float32, device=device)
        step_log_probs: list[torch.Tensor] = []
        step_entropies: list[torch.Tensor] = []
        step_sigma_means: list[torch.Tensor] = []
        step_sigma_stds: list[torch.Tensor] = []

        for idx in range(self.model.num_steps):
            t_cur = timesteps[idx]
            t_next = timesteps[idx + 1]
            delta = (t_cur - t_next).clamp_min(1e-6)

            if requires_grad:
                v_t = self.model.denoise_step(
                    states,
                    prefix_pad_masks,
                    past_key_values,
                    x_t,
                    t_cur.expand(batch_size),
                )
            else:
                with torch.no_grad():
                    v_t = self.model.denoise_step(
                        states,
                        prefix_pad_masks,
                        past_key_values,
                        x_t,
                        t_cur.expand(batch_size),
                    )

            t_cur_safe = t_cur.clamp(min=1e-4, max=1.0 - 1e-4)
            t_cur_exp = t_cur_safe.view(1, 1, 1)
            t_next_exp = t_next.view(1, 1, 1)

            x0_pred = x_t - v_t * t_cur_exp
            x1_pred = x_t + v_t * (1.0 - t_cur_exp)
            x_mean = x0_pred * (1.0 - t_next_exp) + x1_pred * t_next_exp

            sigma_scalar = self._predict_flow_sigma(
                prefix_context=prefix_context,
                states=states,
                x_t=x_t,
                t_cur=t_cur.expand(batch_size, 1),
                noise_scale=noise_scale,
            )
            sigma_t = sigma_scalar * torch.sqrt(delta).view(1, 1, 1)

            if noise_scale > 0:
                eps = torch.randn_like(x_t)
                x_prev = x_mean + sigma_t * eps
            else:
                x_prev = x_mean

            if return_log_prob:
                step_log_probs.append(self._diag_gaussian_log_prob_sum(x_prev, x_mean, sigma_t))
                step_entropies.append(self._diag_gaussian_entropy_sum(sigma_t))

            step_sigma_means.append(sigma_t.mean())
            step_sigma_stds.append(sigma_t.std(unbiased=False))
            x_t = x_prev

        if return_log_prob and self.flow_logprob_mode == "path_exact":
            log_probs = torch.stack(step_log_probs, dim=1).sum(dim=1)
            path_entropy = torch.stack(step_entropies, dim=1).sum(dim=1)
        elif return_log_prob:
            log_probs = None
            path_entropy = None
        else:
            log_probs = None
            path_entropy = None

        metrics = {
            "flow_sigma_mean": float(torch.stack(step_sigma_means).mean().item()),
            "flow_sigma_std": float(torch.stack(step_sigma_stds).mean().item()),
            "flow_logprob_mode": 1.0 if self.flow_logprob_mode == "path_exact" else 0.0,
        }
        if path_entropy is not None:
            metrics["flow_path_entropy_mean"] = float(path_entropy.mean().item())
        if log_probs is not None:
            metrics["flow_path_logprob_mean"] = float(log_probs.mean().item())

        return x_t, log_probs, metrics

    def _build_kv_cache_from_prefix(
        self,
        prefix_features: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ):
        """Build KV cache for prefix. No grad needed."""
        prefix_embs, prefix_pad_masks, prefix_att_masks = prefix_features
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        with torch.no_grad():
            _, past_key_values = self.model.paligemma_with_expert.forward(
                attention_mask=prefix_att_2d_masks,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=self.model.use_cache,
                fill_kv_cache=True,
                adarms_cond=[None, None],
            )
        return past_key_values

    @override
    def sac_init(self):
        """Initialize SAC-related components."""

        self.freeze_vision_tower()

        register_fsdp_forward_method(self, "bc_loss")
        register_fsdp_forward_method(self, "sac_forward_critic")
        register_fsdp_forward_method(self, "sac_forward_actor")
        register_fsdp_forward_method(self, "sac_update_target_network")
        register_fsdp_forward_method(self, "sac_forward_state_features")

    @override
    def sac_forward_actor(
        self,
        state_features: tuple[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            torch.Tensor,
        ],
        is_first_micro_batch: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, float]]:
        actions, log_probs, actor_metrics = self._sample_actions_flow_sde(
            state_features=state_features,
            noise_scale=self.flow_sde_train_noise_scale,
            requires_grad=True,
            return_log_prob=True,
        )
        if is_first_micro_batch:
            self.flow_sde_step.add_(1)
        if self.flow_sde_enable:
            actor_metrics.update({
                "flow_sde_step": float(self.flow_sde_step.item()),
            })
        return actions, log_probs, actor_metrics

    @override
    def sac_forward_critic(
        self,
        a: dict[str, torch.Tensor],
        state_features: tuple[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            torch.Tensor,
        ],
        *,
        use_target_network: bool = False,
        method: Literal["cat", "min"] = "cat",
        requires_grad: bool = False,
    ):
        critic_head = self.target_network_heads if use_target_network else self.critic_heads
        for p in critic_head.parameters():
            p.requires_grad_(requires_grad)
        prefix_cross_attn = self.target_prefix_cross_attn if use_target_network else self.critic_prefix_cross_attn
        for p in prefix_cross_attn.parameters():
            p.requires_grad_(requires_grad)
        if use_target_network:
            self.target_state_token.requires_grad_(requires_grad)
        else:
            self.critic_state_token.requires_grad_(requires_grad)

        prefix_features, states = state_features
        prefix_embs, prefix_pad_masks, _ = prefix_features
        pooled_prefix_embs = self._cross_attention_pool_prefix(
            prefix_embs=prefix_embs,
            prefix_pad_masks=prefix_pad_masks,
            use_target_network=use_target_network,
        )  # (B, 2048)
        actions = a["full_action"][:, : self.critic_action_steps, : self.critic_action_dim]
        flattened_actions = actions.reshape(actions.shape[0], -1)
        critic_input = torch.cat([pooled_prefix_embs, states, flattened_actions], dim=-1)

        q_values = self._multi_heads_value(critic_head, critic_input, method=method)

        return q_values
    
    @override
    def sac_get_critic_parameters(self) -> list[torch.nn.Parameter]:
        critic_head_params = [p for head in self.critic_heads for p in head.parameters()]
        critic_prefix_cross_attn_params = list(self.critic_prefix_cross_attn.parameters())
        return critic_head_params + critic_prefix_cross_attn_params + [self.critic_state_token]

    @override
    def sac_get_named_actor_parameters(self) -> list[tuple[str, torch.nn.Parameter]]:
        named_parameters = [
            (name, param)
            for name, param in self.model.named_parameters()
            if param.requires_grad
        ]
        return named_parameters

    @override
    def sac_forward_state_features(
        self, s: dict[str, torch.Tensor]
    ) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        with torch.no_grad():
            prefix_features = self.model.embed_prefix(
                images=s["images"].unbind(dim=1),
                img_masks=s["image_masks"].unbind(dim=1),
                lang_tokens=s["lang_tokens"],
                lang_masks=s["lang_masks"],
            )
        return (prefix_features, s["states"])

    @override
    @torch.no_grad()
    def sac_update_target_network(self, tau: float):
        for t_head, head in zip(self.target_network_heads, self.critic_heads, strict=True):
            t_sd = t_head.state_dict()
            h_sd = head.state_dict()
            for k in t_sd.keys():
                t_sd[k].mul_(1.0 - tau).add_(h_sd[k], alpha=tau)
            t_head.load_state_dict(t_sd, strict=True)

        t_cross_attn_sd = self.target_prefix_cross_attn.state_dict()
        cross_attn_sd = self.critic_prefix_cross_attn.state_dict()
        for k in t_cross_attn_sd.keys():
            t_cross_attn_sd[k].mul_(1.0 - tau).add_(cross_attn_sd[k], alpha=tau)
        self.target_prefix_cross_attn.load_state_dict(t_cross_attn_sd, strict=True)

        self.target_state_token.data.mul_(1.0 - tau).add_(self.critic_state_token.data, alpha=tau)

    