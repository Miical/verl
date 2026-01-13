from __future__ import annotations

from onnx_ir import Tensor
from typing import Literal
from typing_extensions import override

import torch
from torch import nn
from torch.distributions import Normal
from torch.distributed.fsdp import register_fsdp_forward_method
from transformers import PreTrainedModel
from verl.utils.device import get_device_name

from .configuration_pi0_torch import PI0TorchConfig
from .model.modeling_pi0 import PI0Model, make_att_2d_masks
from .pi0_utils import (
    ImageTransform,
    Normalize,
    PromptTokenizerTransform,
    Unnormalize,
)
from ..modules.mlp import MLP
from ...sac.base import SupportSACTraining

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
        self.image_transform = ImageTransform(resize_imgs_with_padding=(224, 224), enable_image_aug=False)
        max_length = 200 if self.pi05_enabled else 48
        self.prompt_tokenizer_transform = PromptTokenizerTransform(max_length=max_length, discrete_state_input=False)

        # Output transforms
        self.state_unnormalize_transform = Unnormalize(self.state_norm_stats, use_quantiles=self.pi05_enabled)
        self.action_unnormalize_transform = Unnormalize(self.action_norm_stats, use_quantiles=self.pi05_enabled)

        self._to(get_device_name())

        ##### SAC Algorithm Support #####
        if getattr(self.config, "sac_enable", False):
            head_num = 2 if getattr(self.config, "double_q", True) else 1

            self.critic_heads = nn.ModuleList([
                MLP(
                    input_dim=3680,  # config: 2048(prefix mean) + 32(state) + 50*32(action flat)
                    hidden_dims=[256, 256],
                    output_dim=1,
                    activation='relu',
                    init_method='kaiming'
                )
                for _ in range(head_num)
            ])

            self.target_network_heads = nn.ModuleList([
                MLP(
                    input_dim=3680, # config
                    hidden_dims=[256, 256],
                    output_dim=1,
                    activation='relu',
                    init_method='kaiming'
                )
                for _ in range(head_num)
            ])

    def _to(self, device: torch.device | str):
        self.state_normalize_transform.to(device)
        self.state_unnormalize_transform.to(device)
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

    def dummy_forward(self) -> None:
        """Run a dummy forward pass to initialize fsdp sharding."""

        device = get_device_name()
        with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.bfloat16):
            _ = self(
                images=[torch.zeros((1, 3, 224, 224), device=device, dtype=torch.float32)],
                img_masks=[torch.ones((1,), device=device, dtype=torch.bool)],
                lang_tokens=torch.zeros((1, 1), device=device, dtype=torch.long),
                lang_masks=torch.ones((1, 1), device=device, dtype=torch.bool),
                state=torch.zeros((1, 32), device=device, dtype=torch.float32),
                x_t=self.model.sample_noise((1, 50, 32), device=device),
                timestep=torch.full((1,), 0.5, device=device, dtype=torch.float32),
            )

    @torch.no_grad()
    def sample_actions(
        self,
        images: dict[str, torch.Tensor],
        img_masks: list[torch.Tensor],
        task: list[str],
        state: torch.Tensor,
        tokenizer,
    ) -> torch.Tensor:
        """Run one forward pass from raw inputs to final action sequence.

        Args:
            images: Observation images of the robot. Each value is a tensor with shape (B,C,H,W).
            img_masks: A list of image masks corresponding to the images dict, each with shape (B,).
            task: A list of natural language task descriptions.
            state: The robot joint state tensor with shape (B, state_dim).
            tokenizer: The tokenizer used for prompt tokenization.

        Returns:
            A tensor of predicted actions with shape (batch, num_steps, original_action_dim) on the original input device.
        """

        # Input transforms
        state = self.state_normalize_transform(state)
        images, _ = self.image_transform.call_batch(images)
        lang_tokens, lang_masks = self.prompt_tokenizer_transform.call_batch(
            { 'task': task, 'observation.state': state },
            tokenizer
        )

        # Inference
        pred_action = self.model.sample_actions(images, img_masks, lang_tokens, lang_masks, state=state)

        # Output transforms
        state = self.state_unnormalize_transform(state)
        pred_action = self.action_unnormalize_transform(pred_action)

        return pred_action, images, img_masks, lang_tokens, lang_masks, state

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop("config", None)

        if config is None:
            config = PI0TorchConfig.from_pretrained(pretrained_model_name_or_path)

        policy = cls(config)
        policy.model = PI0Model.from_pretrained(pretrained_model_name_or_path)
        return policy

    def freeze_vision_tower(self) -> None:
        """Freeze the vision tower parameters."""

        if self.model is None:
            raise RuntimeError("PI0ForActionPrediction.model is not initialized. Did from_pretrained() run?")
        vision_tower = self.model.paligemma_with_expert.vision_tower
        vision_tower.requires_grad_(False)
        vision_tower.eval()

    # --- SAC Algorithm Support ---

    def _multi_heads_value(
        self,
        value_heads: nn.ModuleList,
        input_tensor: torch.Tensor,
        method: Literal["cat", "min"] = "cat"
    ) -> torch.Tensor:

        q_values = [head(input_tensor) for head in value_heads]
        if method == "cat":
            q_values = torch.cat(q_values, dim=-1)
        elif method == "min":
            q_values = torch.min(torch.cat(q_values, dim=-1), dim=-1).values
        else:
            raise ValueError(f"Unknown method: {method}")

        return q_values

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

    def _get_logprobs(
        self,
        s: dict[str, torch.Tensor],
        prefix_features: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        mode: Literal["train", "eval"] = "train",
        *,
        # 推荐 SAC 用：直接传入“同一次采样的最后一步 transition”
        x_t: torch.Tensor | None = None,     # (B, T, A)
        x_next: torch.Tensor | None = None,  # (B, T, A)
        v_t: torch.Tensor | None = None,     # (B, T, A)
        t: torch.Tensor | None = None,       # (B,)
        step_idx: torch.Tensor | None = None # (B,)
    ) -> torch.Tensor:
        """
        Approx log π via one-step Gaussian approximation.

        - SAC 正确用法：传入同一次采样里最后一步 (x_{K-1}, x_K, v_{K-1}, t_{K-1})
          这样 logπ 至少和最终 action 对齐。
        - 如果没传 transition，则会 fallback 自己采样一遍（不推荐用于 SAC 训练）。
        """
        assert self.model is not None
        prefix_embs, prefix_pad_masks, prefix_att_masks = prefix_features
        states = s["states"]
        B = prefix_embs.shape[0]
        device = prefix_embs.device

        past_key_values = self._build_kv_cache_from_prefix(prefix_features)

        # fallback：没给 transition 就自己跑一遍，并用最后一步
        if x_t is None or x_next is None or v_t is None or t is None:
            actions_shape = (B, self.model.n_action_steps, self.model.max_action_dim)
            x = self.model.sample_noise(actions_shape, device=device)

            dt = -1.0 / float(self.model.num_steps)
            t_grid = torch.arange(1.0, -dt / 2, dt, dtype=torch.float32, device=device)

            x_prev, v_prev, t_prev = None, None, None
            for tt in t_grid:
                x_prev = x
                t_prev = tt
                v_prev = self.model.denoise_step(
                    states,
                    prefix_pad_masks,
                    past_key_values,
                    x,
                    tt.expand(B),
                )
                x = x + dt * v_prev

            x_t = x_prev
            x_next = x
            v_t = v_prev
            t = t_prev.expand(B)

        # sigma schedule step index：默认用最后一步 K-1
        K = int(self.model.num_steps)
        if step_idx is None:
            step_idx = torch.full((B,), K - 1, device=device, dtype=torch.long)
            if mode == "eval":
                # eval 下也用最后一步；你想固定 0 也可以
                pass

        # one-step mean/std
        dt_pos = 1.0 / float(K)
        t_b = t[:, None, None]  # (B,1,1)
        dt_b = torch.full_like(t_b, dt_pos)

        x0_pred = x_t - v_t * t_b
        x1_pred = x_t + v_t * (1.0 - t_b)

        # heuristic sigma schedule (ported family)
        noise_level = 0.5
        t_grid_full = torch.arange(1.0, -dt_pos / 2, -dt_pos, dtype=torch.float32, device=device)  # len=K+1
        t_for_sigma = torch.where(t_grid_full == 1.0, t_grid_full[1], t_grid_full)
        sigmas = noise_level * torch.sqrt(t_grid_full / (1.0 - t_for_sigma).clamp_min(1e-6))
        sigmas = sigmas[:-1]  # len=K

        sigma_i = sigmas[step_idx][:, None, None].clamp_min(1e-6)  # (B,1,1)

        x0_weight = torch.ones_like(t_b) - (t_b - dt_b)
        x1_weight = t_b - dt_b - (sigma_i ** 2) * dt_b / (2.0 * t_b.clamp_min(1e-6))

        x_next_mean = x0_pred * x0_weight + x1_pred * x1_weight
        x_next_std = (dt_b.sqrt() * sigma_i).clamp_min(1e-6)

        dist = Normal(x_next_mean.float(), x_next_std.float())
        log_probs = dist.log_prob(x_next.float()).sum(dim=(1, 2))  # (B,)
        return log_probs

    def _sample_actions_and_logprobs_from_prefix(
        self,
        states: torch.Tensor,
        prefix_features: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample actions amd compute logprob aligned with those sampled actions.

        Args:
            states: (B, state_dim)
            prefix_features: tuple of (prefix_embs, prefix_pad_masks, prefix_att_masks)

        Returns:
            actions: (B, n_action_steps, action_dim)
            log_probs: (B,)
        """

        prefix_embs, prefix_pad_masks, _ = prefix_features
        B = prefix_embs.shape[0]
        device = prefix_embs.device

        past_key_values = self._build_kv_cache_from_prefix(prefix_features)

        actions_shape = (B, self.model.n_action_steps, self.model.max_action_dim)
        x = self.model.sample_noise(actions_shape, device=device)

        dt = -1.0 / float(self.model.num_steps)
        t_grid = torch.arange(1.0, -dt / 2, dt, dtype=torch.float32, device=device)  # len=K

        x_prev, v_prev, t_prev = None, None, None
        for tt in t_grid:
            x_prev = x
            t_prev = tt
            v_prev = self.model.denoise_step(
                states,
                prefix_pad_masks,
                past_key_values,
                x,
                tt.expand(B),
            )
            x = x + dt * v_prev

        actions = x  # x_K

        # aligned logprob: use last transition (K-1)
        step_idx = torch.full((B,), int(self.model.num_steps) - 1, device=device, dtype=torch.long)
        log_probs = self._get_logprobs(
            {"states": states},
            prefix_features,
            x_t=x_prev,
            x_next=actions,
            v_t=v_prev,
            t=t_prev.expand(B),
            step_idx=step_idx,
        )

        return actions, log_probs

    @override
    def sac_init(self):
        """Initialize SAC-related components."""

        self.freeze_vision_tower()

        register_fsdp_forward_method(self, "sac_forward_critic")
        register_fsdp_forward_method(self, "sac_forward_actor")
        register_fsdp_forward_method(self, "sac_update_target_network")

    @override
    def sac_forward_critic(
        self,
        s0: dict[str, torch.Tensor],
        a0: dict[str, torch.Tensor],
        s1: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute Q-values for given state-action pairs using the critic heads.

        Args:
            s0: Dictionary of tensors representing the initial states, with keys:
                - "images": torch.Tensor of shape (B, n_images, C, H, W)
                - "image_masks": torch.Tensor of shape (B, n_images)
                - "lang_tokens": torch.Tensor of shape (B, L)
                - "lang_masks": torch.Tensor of shape (B, L)
                - "states": torch.Tensor of shape (B, state_dim)
            a0: Dictionary of tensors representing the initial actions, with key:
                - "full_action": torch.Tensor of shape (B, n_action_steps, action_dim)
            s1: Dictionary of tensors representing the next states, with same keys as s0.

        Returns:
            q_values_0: torch.Tensor of shape (B, head_num), Q values for (s0, a0), computed by critic heads.
            q_values_1: torch.Tensor of shape (B,), Q values for (s1, a1), computed by target network.
            log_probs_1: torch.Tensor of shape (B,), log probabilities of actions a1 under the current policy.
            shared_features: tuple containing shared features computed by the critic, it will be used in
                             sac_forward_actor.
        """

        # Prepare prefix features for s0 and s1
        with torch.no_grad():
            prefix_features = self.model.embed_prefix(
                images=torch.cat([s0["images"], s1["images"]], dim=0).unbind(dim=1),
                img_masks=torch.cat([s0["image_masks"], s1["image_masks"]], dim=0).unbind(dim=1),
                lang_tokens=torch.cat([s0["lang_tokens"], s1["lang_tokens"]], dim=0),
                lang_masks=torch.cat([s0["lang_masks"], s1["lang_masks"]], dim=0),
            )
        prefix_features_0 = tuple(feature_chunk.chunk(2, dim=0)[0] for feature_chunk in prefix_features)
        prefix_features_1 = tuple(feature_chunk.chunk(2, dim=0)[1] for feature_chunk in prefix_features)
        prefix_embs, _, _ = prefix_features                        # (2B, 968, 2048)
        states = torch.cat([s0["states"], s1["states"]], dim=0)    # (2B, 32)

        # Resample actions for s1
        with torch.no_grad():
            actions_0 = a0["full_action"]
            actions_1, log_probs_1 = self._sample_actions_and_logprobs_from_prefix(
                states= s1["states"],
                prefix_features=prefix_features_1
            )
        actions = torch.cat([actions_0, actions_1], dim=0)         # (2B, 50, 32)

        # Enable grad for critic heads
        for p in self.critic_heads.parameters():
            p.requires_grad_(True)

        # Compute Q-values
        mean_prefix_embs = prefix_embs.mean(dim=1, keepdim=False)                        # (2B, 2048)
        flattened_actions = actions.view(actions.size(0), -1)                            # (2B, 50*32)
        critic_input = torch.cat([mean_prefix_embs, states, flattened_actions], dim=-1)  # (2B, 3680)
        critic_input_0, critic_input_1 = torch.chunk(critic_input, 2, dim=0)             # (B,  3680)

        q_values_0 = self._multi_heads_value(self.critic_heads, critic_input_0, method="cat")
        with torch.no_grad():
            q_values_1 = self._multi_heads_value(self.target_network_heads, critic_input_1, method="min")

        return (q_values_0, q_values_1, log_probs_1, (prefix_features_0,))

    @override
    def sac_forward_actor(
        self,
        s0: dict[str, torch.Tensor],
        shared_features: tuple[torch.Tensor, tuple],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute log probabilities and Q-values for actions sampled by the actor.

        Args:
            s0: Dictionary of tensors representing the initial states, with keys:
                - "images": torch.Tensor of shape (B, n_images, C, H, W)
                - "image_masks": torch.Tensor of shape (B, n_images)
                - "lang_tokens": torch.Tensor of shape (B, L)
                - "lang_masks": torch.Tensor of shape (B, L)
                - "states": torch.Tensor of shape (B, state_dim)
            shared_features: tuple containing shared features computed by the critic in sac_forward_critic.

        Returns:
            log_probs: torch.Tensor of shape (B, 1), log probabilities of the actions sampled by the actor.
            q_values_0: torch.Tensor of shape (B, 1), Q values for (s0, a0) computed using the critic heads.
        """

        for p in self.critic_heads.parameters():
            p.requires_grad_(False)

        prefix_features = shared_features[0]
        prefix_embs, _, _ = prefix_features
        actions_pi, log_probs = self._sample_actions_and_logprobs_from_prefix(s0["states"], prefix_features)

        mean_prefix_embs = prefix_embs.mean(dim=1)
        flat_actions = actions_pi.reshape(actions_pi.size(0), -1)
        critic_input = torch.cat([mean_prefix_embs, s0["states"], flat_actions], dim=-1)
        q_values_0 = self._multi_heads_value(self.critic_heads, critic_input, method="min")

        return log_probs, q_values_0

    @override
    @torch.no_grad()
    def sac_update_target_network(self, tau: float):
        """Update the target network heads using Polyak averaging.

        Args:
            tau: The interpolation parameter for Polyak averaging.
        """

        for target_head, head in zip(self.target_network_heads, self.critic_heads):
            for target_param, param in zip(target_head.parameters(), head.parameters()):
                target_param.data.mul_(1.0 - tau).add_(param.data, alpha=tau)
