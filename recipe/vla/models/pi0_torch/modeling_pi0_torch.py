from __future__ import annotations

from onnx_ir import Tensor
from typing_extensions import override

import torch
from torch import nn
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
                    input_dim=3680, # config
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

    def _multi_heads_min(self, value_heads: nn.ModuleList, input_tensor: torch.Tensor) -> torch.Tensor:
        q_values = [head(input_tensor) for head in value_heads]
        q_values_min = torch.min(torch.cat(q_values, dim=-1), dim=-1).values
        return q_values_min

    def _get_logprobs(
        self,
        s: dict[str, torch.Tensor],
        prefix_features: tuple[torch.Tensor]
    ) -> torch.Tensor:
        prefix_embs, prefix_pad_masks, prefix_att_masks = prefix_features
        batch_size = prefix_embs.shape[0]
        device = prefix_embs.device

        actions_shape = (batch_size, self.model.n_action_steps, self.model.max_action_dim)
        noise = self.model.sample_noise(actions_shape, device)

        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        _, past_key_values = self.model.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.model.use_cache,
            fill_kv_cache=True,
            adarms_cond=[None, None],
        )

        x_t = noise
        dt = -1.0 / self.model.num_steps
        timesteps = torch.arange(1.0, -dt / 2, dt, dtype=torch.float32, device=device)
        for timestep in timesteps:
            v_t = self.model.denoise_step(
                s["states"],
                prefix_pad_masks,
                past_key_values,
                x_t,
                timestep.expand(batch_size),
            )
            x_t += dt * v_t

        actions = x_t
        log_probs = actions.mean(dim=(1, 2)) # TODO: compute log probs properly (flow noise/flow sde)

        return log_probs

    @override
    def sac_init(self):
        """Initialize SAC-related components."""
        pass

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
        a1: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute Q-values for given state-action pairs using the critic heads.

        Args:
            s0: Dictionary of tensors representing the initial states, with keys:
                - "images": torch.Tensor of shape (B, n_images, C, H, W)
                - "img_masks": torch.Tensor of shape (B, n_images)
                - "lang_tokens": torch.Tensor of shape (B, L)
                - "lang_masks": torch.Tensor of shape (B, L)
                - "states": torch.Tensor of shape (B, state_dim)
            a0: Dictionary of tensors representing the initial actions, with key:
                - "actions": torch.Tensor of shape (B, n_action_steps, action_dim)
            s1: Dictionary of tensors representing the next states, with same keys as s0.
            a1: Dictionary of tensors representing the next actions, with same keys as a0.

        Returns:
            q_values_0: torch.Tensor of shape (B,), Q values for (s0, a0), computed by critic heads.
            q_values_1: torch.Tensor of shape (B,), Q values for (s1, a1), computed by target network.
            log_probs_1: torch.Tensor of shape (B,), log probabilities of actions a1 under the current policy.
            shared_features: tuple containing shared features computed by the critic, it will be used in
                             sac_forward_actor.
        """

        with torch.no_grad():
            prefix_features = self.model.embed_prefix(
                images=torch.cat([s0["images"], s1["images"]], dim=0).unbind(dim=1),
                img_masks=torch.cat([s0["img_masks"], s1["img_masks"]], dim=0).unbind(dim=1),
                lang_tokens=torch.cat([s0["lang_tokens"], s1["lang_tokens"]], dim=0),
                lang_masks=torch.cat([s0["lang_masks"], s1["lang_masks"]], dim=0),
            )
        prefix_features_0 = tuple(feature_chunk.chunk(2, dim=0)[0] for feature_chunk in prefix_features)
        prefix_features_1 = tuple(feature_chunk.chunk(2, dim=0)[1] for feature_chunk in prefix_features)
        prefix_embs, _, _ = prefix_features                        # (2B, 968, 2048)
        states = torch.cat([s0["states"], s1["states"]], dim=0)    # (2B, 32)
        actions = torch.cat([a0["actions"], a1["actions"]], dim=0) # (2B, 50, 32)

        mean_prefix_embs = prefix_embs.mean(dim=1, keepdim=False)                        # (2B, 2048)
        flattened_actions = actions.view(actions.size(0), -1)                            # (2B, 50*32)
        critic_input = torch.cat([mean_prefix_embs, states, flattened_actions], dim=-1)  # (2B, 3680)
        critic_input_0, critic_input_1 = torch.chunk(critic_input, 2, dim=0)             # (B,  3680)

        q_values_0 = self._multi_heads_min(self.critic_heads, critic_input_0)
        with torch.no_grad():
            q_values_1 = self._multi_heads_min(self.target_network_heads, critic_input_1)
            log_probs_1 = self._get_logprobs(s1, prefix_features_1)

        return (q_values_0, q_values_1, log_probs_1, (critic_input_0, prefix_features_0))

    @override
    def sac_forward_actor(
        self,
        s0: dict[str, torch.Tensor],
        a0: dict[str, torch.Tensor],
        shared_features: tuple[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute log probabilities and Q-values for actions sampled by the actor.

            Args:
                s0: Dictionary of tensors representing the initial states, with keys:
                    - "images": torch.Tensor of shape (B, n_images, C, H, W)
                    - "img_masks": torch.Tensor of shape (B, n_images)
                    - "lang_tokens": torch.Tensor of shape (B, L)
                    - "lang_masks": torch.Tensor of shape (B, L)
                    - "states": torch.Tensor of shape (B, state_dim)
                a0: Dictionary of tensors representing the initial actions, with key:
                    - "actions": torch.Tensor of shape (B, n_action_steps, action_dim)
                shared_features: tuple containing shared features computed by the critic in sac_forward_critic.

            Returns:
                log_probs: torch.Tensor of shape (B, 1), log probabilities of the actions sampled by the actor.
                q_values_0: torch.Tensor of shape (B, 1), Q values for (s0, a0) computed using the critic heads.
        """

        critic_input, prefix_features = shared_features

        log_probs = self._get_logprobs(s0, prefix_features)
        q_values_0 = self._multi_heads_min(self.critic_heads, critic_input)

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
