from __future__ import annotations

from onnx_ir import Tensor

import torch
from transformers import PreTrainedModel
from verl.utils.device import get_device_name

from .configuration_pi0_torch import PI0TorchConfig
from .model.modeling_pi0 import PI0Model
from .pi0_utils import (
    ImageTransform,
    Normalize,
    PromptTokenizerTransform,
    Unnormalize,
)

class PI0ForActionPrediction(PreTrainedModel):
    config_class = PI0TorchConfig
    base_model_prefix = "pi0_torch"

    def __init__(self, config: PI0TorchConfig):
        super().__init__(config)
        self.model = None
        self.state_norm_stats = config.state_norm_stats
        self.action_norm_stats = config.action_norm_stats
        self.pi05_enabled = self.config.pi05_enabled

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
