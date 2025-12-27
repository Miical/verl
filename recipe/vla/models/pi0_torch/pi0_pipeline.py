import torch

from .pi0_utils import (
    ImageTransform,
    Normalize,
    PromptTokenizerTransform,
    Unnormalize,
)

class Pi0Pipeline():
    """High-level inference pipeline for PI0 policy.

    This pipeline handles preprocessing (state/image/token), model invocation, and postprocessing back to the environment action space.
    """

    def __init__(
        self,
        policy,
        tokenizer_model_path: str,
        state_norm_stats: dict,
        action_norm_stats: dict,
        original_action_dim: int = 14,
    ) -> None:
        """Initialize the PI0 pipeline.

        Args:
            model_path: Path to the pretrained PI0 policy weights.
            tokenizer_model_path: Path or hub id for the tokenizer used in prompt processing.
            state_norm_stats: Normalization stats for robot state; either {'mean','std'} or {'q01','q99'}.
            action_norm_stats: Normalization stats for actions; either {'mean','std'} or {'q01','q99'}.
            original_action_dim: Dimension of the environment's native action space before padding (e.g., 14 or 16).
        """
        super().__init__()
        self.policy = policy
        self.policy.eval()
        self.device = 'cpu'
        self.pi05_enabled = self.policy.pi05_enabled
        # Input transforms
        self.state_normalize_transform = Normalize(state_norm_stats, use_quantiles=self.pi05_enabled)
        self.image_transform = ImageTransform(
            resize_imgs_with_padding=(224, 224),
            enable_image_aug=False,
        )
        max_length = 200 if self.pi05_enabled else 48
        self.prompt_tokenizer_transform = PromptTokenizerTransform(
            tokenizer_model_path=tokenizer_model_path, max_length=max_length, discrete_state_input=False
        )
        # Output transforms
        self.state_unnormalize_transform = Unnormalize(state_norm_stats, use_quantiles=self.pi05_enabled)
        self.action_unnormalize_transform = Unnormalize(action_norm_stats, use_quantiles=self.pi05_enabled)

    def to(self, device: torch.device | str):
        """Move the policy and all transforms to the specified device.

        Args:
            device: Target device (e.g., 'cuda', 'cpu', torch.device('cuda:0')).

        Returns:
            self: Enables chained calls.
        """
        self.device = device
        self.policy.to(device)
        self.state_normalize_transform.to(device)
        self.state_unnormalize_transform.to(device)
        self.action_unnormalize_transform.to(device)
        return self

    def compile(self, **kwargs) -> None:
        """Compile the sampling function for improved runtime speed.

        Note:
            This uses `torch.compile` under the hood and forwards any kwargs.
        """
        self.policy.sample_actions = torch.compile(self.policy.sample_actions, **kwargs)

    @torch.no_grad()
    def __call__(
        self,
        images: dict[str, torch.Tensor],
        task: list[str],
        state: torch.Tensor,
    ) -> torch.Tensor:
        """Run one forward pass from raw inputs to final action sequence.

        Args:
            images: Observation images of the robot. Each value is a tensor with shape (C,H,W).
            task: Natural language task description.
            state: The robot joint state tensor with shape (state_dim,).

        Returns:
            A tensor of predicted actions with shape (batch, num_steps, original_action_dim) on the original input device.
        """
        # Input transforms
        ori_device = state.device
        state = state.to(self.device)
        for key in images:
            images[key] = images[key].to(self.device)

        state = self.state_normalize_transform(state)
        images, img_masks = self.image_transform.call_batch(images)

        bsize = state.shape[0]
        img_masks = [
            torch.ones((bsize,), device=self.device, dtype=torch.bool),
            torch.ones((bsize,), device=self.device, dtype=torch.bool),
            torch.zeros((bsize,), device=self.device, dtype=torch.bool),
        ]

        lang_tokens, lang_masks = self.prompt_tokenizer_transform.call_batch({'task': task, 'observation.state': state})

        # Inference
        pred_action = self.policy.sample_actions(images, img_masks, lang_tokens, lang_masks, state=state)

        # Output transforms
        state = self.state_unnormalize_transform(state)
        pred_action = self.action_unnormalize_transform(pred_action)

        return pred_action, images, img_masks, lang_tokens, lang_masks, state
