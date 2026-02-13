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

from typing import Literal
import logging

import torch
from onnx_ir import Tensor
from torch import nn
from torch.distributed.fsdp import register_fsdp_forward_method
from torch.distributions import Normal
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

logger = logging.getLogger(__name__)


class PI0ForActionPrediction(PreTrainedModel, SupportSACTraining):
    config_class = PI0TorchConfig
    base_model_prefix = "pi0_torch"

    def __init__(self, config: PI0TorchConfig):
        super().__init__(config)
        self.model: PI0Model = None
        self.state_norm_stats = config.state_norm_stats
        self.action_norm_stats = config.action_norm_stats
        self.pi05_enabled = config.pi05_enabled
        
        # ===================== MERGED: Code B 配置 =====================
        self.use_endpose = getattr(config, "use_endpose", False)
        self.no_state = getattr(config, "no_state", False)
        # ===================== END MERGED =====================

        assert self.state_norm_stats, "state_norm_stats must be provided in PI0TorchConfig"
        assert self.action_norm_stats, "action_norm_stats must be provided in PI0TorchConfig"
        assert isinstance(self.pi05_enabled, bool), "pi05_enabled must be provided in PI0TorchConfig"

        # Input transforms
        self.state_normalize_transform = Normalize(self.state_norm_stats, use_quantiles=self.pi05_enabled)
        self.action_normalize_transform = Normalize(self.action_norm_stats, use_quantiles=self.pi05_enabled)
        self.image_transform = ImageTransform(resize_imgs_with_padding=(224, 224), enable_image_aug=False)
        max_length = 200 if self.pi05_enabled else 48
        
        # ===================== MERGED: Code B 的关键修改 =====================
        # Pi0.5 必须开启离散 state 输入，否则 prompt 里没有 state conditioning
        self.prompt_tokenizer_transform = PromptTokenizerTransform(
            max_length=max_length,
            discrete_state_input=self.pi05_enabled
        )
        # ===================== END MERGED =====================

        # Output transforms
        self.state_unnormalize_transform = Unnormalize(self.state_norm_stats, use_quantiles=self.pi05_enabled)
        self.action_unnormalize_transform = Unnormalize(self.action_norm_stats, use_quantiles=self.pi05_enabled)

        # ===================== MERGED: Code B 的警告开关 =====================
        self._warned_no_abs_transform = False
        # ===================== END MERGED =====================

        self._to(get_device_name())

        ##### SAC Algorithm Support #####
        # ===================== 保留 Code A 的网络结构 =====================
        if getattr(self.config, "sac_enable", False):
            head_num = 2 if getattr(self.config, "double_q", True) else 1

            self.critic_heads = nn.ModuleList(
                [
                    MLP(
                        input_dim=2150,  # 2048(prefix mean) + 32(state) + 10*7(action flat)
                        hidden_dims=[1024, 512, 256],
                        output_dim=1,
                        activation="relu",
                        init_method="normal",
                    )
                    for _ in range(head_num)
                ]
            )

            self.target_network_heads = nn.ModuleList(
                [
                    MLP(
                        input_dim=2150,
                        hidden_dims=[1024, 512, 256],
                        output_dim=1,
                        activation="relu",
                        init_method="normal",
                    )
                    for _ in range(head_num)
                ]
            )
        # ===================== END 保留 Code A =====================

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
        env_name: str,
    ) -> tuple[Pi0Output, dict, dict]:
        """Run one forward pass from raw inputs to final action sequence."""

        from .policy.libero_policy import LiberoPi0Input
        from .policy.piper_policy import PiperPi0Input

        if env_name == "piper":
            pi0_input = PiperPi0Input.from_env_obs(env_obs)
        elif env_name == "libero":
            pi0_input = LiberoPi0Input.from_env_obs(env_obs)
        else:
            raise ValueError(f"Unknown env_name={env_name}")

        # 保存原始 state (未归一化) 用于 delta->absolute 后处理
        state_raw_unnormalized = pi0_input.state

        # 让 transforms buffer 跟随输入 device（解决 cuda:0 vs cuda:4）
        target_device = state_raw_unnormalized.device
        try:
            self._to(target_device)
        except Exception:
            pass

        raw_state_dim = int(state_raw_unnormalized.shape[-1])
        cfg = getattr(self, "config", None)

        # ===== CHANGED =====
        # 真实动作维度：piper 优先用 cfg.action_dim（你已设为14），并强制 clamp，避免变成32导致 mask/where 维度炸
        if env_name == "piper" and cfg is not None and getattr(cfg, "action_dim", None) is not None:
            real_act_dim = int(cfg.action_dim)   # 期望=14
        else:
            # libero 等场景：默认跟随输入 state 的真实维度（更安全）
            real_act_dim = raw_state_dim

        # clamp：防御性对齐（避免 state/pred_action 维度更小导致越界）
        real_act_dim = max(1, min(real_act_dim, raw_state_dim))
        # ===== END CHANGED =====

        # Input transforms
        state = self.state_normalize_transform(state_raw_unnormalized)

        if self.no_state:
            logger.info("no_state is True, zeroing state")
            state = torch.zeros_like(state)

        images, _ = self.image_transform.call_batch(pi0_input.images)

        # prompt：pi0.5 下只用真实动作维度（不含 pad）
        if self.pi05_enabled:
            state_for_prompt = state[:, :real_act_dim]
        else:
            state_for_prompt = state

        lang_tokens, lang_masks = self.prompt_tokenizer_transform.call_batch(
            {"task": pi0_input.task, "observation.state": state_for_prompt},
            tokenizer,
        )

        # Inference
        pred_action = self.model.sample_actions(images, pi0_input.img_masks, lang_tokens, lang_masks, state=state)

        # Unnormalize action
        pred_action = self.action_unnormalize_transform(pred_action)

        # ===== CHANGED =====
        # 进一步 clamp 到 pred_action 的最后维度（防止模型输出维度 < real_act_dim 的极端情况）
        real_act_dim = min(real_act_dim, int(pred_action.shape[-1]))

        # mask 改为“按 real_act_dim 动态生成”，避免硬编码14造成 where 维度不匹配
        # 语义保持不变：joint=True（累加），gripper=False（保持delta）
        mask = torch.ones((real_act_dim,), device=pred_action.device, dtype=torch.bool)
        # 双臂：左 gripper=6，右 gripper=13（如果存在）
        if real_act_dim > 6:
            mask[6] = False
        if real_act_dim > 13:
            mask[13] = False
        # ===== END CHANGED =====

        if self.use_endpose:
            logger.info("use_endpose is True, applying endpose transformation")
            from scipy.spatial.transform import Rotation as R

            B, T, A = pred_action.shape

            # ===== CHANGED =====
            # endpose 逻辑只对 piper(14维) 有意义；防御：维度不够就直接跳过转换
            if real_act_dim < 14:
                logger.warning(f"use_endpose=True but real_act_dim={real_act_dim} < 14, skip endpose abs conversion.")
            else:
                pos_indices_left = torch.tensor([0, 1, 2], device=pred_action.device)
                pos_indices_right = torch.tensor([7, 8, 9], device=pred_action.device)
                rot_indices_left = torch.tensor([3, 4, 5], device=pred_action.device)
                rot_indices_right = torch.tensor([10, 11, 12], device=pred_action.device)

                abs_action = pred_action.clone()

                s0_pos = state_raw_unnormalized[:, :real_act_dim]
                delta_pos = pred_action[:, :, :real_act_dim]

                abs_action[:, :, pos_indices_left] = (
                    s0_pos[:, pos_indices_left].unsqueeze(1) + delta_pos[:, :, pos_indices_left]
                )
                abs_action[:, :, pos_indices_right] = (
                    s0_pos[:, pos_indices_right].unsqueeze(1) + delta_pos[:, :, pos_indices_right]
                )

                s0_rot_left = state_raw_unnormalized[:, rot_indices_left]
                s0_rot_right = state_raw_unnormalized[:, rot_indices_right]

                for t in range(T):
                    delta_rot_left = pred_action[:, t, rot_indices_left]
                    delta_rot_right = pred_action[:, t, rot_indices_right]

                    for b in range(B):
                        state_left_rot = R.from_euler("xyz", s0_rot_left[b].cpu().numpy(), degrees=True)
                        delta_left_rot = R.from_euler("xyz", delta_rot_left[b].cpu().numpy(), degrees=True)
                        abs_rot_left = state_left_rot * delta_left_rot
                        abs_euler_left = torch.tensor(
                            abs_rot_left.as_euler("xyz", degrees=True),
                            device=pred_action.device,
                            dtype=pred_action.dtype,
                        )
                        abs_action[b, t, rot_indices_left] = abs_euler_left

                        state_right_rot = R.from_euler("xyz", s0_rot_right[b].cpu().numpy(), degrees=True)
                        delta_right_rot = R.from_euler("xyz", delta_rot_right[b].cpu().numpy(), degrees=True)
                        abs_rot_right = state_right_rot * delta_right_rot
                        abs_euler_right = torch.tensor(
                            abs_rot_right.as_euler("xyz", degrees=True),
                            device=pred_action.device,
                            dtype=pred_action.dtype,
                        )
                        abs_action[b, t, rot_indices_right] = abs_euler_right

                pred_action = abs_action
            # ===== END CHANGED =====

        else:
            # Joint 模式：根据 mask 选择性累加
            s0 = state_raw_unnormalized[:, :real_act_dim].unsqueeze(1)  # (B,1,real_act_dim)
            delta = pred_action[..., :real_act_dim]  # (B,T,real_act_dim)

            abs_action = pred_action.clone()
            abs_action[..., :real_act_dim] = torch.where(
                mask.view(1, 1, -1),   # (1,1,real_act_dim)
                s0 + delta,            # joint: state + delta
                delta,                 # gripper: keep delta
            )
            pred_action = abs_action

            if not self._warned_no_abs_transform:
                logger.warning(
                    "[PI0ForActionPrediction] Using fallback delta->absolute transformation "
                    "(joint mode with mask-based accumulation)."
                )
                self._warned_no_abs_transform = True

        from .policy.libero_policy import LiberoPi0Output
        from .policy.piper_policy import PiperPi0Output

        if env_name == "libero":
            pi0_output = LiberoPi0Output.from_model_output({"full_action": pred_action})
        else:
            pi0_output = PiperPi0Output.from_model_output({"full_action": pred_action})

        # ===== CHANGED =====
        # piper 分支也补齐 s/a（很多训练/rollout链路会依赖）
        s = {
            "states": state,
            "images": torch.stack(images, dim=1),
            "image_masks": torch.stack(pi0_input.img_masks, dim=1),
            "lang_tokens": lang_tokens,
            "lang_masks": lang_masks,
        }
        a = {
            "full_action": self.action_normalize_transform(pred_action),
        }
        # ===== END CHANGED =====

        return pi0_output, s, a


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
        *,
        x_t: torch.Tensor | None = None,  # (B, T, A)
        x_next: torch.Tensor | None = None,  # (B, T, A)
        v_t: torch.Tensor | None = None,  # (B, T, A)
        t: torch.Tensor | None = None,  # (B,)
        step_idx: torch.Tensor | None = None,  # (B,)
    ) -> torch.Tensor:
        """
        Compute log-probability of x_{t+1} given (x_t, v_t) under the Flow-SDE formulation.
        See https://arxiv.org/abs/2510.25889
        """

        prefix_embs, prefix_pad_masks, _ = prefix_features
        states = s["states"]
        B = prefix_embs.shape[0]
        device = prefix_embs.device

        past_key_values = self._build_kv_cache_from_prefix(prefix_features)

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

        # sigma schedule step index
        K = int(self.model.num_steps)
        if step_idx is None:
            step_idx = torch.full((B,), K - 1, device=device, dtype=torch.long)

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
        x1_weight = t_b - dt_b - (sigma_i**2) * dt_b / (2.0 * t_b.clamp_min(1e-6))

        x_next_mean = x0_pred * x0_weight + x1_pred * x1_weight
        x_next_std = (dt_b.sqrt() * sigma_i).clamp_min(1e-6)

        dist = Normal(x_next_mean.float(), x_next_std.float())
        log_probs = dist.log_prob(x_next.float()).sum(dim=2).mean(dim=1)  # (B,)
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
        register_fsdp_forward_method(self, "sac_forward_state_features")

    @override
    def sac_forward_actor(
        self,
        state_features: tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prefix_features, states = state_features
        actions, log_probs = self._sample_actions_and_logprobs_from_prefix(states, prefix_features)
        return actions, log_probs

    @override
    def sac_forward_critic(
        self,
        a: dict[str, torch.Tensor],
        state_features: tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        *,
        use_target_network: bool = False,
        method: Literal["cat", "min"] = "cat",
        requires_grad: bool = False,
    ):
        critic_head = self.target_network_heads if use_target_network else self.critic_heads
        for p in critic_head.parameters():
            p.requires_grad_(requires_grad)

        prefix_features, states = state_features
        prefix_embs, _, _ = prefix_features
        mean_prefix_embs = prefix_embs.mean(dim=1, keepdim=False)  # (B, 2048)
        actions = a["full_action"][:, :10, :7]  # (B, 10, 7)
        flattened_actions = actions.reshape(actions.shape[0], -1)  # (B, 70)
        critic_input = torch.cat([mean_prefix_embs, states, flattened_actions], dim=-1)  # (B, 2150)

        q_values = self._multi_heads_value(critic_head, critic_input, method=method)

        return q_values

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
        for target_head, head in zip(self.target_network_heads, self.critic_heads, strict=False):
            for target_param, param in zip(target_head.parameters(), head.parameters(), strict=False):
                target_param.data.mul_(1.0 - tau).add_(param.data, alpha=tau)
