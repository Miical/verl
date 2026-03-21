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

from transformers import PretrainedConfig


class PI0TorchConfig(PretrainedConfig):
    model_type = "pi0_torch"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.state_norm_stats = kwargs.get("state_norm_stats", {})
        self.action_norm_stats = kwargs.get("action_norm_stats", {})
        self.pi05_enabled = kwargs.get("pi05_enabled", False)
        self.norm_stats_path = kwargs.get("norm_stats_path", None)

        # ReinFlow-lite stochastic flow parameters
        self.flow_logprob_mode = kwargs.get("flow_logprob_mode", "path_exact")
        self.flow_sigma_head_hidden_dim = kwargs.get("flow_sigma_head_hidden_dim", 512)
        self.flow_sigma_min = kwargs.get("flow_sigma_min", 1e-3)
        self.flow_sigma_max = kwargs.get("flow_sigma_max", 5e-1)
        self.flow_sigma_init = kwargs.get("flow_sigma_init", 5e-2)
        self.flow_sigma_use_latent_stats = kwargs.get("flow_sigma_use_latent_stats", True)

        # Rollout debug / dataset replay knobs.
        # These are intentionally optional so normal training / rollout paths stay unchanged.
        self.debug_rollout_mode = kwargs.get("debug_rollout_mode", "off")
        self.debug_dataset_root = kwargs.get("debug_dataset_root", None)
        self.debug_dataset_repo_id = kwargs.get("debug_dataset_repo_id", None)
        self.debug_dataset_index = kwargs.get("debug_dataset_index", -1)
        self.debug_dataset_start_index = kwargs.get("debug_dataset_start_index", 0)
        self.debug_dataset_stride = kwargs.get("debug_dataset_stride", kwargs.get("n_action_steps", 1))
        self.debug_dataset_auto_advance = kwargs.get("debug_dataset_auto_advance", True)
        self.debug_dataset_action_send_variant = kwargs.get("debug_dataset_action_send_variant", "recovered_abs")
        self.debug_dataset_observation_send_ground_truth = kwargs.get(
            "debug_dataset_observation_send_ground_truth", False
        )
