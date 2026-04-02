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

from typing import Any

import torch

from verl import DataProto


class SupportSFTTraining:
    """
    Base class for models that support SFT/BC-style supervised updates.

    This intentionally does NOT inherit from `abc.ABC` because model classes may
    be wrapped or rewritten by FSDP at runtime.
    """

    def sft_init(self):
        """Initialize SFT-related components."""

        raise NotImplementedError("Subclasses must implement sft_init method.")

    def bc_loss(
        self,
        obs: DataProto,
        tokenizer: torch.nn.Module,
        actions: dict[str, torch.Tensor],
        valids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute behavior cloning loss for supervised/SFT training."""

        raise NotImplementedError("Subclasses must implement bc_loss method.")
