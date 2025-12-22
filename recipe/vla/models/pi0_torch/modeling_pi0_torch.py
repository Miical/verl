from __future__ import annotations

from typing import Any
from transformers import PreTrainedModel

from .configuration_pi0_torch import PI0TorchConfig
from .modeling_pi0 import PI0Policy  # you already confirmed this works

class PI0TorchModel(PreTrainedModel):
    config_class = PI0TorchConfig
    base_model_prefix = "pi0_torch"

    def __init__(self, config: PI0TorchConfig):
        super().__init__(config)
        self.policy = None

    def forward(self, *args: Any, **kwargs: Any):
        if self.policy is None:
            raise RuntimeError("PI0TorchModel.policy is not initialized. Did from_pretrained() run?")
        return self.policy(*args, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop("config", None)

        if config is None:
            config = PI0TorchConfig.from_pretrained(pretrained_model_name_or_path)

        model = cls(config)
        model.policy = PI0Policy.from_pretrained(pretrained_model_name_or_path)
        return model

