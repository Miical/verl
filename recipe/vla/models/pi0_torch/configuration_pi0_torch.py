from transformers import PretrainedConfig

class PI0TorchConfig(PretrainedConfig):
    model_type = "pi0_torch"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
