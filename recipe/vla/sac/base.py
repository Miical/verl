import torch

class SupportSACTraining:
    """
    Base class for Soft Actor-Critic (SAC).

    Subclasses implement a Policy that can be plugged directly into SAC training.
    This implementation requires the actor and critic to be integrated within a
    single model instance, e.g., sharing a backbone with an additional MLP head
    that outputs critic values (Q/V) alongside the actor's action distribution.

    Note:
        This class intentionally does NOT inherit from `abc.ABC`.
        The root model may be wrapped or transformed by FSDP (Fully Sharded
        Data Parallel), which performs runtime class substitution; using
        `ABCMeta` can break FSDP's class rewriting mechanism.
    """

    def sac_forward(
        self,
        images: dict[str, torch.Tensor],
        img_masks: list[torch.Tensor],
        task: list[str],
        state: torch.Tensor,
        tokenizer) -> dict:

        raise NotImplementedError("Subclasses must implement sac_forward method.")

