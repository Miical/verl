import torch
from verl import DataProto
from abc import abstractmethod, ABC

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

class BaseSACActor(ABC):
    @abstractmethod
    def update_policy(self, data: DataProto) -> dict:
        """
        Update the policy using the provided data batch.

        Args:
            data: DataProto containing the following entries in `data.batch`:
                - "full_action": Tensor of shape (B, action_steps, action_dim),
                    representing the action chunk for each sample.
                - "states": Tensor of shape (B, state_dim),
                    representing the environment or agent state.
                - "images": Tensor of shape (num_images, B, C, H, W),
                    containing visual observations.
                - "image_masks": Tensor of shape (num_images, B),
                    indicating valid images per sample.
                - "lang_tokens": Tensor of shape (B, max_seq_len),
                    tokenized language instructions.
                - "lang_masks": Tensor of shape (B, max_seq_len),
                    attention masks for language tokens.
                - "reward_tensor": Tensor of shape (B,),
                    chunk-level scalar rewards.
                    Each action chunk corresponds to a single reward value. The reward must be computed at the
                    trajectory level and assigned to chunks externally, as the actor operates purely on chunks
                    and does not have access to full trajectories.
                - "response_mask": Tensor of shape (B,),
                    mask indicating whether each sample has a valid response.
        """
