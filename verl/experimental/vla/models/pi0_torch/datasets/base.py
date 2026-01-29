from verl.protocol import DataProto
from abc import ABC, abstractmethod

class Pi0DatasetInput(ABC):
    def __init__(self):
        
        # dict containing state information:
        #   images: dict[str, torch.Tensor] with keys: [
        #       'observation.images.cam_high',
        #       'observation.images.cam_left_wrist',  
        #       'observation.images.cam_right_wrist'
        #   ], each with shape (B, C, H, W)
        #   img_masks: list[torch.Tensor] each with shape (B,)
        #   task: list[str]
        #   state: torch.Tensor with shape (B, state_dim)
        self.s0, self.s1 = {}, {}

        # dict containing action information:
        #   action: torch.Tensor with shape (B, action_dim)
        self.a0, self.a1 = {}, {}

        # reward information, float tensor with shape (B,)
        self.reward = None

        # valid mask, bool tensor with shape (B,)
        self.valid = None
    
    @classmethod
    @abstractmethod
    def from_dataset_batch(cls, batch: DataProto) -> "Pi0DatasetInput":
        ...



    
