# Copyright 2025 The RLinf Authors.
# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

"""
LeRobot Dataset for verl RL Training

This module provides a PyTorch Dataset wrapper for LeRobot datasets,
compatible with verl's training framework. It supports:
- Loading LeRobot format datasets using giga_datasets
- Processing images and states for VLA training
- Action chunking for temporal consistency
- Integration with HuggingFace tokenizers and processors
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import AutoTokenizer, ProcessorMixin

logger = logging.getLogger(__name__)


def collate_fn(data_list: list[dict]) -> dict:
    """Collate function for LeRobot RL dataset.
    
    Handles batching of:
    - Tensors (images, states, actions) -> stack
    - Lists/strings (prompts, task descriptions) -> convert to numpy array with dtype=object
    - state_ids -> convert to numpy array in non_tensor_batch
    
    Note: DataProto.from_single_dict only accepts torch.Tensor or np.ndarray,
    so all non-tensor data must be converted to numpy arrays.
    
    Args:
        data_list: List of sample dictionaries from dataset
        
    Returns:
        Batched dictionary with stacked tensors and numpy arrays
        Special handling: state_ids -> non_tensor_batch['state_ids'] as numpy array
    """
    if not data_list:
        return {}
    
    # Separate tensors and non-tensors
    tensors = {}
    non_tensors = {}
    
    # Extract state_ids if present
    state_ids = None
    if 'state_id' in data_list[0]:
        state_ids = np.array([item['state_id'] for item in data_list], dtype=np.int32)
    
    for key in data_list[0].keys():
        if key == 'state_id':
            continue  # Handle separately
            
        values = [item[key] for item in data_list]
        
        # Check if all values are tensors
        if all(isinstance(v, torch.Tensor) for v in values):
            try:
                tensors[key] = torch.stack(values, dim=0)
            except RuntimeError as e:
                # If stacking fails (e.g., different shapes), keep as list
                logger.warning(f"Failed to stack tensors for key '{key}': {e}")
                # Convert to numpy array with object dtype for variable shapes
                non_tensors[key] = np.array(values, dtype=object)
        elif all(isinstance(v, np.ndarray) for v in values):
            # Stack numpy arrays
            try:
                non_tensors[key] = np.stack(values, axis=0)
            except ValueError:
                # Variable shapes, use object dtype
                non_tensors[key] = np.array(values, dtype=object)
        else:
            # Convert lists/strings/other types to numpy array with object dtype
            # This is required because DataProto.from_single_dict only accepts
            # torch.Tensor or np.ndarray
            non_tensors[key] = np.array(values, dtype=object)
    
    result = {**tensors, **non_tensors}
    
    # Add state_ids and task_ids as top-level numpy arrays
    # (DataProto.from_single_dict will move them to non_tensor_batch)
    if state_ids is not None:
        result['state_ids'] = state_ids
        result['task_ids'] = state_ids.copy()  # task_ids same as state_ids for now
    
    return result


class LeRobotRLDataset(Dataset):
    """LeRobot Dataset for verl RL Training.
    
    This dataset loads LeRobot format data using giga_datasets and prepares 
    it for VLA policy training.
    
    Args:
        data_files: Path(s) to dataset directory or files
        tokenizer: HuggingFace tokenizer for text processing
        config: Dataset configuration (from hydra config)
        processor: HuggingFace processor for image/multimodal processing
        max_samples: Maximum number of samples to load (-1 for all)
    """
    
    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: AutoTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
        max_samples: int = -1,
    ):
        super().__init__()
        
        if not isinstance(data_files, (list, ListConfig)):
            data_files = [data_files]
        
        self.data_files = data_files
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.max_samples = max_samples
        
        # Extract config parameters
        self.action_chunk = config.get("action_chunk", 50)
        self.max_prompt_length = config.get("max_prompt_length", 512)
        self.image_key = config.get("image_key", "images")
        self.state_key = config.get("state_key", "state")
        self.action_key = config.get("action_key", "action")
        self.task_description = config.get("task_description", "robot manipulation task")
        
        # Load dataset
        data_path = Path(self.data_files[0])
        logger.info(f"Loading LeRobot dataset from: {data_path}")
        
        # Check if dataset path exists
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {data_path}")
        
        # Check for LeRobot format
        if (data_path / 'meta' / 'info.json').exists():
            logger.info("LeRobot format detected, using local loader (skip network requests)")
            # 直接使用本地加载器,避免网络请求
            try:
                self._load_with_local_metadata(data_path)
            except Exception as e:
                logger.error(f"Failed to load dataset: {e}")
                raise
        else:
            raise FileNotFoundError(
                f"Cannot find LeRobot dataset at {data_path}. "
                f"Expected 'meta/info.json' file"
            )
        
        logger.info(f"LeRobotRLDataset initialized with {self.total_samples} samples")
    
    def _load_with_local_metadata(self, data_path: Path):
        """Load dataset using local metadata only (no network requests).
        
        只读取 meta/episodes.jsonl 来获取数据集大小,不加载实际数据。
        这样可以避免网络请求和大文件加载,适合 val_only 模式。
        """
        import json
        
        episodes_file = data_path / 'meta' / 'episodes.jsonl'
        if not episodes_file.exists():
            raise FileNotFoundError(f"Cannot find episodes metadata: {episodes_file}")
        
        logger.info(f"Loading dataset metadata from: {episodes_file}")
        
        # 读取 episodes.jsonl 获取 episode 数量和长度
        episodes = []
        with open(episodes_file, 'r') as f:
            for line in f:
                if line.strip():
                    episodes.append(json.loads(line))
        
        # 计算总样本数(所有 episode 的长度之和)
        self.total_samples = sum(ep['length'] for ep in episodes)
        self.num_episodes = len(episodes)
        self.episodes_metadata = episodes
        self.data_path = data_path
        self.loader_type = 'local_metadata'
        
        # 构建从样本索引到 episode_index 的映射
        # 每个样本的 state_id 就是它所属的 episode_index
        self.sample_to_episode = []
        for ep_idx, ep in enumerate(episodes):
            # 每个 episode 的所有 frame 都使用相同的 episode_index 作为 state_id
            self.sample_to_episode.extend([ep_idx] * ep['length'])
        
        # Apply max_samples limit
        if self.max_samples > 0:
            self.total_samples = min(self.total_samples, self.max_samples)
            self.sample_to_episode = self.sample_to_episode[:self.total_samples]
        
        logger.info(f"Dataset loaded: {self.num_episodes} episodes, {self.total_samples} total frames")
        logger.info(f"Using {self.total_samples} samples for training")
    
    def _load_with_giga_datasets(self, data_path: Path):
        """Load dataset using giga_datasets.
        
        Note: giga_datasets.LeRobotDataset expects data_path to be the parent directory,
        and uses basename as repo_id. But we need to pass the full dataset path.
        
        Workaround: Directly instantiate the underlying FastLeRobotDataset.
        """
        from giga_datasets.datasets.lerobot_dataset import FastLeRobotDataset
        
        # Get repo_id and root from data_path
        repo_id = data_path.name
        root = str(data_path.parent)
        
        logger.info(f"Loading with giga_datasets: repo_id={repo_id}, root={root}")
        
        # Directly use FastLeRobotDataset (bypasses giga_datasets wrapper)
        self.dataset = FastLeRobotDataset(
            repo_id=repo_id,
            root=root,
            delta_timestamps=None,
            video_backend='pyav',
        )
        self.loader_type = 'giga_datasets'
        
        # Apply max_samples limit
        self.total_samples = len(self.dataset)
        if self.max_samples > 0:
            self.total_samples = min(self.total_samples, self.max_samples)
    
    def _load_with_lerobot(self, data_path: Path):
        """Load dataset using LeRobot native format (meta/info.json)."""
        try:
            # Import LeRobot dataset from the embedded lerobot in test_env
            import sys
            lerobot_path = str(Path(__file__).parent.parent / 'envs' / 'test_env' / 'robot' / 'controller' / 'piper' / 'lerobot')
            if lerobot_path not in sys.path:
                sys.path.insert(0, lerobot_path)
            
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
            
            # Get repo_id from path
            repo_id = data_path.name
            
            # Load dataset
            self.dataset = LeRobotDataset(
                repo_id=repo_id,
                root=str(data_path.parent),
                video_backend='pyav',
            )
            self.loader_type = 'lerobot'
            
            # Apply max_samples limit
            self.total_samples = len(self.dataset)
            if self.max_samples > 0:
                self.total_samples = min(self.total_samples, self.max_samples)
                
        except ImportError as e:
            logger.error(f"Failed to import LeRobot dataset: {e}")
            raise ImportError(
                "LeRobot dataset loader not available. Please ensure lerobot is installed or accessible."
            )
    
    def __len__(self) -> int:
        return self.total_samples
    
    def __getitem__(self, idx: int) -> dict:
        """Get a single sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing:
            - input_ids: Tokenized prompt
            - attention_mask: Attention mask for prompt
            - pixel_values: Processed images (if processor available)
            - images: Raw images (if no processor)
            - state: Robot state
            - action: Action labels (for action chunking)
            - prompt: Raw text prompt
            - state_id: Episode index (used as state_id for environment reset)
        """
        # Get state_id (episode_index) for this sample
        if self.loader_type == 'local_metadata':
            state_id = self.sample_to_episode[idx] if idx < len(self.sample_to_episode) else idx % self.num_episodes
        else:
            # For other loaders, try to extract from data or use idx % num_episodes
            state_id = idx % self.num_episodes if hasattr(self, 'num_episodes') else 0
        
        # Handle local_metadata mode (no actual data loading)
        if self.loader_type == 'local_metadata':
            # 返回 dummy 数据(val_only 模式下不会真正使用)
            return self._get_dummy_sample(state_id=state_id)
        
        # Get data from dataset loader
        data = self.dataset[idx]
        
        # Process the sample
        result = self._process_sample(data)
        result['state_id'] = state_id
        return result
    
    def _get_dummy_sample(self, state_id: int = 0) -> dict:
        """返回一个 dummy 样本(用于 val_only 模式)
        
        Args:
            state_id: Episode index to use as state_id
        """
        prompt = self.task_description
        
        # Tokenize prompt
        tokenized = self.tokenizer(
            prompt,
            padding='max_length',
            max_length=self.max_prompt_length,
            truncation=True,
            return_tensors='pt',
        )
        
        return {
            'input_ids': tokenized['input_ids'].squeeze(0),
            'attention_mask': tokenized['attention_mask'].squeeze(0),
            'state': torch.zeros(14, dtype=torch.float32),
            'action': torch.zeros(14, dtype=torch.float32),
            'prompt': prompt,
            'images': {},
            'state_id': state_id,  # Episode index as state_id
        }
    
    def _process_sample(self, data: dict) -> dict:
        """Process a raw data sample into model inputs.
        
        Args:
            data: Raw data from dataset (format depends on loader_type)
            
        Returns:
            Processed sample dictionary with tokenized text and processed images
        """
        # Extract state (convert to tensor if needed)
        state = data.get('observation.state', data.get(self.state_key))
        state = self._to_tensor(state, dtype=torch.float32)
        
        # Extract images - handle both formats
        images = self._extract_images(data)
        
        # Extract action (convert to tensor if needed)
        action = data.get('action', data.get(self.action_key))
        if action is None:
            action = torch.zeros(14, dtype=torch.float32)  # Default action dim
        else:
            action = self._to_tensor(action, dtype=torch.float32)
        
        # Get task description
        prompt = data.get('task', self.task_description)
        
        # Tokenize prompt
        tokenized = self.tokenizer(
            prompt,
            padding='max_length',
            max_length=self.max_prompt_length,
            truncation=True,
            return_tensors='pt',
        )
        
        result = {
            'input_ids': tokenized['input_ids'].squeeze(0),
            'attention_mask': tokenized['attention_mask'].squeeze(0),
            'state': state,
            'action': action,
            'prompt': prompt,
        }
        
        # Process images with processor if available
        if self.processor is not None and images:
            result['pixel_values'] = self._process_images(images, prompt)
        else:
            result['images'] = images
        
        return result
    
    def _extract_images(self, data: dict) -> dict:
        """Extract images from data, handling both giga_datasets and LeRobot formats.
        
        Args:
            data: Raw data dictionary
            
        Returns:
            Dictionary of images with standardized camera names (front, left, right)
        """
        # Try giga_datasets format first
        images = data.get('observation.images', data.get(self.image_key))
        
        if images and isinstance(images, dict):
            # Already in dict format, check if we need to rename keys
            # LeRobot native format uses full keys like "observation.images.top"
            standardized = {}
            camera_map = {
                'top': 'front',
                'cam_high': 'front',
                'observation.images.top': 'front',
                'left_wrist': 'left',
                'cam_left_wrist': 'left',
                'observation.images.left_wrist': 'left',
                'right_wrist': 'right',
                'cam_right_wrist': 'right',
                'observation.images.right_wrist': 'right',
            }
            
            for key, img in images.items():
                # Use mapped name or keep original
                std_name = camera_map.get(key, key)
                standardized[std_name] = img
            
            return standardized
        
        # Try to extract from top-level keys (LeRobot native format)
        standardized = {}
        for key in data.keys():
            if 'observation.images.' in key:
                cam_name = key.split('observation.images.')[-1]
                std_name = {
                    'top': 'front',
                    'left_wrist': 'left',
                    'right_wrist': 'right',
                }.get(cam_name, cam_name)
                standardized[std_name] = data[key]
        
        return standardized if standardized else {}
    
    def _to_tensor(self, data, dtype=torch.float32):
        """Convert data to PyTorch tensor."""
        if isinstance(data, torch.Tensor):
            return data.to(dtype)
        elif isinstance(data, np.ndarray):
            return torch.from_numpy(data).to(dtype)
        else:
            return torch.tensor(data, dtype=dtype)
    
    def _process_images(self, images: dict, prompt: str):
        """Process images using HuggingFace processor.
        
        Args:
            images: Dictionary of camera images
            prompt: Text prompt for multimodal processing
            
        Returns:
            Processed pixel_values tensor or original images dict on failure
        """
        try:
            # Collect images in standard order
            image_list = []
            for cam_name in ['front', 'left', 'right']:
                if cam_name in images:
                    img = images[cam_name]
                    # Convert to numpy if needed
                    if isinstance(img, torch.Tensor):
                        img = img.cpu().numpy()
                    image_list.append(img)
            
            if not image_list:
                return images
            
            # Process with HuggingFace processor
            processed = self.processor(
                images=image_list,
                text=prompt,
                return_tensors='pt',
            )
            return processed['pixel_values'].squeeze(0)
            
        except Exception as e:
            logger.warning(f"Failed to process images with processor: {e}")
            return images
    
    def close(self):
        """Close dataset and release resources."""
        if hasattr(self, 'dataset') and hasattr(self.dataset, 'close'):
            self.dataset.close()


# Export for easy import
__all__ = ['LeRobotRLDataset', 'collate_fn']

