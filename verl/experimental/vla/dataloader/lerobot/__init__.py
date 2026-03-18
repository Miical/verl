from __future__ import annotations

from bisect import bisect_right
from typing import Any

import torch
from torch.utils.data import Dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from .collator import collate_fn
from .sampler import EpisodeSampler


def _to_1d_long_tensor(x: Any) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().long().view(-1)
    return torch.as_tensor(x, dtype=torch.long).view(-1)


def _resolve_episode_boundaries(dataset: LeRobotDataset) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Support both newer metadata layout:
        dataset.meta.episodes["dataset_from_index"], ["dataset_to_index"]
    and older/v2.x style layout:
        dataset.episode_data_index["from"], ["to"]
    """
    # Newer layout
    meta = getattr(dataset, "meta", None)
    if meta is not None and hasattr(meta, "episodes"):
        episodes = meta.episodes
        if isinstance(episodes, dict):
            if "dataset_from_index" in episodes and "dataset_to_index" in episodes:
                return (
                    _to_1d_long_tensor(episodes["dataset_from_index"]),
                    _to_1d_long_tensor(episodes["dataset_to_index"]),
                )

    # Older / v2.x layout
    episode_data_index = getattr(dataset, "episode_data_index", None)
    if episode_data_index is not None:
        if "from" in episode_data_index and "to" in episode_data_index:
            return (
                _to_1d_long_tensor(episode_data_index["from"]),
                _to_1d_long_tensor(episode_data_index["to"]),
            )

    raise KeyError(
        "Cannot find episode boundaries in LeRobotDataset. "
        "Expected either meta.episodes['dataset_from_index'/'dataset_to_index'] "
        "or episode_data_index['from'/'to']."
    )


class SequenceDataset(Dataset):
    """
    Wrap a frame-level LeRobotDataset into a sequence-level dataset:
        one item = [t0_dict, t1_dict]

    Also inject a synthetic `next.done` bool tensor so downstream RL code
    can derive sparse rewards / dones in offline setting.
    """

    def __init__(
        self,
        base_dataset: LeRobotDataset,
        seq_indices: list[list[int]],
        dataset_from_indices: list[int],
        dataset_to_indices: list[int],
    ):
        self.base_dataset = base_dataset
        self.seq_indices = seq_indices
        self.dataset_from_indices = dataset_from_indices
        self.dataset_to_indices = dataset_to_indices

    def __len__(self) -> int:
        return len(self.seq_indices)

    def _find_episode_end_exclusive(self, first_idx: int) -> int:
        ep_pos = bisect_right(self.dataset_from_indices, first_idx) - 1
        assert ep_pos >= 0, f"Cannot locate episode for index {first_idx}"
        return self.dataset_to_indices[ep_pos]

    def __getitem__(self, i: int):
        pair = self.seq_indices[i]
        assert len(pair) == 2, f"Expected seq_len=2, got {len(pair)}"

        idx0, idx1 = pair[0], pair[1]
        sample0 = self.base_dataset[idx0]
        sample1 = self.base_dataset[idx1]

        # dataset_to_indices is exclusive end index
        episode_end_exclusive = self._find_episode_end_exclusive(idx0)
        is_terminal = idx1 == (episode_end_exclusive - 1)

        done_tensor = torch.tensor(is_terminal, dtype=torch.bool)

        # Inject synthetic transition-level done flag
        sample0["next.done"] = done_tensor
        sample1["next.done"] = done_tensor

        return [sample0, sample1]


def make_dataset(repo_id: str, root: str) -> SequenceDataset:
    base_dataset = LeRobotDataset(
        repo_id=repo_id,
        root=root,
        video_backend="pyav",
    )

    # keep 50-step action chunk horizon for downstream processing
    base_dataset.delta_indices = {"action": list(range(50))}

    dataset_from_indices, dataset_to_indices = _resolve_episode_boundaries(base_dataset)
    dataset_from_indices = dataset_from_indices.tolist()
    dataset_to_indices = dataset_to_indices.tolist()

    sampler = EpisodeSampler(
        dataset_from_indices=dataset_from_indices,
        dataset_to_indices=dataset_to_indices,
        episode_indices_to_use=list(range(len(dataset_from_indices))),
        seq_len=2,
        stride=1,
        shuffle=False,
    )

    seq_indices = list(iter(sampler))
    return SequenceDataset(
        base_dataset=base_dataset,
        seq_indices=seq_indices,
        dataset_from_indices=dataset_from_indices,
        dataset_to_indices=dataset_to_indices,
    )


def make_sampler(dataset: Dataset):
    # SequenceDataset already represents sequence-level items,
    # so DataLoader should use ordinary batching by batch_size.
    return None


def make_collator():
    return collate_fn