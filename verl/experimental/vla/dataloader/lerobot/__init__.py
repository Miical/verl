from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from .collator import collate_fn


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
    meta = getattr(dataset, "meta", None)
    if meta is not None and hasattr(meta, "episodes"):
        episodes = meta.episodes
        if isinstance(episodes, dict):
            if "dataset_from_index" in episodes and "dataset_to_index" in episodes:
                return (
                    _to_1d_long_tensor(episodes["dataset_from_index"]),
                    _to_1d_long_tensor(episodes["dataset_to_index"]),
                )

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


def _build_horizon_pairs(
    dataset_from_indices: list[int],
    dataset_to_indices: list[int],
    transition_horizon: int,
) -> list[list[int]]:
    """
    Build sequence pairs with chunk-horizon semantics.

    For each episode and start index t, produce:
        [t, min(t + transition_horizon, episode_end - 1)]

    This yields sliding windows:
        (t, t+H), (t+1, t+1+H), ...
    and for the tail of the episode where fewer than H future steps remain,
    it pairs to the terminal frame:
        (T-H+1, T), ..., (T-1, T)

    This matches the semantics:
      - s0 is the state before executing the chunk at time t
      - a0 is the future H-step action chunk from time t
      - s1 is the state after up to H steps, or earlier if the episode ends
    """
    if transition_horizon <= 0:
        raise ValueError(f"transition_horizon must be positive, got {transition_horizon}")

    seq_indices: list[list[int]] = []
    for start, end_exclusive in zip(dataset_from_indices, dataset_to_indices, strict=True):
        terminal_idx = end_exclusive - 1
        # Need at least one next state.
        for idx0 in range(start, terminal_idx):
            idx1 = min(idx0 + transition_horizon, terminal_idx)
            seq_indices.append([idx0, idx1])
    return seq_indices


class SequenceDataset(Dataset):
    """
    Wrap a frame-level LeRobotDataset into a sequence-level dataset:
        one item = [t0_dict, t1_dict]

    Here t1 is not the adjacent frame by default. Instead, it is the frame after
    `transition_horizon` steps, or the episode terminal frame if fewer than
    `transition_horizon` future steps remain.

    We also inject a synthetic `next.done` flag with chunk semantics:
        next.done = 1 iff the transition reaches episode terminal within horizon.
    """

    def __init__(
        self,
        base_dataset: LeRobotDataset,
        seq_indices: list[list[int]],
        dataset_to_indices: list[int],
        transition_horizon: int,
    ):
        self.base_dataset = base_dataset
        self.seq_indices = seq_indices
        self.dataset_to_indices = dataset_to_indices
        self.transition_horizon = transition_horizon

    def __len__(self) -> int:
        return len(self.seq_indices)

    def __getitem__(self, i: int):
        idx0, idx1 = self.seq_indices[i]
        sample0 = self.base_dataset[idx0]
        sample1 = self.base_dataset[idx1]

        # Terminal with chunk semantics: the chunk reaches episode end within horizon.
        is_terminal = False
        for end_exclusive in self.dataset_to_indices:
            if idx0 < end_exclusive:
                is_terminal = idx1 == (end_exclusive - 1)
                break

        done_tensor = torch.tensor(is_terminal, dtype=torch.bool)

        sample0["next.done"] = done_tensor
        sample1["next.done"] = done_tensor
        sample0["transition.horizon"] = torch.tensor(self.transition_horizon, dtype=torch.long)
        sample1["transition.horizon"] = torch.tensor(self.transition_horizon, dtype=torch.long)
        return [sample0, sample1]


def make_dataset(repo_id: str, root: str) -> SequenceDataset:
    base_dataset = LeRobotDataset(
        repo_id=repo_id,
        root=root,
        video_backend="pyav",
    )

    # Keep 50-step action chunk horizon for downstream processing.
    action_horizon = 50
    base_dataset.delta_indices = {"action": list(range(action_horizon))}

    dataset_from_indices, dataset_to_indices = _resolve_episode_boundaries(base_dataset)
    dataset_from_indices = dataset_from_indices.tolist()
    dataset_to_indices = dataset_to_indices.tolist()

    seq_indices = _build_horizon_pairs(
        dataset_from_indices=dataset_from_indices,
        dataset_to_indices=dataset_to_indices,
        transition_horizon=action_horizon,
    )

    return SequenceDataset(
        base_dataset=base_dataset,
        seq_indices=seq_indices,
        dataset_to_indices=dataset_to_indices,
        transition_horizon=action_horizon,
    )


def make_sampler(dataset: Dataset):
    # SequenceDataset already represents sequence-level items,
    # so DataLoader should use ordinary batching by batch_size.
    return None


def make_collator():
    return collate_fn
