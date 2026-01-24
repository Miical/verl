from typing_extensions import override
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from .sampler import EpisodeSampler
from .collator import collate_func

class _LeRobotDataset(LeRobotDataset):
    @override
    def __getitem__(self, index):
        if isinstance(index, (list, tuple)):
            data = [
                super(_LeRobotDataset, self).__getitem__(i)
                for i in index
            ]
        else:
            data = super(_LeRobotDataset, self).__getitem__(index)

        return data

def make_dataset(repo_id: str, root: str) -> _LeRobotDataset:
    dataset = _LeRobotDataset(
        repo_id=repo_id,
        root=root
    )
    dataset.delta_indices = {'action': list(range(50))}
    return dataset


def make_sampler(dataset: _LeRobotDataset) -> EpisodeSampler:
    sampler = EpisodeSampler(
        dataset_from_indices=dataset.meta.episodes["dataset_from_index"],
        dataset_to_indices=dataset.meta.episodes["dataset_to_index"],
        seq_len=2
    )
    return sampler

def make_collator() -> callable:
    return collate_func




