
import torch

class EpisodeSampler:
    def __init__(
        self,
        dataset_from_indices,
        dataset_to_indices,
        episode_indices_to_use=None,
        drop_n_first_frames=0,
        drop_n_last_frames=0,
        seq_len=50,
        stride=None,
        shuffle=False,
    ):
        if stride is None:
            stride = seq_len

        indices = []
        for episode_idx, (start_index, end_index) in enumerate(
            zip(dataset_from_indices, dataset_to_indices, strict=True)
        ):
            if episode_indices_to_use is None or episode_idx in episode_indices_to_use:
                episode_range = list(
                    range(
                        start_index + drop_n_first_frames,
                        end_index - drop_n_last_frames
                    )
                )

                for i in range(0, len(episode_range) - seq_len + 1, stride):
                    indices.append(episode_range[i:i + seq_len])

        self.indices = indices
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            for i in torch.randperm(len(self.indices)):
                yield self.indices[i]
        else:
            for seq in self.indices:
                yield seq

    def __len__(self):
        return len(self.indices)
