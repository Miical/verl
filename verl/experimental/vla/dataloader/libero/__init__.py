import glob
import os
from typing import Callable

import h5py
import torch
from torch.utils.data import Dataset

from .collator import libero_collate_fn

N_ACTIONS_STEPS = 10


class LiberoTrajDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.hdf5_files = sorted(glob.glob(os.path.join(root_dir, "*_demo.hdf5")))
        if len(self.hdf5_files) == 0:
            raise FileNotFoundError(f"No *_demo.hdf5 found under: {root_dir}")

        self.samples = []
        for path in self.hdf5_files:
            with h5py.File(path, "r") as f:
                if "data" not in f:
                    continue
                for demo_key in f["data"].keys():
                    self.samples.append((path, demo_key))

        print(f"[LiberoTrajDataset] Found {len(self.hdf5_files)} task files, {len(self.samples)} demos total.")

    def __len__(self):
        return len(self.samples)

    def get_demo_ref(self, idx):
        return self.samples[idx]


class LiberoTwoStepDataset(Dataset):
    """
    dataset_keys = ['obs', 'actions', 'dones', 'rewards', 'robot_states', 'states']
    """

    def __init__(self, root_dir, step=N_ACTIONS_STEPS, build_index=True):
        self.root_dir = root_dir
        self.step = int(step)
        self.traj_ds = LiberoTrajDataset(root_dir)
        self.index = []  # list of (hdf5_path, demo_key, t0)
        self.pair_index = []  # list of ((hdf5_path, demo_key, t0), (hdf5_path, demo_key, t1))

        if build_index:
            self._build_index()

    def _build_index(self):
        for i in range(len(self.traj_ds)):
            hdf5_path, demo_key = self.traj_ds.get_demo_ref(i)
            with h5py.File(hdf5_path, "r") as f:
                demo = f["data"][demo_key]
                T = demo["actions"].shape[0]
                max_t0 = T - self.step
                if max_t0 < 0:
                    continue
                demo_chunks = []
                for t0 in range(0, max_t0 + 1, self.step):
                    ref = (hdf5_path, demo_key, t0)
                    self.index.append(ref)
                    demo_chunks.append(ref)

                for j in range(len(demo_chunks) - 1):
                    self.pair_index.append((demo_chunks[j], demo_chunks[j + 1]))

        print(
            f"[LiberoTwoStepDataset] Built step index: {len(self.index)} chunks, "
            f"{len(self.pair_index)} transition pairs (stride={self.step}, horizon={self.step})."
        )

    def __len__(self):
        return len(self.pair_index)

    def _read_one_ref(self, ref, prefix=""):
        hdf5_path, demo_key, t0 = ref

        out = {"t0": t0, "hdf5_path": hdf5_path, "demo_key": demo_key}

        with h5py.File(hdf5_path, "r") as f:
            demo = f["data"][demo_key]

            obs = {}
            for k in demo["obs"].keys():
                obs[k] = demo["obs"][k][t0]
            out["obs"] = obs

            actions = demo["actions"][t0 : t0 + self.step]
            out["actions"] = actions

            T = demo["actions"].shape[0]
            out["chunk_dones"] = int(t0 + 2 * self.step - 1 >= T)
            for key in ["dones", "rewards", "robot_states", "states"]:
                if key in demo:
                    out[key] = demo[key][t0]
                else:
                    out[key] = None

        for k in list(out.keys()):
            out[prefix + k] = out.pop(k)

        return out

    def _read_one(self, idx, prefix=""):
        return self._read_one_ref(self.index[idx], prefix=prefix)

    def __getitem__(self, idx):
        t0_ref, t1_ref = self.pair_index[idx]
        out = {}
        out.update(self._read_one_ref(t0_ref, prefix="t0."))
        out.update(self._read_one_ref(t1_ref, prefix="t1."))
        return out


def make_dataset(repo_id: str, root: str) -> LiberoTwoStepDataset:
    dataset = LiberoTwoStepDataset(root)
    return dataset


def make_sampler(dataset: LiberoTwoStepDataset):
    return None


def make_collator() -> Callable:
    return libero_collate_fn
