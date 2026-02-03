import os
import glob
import h5py
import numpy as np
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

        print(f"[LiberoTrajDataset] Found {len(self.hdf5_files)} task files, "
              f"{len(self.samples)} demos total.")

    def __len__(self):
        return len(self.samples)

    def get_demo_ref(self, idx):
        return self.samples[idx]


class LiberoStepDataset(Dataset):
    """
    dataset_keys = ['obs', 'actions', 'dones', 'rewards', 'robot_states', 'states']
    """

    def __init__(self, root_dir, step=N_ACTIONS_STEPS, build_index=True):
        self.root_dir = root_dir
        self.step = int(step)
        self.traj_ds = LiberoTrajDataset(root_dir)
        self.index = []  # list of (hdf5_path, demo_key, t0)

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
                for t0 in range(0, max_t0 + 1, self.step):
                    self.index.append((hdf5_path, demo_key, t0))

        print(f"[LiberoStepDataset] Built step index: {len(self.index)} samples "
              f"(stride={self.step}, horizon={self.step}).")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        hdf5_path, demo_key, t0 = self.index[idx]

        out = {"t0": t0, "hdf5_path": hdf5_path, "demo_key": demo_key}

        with h5py.File(hdf5_path, "r") as f:
            demo = f["data"][demo_key]

            # obs: all fields, only take first step t0
            obs = {}
            for k in demo["obs"].keys():
                obs[k] = demo["obs"][k][t0]
            out["obs"] = obs

            # dataset keys
            # actions: take horizon [t0:t0+step]
            actions = demo["actions"][t0: t0 + self.step]
            out["actions"] = actions

            # others: only take t0
            for key in ["dones", "rewards", "robot_states", "states"]:
                if key in demo:
                    out[key] = demo[key][t0]
                else:
                    out[key] = None

        return out

    
def make_dataset(repo_id: str, root: str) -> LiberoStepDataset:
    dataset = LiberoStepDataset(root)
    return dataset

def make_sampler(dataset: LiberoStepDataset):
    return None

def make_collator() -> callable:
    return libero_collate_fn 

