from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .collator import collate_fn


def _to_float_tensor(x: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32)


def _to_uint8_tensor(x: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.uint8)


def _parse_task_from_path(path: str) -> str:
    name = os.path.basename(path)
    if name.endswith("_demo.hdf5"):
        name = name[: -len("_demo.hdf5")]
    name = re.sub(r"^[A-Z_]+_SCENE\d+_", "", name)
    return name.replace("_", " ").lower()


@dataclass(frozen=True)
class _TransitionRef:
    file_path: str
    demo_key: str
    idx0: int
    idx1: int
    action_horizon: int


class LiberoSequenceDataset(Dataset):
    """
    Raw LIBERO HDF5 -> sequence dataset for current RLPD pipeline.

    One item returns [t0_dict, t1_dict], where:
      - t0 is frame idx0
      - t1 is frame min(idx0 + H, terminal_idx)
      - actions are fixed-length action chunks starting at idx0 / idx1
      - chunk_dones is True iff the transition reaches episode terminal within horizon

    This matches the fields consumed by:
      verl.experimental.vla.models.pi0_torch.datasets.libero_dataset.LiberoPi0DatasetInput
    """

    def __init__(self, refs: list[_TransitionRef]):
        self.refs = refs
        self._file_cache: dict[str, h5py.File] = {}

    def __len__(self) -> int:
        return len(self.refs)

    def _get_file(self, path: str) -> h5py.File:
        f = self._file_cache.get(path)
        if f is None:
            f = h5py.File(path, "r")
            self._file_cache[path] = f
        return f

    def _read_step(self, demo: h5py.Group, idx: int, hdf5_path: str, action_horizon: int) -> dict[str, Any]:
        obs = demo["obs"]
        actions = np.asarray(demo["actions"])
        dones = np.asarray(demo["dones"])
        rewards = np.asarray(demo["rewards"]) if "rewards" in demo else None

        terminal_idx = actions.shape[0] - 1
        chunk_end = min(idx + action_horizon, terminal_idx)
        chunk_done = bool(chunk_end == terminal_idx)

        action_chunk = actions[idx : min(idx + action_horizon, actions.shape[0])]
        if action_chunk.shape[0] == 0:
            raise IndexError(f"Empty action chunk at idx={idx} for {hdf5_path}")

        # Fixed horizon is required for torch.stack in collator.
        if action_chunk.shape[0] < action_horizon:
            pad = np.repeat(action_chunk[-1:], action_horizon - action_chunk.shape[0], axis=0)
            action_chunk = np.concatenate([action_chunk, pad], axis=0)

        out = {
            "actions": _to_float_tensor(action_chunk),                     # (H, 7)
            "chunk_dones": torch.tensor(chunk_done, dtype=torch.bool),
            "dones": torch.tensor(bool(dones[idx]), dtype=torch.bool),
            "obs.agentview_rgb": _to_uint8_tensor(np.asarray(obs["agentview_rgb"][idx])),
            "obs.eye_in_hand_rgb": _to_uint8_tensor(np.asarray(obs["eye_in_hand_rgb"][idx])),
            "obs.ee_pos": _to_float_tensor(np.asarray(obs["ee_pos"][idx])),
            "obs.ee_ori": _to_float_tensor(np.asarray(obs["ee_ori"][idx])),
            "obs.gripper_states": _to_float_tensor(np.asarray(obs["gripper_states"][idx])),
            "hdf5_path": hdf5_path,
            "task": _parse_task_from_path(hdf5_path),
        }

        # Optional extras kept for debugging / future use.
        if "joint_states" in obs:
            out["obs.joint_states"] = _to_float_tensor(np.asarray(obs["joint_states"][idx]))
        if "ee_states" in obs:
            out["obs.ee_states"] = _to_float_tensor(np.asarray(obs["ee_states"][idx]))
        if "robot_states" in demo:
            out["robot_states"] = _to_float_tensor(np.asarray(demo["robot_states"][idx]))
        if "states" in demo:
            out["states"] = _to_float_tensor(np.asarray(demo["states"][idx]))
        if rewards is not None:
            out["rewards"] = torch.tensor(float(rewards[idx]), dtype=torch.float32)

        return out

    def __getitem__(self, i: int):
        ref = self.refs[i]
        f = self._get_file(ref.file_path)
        demo = f["data"][ref.demo_key]
        sample0 = self._read_step(demo, ref.idx0, ref.file_path, ref.action_horizon)
        sample1 = self._read_step(demo, ref.idx1, ref.file_path, ref.action_horizon)

        # Keep chunk terminal semantics aligned across the pair.
        pair_chunk_done = torch.tensor(ref.idx1 >= (demo["actions"].shape[0] - 1), dtype=torch.bool)
        sample0["chunk_dones"] = pair_chunk_done
        sample1["chunk_dones"] = pair_chunk_done
        return [sample0, sample1]



def _resolve_hdf5_files(root: str) -> list[str]:
    if os.path.isfile(root):
        if root.endswith(".hdf5"):
            return [root]
        raise ValueError(f"Expected .hdf5 file, got: {root}")

    if not os.path.isdir(root):
        raise FileNotFoundError(f"RLPD root not found: {root}")

    paths = [
        os.path.join(root, name)
        for name in sorted(os.listdir(root))
        if name.endswith(".hdf5")
    ]
    if not paths:
        raise FileNotFoundError(f"No .hdf5 files found under: {root}")
    return paths


def _build_refs_from_file(path: str, action_horizon: int) -> list[_TransitionRef]:
    refs: list[_TransitionRef] = []
    with h5py.File(path, "r") as f:
        if "data" not in f:
            raise KeyError(f"Missing top-level group 'data' in {path}")

        for demo_key in sorted(f["data"].keys()):
            demo = f["data"][demo_key]
            if "actions" not in demo or "obs" not in demo:
                raise KeyError(f"{path}:{demo_key} missing required keys 'actions' / 'obs'")

            obs = demo["obs"]
            required_obs = [
                "agentview_rgb",
                "eye_in_hand_rgb",
                "ee_pos",
                "ee_ori",
                "gripper_states",
            ]
            for key in required_obs:
                if key not in obs:
                    raise KeyError(f"{path}:{demo_key}/obs missing required key '{key}'")

            T = int(demo["actions"].shape[0])
            if T < 2:
                continue

            terminal_idx = T - 1
            for idx0 in range(0, terminal_idx):
                idx1 = min(idx0 + action_horizon, terminal_idx)
                refs.append(
                    _TransitionRef(
                        file_path=path,
                        demo_key=demo_key,
                        idx0=idx0,
                        idx1=idx1,
                        action_horizon=action_horizon,
                    )
                )
    return refs


def make_dataset(repo_id: str, root: str) -> LiberoSequenceDataset:
    del repo_id  # unused for raw local HDF5 loader

    action_horizon = int(os.environ.get("LIBERO_RLPD_ACTION_HORIZON", "10"))
    hdf5_files = _resolve_hdf5_files(root)

    refs: list[_TransitionRef] = []
    for path in hdf5_files:
        refs.extend(_build_refs_from_file(path, action_horizon=action_horizon))

    if not refs:
        raise RuntimeError(f"No valid transitions built from raw LIBERO HDF5 under: {root}")

    return LiberoSequenceDataset(refs)


def make_sampler(dataset: Dataset):
    del dataset
    return None


def make_collator():
    return collate_fn
