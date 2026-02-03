import torch
import numpy as np

def _to_tensor(x, kind="float"):
    """
    kind:
      - "float": cast to float32
      - "int": cast to int64
      - "byte": cast to uint8
    """
    if x is None:
        return None

    if isinstance(x, (np.generic,)):
        x = x.item()

    if isinstance(x, torch.Tensor):
        t = x
    elif isinstance(x, np.ndarray):
        t = torch.from_numpy(x)
    elif isinstance(x, (float, int, bool)):
        t = torch.tensor(x)
    else:
        t = torch.tensor(x)

    if kind == "float":
        t = t.to(torch.float32)
    elif kind == "int":
        t = t.to(torch.int64)
    elif kind == "byte":
        t = t.to(torch.uint8)

    if t.dim() == 0:
        t = t.view(1)

    return t


import torch
import numpy as np

def libero_collate_fn(batch):
    out = {}

    # obs
    obs_keys = batch[0]["obs"].keys()
    for k in sorted(obs_keys):
        vals = [b["obs"].get(k, None) for b in batch]

        v0 = vals[0]
        kind = "byte" if isinstance(v0, np.ndarray) and v0.dtype == np.uint8 else "float"
        out["obs." + k] = torch.stack([_to_tensor(v, kind=kind) for v in vals], dim=0)

    # actions (B, N_ACTIONS_STEPS, A)
    out["actions"] = torch.stack([_to_tensor(b["actions"], kind="float") for b in batch], dim=0)

    # rewards/dones
    for key in ["dones", "rewards"]:
        vals = [b[key] for b in batch]
        out[key] = torch.stack([_to_tensor(v, kind="int") for v in vals], dim=0)
        out[key] = out[key].squeeze(-1)

    # robot_states, ignore states
    for key in ["robot_states"]:
        vals = [b[key] for b in batch]
        out[key] = torch.stack([_to_tensor(v, kind="float") for v in vals], dim=0)

    # metadata
    out["t0"] = np.array([b["t0"] for b in batch])
    out["hdf5_path"] = np.array([b["hdf5_path"] for b in batch])
    out["demo_key"] = np.array([b["demo_key"] for b in batch])

    return out
