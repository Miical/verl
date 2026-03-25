import os
import pickle
import struct
import time
from pathlib import Path
from typing import Any


def _ipc_paths(rank: int, stage_id: int, host: str | None = None) -> tuple[str, str]:
    if host is None:
        host = os.uname().nodename
    base = f"/tmp/lerobot_ipc_{host}_rank{rank}_stage{stage_id}"
    return f"{base}.req.fifo", f"{base}.resp.fifo"


def _ensure_fifo(path: str) -> None:
    p = Path(path)
    if p.exists():
        p.unlink()
    os.mkfifo(p)


def _remove_fifo(path: str) -> None:
    p = Path(path)
    if p.exists():
        p.unlink()


def _send_raw_obj(path: str, obj: Any) -> None:
    payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    header = struct.pack("!I", len(payload))
    with open(path, "wb", buffering=0) as fifo:
        fifo.write(header)
        fifo.write(payload)
        fifo.flush()


def _recv_raw_obj(path: str) -> Any:
    with open(path, "rb", buffering=0) as fifo:
        header = fifo.read(4)
        if len(header) != 4:
            raise RuntimeError("Failed to read object header from FIFO")
        size = struct.unpack("!I", header)[0]
        payload = fifo.read(size)
        if len(payload) != size:
            raise RuntimeError("Failed to read object payload from FIFO")
        return pickle.loads(payload)


def setup_ipc(rank: int, stage_id: int) -> None:
    req_path, resp_path = _ipc_paths(rank=rank, stage_id=stage_id)
    _ensure_fifo(req_path)
    _ensure_fifo(resp_path)


def clear_ipc(rank: int, stage_id: int) -> None:
    req_path, resp_path = _ipc_paths(rank=rank, stage_id=stage_id)
    _remove_fifo(req_path)
    _remove_fifo(resp_path)


def send_obj(type: str, content: Any, rank: int, stage_id: int, timeout_s: float = 60.0) -> Any:
    req_path, resp_path = _ipc_paths(rank=rank, stage_id=stage_id)
    deadline = time.time() + timeout_s
    while not (Path(req_path).exists() and Path(resp_path).exists()):
        if time.time() >= deadline:
            raise TimeoutError(f"IPC FIFO is not ready: req={req_path}, resp={resp_path}")
        time.sleep(0.05)

    _send_raw_obj(req_path, {"type": type, "content": content})
    return _recv_raw_obj(resp_path)


def recv_obj(rank: int, stage_id: int) -> Any:
    req_path, _ = _ipc_paths(rank=rank, stage_id=stage_id)
    return _recv_raw_obj(req_path)


def reply_obj(content: Any, rank: int, stage_id: int) -> None:
    _, resp_path = _ipc_paths(rank=rank, stage_id=stage_id)
    _send_raw_obj(resp_path, content)
