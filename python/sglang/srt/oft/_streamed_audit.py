"""Env-gated audit hook for SGLang's OFT streamed weight loader.

Set ``SGLANG_OFT_STREAMED_AUDIT=<base_path>`` to record every dense
fused-buffer write and every expert-tensor partition event. Each rank writes
to its own file ``<base_path>.sglang_streamed.tp{T}.jsonl``; lines are

    {"time": <unix>, "tp_rank": <int>, "event": "dense_write"|"expert_partition", ...}

Debug-only: with the env var set, every recorded write does a CUDA->CPU sync
(`.abs().max().item()` and friends) which adds non-trivial per-sync latency.
Do not enable in production. Production runs without the env var pay nothing.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import torch


_LOCK = threading.Lock()


def _base() -> str | None:
    value = os.getenv("SGLANG_OFT_STREAMED_AUDIT", "").strip()
    return value or None


def _distributed_rank() -> int:
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return int(torch.distributed.get_rank())
    except Exception:
        pass
    return 0


def _path_for_rank(base: str, tp_rank: int) -> Path:
    return Path(f"{base}.sglang_streamed.tp{tp_rank}.jsonl")


def enabled() -> bool:
    return _base() is not None


def _tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    detached = tensor.detach()
    if detached.numel() == 0:
        return {
            "shape": list(detached.shape),
            "dtype": str(detached.dtype),
            "device": str(detached.device),
            "max_abs": 0.0,
            "mean_abs": 0.0,
            "nonzero": 0,
        }
    as_float = detached.float()
    return {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "max_abs": float(as_float.abs().max().item()),
        "mean_abs": float(as_float.abs().mean().item()),
        "nonzero": int((detached != 0).sum().item()),
    }


def record(event: str, *, tp_rank: int | None = None, **fields: Any) -> None:
    base = _base()
    if base is None:
        return
    rank = int(_distributed_rank() if tp_rank is None else tp_rank)
    payload = {"time": time.time(), "tp_rank": rank, "event": event, **fields}
    path = _path_for_rank(base, rank)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, sort_keys=True)
    with _LOCK:
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def record_dense_write(
    fused_target: str,
    layer_id: int,
    r: torch.Tensor,
    *,
    tp_rank: int | None = None,
) -> None:
    record(
        "dense_write",
        tp_rank=tp_rank,
        fused_target=fused_target,
        layer_id=int(layer_id),
        tensor=_tensor_summary(r),
    )


def record_expert_partition(
    layer_id: int,
    expert_id: int,
    proj: str,
    tensor: torch.Tensor,
    *,
    tp_rank: int | None = None,
    num_local_experts: int | None = None,
) -> None:
    record(
        "expert_partition",
        tp_rank=tp_rank,
        layer_id=int(layer_id),
        expert_id=int(expert_id),
        proj=proj,
        num_local_experts=(None if num_local_experts is None else int(num_local_experts)),
        tensor=_tensor_summary(tensor),
    )
