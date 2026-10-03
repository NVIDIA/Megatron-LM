"""Deterministic computation utilities for bitwise reproducible experiments."""
from __future__ import annotations

import os

import torch

_is_deterministic: bool = False


def set_deterministic(seed: int = 42) -> None:
    """Enable all deterministic flags. Must call before first CUDA op in process."""
    global _is_deterministic
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    _is_deterministic = True


def is_deterministic() -> bool:
    return _is_deterministic
