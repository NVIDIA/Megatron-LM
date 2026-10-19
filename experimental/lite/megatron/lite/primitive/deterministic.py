# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Deterministic computation utilities for bitwise reproducible experiments."""

from __future__ import annotations

import os
import sys

import torch

_is_deterministic: bool = False
_TRUE_VALUES = {"1", "true", "yes", "on"}
_DETERMINISTIC_ENV = "MEGATRON_LITE_DETERMINISTIC"


def deterministic_requested() -> bool:
    """Return whether bitwise-validation deterministic mode was requested."""
    return os.environ.get(_DETERMINISTIC_ENV, "").strip().lower() in _TRUE_VALUES


def set_deterministic(seed: int = 42) -> None:
    """Enable all deterministic flags. Must call before first CUDA op in process."""
    global _is_deterministic
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "0"
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    disable_jit_fuser()
    _is_deterministic = True


def disable_jit_fuser() -> None:
    """Run jit_fuser (torch.compile) kernels eagerly, as MMF's ``disable_jit_fuser=True``."""
    os.environ["MEGATRON_LITE_DISABLE_JIT_FUSER"] = "1"
    from megatron.lite.primitive.kernels import jit as lite_jit

    lite_jit.jit_fuser = lite_jit.noop_decorator
    try:
        from megatron.core import jit as core_jit
    except ModuleNotFoundError:
        pass
    else:
        core_jit.disable_jit_fuser()
    # Already-imported modules hold compiled wrappers; swap in the original functions.
    for name, module in list(sys.modules.items()):
        if module is None or not name.startswith("megatron."):
            continue
        for attr, value in list(vars(module).items()):
            orig = getattr(value, "_torchdynamo_orig_callable", None)
            if callable(orig) and callable(value):
                setattr(module, attr, orig)


def is_deterministic() -> bool:
    return _is_deterministic
