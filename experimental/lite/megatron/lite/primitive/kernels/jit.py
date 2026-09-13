# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Local JIT/compile decorator shim for small fused Python kernels."""

from __future__ import annotations

import os

import torch


def noop_decorator(func):
    return func


def _build_jit_fuser():
    if os.environ.get("MEGATRON_LITE_DISABLE_JIT_FUSER", "0") == "1":
        return noop_decorator
    compile_fn = getattr(torch, "compile", None)
    if compile_fn is not None:
        return compile_fn
    return torch.jit.script


jit_fuser = _build_jit_fuser()


__all__ = ["jit_fuser", "noop_decorator"]
