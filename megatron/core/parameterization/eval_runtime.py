# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

_SCALING_POLICY_EVAL_DEPTH: ContextVar[int] = ContextVar(
    'scaling_policy_eval_depth', default=0
)


def is_scaling_policy_eval_allowed() -> bool:
    return _SCALING_POLICY_EVAL_DEPTH.get() > 0


@contextmanager
def allow_scaling_policy_eval(enabled: bool) -> Iterator[None]:
    if not enabled:
        yield
        return

    token = _SCALING_POLICY_EVAL_DEPTH.set(_SCALING_POLICY_EVAL_DEPTH.get() + 1)
    try:
        yield
    finally:
        _SCALING_POLICY_EVAL_DEPTH.reset(token)
