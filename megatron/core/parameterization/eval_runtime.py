from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator


_DEPTH_MUP_EVAL_DEPTH: ContextVar[int] = ContextVar('depth_mup_eval_depth', default=0)


def is_depth_mup_eval_enabled() -> bool:
    return _DEPTH_MUP_EVAL_DEPTH.get() > 0


@contextmanager
def depth_mup_eval_context(enabled: bool) -> Iterator[None]:
    if not enabled:
        yield
        return

    token = _DEPTH_MUP_EVAL_DEPTH.set(_DEPTH_MUP_EVAL_DEPTH.get() + 1)
    try:
        yield
    finally:
        _DEPTH_MUP_EVAL_DEPTH.reset(token)
