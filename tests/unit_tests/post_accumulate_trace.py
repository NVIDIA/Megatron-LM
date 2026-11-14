# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Keep coverage tracers visible inside CUDA post-accumulate grad hooks.

``Tensor.register_post_accumulate_grad_hook`` callbacks that the CUDA autograd
engine invokes run with ``sys.gettrace() is None``. coverage.py then leaves the
hook body unmarked, and pytest-testmon does not select the CUDA tests that
execute it. See https://github.com/NVIDIA/Megatron-LM/issues/5633.

Registration happens on the test thread, where coverage is already tracing.
The wrapper captures that tracer and installs it for the hook call when the
autograd thread has none.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any, TypeVar

import torch
from torch.overrides import handle_torch_function, has_torch_function_unary

_Hook = TypeVar("_Hook", bound=Callable[..., Any])
_INSTALLED = False
_ORIGINAL = torch.Tensor.register_post_accumulate_grad_hook


def preserve_caller_trace(hook: _Hook) -> _Hook:
    """Return ``hook`` so a traceless caller still runs it under the registering tracer.

    Hooks registered while no tracer is active are returned unchanged, so runs
    that are not collecting coverage pay only for an identity check at registration.

    Args:
        hook: Post-accumulate grad hook. It is called as ``hook(parameter)``.

    Returns:
        The original hook, or a wrapper that reinstalls the captured tracer.
    """
    tracer = sys.gettrace()
    if tracer is None:
        return hook

    def wrapped(parameter: torch.Tensor) -> Any:
        previous = sys.gettrace()
        if previous is not None:
            return hook(parameter)
        sys.settrace(tracer)
        try:
            return hook(parameter)
        finally:
            # The autograd thread had no tracer. Leave it that way so later
            # engine callbacks on this thread are not attributed to this test.
            sys.settrace(previous)

    return wrapped  # type: ignore[return-value]


def install_post_accumulate_trace() -> None:
    """Patch tensor hook registration so unit tests trace CUDA hook bodies.

    The patch is process-global and idempotent. It wraps the hook before the
    original implementation stores it, including the ``__torch_function__``
    path used by tensor subclasses.
    """
    global _INSTALLED
    if _INSTALLED:
        return

    def register(self: torch.Tensor, hook: Callable[..., Any]) -> Any:
        if has_torch_function_unary(self):
            return handle_torch_function(register, (self,), self, hook)
        return _ORIGINAL(self, preserve_caller_trace(hook))

    torch.Tensor.register_post_accumulate_grad_hook = register
    _INSTALLED = True
