# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Opt-in L1 observations of TransformerLayer boundaries used with TE.

Metadata still incurs Python, locking, and serialization overhead. CUDA graph
capture can emit records, but replay does not rerun Python observers or hooks;
these records cannot certify replay coverage. Value-bearing capture retains the
recorder's CUDA graph restrictions. No distributed collectives are introduced.
"""

from contextvars import ContextVar
from dataclasses import replace
from functools import wraps

import torch

from .context import determinism_context, get_determinism_context
from .trace import get_determinism_trace

_REGISTER_BACKWARD_HOOKS = ContextVar("te_trace_register_backward_hooks", default=True)


def bind_layer_trace_context(function):
    """Restore checkpoint identity and avoid hooks on non-reentrant recompute."""
    snapshot = get_determinism_context()
    first_grad_enabled = None

    @wraps(function)
    def wrapped(*args, **kwargs):
        nonlocal first_grad_enabled
        recomputing = first_grad_enabled is not None
        if not recomputing:
            first_grad_enabled = torch.is_grad_enabled()
        register = _REGISTER_BACKWARD_HOOKS.get() and not (recomputing and first_grad_enabled)
        token = _REGISTER_BACKWARD_HOOKS.set(register)
        try:
            with determinism_context(snapshot):
                return function(*args, **kwargs)
        finally:
            _REGISTER_BACKWARD_HOOKS.reset(token)

    return wrapped


def trace_transformer_layer_forward(forward):
    """Observe hidden states and their gradients without retaining activations."""

    @wraps(forward)
    def wrapped(self, hidden_states, /, *args, **kwargs):
        if not self.config.determinism_trace_layers:
            return forward(self, hidden_states, *args, **kwargs)
        trace = get_determinism_trace()
        if trace is None:
            return forward(self, hidden_states, *args, **kwargs)

        snapshot = replace(get_determinism_context(), phase="forward")
        metadata = {
            "boundary_level": "L1",
            "layer_number": self.layer_number,
            "missing_context": [
                name for name in ("iteration", "microbatch") if getattr(snapshot, name) is None
            ],
            "cuda_graph_replay_observed": False,
        }

        def observe(tensor, boundary):
            name = f"te.layer.{self.layer_number}.{boundary}"
            trace.record_tensor(
                name,
                tensor,
                iteration=snapshot.iteration,
                microbatch=snapshot.microbatch,
                phase="forward",
                metadata=metadata,
            )
            if _REGISTER_BACKWARD_HOOKS.get() and torch.is_grad_enabled() and tensor.requires_grad:

                def backward(gradient):
                    with determinism_context(replace(snapshot, phase="backward")):
                        trace.record_tensor(
                            name,
                            gradient,
                            iteration=snapshot.iteration,
                            microbatch=snapshot.microbatch,
                            phase="backward",
                            metadata=metadata,
                        )
                    return gradient

                tensor.register_hook(backward)

        with determinism_context(snapshot):
            observe(hidden_states, "in")
            output, context = forward(self, hidden_states, *args, **kwargs)
            observe(output, "out")
        return output, context

    return wrapped
