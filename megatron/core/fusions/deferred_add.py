# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Cross-layer deferred-add protocol.

A layer can skip its exit residual-add and instead hand the pair
``(residual, delta)`` to the next layer as a :class:`DeferredAdd`. The
next layer then folds that add into its own entry RMSNorm via
:func:`~megatron.core.fusions.fused_add_rmsnorm.fused_add_rmsnorm`,
collapsing a residual-add plus an RMSNorm into one Triton kernel.

The protocol has three moving parts:

* :class:`DeferredAdd` -- the container carrying ``(residual, delta)``
  across a layer boundary.
* :class:`AbsorbMode` / :class:`DeferMode` -- per-layer enums recorded
  by :func:`wire_add_fusion` at build time. They identify *where* in
  the layer the fusion attaches (attention entry, pre-MLP entry, Mamba
  entry, attention exit, MLP exit, Mamba exit).
* :func:`should_absorb_add` / :func:`should_defer_add` -- runtime
  predicates that gate the wiring behind training / grad-enabled /
  fp32-residual checks so fallback remains correct.

See :func:`~megatron.core.fusions.fused_add_rmsnorm.fused_add_rmsnorm`
for the kernel the absorb side calls.
"""

import os
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Union

import torch

_DISABLE_ENV_VAR = "MEGATRON_DISABLE_CROSS_LAYER_ADD_FUSION"


@dataclass
class DeferredAdd:
    """A residual-add deferred from one layer to the next.

    When a layer defers its exit residual-add, it returns one of these
    instead of a plain ``Tensor``. The next layer runs
    ``fused_add_rmsnorm(delta, residual, entry_norm_weight)`` as its
    entry step, collapsing the deferred add and its own input RMSNorm
    into one Triton kernel.

    The class is *iterable* (yielding ``residual`` then ``delta``) so
    the per-layer CUDA-graph machinery in ``cuda_graphs.py`` can walk
    tensor containers. It is **not** a ``tuple`` subclass, so
    ``isinstance(x, tuple)`` checks used elsewhere (e.g. to unwrap
    ``TransformerLayer``'s ``(output, context)`` return) correctly skip
    over ``DeferredAdd`` instances.
    """

    residual: "torch.Tensor"
    delta: "torch.Tensor"

    def __iter__(self):
        yield self.residual
        yield self.delta

    def __len__(self):
        return 2

    def __getitem__(self, i):
        # Indexing support for CUDA graph buffer-reuse paths in
        # ``cuda_graphs.py`` that sometimes do
        # ``previous_runner.fwd_graph_outputs[0]`` on a layer's return value.
        return (self.residual, self.delta)[i]


class AbsorbMode(Enum):
    """Where a layer folds an incoming ``DeferredAdd`` into its entry norm."""

    NONE = 0
    ATTN_ENTRY = 1      # TransformerLayer: fuse into linear_qkv's entry norm
    PRE_MLP_ENTRY = 2   # TransformerLayer (no self-attn): fuse into pre_mlp_layernorm
    MAMBA_ENTRY = 3     # MambaLayer: fuse into mixer.in_proj (or standalone norm)


class DeferMode(Enum):
    """Where a layer skips its exit residual-add and returns a ``DeferredAdd``."""

    NONE = 0
    ATTN_EXIT = 1       # TransformerLayer attn-only: skip self_attn_bda
    MLP_EXIT = 2        # TransformerLayer: skip mlp_bda
    MAMBA_EXIT = 3      # MambaLayer: skip mamba_bda


def add_fusion_disabled_by_env() -> bool:
    """True iff the env var asks us to skip deferred-add fusion."""
    return bool(os.environ.get(_DISABLE_ENV_VAR))


def add_fusion_enabled() -> bool:
    """True iff deferred-add fusion is globally available (Triton + env)."""
    from megatron.core.fusions.fused_add_rmsnorm import HAVE_TRITON

    return HAVE_TRITON and not add_fusion_disabled_by_env()


def materialize(x: Union["torch.Tensor", DeferredAdd]) -> "torch.Tensor":
    """Return a plain tensor, collapsing a ``DeferredAdd`` via
    ``residual + delta`` if needed.

    Layer-stack code that does not participate in the protocol can call
    this to keep the deferred-add object from leaking past the stack.
    """
    if isinstance(x, DeferredAdd):
        return x.residual + x.delta
    return x


def should_defer_add(layer) -> bool:
    """Should ``layer`` defer its exit residual-add this call?

    Combines the wiring-time ``_defer_mode`` with the shared runtime
    gates (training / grad-enabled / fp32 residual).
    """
    return (
        getattr(layer, "_defer_mode", DeferMode.NONE) is not DeferMode.NONE
        and not layer.training
        and not torch.is_grad_enabled()
        and not layer.config.fp32_residual_connection
    )


def should_absorb_add(layer, x) -> bool:
    """Should ``layer`` absorb ``x`` as an incoming :class:`DeferredAdd`?"""
    return (
        isinstance(x, DeferredAdd)
        and getattr(layer, "_absorb_mode", AbsorbMode.NONE) is not AbsorbMode.NONE
        and not layer.training
        and not torch.is_grad_enabled()
        and not layer.config.fp32_residual_connection
    )


def wire_add_fusion(layers: Iterable) -> int:
    """Enable deferred-add fusion on every compatible adjacent pair.

    Each layer exposes ``native_absorb_mode()`` / ``native_defer_mode()``
    returning the ``AbsorbMode`` / ``DeferMode`` it would use if asked
    (or ``NONE`` if it cannot). For each adjacent ``(prev, nxt)`` pair
    where both sides advertise a non-``NONE`` mode, we copy those modes
    onto ``prev._defer_mode`` and ``nxt._absorb_mode`` so the forward
    paths can dispatch without re-deriving the decision.

    Returns the number of boundaries wired (``0`` when disabled).
    """
    if not add_fusion_enabled():
        return 0
    layers = list(layers)
    wired = 0
    for prev, nxt in zip(layers, layers[1:]):
        d = getattr(prev, "native_defer_mode", lambda: DeferMode.NONE)()
        a = getattr(nxt, "native_absorb_mode", lambda: AbsorbMode.NONE)()
        if d is not DeferMode.NONE and a is not AbsorbMode.NONE:
            prev._defer_mode = d
            nxt._absorb_mode = a
            wired += 1
    return wired
