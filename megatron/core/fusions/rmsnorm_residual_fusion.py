# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Cross-layer RMSNorm + residual-add fusion protocol.

A layer can skip its exit residual-add (the ``bda`` step in
``Transformer``/``Mamba`` layers) and instead hand the pair
``(residual, delta)`` to the next layer as a :class:`DeferredAdd`.
The next layer folds that add into its own entry RMSNorm via
:func:`~megatron.core.fusions.fused_add_rmsnorm.fused_add_rmsnorm`,
collapsing a residual-add plus an RMSNorm into one Triton kernel and
saving one global-memory round-trip per layer boundary.

This is *cross-layer* fusion and should not be confused with TE's
intra-module ``fused_residual_rmsnorm``.

Pieces:

* :class:`DeferredAdd` -- container carrying ``(residual, delta)``
  across a boundary.
* :class:`AbsorbMode` / :class:`DeferMode` -- per-layer enums stamped
  by :func:`wire_rmsnorm_residual_fusion` at build time. They identify
  *where* in the layer the fusion attaches (attention entry, pre-MLP
  entry, Mamba entry, attention exit, MLP exit, Mamba exit).
* :func:`should_absorb_add` / :func:`should_defer_add` -- runtime
  predicates that gate the wiring behind training / grad-enabled /
  fp32-residual checks so the unfused fallback remains correct.

The kernel lives in
:mod:`~megatron.core.fusions.fused_add_rmsnorm`; this module is the
control-plane that decides which boundaries call it.
"""

import os
from dataclasses import dataclass
from enum import Enum
from typing import Iterable

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
    ATTN_ENTRY = 1  # TransformerLayer: fuse into linear_qkv's entry norm
    PRE_MLP_ENTRY = 2  # TransformerLayer (no self-attn): fuse into pre_mlp_layernorm
    MAMBA_ENTRY = 3  # MambaLayer: fuse into mixer.in_proj (or standalone norm)


class DeferMode(Enum):
    """Where a layer skips its exit residual-add and returns a ``DeferredAdd``."""

    NONE = 0
    ATTN_EXIT = 1  # TransformerLayer attn-only: skip self_attn_bda
    MLP_EXIT = 2  # TransformerLayer: skip mlp_bda
    MAMBA_EXIT = 3  # MambaLayer: skip mamba_bda


def rmsnorm_fusion_disabled_by_env() -> bool:
    """True iff the env var asks us to skip cross-layer RMSNorm fusion."""
    return bool(os.environ.get(_DISABLE_ENV_VAR))


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


def wire_rmsnorm_residual_fusion(layers: Iterable) -> int:
    """Enable cross-layer RMSNorm + residual-add fusion on every compatible pair.

    Each layer exposes ``native_fusion_modes()`` returning
    ``(AbsorbMode, DeferMode)`` -- the modes it would use if paired with
    a compatible neighbour (or ``NONE`` if it cannot participate). For
    each adjacent ``(prev, nxt)`` pair where ``prev`` can defer and
    ``nxt`` can absorb, we stamp the modes onto the layers so their
    forward paths dispatch without re-deriving the decision.

    Returns the number of boundaries wired (``0`` when disabled via env
    var or when Triton is unavailable).
    """
    from megatron.core.fusions.fused_add_rmsnorm import HAVE_TRITON

    if not HAVE_TRITON or rmsnorm_fusion_disabled_by_env():
        return 0
    layers = list(layers)
    wired = 0
    none = (AbsorbMode.NONE, DeferMode.NONE)
    for prev, nxt in zip(layers, layers[1:]):
        _, prev_defer = getattr(prev, "native_fusion_modes", lambda: none)()
        nxt_absorb, _ = getattr(nxt, "native_fusion_modes", lambda: none)()
        if prev_defer is not DeferMode.NONE and nxt_absorb is not AbsorbMode.NONE:
            prev._defer_mode = prev_defer
            nxt._absorb_mode = nxt_absorb
            wired += 1
    return wired
