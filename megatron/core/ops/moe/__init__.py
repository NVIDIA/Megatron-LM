# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Mixture of experts: what a backend must provide, and which ones exist.

The implementations are in :mod:`.backends`; a provider in
``megatron.core.models.backends`` or ``megatron.core.extensions`` picks between them.

**Contract**

* **Targets**: ``grouped_mlp_modules`` returns an experts builder called with the local expert
  count and config; ``activation_func`` returns a builder for an activation *module*, or
  ``None`` to fall back to ``config.activation_func``; ``moe_router`` returns a router class,
  or ``None`` to keep the ``MoESubmodules`` default.
* **State**: the experts own their own weights. The router owns its gate. Neither owns the
  dispatcher's buffers.
* **Process groups**: the dispatcher, not these targets, owns expert- and tensor-parallel
  communication; it receives its groups from the MoE layer.
* **Modes**: ``moe_use_grouped_gemm`` is a construction-time input, so a backend picks its
  grouped or sequential experts once rather than branching per step. The inference backend
  needs compact ``[tokens, topk]`` index routing, which is why the router is a slot at all.
"""

from __future__ import annotations

from megatron.core.ops.moe.backends import MoeLocal, MoeTE

__all__ = ["MoeLocal", "MoeTE"]
