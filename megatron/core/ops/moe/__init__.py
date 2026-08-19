# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Mixture of experts: what a backend must provide, and which ones exist.

The implementations are in :mod:`.backends`.

Contract
--------
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

from typing import TYPE_CHECKING, Optional

from megatron.core.inference.ops.backends import MoeInference
from megatron.core.ops.moe.backends import MoeLocal, MoeTE
from megatron.core.ops.operations import Operation, unowned

if TYPE_CHECKING:
    from megatron.core.transformer.mlp import TEActivationFunctionBuilder
    from megatron.core.transformer.moe.moe_layer import ExpertsBuilder

FAMILY = "moe"

GROUPED_MLP_MODULES = Operation(FAMILY, "grouped_mlp_modules")
ACTIVATION_FUNC = Operation(FAMILY, "activation_func")
MOE_ROUTER = Operation(FAMILY, "moe_router")

OPERATIONS = (GROUPED_MLP_MODULES, ACTIVATION_FUNC, MOE_ROUTER)


class MoeSlots:
    """The mixture-of-experts slots."""

    def grouped_mlp_modules(self, moe_use_grouped_gemm: bool) -> "ExpertsBuilder":
        """Which module and submodules to use for grouped mlp."""
        raise unowned(GROUPED_MLP_MODULES)

    def activation_func(self) -> Optional["TEActivationFunctionBuilder"]:
        """Which module to use for the activation function, or None for config.activation_func."""
        raise unowned(ACTIVATION_FUNC)

    def moe_router(self) -> Optional[type]:
        """Which MoE router to use, or None to keep the MoESubmodules default."""
        raise unowned(MOE_ROUTER)


#: Backend name -> the class that owns this family's slots. Add a backend by adding its class
#: to backends.py and one entry here; one that needs an optional package declares ``REQUIRES``.
BACKENDS = {"local": MoeLocal, "transformer_engine": MoeTE, "inference_optimized": MoeInference}

#: Used when the selected preset has no entry above.
DEFAULT = "local"

__all__ = [
    "ACTIVATION_FUNC",
    "BACKENDS",
    "DEFAULT",
    "FAMILY",
    "GROUPED_MLP_MODULES",
    "MOE_ROUTER",
    "OPERATIONS",
    "MoeSlots",
]
