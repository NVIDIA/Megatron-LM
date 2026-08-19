# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Normalization: what a backend must provide, and which ones exist.

The implementations are in :mod:`.backends`.

Contract
--------
* **Target**: a builder called as ``target(config=..., hidden_size=..., eps=...)`` returning a
  module whose ``forward(x) -> Tensor`` preserves shape and dtype.
* **Selection inputs**: ``rms_norm`` (RMSNorm vs LayerNorm), ``for_qk`` (query/key norm, which
  some backends implement differently), ``has_residual`` (this norm is followed by a residual
  add, which some backends can fuse). All three are known while the model is built.
* **State**: the norm owns its own weight, and its bias for LayerNorm. Backends must agree on
  the parameter names so a checkpoint stays loadable across backends.
* **Process groups**: none. Sequence-parallel marking is read from ``config`` by the target
  itself; the owning module keeps any surrounding communication.
* **Modes**: every backend supports training, backward, and inference. Only Transformer Engine
  fuses a residual, and only for RMSNorm. Apex implements LayerNorm only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from megatron.core.inference.ops.backends import NormInference
from megatron.core.ops.norm.backends import NormApex, NormLocal, NormTE, NormTorch
from megatron.core.ops.operations import Operation, unowned
from megatron.core.transformer.torch_norm import L2Norm, WrappedTorchNorm

if TYPE_CHECKING:
    from megatron.core.transformer.torch_norm import LayerNormBuilder

FAMILY = "norm"

LAYER_NORM = Operation(FAMILY, "layer_norm")

OPERATIONS = (LAYER_NORM,)


class NormSlots:
    """The normalization slots, with the error a slot gives when no backend owns it."""

    def layer_norm(
        self, rms_norm: bool = False, for_qk: bool = False, has_residual: bool = False
    ) -> "LayerNormBuilder":
        """Which module to use for layernorm."""
        raise unowned(LAYER_NORM)


#: Backend name -> the class that owns this family's slots. Add a backend by adding its class
#: to backends.py and one entry here; one that needs an optional package declares ``REQUIRES``.
BACKENDS = {
    "local": NormLocal,
    "transformer_engine": NormTE,
    "inference_optimized": NormInference,
    "torch": NormTorch,
    "apex": NormApex,
}

#: Used when the selected preset has no entry above.
DEFAULT = "local"

__all__ = [
    "BACKENDS",
    "DEFAULT",
    "FAMILY",
    "LAYER_NORM",
    "OPERATIONS",
    "L2Norm",
    "NormSlots",
    "WrappedTorchNorm",
]
