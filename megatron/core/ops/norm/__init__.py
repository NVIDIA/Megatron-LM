# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Normalization implementations, grouped by backend.

Contract for this family
------------------------
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

from megatron.core.ops.norm.apex import ApexNormBackend, apex_layer_norm, have_apex
from megatron.core.ops.norm.reference import L2Norm, TorchNormBackend, WrappedTorchNorm

__all__ = [
    "ApexNormBackend",
    "L2Norm",
    "TorchNormBackend",
    "WrappedTorchNorm",
    "apex_layer_norm",
    "have_apex",
]
