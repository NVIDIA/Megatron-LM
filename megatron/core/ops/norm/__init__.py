# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Normalization: what a backend must provide, and which ones exist.

The implementations are in :mod:`.backends`; a provider in
``megatron.core.models.backends`` or ``megatron.core.extensions`` picks between them.

**Contract**

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

from megatron.core.ops.norm.backends import NormApex, NormTE, NormTorch, TENormWithResidual
from megatron.core.transformer.torch_norm import L2Norm, WrappedTorchNorm

__all__ = ["L2Norm", "NormApex", "NormTE", "NormTorch", "TENormWithResidual", "WrappedTorchNorm"]
