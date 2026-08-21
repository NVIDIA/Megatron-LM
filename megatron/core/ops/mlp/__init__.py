# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The dense MLP block: what a backend must provide, and which ones exist.

The implementations are in :mod:`.backends`; a provider in
``megatron.core.models.backends`` or ``megatron.core.extensions`` picks between them.

**Contract**

* **Target**: a class exposing ``as_mlp_submodule``, which the transformer layer calls with
  the MLP submodules to build the block. The block owns its own linears and activation; which
  linears it gets come from the ``linear`` family, not from here.
* **Selection input**: ``grouped`` says the dense MLP should use grouped GEMM, which is known
  while the model is built. A backend without a grouped form ignores it.
* **State**: the block owns the weights of the linears it builds. Its state dict must match
  the unfused MLP's, since the two are interchangeable at the spec level.
* **Process groups**: none of its own; the linears it builds own their tensor-parallel
  collectives, using the group the owning module passes in.
* **Modes**: only Transformer Engine fuses, and only from TE 1.13. The fused form does not
  support mixture-of-experts, which is why the MoE path uses ``moe.grouped_mlp_modules``.
"""

from __future__ import annotations

from megatron.core.ops.mlp.backends import MlpMegatron, MlpTEOpFuser

__all__ = ["MlpMegatron", "MlpTEOpFuser"]
