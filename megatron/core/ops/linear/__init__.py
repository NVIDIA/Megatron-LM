# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Linear layers: what a backend must provide, and which ones exist.

The implementations are in :mod:`.backends`; a provider in
``megatron.core.models.backends`` or ``megatron.core.extensions`` picks between them.

**Contract**

* **Targets**: classes the owning module constructs with Megatron's linear signature.
  ``column_parallel_layer_norm_linear`` returns ``None`` when the backend does not fuse the
  preceding norm into the projection.
* **State**: each target owns its weight and optional bias; the fused variant additionally owns
  the norm weight under ``layer_norm_weight``. Fused and unfused targets therefore have
  different state dicts, which is why the layer spec, not the backend, decides which is used.
* **Process groups**: the target owns the tensor-parallel collectives for its own weights,
  using the group the owning module passes in. Nothing here creates a group.
* **Modes**: every backend supports training, backward, and inference. Only Transformer Engine
  and Kitchen support quantized execution.
"""

from __future__ import annotations

from megatron.core.ops.linear.backends import LinearLocal, LinearTE

__all__ = ["LinearLocal", "LinearTE"]
