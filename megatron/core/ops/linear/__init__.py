# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Linear layers: what a backend must provide, and which ones exist.

The implementations are in :mod:`.backends`.

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

from typing import Optional

from megatron.core.inference.ops.backends import LinearInference
from megatron.core.ops.linear.backends import LinearLocal, LinearTE
from megatron.core.ops.operations import Operation, unowned

FAMILY = "linear"

# Megatron Core has no non-tensor-parallel linear, so a backend may leave this one alone.
LINEAR = Operation(FAMILY, "linear", optional=True)
COLUMN_PARALLEL_LINEAR = Operation(FAMILY, "column_parallel_linear")
ROW_PARALLEL_LINEAR = Operation(FAMILY, "row_parallel_linear")
COLUMN_PARALLEL_LAYER_NORM_LINEAR = Operation(FAMILY, "column_parallel_layer_norm_linear")

OPERATIONS = (
    LINEAR,
    COLUMN_PARALLEL_LINEAR,
    ROW_PARALLEL_LINEAR,
    COLUMN_PARALLEL_LAYER_NORM_LINEAR,
)


class LinearSlots:
    """The linear slots, plus the one query derived from them."""

    def linear(self) -> type:
        """Which non-parallel linear module the backend uses."""
        raise unowned(LINEAR)

    def column_parallel_linear(self) -> type:
        """Which column parallel linear module the backend uses."""
        raise unowned(COLUMN_PARALLEL_LINEAR)

    def row_parallel_linear(self) -> type:
        """Which row parallel linear module the backend uses."""
        raise unowned(ROW_PARALLEL_LINEAR)

    def column_parallel_layer_norm_linear(self) -> Optional[type]:
        """Which module fuses layernorm and column parallel linear, or None if unfused."""
        raise unowned(COLUMN_PARALLEL_LAYER_NORM_LINEAR)

    def fuse_layernorm_and_linear(self) -> bool:
        """Whether the backend fuses layernorm into the column parallel linear.

        Derived, never owned: asking the slot keeps this answer and the target it describes
        from disagreeing.
        """
        return self.column_parallel_layer_norm_linear() is not None


#: Backend name -> the class that owns this family's slots. Add a backend by adding its class
#: to backends.py and one entry here; one that needs an optional package declares ``REQUIRES``.
BACKENDS = {
    "local": LinearLocal,
    "transformer_engine": LinearTE,
    "inference_optimized": LinearInference,
}

#: Used when the selected preset has no entry above.
DEFAULT = "local"

__all__ = [
    "BACKENDS",
    "COLUMN_PARALLEL_LAYER_NORM_LINEAR",
    "COLUMN_PARALLEL_LINEAR",
    "DEFAULT",
    "FAMILY",
    "LINEAR",
    "OPERATIONS",
    "ROW_PARALLEL_LINEAR",
    "LinearSlots",
]
