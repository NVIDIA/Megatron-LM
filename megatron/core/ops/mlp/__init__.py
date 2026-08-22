# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The dense MLP block: what a backend must provide, and which ones exist.

The implementations are in :mod:`.backends`.

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
from megatron.core.ops.operations import Operation, unowned

FAMILY = "mlp"

MLP_MODULE = Operation(FAMILY, "mlp_module")

OPERATIONS = (MLP_MODULE,)


class MlpSlots:
    """The dense MLP slots."""

    def mlp_module(self, grouped: bool = False) -> type:
        """Which module to use for the dense MLP block."""
        raise unowned(MLP_MODULE)


#: Backend name -> the class that owns this family's slots. Add a backend by adding its class
#: to backends.py and one entry here; one that needs an optional package declares ``REQUIRES``.
#:
#: Deliberately no "transformer_engine" key: a Transformer Engine run uses the plain MLP unless
#: --use-transformer-engine-op-fuser asks for the fused one.
BACKENDS = {"megatron": MlpMegatron, "te_op_fuser": MlpTEOpFuser}

#: Used when the selected preset has no entry above, which for this family is always.
DEFAULT = "megatron"


def legacy_backends(options) -> dict:
    """Which backend ``--use-transformer-engine-op-fuser`` selects.

    That flag predates ``--op-backend``, so it is a second vocabulary for this slot. The
    translation lives here, beside BACKENDS, rather than in the resolver.
    """
    if not options.use_te_op_fuser:
        return {}
    return {MLP_MODULE: "te_op_fuser"}


__all__ = [
    "BACKENDS",
    "DEFAULT",
    "FAMILY",
    "MLP_MODULE",
    "OPERATIONS",
    "MlpSlots",
    "legacy_backends",
]
