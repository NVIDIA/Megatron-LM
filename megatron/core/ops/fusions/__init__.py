# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Fusions that span several families: what one must declare, and which ones exist.

Every other family under ``megatron.core.ops`` owns one boundary, and its backends are
interchangeable -- ``NormTE`` and ``NormApex`` answer the same question, so they are read side
by side in one ``backends.py``. A fusion owns a *larger* boundary than any single family, and
two fusions are not alternatives to each other: each defines its own span. So this family keeps
one file per fusion, and two vendors adding two megakernels never touch the same file.

**Contract**

* **Target**: a ``TransformerLayer`` subclass, or ``None`` to keep the ordinary layer. It is
  handed the same ``TransformerLayerSubmodules`` the ordinary layer would have received, and
  ignores the ones it performs itself -- those are specs, so an ignored one is never built.
* **Slots**: one per point where the layer hands over control, *not* one per kernel. The two
  here are the two layer specs the builder already makes, dense and MoE, so a kernel spanning
  attention and the dense MLP and a kernel spanning attention and the experts are separate
  choices that can both be made. Both are optional: a fusion fills the one it covers.

**When a new slot is *not* the answer**
This class stays short only if the rule is held to, so it is written down. A new slot is
warranted when the layer hands over control somewhere it currently does not -- a span reaching
into the next layer, for instance, which is a block-level handover. It is **not** warranted for:

* **A wider footprint.** A kernel that also eats the input norm fills ``fused_moe_layer`` and
  lists ``LAYER_NORM`` in ``SPANS``. The handover point did not move.
* **Different internals.** An MLA-plus-MoE kernel and a GQA-plus-MoE kernel hand over at the
  same point, so they share ``fused_moe_layer`` and each refuses the configuration it cannot
  serve from ``from_options`` -- the same thing ``NormTorch`` does.
* **Another vendor.** Two kernels at one handover point are two entries in ``BACKENDS``.

``tests/unit_tests/ops/test_fusions.py:test_one_slot_per_handover_point`` pins the list, so
adding one is a deliberate, reviewed change rather than something that accumulates.
* **State**: the fusion is a transformer layer, so it owns what a transformer layer owns. Its
  state dict has to load a checkpoint the ordinary layer wrote, or it is a different model
  rather than a faster one.
* **Process groups**: whatever the layer it replaces owned, received the same way.
* **Declarations**: ``SPANS`` lists the operations it performs itself, so that selecting a
  backend for one of them is refused instead of silently ignored. ``REQUIRES``, ``DETERMINISM``
  and ``FUSES`` mean exactly what they mean on any other backend.

The freedom is real and the rule is short: fuse across whatever you like, but say which
operations you swallowed, what you need, and whether you are deterministic.
"""

from __future__ import annotations

from typing import Optional

from megatron.core.ops.fusions.attn_moe import AttnMoeReference
from megatron.core.ops.operations import Operation, unowned

FAMILY = "fusions"

#: Named for where control is handed over, not for what any one kernel swallows -- a kernel
#: reaching wider says so in SPANS rather than needing a slot of its own.
FUSED_DENSE_LAYER = Operation(FAMILY, "fused_dense_layer", optional=True)
FUSED_MOE_LAYER = Operation(FAMILY, "fused_moe_layer", optional=True)

OPERATIONS = (FUSED_DENSE_LAYER, FUSED_MOE_LAYER)


class FusionSlots:
    """The fusion slots: one per handover point, not one per fusion."""

    def fused_dense_layer(self) -> Optional[type]:
        """Which layer runs a dense layer as one kernel, or None to build it from parts."""
        raise unowned(FUSED_DENSE_LAYER)

    def fused_moe_layer(self) -> Optional[type]:
        """Which layer runs an MoE layer as one kernel, or None to build it from parts."""
        raise unowned(FUSED_MOE_LAYER)


class FusionNone:
    """No fusion: every operation keeps the backend its own family selected.

    Returning ``None`` to mean "keep the default" is the same shape ``moe_router`` already
    uses, so an unfused build asks the question and gets an answer rather than taking a
    different path through the spec builder.
    """

    #: It selects nothing, so it cannot make anything nondeterministic.
    DETERMINISM = "deterministic"

    def fused_dense_layer(self) -> None:
        """No fused layer; a dense layer is built from its parts."""
        return None

    def fused_moe_layer(self) -> None:
        """No fused layer; an MoE layer is built from its parts."""
        return None


#: Backend name -> the class that owns this family's slots. Add a fusion by adding its own
#: file beside attn_moe.py and one entry here.
BACKENDS = {"none": FusionNone, "attn_moe_reference": AttnMoeReference}

#: Used when the selected preset has no entry above, which for this family is always: a fusion
#: is never implied by --transformer-impl, only ever asked for by name.
DEFAULT = "none"

__all__ = [
    "BACKENDS",
    "DEFAULT",
    "FAMILY",
    "FUSED_DENSE_LAYER",
    "FUSED_MOE_LAYER",
    "OPERATIONS",
    "FusionNone",
    "FusionSlots",
]
