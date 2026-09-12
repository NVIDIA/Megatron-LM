# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Attention fused with the expert MLPs.

One file, one fusion. This is the file a vendor copies to contribute a megakernel: nothing
outside it has to change, and no other vendor's file is touched. The contract it meets is in
this package's ``__init__``.
"""

from __future__ import annotations

from megatron.core.ops.attention import CORE_ATTENTION
from megatron.core.ops.moe import GROUPED_MLP_MODULES
from megatron.core.transformer.transformer_layer import TransformerLayer

__all__ = ["AttnMoeReference", "ReferenceFusedAttentionMoELayer"]


class ReferenceFusedAttentionMoELayer(TransformerLayer):
    """Where the kernel goes.

    A fused layer is handed the same :class:`TransformerLayerSubmodules` the ordinary layer
    would have received, and is free to ignore the ones it performs itself -- those are specs,
    not modules, so ignoring one means it is simply never built.

    This reference ignores none of them and calls the ordinary path, so a run with it selected
    trains identically to a run without. That is the point: the wiring can be tested on its
    own, before anyone writes a kernel.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #: Counts calls, so a test can tell the fused layer is the one that ran.
        self.fused_steps = 0

    def forward(self, *args, **kwargs):
        """One entry point for attention through the experts.

        A real kernel does here, in one call into its own library, what
        ``TransformerLayer.forward`` does across attention, the residual, and the experts.
        """
        self.fused_steps += 1
        return super().forward(*args, **kwargs)


class AttnMoeReference:
    """A worked attention-plus-MoE fusion, selected like any other backend::

        --op-backend fused_moe_layer=attn_moe_reference

    It fills the MoE handover point and leaves the dense one alone, so an attention-plus-dense
    -MLP kernel from somewhere else can be selected alongside it.

    A vendor's version differs in three places and nowhere else::

        REQUIRES = "vendor_kernels>=0.4"       # checked once, while arguments are parsed
        DETERMINISM = "nondeterministic"       # refused under --deterministic-mode

        def fused_moe_layer(self):
            from vendor_kernels.megatron import FusedAttentionMoELayer   # here, never at
            return FusedAttentionMoELayer                                # module scope
    """

    #: In tree, so there is nothing to require. An optional package is named here instead of
    #: being checked in code, so the dependency is visible in the class header.
    REQUIRES = None

    #: It delegates to the ordinary layer, which has not been audited for bit-exactness.
    DETERMINISM = "unknown"

    #: The operations this fusion performs itself. Selecting a backend for one of them is
    #: refused rather than silently ignored -- see ``resolve.py:_check_spans``. This is the
    #: declaration that makes an arbitrary span checkable: fuse what you like, but say what
    #: you swallowed.
    SPANS = (CORE_ATTENTION, GROUPED_MLP_MODULES)

    #: A kernel that can only accept particular *neighbours* -- slots it does not perform but
    #: is handed -- declares them the way any backend does, and would say, for example::
    #:
    #:     FUSES = {COLUMN_PARALLEL_LAYER_NORM_LINEAR: "transformer_engine"}
    #:
    #: The reference accepts anything, so it declares nothing.

    def fused_moe_layer(self) -> type:
        """The layer that runs an MoE layer as one kernel."""
        return ReferenceFusedAttentionMoELayer
