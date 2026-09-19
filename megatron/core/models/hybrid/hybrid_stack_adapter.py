# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Spec-injected forward extensions for the standard HybridStack layer loop."""

from dataclasses import dataclass
from typing import Protocol

from torch import Tensor

from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.hyper_connection import SinglePassMHCState
from megatron.core.transformer.transformer_layer import CrossLayerState


@dataclass
class HybridStackForwardContext:
    """Working state for one forward, never retained on a stack or adapter."""

    cross_layer_state: CrossLayerState | None = None
    mhc_state: SinglePassMHCState | None = None


class HybridStackForwardAdapter(Protocol):
    """Optional layer-owned state lifecycle, selected explicitly by a stack spec.

    An adapter stores construction metadata only. Activation tensors belong to
    the context returned by prepare_forward, including for overlapping microbatches.
    """

    def prepare_forward(
        self,
        hidden_states: Tensor,
        packed_seq_params: PackedSeqParams | None,
        inference_context: BaseInferenceContext | None,
        context: HybridStackForwardContext,
    ) -> HybridStackForwardContext:
        """Initialize or validate the caller's per-forward state before the layer loop."""
        ...

    def finalize_forward(self, output: Tensor, context: HybridStackForwardContext) -> Tensor:
        """Return the stack output after final normalization and stream contraction."""
        ...
