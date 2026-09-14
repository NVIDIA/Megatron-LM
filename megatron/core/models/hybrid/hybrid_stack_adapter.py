# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Model-specific forward extensions for the generic Hybrid layer loop."""

from dataclasses import dataclass, field
from typing import Any, Protocol

from torch import Tensor
from torch.distributed import ProcessGroup

from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.pipeline_parallel.pipeline_payload import PipelinePayloadFactory
from megatron.core.transformer.hyper_connection import SinglePassMHCState


@dataclass
class HybridStackForwardContext:
    """Working state for one forward; never retained on the stack or its adapter.

    ``layer_kwargs`` are passed through to Transformer layers, including those
    inside mHC wrappers. The stack owns mHC execution; an adapter may restore
    its incoming mixing state and export it after the layer loop.
    """

    layer_kwargs: dict[str, Any] = field(default_factory=dict)
    mhc_state: SinglePassMHCState | None = None


class HybridStackForwardAdapter(Protocol):
    """Optional, spec-injected adapter for model-specific state and boundaries.

    Adapters are plain objects with static configuration, not parameter-bearing
    modules. HybridStack constructs one using ``config``, ``layer_type_list``,
    ``pp_layer_offset``, ``pre_process``, ``post_process``, ``is_mtp_layer`` and
    the stack's explicit ``pg_collection``.
    All activation state must be returned in a fresh per-forward context.
    """

    def configure_distributed_pipeline(
        self, pattern: str, pp_group: ProcessGroup, vp_stage: int | None = None
    ) -> PipelinePayloadFactory | None:
        """Bind static boundaries and return the receive factory, or None for PP=1."""
        ...

    @property
    def consumes_input_tensor(self) -> bool:
        """Whether to clear the pending input before attempting this forward."""
        ...

    def validate_input(self, input_tensor: Any) -> None:
        """Validate an input supplied through the existing model/stack setter."""
        ...

    def prepare_forward(
        self,
        hidden_states: Any,
        packed_seq_params: PackedSeqParams | None,
        inference_context: BaseInferenceContext | None,
    ) -> tuple[Any, PackedSeqParams | None, HybridStackForwardContext]:
        """Restore input and metadata and create working state before the layer loop."""
        ...

    def finalize_forward(
        self,
        output: Tensor | tuple[Tensor, Tensor],
        packed_seq_params: PackedSeqParams | None,
        context: HybridStackForwardContext,
    ) -> Any:
        """Export model-specific output after mHC contraction and final normalization."""
        ...

    def recompute_boundary_tensors(self, context: HybridStackForwardContext) -> tuple[Tensor, ...]:
        """Return live side outputs whose backward may enter the current recompute group.

        Tensor references must describe this boundary now, rather than being
        read from mutable working state when backward eventually runs.
        """
        ...
