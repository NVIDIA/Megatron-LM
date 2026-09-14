# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Model-specific forward extensions for the generic Hybrid layer loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, ContextManager, Protocol

from torch import Tensor
from torch.distributed import ProcessGroup
from torch.nn import Module, ModuleList

from megatron.core import tensor_parallel
from megatron.core.extensions.transformer_engine import te_checkpoint
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.pipeline_parallel.pipeline_payload import (
    PipelinePayloadFactory,
    PipelinePayloadSpec,
)
from megatron.core.transformer.hyper_connection import SinglePassMHCState
from megatron.core.transformer.transformer_config import TransformerConfig

if TYPE_CHECKING:
    from megatron.core.transformer.transformer_layer import CrossLayerState, RecomputeTensors


@dataclass
class HybridStackForwardContext:
    """Working state for one forward; never retained on the stack or its adapter.

    ``cross_layer_state`` is passed to participating layers, including those
    inside mHC wrappers. The stack owns mHC execution; an adapter may restore
    its incoming mixing state and export it after the layer loop.
    """

    cross_layer_state: CrossLayerState | None = None
    mhc_state: SinglePassMHCState | None = None

    def recompute_boundary_tensors(self) -> tuple[Tensor, ...]:
        """Snapshot all live side outputs of the current layer group."""
        tensors = (
            ()
            if self.cross_layer_state is None
            else self.cross_layer_state.recompute_boundary_tensors()
        )
        if self.mhc_state is not None and self.mhc_state.pre_mix is not None:
            tensors += (self.mhc_state.pre_mix,)
        return tensors

    def save_for_recompute(
        self,
    ) -> tuple[RecomputeTensors, Callable[[RecomputeTensors], "HybridStackForwardContext"]]:
        """Keep model state and the stack's mHC state on explicit checkpoint edges."""
        tensors: RecomputeTensors = ()
        restore_state: Callable[[RecomputeTensors], CrossLayerState] | None = None
        if self.cross_layer_state is not None:
            tensors, restore_state = self.cross_layer_state.save_for_recompute()
        has_mhc_state = self.mhc_state is not None
        pre_mix = self.mhc_state.pre_mix if self.mhc_state is not None else None

        def restore(values: RecomputeTensors) -> "HybridStackForwardContext":
            return HybridStackForwardContext(
                cross_layer_state=restore_state(values[1:]) if restore_state is not None else None,
                mhc_state=SinglePassMHCState(values[0]) if has_mhc_state else None,
            )

        return (pre_mix, *tensors), restore


class HybridStackForwardAdapter(Protocol):
    """Optional, spec-injected adapter for model-specific state and boundaries.

    Adapters are plain objects with static configuration, not parameter-bearing
    modules. HybridStack constructs one using ``config``, ``layer_type_list``,
    ``pp_layer_offset``, ``pre_process``, ``post_process``, ``is_mtp_layer`` and
    the stack's explicit ``pg_collection``.
    All activation state must be returned in a fresh per-forward context.
    """

    def configure_cuda_graphs(self, layers: ModuleList) -> None:
        """Attach model-specific graph schemas, or do nothing when graphs are unused."""
        ...

    def configure_distributed_pipeline(
        self, pattern: str, pp_group: ProcessGroup, vp_stage: int | None = None
    ) -> PipelinePayloadFactory | None:
        """Bind static boundaries and return the receive factory, or None for PP=1."""
        ...

    def pipeline_payload_spec(
        self,
        seq_length: int,
        micro_batch_size: int,
        packed_seq_params: PackedSeqParams | None = None,
        *,
        requires_grad: bool = True,
    ) -> tuple[PipelinePayloadSpec | None, PipelinePayloadSpec | None]:
        """Describe incoming/outgoing tensor boundaries, or return (None, None) for PP=1."""
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


def checkpointed_hybrid_forward(
    config: TransformerConfig,
    layers: ModuleList,
    hidden_states: Tensor,
    context: HybridStackForwardContext,
    *,
    tp_group: ProcessGroup,
    quantization_context: Callable[[TransformerConfig, int], ContextManager],
    layer_forward: Callable[[Module, Tensor, HybridStackForwardContext], Tensor],
) -> Tensor:
    """Run full-recompute groups with explicit model and mHC tensor boundaries.

    ``layer_forward`` must use the supplied working context; it must not capture
    the caller's mutable context. Routing/position metadata may be captured if
    immutable for this microbatch. Only returned tensor edges and independent
    restored state reach the next group or pipeline export.
    """

    def group_forward(start, end, restore):
        output_restore = [None]

        def forward(hidden, *state_tensors):
            working = restore(state_tensors)
            for index in range(start, end):
                layer = layers[index]
                with quantization_context(config, layer.layer_number - 1):
                    hidden = layer_forward(layer, hidden, working)
            tensors, output_restore[0] = working.save_for_recompute()
            return hidden, *tensors

        return forward, output_restore

    uniform = config.recompute_method == "uniform"
    count = config.recompute_num_layers
    group_size = count if uniform else 1
    remaining = count
    for start in range(0, len(layers), group_size):
        state_inputs, restore = context.save_for_recompute()
        args = (hidden_states, *state_inputs)
        forward, output_restore = group_forward(
            start, min(start + group_size, len(layers)), restore
        )
        # Reentrant checkpointing needs a differentiable input. Frozen groups
        # run normally; block mode counts only eligible groups, including those
        # with frozen hidden states but live shared side inputs.
        use_checkpoint = (uniform or remaining > 0) and any(
            tensor is not None and tensor.requires_grad for tensor in args
        )
        if use_checkpoint:
            if config.fp8 or config.fp4 or config.quant_recipe is not None:
                outputs = te_checkpoint(
                    forward,
                    config.distribute_saved_activations,
                    tensor_parallel.random.get_cuda_rng_tracker,
                    tp_group,
                    *args,
                )
            else:
                outputs = tensor_parallel.checkpoint(
                    forward, config.distribute_saved_activations, *args
                )
            remaining -= 1
        else:
            outputs = forward(*args)
        hidden_states, *state_outputs = outputs
        restored = output_restore[0](tuple(state_outputs))
        context.cross_layer_state = restored.cross_layer_state
        context.mhc_state = restored.mhc_state
    return hidden_states
