# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CUDA-graph spans for hybrid stacks.

The local CUDA-graph implementation normally creates one runner for every selected module. A
hybrid stack has a stronger execution structure: Mamba and attention layers are static, while a
MoE layer has one dynamic expert-compute island between a static router and static postprocess.
This module coalesces adjacent static operations into maximal spans, so CUDA graphs meet only
eager execution at their boundaries.
"""

from contextlib import nullcontext
from dataclasses import dataclass
from enum import Enum, auto
from typing import Iterator, Sequence, TypeAlias

import torch

from megatron.core.enums import Fp8Recipe
from megatron.core.fp4_utils import get_fp4_context
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.models.hybrid.layers import utils as layer_utils
from megatron.core.ssm.mamba_layer import MambaLayer
from megatron.core.transformer.enums import CudaGraphModule
from megatron.core.transformer.module import GraphableMegatronModule, MegatronModule
from megatron.core.transformer.transformer_layer import MoETransformerLayer


class HybridCudaGraphOperation(Enum):
    """One statically graphable operation in a hybrid stack."""

    LAYER = auto()
    MOE_ROUTER = auto()
    MOE_POSTPROCESS = auto()


@dataclass(frozen=True)
class HybridCudaGraphOperationSpec:
    """A graph operation and the HybridStack layer that owns it."""

    operation: HybridCudaGraphOperation
    layer_index: int


@dataclass(frozen=True)
class HybridCudaGraphSpanSpec:
    """A maximal sequence of adjacent graphable operations."""

    operations: tuple[HybridCudaGraphOperationSpec, ...]


@dataclass(frozen=True)
class HybridCudaGraphEagerExpertSpec:
    """The dynamic expert-compute island of one MoE layer."""

    layer_index: int


@dataclass(frozen=True)
class HybridCudaGraphEagerLayerSpec:
    """A complete layer that is outside the selected CUDA-graph modules."""

    layer_index: int


HybridCudaGraphPlanEntry: TypeAlias = (
    HybridCudaGraphSpanSpec | HybridCudaGraphEagerExpertSpec | HybridCudaGraphEagerLayerSpec
)

_SUPPORTED_HYBRID_CUDA_GRAPH_MODULES = frozenset(
    {
        CudaGraphModule.mamba,
        CudaGraphModule.attn,
        CudaGraphModule.moe_router,
        CudaGraphModule.moe_preprocess,
    }
)


def should_use_hybrid_cuda_graph_spans(config, *, cp_size: int = 1) -> bool:
    """Select spans for eligible HybridStack local partial-CG configurations."""

    if not config.cuda_graph_coalesce_partial_captures or config.cuda_graph_impl != "local":
        return False
    if cp_size != 1 or config.enable_mhc_connections:
        return False
    if config.recompute_granularity == "full":
        return False

    modules = set(config.cuda_graph_modules)
    unsupported = modules - _SUPPORTED_HYBRID_CUDA_GRAPH_MODULES
    if not modules or unsupported:
        return False

    return True


def build_hybrid_cuda_graph_span_plan(
    layer_config_list: Sequence, cuda_graph_modules: Sequence[CudaGraphModule]
) -> tuple[HybridCudaGraphPlanEntry, ...]:
    """Build maximal graph spans separated by eager layers or MoE expert compute."""

    scopes = set(cuda_graph_modules)
    plan: list[HybridCudaGraphPlanEntry] = []
    pending: list[HybridCudaGraphOperationSpec] = []

    def flush_span() -> None:
        if pending:
            plan.append(HybridCudaGraphSpanSpec(tuple(pending)))
            pending.clear()

    for layer_index, layer_config in enumerate(layer_config_list):
        config_type = type(layer_config)
        if config_type is layer_utils.MambaLayerConfig and CudaGraphModule.mamba in scopes:
            pending.append(
                HybridCudaGraphOperationSpec(HybridCudaGraphOperation.LAYER, layer_index)
            )
        elif config_type is layer_utils.AttentionLayerConfig and CudaGraphModule.attn in scopes:
            pending.append(
                HybridCudaGraphOperationSpec(HybridCudaGraphOperation.LAYER, layer_index)
            )
        elif config_type is layer_utils.MoELayerConfig and CudaGraphModule.moe_router in scopes:
            pending.append(
                HybridCudaGraphOperationSpec(HybridCudaGraphOperation.MOE_ROUTER, layer_index)
            )
            flush_span()
            plan.append(HybridCudaGraphEagerExpertSpec(layer_index))
            pending.append(
                HybridCudaGraphOperationSpec(HybridCudaGraphOperation.MOE_POSTPROCESS, layer_index)
            )
        else:
            flush_span()
            plan.append(HybridCudaGraphEagerLayerSpec(layer_index))

    flush_span()
    return tuple(plan)


def _call_module_without_local_cudagraph(module, **kwargs):
    """Run a graphable module through normal PyTorch hooks, bypassing its local graph manager."""

    return super(MegatronModule, module).__call__(**kwargs)


def _inner_quantization_context(layer):
    config = layer.config
    if config.fp8 and config.fp8_recipe != Fp8Recipe.delayed:
        return get_fp8_context(config, layer.layer_number - 1)
    if config.fp4:
        return get_fp4_context(config, layer.layer_number - 1)
    return nullcontext()


class HybridCudaGraphSpan(GraphableMegatronModule):
    """One local CUDA graph spanning adjacent operations from several HybridStack layers.

    Referenced layers stay registered only under ``HybridStack.layers``. The span exposes their
    modules, parameters, and buffers to ``CudaGraphManager`` without registering duplicate module
    paths in the model state dict.
    """

    def __init__(self, config, spec: HybridCudaGraphSpanSpec, layers: Sequence[torch.nn.Module]):
        from megatron.core.transformer.cuda_graphs import set_cuda_graph_stream_pool_size

        # One shared replay stream serializes graph-pool scratch reuse through the Stage-2 RS tail.
        set_cuda_graph_stream_pool_size(1)
        super().__init__(config)
        self.spec = spec
        self.is_hybrid_cuda_graph_span = True
        self._span_layers = tuple(layers[op.layer_index] for op in spec.operations)

    def _unique_layers(self) -> Iterator[torch.nn.Module]:
        seen = set()
        for layer in self._span_layers:
            if id(layer) not in seen:
                seen.add(id(layer))
                yield layer

    def modules(self) -> Iterator[torch.nn.Module]:
        """Expose referenced layer modules to CUDA-graph DDP and FP8 handling."""

        yield self
        seen = {id(self)}
        for layer in self._unique_layers():
            for module in layer.modules():
                if id(module) not in seen:
                    seen.add(id(module))
                    yield module

    def parameters(self, recurse: bool = True) -> Iterator[torch.nn.Parameter]:
        """Expose referenced parameters without registering duplicate module ownership."""

        if not recurse:
            return
        seen = set()
        for layer in self._unique_layers():
            for param in layer.parameters():
                if id(param) not in seen:
                    seen.add(id(param))
                    yield param

    def buffers(self, recurse: bool = True) -> Iterator[torch.Tensor]:
        """Expose referenced buffers for capture-time backup and restore."""

        if not recurse:
            return
        seen = set()
        for layer in self._unique_layers():
            for buffer in layer.buffers():
                if id(buffer) not in seen:
                    seen.add(id(buffer))
                    yield buffer

    def forward(
        self,
        *state,
        attention_mask=None,
        inference_context=None,
        rotary_pos_emb=None,
        packed_seq_params=None,
        sequence_len_offset=None,
        padding_mask=None,
    ):
        """Execute every operation in this span as one graphable callable."""

        if inference_context is not None:
            raise ValueError("Hybrid CUDA-graph spans currently support training only")

        for op, layer in zip(self.spec.operations, self._span_layers, strict=True):
            with _inner_quantization_context(layer):
                if op.operation is HybridCudaGraphOperation.LAYER:
                    if len(state) != 1:
                        raise RuntimeError(
                            "A complete hybrid layer expects one hidden-state tensor"
                        )
                    if isinstance(layer, MambaLayer):
                        output = _call_module_without_local_cudagraph(
                            layer,
                            hidden_states=state[0],
                            attention_mask=attention_mask,
                            inference_context=inference_context,
                            rotary_pos_emb=rotary_pos_emb,
                            packed_seq_params=packed_seq_params,
                        )
                    else:
                        output = _call_module_without_local_cudagraph(
                            layer,
                            hidden_states=state[0],
                            attention_mask=attention_mask,
                            inference_context=inference_context,
                            rotary_pos_emb=rotary_pos_emb,
                            packed_seq_params=packed_seq_params,
                            sequence_len_offset=sequence_len_offset,
                            padding_mask=padding_mask,
                        )
                    state = (output[0] if isinstance(output, tuple) else output,)
                elif op.operation is HybridCudaGraphOperation.MOE_ROUTER:
                    if len(state) != 1 or not isinstance(layer, MoETransformerLayer):
                        raise RuntimeError("A MoE router span expects one hidden-state tensor")
                    hidden_states, context = layer._forward_attention(
                        hidden_states=state[0],
                        attention_mask=attention_mask,
                        inference_context=inference_context,
                        rotary_pos_emb=rotary_pos_emb,
                        packed_seq_params=packed_seq_params,
                        sequence_len_offset=sequence_len_offset,
                        padding_mask=padding_mask,
                    )
                    if context is not None:
                        raise NotImplementedError("Cross attention is unsupported in hybrid spans")
                    state = tuple(
                        layer._forward_mlp_router(hidden_states, padding_mask=padding_mask)
                    )
                elif op.operation is HybridCudaGraphOperation.MOE_POSTPROCESS:
                    if len(state) != 4 or not isinstance(layer, MoETransformerLayer):
                        raise RuntimeError(
                            "A MoE postprocess span expects residual, expert output, shared "
                            "expert output, and MLP bias"
                        )
                    state = (layer._forward_mlp_postprocess(*state),)
                else:
                    raise AssertionError(f"Unknown hybrid CUDA-graph operation: {op.operation}")

        return state[0] if len(state) == 1 else tuple(state)


def run_hybrid_eager_expert(layer: MoETransformerLayer, router_outputs):
    """Run the dynamic expert-compute island between two CUDA-graph spans."""

    residual, hidden_states, probs, shared_expert_output, *token_dispatcher_attr_outputs = (
        router_outputs
    )
    layer._synchronize_router_host_outputs(token_dispatcher_attr_outputs)
    with _inner_quantization_context(layer):
        expert_output, mlp_bias = layer._forward_mlp_expert_compute(
            hidden_states, probs, token_dispatcher_attr_outputs
        )
    return residual, expert_output, shared_expert_output, mlp_bias


def run_hybrid_eager_layer(
    layer,
    hidden_states,
    *,
    attention_mask,
    inference_context,
    rotary_pos_emb,
    packed_seq_params,
    sequence_len_offset,
    padding_mask,
):
    """Run a complete non-span layer while bypassing any per-layer graph manager."""

    with _inner_quantization_context(layer):
        if isinstance(layer, MambaLayer):
            output = _call_module_without_local_cudagraph(
                layer,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                inference_context=inference_context,
                rotary_pos_emb=rotary_pos_emb,
                packed_seq_params=packed_seq_params,
            )
        else:
            output = _call_module_without_local_cudagraph(
                layer,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                inference_context=inference_context,
                rotary_pos_emb=rotary_pos_emb,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
                padding_mask=padding_mask,
            )
    return output[0] if isinstance(output, tuple) else output
