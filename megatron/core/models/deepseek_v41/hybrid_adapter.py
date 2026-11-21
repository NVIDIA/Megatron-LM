# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""V4.1 execution on HybridStack, with shared state explicit at every replay boundary."""

from dataclasses import dataclass, replace

import torch
from torch import nn

from megatron.core.extensions.transformer_engine import TENorm, te_checkpoint
from megatron.core.fp8_utils import get_fp8_context, get_fp8_recipe, is_first_last_bf16_layer
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.engram import EngramLookupResult
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_stack_adapter import HybridStackForwardContext
from megatron.core.pipeline_parallel.pipeline_stage import PipelineStageState
from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
from megatron.core.transformer.experimental_attention_variant.csa2_module_spec import (
    csa2_attention_spec,
)
from megatron.core.transformer.experimental_attention_variant.dsa import (
    DSAIndexerLossAutoScaler,
    DSAIndexerLossLoggingHelper,
)
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.state_boundary import TensorField
from megatron.core.transformer.stateful_module import StatefulGraphs, StatefulModule
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules

from .state import V41StateCodec


@dataclass
class V41ForwardContext(HybridStackForwardContext, PipelineStageState):
    """Microbatch-local tensors; a module never caches this object."""

    codec: V41StateCodec | None = None
    token_ids: torch.Tensor | None = None
    engram_tokens: torch.Tensor | None = None
    capture: dict | None = None


class _LayerRegion(nn.Module):
    """Rebuild mutable state from tensor arguments on every forward and replay."""

    def __init__(self, layer, codec, kwargs, loss):
        super().__init__()
        self.layer, self.codec, self.kwargs, self.loss = layer, codec, kwargs, loss
        self.training = layer.training

    def forward(self, hidden, values):
        """Export the objective so logging and loss scaling execute only once."""
        boundary = self.layer.layer_number - 1
        csa, mhc, packed = self.codec.restore(values, boundary)
        # Checkpoint replay runs after the stack's per-layer FP8 context has exited.
        with get_fp8_context(self.layer.config, boundary):
            engram = getattr(self.layer, 'engram', None)
            if engram is not None:
                layout = csa.thd_layout
                lookup = EngramLookupResult(
                    values["engram_memory"],
                    None if layout is None else layout.valid_tokens.view(1, -1, 1),
                    None if layout is None else layout.cu_seqlens_padded,
                )
                hidden = engram(hidden, add_to_hidden_states=True, lookup=lookup)
            hidden, _ = self.layer(
                hidden,
                cross_layer_state=csa,
                mhc_state=mhc,
                packed_seq_params=packed,
                **self.kwargs,
            )
        return hidden, self.codec.snapshot(csa, mhc, values, boundary + 1, loss=self.loss)


class DeepSeekV41Adapter:
    """Model-owned layer execution; the generic stack retains its sole layer loop."""

    def __init__(
        self,
        config,
        *,
        layer_type_list,
        pp_layer_offset,
        pre_process,
        post_process,
        is_mtp_layer,
        pg_collection,
    ):
        if is_mtp_layer:
            raise ValueError('V4.1 backbone integration does not include MTP or DSpark')
        expected = ('VE' * (config.num_layers // 2))[
            pp_layer_offset : pp_layer_offset + len(layer_type_list)
        ]
        if ''.join(layer_type_list) != expected:
            raise ValueError('V4.1 uses an explicit VE pair per logical block')
        self.config = config
        self.start, self.end = pp_layer_offset, pp_layer_offset + len(layer_type_list)
        self.pre_process, self.post_process = pre_process, post_process
        self.pg_collection = pg_collection
        # Captures retain independent slots and static profiles, never caller activations.
        self.graphs = {}
        self._next_slot = {}

    def prepare_forward(self, hidden_states, packed_seq_params, inference_context, context):
        """Validate explicit state supplied by the model wrapper or pipeline receiver."""
        if inference_context is not None:
            raise NotImplementedError('V4.1 draft supports training/full-sequence evaluation')
        if not isinstance(context, V41ForwardContext) or context.codec is None:
            raise ValueError('V4.1 requires a model-owned tensor-state forward context')
        if context.cross_layer_state is None:
            context.cross_layer_state, context.mhc_state, _ = context.codec.restore(
                context.values, self.start
            )
        return context

    @staticmethod
    def _lookup_engram(memory, context):
        tokens = context.token_ids
        if context.engram_tokens is None:
            context.engram_tokens = memory.tokens_with_history(tokens)
        return memory.lookup_with_history(
            tokens, context.engram_tokens, context.codec.packed(context.values)
        )

    def _prepare_region(self, layer, hidden, context, kwargs, loss):
        kwargs = dict(kwargs)
        kwargs.pop('packed_seq_params', None)
        kwargs.pop('mhc_recompute_manager', None)
        boundary = layer.layer_number - 1
        values = context.codec.snapshot(
            context.cross_layer_state, context.mhc_state, context.values, boundary
        )
        inputs = context.codec.schema(boundary).fields
        if getattr(layer, 'engram', None) is not None:
            lookup = self._lookup_engram(layer.engram, context)
            values["engram_memory"] = lookup.memory
            inputs = (
                *inputs,
                TensorField(
                    "engram_memory",
                    tuple(lookup.memory.shape),
                    lookup.memory.dtype,
                    "contiguous",
                    lookup.memory.requires_grad,
                ),
            )
        region = _LayerRegion(layer, context.codec, kwargs, loss)
        return (
            StatefulModule(region, inputs, context.codec.schema(boundary + 1, loss=loss).fields),
            values,
        )

    def run_layer(self, layer, hidden_states, context, **kwargs):
        """Run eagerly, with explicit checkpointing, or in a prepared graph slot."""
        c, codec, csa = self.config, context.codec, context.cross_layer_state
        layer_id = layer.layer_number - 1
        loss = (
            layer_id in c.csa2_index_source_layers
            and layer.training
            and torch.is_grad_enabled()
            and (c.dsa_indexer_loss_coeff or 0) > 0
        )
        recompute = layer.training and c.recompute_granularity == 'full'
        graph_key = (layer_id, codec.shape, loss, layer.training)
        capture = context.capture is not None and layer_id in context.capture['layers']
        if capture and layer_id % 2:
            raise ValueError('Variable-token MoE execution remains outside layer capture')
        if recompute or graph_key in self.graphs or capture:
            region, values = self._prepare_region(layer, hidden_states, context, kwargs, loss)
            if capture:
                if recompute:
                    raise ValueError(
                        'Full recompute and layer capture are separate execution profiles'
                    )
                capture_kwargs = {}
                if c.fp8:
                    if context.capture['backend'] != 'transformer_engine':
                        raise ValueError('V4.1 FP8 layer capture requires the TE graph backend')
                    capture_kwargs = dict(
                        fp8_enabled=not is_first_last_bf16_layer(c, layer_id),
                        fp8_recipe=get_fp8_recipe(c),
                        fp8_group=self.pg_collection.tp,
                    )
                self.graphs[graph_key] = StatefulGraphs(
                    region,
                    hidden_states,
                    values,
                    slots=context.capture['slots'],
                    backend=context.capture['backend'],
                    capture_kwargs=capture_kwargs,
                )
                self._next_slot[graph_key] = (0, context.capture['slots'])
            graph = self.graphs.get(graph_key)
            if graph is None:
                if recompute and c.fp8:
                    result = te_checkpoint(
                        region,
                        False,
                        get_cuda_rng_tracker,
                        self.pg_collection.tp,
                        hidden_states,
                        *region.inputs.pack(values),
                        use_reentrant=False,
                    )
                    hidden_states, values = region.restore_result(result, values)
                else:
                    hidden_states, values = region.run(hidden_states, values, recompute=recompute)
            else:
                slot, slots = self._next_slot[graph_key]
                hidden_states, values = graph.run(hidden_states, values, slot=slot)
                self._next_slot[graph_key] = ((slot + 1) % slots, slots)
            csa, mhc, _ = codec.restore(values, layer_id + 1)
            objective = values.pop('indexer_loss', None)
            context.cross_layer_state, context.mhc_state, context.values = csa, mhc, values
        else:
            csa.indexer_loss = None
            if getattr(layer, "engram", None) is not None:
                hidden_states = layer.engram(
                    hidden_states,
                    lookup=self._lookup_engram(layer.engram, context),
                    add_to_hidden_states=True,
                )
            hidden_states, _ = layer(
                hidden_states, cross_layer_state=csa, mhc_state=context.mhc_state, **kwargs
            )
            objective = csa.indexer_loss
            context.values = codec.snapshot(csa, context.mhc_state, context.values, layer_id + 1)
        if objective is not None:
            logged = objective.detach()
            if c.calculate_per_token_loss:
                count = (
                    context.values['cu'][-1]
                    if codec.shape.packed
                    else codec.shape.sequence * codec.shape.batch
                )
                logged = logged / count
            DSAIndexerLossLoggingHelper.save_loss_to_tracker(
                loss=logged,
                layer_number=layer_id + 1,
                num_layers=c.num_layers,
                reduce_group=self.pg_collection.cp if codec.shape.cp_size > 1 else None,
            )
            hidden_states = DSAIndexerLossAutoScaler.apply(hidden_states, objective)
        # Do not retain retired producers or layer objectives through subsequent forwards.
        csa = context.cross_layer_state
        csa.indexer_loss = None
        if 'global_kv' not in context.values:
            csa.global_kv = csa.global_kv_flat = None
        if 'indexer_k' not in context.values:
            csa.indexer_k = csa.indexer_k_flat = None
        return hidden_states

    def finalize_forward(self, output, context):
        """Keep the ordinary stack contract; the model owns pipeline publication."""
        context.engram_tokens = None
        if not self.post_process and context.token_ids is not None:
            context.values["tokens"] = context.token_ids
        return output


csa2_layer_spec = ModuleSpec(
    module=TransformerLayer,
    submodules=TransformerLayerSubmodules(
        input_layernorm=TENorm,
        self_attention=csa2_attention_spec,
        self_attn_bda=get_bias_dropout_add,
    ),
)


deepseek_v41_stack_spec = replace(
    hybrid_stack_spec,
    submodules=replace(
        hybrid_stack_spec.submodules, csa2_layer=csa2_layer_spec, forward_adapter=DeepSeekV41Adapter
    ),
)
