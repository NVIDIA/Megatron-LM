# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""V4.1 execution on HybridStack, with shared state explicit at every replay boundary."""

from dataclasses import dataclass, field, replace

import torch
from torch import nn

from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_stack_adapter import HybridStackForwardContext
from megatron.core.transformer.experimental_attention_variant.csa2_module_spec import (
    csa2_layer_spec,
)
from megatron.core.transformer.experimental_attention_variant.dsa import (
    DSAIndexerLossAutoScaler,
    DSAIndexerLossLoggingHelper,
)
from megatron.core.transformer.state_boundary import TensorField
from megatron.core.transformer.stateful_module import StatefulGraphs, StatefulModule

from .state import V41StateCodec


@dataclass
class V41ForwardContext(HybridStackForwardContext):
    """Microbatch-local tensors; a module never caches this object."""

    codec: V41StateCodec | None = None
    values: dict = field(default_factory=dict)
    hashes: torch.Tensor | None = None
    hasher: nn.Module | None = None
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
        engram = getattr(self.layer, 'engram', None)
        if engram is not None:
            mask = csa.thd_layout.valid_tokens.unsqueeze(0) if packed is not None else None
            hidden = engram(hidden, values['hashes'], mask)
        hidden, _ = self.layer(
            hidden_states=hidden,
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
        context.cross_layer_state, context.mhc_state, _ = context.codec.restore(
            context.values, self.start
        )
        return context

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
            values['hashes'] = context.hashes
            inputs = (
                *inputs,
                TensorField(
                    'hashes', tuple(context.hashes.shape), torch.int64, 'contiguous', False
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
        if getattr(layer, 'engram', None) is not None and context.hashes is None:
            layout = csa.thd_layout
            context.hashes = context.hasher(
                context.values['tokens'],
                None if layout is None else layout.valid_tokens.unsqueeze(0),
                None if layout is None else layout.sequence_ids.unsqueeze(0),
                cp_group=self.pg_collection.cp,
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
                if getattr(layer, 'engram', None) is not None:
                    raise ValueError(
                        'Variable-split Engram EP lookup remains outside layer capture'
                    )
                self.graphs[graph_key] = StatefulGraphs(
                    region,
                    hidden_states,
                    values,
                    slots=context.capture['slots'],
                    backend=context.capture['backend'],
                )
                self._next_slot[graph_key] = (0, context.capture['slots'])
            graph = self.graphs.get(graph_key)
            if graph is None:
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
            if getattr(layer, 'engram', None) is not None:
                layout = csa.thd_layout
                hidden_states = layer.engram(
                    hidden_states,
                    context.hashes,
                    None if layout is None else layout.valid_tokens.unsqueeze(0),
                )
            hidden_states, _ = layer(
                hidden_states=hidden_states,
                cross_layer_state=csa,
                mhc_state=context.mhc_state,
                **kwargs,
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
        if 'tokens' not in context.values:
            context.hashes = None
        return hidden_states

    def finalize_forward(self, output, context):
        """Keep the ordinary stack contract; the model owns pipeline publication."""
        return output


deepseek_v41_stack_spec = replace(
    hybrid_stack_spec,
    submodules=replace(
        hybrid_stack_spec.submodules, csa2_layer=csa2_layer_spec, forward_adapter=DeepSeekV41Adapter
    ),
)
