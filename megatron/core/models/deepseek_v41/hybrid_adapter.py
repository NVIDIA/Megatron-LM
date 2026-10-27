# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""V4.1 execution on HybridStack, with shared state explicit at every replay boundary."""

from dataclasses import dataclass, field, replace

import torch
from torch import nn

from megatron.core.extensions.transformer_engine import te_checkpoint
from megatron.core.fp8_utils import get_fp8_context, get_fp8_recipe, is_first_last_bf16_layer
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_stack_adapter import HybridStackForwardContext
from megatron.core.tensor_observation import observe_layer_residuals
from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
from megatron.core.transformer.experimental_attention_variant.csa2_module_spec import (
    csa2_layer_spec,
)
from megatron.core.transformer.experimental_attention_variant.dsa import (
    DSAIndexerLossAutoScaler,
    DSAIndexerLossLoggingHelper,
)
from megatron.core.transformer.state_boundary import TensorField
from megatron.core.transformer.stateful_module import StatefulGraphs, StatefulModule

from .graph import attention_region, router_region
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
        # Checkpoint replay runs after the stack's per-layer FP8 context has exited.
        with get_fp8_context(self.layer.config, boundary):
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


class _LayerGroupRegion(nn.Module):
    """Replay a local layer group with fresh shared state and explicit auxiliary losses."""

    def __init__(self, layers, codec, kwargs, loss_layers):
        super().__init__()
        self.regions = nn.ModuleList(
            _LayerRegion(layer, codec, kwargs, layer.layer_number - 1 in loss_layers)
            for layer in layers
        )
        self.training = layers[0].training

    def forward(self, hidden, values):
        """Keep intermediate producers live inside the group, exporting only its boundary."""
        hashes = values.get('hashes')
        objectives = {}
        for region in self.regions:
            if getattr(region.layer, 'engram', None) is not None:
                values = {**values, 'hashes': hashes}
            accumulator = hidden
            hidden, values = region(hidden, values)
            observe_layer_residuals(region.layer, accumulator, hidden)
            if region.loss:
                objectives[f'indexer_loss_{region.layer.layer_number - 1}'] = values.pop(
                    'indexer_loss'
                )
        return hidden, {**values, **objectives}


class DeepSeekV41Adapter:
    """Model-owned eager, graph and checkpoint execution of explicit V4.1 state."""

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
        self._automatic_capture = None
        self._make_forward_pre_hook = None

    def setup_cuda_graph_hooks(self, make_hook_func):
        """Retain the common DDP hook factory without retaining microbatch tensors."""
        self._make_forward_pre_hook = make_hook_func
        for graph in self.graphs.values():
            graph.setup_manual_hooks(make_hook_func)

    def prepare_cuda_graphs(self, *, slots, backend):
        """Arm CLI-selected consumer profiles after the common training warmup."""
        scopes = {getattr(scope, 'name', scope) for scope in self.config.cuda_graph_modules}
        if not scopes or not scopes <= {'attn', 'moe_router'}:
            raise ValueError('V4.1 automatic graphs require explicit attn/moe_router scopes')
        if self.config.recompute_granularity == 'full':
            raise ValueError('Full recompute and layer capture are separate execution profiles')
        self._automatic_capture = dict(
            layers={
                i
                for i in range(self.start, self.end)
                if ('attn' if i % 2 == 0 else 'moe_router') in scopes
            },
            slots=slots,
            backend=backend,
        )

    def has_cuda_graphs(self):
        """Report completed captures rather than merely an armed lazy profile."""
        return bool(self.graphs)

    def reset_cuda_graphs(self):
        """Release completed profiles through their public slot lifecycle."""
        for graph in self.graphs.values():
            graph.close()
        self.graphs.clear()
        self._next_slot.clear()
        self._automatic_capture = None
        self._make_forward_pre_hook = None

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

    def _has_indexer_loss(self, layer):
        return (
            layer.layer_number - 1 in self.config.csa2_index_source_layers
            and layer.training
            and torch.is_grad_enabled()
            and (self.config.dsa_indexer_loss_coeff or 0) > 0
        )

    def _prepare_hashes(self, context):
        if context.hashes is not None:
            return
        layout = context.cross_layer_state.thd_layout
        context.hashes = context.hasher(
            context.values['tokens'],
            None if layout is None else layout.valid_tokens.unsqueeze(0),
            None if layout is None else layout.sequence_ids.unsqueeze(0),
            cp_group=self.pg_collection.cp,
        )

    def _run_graph(self, key, region, hidden, values, capture):
        """Use the shared graph owner for capture, replay and in-flight slot protection."""
        if capture is not None:
            if key in self.graphs:
                raise ValueError('This V4.1 graph profile is already captured')
            capture_kwargs = {}
            if self.config.fp8:
                if capture['backend'] != 'transformer_engine':
                    raise ValueError('V4.1 FP8 layer capture requires the TE graph backend')
                capture_kwargs = dict(
                    fp8_enabled=not is_first_last_bf16_layer(self.config, key[0]),
                    fp8_recipe=get_fp8_recipe(self.config),
                    fp8_group=self.pg_collection.tp,
                )
            self.graphs[key] = StatefulGraphs(
                region,
                hidden,
                values,
                slots=capture['slots'],
                backend=capture['backend'],
                capture_kwargs=capture_kwargs,
                make_forward_pre_hook=self._make_forward_pre_hook,
            )
            self._next_slot[key] = (0, capture['slots'])
        graph = self.graphs[key]
        slot, slots = self._next_slot[key]
        result = graph.run(hidden, values, slot=slot)
        self._next_slot[key] = ((slot + 1) % slots, slots)
        return result

    def _run_consumer_graph(self, layer, hidden, context, kwargs, loss, key, capture):
        """Keep mHC checkpoints and eager parameter hooks outside the consumer region."""
        codec, inner = context.codec, layer.inner_layer
        manager = kwargs.pop('mhc_recompute_manager', None)
        kwargs.pop('packed_seq_params', None)
        aggregated, h_res, h_post, residual = layer.hyper_connection(
            hidden, mhc_state=context.mhc_state, mhc_recompute_manager=manager, return_residual=True
        )
        values = codec.snapshot(
            context.cross_layer_state, context.mhc_state, context.values, key[0]
        )
        if key[0] % 2 == 0:
            region = (
                attention_region(inner, codec, kwargs, loss)
                if capture is not None
                else self.graphs[key].region
            )
            branch, values = self._run_graph(key, region, aggregated, values, capture)
            values['mhc'] = context.mhc_state.pre_mix
            values = {
                field.key: values[field.key] for field in codec.schema(key[0] + 1, loss=loss).fields
            }
            csa, mhc, _ = codec.restore(values, key[0] + 1)
            context.cross_layer_state, context.mhc_state = csa, mhc
            objective = values.pop('indexer_loss', None)
            output = (branch, None)
        else:
            moe = inner.mlp
            if not callable(getattr(moe, 'compute_routing', None)):
                raise ValueError('V4.1 router capture requires an explicit MoE routing interface')
            moe.select_token_dispatcher()
            normalized, _, _ = inner._pre_mlp_layernorm_and_residual(
                aggregated, mhc_recompute_manager=manager
            )
            packed = codec.packed(values)
            normalized, padding, ids, unflatten = inner._maybe_unflatten_for_moe(
                normalized, kwargs.get('padding_mask'), kwargs.get('input_ids'), packed
            )
            if capture is not None:
                region, router_values = router_region(
                    moe, codec, normalized, values, padding_mask=padding, input_ids=ids
                )
            else:
                region = self.graphs[key].region
                router_values = {
                    f.key: values[f.key]
                    for f in region.inputs.fields
                    if f.key not in ('padding_mask', 'input_ids')
                }
                if padding is not None:
                    router_values['padding_mask'] = padding
                if ids is not None:
                    router_values['input_ids'] = ids
            normalized, routed = self._run_graph(key, region, normalized, router_values, capture)
            output = moe(
                normalized,
                router_output=(routed['probs'], routed['routes']),
                padding_mask=padding,
                input_ids=ids,
                packed_seq_params=packed,
            )
            if unflatten is not None:
                output = (inner._maybe_reflatten_from_moe(output[0], packed, unflatten), output[1])
            if inner.recompute_pre_mlp_layernorm or (
                manager is not None and inner.mhc_checkpoint_pre_mlp_layernorm
            ):
                inner.pre_mlp_norm_checkpoint.discard_output_and_register_recompute(output[0])
            objective = None
            values = codec.snapshot(
                context.cross_layer_state, context.mhc_state, values, key[0] + 1
            )
        hidden = layer.combine_branch(
            aggregated, output, h_res, h_post, residual, mhc_recompute_manager=manager
        )
        context.values = values
        return hidden, objective

    def checkpointed_forward(self, layers, hidden_states, context, **kwargs):
        """Honor uniform group sizes or a block count within this PP/VPP stage."""
        c, codec = self.config, context.codec
        if context.capture is not None or self.graphs:
            raise ValueError('Full recompute and layer capture are separate execution profiles')
        uniform = c.recompute_method == 'uniform'
        group_size = c.recompute_num_layers if uniform else 1
        kwargs = dict(kwargs)
        kwargs.pop('packed_seq_params', None)
        for start in range(0, len(layers), group_size):
            group = layers[start : start + group_size]
            if not uniform and start >= c.recompute_num_layers:
                layer = group[0]
                accumulator = hidden_states
                with get_fp8_context(layer.config, layer.layer_number - 1):
                    hidden_states = self.run_layer(
                        layer,
                        hidden_states,
                        context,
                        recompute=False,
                        packed_seq_params=codec.packed(context.values),
                        **kwargs,
                    )
                observe_layer_residuals(layer, accumulator, hidden_states)
                continue
            first, end = group[0].layer_number - 1, group[-1].layer_number
            values = codec.snapshot(
                context.cross_layer_state, context.mhc_state, context.values, first
            )
            inputs = codec.schema(first).fields
            if any(getattr(layer, 'engram', None) is not None for layer in group):
                self._prepare_hashes(context)
                values['hashes'] = context.hashes
                inputs = (
                    *inputs,
                    TensorField('hashes', tuple(context.hashes.shape), torch.int64, 'contiguous'),
                )
            loss_layers = tuple(
                layer.layer_number - 1 for layer in group if self._has_indexer_loss(layer)
            )
            loss_fields = tuple(
                TensorField(f'indexer_loss_{i}', (), torch.float32, 'contiguous', True)
                for i in loss_layers
            )
            region = StatefulModule(
                _LayerGroupRegion(group, codec, kwargs, loss_layers),
                inputs,
                (*codec.schema(end).fields, *loss_fields),
            )
            if c.fp8:
                result = te_checkpoint(
                    region,
                    c.distribute_saved_activations,
                    get_cuda_rng_tracker,
                    self.pg_collection.tp,
                    hidden_states,
                    *region.inputs.pack(values),
                    use_reentrant=False,
                )
                hidden_states, values = region.restore_result(result, values)
            else:
                hidden_states, values = region.run(hidden_states, values, recompute=True)
            objectives = [(i, values.pop(f'indexer_loss_{i}')) for i in loss_layers]
            csa, mhc, _ = codec.restore(values, end)
            context.cross_layer_state, context.mhc_state, context.values = csa, mhc, values
            for layer_id, objective in objectives:
                hidden_states = self._attach_loss(hidden_states, context, objective, layer_id)
            self._retire_state(context)
        return hidden_states

    def run_layer(self, layer, hidden_states, context, *, recompute=None, **kwargs):
        """Run eagerly, with explicit checkpointing, or in a prepared graph slot."""
        c, codec, csa = self.config, context.codec, context.cross_layer_state
        layer_id = layer.layer_number - 1
        loss = self._has_indexer_loss(layer)
        if getattr(layer, 'engram', None) is not None:
            self._prepare_hashes(context)
        if recompute is None:
            recompute = layer.training and c.recompute_granularity == 'full'
        graph_key = (layer_id, codec.shape, loss, layer.training, torch.is_grad_enabled())
        request = context.capture
        if request is None and layer.training and torch.is_grad_enabled():
            request = self._automatic_capture
        capture = (
            request is not None and layer_id in request['layers'] and graph_key not in self.graphs
        )
        if recompute and (capture or graph_key in self.graphs):
            raise ValueError('Full recompute and layer capture are separate execution profiles')
        if not recompute and (capture or graph_key in self.graphs):
            engram = getattr(layer, 'engram', None)
            if engram is not None:
                layout = csa.thd_layout
                hidden_states = engram(
                    hidden_states,
                    context.hashes,
                    None if layout is None else layout.valid_tokens.unsqueeze(0),
                )
            hidden_states, objective = self._run_consumer_graph(
                layer,
                hidden_states,
                context,
                dict(kwargs),
                loss,
                graph_key,
                request if capture else None,
            )
            hidden_states = self._attach_loss(hidden_states, context, objective, layer_id)
            self._retire_state(context)
            return hidden_states
        if recompute:
            region, values = self._prepare_region(layer, hidden_states, context, kwargs, loss)
            if c.fp8:
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
                hidden_states, values = region.run(hidden_states, values, recompute=True)
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
        hidden_states = self._attach_loss(hidden_states, context, objective, layer_id)
        self._retire_state(context)
        return hidden_states

    def _attach_loss(self, hidden_states, context, objective, layer_id):
        c, codec = self.config, context.codec
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
        return hidden_states

    @staticmethod
    def _retire_state(context):
        # Do not retain retired producers or layer objectives through subsequent forwards.
        csa = context.cross_layer_state
        csa.indexer_loss = None
        if 'global_kv' not in context.values:
            csa.global_kv = csa.global_kv_flat = None
        if 'indexer_k' not in context.values:
            csa.indexer_k = csa.indexer_k_flat = None
        if 'tokens' not in context.values:
            context.hashes = None

    def finalize_forward(self, output, context):
        """Keep the ordinary stack contract; the model owns pipeline publication."""
        return output


deepseek_v41_stack_spec = replace(
    hybrid_stack_spec,
    submodules=replace(
        hybrid_stack_spec.submodules, csa2_layer=csa2_layer_spec, forward_adapter=DeepSeekV41Adapter
    ),
)
