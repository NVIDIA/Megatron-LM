# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import warnings
from typing import List, Optional, Tuple

import torch
from torch import Tensor

from megatron.core.fp8_utils import is_first_last_bf16_layer
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.inference.utils import InferenceMode
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.ssm.context_parallel.chunkwise import PackedSequenceCPMetadata
from megatron.core.tensor_parallel.random import MHCCheckpointManager
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.enums import CudaGraphModule
from megatron.core.transformer.hyper_connection import HyperConnectionModule
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mhc_recompute import uses_mhc_recompute_attn_cuda_graph_split
from megatron.core.transformer.module import (
    GraphableMegatronModule,
    MegatronModule,
    convert_module_to_dtype_except_fp32_marked,
)
from megatron.core.transformer.transformer_layer import TransformerLayer


class HyperConnectionHybridLayer(GraphableMegatronModule):
    """Layer-boundary mHC wrapper for HybridStack layers.

    Hybrid layers already own their local residual paths. For this initial
    integration we treat each hybrid layer as a single function by aggregating
    n streams to the layer input, running the existing layer, and feeding only
    the layer delta back through mHC expansion. The expansion path intentionally
    uses zero additional dropout because the wrapped hybrid layer has already
    applied its local dropout/residual update before the delta is computed.

    Checkpoint compatibility: this is a *wrapper* (the inner layer is held as
    `self.inner_layer`), so wrapped-layer state_dict keys are nested under
    `inner_layer.` (e.g. `layers.0.inner_layer.input_layernorm.weight` instead
    of `layers.0.input_layernorm.weight`). HybridStack checkpoints saved with
    `enable_hyper_connections=False` cannot be loaded into a model with
    `enable_hyper_connections=True` (and vice versa) without a key-mapping
    migration. Note: this differs from `HyperConnectionTransformerLayer`,
    which subclasses `TransformerLayer` and only adds new sibling fields,
    keeping all base keys stable.

    CUDA graphs: this wrapper subclasses ``GraphableMegatronModule`` so that, with
    ``cuda_graph_impl="transformer_engine"``, wrapped layers are captured per-layer —
    mirroring ``HyperConnectionTransformerLayer`` on the GPT path. Without this, the TE
    graph discovery (``_layer_is_graphable``) only inspects the top-level layer type and
    silently skips every wrapped layer, so an mHC-enabled HybridStack would run entirely
    eager. Capture modes:

    * With ``mhc_recompute_attn_cuda_graph_split``, attention-only TransformerLayer
      wrappers keep mHC aggregation and BDA eager and capture only the inner norm/attention.
      The aggregate writes directly into the one-stream static graph input, including
      during backward recomputation.

    * Without the split, non-MoE inner layers (attention variants, Mamba): the whole
      wrapper forward (mHC aggregate + inner layer + n-stream BDA) is captured as one
      graph. The inner layer's own ``__call__`` graph routing is bypassed during capture
      (see ``_call_inner_layer``) to avoid nested capture.
    * MoE inner layers, when ``moe_router`` is in ``cuda_graph_modules``: the expert
      all-to-all is not graph-safe, so only the deterministic prefix is graphed (mHC
      ``compute_mappings``/``aggregate`` + the inner layer's router/preprocess). The graph
      outputs the router intermediates, the mHC state (``h_post``, ``h_res``) and the
      n-stream residual; on replay the experts run eagerly and the n-stream BDA (eager)
      consumes the inner's raw ``mlp_output_with_bias`` as the layer delta. Routing the
      residual *through the graph* (not reusing the layer input directly in the eager BDA)
      keeps the backward gradient flowing into the captured graph, which is required for
      bit-identical training — again mirroring ``HyperConnectionTransformerLayer``.

    ``_get_submodules_under_cudagraphs`` returns the submodules whose params the wrapper
    graph's manual hooks must drive: the inner norm/attention for split attention, ``[self]``
    for whole-wrapper capture, or the mHC module + the inner router/preprocess submodules
    for partial MoE capture (experts stay eager).
    """

    supports_hybrid_recompute_kwargs = True

    def __init__(self, config: TransformerConfig, layer: MegatronModule) -> None:
        super().__init__(config=config)
        if (
            config.cuda_graph_impl in ("transformer_engine", "full_iteration")
            and config.recompute_granularity == "selective"
            and "mhc" in (config.recompute_modules or [])
            and not uses_mhc_recompute_attn_cuda_graph_split(config)
        ):
            # Warn rather than reject: this combination was constructible before the
            # attention-only split existed and nothing here is known to be wrong, it
            # is just unlikely to pay. Under per-layer Transformer Engine capture the
            # hybrid wrapper captures the mHC producer inside the graph, so that
            # checkpoint's per-microbatch registration is swallowed and its activation
            # is not recovered -- the rest of the mHC recompute group sits outside the
            # graph and still works. The attention-only split keeps the producer eager.
            # No manual dedup: the default warning filter already reports once per
            # (message, category, module, lineno), and a module-level latch would
            # leak across tests.
            warnings.warn(
                "mHC selective recompute with CUDA Graphs (cuda_graph_impl="
                f"{config.cuda_graph_impl!r}) is not validated for HybridStack mHC "
                "layers: per-layer capture takes the mHC producer with it, and "
                "full-iteration capture records the recompute itself, so this "
                "wrapper's aggregate checkpoint is not the saving it is on the GPT "
                "path. The rest of the recompute group is unaffected.",
                UserWarning,
                stacklevel=2,
            )
        self.inner_layer = layer
        self.layer_number = layer.layer_number
        if self._uses_mhc_recompute_attn_cuda_graph_split() and (
            not isinstance(layer.cross_attention, IdentityOp)
            or not isinstance(layer.mlp, IdentityOp)
        ):
            raise ValueError(
                "Hybrid mHC attention CUDA Graph split requires an attention-only "
                "TransformerLayer with IdentityOp cross-attention and MLP."
            )
        self._offload_module_in_cuda_graph_cached: Optional[bool] = None
        self.hyper_connection = HyperConnectionModule(config=config, layer_number=self.layer_number)
        # This wrapper is the TE graph callable, so it owns the captured offload event
        # boundary. Reuse the inner Transformer's interface instead of reaching across
        # modules for its private factory. Only TransformerLayer-backed wrappers can
        # report an offload boundary in ``offload_module_in_cuda_graph``.
        self.off_interface = layer.off_interface if isinstance(layer, TransformerLayer) else None
        if config.params_dtype is not None:
            convert_module_to_dtype_except_fp32_marked(self.hyper_connection, config.params_dtype)
        if hasattr(layer, 'tp_group'):
            self.tp_group = layer.tp_group

    def _uses_mhc_recompute_attn_cuda_graph_split(self) -> bool:
        """Whether this wrapper captures an inner attention consumer after eager mHC."""
        return (
            uses_mhc_recompute_attn_cuda_graph_split(self.config)
            and isinstance(self.inner_layer, TransformerLayer)
            and not (
                isinstance(self.inner_layer.self_attention, IdentityOp)
                and isinstance(self.inner_layer.cross_attention, IdentityOp)
            )
        )

    def get_layer_static_inputs(self, seq_length, micro_batch_size):
        """Use one stream for split attention, and n streams for whole-wrapper capture.

        CUDA graph capture allocates static buffers sized by this method. The base
        returns [s, b, C], but mHC layers carry n-stream hidden states [s, b, n*C].
        Split attention consumes the aggregate and keeps the inner layer's [s, b, C] input.
        """
        if hasattr(self.inner_layer, "get_layer_static_inputs"):
            static_inputs = self.inner_layer.get_layer_static_inputs(seq_length, micro_batch_size)
        else:
            static_inputs = super().get_layer_static_inputs(seq_length, micro_batch_size)
        if self._uses_mhc_recompute_attn_cuda_graph_split():
            return static_inputs
        hs = static_inputs["hidden_states"]
        n = self.config.num_residual_streams
        static_inputs["hidden_states"] = torch.ones(
            (hs.shape[0], hs.shape[1], n * self.config.hidden_size),
            dtype=hs.dtype,
            requires_grad=hs.requires_grad,
            device=hs.device,
        )
        return static_inputs

    def _uses_graph_dynamic_dsa_route(self):
        """Whether the wrapped layer's captured attention consumes route inputs."""
        predicate = getattr(self.inner_layer, "_uses_graph_dynamic_dsa_route", None)
        return bool(predicate is not None and predicate())

    def _decompose_packed_seq_params_to_kwargs(self, kwargs):
        """Decompose PackedSeqParams into tensor kwargs for TE CUDA graphs."""
        packed_seq_params = kwargs.pop('packed_seq_params', None)
        if packed_seq_params is None:
            return
        kwargs['cu_seqlens_q'] = packed_seq_params.cu_seqlens_q
        kwargs['cu_seqlens_kv'] = packed_seq_params.cu_seqlens_kv
        kwargs['cu_seqlens_q_padded'] = packed_seq_params.cu_seqlens_q_padded
        kwargs['cu_seqlens_kv_padded'] = packed_seq_params.cu_seqlens_kv_padded
        from megatron.core.transformer.experimental_attention_variant import cp_balanced_indexer

        if self._uses_graph_dynamic_dsa_route():
            cp_group = self.inner_layer.pg_collection.cp
            if cp_group is None or cp_group.size() != self.config.context_parallel_size:
                raise RuntimeError(
                    "graph-dynamic balanced CP route requires the wrapped layer's explicit CP "
                    f"group to have size {self.config.context_parallel_size}"
                )
            cp_balanced_indexer.validate_graph_dynamic_plan_contract(
                packed_seq_params,
                self.config.context_parallel_size,
                cp_group.rank(),
                self.config.max_seqlen_per_dp_cp_rank,
            )
            cp_balanced_indexer.add_graph_dynamic_plan_to_kwargs(
                packed_seq_params, kwargs, required=True
            )
            self._set_te_cuda_graph_route_replay_state(packed_seq_params)

    def _reconstruct_packed_seq_params_from_kwargs(self, kwargs):
        """Reconstruct THD PackedSeqParams from tensor kwargs in the graph capture path."""
        if 'cu_seqlens_q' not in kwargs:
            return
        max_seqlen = self.config.max_seqlen_per_dp_cp_rank * self.config.context_parallel_size
        from megatron.core.transformer.experimental_attention_variant import cp_balanced_indexer

        graph_dynamic_plan = cp_balanced_indexer.pop_graph_dynamic_plan_from_kwargs(
            kwargs, self.config.context_parallel_size, self.config.max_seqlen_per_dp_cp_rank
        )
        packed_seq_params = PackedSeqParams(
            qkv_format='thd',
            cp_partition_mode=self.config.cp_partition_mode,
            cu_seqlens_q=kwargs.pop('cu_seqlens_q'),
            cu_seqlens_kv=kwargs.pop('cu_seqlens_kv'),
            cu_seqlens_q_padded=kwargs.pop('cu_seqlens_q_padded'),
            cu_seqlens_kv_padded=kwargs.pop('cu_seqlens_kv_padded'),
            max_seqlen_q=max_seqlen,
            max_seqlen_kv=max_seqlen,
            # This Python flag is baked into the captured graph and cannot vary
            # between replay batches. Use the conservative THD-safe branch.
            pad_between_seqs=True,
        )
        if graph_dynamic_plan is not None:
            cp_balanced_indexer.attach_graph_dynamic_plan(packed_seq_params, graph_dynamic_plan)
        elif self._uses_graph_dynamic_dsa_route():
            raise RuntimeError(
                "TE CUDA graph input is missing graph-dynamic balanced CP route metadata."
            )
        kwargs['packed_seq_params'] = packed_seq_params

    def __call__(self, *args, **kwargs):
        # Keep non-tensor recompute state out of TE CUDA graph inputs; the GPT
        # hyper-connection path follows the same pattern.
        self._mhc_recompute_manager = kwargs.pop("mhc_recompute_manager", None)
        return super().__call__(*args, **kwargs)

    def _inner_is_moe(self) -> bool:
        """True when the inner layer is an MoE ``TransformerLayer``. Such layers use the
        GPT-style raw-delta path (feed the inner's ``mlp_output_with_bias`` straight to the
        n-stream BDA) in both the eager forward and the CUDA-graph replay."""
        from megatron.core.transformer.moe.moe_layer import MoELayer

        return isinstance(self.inner_layer, TransformerLayer) and isinstance(
            getattr(self.inner_layer, 'mlp', None), MoELayer
        )

    def _inner_is_partial_moe_capture(self) -> bool:
        """True when the inner layer is MoE and the configured ``cuda_graph_modules`` request
        partial MoE capture (``moe_router``).

        In that case the wrapper does NOT capture the whole forward as one graph (the expert
        all-to-all is not graph-safe). Instead it graphs the deterministic prefix (mHC aggregate
        + the inner layer's router/preprocess) and runs the experts + mHC BDA eagerly — mirroring
        how ``HyperConnectionTransformerLayer`` graphs MoE layers on the GPT path. Whole-wrapper
        capture is still used for non-MoE inner layers (attention variants, Mamba).
        """
        return (
            self._inner_is_moe()
            and bool(self.config.cuda_graph_modules)
            and CudaGraphModule.moe_router in self.config.cuda_graph_modules
        )

    def _compute_inner_offload_module_in_cuda_graph(self) -> bool:
        """Whether the captured inner TransformerLayer contains an offload boundary.

        HybridStack can split attention and MoE into separate TransformerLayers while
        sharing one global config, so require the configured scope to have a concrete
        branch in this split layer.
        """
        if not isinstance(self.inner_layer, TransformerLayer):
            return False
        return self.inner_layer.offload_scope_in_cuda_graph(require_concrete_modules=True)

    def _compute_offload_module_in_cuda_graph(self) -> bool:
        """Compute the effective offload state for this complete TE callable."""
        if self._compute_inner_offload_module_in_cuda_graph():
            return True
        group_tail = self._get_te_cuda_graph_group_tail()
        return bool(
            group_tail is not None and group_tail._compute_inner_offload_module_in_cuda_graph()
        )

    @property
    def offload_module_in_cuda_graph(self) -> bool:
        """Whether TE must join the wrapper graph with offload streams.

        The graph-group tail setter is the only grouping mutation and invalidates
        this replay-hot-path cache whenever it attaches a tail.
        """
        cached = self._offload_module_in_cuda_graph_cached
        if cached is None:
            cached = self._compute_offload_module_in_cuda_graph()
            self._offload_module_in_cuda_graph_cached = cached
        return cached

    def _can_group_te_cuda_graph_with(self, next_layer: MegatronModule) -> bool:
        """Whether this attention layer and the following MoE prefix can share one TE graph.

        HybridStack represents attention and MoE as separate layers, unlike the GPT path where
        both scopes live in one TransformerLayer callable. Group only the matching mHC pair and
        only when HybridStack executes its normal per-layer loop. Full/MHC-selective recompute
        use different layer scheduling and retain the existing per-layer graphs.
        """
        if not isinstance(next_layer, HyperConnectionHybridLayer):
            return False
        if (
            self.config.context_parallel_size > 1
            and self.config.attention_cp_layout != next_layer.config.linear_cp_layout
        ):
            return False
        for field in ("params_dtype", "fp8", "fp8_recipe", "fp4", "fp4_recipe", "quant_recipe"):
            if getattr(self.config, field, None) != getattr(next_layer.config, field, None):
                return False
        if self.config.recompute_granularity == 'full' or (
            self.config.recompute_granularity == 'selective'
            and 'mhc' in (self.config.recompute_modules or [])
        ):
            return False
        if not isinstance(self.inner_layer, TransformerLayer):
            return False
        if (
            self.config.fp8
            and self.config.first_last_layers_bf16
            and not getattr(self.inner_layer, 'is_mtp_layer', False)
            and is_first_last_bf16_layer(self.config, self.layer_number - 1)
            != is_first_last_bf16_layer(self.config, next_layer.layer_number - 1)
        ):
            return False
        is_attention_only = not (
            isinstance(self.inner_layer.self_attention, IdentityOp)
            and isinstance(self.inner_layer.cross_attention, IdentityOp)
        ) and isinstance(self.inner_layer.mlp, IdentityOp)
        return is_attention_only and next_layer._inner_is_partial_moe_capture()

    def _set_te_cuda_graph_group_tail(self, next_layer: MegatronModule) -> None:
        """Attach a non-registered capture-only tail while preserving checkpoint keys."""
        assert self._can_group_te_cuda_graph_with(next_layer)
        # Bypass nn.Module.__setattr__: next_layer remains registered exactly once in
        # HybridStack.layers, so this capture-only reference cannot alter state_dict keys.
        object.__setattr__(self, '_te_cuda_graph_group_tail', next_layer)
        self._offload_module_in_cuda_graph_cached = None

    def _get_te_cuda_graph_group_tail(self) -> Optional['HyperConnectionHybridLayer']:
        """Return the capture-only MoE tail, if discovery grouped this layer."""
        return getattr(self, '_te_cuda_graph_group_tail', None)

    def _get_active_te_cuda_graph_group_tail(self) -> Optional['HyperConnectionHybridLayer']:
        """Return the grouped tail only while this layer is replaying training graphs."""
        if self.training and getattr(self, 'cuda_graphs', None):
            return self._get_te_cuda_graph_group_tail()
        return None

    def parameters(self, recurse: bool = True):
        """Expose grouped-prefix parameters to TE without registering the group tail.

        Transformer Engine derives a graphed callable's autograd input surface from
        ``callable.parameters()``. The grouped MoE tail stays registered only in
        ``HybridStack.layers`` for checkpoint compatibility, so include just its graph-covered
        prefix parameters here. Parent model traversal and ``state_dict`` continue to use the
        unchanged module hierarchy.
        """
        seen = set()
        for param in super().parameters(recurse=recurse):
            seen.add(id(param))
            yield param

        group_tail = self._get_te_cuda_graph_group_tail()
        if not recurse or group_tail is None:
            return
        for submodule in group_tail._get_submodules_under_cudagraphs():
            for param in submodule.parameters():
                if id(param) not in seen:
                    seen.add(id(param))
                    yield param

    def _te_cuda_graph_capture(self, *args, **kwargs):
        """Capture the graph-safe portion of the wrapper forward.

        Split attention captures only the inner norm/attention after eager mHC aggregation.
        Otherwise, for non-MoE inner layers the whole wrapper forward (mHC aggregate + inner layer +
        n-stream BDA) is captured as one graph. For MoE inner layers under ``moe_router``
        partial capture, only the deterministic prefix is graphed: the mHC
        ``compute_mappings``/``aggregate`` followed by the inner layer's router/preprocess.
        The captured outputs are the inner router/preprocess intermediates plus the mHC
        state (``h_post``, ``h_res``) and the aggregated single-stream input needed to
        reconstruct the layer delta on replay. ``context`` is ``None`` for the graphed
        hybrid layer types, so it is dropped (a tuple containing ``None`` cannot be a
        CUDA-graph output).
        """
        self._reconstruct_packed_seq_params_from_kwargs(kwargs)

        # The Hybrid wrapper, not its inner TransformerLayer(s), is the TE graph
        # callable. Place the offload events at this outer boundary so every D2H/H2D
        # stream dependency belongs to the graph being captured.
        offload_in_graph = self.offload_module_in_cuda_graph
        off_interface = self.off_interface
        if offload_in_graph:
            assert off_interface is not None, (
                "offload_module_in_cuda_graph requires a TransformerLayer-backed "
                "Hybrid wrapper with an offload interface."
            )
            if args:
                hidden_states = off_interface.backward_record(args[0])
                args = (hidden_states,) + args[1:]
            else:
                hidden_states = off_interface.backward_record(kwargs.pop('hidden_states'))
                kwargs['hidden_states'] = hidden_states

        cuda_graph_outputs = self._te_cuda_graph_capture_impl(*args, **kwargs)

        if offload_in_graph:
            off_interface.forward_record()
        return cuda_graph_outputs

    def _te_cuda_graph_capture_impl(self, *args, **kwargs):
        """Capture the wrapper body without adding offload boundary events."""
        assert 'cu_seqlens_q' not in kwargs, (
            "Hybrid CUDA graph capture body received raw THD sequence tensors. "
            "The outer capture boundary must reconstruct PackedSeqParams first."
        )

        if self._uses_mhc_recompute_attn_cuda_graph_split():
            return self._forward_mhc_attention_cuda_graph_consumer(*args, **kwargs)

        group_tail = self._get_te_cuda_graph_group_tail()
        if group_tail is not None:
            hidden_states, context = self.forward(*args, **kwargs)
            assert context is None, "Grouped hybrid CUDA graphs do not support cross-attention."
            tail_kwargs = dict(kwargs)
            tail_kwargs.pop("hidden_states", None)
            return group_tail._te_cuda_graph_capture_impl(hidden_states, **tail_kwargs)

        if self._inner_is_partial_moe_capture():
            hidden_states = args[0] if args else kwargs["hidden_states"]
            aggregated, h_res, h_post, residual = self.hyper_connection(
                hidden_states, return_residual=True
            )
            inner_kwargs = dict(kwargs)
            inner_kwargs.pop("hidden_states", None)
            inner_out = list(
                self.inner_layer._te_cuda_graph_capture_impl(aggregated, **inner_kwargs)
            )
            # inner_out = router/preprocess intermediates ending in the inner residual;
            # append the mHC state AND the n-stream `residual` returned by the (graphed)
            # hyper_connection. Routing `residual` through the graph as an output keeps its
            # backward grad flowing into the graph's backward (mirrors
            # HyperConnectionTransformerLayer), instead of a second autograd path the captured
            # backward does not account for. The experts' raw mlp_output_with_bias (produced
            # on replay) is the layer delta, so `aggregated` need not be captured.
            return tuple(inner_out) + (h_post, h_res, residual)

        hidden_states, context = self.forward(*args, **kwargs)
        cuda_graph_outputs = [hidden_states]
        if context is not None:
            cuda_graph_outputs.append(context)
        return tuple(cuda_graph_outputs)

    def _te_cuda_graph_replay(self, *args, **kwargs):
        """Replay the captured graph and restore the (hidden_states, context) contract.

        Split attention: run mHC aggregation eagerly into the static graph input, replay
        the inner attention, and apply the n-stream BDA eagerly.

        Non-MoE inner layers: the whole wrapper forward was captured, so the only graph
        output is the layer's n-stream hidden_states; re-append ``None`` for context.

        MoE inner layers (partial capture): replay the graphed prefix, then run the
        experts eagerly and apply the mHC n-stream BDA — reproducing exactly the eager
        wrapper tail (``layer_delta = layer_output - aggregated`` then
        ``fused_h_res_h_post_bda``), just with the deterministic prefix graphed.

        Captured ``core_attn`` offload synchronization uses the wrapper-level events in
        ``_te_cuda_graph_capture`` and does not use the delayed-offload queue. Delayed
        expert offload is intentionally outside the Hybrid CUDA Graph support in this
        change, so this wrapper must not enter or flush that queue around replay.
        """
        try:
            self._decompose_packed_seq_params_to_kwargs(kwargs)
            return self._te_cuda_graph_replay_impl(args, kwargs)
        finally:
            self._te_cuda_graph_route_replay_state = None

    def _forward_mhc_attention_cuda_graph_consumer(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        sequence_len_offset: Optional[Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        **_unused_kwargs,
    ):
        """Capture the raw attention branch, leaving both mHC operations eager."""
        output_with_bias, _, _ = self.inner_layer._forward_self_attention_output_with_bias(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            rotary_pos_emb=rotary_pos_emb,
            sequence_len_offset=sequence_len_offset,
            packed_seq_params=packed_seq_params,
        )
        output, bias = output_with_bias
        return (output,) if bias is None else (output, bias)

    def _get_te_cuda_graph_replay_args(self, *args, **kwargs):
        """Use the inner attention's mask normalization for a split graph."""
        if self._uses_mhc_recompute_attn_cuda_graph_split():
            return self.inner_layer._get_te_cuda_graph_replay_args(*args, **kwargs)
        return super()._get_te_cuda_graph_replay_args(*args, **kwargs)

    def _replay_mhc_attention_cuda_graph(self, args, kwargs):
        """Direct-write eager mHC into the graph input, replay attention, then apply BDA."""
        kwargs = kwargs.copy()
        if len(args) > 1 or (args and "hidden_states" in kwargs):
            raise ValueError("Hybrid mHC attention CUDA Graph expects one hidden_states argument")
        hidden_states = args[0] if args else kwargs.pop("hidden_states")

        graph_kwargs = {
            name: kwargs[name]
            for name in ("attention_mask", "rotary_pos_emb", "sequence_len_offset", "padding_mask")
            if name in kwargs and (kwargs[name] is not None or name == "attention_mask")
        }
        # The outer replay boundary already decomposed packed metadata and resolved its slot.
        graph_kwargs.update(
            {
                name: kwargs[name]
                for name in (
                    "cu_seqlens_q",
                    "cu_seqlens_kv",
                    "cu_seqlens_q_padded",
                    "cu_seqlens_kv_padded",
                    "dsa_cp_graph_layout_buffer",
                    "dsa_cp_graph_route_buffer",
                )
                if name in kwargs
            }
        )

        manager = getattr(self, '_mhc_recompute_manager', None)
        output_slot = None
        if manager is not None:
            output_slot = manager.mhc_arena.bind_external_slot(
                ("attention", self.layer_number, "aggregate", 0),
                self.get_te_cuda_graph_static_hidden_input(),
            )
        aggregated, h_res, h_post, residual = self.hyper_connection(
            hidden_states,
            mhc_recompute_manager=manager,
            output_slot=output_slot,
            return_residual=True,
        )
        outputs = super()._te_cuda_graph_replay(aggregated, **graph_kwargs)
        if isinstance(outputs, Tensor):
            outputs = (outputs,)
        if len(outputs) not in (1, 2):
            raise RuntimeError(
                "Hybrid mHC attention CUDA Graph must return output and optional bias"
            )
        output_with_bias = (outputs[0], outputs[1] if len(outputs) == 2 else None)
        hidden_states = self._forward_mhc_post(
            aggregated,
            h_res,
            h_post,
            residual,
            output_with_bias,
            self.inner_layer.hidden_dropout,
            self.inner_layer.config.bias_dropout_fusion,
            manager,
        )
        return hidden_states, None

    def _te_cuda_graph_replay_impl(self, args, kwargs):
        """Replay the wrapper graph, then run any eager continuation."""
        if self._uses_mhc_recompute_attn_cuda_graph_split():
            return self._replay_mhc_attention_cuda_graph(args, kwargs)

        group_tail = self._get_te_cuda_graph_group_tail()
        cuda_graph_output = list(super()._te_cuda_graph_replay(*args, **kwargs))

        if group_tail is not None:
            return group_tail._resume_partial_moe_cuda_graph(cuda_graph_output)

        if self._inner_is_partial_moe_capture():
            return self._resume_partial_moe_cuda_graph(cuda_graph_output)

        return cuda_graph_output[0], None

    def _resume_partial_moe_cuda_graph(self, out: List[Tensor]) -> Tuple[Tensor, Optional[Tensor]]:
        """Run the eager expert/BDA tail from captured router/preprocess outputs."""
        assert self._inner_is_partial_moe_capture()
        residual = out.pop()  # n-stream [s, b, n*C] — graph output (see capture)
        h_res = out.pop()
        h_post = out.pop()
        # Resume the inner MoE experts eagerly to the raw delta (mlp_output_with_bias),
        # then let the n-stream BDA own the residual — identical to the eager forward
        # (`_call_inner_transformer_layer_without_local_bda` fast path →
        # fused_h_res_h_post_bda), just with the router/preprocess prefix graphed.
        mlp_output_with_bias = self.inner_layer.resume_moe_experts_after_partial_cudagraph(out)
        hidden_states = self.hyper_connection.fused_h_res_h_post_bda(
            h_res,
            residual,
            h_post,
            mlp_output_with_bias,
            dropout_prob=self.inner_layer.hidden_dropout,
            training=self.training,
            fused=self.inner_layer.config.bias_dropout_fusion,
            manager=None,
        )
        if (
            self.config.fp32_residual_connection
            and self.config.params_dtype is not None
            and hidden_states.dtype != self.config.params_dtype
        ):
            hidden_states = hidden_states.to(self.config.params_dtype)
        return hidden_states, None

    def _get_submodules_under_cudagraphs(self):
        """Submodules whose params are driven by the wrapper graph's manual hooks.

        Whole-wrapper capture covers the entire wrapper (``[self]``, the base default).
        Split attention covers only the inner norm/attention; mHC keeps its eager hooks.
        For partial MoE capture only the graphed prefix is covered — the mHC module plus
        the inner layer's router/preprocess submodules — so the experts (run eagerly)
        keep their normal forward hooks.
        """
        if self._uses_mhc_recompute_attn_cuda_graph_split():
            return [self.inner_layer.input_layernorm, self.inner_layer.self_attention]
        if self._inner_is_partial_moe_capture():
            submodules = [
                self.hyper_connection,
                *self.inner_layer._get_submodules_under_cudagraphs(),
            ]
        else:
            submodules = super()._get_submodules_under_cudagraphs()

        group_tail = self._get_te_cuda_graph_group_tail()
        if group_tail is not None:
            submodules += group_tail._get_submodules_under_cudagraphs()
        return submodules

    def mamba_state_shapes_per_request(self) -> Optional[Tuple[Tuple[int], Tuple[int]]]:
        """Delegate Mamba inference state shape requests to the wrapped layer."""
        if hasattr(self.inner_layer, 'mamba_state_shapes_per_request'):
            return self.inner_layer.mamba_state_shapes_per_request()
        mixer = getattr(self.inner_layer, 'self_attention', None)
        if mixer is not None and hasattr(mixer, 'mamba_state_shapes_per_request'):
            return mixer.mamba_state_shapes_per_request()
        return None

    def _call_inner_layer(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        inference_context: Optional[BaseInferenceContext],
        rotary_pos_emb: Optional[Tensor],
        sequence_len_offset: Optional[Tensor],
        packed_seq_params: Optional[PackedSeqParams],
        padding_mask: Optional[Tensor],
        input_ids: Optional[Tensor] = None,
        packed_sequence_cp_metadata: Optional[PackedSequenceCPMetadata] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        # When this wrapper is itself being CUDA-graph captured, the inner layer
        # must run as a plain forward: routing through its ``__call__`` would
        # trigger nested TE graph capture (the inner layer is also a
        # GraphableMegatronModule). During eager steps we keep ``__call__`` so the
        # inner layer's forward pre-hooks (e.g. param all-gather) fire normally;
        # under graph replay these are driven by the wrapper's manual hooks.
        from megatron.core.transformer.cuda_graphs import is_graph_capturing

        inner = self.inner_layer.forward if is_graph_capturing() else self.inner_layer

        if isinstance(self.inner_layer, TransformerLayer):
            output = inner(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                inference_context=inference_context,
                rotary_pos_emb=rotary_pos_emb,
                sequence_len_offset=sequence_len_offset,
                packed_seq_params=packed_seq_params,
                padding_mask=padding_mask,
                input_ids=input_ids,
                _called_from_hybrid_mhc_wrapper=True,
            )
        else:
            # Non-transformer layers (e.g. MambaLayer; GatedDeltaNet which does
            # accept `sequence_len_offset` is currently always wrapped inside a
            # TransformerLayer spec, so it takes the branch above) do not accept
            # rotary_pos_emb / sequence_len_offset / padding_mask — pass only
            # the common arguments. New layer types that consume any of these
            # must add explicit handling here.
            extra_kwargs = {}
            if packed_sequence_cp_metadata is not None:
                extra_kwargs["packed_sequence_cp_metadata"] = packed_sequence_cp_metadata
            output = inner(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                inference_context=inference_context,
                packed_seq_params=packed_seq_params,
                **extra_kwargs,
            )

        if isinstance(output, tuple):
            context = output[1] if len(output) > 1 else None
            return output[0], context
        return output, None

    def _call_inner_transformer_layer_without_local_bda(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        inference_context: Optional[BaseInferenceContext],
        rotary_pos_emb: Optional[Tensor],
        sequence_len_offset: Optional[Tensor],
        packed_seq_params: Optional[PackedSeqParams],
        padding_mask: Optional[Tensor],
        input_ids: Optional[Tensor] = None,
        mhc_recompute_manager: Optional[MHCCheckpointManager] = None,
    ) -> Optional[Tuple[Tuple[Tensor, Optional[Tensor]], Optional[Tensor], float, bool]]:
        """Return a raw TransformerLayer branch output when the wrapped layer is split.

        Hybrid DSv4 layers are usually attention-only (`W/C/H/D`) or MLP/MoE-only (`-/E`)
        TransformerLayer instances. For those layers, skip the inner layer's local
        residual+BDA and feed the raw branch output directly into the mHC BDA, matching the
        GPT mHC path and avoiding a residual add followed by `layer_output - aggregated`.
        """
        if not isinstance(self.inner_layer, TransformerLayer):
            return None

        layer = self.inner_layer
        if InferenceMode.is_active() and layer.config.inference_fuse_tp_communication:
            return None

        has_attention = not isinstance(layer.self_attention, IdentityOp)
        has_cross_attention = not isinstance(layer.cross_attention, IdentityOp)
        has_mlp = not isinstance(layer.mlp, IdentityOp)

        if has_cross_attention or has_attention == has_mlp:
            return None

        if has_attention:
            output_with_bias, attn_norm_manager, residual = (
                layer._forward_self_attention_output_with_bias(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    inference_context=inference_context,
                    rotary_pos_emb=rotary_pos_emb,
                    packed_seq_params=packed_seq_params,
                    sequence_len_offset=sequence_len_offset,
                    mhc_recompute_manager=mhc_recompute_manager,
                )
            )
            output_with_bias = layer._group_offload_output_with_bias(
                output_with_bias, attn_norm_manager, forced_released_tensors=[residual]
            )
            return output_with_bias, None, layer.hidden_dropout, layer.config.bias_dropout_fusion

        output_with_bias, residual = layer._forward_mlp_output_with_bias(
            hidden_states,
            inference_context=inference_context,
            padding_mask=padding_mask,
            input_ids=input_ids,
            packed_seq_params=packed_seq_params,
            mhc_recompute_manager=mhc_recompute_manager,
        )
        # This fast path bypasses TransformerLayer._forward_post_mlp(), which normally
        # discards the selective pre-MLP layernorm checkpoint before MLP backward.
        if layer.recompute_pre_mlp_layernorm or (
            mhc_recompute_manager is not None and layer.mhc_checkpoint_pre_mlp_layernorm
        ):
            layer.pre_mlp_norm_checkpoint.discard_output_and_register_recompute(output_with_bias[0])
        if layer.mlp_norm_manager is not None:
            output_with_bias = layer._group_offload_output_with_bias(
                output_with_bias, layer.mlp_norm_manager, forced_released_tensors=[residual]
            )
            layer.mlp_norm_manager = None
        return output_with_bias, None, layer.hidden_dropout, layer.config.bias_dropout_fusion

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        sequence_len_offset: Optional[Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        padding_mask: Optional[Tensor] = None,
        input_ids: Optional[Tensor] = None,
        mhc_recompute_manager=None,
        packed_sequence_cp_metadata: Optional[PackedSequenceCPMetadata] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Run the wrapped hybrid layer through one layer-boundary mHC update.

        ``attention_mask`` defaults to ``None`` so that CUDA-graph capture, which
        calls this forward with only the static ``hidden_states`` input, does not
        fail on a missing positional argument (causal masking is inferred by the
        attention backend when the mask is ``None``).
        """

        if mhc_recompute_manager is None:
            mhc_recompute_manager = getattr(self, '_mhc_recompute_manager', None)

        aggregated, h_res, h_post, residual = self.hyper_connection(
            hidden_states, mhc_recompute_manager=mhc_recompute_manager, return_residual=True
        )
        fast_path_result = self._call_inner_transformer_layer_without_local_bda(
            aggregated,
            attention_mask,
            inference_context,
            rotary_pos_emb,
            sequence_len_offset,
            packed_seq_params,
            padding_mask,
            input_ids,
            mhc_recompute_manager=mhc_recompute_manager,
        )

        if fast_path_result is None:
            layer_output, context = self._call_inner_layer(
                aggregated,
                attention_mask,
                inference_context,
                rotary_pos_emb,
                sequence_len_offset,
                packed_seq_params,
                padding_mask,
                input_ids,
                packed_sequence_cp_metadata=packed_sequence_cp_metadata,
            )
            # The inner hybrid layer already applied its own local residual/dropout, so
            # it returns `aggregated + f(aggregated)`. We feed only the function
            # delta `f(aggregated)` into the n-stream BDA so it does not double-count
            # the residual that mHC owns.
            if self.config.fp32_residual_connection and aggregated.dtype != layer_output.dtype:
                aggregated = aggregated.to(layer_output.dtype)
            layer_output_with_bias = (layer_output - aggregated, None)
            dropout_prob = 0.0
            bias_dropout_fusion = False
        else:
            layer_output_with_bias, context, dropout_prob, bias_dropout_fusion = fast_path_result

        hidden_states = self._forward_mhc_post(
            aggregated,
            h_res,
            h_post,
            residual,
            layer_output_with_bias,
            dropout_prob,
            bias_dropout_fusion,
            mhc_recompute_manager,
        )
        return hidden_states, context

    def _forward_mhc_post(
        self,
        aggregated: Tensor,
        h_res: Tensor,
        h_post: Tensor,
        residual: Tensor,
        layer_output_with_bias: Tuple[Tensor, Optional[Tensor]],
        dropout_prob: float,
        bias_dropout_fusion: bool,
        mhc_recompute_manager: Optional[MHCCheckpointManager],
    ) -> Tensor:
        """Apply the same mHC group boundary and output dtype in eager and graph replay."""
        layer_output = layer_output_with_bias[0]
        # Sanity check: this contract requires the branch output to preserve shape;
        # any mismatch indicates a future layer type is breaking the residual assumption
        # and would silently corrupt the n-stream state.
        if layer_output.shape != aggregated.shape:
            raise RuntimeError(
                "HyperConnectionHybridLayer requires wrapped branches to preserve "
                f"hidden-state shape. Got {tuple(layer_output.shape)} from wrapped branch "
                f"vs {tuple(aggregated.shape)} input."
            )
        is_last_in_recompute_block = bool(
            mhc_recompute_manager is not None
            and getattr(mhc_recompute_manager, "is_last_layer_in_recompute_block", False)
        )
        mhc_bda_manager = None if is_last_in_recompute_block else mhc_recompute_manager

        hidden_states = self.hyper_connection.fused_h_res_h_post_bda(
            h_res,
            residual,
            h_post,
            layer_output_with_bias,
            dropout_prob=dropout_prob,
            training=self.training,
            fused=bias_dropout_fusion,
            manager=mhc_bda_manager,
        )
        # In `HyperConnectionTransformerLayer` the n-stream output stays in compute
        # dtype because the post-attention `x` is in compute dtype. In the hybrid
        # wrapper, `layer_delta` may be fp32 (when `fp32_residual_connection=True`
        # or an inner layer upcasts), so `fused_h_res_h_post_bda`'s `output.to(x.dtype)`
        # would leave the result in fp32 and silently propagate fp32 n-stream
        # hidden states to every subsequent layer (~2x activation memory). Restore
        # the compute-dtype contract here.
        if (
            self.config.fp32_residual_connection
            and self.config.params_dtype is not None
            and hidden_states.dtype != self.config.params_dtype
        ):
            hidden_states = hidden_states.to(self.config.params_dtype)
        return hidden_states
