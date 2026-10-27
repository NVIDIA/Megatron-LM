# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import warnings
from typing import List, Optional, Tuple

import torch
from torch import Tensor

from megatron.core.fp8_utils import is_first_last_bf16_layer
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.inference.utils import InferenceMode
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.ssm.context_parallel.chunkwise import PackedSequenceCPMetadata
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.enums import CudaGraphModule
from megatron.core.transformer.hyper_connection import HyperConnectionModule
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.module import (
    GraphableMegatronModule,
    MegatronModule,
    convert_module_to_dtype_except_fp32_marked,
)
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.utils import nvtx_range_pop, nvtx_range_push


class HyperConnectionHybridLayer(GraphableMegatronModule):
    """Layer-boundary mHC wrapper for HybridStack layers.

    Hybrid layers already own their local residual paths. Each wrapped layer is
    treated as one function by aggregating n streams to its input, running the
    existing layer, and feeding only the layer delta back through mHC expansion.

    This wrapper nests the inner layer under inner_layer. Checkpoints cannot
    switch between mHC-enabled and ordinary HybridStacks without key migration.

    CUDA graphs: the wrapper is a ``GraphableMegatronModule`` so that, with
    ``cuda_graph_impl="transformer_engine"``, wrapped layers are captured per layer.
    Otherwise TE graph discovery (``_layer_is_graphable``) inspects only the top-level
    layer type and silently leaves every wrapped layer eager. Capture modes:

    * Non-MoE inner layers (attention variants, Mamba): the whole wrapper forward (mHC
      aggregate + inner layer + n-stream BDA) is captured as one graph. The inner layer's
      own ``__call__`` graph routing is bypassed during capture (see ``_call_inner_layer``)
      to avoid nested capture.
    * MoE inner layers, when ``moe_router`` is in ``cuda_graph_modules``: the expert
      all-to-all is not graph-safe, so only the deterministic prefix is graphed (mHC
      ``compute_mappings``/``aggregate`` + the inner layer's router/preprocess). The graph
      outputs the router intermediates, the mHC state (``h_post``, ``h_res``) and the
      n-stream residual; on replay the experts run eagerly and the n-stream BDA (eager)
      consumes the inner's raw ``mlp_output_with_bias`` as the layer delta. Routing the
      residual *through the graph* (not reusing the layer input directly in the eager BDA)
      keeps the backward gradient flowing into the captured graph, which is required for
      bit-identical training.
    * An attention-only wrapper followed by an MoE wrapper under partial capture can be
      grouped into one graph (attention wrapper + MoE router prefix), see
      ``_can_group_te_cuda_graph_with``. HybridStack then skips the grouped MoE layer and
      the head's replay runs its eager expert/BDA tail.

    ``_get_submodules_under_cudagraphs`` returns the submodules whose params the wrapper
    graph's manual hooks must drive: ``[self]`` for whole-wrapper capture, or the mHC
    module + the inner router/preprocess submodules for partial MoE capture (experts stay
    eager).

    Only Transformer Engine per-layer capture uses this; under ``cuda_graph_impl="local"``
    the wrapper creates no graph manager and stays eager, as before.
    """

    def __init__(self, config: TransformerConfig, layer: MegatronModule) -> None:
        super().__init__(config=config)
        if (
            config.cuda_graph_impl == "transformer_engine"
            and config.recompute_granularity == "selective"
            and "mhc" in (config.recompute_modules or [])
        ):
            # Warn rather than reject: nothing here is known to be wrong, it is just unlikely
            # to pay. Under per-layer Transformer Engine capture the wrapper captures the mHC
            # producer inside the graph, so that checkpoint's per-microbatch registration is
            # swallowed and its activation is not recovered -- the rest of the mHC recompute
            # group sits outside the graph and still works. No manual dedup: the default
            # warning filter already reports once per (message, category, module, lineno).
            warnings.warn(
                "mHC selective recompute with per-layer Transformer Engine CUDA Graphs is not "
                "validated for HybridStack mHC layers: the captured wrapper takes the mHC "
                "producer with it, so this wrapper's aggregate checkpoint is not the saving it "
                "is on the eager path. The rest of the recompute group is unaffected.",
                UserWarning,
                stacklevel=2,
            )
        self.inner_layer = layer
        self.layer_number = layer.layer_number
        # Set by __call__ from kwargs: the manager is not a tensor, so it must stay out of the
        # TE CUDA graph inputs; forward reads it back from here.
        self._mhc_recompute_manager = None
        # Set by __call__ while it drives a layer whose Engram lookup it has already applied
        # in front of the CUDA graph (cuda_graph_capture_engram_host); forward then skips it.
        self._engram_applied_by_caller = False
        if config.mhc_connection_variant == "gated_residual":
            from megatron.core.transformer.gated_residual import GatedResidualModule

            self.hyper_connection = GatedResidualModule(
                config=config, layer_number=self.layer_number
            )
        else:
            self.hyper_connection = HyperConnectionModule(
                config=config, layer_number=self.layer_number
            )
        if config.params_dtype is not None:
            convert_module_to_dtype_except_fp32_marked(self.hyper_connection, config.params_dtype)
        if hasattr(layer, 'tp_group'):
            self.tp_group = layer.tp_group

    def create_mcore_cudagraph_manager(self, config):
        """Do not graph the wrapper with the local implementation.

        ``GraphableMegatronModule`` calls this for ``cuda_graph_impl="local"``. Only the
        Transformer Engine per-layer capture is implemented for this wrapper, so it keeps no
        local graph manager and runs eagerly, exactly as the plain-module wrapper did.
        """

    def is_cuda_graph_capturable(self) -> bool:
        """Delegate per-layer capture eligibility to the wrapped layer.

        The wrapper itself adds only the mHC aggregate and BDA, which are capturable; whether
        the layer as a whole may be captured is decided by the inner layer (e.g. an inner
        layer that hosts Engram, or whose attention variant performs host synchronizations).
        """
        if self._offloads_with_cuda_graph():
            return False
        if self._engram_outside_cuda_graph():
            # The Engram lookup runs eagerly in __call__, in front of the graph; whether the
            # rest of the layer may be captured is decided by its other submodules.
            predicate = getattr(self.inner_layer, "submodules_are_cuda_graph_capturable", None)
            return predicate is None or predicate()
        predicate = getattr(self.inner_layer, "is_cuda_graph_capturable", None)
        return predicate is None or predicate()

    def _offloads_with_cuda_graph(self) -> bool:
        """Whether fine-grained activation offloading would interact with this wrapper's graph.

        The wrapper graph does not plant the offload stream events (``backward_record`` /
        ``forward_record``) that ``TransformerLayer._te_cuda_graph_capture`` plants, nor does
        its replay enter the delayed-offload queue. A wrapped layer that offloads a module
        inside the captured scope, or any wrapped layer when offloads are delayed until
        graph replay, therefore stays eager.
        """
        if not self.config.fine_grained_activation_offloading:
            return False
        if self.config.delay_offload_until_cuda_graph:
            return True
        inner = self.inner_layer
        if not isinstance(inner, TransformerLayer):
            return False
        modules = self.config.cuda_graph_modules or []
        has_attention = not isinstance(inner.self_attention, IdentityOp)
        if (
            has_attention
            and (not modules or CudaGraphModule.attn in modules)
            and (inner.offload_core_attn or inner.offload_attn_proj or inner.offload_qkv_linear)
        ):
            return True
        if (
            not inner.is_moe_layer
            and not isinstance(inner.mlp, IdentityOp)
            and (not modules or CudaGraphModule.mlp in modules)
            and inner.offload_mlp_norm
        ):
            return True
        return False

    def _engram_outside_cuda_graph(self) -> bool:
        """Whether this wrapper runs its inner layer's Engram lookup outside the layer graph.

        Opt-in (``cuda_graph_capture_engram_host``). The Engram residual is added to the
        n-stream input before the read gate in either case; the lookup's data-dependent
        shapes and all-to-all simply stay outside the captured region, which then starts at
        the mHC aggregate and covers the rest of the layer.
        """
        return bool(getattr(self.config, "cuda_graph_capture_engram_host", False)) and (
            getattr(self.inner_layer, "engram", None) is not None
        )

    def get_layer_static_inputs(self, seq_length, micro_batch_size):
        """Return the inner layer's static inputs with an n-stream ``hidden_states``.

        CUDA graph capture allocates static buffers sized by this method. The inner layer
        returns [s, b, C], but mHC layers carry n-stream hidden states [s, b, n*C]. A grouped
        MoE tail additionally contributes its router's ``padding_mask`` input.
        """
        static_inputs = self.inner_layer.get_layer_static_inputs(seq_length, micro_batch_size)
        hs = static_inputs["hidden_states"]
        n = self.config.mhc_num_residual_streams
        static_inputs["hidden_states"] = torch.ones(
            (hs.shape[0], hs.shape[1], n * self.config.hidden_size),
            dtype=hs.dtype,
            requires_grad=hs.requires_grad,
            device=hs.device,
        )
        group_tail = self._get_te_cuda_graph_group_tail()
        if group_tail is not None:
            tail_inputs = group_tail.inner_layer.get_layer_static_inputs(
                seq_length, micro_batch_size
            )
            if "padding_mask" in tail_inputs:
                static_inputs["padding_mask"] = tail_inputs["padding_mask"]
        return static_inputs

    def _attention_in_cuda_graph(self) -> bool:
        """Whether the whole-wrapper graph contains the inner layer's attention."""
        predicate = getattr(self.inner_layer, "_attention_in_cuda_graph", None)
        return predicate is not None and predicate()

    def _decompose_packed_seq_params_to_kwargs(self, kwargs):
        """Decompose PackedSeqParams into tensor kwargs for TE CUDA graphs."""
        packed_seq_params = kwargs.pop('packed_seq_params', None)
        if packed_seq_params is None:
            return
        for name, offsets in self._thd_graph_cu_seqlens_kwargs(packed_seq_params).items():
            if self._is_thd_cuda_graph() and (
                offsets is None or offsets.numel() != self.config.thd_max_packed_sequences + 1
            ):
                raise ValueError(
                    "Packed CUDA graph inputs must use the configured metadata capacity."
                )
            kwargs[name] = offsets

    def _reconstruct_packed_seq_params_from_kwargs(self, kwargs):
        """Reconstruct THD PackedSeqParams from tensor kwargs in the graph capture path."""
        if 'cu_seqlens_q' not in kwargs:
            return
        max_seqlen = self.config.max_seqlen_per_dp_cp_rank * self.config.context_parallel_size
        kwargs['packed_seq_params'] = PackedSeqParams(
            qkv_format='thd',
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

    def __call__(self, *args, **kwargs):
        # Keep non-tensor recompute state out of TE CUDA graph inputs; the GPT
        # hyper-connection path follows the same pattern.
        self._mhc_recompute_manager = kwargs.pop("mhc_recompute_manager", None)
        if not self._engram_outside_cuda_graph():
            return super().__call__(*args, **kwargs)

        from megatron.core.transformer.cuda_graphs import is_graph_capturing

        # Run the Engram lookup eagerly here, then hand the post-Engram n-stream tensor to the
        # (captured or eager) rest of the layer. During capture the static sample input
        # stands for that post-Engram tensor, so nothing is applied.
        if not is_graph_capturing():
            hidden_states = args[0] if args else kwargs["hidden_states"]
            nvtx_range_push("HC::engram")
            hidden_states = self.inner_layer._maybe_apply_engram(
                hidden_states, kwargs.get("input_ids"), kwargs.get("packed_seq_params")
            )
            nvtx_range_pop("HC::engram")
            if args:
                args = (hidden_states,) + tuple(args[1:])
            else:
                kwargs["hidden_states"] = hidden_states
        self._engram_applied_by_caller = True
        try:
            return super().__call__(*args, **kwargs)
        finally:
            self._engram_applied_by_caller = False

    def _inner_is_moe(self) -> bool:
        """True when the inner layer is an MoE ``TransformerLayer``. Such layers use the
        raw-delta path (feed the inner's ``mlp_output_with_bias`` straight to the n-stream
        BDA) in both the eager forward and the CUDA-graph replay."""
        from megatron.core.transformer.moe.moe_layer import MoELayer

        return isinstance(self.inner_layer, TransformerLayer) and isinstance(
            getattr(self.inner_layer, 'mlp', None), MoELayer
        )

    def _inner_is_partial_moe_capture(self) -> bool:
        """True when the inner layer is MoE and the configured ``cuda_graph_modules`` request
        partial MoE capture (``moe_router``).

        In that case the wrapper does NOT capture the whole forward as one graph (the expert
        all-to-all is not graph-safe). Instead it graphs the deterministic prefix (mHC
        aggregate + the inner layer's router/preprocess) and runs the experts + mHC BDA
        eagerly. Whole-wrapper capture is still used for non-MoE inner layers (attention
        variants, Mamba).
        """
        return (
            self._inner_is_moe()
            and bool(self.config.cuda_graph_modules)
            and CudaGraphModule.moe_router in self.config.cuda_graph_modules
        )

    def _can_group_te_cuda_graph_with(self, next_layer: MegatronModule) -> bool:
        """Whether this attention layer and the following MoE prefix can share one TE graph.

        HybridStack represents attention and MoE as separate layers, unlike the GPT path where
        both scopes live in one TransformerLayer callable. Group only the matching mHC pair and
        only when HybridStack executes its normal per-layer loop. MHC-selective recompute
        uses different layer scheduling and retains the existing per-layer graphs.
        """
        if not isinstance(next_layer, HyperConnectionHybridLayer):
            return False
        if getattr(next_layer.inner_layer, "engram", None) is not None:
            # A tail's Engram lookup would have to run eagerly in the middle of the group.
            return False
        if self.config.recompute_granularity == 'full' or (
            self.config.recompute_granularity == 'selective'
            and 'mhc' in (self.config.recompute_modules or [])
        ):
            return False
        if not isinstance(self.inner_layer, TransformerLayer):
            return False
        if getattr(getattr(next_layer.inner_layer, "mlp", None), "router", None) is not None and (
            getattr(next_layer.inner_layer.mlp.router, "is_hash_layer", False)
        ):
            # HybridStack passes input_ids only to the layers that consume them; the head
            # would be replayed without the tail's token IDs.
            return False
        if (
            self.config.context_parallel_size > 1
            and self.config.attention_cp_layout != next_layer.config.linear_cp_layout
        ):
            # HybridStack converts the CP layout between the two layers; a group would skip it.
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

        For non-MoE inner layers the whole wrapper forward (mHC aggregate + inner layer +
        n-stream BDA) is captured as one graph. For MoE inner layers under ``moe_router``
        partial capture, only the deterministic prefix is graphed: the mHC
        ``compute_mappings``/``aggregate`` followed by the inner layer's router/preprocess.
        The captured outputs are the inner router/preprocess intermediates plus the mHC
        state (``h_post``, ``h_res``) and the n-stream residual needed to apply the
        n-stream BDA on replay. ``context`` is ``None`` for the graphed hybrid layer types,
        so it is dropped (a tuple containing ``None`` cannot be a CUDA-graph output).
        """
        self._reconstruct_packed_seq_params_from_kwargs(kwargs)
        return self._te_cuda_graph_capture_impl(*args, **kwargs)

    def _te_cuda_graph_capture_impl(self, *args, **kwargs):
        """Capture the wrapper body; ``packed_seq_params`` is already reconstructed."""
        assert 'cu_seqlens_q' not in kwargs, (
            "Hybrid CUDA graph capture body received raw THD sequence tensors. "
            "The outer capture boundary must reconstruct PackedSeqParams first."
        )

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
            # backward grad flowing into the graph's backward, instead of a second autograd
            # path the captured backward does not account for. The experts' raw
            # mlp_output_with_bias (produced on replay) is the layer delta, so `aggregated`
            # need not be captured. The h_res slot is omitted for identity-residual variants
            # (gated_residual), which leave it None -- a None cannot be a CUDA graph output.
            # Keyed on the module rather than on the variant name;
            # _resume_partial_moe_cuda_graph mirrors this.
            hc_state = (h_post, h_res) if self.hyper_connection.produces_h_res else (h_post,)
            return tuple(inner_out) + hc_state + (residual,)

        hidden_states, context = self.forward(*args, **kwargs)
        cuda_graph_outputs = [hidden_states]
        if context is not None:
            cuda_graph_outputs.append(context)
        return tuple(cuda_graph_outputs)

    def _te_cuda_graph_replay(self, *args, **kwargs):
        """Replay the captured graph and restore the (hidden_states, context) contract.

        Non-MoE inner layers: the whole wrapper forward was captured, so the only graph
        output is the layer's n-stream hidden_states; re-append ``None`` for context.

        MoE inner layers (partial capture, also as a grouped tail): replay the graphed
        prefix, then run the experts eagerly and apply the mHC n-stream BDA -- the eager
        wrapper tail with the deterministic prefix graphed.
        """
        kwargs = dict(kwargs)
        self._decompose_packed_seq_params_to_kwargs(kwargs)
        assert kwargs.get('inference_context') is None, (
            "CUDA graph accepts only Tensor inputs. "
            "For inference cuda graph, please use cuda_graph_impl=local instead."
        )
        kwargs.pop('inference_context', None)
        return self._te_cuda_graph_replay_impl(args, kwargs)

    def _te_cuda_graph_replay_impl(self, args, kwargs):
        """Replay the wrapper graph, then run any eager continuation."""
        group_tail = self._get_te_cuda_graph_group_tail()
        # The eager MoE continuation keeps the caller's mask; the graph takes one only when
        # the router layer whose prefix it holds reserved that input at capture.
        padding_mask = kwargs.pop("padding_mask", None)
        mask_owner = (group_tail if group_tail is not None else self).inner_layer
        if isinstance(mask_owner, TransformerLayer):
            hidden_states = args[0] if args else kwargs["hidden_states"]
            graph_padding_mask = mask_owner._te_cuda_graph_padding_mask(
                padding_mask, hidden_states.device
            )
        else:
            graph_padding_mask = padding_mask
        if graph_padding_mask is not None:
            kwargs["padding_mask"] = graph_padding_mask

        cuda_graph_output = list(super()._te_cuda_graph_replay(*args, **kwargs))

        if group_tail is not None:
            return group_tail._resume_partial_moe_cuda_graph(cuda_graph_output, padding_mask)

        if self._inner_is_partial_moe_capture():
            return self._resume_partial_moe_cuda_graph(cuda_graph_output, padding_mask)

        return cuda_graph_output[0], None

    def _get_te_cuda_graph_replay_args(self, *args, **kwargs):
        """Replace a ``None`` attention mask the way ``TransformerLayer`` does.

        A graph that contains attention was captured with a static ``attention_mask`` input
        (SBHD) or with none (static THD). The caller may pass ``None`` when the input has no
        padding; the graph still needs a tensor, so an equivalent all-False mask is used.
        """
        cudagraph_args, cudagraph_kwargs = super()._get_te_cuda_graph_replay_args(*args, **kwargs)
        if (
            'attention_mask' in cudagraph_kwargs
            and cudagraph_kwargs['attention_mask'] is None
            and self._attention_in_cuda_graph()
        ):
            if self._is_thd_cuda_graph():
                cudagraph_kwargs.pop('attention_mask')
            else:
                hidden_states = cudagraph_args[0]
                slen_per_cp = hidden_states.size(0)
                if self.config.sequence_parallel:
                    slen_per_cp *= self.config.tensor_model_parallel_size
                slen = slen_per_cp * self.config.context_parallel_size
                cudagraph_kwargs['attention_mask'] = torch.zeros(
                    (hidden_states.size(1), 1, slen_per_cp, slen),
                    dtype=torch.bool,
                    device=torch.cuda.current_device(),
                )
        return cudagraph_args, cudagraph_kwargs

    def _resume_partial_moe_cuda_graph(
        self, out: List[Tensor], padding_mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Run the eager expert/BDA tail from captured router/preprocess outputs."""
        assert self._inner_is_partial_moe_capture()
        residual = out.pop()  # n-stream [s, b, n*C] -- graph output (see capture)
        # Only packed by variants that produce one; see the capture side.
        h_res = out.pop() if self.hyper_connection.produces_h_res else None
        h_post = out.pop()
        # Resume the inner MoE experts eagerly to the raw delta (mlp_output_with_bias),
        # then let the n-stream BDA own the residual -- identical to the eager forward
        # (`_call_inner_transformer_layer_without_local_bda` fast path ->
        # fused_h_res_h_post_bda), just with the router/preprocess prefix graphed.
        mlp_output_with_bias = self.inner_layer.resume_moe_experts_after_partial_cudagraph(
            out, padding_mask=padding_mask
        )
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
        For partial MoE capture only the graphed prefix is covered -- the mHC module plus
        the inner layer's router/preprocess submodules -- so the experts (run eagerly)
        keep their normal forward hooks. A grouped tail adds its covered prefix.
        """
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
        packed_sequence_cp_metadata: Optional[PackedSequenceCPMetadata],
        padding_mask: Optional[Tensor],
        input_ids: Optional[Tensor] = None,
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
                # This wrapper already added the n-gram memory to the n-stream tensor,
                # before the read gate; the inner layer must not add it a second time.
                # Only passed when Engram is on: an unrecognized non-tensor kwarg would
                # otherwise reach the inner layer's CUDA-graph machinery on every mHC run.
                **({"skip_engram": True} if self.config.engram_enabled else {}),
            )
        else:
            # Mamba-like layers only consume the common HybridStack arguments.
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
        mhc_recompute_manager=None,
    ) -> Optional[Tuple[Tuple[Tensor, Optional[Tensor]], Optional[Tensor], float, bool]]:
        """Return a raw branch output for split Hybrid TransformerLayer instances.

        Hybrid layers are normally attention-only or MLP/MoE-only. For those
        layers, bypass the inner layer's local residual/BDA and let the mHC BDA
        own that operation directly.
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
            packed_seq_params=packed_seq_params,
            input_ids=input_ids,
            mhc_recompute_manager=mhc_recompute_manager,
        )
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
        packed_sequence_cp_metadata: Optional[PackedSequenceCPMetadata] = None,
        mhc_recompute_manager=None,
        input_ids: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Run the wrapped hybrid layer through one layer-boundary mHC update.

        ``attention_mask`` defaults to ``None`` so that CUDA-graph capture, which
        calls this forward with only the static ``hidden_states`` input for layers
        without attention, does not fail on a missing positional argument.
        """
        if mhc_recompute_manager is None:
            mhc_recompute_manager = self._mhc_recompute_manager

        if (
            padding_mask is not None
            and isinstance(self.inner_layer, TransformerLayer)
            and self.inner_layer._padding_mask_shape_seen is None
        ):
            # The fast path below skips TransformerLayer._forward_attention, which records
            # the whole-layer mask shape that TE CUDA-graph capture reserves for the router.
            self.inner_layer._padding_mask_shape_seen = tuple(padding_mask.shape)

        # The n-gram memory is added to the n-stream tensor *before* the read gate: its
        # output belongs to the residual streams, not to the wrapped layer's branch delta.
        # The wrapper has to do it because both inner paths below bypass that part of
        # TransformerLayer.forward -- the fast path calls the sublayer helpers directly,
        # and the fallback path passes skip_engram=True. With cuda_graph_capture_engram_host
        # __call__ has already applied it in front of the graph.
        if (
            getattr(self.inner_layer, "engram", None) is not None
            and not self._engram_applied_by_caller
        ):
            nvtx_range_push("HC::engram")
            hidden_states = self.inner_layer._maybe_apply_engram(
                hidden_states, input_ids, packed_seq_params
            )
            nvtx_range_pop("HC::engram")

        aggregated, h_res, h_post, residual = self.hyper_connection(
            hidden_states, mhc_recompute_manager=mhc_recompute_manager, return_residual=True
        )
        nvtx_range_push("HC::inner")
        fast_path_result = self._call_inner_transformer_layer_without_local_bda(
            aggregated,
            attention_mask,
            inference_context,
            rotary_pos_emb,
            sequence_len_offset,
            packed_seq_params,
            padding_mask,
            input_ids,
            mhc_recompute_manager,
        )

        if fast_path_result is None:
            layer_output, context = self._call_inner_layer(
                aggregated,
                attention_mask,
                inference_context,
                rotary_pos_emb,
                sequence_len_offset,
                packed_seq_params,
                packed_sequence_cp_metadata,
                padding_mask,
                input_ids,
            )
            if self.config.fp32_residual_connection and aggregated.dtype != layer_output.dtype:
                aggregated = aggregated.to(layer_output.dtype)
            layer_output_with_bias = (layer_output - aggregated, None)
            dropout_prob = 0.0
            bias_dropout_fusion = False
        else:
            layer_output_with_bias, context, dropout_prob, bias_dropout_fusion = fast_path_result
        nvtx_range_pop("HC::inner")

        layer_output = layer_output_with_bias[0]
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
        if (
            self.config.fp32_residual_connection
            and self.config.params_dtype is not None
            and hidden_states.dtype != self.config.params_dtype
        ):
            hidden_states = hidden_states.to(self.config.params_dtype)
        return hidden_states, context
