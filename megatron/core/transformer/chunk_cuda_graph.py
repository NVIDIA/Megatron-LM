# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Chunk-wise Transformer Engine CUDA graphs.

With ``cuda_graph_impl="chunk_wise"`` the decoder block of every PP/VPP model chunk (and, on the
last stage of packed-sequence (THD) runs, the post-process block: MTP, LM head and loss) is the
callable handed to Transformer Engine's ``make_graphed_callables`` instead of the individual
layers. One forward graph and one
backward graph are captured per callable and microbatch slot, so activation recompute, MoE
dispatch/combine and the hyper-connection residual streams are all recorded inside one graph and
replayed with a single launch per pass.
"""

from collections import OrderedDict
from contextlib import nullcontext

import torch

from megatron.core import tensor_parallel
from megatron.core.extensions.te_cuda_graphs import TECudaGraphMixin
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import GraphableMegatronModule


class ChunkCudaGraphBlockMixin(TECudaGraphMixin):
    """Make a complete decoder block a Transformer Engine graph callable.

    The mixin must precede ``GraphableMegatronModule`` in the bases. It is the whole-block
    counterpart of the per-layer ``TECudaGraphMixin`` users: the block registers its graph state
    under ``cuda_graph_impl="chunk_wise"`` (its layers register under ``"transformer_engine"``
    and therefore run their ordinary forward inside the block capture). It contributes the static
    capture inputs of the block-level call signature and the capture/replay adapters that convert
    between that signature and the tensor-only interface CUDA graphs accept (``PackedSeqParams``
    is decomposed into its ``cu_seqlens`` tensors, and the pipeline input tensor is exposed as the
    positional graph input). With fine-grained activation offloading the offload manager runs in
    its whole-block capture mode (``PipelineOffloadManager.enter_block_capture``): the D2H/H2D
    copies are part of the captured graphs and every slot's graphs are self-contained.
    """

    _te_cuda_graph_impl = "chunk_wise"
    is_cuda_graph_chunk_callable = True
    # Whether the graphs were captured with the dataloader's attention mask as a static input
    # (get_layer_static_inputs), and the mask a replay passes when the caller has none.
    _chunk_graph_captures_attention_mask = False
    _attention_mask_for_a_missing_mask = None

    def _chunk_cp_group(self):
        """Explicit CP group of the captured block (the post-process block uses its owner's)."""
        pg_collection = getattr(self, "pg_collection", None)
        if pg_collection is None and getattr(self, "_owner", None) is not None:
            pg_collection = getattr(self._owner, "pg_collection", None)
        return getattr(pg_collection, "cp", None)

    def _block_uses_graph_dynamic_dsa_route(self):
        """Whether a DSA layer of this block consumes the balanced-CP per-pack route inputs.

        Deliberately not named like the layer-level ``_uses_graph_dynamic_dsa_route``: the
        per-layer capture's block-owned arena staging does not apply to a block callable. Here
        the two route buffers are ordinary static inputs of the block graph, refreshed per replay
        like ``cu_seqlens``, and the ``PackedSeqParams`` rebuilt inside the graph hands them to
        every DSA layer of the block.
        """
        if not getattr(self.config, "dsa_cp_balance_indexer_graph_dynamic_packs", False):
            return False
        return any(
            module is not self
            and getattr(module, "_uses_graph_dynamic_dsa_route", None) is not None
            and module._uses_graph_dynamic_dsa_route()
            for module in self.modules()
        )

    def _chunk_graph_has_quantized_modules(self):
        """Whether the captured block holds modules the block-level FP8 autocast applies to.

        The decoder block always does. The post-process block overrides this: only its MTP
        layers and an MXFP8 output projection are Transformer Engine modules run under FP8.
        """
        return True

    def _chunk_graph_replay_quantization_context(self):
        """Quantization context of a chunk graph replay (``nullcontext`` without FP8).

        Replay enters the graphed callable directly, so the ``fp8_autocast`` the eager block
        forward opens around its layers (``TransformerBlock.forward``, the MTP layers) never
        runs. Transformer Engine's graphed callable takes all of its FP8 bookkeeping from the
        autocast enclosing the replay: inside one it registers the block's FP8 metas with the
        global amax buffer, the first callable of the autocast owns the backward-side
        ``reduce_and_update_fp8_tensors`` and the autocast exit runs the forward-side one (delayed
        scaling: without the context the amax history is never reduced and every scale keeps its
        capture-time value), and that same first callable forwards ``is_first_microbatch`` to the
        FP8 weight-cache update (every recipe). The graphs themselves are fixed, so the context
        only drives this bookkeeping; a BF16 boundary layer stays BF16 as it was recorded.

        NVFP4 goes through the same TE autocast (``get_fp4_context``): without it the replay
        would never hand ``is_first_microbatch`` to the weight cache and would re-quantize the
        BF16 weights on every microbatch.
        """
        if not self._chunk_graph_has_quantized_modules():
            return nullcontext()
        if getattr(self.config, "fp8", None):
            from megatron.core.fp8_utils import get_fp8_context

            return get_fp8_context(self.config)
        if getattr(self.config, "fp4", None):
            from megatron.core.fp4_utils import get_fp4_context

            return get_fp4_context(self.config)
        return nullcontext()

    def is_graphable_for_scope(self, modules):
        """A block covers none of the per-layer scopes of ``cuda_graph_impl="transformer_engine"``:
        there its layers are the callables (``_te_cuda_graph_layer_is_graphable``)."""
        return False

    def _should_call_te_cudagraph(self):
        """Whether this call captures or replays the block graph.

        Only a block the pipeline schedule drives is a callable; a stack nested in one (the
        hybrid MTP stack, which runs inside the post-process callable) runs its forward.
        """
        # getattr: unit tests call blocks built on stand-in configs.
        if getattr(self.config, "cuda_graph_impl", "none") != self._te_cuda_graph_impl:
            return False
        return self.is_cuda_graph_chunk_callable and super()._should_call_te_cudagraph()

    def _te_cuda_graph_replay_microbatch(self):
        """The schedule sets ``current_microbatch`` on the block (``set_current_microbatch``)."""
        return getattr(self, "current_microbatch", 0)

    def __call__(self, *args, **kwargs):
        # Note a padding mask where the model hands it to the block: the hyper-connection
        # layers consume it without going through TransformerLayer._forward_attention, the
        # only place that records it per layer.
        if (
            kwargs.get("padding_mask") is not None
            and getattr(self, "_padding_mask_shape_seen", None) is None
            and getattr(self.config, "cuda_graph_impl", "none") == "chunk_wise"
        ):
            self._padding_mask_shape_seen = tuple(kwargs["padding_mask"].shape)
        packed_seq_params = kwargs.get("packed_seq_params")
        if (
            packed_seq_params is not None
            and getattr(self.config, "cuda_graph_impl", "none") == "chunk_wise"
        ):
            # Some data paths hand over one tensor as both the query boundaries and the padded
            # ones (the physical boundaries of a zigzag layout), and code inside the block
            # tells the two cases apart by the identity of the tensors (the roll of the MTP
            # hidden-state mixing). A capture rebuilds the metadata from the graph's static
            # inputs, so note what the model hands over in its eager calls.
            self._packed_boundaries_shared_seen = (
                packed_seq_params.cu_seqlens_q_padded is not None
                and packed_seq_params.cu_seqlens_q_padded is packed_seq_params.cu_seqlens_q
            )
        return super().__call__(*args, **kwargs)

    def get_layer_static_inputs(self, seq_length, micro_batch_size):
        """Build the static inputs of a complete decoder-block graph."""
        layout_manager = getattr(self, "_cp_layout_manager", None)
        if layout_manager is not None and layout_manager.requires_conversion:
            raise ValueError(
                "cuda_graph_impl='chunk_wise' cannot capture a hybrid stack that converts "
                "between context-parallel layouts (boundary layout "
                f"{layout_manager.boundary_layout}, layer layouts "
                f"{sorted(set(layout_manager.layer_layouts))}): the conversion exchanges tokens "
                "with data-dependent split sizes. Give every layer the boundary layout "
                "(attention_cp_layout / linear_cp_layout) or run without chunk graphs."
            )
        if (
            getattr(self, "_has_linear_layer_with_chunkwise_cp", False)
            # The flag says which layers would run chunkwise; the stack only does so with more
            # than one context-parallel rank (dynamic CP is not captured, so the size is static).
            and self.config.context_parallel_size > 1
            and self._is_thd_cuda_graph()
        ):
            raise ValueError(
                "cuda_graph_impl='chunk_wise' cannot capture packed (THD) sequences through "
                "Mamba layers with chunkwise context parallelism (linear_cp_mode='chunkwise'): "
                "their CP metadata is built on the host from the sequence boundaries of each "
                "pack, so a captured graph would replay the layout of the capture sample. Use "
                "fixed-length batches or run without chunk graphs."
            )
        static_inputs = super().get_layer_static_inputs(seq_length, micro_batch_size)
        config = self.config
        device = torch.cuda.current_device()

        # getattr: unit tests build blocks on stand-in configs.
        if getattr(config, "fp32_residual_connection", False):
            # The residual stream enters every block in float32: the embedding casts it and the
            # pipeline carries it in float32 (TransformerConfig sets pipeline_dtype). A static
            # input of the parameter dtype would have every replay's input copy round the
            # stream, and the returned input gradient with it, at each block boundary.
            hidden_states = static_inputs["hidden_states"]
            static_inputs["hidden_states"] = torch.ones(
                hidden_states.shape, dtype=torch.float32, requires_grad=True, device=device
            )

        if config.enable_hyper_connections and not self.pre_process:
            # mHC keeps the expanded residual streams across chunk boundaries. Only the first
            # decoder chunk receives the H-wide embedding and expands it internally.
            hidden_states = static_inputs["hidden_states"]
            static_inputs["hidden_states"] = torch.ones(
                (*hidden_states.shape[:-1], config.hidden_size * config.num_residual_streams),
                dtype=hidden_states.dtype,
                requires_grad=True,
                device=device,
            )

        if self._is_thd_cuda_graph():
            self._check_packed_layers_are_capturable()
            max_num_seqs = _thd_max_packed_sequences(config)
            max_tokens = config.max_seqlen_per_dp_cp_rank * config.context_parallel_size
            cu_seqlens = torch.zeros(max_num_seqs + 1, dtype=torch.int32, device=device)
            cu_seqlens[1:] = max_tokens
            static_inputs.update(
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_kv=cu_seqlens.clone(),
                cu_seqlens_q_padded=cu_seqlens.clone(),
                cu_seqlens_kv_padded=cu_seqlens.clone(),
            )
            self._add_graph_dynamic_dsa_route_static_inputs(static_inputs, cu_seqlens, max_tokens)
            captures_padding_mask = True
        else:
            if config.create_attention_mask_in_dataloader:
                local_seq_length = seq_length // config.context_parallel_size
                static_inputs["attention_mask"] = (
                    ~(torch.tril(torch.ones((local_seq_length, seq_length))).bool())
                    .to(device)
                    .reshape(1, 1, local_seq_length, seq_length)
                    .tile(micro_batch_size, 1, 1, 1)
                )
            # Fixed-length batches carry a padding mask only for some datasets. Like the
            # per-layer capture, reserve the input when this block (__call__) or one of its
            # layers (TransformerLayer._padding_mask_shape_seen) received a mask during the
            # eager warm-up steps.
            captures_padding_mask = any(
                getattr(module, "_padding_mask_shape_seen", None) is not None
                for module in self.modules()
            )

        # padding_mask and the hash router's input_ids are batch-first token metadata. They are
        # captured in the block's own token frame, that of its hidden_states ([tokens, batch,
        # hidden]: CP-local and, with sequence parallelism, TP-local), on every stage; replay
        # brings what the model passes to that frame (_align_token_metadata_to_block).
        tokens, batch = static_inputs["hidden_states"].shape[:2]
        if captures_padding_mask:
            static_inputs["padding_mask"] = torch.zeros(
                (batch, tokens), dtype=torch.bool, device=device
            )
        self._chunk_graph_captures_padding_mask = captures_padding_mask
        self._chunk_graph_captures_attention_mask = "attention_mask" in static_inputs
        if config.moe_n_hash_layers > 0 and any(
            getattr(module, "is_hash_layer", False) for module in self.modules()
        ):
            static_inputs["input_ids"] = torch.zeros(
                (batch, tokens), dtype=torch.long, device=device
            )
        return static_inputs

    def _check_packed_layers_are_capturable(self):
        """Refuse layers that derive host-side state from the values of a pack's ``cu_seqlens``.

        The config rejects GDN / KDA with packed sequences by ``experimental_attention_variant``
        and by hybrid pattern symbol ('G', 'K'). Which mixer a layer runs is the layer spec's
        choice (a Mamba-symbol layer can hold a gated delta product mixer, which runs the same
        FLA chunk kernels), so the layers themselves are checked here.
        """
        from megatron.core.ssm.gated_delta_net.common import _GDNBase

        def derives_chunk_indices_on_the_host(module):
            # The gated delta product mixer is matched by name, so the check does not depend
            # on the module that defines it.
            return isinstance(module, _GDNBase) or any(
                cls.__name__ == "GatedDeltaProductMixer" for cls in type(module).__mro__
            )

        if any(derives_chunk_indices_on_the_host(module) for module in self.modules()):
            raise ValueError(
                "cuda_graph_impl='chunk_wise' does not support packed-sequence (THD) training "
                "with GDN / KDA / GDP layers: the FLA kernels derive their chunk indices on the "
                "host from the cu_seqlens values and cache them per tensor, so a captured graph "
                "replays the pack layout of the capture sample. Use fixed-length sequences (no "
                "sequence_packing_scheduler) with chunk-wise graphs, or run the model without "
                "CUDA graphs."
            )

    def _align_token_metadata_to_block(self, hidden_states, kwargs):
        """Bring a replay's ``padding_mask`` / ``input_ids`` to the captured token frame.

        The models differ in what they hand the block under sequence parallelism: ``GPTModel``
        scatters the padding mask on the first stage only and never the ids, ``HybridModel``
        scatters both. In eager mode every consumer shards full-length metadata itself
        (``MoELayer._align_batch_first_to_sequence_parallel``); a graph input has one fixed
        shape, so the same shard is taken here, before the replay copies it into the static
        buffer.

        The graph's keyword set is fixed at capture as well: a graph that reserved a mask gets
        an all-False one when a batch has none, and a mask arriving at a graph captured without
        one would be dropped silently by the graphed callable, so it raises when the block
        routes tokens (the routers would diverge from eager) and is dropped when nothing in the
        block reads it.
        """
        if self.config.sequence_parallel:
            for key in ("padding_mask", "input_ids"):
                value = kwargs.get(key)
                if value is not None and value.shape[1] != hidden_states.shape[0]:
                    kwargs[key] = (
                        tensor_parallel.scatter_to_sequence_parallel_region(
                            value.transpose(0, 1).contiguous(), group=self.tp_group
                        )
                        .transpose(0, 1)
                        .contiguous()
                    )
        if getattr(self, "_chunk_graph_captures_padding_mask", False):
            if kwargs.get("padding_mask") is None:
                tokens, batch = hidden_states.shape[:2]
                kwargs["padding_mask"] = torch.zeros(
                    (batch, tokens), dtype=torch.bool, device=hidden_states.device
                )
        elif kwargs.pop("padding_mask", None) is not None:
            from megatron.core.transformer.moe.moe_layer import MoELayer

            if any(isinstance(module, MoELayer) for module in self.modules()):
                raise RuntimeError(
                    "padding_mask was passed to a chunk CUDA graph that was captured without a "
                    "padding_mask input: no mask reached this block during "
                    "cuda_graph_warmup_steps (is it 0?). The graphed MoE routers would silently "
                    "ignore the mask. Run at least one eager warm-up step with masked batches."
                )

    def _add_graph_dynamic_dsa_route_static_inputs(self, static_inputs, cu_seqlens, max_tokens):
        """Add the balanced-CP route pair (full-capacity sample) to the block's static inputs."""
        if not self._block_uses_graph_dynamic_dsa_route():
            return
        from megatron.core.transformer.experimental_attention_variant import cp_balanced_indexer

        cp_balanced_indexer.add_graph_dynamic_plan_static_inputs(
            static_inputs, cu_seqlens, self._chunk_cp_group(), max_tokens
        )

    def _drop_cp_layout_kwargs(self, kwargs):
        """Drop the per-layout metadata (``packed_seq_params_by_layout``,
        ``padding_mask_by_layout``, ``cp_layout_plan``) that a hybrid stack receives alongside
        ``packed_seq_params`` and ``padding_mask``.

        A stack reads them only when it converts between CP layouts across its layers
        (``ContextParallelLayoutManager.requires_conversion``); that conversion routes tokens
        with data-dependent split sizes and cannot be captured, so ``get_layer_static_inputs``
        rejects such stacks. Every other stack ignores them, as its eager forward does: the
        dicts repeat the ``packed_seq_params`` / ``padding_mask`` already passed, and the plan
        is ``None`` or, where the data pipeline routes every pack between the contiguous and
        the zigzag layout (``linear_cp_layout`` differs from ``attention_cp_layout``) although
        this stack has layers of one layout only, a route the stack does not take.
        """
        kwargs.pop("packed_seq_params_by_layout", None)
        kwargs.pop("padding_mask_by_layout", None)
        plan = kwargs.pop("cp_layout_plan", None)
        layout_manager = getattr(self, "_cp_layout_manager", None)
        if plan is not None and layout_manager is not None and layout_manager.requires_conversion:
            raise RuntimeError(
                "Chunk CUDA graphs received a THD CP layout plan for a stack that converts "
                "between context-parallel layouts: per-layer CP layout conversion is not "
                "supported inside a captured block."
            )

    def _decompose_packed_seq_params_to_kwargs(self, kwargs):
        """Replace ``packed_seq_params`` by its graph-copyable tensors (``cu_seqlens`` and, with
        the graph-dynamic balanced CP indexer, the per-pack route pair)."""
        self._drop_cp_layout_kwargs(kwargs)
        packed_seq_params = kwargs.pop("packed_seq_params", None)
        if packed_seq_params is None:
            return
        if (
            packed_seq_params.local_cp_size is not None
            and packed_seq_params.local_cp_size != self.config.context_parallel_size
        ):
            # dynamic_context_parallel: the captured collectives use the capture's CP group.
            raise RuntimeError(
                "A chunk CUDA graph cannot replay a microbatch scheduled on a context-parallel "
                f"group of size {packed_seq_params.local_cp_size}: its collectives were "
                f"captured for context_parallel_size={self.config.context_parallel_size}."
            )
        if not self._is_thd_cuda_graph():
            # Captured from fixed-length samples, the graph has no cu_seqlens inputs: the
            # graphed callable would drop them and replay plain causal attention across the
            # sequence boundaries.
            raise RuntimeError(
                "A chunk CUDA graph captured for fixed-length batches received packed_seq_params. "
                "Packed sequences are captured only with sequence_packing_scheduler (which "
                "bounds the packed layout through max_seqlen_per_dp_cp_rank and "
                "thd_max_packed_sequences); run this data path without chunk-wise CUDA graphs."
            )
        kwargs["cu_seqlens_q"] = packed_seq_params.cu_seqlens_q
        kwargs["cu_seqlens_kv"] = packed_seq_params.cu_seqlens_kv
        kwargs["cu_seqlens_q_padded"] = packed_seq_params.cu_seqlens_q_padded
        kwargs["cu_seqlens_kv_padded"] = packed_seq_params.cu_seqlens_kv_padded
        if self._block_uses_graph_dynamic_dsa_route():
            from megatron.core.transformer.experimental_attention_variant import cp_balanced_indexer

            cp_group = self._chunk_cp_group()
            cp_balanced_indexer.validate_graph_dynamic_plan_contract(
                packed_seq_params,
                self.config.context_parallel_size,
                cp_group.rank(),
                self.config.max_seqlen_per_dp_cp_rank,
            )
            cp_balanced_indexer.add_graph_dynamic_plan_to_kwargs(
                packed_seq_params, kwargs, required=True
            )

    def _reconstruct_packed_seq_params_from_kwargs(self, kwargs):
        """Rebuild the graph-static ``PackedSeqParams`` inside the captured block."""
        if "cu_seqlens_q" not in kwargs:
            return
        from megatron.core.transformer.experimental_attention_variant import cp_balanced_indexer

        graph_dynamic_plan = cp_balanced_indexer.pop_graph_dynamic_plan_from_kwargs(
            kwargs, self.config.context_parallel_size, self.config.max_seqlen_per_dp_cp_rank
        )
        max_seqlen = self.config.max_seqlen_per_dp_cp_rank * self.config.context_parallel_size
        cu_seqlens_q, cu_seqlens_kv = kwargs.pop("cu_seqlens_q"), kwargs.pop("cu_seqlens_kv")
        cu_seqlens_q_padded = kwargs.pop("cu_seqlens_q_padded")
        cu_seqlens_kv_padded = kwargs.pop("cu_seqlens_kv_padded")
        if getattr(self, "_packed_boundaries_shared_seen", False):
            # The runtime metadata carries one tensor in both roles (see __call__): so does the
            # rebuilt one. The padded graph inputs then go unread; every replay refreshes the
            # query boundaries with the same values.
            cu_seqlens_q_padded, cu_seqlens_kv_padded = cu_seqlens_q, cu_seqlens_kv
        packed_seq_params = PackedSeqParams(
            qkv_format="thd",
            cp_partition_mode=self.config.cp_partition_mode,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            cu_seqlens_q_padded=cu_seqlens_q_padded,
            cu_seqlens_kv_padded=cu_seqlens_kv_padded,
            max_seqlen_q=max_seqlen,
            max_seqlen_kv=max_seqlen,
            pad_between_seqs=True,
        )
        if graph_dynamic_plan is not None:
            cp_balanced_indexer.attach_graph_dynamic_plan(packed_seq_params, graph_dynamic_plan)
        elif self._block_uses_graph_dynamic_dsa_route():
            raise RuntimeError(
                "Chunk CUDA graph input is missing the graph-dynamic balanced CP route metadata."
            )
        kwargs["packed_seq_params"] = packed_seq_params

    def _prepare_pipeline_input_for_chunk_cuda_graph(self, args, kwargs):
        """Expose the pipeline input tensor as the positional graph input."""
        args = tuple(args)
        kwargs = kwargs.copy()
        if self.pre_process:
            return args, kwargs

        hidden_states = args[0] if args else kwargs.get("hidden_states")
        if hidden_states is None:
            hidden_states = self.input_tensor
        assert torch.is_tensor(
            hidden_states
        ), "A non-pre-process decoder chunk requires input_tensor before TE capture or replay."

        if args:
            args = (hidden_states,) + args[1:]
        else:
            kwargs["hidden_states"] = hidden_states
        self.input_tensor = hidden_states
        return args, kwargs

    def _te_cuda_graph_capture(self, *args, **kwargs):
        """Run the block forward on the static inputs so TE records it as one graph."""
        args, kwargs = self._prepare_pipeline_input_for_chunk_cuda_graph(args, kwargs)
        self._reconstruct_packed_seq_params_from_kwargs(kwargs)
        kwargs.setdefault("attention_mask", None)

        if self.config.moe_paged_stash:
            from megatron.core.transformer.moe.paged_stash import (
                PagedStashManager,
                paged_stash_init_chunk_handler,
            )

            # TE calls the block directly, so the model-level chunk handler does not run: start
            # a new (microbatch, layer=1) key range for this capture-time forward. The
            # post-process block continues the decoder's layer numbering, like eager MTP.
            if PagedStashManager.get_instance().enabled and not getattr(
                self, "is_cuda_graph_postprocess_callable", False
            ):
                paged_stash_init_chunk_handler(
                    self.config.virtual_pipeline_model_parallel_size,
                    getattr(self, "vp_stage", None),
                )

        if self.config.fine_grained_activation_offloading:
            from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
                FineGrainedActivationOffloadingInterface as off_interface,
            )

            # One offload handler per capture-time callable forward; the post-process block's
            # handler resumes where the decoder's groups end, like eager MTP, but is its own
            # (the two callables are separate graphs). The TE graph interface's
            # backward_record/forward_record mark the capture boundary of a callable, as they
            # do for a per-layer callable: the backward graph's tail joins the H2D stream and
            # the forward graph's tail joins the D2H stream.
            te_graph_boundary = self.get_offload_interface()
            is_postprocess = getattr(self, "is_cuda_graph_postprocess_callable", False)
            off_interface.begin_block_capture_chunk(
                getattr(self, "vp_stage", None), continue_current=is_postprocess
            )
            off_interface.block_capture_forward_start()
            if args:
                hidden_states = te_graph_boundary.backward_record(args[0])
                args = (hidden_states,) + args[1:]
            else:
                hidden_states = te_graph_boundary.backward_record(kwargs["hidden_states"])
                kwargs["hidden_states"] = hidden_states
            if not self.pre_process:
                self.input_tensor = hidden_states
            # The callable's static inputs (hidden_states, and for the post-process block
            # mhc_multistream, labels, ...) are Transformer Engine's sample args: it keeps them
            # alive for the slot and refreshes them at replay. The first layer's input / residual
            # (attn_norm's forced release) and the loss inputs alias them, so the offload release
            # must not resize_(0) their storage.
            for value in (*args, *kwargs.values()):
                if torch.is_tensor(value):
                    off_interface.register_block_static_input(value)

        output = self.forward(*args, **kwargs)

        if self.config.fine_grained_activation_offloading:
            output = off_interface.block_capture_backward_start(output)
            te_graph_boundary.forward_record()
            off_interface.end_block_capture_chunk(record_decoder_end=not is_postprocess)

        if self.config.moe_paged_stash:
            from megatron.core.transformer.moe.paged_stash import (
                paged_stash_wait_for_stash_to_complete,
            )

            # The forward graph must not end with work pending on the stash's pack stream.
            paged_stash_wait_for_stash_to_complete()

        return output

    def _supply_attention_mask_of_a_fixed_length_graph(self, hidden_states, kwargs):
        """Give a fixed-length graph that was captured with an attention mask one at replay.

        With ``create_attention_mask_in_dataloader`` the fixed-length capture takes the
        dataloader's attention mask as a static input (``get_layer_static_inputs``). The
        training script hands that mask to the first and the last pipeline stage only: a stage
        in between receives no batch, so its block is called with ``attention_mask=None``, and
        the graphed callable refuses a replay that lacks a captured keyword. Pass the mask that
        means the same as a missing one to the attention of this block:

        * Transformer Engine's attention applies the mask of its ``attn_mask_type`` itself and
          ignores a given mask or combines the two, so a mask that masks nothing equals none
          (what the per-layer replay passes, ``TransformerLayer._get_te_cuda_graph_replay_args``);
        * Megatron's own ``DotProductAttention`` (``fallback_to_eager_attn``) applies a given
          mask as its only mask and builds the causal one just for a missing mask, so there
          the causal mask equals none.
        """
        if not self._chunk_graph_captures_attention_mask:
            return
        if kwargs.get("attention_mask") is not None:
            return
        # hidden_states is [tokens, batch, hidden] in the block's own frame: CP-local and, with
        # sequence parallelism, TP-local. The mask is [batch, 1, CP-local tokens, tokens].
        local_seq_length, micro_batch_size = hidden_states.shape[0], hidden_states.shape[1]
        if self.config.sequence_parallel:
            local_seq_length *= self.config.tensor_model_parallel_size
        shape = (
            micro_batch_size,
            1,
            local_seq_length,
            local_seq_length * self.config.context_parallel_size,
        )
        mask = self._attention_mask_for_a_missing_mask
        if mask is None or mask.shape != shape or mask.device != hidden_states.device:
            mask = torch.zeros(shape, dtype=torch.bool, device=hidden_states.device)
            mask_reading_attention = [
                module for module in self.modules() if isinstance(module, DotProductAttention)
            ]
            if mask_reading_attention:
                if self.config.context_parallel_size > 1 or any(
                    module.attn_mask_type != AttnMaskType.causal
                    for module in mask_reading_attention
                ):
                    raise ValueError(
                        "cuda_graph_impl='chunk_wise' captured this block with the dataloader's "
                        "attention mask, but the block was called without one (a pipeline stage "
                        "that is neither the first nor the last receives no batch). With "
                        "fallback_to_eager_attn the missing mask can only be replaced for causal "
                        "attention without context parallelism: pass the attention mask on "
                        "every stage or set create_attention_mask_in_dataloader=False."
                    )
                mask = ~torch.tril(torch.ones_like(mask))
            self._attention_mask_for_a_missing_mask = mask
        kwargs["attention_mask"] = mask

    def _te_cuda_graph_replay(self, *args, **kwargs):
        """Replay the graph of microbatch ``current_microbatch``."""
        args, kwargs = self._prepare_pipeline_input_for_chunk_cuda_graph(args, kwargs)
        self._decompose_packed_seq_params_to_kwargs(kwargs)
        # The pipeline input is the positional graph input of the shared replay.
        if args:
            hidden_states, args = args[0], args[1:]
        else:
            hidden_states = kwargs.pop("hidden_states")
        self._align_token_metadata_to_block(hidden_states, kwargs)
        self._supply_attention_mask_of_a_fixed_length_graph(hidden_states, kwargs)
        # Only tensors are graph inputs; a keyword that is None at runtime was not captured.
        kwargs = {key: value for key, value in kwargs.items() if value is not None}

        microbatch_id = self._te_cuda_graph_replay_microbatch()
        num_slots = len(self.te_cuda_graphs)
        # Without pipeline parallelism one slot serves every microbatch (a microbatch's backward
        # completes before the next forward). With PP the capture holds one slot per microbatch,
        # so the shared replay's ``microbatch % len(te_cuda_graphs)`` must not wrap: reusing a
        # slot whose backward has not retired would corrupt its static buffers.
        if self.config.pipeline_model_parallel_size > 1 and microbatch_id >= num_slots:
            # Not an assert: under ``python -O`` the replay below would wrap around silently.
            raise RuntimeError(
                f"Chunk CUDA graph replay requested microbatch {microbatch_id}, but the capture "
                f"only contains {num_slots} graph slots. The slots are sized at "
                "capture from the global batch size (one sequence per dataset item for packed "
                "data). A step with more microbatches needs a new capture: keep the global "
                "batch size fixed, or relaunch from a checkpoint after it changes."
            )
        with self._chunk_graph_replay_quantization_context():
            return super()._te_cuda_graph_replay(hidden_states, *args, **kwargs)


class ChunkCudaGraphPostProcessBlock(ChunkCudaGraphBlockMixin, GraphableMegatronModule):
    """Base class for the last stage's post-process (MTP block, LM head, loss) as a chunk callable.

    It is captured right after the decoder chunk in the same schedule-ordered TE capture, so it
    shares the graph memory pool with the decoder graphs (replay-safe by construction, because the
    allocator sees the true forward/backward liveness of every microbatch).

    The block owns no parameters of its own. The model's ``mtp`` / ``output_layer`` / ``embedding``
    are registered as shared children so Transformer Engine collects their parameters for the
    captured backward; state dicts are suppressed so the shared parameters are not serialized
    twice, and the missing-key report of a strict ``load_state_dict`` skips this subtree. Models
    attach it under the attribute ``postprocess_block``.
    """

    is_cuda_graph_postprocess_callable = True

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__(config=model.config)
        object.__setattr__(self, "_owner", model)  # plain reference, not a registered child
        if getattr(model, "mtp", None) is not None:
            self.mtp = model.mtp
        self.output_layer = model.output_layer
        if getattr(model, "embedding", None) is not None:
            self.embedding = model.embedding
        self.pre_process = False
        self.post_process = True
        self.input_tensor = None
        self.vp_stage = getattr(model, "vp_stage", None)
        self.register_load_state_dict_post_hook(self._drop_shared_missing_keys)

    @staticmethod
    def _drop_shared_missing_keys(module, incompatible_keys):
        """The shared parameters are loaded through the owner's own attributes."""
        marker = "postprocess_block."
        incompatible_keys.missing_keys[:] = [
            key for key in incompatible_keys.missing_keys if marker not in key
        ]

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        """Shared parameters are serialized by their owning module."""
        if destination is None:
            destination = OrderedDict()
        return destination

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Shared parameters are serialized by their owning module."""
        return {}

    def eager_only_reason(self):
        """Why this post-process has to stay eager, or ``None`` when it can be captured.

        A replay runs no Python, so code that has to see the logits of every microbatch keeps
        the eager post-process (after the decoder replay): forward hooks on the LM head (logits
        savers, distillation captures) here, and what a model's own post-process adds.
        """
        if self.output_layer._forward_hooks:
            return "forward hooks are registered on the output layer"
        return None

    def receives_every_captured_input(self, **optional_inputs):
        """Whether a call hands the post-process every optional tensor its graph reserves.

        The graph takes ``labels``, ``loss_mask`` and the packed sequence parameters and, where
        the MTP runs on this stage, ``input_ids``, ``position_ids`` and ``padding_mask`` as well
        (``get_layer_static_inputs``). The models declare all of them optional and the eager
        post-process has a default for each; a graph cannot be replayed without an input it
        was captured with, so a call that leaves one out takes the eager post-process.
        """
        required = ["labels", "loss_mask", "packed_seq_params"]
        if (self.config.mtp_num_layers or 0) > 0 and self._owner.mtp_process:
            required += ["input_ids", "position_ids", "padding_mask"]
        return all(optional_inputs.get(name) is not None for name in required)

    def _chunk_graph_has_quantized_modules(self):
        """The MTP layers run under FP8 like eager (``get_layer_fp8_context``) unless
        ``keep_mtp_in_bf16`` holds them out, and the LM head is a Transformer Engine module
        only with the MXFP8 output projection; a post-process block with neither is captured
        and replayed without an autocast, exactly as the eager post-process opens none."""
        from megatron.core.fp8_utils import is_mxfp8_output_proj_active

        mtp_on_this_stage = (self.config.mtp_num_layers or 0) > 0 and self._owner.mtp_process
        quantized_mtp = mtp_on_this_stage and not getattr(self.config, "keep_mtp_in_bf16", False)
        return quantized_mtp or is_mxfp8_output_proj_active(self.config)

    def _align_token_metadata_to_block(self, hidden_states, kwargs):
        """The post-process consumes the caller's full-length ``input_ids`` / ``padding_mask``
        (the MTP rolls them together and gathers its own hidden states), captured as such."""

    def get_layer_static_inputs(self, seq_length, micro_batch_size):
        """Static inputs of the post-process graph (packed sequences only)."""
        config = self.config
        assert (
            self._is_thd_cuda_graph()
        ), "Post-process chunk CUDA graphs are supported for packed (THD) sequences only."
        self._check_packed_layers_are_capturable()
        device = torch.cuda.current_device()
        if config.bf16:
            dtype = torch.bfloat16
        elif config.fp16:
            dtype = torch.float16
        else:
            dtype = torch.float32
        tokens = config.max_seqlen_per_dp_cp_rank
        hidden_tokens = tokens
        if config.sequence_parallel:
            hidden_tokens //= config.tensor_model_parallel_size
        inputs = {
            "hidden_states": torch.ones(
                (hidden_tokens, 1, config.hidden_size),
                dtype=dtype,
                requires_grad=True,
                device=device,
            )
        }
        mtp_on_this_stage = (config.mtp_num_layers or 0) > 0 and self._owner.mtp_process
        if config.enable_hyper_connections and mtp_on_this_stage:
            # The decoder hands MTP the pre-contraction residual streams.
            inputs["mhc_multistream"] = torch.ones(
                (hidden_tokens, 1, config.hidden_size * config.num_residual_streams),
                dtype=dtype,
                requires_grad=True,
                device=device,
            )
        if mtp_on_this_stage:
            # Only the MTP block reads these (and the padding mask below) in post-process. A last
            # stage without MTP receives no tokens / position ids in its packed batch
            # (data_schedule.py), the replay drops None keywords, and a keyword that was captured
            # but never arrives at replay makes the graphed callable raise, so they are captured
            # only where the stage actually consumes them.
            # Zeros: valid indices for any vocabulary and any learned position table (which can
            # be shorter than a pack); the runtime tensors overwrite them at every replay.
            inputs["input_ids"] = torch.zeros((1, tokens), dtype=torch.long, device=device)
            inputs["position_ids"] = torch.zeros((1, tokens), dtype=torch.long, device=device)
        inputs["labels"] = torch.zeros((1, tokens), dtype=torch.long, device=device)
        inputs["loss_mask"] = torch.ones((1, tokens), dtype=torch.float32, device=device)
        max_num_seqs = _thd_max_packed_sequences(config)
        max_tokens = tokens * config.context_parallel_size
        cu_seqlens = torch.zeros(max_num_seqs + 1, dtype=torch.int32, device=device)
        cu_seqlens[1:] = max_tokens
        inputs.update(
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens.clone(),
            cu_seqlens_q_padded=cu_seqlens.clone(),
            cu_seqlens_kv_padded=cu_seqlens.clone(),
        )
        self._add_graph_dynamic_dsa_route_static_inputs(inputs, cu_seqlens, max_tokens)
        if mtp_on_this_stage:
            # Full length on every stage: the model hands the post-process the caller's padding
            # mask (the MTP rolls it alongside input_ids); only the decoder gets the SP-scattered
            # copy.
            inputs["padding_mask"] = torch.zeros(1, tokens, dtype=torch.bool, device=device)
        return inputs


def _thd_max_packed_sequences(config) -> int:
    """Upper bound on packed sequences per microbatch, from either config spelling."""
    max_num_seqs = config.thd_max_packed_sequences
    if max_num_seqs is None:
        max_num_seqs = getattr(config, "cuda_graph_thd_max_packed_sequences", None)
    assert max_num_seqs is not None, (
        "thd_max_packed_sequences (or cuda_graph_thd_max_packed_sequences) must be set for "
        "THD chunk CUDA graphs."
    )
    return max_num_seqs


def build_postprocess_block(model, block_cls):
    """Attach a post-process chunk callable to ``model`` when the configuration supports it.

    Post-process capture is part of chunk-wise graphs for packed-sequence (THD) training on the
    last pipeline stage; other layouts keep the eager post-process.
    """
    config = model.config
    if not (
        model.post_process
        and config.cuda_graph_impl == "chunk_wise"
        and config.sequence_packing_scheduler is not None
    ):
        return None
    return block_cls(model)
