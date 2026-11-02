# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Chunk-wise Transformer Engine CUDA graphs.

With ``cuda_graph_impl="chunk_wise"`` the decoder block of every PP/VPP model chunk (and, on the
last stage, the post-process block: MTP, LM head and loss) is the callable handed to Transformer
Engine's ``make_graphed_callables`` instead of the individual layers. One forward graph and one
backward graph are captured per callable and microbatch slot, so activation recompute, MoE
dispatch/combine and the hyper-connection residual streams are all recorded inside one graph and
replayed with a single launch per pass.
"""

from collections import OrderedDict
from contextlib import nullcontext

import torch

from megatron.core.packed_seq_params import PackedSeqParams, packed_seq_idx_on_device
from megatron.core.transformer.module import GraphableMegatronModule

# Boundary tensors that passed validate_packed_boundaries_for_linear_attention, keyed by tensor
# identity: (cu_q, cu_kv, (versions, capacity, max_num_seqs, cp_size)). Holding the (tiny)
# tensors keeps their ids from being reused while they are cached; in-place refreshes of a
# reused buffer bump the version and are validated again.
_VALIDATED_PACKS_CAPACITY = 16
_VALIDATED_PACKS: "OrderedDict[tuple[int, int], tuple]" = OrderedDict()


def validate_packed_boundaries_for_linear_attention(
    packed_seq_params, capacity, max_num_seqs, cp_size
):
    """Host-side check of one pack before its boundaries are written into a captured graph.

    GDN-family layers consume ``cu_seqlens`` inside the captured block, where their own
    device-to-host checks cannot run. The replay therefore validates the pack here, in eager
    code, with a single device-to-host copy: ``max_num_seqs + 1`` boundaries, q == kv,
    non-decreasing, starting at 0 and covering the static capacity exactly (the dummy tail
    sequence of ``thd_tail_padding_policy='append_dummy_seq'``; a shorter pack would leave rows
    the kernels never write), and every physical length a multiple of ``2 * cp_size`` (the
    zigzag layout headwise CP redistributes, see ``_build_thd_cp_a2a_perm``).

    Raises:
        ValueError: with the offending boundaries, instead of a silently wrong replay.
    """
    cu_q = (
        packed_seq_params.cu_seqlens_q_padded
        if packed_seq_params.cu_seqlens_q_padded is not None
        else packed_seq_params.cu_seqlens_q
    )
    cu_kv = (
        packed_seq_params.cu_seqlens_kv_padded
        if packed_seq_params.cu_seqlens_kv_padded is not None
        else packed_seq_params.cu_seqlens_kv
    )
    if cu_q is None or cu_kv is None:
        raise ValueError("Packed GDN chunk graphs need cu_seqlens_q and cu_seqlens_kv.")
    # Every captured block of a microbatch (decoder chunks, the post-process block) replays the
    # same pack: only the first one pays the device-to-host copy, the others find the boundary
    # tensors (by identity and in-place version) in the small cache of validated packs.
    cache_key = (id(cu_q), id(cu_kv))
    stamp = (cu_q._version, cu_kv._version, capacity, max_num_seqs, cp_size)
    cached = _VALIDATED_PACKS.get(cache_key)
    if cached is not None and cached[0] is cu_q and cached[1] is cu_kv and cached[2] == stamp:
        return
    host = torch.stack([cu_q, cu_kv]).detach().to(device="cpu", dtype=torch.long)
    boundaries, boundaries_kv = host[0], host[1]
    problems = []
    if boundaries.numel() != max_num_seqs + 1:
        problems.append(
            f"{boundaries.numel()} boundaries, expected thd_max_packed_sequences + 1 = "
            f"{max_num_seqs + 1}"
        )
    if not torch.equal(boundaries, boundaries_kv):
        problems.append("cu_seqlens_q and cu_seqlens_kv differ")
    lengths = boundaries[1:] - boundaries[:-1]
    if boundaries.numel() and boundaries[0].item() != 0:
        problems.append(f"first boundary is {boundaries[0].item()}, expected 0")
    if (lengths < 0).any():
        problems.append("boundaries are not non-decreasing")
    if boundaries.numel() and boundaries[-1].item() != capacity:
        problems.append(
            f"last boundary is {boundaries[-1].item()}, expected the static capacity "
            f"{capacity} (thd_tail_padding_policy='append_dummy_seq' covers the padding)"
        )
    if cp_size > 1 and (lengths % (2 * cp_size) != 0).any():
        problems.append(
            f"a physical sequence length is not a multiple of 2 * cp_size = {2 * cp_size}"
        )
    if problems:
        raise ValueError(
            "Packed sequence boundaries cannot be replayed through the GDN layers of a chunk "
            f"CUDA graph: {'; '.join(problems)}. cu_seqlens={boundaries.tolist()}"
        )
    _VALIDATED_PACKS[cache_key] = (cu_q, cu_kv, stamp)
    while len(_VALIDATED_PACKS) > _VALIDATED_PACKS_CAPACITY:
        _VALIDATED_PACKS.popitem(last=False)


class ChunkCudaGraphBlockMixin:
    """Make a complete decoder block a Transformer Engine graph callable.

    The mixin must precede ``GraphableMegatronModule`` in the bases. It contributes the static
    capture inputs of the block-level call signature and the capture/replay adapters that convert
    between that signature and the tensor-only interface CUDA graphs accept (``PackedSeqParams``
    is decomposed into its ``cu_seqlens`` tensors, and the pipeline input tensor is exposed as the
    positional graph input). With fine-grained activation offloading the offload manager runs in
    its whole-block capture mode (``PipelineOffloadManager.enter_block_capture``): the D2H/H2D
    copies are part of the captured graphs and every slot's graphs are self-contained.
    """

    is_cuda_graph_chunk_callable = True

    def _should_call_local_cudagraph(self, *args, **kwargs):
        """A chunk callable is graphed through Transformer Engine only."""
        return False

    def _chunk_cp_group(self):
        """Explicit CP group of the captured block (the post-process block uses its owner's)."""
        pg_collection = getattr(self, 'pg_collection', None)
        if pg_collection is None and getattr(self, '_owner', None) is not None:
            pg_collection = getattr(self._owner, 'pg_collection', None)
        return getattr(pg_collection, 'cp', None)

    def _block_uses_graph_dynamic_dsa_route(self):
        """Whether a DSA layer of this block consumes the balanced-CP per-pack route inputs.

        Deliberately not named like the layer-level ``_uses_graph_dynamic_dsa_route``: the
        per-layer capture's block-owned arena staging does not apply to a block callable. Here
        the two route buffers are ordinary static inputs of the block graph, refreshed per replay
        like ``cu_seqlens``, and the ``PackedSeqParams`` rebuilt inside the graph hands them to
        every DSA layer of the block.
        """
        if not getattr(self.config, 'dsa_cp_balance_indexer_graph_dynamic_packs', False):
            return False
        return any(
            module is not self
            and getattr(module, '_uses_graph_dynamic_dsa_route', None) is not None
            and module._uses_graph_dynamic_dsa_route()
            for module in self.modules()
        )

    def _gated_delta_net_layers(self):
        """The GDN-family layers (GDN / GDN2 / KDA) captured in this block, with their names."""
        cached = getattr(self, '_chunk_graph_gated_delta_net_layers', None)
        if cached is None:
            from megatron.core.ssm.gated_delta_net.common import _GDNBase

            cached = [
                (name, module)
                for name, module in self.named_modules()
                if isinstance(module, _GDNBase)
            ]
            self._chunk_graph_gated_delta_net_layers = cached
        return cached

    def _block_has_gated_delta_net_layers(self):
        """Whether a GDN-family layer (GDN / GDN2 / KDA) is captured in this block."""
        return bool(self._gated_delta_net_layers())

    def _check_packed_gated_delta_net_capture(self):
        """Refuse to capture packed GDN-family layers whose kernels cannot replay other packs.

        The stack-level configuration gate judges ``experimental_attention_variant``; layers
        built from a hybrid layer pattern ('G' / 'K') carry no variant, so every captured layer
        is asked for its own blockers here, before the first capture and the first replay.
        """
        if getattr(self, '_chunk_graph_gated_delta_net_checked', False):
            return
        problems = []
        for name, module in self._gated_delta_net_layers():
            blockers = module.packed_chunk_graph_blockers()
            if blockers:
                problems.append(f"{name or type(module).__name__}: {'; '.join(blockers)}")
        if problems:
            raise ValueError(
                "cuda_graph_impl='chunk_wise' cannot capture packed-sequence (THD) training with "
                "the GDN-family layers of this block, whose kernels derive their chunk indices on "
                "the host from the capture sample's cu_seqlens values: "
                + " | ".join(problems)
                + ". Only Gated DeltaNet on gdn_kernel_backend='transformer_engine' and KDA on "
                "flash-linear-attention >= 0.6.0 read the pack boundaries on the device. Use "
                "fixed-length sequences (no sequence_packing_scheduler) or run without CUDA "
                "graphs."
            )
        self._chunk_graph_gated_delta_net_checked = True

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
        """
        if not getattr(self.config, 'fp8', None) or not self._chunk_graph_has_quantized_modules():
            return nullcontext()
        from megatron.core.fp8_utils import get_fp8_context

        return get_fp8_context(self.config)

    def get_layer_static_inputs(self, seq_length, micro_batch_size):
        """Build the static inputs of a complete decoder-block graph."""
        layout_manager = getattr(self, '_cp_layout_manager', None)
        if layout_manager is not None and layout_manager.requires_conversion:
            raise ValueError(
                "cuda_graph_impl='chunk_wise' cannot capture a hybrid stack whose layers use "
                "different context-parallel layouts "
                f"(layer layouts {sorted(set(layout_manager.layer_layouts))}): the per-layer "
                "layout conversion exchanges tokens with data-dependent split sizes. Use one "
                "cp_partition_mode for every layer or run without chunk graphs."
            )
        static_inputs = super().get_layer_static_inputs(seq_length, micro_batch_size)
        config = self.config
        device = torch.cuda.current_device()

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
            self._check_packed_gated_delta_net_capture()
            max_num_seqs = config.thd_max_packed_sequences
            assert (
                max_num_seqs is not None
            ), "thd_max_packed_sequences must be set for THD chunk CUDA graphs."
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
            local_tokens = config.max_seqlen_per_dp_cp_rank
            if config.sequence_parallel and self.pre_process:
                local_tokens //= config.tensor_model_parallel_size
            static_inputs["padding_mask"] = torch.zeros(
                1, local_tokens, dtype=torch.bool, device=device
            )
        elif config.create_attention_mask_in_dataloader:
            local_seq_length = seq_length // config.context_parallel_size
            static_inputs["attention_mask"] = (
                ~(torch.tril(torch.ones((local_seq_length, seq_length))).bool())
                .to(device)
                .reshape(1, 1, local_seq_length, seq_length)
                .tile(micro_batch_size, 1, 1, 1)
            )

        has_hash_router = config.moe_n_hash_layers > 0 and any(
            getattr(module, 'is_hash_layer', False) for module in self.modules()
        )
        if has_hash_router:
            input_ids_shape = (
                (1, config.max_seqlen_per_dp_cp_rank)
                if self._is_thd_cuda_graph()
                else (micro_batch_size, seq_length)
            )
            static_inputs["input_ids"] = torch.zeros(
                input_ids_shape, dtype=torch.long, device=device
            )
        return static_inputs

    def _add_graph_dynamic_dsa_route_static_inputs(self, static_inputs, cu_seqlens, max_tokens):
        """Add the balanced-CP route pair (full-capacity sample) to the block's static inputs."""
        if not self._block_uses_graph_dynamic_dsa_route():
            return
        from megatron.core.transformer.experimental_attention_variant import cp_balanced_indexer

        cp_balanced_indexer.add_graph_dynamic_plan_static_inputs(
            static_inputs, cu_seqlens, self._chunk_cp_group(), max_tokens
        )

    def _drop_cp_layout_kwargs(self, kwargs):
        """Drop the per-layout packed metadata (``packed_seq_params_by_layout``, ``cp_layout_plan``)
        that a hybrid stack receives alongside ``packed_seq_params``.

        ``packed_seq_params_by_layout`` and ``cp_layout_plan`` only carry information when the
        stack converts between CP layouts across its layers (``ContextParallelLayoutManager.
        requires_conversion``); that conversion routes tokens with data-dependent split sizes and
        cannot be captured, so ``get_layer_static_inputs`` rejects such stacks. For a single layout
        the dict holds the one ``packed_seq_params`` already passed and the plan is ``None``, and
        the block rebuilds nothing from them.
        """
        kwargs.pop('packed_seq_params_by_layout', None)
        plan = kwargs.pop('cp_layout_plan', None)
        assert plan is None, (
            "Chunk CUDA graphs received a THD CP layout plan: per-layer CP layout conversion is "
            "not supported inside a captured block."
        )

    def _decompose_packed_seq_params_to_kwargs(self, kwargs):
        """Replace ``packed_seq_params`` by its graph-copyable tensors (``cu_seqlens`` and, with
        the graph-dynamic balanced CP indexer, the per-pack route pair)."""
        self._drop_cp_layout_kwargs(kwargs)
        packed_seq_params = kwargs.pop('packed_seq_params', None)
        if packed_seq_params is None:
            return
        if self._block_has_gated_delta_net_layers():
            # The GDN layers skip their device-to-host boundary checks inside the graph.
            self._check_packed_gated_delta_net_capture()
            validate_packed_boundaries_for_linear_attention(
                packed_seq_params,
                self.config.max_seqlen_per_dp_cp_rank * self.config.context_parallel_size,
                self.config.thd_max_packed_sequences,
                self.config.context_parallel_size,
            )
        kwargs['cu_seqlens_q'] = packed_seq_params.cu_seqlens_q
        kwargs['cu_seqlens_kv'] = packed_seq_params.cu_seqlens_kv
        kwargs['cu_seqlens_q_padded'] = packed_seq_params.cu_seqlens_q_padded
        kwargs['cu_seqlens_kv_padded'] = packed_seq_params.cu_seqlens_kv_padded
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
        if 'cu_seqlens_q' not in kwargs:
            return
        from megatron.core.transformer.experimental_attention_variant import cp_balanced_indexer

        graph_dynamic_plan = cp_balanced_indexer.pop_graph_dynamic_plan_from_kwargs(
            kwargs, self.config.context_parallel_size, self.config.max_seqlen_per_dp_cp_rank
        )
        max_seqlen = self.config.max_seqlen_per_dp_cp_rank * self.config.context_parallel_size
        cu_seqlens_q_padded = kwargs.pop('cu_seqlens_q_padded')
        # GDN-family layers consume the pack inside the graph: the per-token sequence index is
        # built from the static boundary buffer with fixed-shape device ops (the eager data path
        # uses a host-synchronizing repeat_interleave), and the pack was validated on the host by
        # _decompose_packed_seq_params_to_kwargs before this replay.
        linear_attention = self._block_has_gated_delta_net_layers()
        seq_idx = (
            packed_seq_idx_on_device(cu_seqlens_q_padded, max_seqlen) if linear_attention else None
        )
        packed_seq_params = PackedSeqParams(
            qkv_format='thd',
            cp_partition_mode=self.config.cp_partition_mode,
            cu_seqlens_q=kwargs.pop('cu_seqlens_q'),
            cu_seqlens_kv=kwargs.pop('cu_seqlens_kv'),
            cu_seqlens_q_padded=cu_seqlens_q_padded,
            cu_seqlens_kv_padded=kwargs.pop('cu_seqlens_kv_padded'),
            max_seqlen_q=max_seqlen,
            max_seqlen_kv=max_seqlen,
            pad_between_seqs=True,
            total_tokens=max_seqlen if linear_attention else None,
            seq_idx=seq_idx,
            cu_seqlens_validated=linear_attention,
        )
        if graph_dynamic_plan is not None:
            cp_balanced_indexer.attach_graph_dynamic_plan(packed_seq_params, graph_dynamic_plan)
        elif self._block_uses_graph_dynamic_dsa_route():
            raise RuntimeError(
                "Chunk CUDA graph input is missing the graph-dynamic balanced CP route metadata."
            )
        kwargs['packed_seq_params'] = packed_seq_params

    def _prepare_pipeline_input_for_chunk_cuda_graph(self, args, kwargs):
        """Expose the pipeline input tensor as the positional graph input."""
        args = tuple(args)
        kwargs = kwargs.copy()
        if self.pre_process:
            return args, kwargs

        hidden_states = args[0] if args else kwargs.get('hidden_states')
        if hidden_states is None:
            hidden_states = self.input_tensor
        assert torch.is_tensor(
            hidden_states
        ), "A non-pre-process decoder chunk requires input_tensor before TE capture or replay."

        if args:
            args = (hidden_states,) + args[1:]
        else:
            kwargs['hidden_states'] = hidden_states
        self.input_tensor = hidden_states
        return args, kwargs

    def _te_cuda_graph_capture(self, *args, **kwargs):
        """Run the block forward on the static inputs so TE records it as one graph."""
        if self.config.cuda_graph_impl != "chunk_wise":
            return super()._te_cuda_graph_capture(*args, **kwargs)

        args, kwargs = self._prepare_pipeline_input_for_chunk_cuda_graph(args, kwargs)
        self._reconstruct_packed_seq_params_from_kwargs(kwargs)
        kwargs.setdefault('attention_mask', None)

        if self.config.moe_paged_stash:
            from megatron.core.transformer.moe.paged_stash import (
                PagedStashManager,
                paged_stash_init_chunk_handler,
            )

            # TE calls the block directly, so the model-level chunk handler does not run: start
            # a new (microbatch, layer=1) key range for this capture-time forward. The
            # post-process block continues the decoder's layer numbering, like eager MTP.
            if PagedStashManager.get_instance().enabled and not getattr(
                self, 'is_cuda_graph_postprocess_callable', False
            ):
                paged_stash_init_chunk_handler(
                    self.config.virtual_pipeline_model_parallel_size,
                    getattr(self, 'vp_stage', None),
                )

        if self.config.fine_grained_activation_offloading:
            from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
                FineGrainedActivationOffloadingInterface as off_interface,
            )

            # One offload handler per capture-time model-chunk forward; the post-process block
            # continues the decoder's handler, like eager MTP. backward_record/forward_record are
            # dev's per-callable capture boundary: the backward graph's tail joins the H2D stream
            # and the forward graph's tail joins the D2H stream.
            is_postprocess = getattr(self, 'is_cuda_graph_postprocess_callable', False)
            off_interface.begin_block_capture_chunk(
                getattr(self, 'vp_stage', None), continue_current=is_postprocess
            )
            off_interface.block_capture_forward_start()
            if args:
                hidden_states = off_interface.backward_record(args[0])
                args = (hidden_states,) + args[1:]
            else:
                hidden_states = off_interface.backward_record(kwargs['hidden_states'])
                kwargs['hidden_states'] = hidden_states
            if not self.pre_process:
                self.input_tensor = hidden_states

        output = self.forward(*args, **kwargs)

        if self.config.fine_grained_activation_offloading:
            output = off_interface.block_capture_backward_start(output)
            off_interface.forward_record()
            off_interface.end_block_capture_chunk(record_decoder_end=not is_postprocess)

        if self.config.moe_paged_stash:
            from megatron.core.transformer.moe.paged_stash import (
                paged_stash_wait_for_stash_to_complete,
            )

            # The forward graph must not end with work pending on the stash's pack stream.
            paged_stash_wait_for_stash_to_complete()

        return output

    def _te_cuda_graph_replay(self, *args, **kwargs):
        """Replay the graph of microbatch ``current_microbatch``."""
        if self.config.cuda_graph_impl != "chunk_wise":
            return super()._te_cuda_graph_replay(*args, **kwargs)

        args, kwargs = self._prepare_pipeline_input_for_chunk_cuda_graph(args, kwargs)
        self._decompose_packed_seq_params_to_kwargs(kwargs)
        # Only tensors are graph inputs; a keyword that is None at runtime was not captured.
        kwargs = {key: value for key, value in kwargs.items() if value is not None}

        microbatch_id = getattr(self, 'current_microbatch', 0)
        # Without pipeline parallelism one slot serves every microbatch (a microbatch's backward
        # completes before the next forward). With PP the capture holds one slot per microbatch,
        # so the base replay's ``current_microbatch % len(cuda_graphs)`` must not wrap: reusing a
        # slot whose backward has not retired would corrupt its static buffers.
        assert self.config.pipeline_model_parallel_size == 1 or microbatch_id < len(
            self.cuda_graphs
        ), (
            f"Chunk CUDA graph replay requested microbatch {microbatch_id}, but the capture only "
            f"contains {len(self.cuda_graphs)} graph slots."
        )
        with self._chunk_graph_replay_quantization_context():
            return super()._te_cuda_graph_replay(*args, **kwargs)


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
    ATTRIBUTE_NAME = 'postprocess_block'

    def __init__(self, model):
        super().__init__(config=model.config)
        object.__setattr__(self, '_owner', model)  # plain reference, not a registered child
        if getattr(model, 'mtp', None) is not None:
            self.mtp = model.mtp
        self.output_layer = model.output_layer
        if getattr(model, 'embedding', None) is not None:
            self.embedding = model.embedding
        self.pre_process = False
        self.post_process = True
        self.input_tensor = None
        self.vp_stage = getattr(model, 'vp_stage', None)
        self.register_load_state_dict_post_hook(self._drop_shared_missing_keys)

    @staticmethod
    def _drop_shared_missing_keys(module, incompatible_keys):
        """The shared parameters are loaded through the owner's own attributes."""
        marker = f"{ChunkCudaGraphPostProcessBlock.ATTRIBUTE_NAME}."
        incompatible_keys.missing_keys[:] = [
            key for key in incompatible_keys.missing_keys if marker not in key
        ]

    def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
        """Shared parameters are serialized by their owning module."""
        if destination is None:
            destination = OrderedDict()
        return destination

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        """Shared parameters are serialized by their owning module."""
        return {}

    def _chunk_graph_has_quantized_modules(self):
        """The MTP layers run under FP8 like eager (``get_layer_fp8_context``) and the LM head
        is a Transformer Engine module only with the MXFP8 output projection; a post-process
        block with neither opens no autocast, exactly as the eager post-process does not."""
        from megatron.core.fp8_utils import is_mxfp8_output_proj_active

        mtp_on_this_stage = (self.config.mtp_num_layers or 0) > 0 and self._owner.mtp_process
        return mtp_on_this_stage or is_mxfp8_output_proj_active(self.config)

    def get_layer_static_inputs(self, seq_length, micro_batch_size):
        """Static inputs of the post-process graph (packed sequences only)."""
        config = self.config
        assert (
            self._is_thd_cuda_graph()
        ), "Post-process chunk CUDA graphs are supported for packed (THD) sequences only."
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
            inputs["input_ids"] = torch.zeros((1, tokens), dtype=torch.long, device=device)
            inputs["position_ids"] = torch.arange(
                tokens, dtype=torch.long, device=device
            ).unsqueeze(0)
        inputs["labels"] = torch.zeros((1, tokens), dtype=torch.long, device=device)
        inputs["loss_mask"] = torch.ones((1, tokens), dtype=torch.float32, device=device)
        max_num_seqs = config.thd_max_packed_sequences
        assert (
            max_num_seqs is not None
        ), "thd_max_packed_sequences must be set for THD chunk CUDA graphs."
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
