# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Chunk-granularity Transformer Engine CUDA graphs.

With ``cuda_graph_granularity="chunk"`` the decoder block of every PP/VPP model chunk (and, on the
last stage, the post-process block: MTP, LM head and loss) is the callable handed to Transformer
Engine's ``make_graphed_callables`` instead of the individual layers. One forward graph and one
backward graph are captured per callable and microbatch slot, so activation recompute, MoE
dispatch/combine and the hyper-connection residual streams are all recorded inside one graph and
replayed with a single launch per pass.
"""

from collections import OrderedDict

import torch

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.module import GraphableMegatronModule


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

    def get_layer_static_inputs(self, seq_length, micro_batch_size):
        """Build the static inputs of a complete decoder-block graph."""
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

    def _decompose_packed_seq_params_to_kwargs(self, kwargs):
        """Replace ``packed_seq_params`` by its graph-copyable tensors (``cu_seqlens`` and, with
        the graph-dynamic balanced CP indexer, the per-pack route pair)."""
        packed_seq_params = kwargs.pop('packed_seq_params', None)
        if packed_seq_params is None:
            return
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
        packed_seq_params = PackedSeqParams(
            qkv_format='thd',
            cp_partition_mode=self.config.cp_partition_mode,
            cu_seqlens_q=kwargs.pop('cu_seqlens_q'),
            cu_seqlens_kv=kwargs.pop('cu_seqlens_kv'),
            cu_seqlens_q_padded=kwargs.pop('cu_seqlens_q_padded'),
            cu_seqlens_kv_padded=kwargs.pop('cu_seqlens_kv_padded'),
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
        if self.config.cuda_graph_granularity != "chunk":
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
        if self.config.cuda_graph_granularity != "chunk":
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
            # mask (the MTP rolls it alongside input_ids); only the decoder gets the SP-scattered copy.
            inputs["padding_mask"] = torch.zeros(1, tokens, dtype=torch.bool, device=device)
        return inputs


def build_postprocess_block(model, block_cls):
    """Attach a post-process chunk callable to ``model`` when the configuration supports it.

    Post-process capture is part of chunk granularity for packed-sequence (THD) training on the
    last pipeline stage; other layouts keep the eager post-process.
    """
    config = model.config
    if not (
        model.post_process
        and config.cuda_graph_impl == "transformer_engine"
        and config.cuda_graph_granularity == "chunk"
        and config.sequence_packing_scheduler is not None
    ):
        return None
    return block_cls(model)
