# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Shared TE chunk CUDA graph contract for decoder block implementations."""

import torch

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.utils import is_te_min_version, is_torch_min_version


class ChunkCudaGraphBlockMixin:
    """Make a complete decoder block a Transformer Engine graph callable."""

    is_cuda_graph_chunk_callable = True
    _supports_dynamic_cp_cuda_graph_replay = True

    def _initialize_chunk_cuda_graph_support(self):
        """Validate and initialize block-boundary offload integration."""
        self.offload_module_in_cuda_graph = bool(
            self.config.cuda_graph_impl == "transformer_engine"
            and getattr(self.config, 'cuda_graph_granularity', 'layer') == "chunk"
            and self.config.fine_grained_activation_offloading
        )
        if self.offload_module_in_cuda_graph:
            assert is_torch_min_version(
                "2.9.0a0"
            ), "Offloading modules captured in a chunk CUDA graph requires torch>=2.9.0."
            assert is_te_min_version(
                "2.14.0"
            ), "Offloading modules captured in a chunk CUDA graph requires TE>=2.14.0."
            assert self.config.cuda_graph_warmup_steps > 0, (
                "Fine-grained activation offloading with chunk CUDA graphs requires "
                "cuda_graph_warmup_steps > 0."
            )

    def _should_call_local_cudagraph(self, *args, **kwargs):
        """Do not change the existing HybridStack local-backend ownership."""
        if getattr(self.config, 'cuda_graph_granularity', 'layer') == 'chunk':
            return False
        return super()._should_call_local_cudagraph(*args, **kwargs)

    def get_layer_static_inputs(self, seq_length, micro_batch_size):
        """Build static inputs for a complete decoder-block TE graph."""
        static_inputs = super().get_layer_static_inputs(seq_length, micro_batch_size)
        device = torch.cuda.current_device()

        # Hybrid mHC keeps the expanded residual streams across PP/VPP chunk
        # boundaries. Only the first decoder chunk receives the ordinary H-wide
        # embedding and performs input_expand internally.
        if self.config.enable_hyper_connections and not self.pre_process:
            hidden_states = static_inputs["hidden_states"]
            static_inputs["hidden_states"] = torch.ones(
                (
                    *hidden_states.shape[:-1],
                    self.config.hidden_size * self.config.num_residual_streams,
                ),
                dtype=hidden_states.dtype,
                requires_grad=True,
                device=device,
            )

        if self._is_thd_cuda_graph():
            max_num_seqs = self.config.thd_max_packed_sequences
            assert (
                max_num_seqs is not None
            ), "thd_max_packed_sequences must be set for THD chunk CUDA graphs."
            capture_cp_size, _ = self._get_thd_cuda_graph_capture_cp()
            max_tokens = self.config.max_seqlen_per_dp_cp_rank * capture_cp_size
            cu_seqlens = torch.zeros(max_num_seqs + 1, dtype=torch.int32, device=device)
            cu_seqlens[1:] = max_tokens
            static_inputs.update(
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_kv=cu_seqlens.clone(),
                cu_seqlens_q_padded=cu_seqlens.clone(),
                cu_seqlens_kv_padded=cu_seqlens.clone(),
            )

            local_tokens = self.config.max_seqlen_per_dp_cp_rank
            if self.config.sequence_parallel and self.pre_process:
                local_tokens //= self.config.tensor_model_parallel_size
            static_inputs["padding_mask"] = torch.zeros(
                1, local_tokens, dtype=torch.bool, device=device
            )
        elif self.config.create_attention_mask_in_dataloader:
            local_seq_length = seq_length // self.config.context_parallel_size
            static_inputs["attention_mask"] = (
                ~(torch.tril(torch.ones((local_seq_length, seq_length))).bool())
                .to(device)
                .reshape(1, 1, local_seq_length, seq_length)
                .tile(micro_batch_size, 1, 1, 1)
            )

        has_hash_router = self.config.moe_n_hash_layers > 0 and any(
            getattr(module, 'is_hash_layer', False) for module in self.modules()
        )
        if has_hash_router:
            input_ids_shape = (
                (1, self.config.max_seqlen_per_dp_cp_rank)
                if self._is_thd_cuda_graph()
                else (micro_batch_size, seq_length)
            )
            static_inputs["input_ids"] = torch.zeros(
                input_ids_shape, dtype=torch.long, device=device
            )
        return static_inputs

    @staticmethod
    def _decompose_packed_seq_params_to_kwargs(kwargs):
        """Replace PackedSeqParams with graph-copyable tensor fields."""
        packed_seq_params = kwargs.pop('packed_seq_params', None)
        if packed_seq_params is None:
            return
        kwargs['cu_seqlens_q'] = packed_seq_params.cu_seqlens_q
        kwargs['cu_seqlens_kv'] = packed_seq_params.cu_seqlens_kv
        kwargs['cu_seqlens_q_padded'] = packed_seq_params.cu_seqlens_q_padded
        kwargs['cu_seqlens_kv_padded'] = packed_seq_params.cu_seqlens_kv_padded

    def _reconstruct_packed_seq_params_from_kwargs(self, kwargs):
        """Rebuild graph-static PackedSeqParams inside the captured block."""
        if 'cu_seqlens_q' not in kwargs:
            return
        capture_cp_size, capture_cp_group = self._get_thd_cuda_graph_capture_cp()
        max_seqlen = self.config.max_seqlen_per_dp_cp_rank * capture_cp_size
        kwargs['packed_seq_params'] = PackedSeqParams(
            qkv_format='thd',
            cp_partition_mode=self.config.cp_partition_mode,
            cu_seqlens_q=kwargs.pop('cu_seqlens_q'),
            cu_seqlens_kv=kwargs.pop('cu_seqlens_kv'),
            cu_seqlens_q_padded=kwargs.pop('cu_seqlens_q_padded'),
            cu_seqlens_kv_padded=kwargs.pop('cu_seqlens_kv_padded'),
            max_seqlen_q=max_seqlen,
            max_seqlen_kv=max_seqlen,
            local_cp_size=(capture_cp_size if self.config.dynamic_context_parallel else None),
            cp_group=(capture_cp_group if self.config.dynamic_context_parallel else None),
            pad_between_seqs=True,
        )

    def _prepare_pipeline_input_for_chunk_cuda_graph(self, args, kwargs):
        """Expose the real PP input tensor as the TE callable input."""
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
        """Capture one complete decoder chunk with the TE callable interface."""
        if getattr(self.config, 'cuda_graph_granularity', 'layer') != "chunk":
            return super()._te_cuda_graph_capture(*args, **kwargs)

        if self.config.moe_paged_stash:
            from megatron.core.transformer.moe.paged_stash import (
                PagedStashManager,
                paged_stash_init_chunk_handler,
            )

            if PagedStashManager.get_instance().enabled:
                paged_stash_init_chunk_handler(
                    getattr(self, '_te_cuda_graph_vp_size', 1),
                    getattr(self, '_te_cuda_graph_vp_stage', 0),
                )

        args, kwargs = self._prepare_pipeline_input_for_chunk_cuda_graph(args, kwargs)
        self._reconstruct_packed_seq_params_from_kwargs(kwargs)
        kwargs.setdefault('attention_mask', None)

        if self.offload_module_in_cuda_graph:
            from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
                FineGrainedActivationOffloadingInterface as off_interface,
            )

            off_interface.forward_capture_start()
            if args:
                hidden_states = off_interface.backward_record(args[0])
                args = (hidden_states,) + args[1:]
            else:
                hidden_states = off_interface.backward_record(kwargs['hidden_states'])
                kwargs['hidden_states'] = hidden_states
            if not self.pre_process:
                self.input_tensor = hidden_states

        output = self.forward(*args, **kwargs)

        if self.config.moe_paged_stash:
            from megatron.core.transformer.moe.paged_stash import (
                paged_stash_wait_for_stash_to_complete,
            )

            paged_stash_wait_for_stash_to_complete()

        if self.offload_module_in_cuda_graph:
            output = off_interface.backward_capture_start(output)
            off_interface.forward_record()
        return output

    def _te_cuda_graph_replay(self, *args, **kwargs):
        """Replay graph ``microbatch_id`` from the Nmax chunk capture."""
        if getattr(self.config, 'cuda_graph_granularity', 'layer') != "chunk":
            return super()._te_cuda_graph_replay(*args, **kwargs)

        args, kwargs = self._prepare_pipeline_input_for_chunk_cuda_graph(args, kwargs)
        self._activate_dynamic_cp_cuda_graph(kwargs.get('packed_seq_params'))
        self._decompose_packed_seq_params_to_kwargs(kwargs)
        if self._is_thd_cuda_graph() and kwargs.get('attention_mask') is None:
            kwargs.pop('attention_mask', None)

        microbatch_id = getattr(self, 'current_microbatch', 0)
        assert microbatch_id >= 0, "Chunk CUDA graph replay requires a nonnegative microbatch ID."
        assert self.cuda_graphs, "Chunk CUDA graph replay requires a captured graph bank."
        # ``current_microbatch`` is the logical pipeline-schedule ID, while dynamic
        # microbatch capture may keep only a bounded ring of physical graph slots.
        # GraphableMegatronModule owns the logical-to-physical modulo mapping.
        return super()._te_cuda_graph_replay(*args, **kwargs)
