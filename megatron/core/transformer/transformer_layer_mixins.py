# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Transformer layer mixins for TE CUDA graph integration."""

from __future__ import annotations

import logging
from functools import partial
from inspect import signature
from typing import Optional

import torch
from torch import Tensor

from megatron.core.extensions.te_cuda_graphs import TECudaGraphMixin
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import CudaGraphModule
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP
from megatron.core.typed_torch import apply_module
from megatron.core.utils import is_te_min_version, is_torch_min_version, log_single_rank

logger = logging.getLogger(__name__)


class TECudaGraphTransformerLayerMixin(TECudaGraphMixin):
    """Connect TransformerLayer's attention and MLP stages to a single TE graph."""

    def register_te_cuda_graph(self):
        """Initialize Transformer graph state and validate its activation offload scopes."""
        super().register_te_cuda_graph()
        # Batch-first shape of the padding_mask this layer received in eager execution (the
        # cuda_graph_warmup_steps), recorded in _forward_attention before any MLP chunking.
        self._te_cuda_graph_padding_mask_shape_seen: Optional[tuple[int, ...]] = None
        # Batch-first shape of the padding_mask input reserved by the TE CUDA graph of this layer;
        # set in get_layer_static_inputs at capture, None when the graph has no such input.
        self._te_cuda_graph_padding_mask_shape: Optional[tuple[int, ...]] = None
        # (shape, dtype) of the rotary_pos_emb this layer received in eager training (the
        # cuda_graph_warmup_steps), recorded in _forward_attention.
        self._te_cuda_graph_rotary_pos_emb_seen: Optional[tuple[tuple[int, ...], torch.dtype]] = (
            None
        )
        # Whether the TE CUDA graph of this layer takes rotary_pos_emb as an input; set when the
        # graph is captured with the self-attention inside it, None otherwise.
        self._te_cuda_graph_has_rotary_pos_emb: Optional[bool] = None

        # Check the compatibility of fine-grained activation offloading and cuda graph.
        if self.config.fine_grained_activation_offloading:
            cuda_graph_modules = self.config.cuda_graph_modules or []
            if CudaGraphModule.attn in cuda_graph_modules:
                self.offload_attn_norm = False
                log_single_rank(
                    logger,
                    logging.WARNING,
                    "attn_norm offloading is not supported with attn cudagraph. "
                    "Disabling attn_norm offloading.",
                )
            mark_mlp_norm_offloading_not_supported = False
            # For moe layer, mlp_norm offloading isn't supported with attn or moe_router cudagraph.
            if self.is_moe_layer:
                if (
                    CudaGraphModule.attn in cuda_graph_modules
                    or CudaGraphModule.moe_router in cuda_graph_modules
                ):
                    mark_mlp_norm_offloading_not_supported = True
            # For non-moe layer, mlp_norm is the boundary of attn or mlp cudagraph.
            # The only case where mlp_norm offloading is supported is when whole layer is captured.
            elif (
                CudaGraphModule.attn in cuda_graph_modules
                and CudaGraphModule.mlp not in cuda_graph_modules
            ) or (
                CudaGraphModule.attn not in cuda_graph_modules
                and CudaGraphModule.mlp in cuda_graph_modules
            ):
                mark_mlp_norm_offloading_not_supported = True
            if mark_mlp_norm_offloading_not_supported:
                self.offload_mlp_norm = False
                log_single_rank(
                    logger,
                    logging.WARNING,
                    "mlp_norm offloading is not supported with the current cudagraph scope. "
                    "Disabling mlp_norm offloading.",
                )
        # Set the offload module in cuda graph flag.
        self.offload_module_in_cuda_graph = False
        cuda_graph_modules = self.config.cuda_graph_modules or []
        if CudaGraphModule.attn in cuda_graph_modules:
            if self.offload_core_attn or self.offload_attn_proj or self.offload_qkv_linear:
                self.offload_module_in_cuda_graph = True
        if not self.is_moe_layer and CudaGraphModule.mlp in cuda_graph_modules:
            if self.offload_mlp_norm:
                self.offload_module_in_cuda_graph = True
        if self.offload_module_in_cuda_graph:
            assert is_torch_min_version(
                "2.9.0a0"
            ), "Offloading modules captured in cuda graph requires torch>=2.9.0."
            assert is_te_min_version(
                "2.14.0"
            ), "Offloading modules captured in cuda graph requires TE>=2.14.0."
            assert (
                self.config.cuda_graph_warmup_steps > 0
            ), "Fine-grained activation offloading needs cuda_graph_warmup_steps > 0."
        return self

    @staticmethod
    def can_recompute_pre_mlp_layernorm_for_te_cudagraph(config, is_moe_layer):
        """Check whether a TE MoE router graph permits pre-MLP layernorm recomputation."""
        if (
            config.cuda_graph_impl != "transformer_engine"
            or not is_moe_layer
            or CudaGraphModule.moe_router not in config.cuda_graph_modules
        ):
            # TE router capture does not constrain this layernorm.
            return True
        if (
            config.moe_shared_expert_intermediate_size is not None
            and config.moe_shared_expert_overlap
        ):
            # If shared expert overlap is used, we cannot make the pre-mlp layernorm
            # recomputation, because the shared expert takes the layernorm output as
            # input, and it is outside of the CUDA graph scope.
            log_single_rank(
                logger,
                logging.WARNING,
                "pre_mlp_layernorm recompute is not supported with moe router "
                "cudagraph + shared expert overlap. Disabling pre_mlp_layernorm "
                "recompute.",
            )
            return False
        if CudaGraphModule.moe_preprocess in config.cuda_graph_modules and (
            config.moe_token_dispatcher_type == "alltoall" or config.moe_latent_size
        ):
            # Only when capturing the preprocess part and using alltoall token
            # dispatcher or latent MoE can we make the pre-mlp layernorm recomputation.
            # Because in other cases the layernorm output returns directly as one of the
            # outputs of the cudagraph, which will be allocated a static buffer, thus
            # not able to be released.
            return True
        log_single_rank(
            logger,
            logging.WARNING,
            "pre_mlp_layernorm recompute is only supported with moe router + "
            "preprocess cudagraph will alltoall token dispatcher or latent MoE. "
            "Disabling pre_mlp_layernorm recompute.",
        )
        return False

    def _forward_attention(self, *args, **kwargs):
        """Observe the whole-layer mask and the rotary embedding of the warmup calls."""
        if self.config.cuda_graph_impl == "transformer_engine" and (
            self._te_cuda_graph_padding_mask_shape_seen is None
            or (self.training and self._te_cuda_graph_rotary_pos_emb_seen is None)
        ):
            inputs = signature(super()._forward_attention).bind(*args, **kwargs).arguments
            self._te_cuda_graph_record_padding_mask_shape(inputs.get("padding_mask"))
            rotary_pos_emb = inputs.get("rotary_pos_emb")
            if (
                self.training
                and isinstance(rotary_pos_emb, Tensor)
                and self._te_cuda_graph_rotary_pos_emb_seen is None
            ):
                # Capture reserves a rotary_pos_emb input of this layout and dtype.
                self._te_cuda_graph_rotary_pos_emb_seen = (
                    tuple(rotary_pos_emb.shape),
                    rotary_pos_emb.dtype,
                )
        return super()._forward_attention(*args, **kwargs)

    def _te_cuda_graph_record_padding_mask_shape(self, padding_mask):
        if padding_mask is not None and self._te_cuda_graph_padding_mask_shape_seen is None:
            # Whole-layer batch-first mask shape, taken before MLP chunking splits the mask, so TE
            # CUDA graph capture reserves (and replay passes) a mask of the layer's input shape.
            self._te_cuda_graph_padding_mask_shape_seen = tuple(padding_mask.shape)

    def get_layer_static_inputs(self, seq_length, micro_batch_size):
        """
        Get the static inputs for the transformer layer. Besides the hidden_states that is
        generated in TECudaGraphMixin, we also add the attention_mask.

        For THD + CUDA Graph: generates static cu_seqlens tensors instead of attention_mask.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the static inputs for the layer.
        """
        static_inputs = super().get_layer_static_inputs(seq_length, micro_batch_size)

        # Captured forward needs attention-side static input only when this
        # layer's attention is inside the captured scope.
        attn_in_graph = self._attention_in_cuda_graph()

        if self._is_thd_cuda_graph():
            if attn_in_graph:
                # Static cu_seqlens shaped [cuda_graph_thd_max_packed_sequences + 1]. We seed it as
                # one full-length sequence (covers the worst case at capture):
                #   cu_seqlens = [0, max_T, max_T, ..., max_T]
                # which represents a single packed sequence followed by zero-length
                # entries. cu_seqlens_q / kv / *_padded all share this layout.
                max_T = self.config.max_seqlen_per_dp_cp_rank * self.config.context_parallel_size
                cu_seqlens = torch.zeros(
                    self.config.cuda_graph_thd_max_packed_sequences + 1,
                    dtype=torch.int32,
                    device=static_inputs["hidden_states"].device,
                )
                cu_seqlens[1:] = max_T
                for name in (
                    'cu_seqlens_q',
                    'cu_seqlens_kv',
                    'cu_seqlens_q_padded',
                    'cu_seqlens_kv_padded',
                ):
                    static_inputs[name] = cu_seqlens.clone()
        elif attn_in_graph:
            slen_per_cp = seq_length // self.config.context_parallel_size
            static_inputs["attention_mask"] = (
                ~(torch.tril(torch.ones((slen_per_cp, seq_length))).bool())
                .to(torch.cuda.current_device())
                .reshape(1, 1, slen_per_cp, seq_length)
                .tile(micro_batch_size, 1, 1, 1)
            )
        # TE copies only the captured inputs into the graph at replay and ignores any other
        # keyword argument, so a rotary embedding that is not a graph input never reaches the
        # graphed attention. Reserve one with the layout and dtype that the eager warmup steps
        # passed to this layer, sized to the graph's sequence: its own buffer per (layer,
        # microbatch), since e.g. multimodal RoPE ('mrope') or a vision encoder's 2D RoPE
        # changes with every batch.
        if attn_in_graph and self._te_cuda_graph_rotary_pos_emb_seen is not None:
            # Rows per graph: packed THD sequences keep the rotary table of the whole packed
            # capacity (the models skip the context-parallel slice for them, as cu_seqlens above),
            # every other layout is sliced to the CP-local sequence length.
            if self._is_thd_cuda_graph():
                rotary_rows = (
                    self.config.max_seqlen_per_dp_cp_rank * self.config.context_parallel_size
                )
            else:
                rotary_rows = seq_length // self.config.context_parallel_size
            rotary_shape, rotary_dtype = self._te_cuda_graph_rotary_pos_emb_seen
            rotary_shape = list(rotary_shape)
            mrope_section = getattr(self.config, "mrope_section", None)
            if (
                mrope_section is not None
                and len(rotary_shape) == 4
                and rotary_shape[0] == 3
                and rotary_shape[-1] == sum(mrope_section)
            ):
                # Raw per-axis mRoPE frequencies [3, b, s, d / 2] (fused mRoPE path):
                # the sequence is axis 2.
                rotary_shape[2] = rotary_rows
            else:
                rotary_shape[0] = rotary_rows
            static_inputs["rotary_pos_emb"] = torch.zeros(
                rotary_shape, dtype=rotary_dtype, device=torch.cuda.current_device()
            )
        if self.is_moe_layer and getattr(self.mlp.router, 'is_hash_layer', False):
            # [batch, seq] of the static hidden states; THD graphs use a single packed row.
            static_inputs["input_ids"] = torch.zeros(
                (static_inputs["hidden_states"].shape[1], static_inputs["hidden_states"].shape[0]),
                dtype=torch.long,
                device=torch.cuda.current_device(),
            )

        # Reserve a padding_mask graph input when the eager warmup steps before capture gave this
        # layer one, so the graphed router/z-loss/expert-bias see the same mask as eager mode. The
        # shape is the layer's whole-batch mask (recorded in _forward_attention, before any MLP
        # chunking), which is also what replay passes. TE fixes the kwarg set at capture, so
        # replays must then always pass a mask; see _te_cuda_graph_replay for the all-False default.
        self._te_cuda_graph_padding_mask_shape = None
        if self._moe_router_in_cuda_graph():
            mask_shape = self._te_cuda_graph_padding_mask_shape_seen
            if mask_shape is not None:
                self._te_cuda_graph_padding_mask_shape = tuple(mask_shape)
                static_inputs["padding_mask"] = torch.zeros(
                    mask_shape, dtype=torch.bool, device=torch.cuda.current_device()
                )
        return static_inputs

    def _te_cuda_graph_padding_mask(
        self, padding_mask: Optional[Tensor], device: torch.device
    ) -> Optional[Tensor]:
        """Reconcile the replay padding_mask with the graph's captured input set.

        Returns the mask to pass to the graph: the caller's mask, an all-False mask when the graph
        was captured with a padding_mask input but this batch has none, or None when the graph has
        no such input. Raises when a mask is passed to a graph captured without one, since the
        graph would silently ignore it and diverge from eager execution.
        """
        graph_mask_shape = self._te_cuda_graph_padding_mask_shape
        if graph_mask_shape is None:
            if padding_mask is not None and self._moe_router_in_cuda_graph():
                raise RuntimeError(
                    "padding_mask was passed to a TE CUDA graph that was captured without a "
                    "padding_mask input: no mask reached this layer during "
                    "cuda_graph_warmup_steps (is it 0?). The graphed router would silently ignore "
                    "the mask. Run at least one eager warmup step with masked batches."
                )
            return None
        if padding_mask is None:
            padding_mask = torch.zeros(graph_mask_shape, dtype=torch.bool, device=device)
        return padding_mask

    def _get_te_cuda_graph_replay_args(
        self, hidden_states: Tensor, *, padding_mask: Optional[Tensor] = None, **kwargs
    ):
        """Adapt padding and attention masks, then prepare tensor inputs for TE."""
        kwargs = {k: v for k, v in kwargs.items() if v is not None or k == "attention_mask"}
        # Attention inputs are absent when attention ran eagerly before the graph.
        if (
            self.config.cuda_graph_modules
            and CudaGraphModule.attn not in self.config.cuda_graph_modules
        ):
            kwargs.pop("attention_mask", None)
        # The graph only takes padding_mask when it was captured with that input; the eager MoE
        # steps after the graph keep using the caller's mask.
        graph_padding_mask = self._te_cuda_graph_padding_mask(padding_mask, hidden_states.device)
        if graph_padding_mask is not None:
            kwargs["padding_mask"] = graph_padding_mask
        cudagraph_args, cudagraph_kwargs = super()._get_te_cuda_graph_replay_args(
            hidden_states, **kwargs
        )

        try:
            import transformer_engine.pytorch as te  # pylint: disable=unused-import

            def get_zero_attention_mask(slen_per_tpcp, micro_batch_size):
                sequence_parallel = self.config.sequence_parallel
                tensor_model_parallel_size = self.config.tensor_model_parallel_size
                slen_per_cp = (
                    slen_per_tpcp * tensor_model_parallel_size
                    if sequence_parallel
                    else slen_per_tpcp
                )
                slen = slen_per_cp * self.config.context_parallel_size
                return torch.zeros(
                    (micro_batch_size, 1, slen_per_cp, slen),
                    dtype=torch.bool,
                    device=torch.cuda.current_device(),
                )

            if not is_te_min_version("1.10.0"):
                # TE version < 1.10.0 does not support keyword arguments with CUDA graph.
                cudagraph_args = list(cudagraph_args)
                for k, v in cudagraph_kwargs.items():
                    if k == "attention_mask":
                        if v is not None:
                            cudagraph_args.append(v)
                            cudagraph_kwargs[k] = None
                        else:
                            cudagraph_args.append(
                                get_zero_attention_mask(
                                    hidden_states.size(0), hidden_states.size(1)
                                )
                            )
                    elif k != 'is_first_microbatch':
                        assert v is None, "Keyword Arguments not supported with CUDA graph."
            elif (
                'attention_mask' in cudagraph_kwargs and cudagraph_kwargs['attention_mask'] is None
            ):
                # The attention_mask can be None when there is no padding to the input sequence.
                # However, an attention_mask Tensor must be passed into cudagraph for replay, so
                # we create an equivalent zero Tensor as the attention_mask.
                if self._is_thd_cuda_graph():
                    cudagraph_kwargs.pop("attention_mask")
                else:
                    cudagraph_kwargs["attention_mask"] = get_zero_attention_mask(
                        hidden_states.size(0), hidden_states.size(1)
                    )
        except ImportError:
            raise RuntimeError("CUDAGraph requires TransformerEngine, but not installed")
        return tuple(cudagraph_args), cudagraph_kwargs

    def is_graphable_for_scope(self, modules):
        """Return whether this layer contains a requested TE graph scope."""
        # If cuda_graph_modules is not set, every layer is graphed.
        if not modules:
            return True

        # import modules here to avoid a circular import
        from megatron.core.transformer.moe.moe_layer import MoELayer

        if CudaGraphModule.attn in modules and not (
            isinstance(self.self_attention, IdentityOp)
            and isinstance(self.cross_attention, IdentityOp)
        ):
            # attn layer.
            return True
        if (
            CudaGraphModule.moe in modules
            or CudaGraphModule.moe_router in modules
            or CudaGraphModule.moe_preprocess in modules
        ) and isinstance(self.mlp, MoELayer):
            # moe layer.
            return True
        if CudaGraphModule.mlp in modules and isinstance(self.mlp, MLP):
            # mlp layer.
            return True
        return False

    def _attention_in_cuda_graph(self) -> bool:
        """Whether this layer's attention is inside the configured capture scope."""
        return not isinstance(self.self_attention, IdentityOp) and (
            not self.config.cuda_graph_modules
            or CudaGraphModule.attn in self.config.cuda_graph_modules
        )

    def _moe_cuda_graph_outputs_shared_expert(self) -> bool:
        """Whether a ``moe_router`` partial CUDA graph of this MoE layer outputs the shared
        expert's result (see TEPartialCaptureMoEMixin._te_cuda_graph_outputs_shared_expert)."""
        return self.mlp._te_cuda_graph_outputs_shared_expert()

    def _moe_router_in_cuda_graph(self) -> bool:
        """Return True when this layer's MoE router runs inside the TE CUDA graph."""
        return self.is_moe_layer and (
            not self.config.cuda_graph_modules
            or CudaGraphModule.moe in self.config.cuda_graph_modules
            or CudaGraphModule.moe_router in self.config.cuda_graph_modules
        )

    def _te_cuda_graph_mlp_capture_output(self, mlp_output_with_bias, residual):
        """Return partial-capture outputs, or None to continue normal execution."""
        from megatron.core.transformer.cuda_graphs import is_graph_capturing

        if (
            self.is_moe_layer
            and self.training
            and is_graph_capturing()
            and CudaGraphModule.moe_router in self.config.cuda_graph_modules
        ):
            if self.recompute_pre_mlp_layernorm:
                # Register the recompute hooks to all the cudagraph output tensors, because some
                # tensors are in parallel execution paths and they all need pre_mlp_layernorm to be
                # recomputed in backward pass. For example, the router path and the shared expert
                # path. So only register in one path is risky.
                for tensor in mlp_output_with_bias:
                    self.pre_mlp_norm_checkpoint.discard_output_and_register_recompute(tensor)
            return list(mlp_output_with_bias) + [residual]

        return None

    def _get_submodules_under_cudagraphs(self):
        """
        Get the submodules that are covered by cudagraphs.
        """
        if not self.config.cuda_graph_modules:
            return super()._get_submodules_under_cudagraphs()

        submodules = []
        if CudaGraphModule.attn in self.config.cuda_graph_modules:
            submodules += [
                self.input_layernorm,
                self.self_attention,
                self.pre_cross_attn_layernorm,
                self.cross_attention,
            ]
        if (not self.is_moe_layer and CudaGraphModule.mlp in self.config.cuda_graph_modules) or (
            self.is_moe_layer and CudaGraphModule.moe in self.config.cuda_graph_modules
        ):
            submodules += [self.pre_mlp_layernorm, self.mlp]
        elif self.is_moe_layer and CudaGraphModule.moe_router in self.config.cuda_graph_modules:
            submodules += [self.pre_mlp_layernorm, self.mlp.router]
            if self._moe_cuda_graph_outputs_shared_expert():
                submodules += [self.mlp.shared_experts]
        return submodules

    def _decompose_packed_seq_params_to_kwargs(self, kwargs):
        """Decompose PackedSeqParams into individual tensor kwargs for CUDA graph.

        CUDA graph requires all inputs to be tensors. This extracts the cu_seqlens
        tensor fields from PackedSeqParams into individual kwargs. max_seqlen_q/kv
        are omitted because they are static and reading them from a CUDA tensor is
        forbidden during graph capture. They are restored from config in
        _reconstruct_packed_seq_params_from_kwargs.
        """
        packed_seq_params = kwargs.pop('packed_seq_params', None)
        if packed_seq_params is None:
            return
        for name, offsets in self._thd_graph_cu_seqlens_kwargs(packed_seq_params).items():
            if (
                offsets is None
                or offsets.numel() != self.config.cuda_graph_thd_max_packed_sequences + 1
            ):
                raise ValueError(
                    "Packed CUDA graph inputs must use the configured metadata capacity."
                )
            kwargs[name] = offsets

    def _reconstruct_packed_seq_params_from_kwargs(self, kwargs):
        """Reconstruct PackedSeqParams from individual tensor kwargs (CUDA graph path).

        During CUDA graph capture/replay, PackedSeqParams fields are decomposed into
        individual cu_seqlens tensor kwargs. This method reassembles them into a
        PackedSeqParams. max_seqlen_q/kv are taken from config since they are always
        the padded static value and cannot be read from CUDA tensors during graph capture.
        pad_between_seqs is conservatively enabled because its Python value is not a
        graph input and inferring it from CUDA tensors would synchronize during capture.
        """
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
            # CUDA graph inputs do not carry this Python bool. Sequence-packing
            # metadata may contain valid/physical gaps, so use the conservative
            # graph-static value instead of asking TE to infer it with a CUDA
            # tensor comparison during capture. This restricts TE backend
            # selection for THD to cuDNN fused attention.
            pad_between_seqs=True,
        )

    def _te_cuda_graph_capture_forward(self, hidden_states: Tensor, *args, **kwargs):
        """
        CUDA Graph capture for this layer using TE interface.
        There are some differences from the normal pass:
        1. In some conditions CUDA graph cannot cover the entire layer. The `cuda_graph_modules`
           attribute can be set to control the scope of the CUDA graph.
        2. If context is None, it cannot be returned as output.
        """
        if args:
            positional_inputs = (
                signature(super()._forward_attention).bind(hidden_states, *args).arguments
            )
            positional_inputs.pop("hidden_states")
            kwargs = dict(**positional_inputs, **kwargs)
        kwargs.setdefault("attention_mask", None)
        self._reconstruct_packed_seq_params_from_kwargs(kwargs)
        kwargs = {k: v for k, v in kwargs.items() if v is not None or k == "attention_mask"}

        context = None
        if (
            not self.config.cuda_graph_modules
            or CudaGraphModule.attn in self.config.cuda_graph_modules
        ):
            if not isinstance(self.self_attention, IdentityOp):
                # Replays must pass a rotary embedding exactly when the graph takes one.
                self._te_cuda_graph_has_rotary_pos_emb = kwargs.get("rotary_pos_emb") is not None
            hidden_states, context = self._forward_attention(hidden_states, **kwargs)

        if (
            not self.config.cuda_graph_modules
            or (not self.is_moe_layer and CudaGraphModule.mlp in self.config.cuda_graph_modules)
            or (
                self.is_moe_layer
                and (
                    CudaGraphModule.moe in self.config.cuda_graph_modules
                    or CudaGraphModule.moe_router in self.config.cuda_graph_modules
                )
            )
        ):
            # padding_mask is a graph input only when get_layer_static_inputs reserved it.
            hidden_states = self._forward_mlp(
                hidden_states,
                padding_mask=kwargs.get("padding_mask"),
                input_ids=kwargs.get("input_ids", None),
            )
        if not isinstance(hidden_states, list) and not isinstance(hidden_states, tuple):
            cuda_graph_outputs = [hidden_states]
        else:
            cuda_graph_outputs = list(hidden_states)
        if context is not None:
            cuda_graph_outputs.append(context)
        return tuple(cuda_graph_outputs)

    def _te_cuda_graph_prepare_replay(self, hidden_states: Tensor, *args, **kwargs):
        """Run the eager prefix and bind this call's graph and continuation inputs."""
        if args:
            positional_inputs = (
                signature(super()._forward_attention).bind(hidden_states, *args).arguments
            )
            positional_inputs.pop("hidden_states")
            kwargs = dict(**positional_inputs, **kwargs)
        kwargs.setdefault("attention_mask", None)
        kwargs = {k: v for k, v in kwargs.items() if v is not None or k == "attention_mask"}

        context = None
        if (
            self.config.cuda_graph_modules
            and CudaGraphModule.attn not in self.config.cuda_graph_modules
        ):
            input_ids = kwargs.get("input_ids", None)
            hidden_states, context = self._forward_attention(hidden_states, **kwargs)
            # Attention consumed the other kwargs eagerly; the MoE padding mask is still needed
            # by the graphed router and by the eager MoE steps after the graph, and hash-routing
            # layers still need input_ids.
            kwargs = {"padding_mask": kwargs.get("padding_mask")}
            if input_ids is not None:
                kwargs["input_ids"] = input_ids

        packed = kwargs.get('packed_seq_params')
        if packed is not None and self._is_thd_cuda_graph():
            kwargs = dict(kwargs)
            self._decompose_packed_seq_params_to_kwargs(kwargs)

        assert (kwargs.get('inference_context') is None) and (
            kwargs.get('packed_seq_params') is None
        ), (
            "CUDA graph accepts only Tensor inputs. "
            "inference_context and packed_seq_params are excluded from input list. "
            "For inference cuda graph, please use cuda_graph_impl=local instead."
        )
        # TE ignores keyword arguments that a graph was not captured with, so a rotary embedding
        # passed to a graph without that input would silently not be applied.
        passes_rotary_pos_emb = kwargs.get("rotary_pos_emb") is not None
        if self._te_cuda_graph_has_rotary_pos_emb not in (None, passes_rotary_pos_emb):
            raise RuntimeError(
                f"TransformerLayer {self.layer_number}: rotary_pos_emb was passed to a TE CUDA "
                "graph that was captured without a rotary_pos_emb input and would ignore it. The "
                "input is reserved only when the eager steps before the capture "
                "(cuda_graph_warmup_steps) pass a rotary embedding to the layer."
                if passes_rotary_pos_emb
                else f"TransformerLayer {self.layer_number}: the TE CUDA graph was captured with a "
                "rotary_pos_emb input, but the replay passes none."
            )

        kwargs.pop("inference_context", None)
        kwargs.pop("packed_seq_params", None)
        return (
            partial(self._te_cuda_graph_replay_graph, hidden_states, **kwargs),
            partial(
                self._te_cuda_graph_finish_replay,
                context=context,
                graph_has_context=kwargs.get("context") is not None,
                padding_mask=kwargs.get("padding_mask"),
                input_ids=kwargs.get("input_ids"),
                packed_seq_params=packed,
            ),
        )

    def _te_cuda_graph_finish_replay(
        self,
        cuda_graph_output,
        *,
        context: Optional[Tensor] = None,
        graph_has_context: bool = False,
        padding_mask: Optional[Tensor] = None,
        input_ids: Optional[Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ):
        """Finish the eager layer stages using this call's continuation inputs."""
        cuda_graph_output = list(cuda_graph_output)
        if graph_has_context:
            context = cuda_graph_output.pop()

        if (
            not self.config.cuda_graph_modules
            or (not self.is_moe_layer and CudaGraphModule.mlp in self.config.cuda_graph_modules)
            or (self.is_moe_layer and CudaGraphModule.moe in self.config.cuda_graph_modules)
        ):
            # CUDA Graph captures the whole MLP/MoE part. CUDA Graph output is the layer output.
            assert len(cuda_graph_output) == 1, "CUDA Graph output should be the layer output."
            output = cuda_graph_output.pop()
            assert not self.config.overlap_moe_expert_parallel_comm, "EP overlap must be \
                disabled when CUDA graph captures the whole MLP/MoE part."
        elif self.is_moe_layer and CudaGraphModule.moe_router in self.config.cuda_graph_modules:
            # CUDA Graph partially captures the MoE.
            # The rest of the layer should go to the normal pass.
            # residual is the last element in the CUDA graph output.
            residual = cuda_graph_output.pop()
            mlp_output_with_bias = self.mlp._te_cuda_graph_resume(
                cuda_graph_output, padding_mask, packed_seq_params=packed_seq_params
            )
            if self.config.overlap_moe_expert_parallel_comm:
                return (residual, *mlp_output_with_bias)

            # If we early returned, layernorm recompute hooks were attached to the output buffer
            # of the cudagraph, so disable the recompute hooks inside _apply_mlp_bda_step
            recompute_pre_mlp_layernorm = self.recompute_pre_mlp_layernorm
            self.recompute_pre_mlp_layernorm = False
            output = self._apply_mlp_bda_step(mlp_output_with_bias, residual)
            self.recompute_pre_mlp_layernorm = recompute_pre_mlp_layernorm
        else:
            # If EP overlap is enabled, needs to return same outputs as submodule.attn
            if self.config.overlap_moe_expert_parallel_comm:
                assert len(cuda_graph_output) == 1, "CUDA Graph output should be the layer output."
                residual = cuda_graph_output.pop()
                if not self.is_moe_layer:
                    return residual, None, None, None
                hidden_states = apply_module(self.pre_mlp_layernorm)(residual)
                if self._pre_mlp_layernorm_returns_residual:
                    hidden_states, residual = hidden_states

                return (
                    residual,
                    *self.mlp._te_cuda_graph_pre_dispatch(
                        hidden_states, padding_mask, packed_seq_params=packed_seq_params
                    ),
                )

            # CUDA Graph does not capture the MLP/MoE part at all.
            output = self._forward_mlp(
                *cuda_graph_output,
                padding_mask=padding_mask,
                input_ids=input_ids,
                packed_seq_params=packed_seq_params,
            )
        return output, context


class TECudaGraphHyperConnectionTransformerLayerMixin(TECudaGraphTransformerLayerMixin):
    """Adapt TE graph inputs and parameter hooks for hyper connection layers."""

    def get_layer_static_inputs(self, seq_length, micro_batch_size):
        """Override to produce n-stream hidden_states of shape [s, b, n*C].

        CUDA graph capture creates static buffers whose shapes are determined by
        this method. The base class returns [s, b, C], but mHC layers operate on
        n-stream hidden states of shape [s, b, n*C].
        """
        static_inputs = super().get_layer_static_inputs(seq_length, micro_batch_size)
        hs = static_inputs["hidden_states"]
        n = self.config.mhc_num_residual_streams
        static_inputs["hidden_states"] = torch.ones(
            (hs.shape[0], hs.shape[1], n * self.config.hidden_size),
            dtype=hs.dtype,
            requires_grad=hs.requires_grad,
            device=hs.device,
        )
        return static_inputs

    def _get_submodules_under_cudagraphs(self):
        """Override to include hyper connection modules.

        The base TransformerLayer._get_submodules_under_cudagraphs does not include
        self_attention_hyper_connection / mlp_hyper_connection. Their learnable
        parameters (mapping_proj, alpha_*, bias) need manual pre-forward hooks
        during CUDA graph replay so that parameter all-gathers are triggered.
        """
        submodules = super()._get_submodules_under_cudagraphs()

        if not self.config.cuda_graph_modules:
            return submodules

        if (
            CudaGraphModule.attn in self.config.cuda_graph_modules
            and self.self_attention_hyper_connection is not None
        ):
            submodules.append(self.self_attention_hyper_connection)
        if (
            CudaGraphModule.mlp in self.config.cuda_graph_modules
            and not self.is_moe_layer
            and self.mlp_hyper_connection is not None
        ):
            submodules.append(self.mlp_hyper_connection)
        return submodules
