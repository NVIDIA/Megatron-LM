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
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.typed_torch import apply_module
from megatron.core.utils import (
    is_te_min_version,
    is_torch_min_version,
    log_single_rank,
    nvtx_range_pop,
    nvtx_range_push,
)

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
        """Observe the whole-layer mask for both normal and EP-overlap warmup calls."""
        if (
            self.config.cuda_graph_impl == "transformer_engine"
            and self._te_cuda_graph_padding_mask_shape_seen is None
        ):
            inputs = signature(super()._forward_attention).bind(*args, **kwargs)
            self._te_cuda_graph_record_padding_mask_shape(inputs.arguments.get("padding_mask"))
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

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the static inputs for the layer.
        """
        static_inputs = super().get_layer_static_inputs(seq_length, micro_batch_size)

        if getattr(self.config, "sequence_packing_scheduler", None) is not None:
            capacity = self.config.max_seqlen_per_dp_cp_rank * self.config.context_parallel_size
            offsets = torch.zeros(
                self.config.cuda_graph_thd_max_packed_sequences + 1,
                dtype=torch.int32,
                device=static_inputs["hidden_states"].device,
            )
            offsets[1:] = capacity
            for name in (
                "cu_seqlens_q",
                "cu_seqlens_kv",
                "cu_seqlens_q_padded",
                "cu_seqlens_kv_padded",
            ):
                static_inputs[name] = offsets.clone()
        elif not isinstance(self.self_attention, IdentityOp) and (
            not self.config.cuda_graph_modules
            or CudaGraphModule.attn in self.config.cuda_graph_modules
        ):
            slen_per_cp = seq_length // self.config.context_parallel_size
            static_inputs["attention_mask"] = (
                ~(torch.tril(torch.ones((slen_per_cp, seq_length))).bool())
                .to(torch.cuda.current_device())
                .reshape(1, 1, slen_per_cp, seq_length)
                .tile(micro_batch_size, 1, 1, 1)
            )
        if self.is_moe_layer and getattr(self.mlp.router, 'is_hash_layer', False):
            static_inputs["input_ids"] = torch.zeros(
                (micro_batch_size, static_inputs["hidden_states"].shape[0]),
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
            # TE runs the graphed callable eagerly on these static inputs (warmup) before it
            # captures it. The all-ones hidden states give every token the same router logits, so
            # every token of every EP rank picks the same top-k experts. A megakernel inside the
            # graph sizes its per-expert schedule for real routing (MOK's
            # schedule_capacity_multiplier) and can trap on that skew, so its static mask marks
            # every token as padding and the warmup routes none. Replay copies the real mask into
            # this buffer. Native MoE layers keep the all-False mask.
            megakernel_in_graph = self._cuda_graph_megakernel() is not None
            if (
                megakernel_in_graph
                and mask_shape is None
                and not self.config.moe_router_force_load_balancing
            ):
                # Without a mask input, nothing keeps the warmup's routes off that skew, except
                # forced load balancing, a benchmark-only mode that replaces the router logits with
                # random ones. TECudaGraphHelper builds the static inputs of every layer before it
                # calls TE's make_graphed_callables, so this raises before any warmup forward.
                raise RuntimeError(
                    "This layer's TE CUDA graph contains the MoE megakernel, but no padding_mask "
                    "reached the layer during the eager cuda_graph_warmup_steps. Without a "
                    "padding_mask graph input, TE's capture warmup would route every token of its "
                    "all-ones static input to the same experts, which can exceed the megakernel's "
                    "static schedule capacity (MOK traps on the device). Pass a padding_mask model "
                    "input (an all-False mask is enough) in at least one eager warmup step "
                    "(cuda_graph_warmup_steps > 0), or leave the MoE layer out of the CUDA graph "
                    "(for example cuda_graph_modules=['attn']). Benchmarks that set "
                    "moe_router_force_load_balancing, which replaces the router logits with "
                    "random ones, are exempt."
                )
            if mask_shape is not None:
                self._te_cuda_graph_padding_mask_shape = tuple(mask_shape)
                static_inputs["padding_mask"] = torch.full(
                    mask_shape,
                    megakernel_in_graph,
                    dtype=torch.bool,
                    device=torch.cuda.current_device(),
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
                if self.config.sequence_packing_scheduler is not None:
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
            from megatron.core.ssm.gated_delta_net.common import _GDNBase

            # GDN reads packed sequence offsets on the host, which capture forbids, so it
            # stays eager for scheduler-packed THD batches.
            if getattr(self.config, "sequence_packing_scheduler", None) is not None and isinstance(
                self.self_attention, _GDNBase
            ):
                return False
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

    def _moe_router_in_cuda_graph(self) -> bool:
        """Return True when this layer's MoE router runs inside the TE CUDA graph."""
        return self.is_moe_layer and (
            not self.config.cuda_graph_modules
            or CudaGraphModule.moe in self.config.cuda_graph_modules
            or CudaGraphModule.moe_router in self.config.cuda_graph_modules
        )

    def _cuda_graph_megakernel(self):
        """Return the MoE megakernel backend that this layer's TE CUDA graph runs, else None.

        The graph contains the megakernel when its scope covers the whole MoE (``moe``, or every
        module when ``cuda_graph_modules`` is empty). Otherwise the whole MoE layer runs eagerly:
        TransformerConfig rejects the ``moe_router`` and ``moe_preprocess`` scopes for MOK, whose
        graphs would end between the router and the megakernel.
        """
        if not self.is_moe_layer or (
            self.config.cuda_graph_modules
            and CudaGraphModule.moe not in self.config.cuda_graph_modules
        ):
            return None
        return getattr(self.mlp, "megakernel_experts", None)

    def _before_te_cuda_graph_replay(self):
        """Refresh a megakernel's cached weight views before a graph that contains it replays.

        The replay runs no Python, so a backend that converts weights once per optimizer step
        (MOK's MXFP8 scale swizzle) does it here: after the manual parameter all-gather hooks and
        before the graph on the same stream. MOK refreshes only at the first microbatch.
        """
        megakernel = self._cuda_graph_megakernel()
        if megakernel is not None:
            megakernel.refresh_weight_views()

    def _after_te_cuda_graphs_deleted(self):
        """Release a megakernel from the CUDA graphs of this layer, which were just deleted.

        A backend may pin state while a graph references it (MOK's cached weight views must keep
        their addresses); with the graphs gone, reloading or moving its parameters is safe again.
        """
        megakernel = self._cuda_graph_megakernel()
        if megakernel is not None:
            megakernel.cuda_graphs_deleted()

    def _te_cuda_graph_mlp_capture_output(self, mlp_output_with_bias, residual, mlp_state):
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
            return self._pack_mlp_cuda_graph_state(mlp_output_with_bias, residual, mlp_state)

        return None

    def _pack_mlp_cuda_graph_state(self, outputs, residual, mlp_state):
        """Append the layer's residual state to the partial MoE graph outputs."""
        assert not mlp_state, "Override partial graph state packing for subclass MLP state."
        return list(outputs) + [residual]

    def _unpack_mlp_cuda_graph_state(self, outputs):
        """Remove the residual state, leaving only router/preprocess intermediates."""
        return outputs.pop(), ()

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
            if (
                self.config.moe_shared_expert_intermediate_size is not None
                and not self.config.moe_shared_expert_overlap
            ):
                submodules += [self.mlp.shared_experts]
        return submodules

    def _reconstruct_packed_graph_inputs(self, kwargs):
        """Rebuild THD metadata inside capture using graph-static scalar capacities."""
        if "cu_seqlens_q" not in kwargs:
            return
        capacity = self.config.max_seqlen_per_dp_cp_rank * self.config.context_parallel_size
        kwargs["packed_seq_params"] = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=kwargs.pop("cu_seqlens_q"),
            cu_seqlens_kv=kwargs.pop("cu_seqlens_kv"),
            cu_seqlens_q_padded=kwargs.pop("cu_seqlens_q_padded"),
            cu_seqlens_kv_padded=kwargs.pop("cu_seqlens_kv_padded"),
            max_seqlen_q=capacity,
            max_seqlen_kv=capacity,
            # Runtime packs can contain valid/physical gaps. A fixed True value
            # avoids a CUDA-to-host tensor comparison during attention capture.
            pad_between_seqs=True,
        )

    def _decompose_packed_graph_inputs(self, kwargs):
        """Replace scheduler THD metadata with the graph's fixed-capacity tensor inputs.

        Returns a new kwargs dict without ``packed_seq_params`` when the call carries
        scheduler-packed metadata, and ``kwargs`` unchanged otherwise. This is the replay
        counterpart of ``_reconstruct_packed_graph_inputs``.
        """
        packed = kwargs.get("packed_seq_params")
        if packed is None or self.config.sequence_packing_scheduler is None:
            return kwargs
        kwargs = dict(kwargs)
        kwargs.pop("packed_seq_params")
        for name in (
            "cu_seqlens_q",
            "cu_seqlens_kv",
            "cu_seqlens_q_padded",
            "cu_seqlens_kv_padded",
        ):
            offsets = getattr(packed, name)
            if (
                offsets is None
                or offsets.numel() != self.config.cuda_graph_thd_max_packed_sequences + 1
            ):
                raise ValueError(
                    "Packed CUDA graph inputs must use the configured metadata capacity."
                )
            kwargs[name] = offsets
        return kwargs

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
        self._reconstruct_packed_graph_inputs(kwargs)
        kwargs = {k: v for k, v in kwargs.items() if v is not None or k == "attention_mask"}

        context = None
        if (
            not self.config.cuda_graph_modules
            or CudaGraphModule.attn in self.config.cuda_graph_modules
        ):
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

        packed = kwargs.get("packed_seq_params")
        kwargs = self._decompose_packed_graph_inputs(kwargs)

        assert (kwargs.get('inference_context') is None) and (
            kwargs.get('packed_seq_params') is None
        ), (
            "CUDA graph accepts only Tensor inputs. "
            "inference_context and packed_seq_params are excluded from input list. "
            "For inference cuda graph, please use cuda_graph_impl=local instead."
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
            # The residual state follows the router/preprocess outputs.
            residual, mlp_state = self._unpack_mlp_cuda_graph_state(cuda_graph_output)
            mlp_output_with_bias = self.mlp._te_cuda_graph_resume(
                cuda_graph_output, padding_mask, packed_seq_params=packed_seq_params
            )
            if self.config.overlap_moe_expert_parallel_comm:
                return (residual, *mlp_output_with_bias)

            # If we early returned, layernorm recompute hooks were attached to the output buffer
            # of the cudagraph, so disable the recompute hooks inside _apply_mlp_bda_step
            recompute_pre_mlp_layernorm = self.recompute_pre_mlp_layernorm
            self.recompute_pre_mlp_layernorm = False
            output = self._apply_mlp_bda_step(mlp_output_with_bias, residual, mlp_state)
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


def _uses_mhc_attention_cuda_graph_split(config: TransformerConfig) -> bool:
    """Whether mHC attention graphs capture only the input layernorm and self-attention.

    ``TransformerConfig`` validates the remaining requirements of
    ``mhc_recompute_attn_cuda_graph_split``.
    """
    return (
        config.mhc_recompute_attn_cuda_graph_split
        and config.cuda_graph_impl == "transformer_engine"
    )


def _validate_mhc_te_graph_inputs(
    kwargs: dict, *, attention_graph_split: bool = False, packed_sequences: bool = False
) -> None:
    """Reject eager-only inputs that have no static TE training sample.

    The attention-only graph split keeps the padding mask and the mHC recompute manager
    outside the graph, where its eager MLP and mHC steps consume them. With scheduler-packed
    sequences, the split graph also takes their metadata as fixed-capacity offsets.
    """
    unsupported = ["packed_sequence_cp_metadata", "inference_context"]
    if not (attention_graph_split and packed_sequences):
        unsupported.append("packed_seq_params")
    if not attention_graph_split:
        unsupported.append("padding_mask")
    for name in unsupported:
        if kwargs.get(name) is not None:
            raise NotImplementedError(
                f"mHC TE graphs do not support {name} here. Packed THD sequences require "
                "mhc_recompute_attn_cuda_graph_split with sequence_packing_scheduler, and "
                "padding_mask requires the attention graph split."
            )
    if attention_graph_split:
        return
    if kwargs.get("mhc_recompute_manager") is not None:
        raise NotImplementedError(
            "mHC TE graphs support mHC selective recompute only with "
            "mhc_recompute_attn_cuda_graph_split."
        )
    kwargs.pop("mhc_recompute_manager", None)


class TECudaGraphHyperConnectionTransformerLayerMixin(TECudaGraphTransformerLayerMixin):
    """Adapt TE graph inputs, parameter hooks and graph scopes for hyper connection layers.

    The whole-layer and attention graphs start from the n-stream hidden states. With
    ``mhc_recompute_attn_cuda_graph_split``, the graph covers only the input layernorm and
    self-attention, and the mHC read, write and MLP run eagerly around it.
    """

    def register_te_cuda_graph(self):
        """Validate the mHC layer's TE capture scopes after the base layer is built."""
        if self.is_moe_layer and CudaGraphModule.moe in self.config.cuda_graph_modules:
            raise NotImplementedError(
                "mHC TE graphs support MoE router capture with optional preprocessing only."
            )
        if _uses_mhc_attention_cuda_graph_split(self.config) and not isinstance(
            self.cross_attention, IdentityOp
        ):
            raise NotImplementedError(
                "mhc_recompute_attn_cuda_graph_split does not support cross-attention."
            )
        return super().register_te_cuda_graph()

    def get_layer_static_inputs(self, seq_length, micro_batch_size):
        """Override to produce n-stream hidden_states of shape [s, b, n*C].

        CUDA graph capture creates static buffers whose shapes are determined by
        this method. The base class returns [s, b, C], but mHC layers operate on
        n-stream hidden states of shape [s, b, n*C]. The attention graph split
        starts from the aggregated [s, b, C] input instead.
        """
        static_inputs = super().get_layer_static_inputs(seq_length, micro_batch_size)
        if _uses_mhc_attention_cuda_graph_split(self.config):
            # The eager MLP consumes these; the attention graph is captured without them.
            static_inputs.pop("input_ids", None)
            static_inputs.pop("padding_mask", None)
            return static_inputs
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

        if not self.config.cuda_graph_modules or _uses_mhc_attention_cuda_graph_split(self.config):
            # The attention graph split runs both hyper connections eagerly.
            return submodules

        if (
            CudaGraphModule.attn in self.config.cuda_graph_modules
            and self.self_attention_hyper_connection is not None
        ):
            submodules.append(self.self_attention_hyper_connection)
        if self.mlp_hyper_connection is not None and (
            (CudaGraphModule.mlp in self.config.cuda_graph_modules and not self.is_moe_layer)
            or (self.is_moe_layer and CudaGraphModule.moe_router in self.config.cuda_graph_modules)
        ):
            submodules.append(self.mlp_hyper_connection)
        return submodules

    def _te_cuda_graph_capture_forward(self, hidden_states: Tensor, *args, **kwargs):
        """Capture the attention graph split, or the layer with mHC-supported inputs."""
        if _uses_mhc_attention_cuda_graph_split(self.config):
            self._reconstruct_packed_graph_inputs(kwargs)
            _validate_mhc_te_graph_inputs(
                kwargs,
                attention_graph_split=True,
                packed_sequences=self.config.sequence_packing_scheduler is not None,
            )
            return self._forward_mhc_attention_graph(hidden_states, *args, **kwargs)
        _validate_mhc_te_graph_inputs(kwargs)
        return super()._te_cuda_graph_capture_forward(hidden_states, *args, **kwargs)

    def _te_cuda_graph_prepare_replay(self, hidden_states: Tensor, *args, **kwargs):
        """Prepare the attention graph split, or the layer with mHC-supported inputs."""
        if _uses_mhc_attention_cuda_graph_split(self.config):
            _validate_mhc_te_graph_inputs(
                kwargs,
                attention_graph_split=True,
                packed_sequences=self.config.sequence_packing_scheduler is not None,
            )
            return self._prepare_mhc_attention_graph_split_replay(hidden_states, *args, **kwargs)
        _validate_mhc_te_graph_inputs(kwargs)
        return super()._te_cuda_graph_prepare_replay(hidden_states, *args, **kwargs)

    def _forward_mhc_attention_graph(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ) -> tuple[Tensor, ...]:
        """Captured body of the attention graph split: input layernorm and self-attention.

        Takes the aggregated single-stream input and returns the self-attention output, plus
        its bias when present. The mHC read and write stay outside the graph.
        """
        input_layernorm_output = apply_module(self.input_layernorm)(hidden_states)
        if isinstance(input_layernorm_output, tuple):
            raise ValueError(
                "input_layernorm cannot return a residual when a residual connection "
                "owns the carried state."
            )
        nvtx_range_push(suffix="self_attention")
        attention_output, attention_bias = self.self_attention(
            input_layernorm_output,
            attention_mask=attention_mask,
            rotary_pos_emb=rotary_pos_emb,
            packed_seq_params=packed_seq_params,
        )
        nvtx_range_pop(suffix="self_attention")
        if attention_bias is None:
            return (attention_output,)
        return attention_output, attention_bias

    def _prepare_mhc_attention_graph_split_replay(self, hidden_states: Tensor, *args, **kwargs):
        """Run the eager mHC read and bind the attention graph and its eager continuation."""
        from megatron.core.transformer.hyper_connection import HyperConnectionExecutionContext

        if args:
            raise TypeError(
                "mhc_recompute_attn_cuda_graph_split takes the layer inputs after "
                "hidden_states as keyword arguments."
            )
        if kwargs.get("context") is not None:
            raise NotImplementedError(
                "mhc_recompute_attn_cuda_graph_split does not support cross-attention."
            )
        # The eager MLP consumes these; the attention graph was captured without them, and
        # takes scheduler-packed metadata as fixed-capacity offsets instead.
        padding_mask = kwargs.pop("padding_mask", None)
        input_ids = kwargs.pop("input_ids", None)
        packed_seq_params = kwargs.get("packed_seq_params")
        kwargs = self._decompose_packed_graph_inputs(kwargs)

        nvtx_range_push(suffix="self_attention_hyper_connection")
        # TE copies the aggregate into the graph's static input and keeps that copy for the
        # graph's backward, so the read is eager rather than checkpointed.
        aggregated, attn_state = apply_module(self.residual_read_self_attn)(
            hidden_states,
            execution_context=HyperConnectionExecutionContext(self.self_attention_hyper_connection),
        )
        nvtx_range_pop(suffix="self_attention_hyper_connection")
        return (
            partial(self._te_cuda_graph_replay_graph, aggregated, **kwargs),
            partial(
                self._finish_mhc_attention_graph_split_replay,
                attn_state=attn_state,
                padding_mask=padding_mask,
                input_ids=input_ids,
                packed_seq_params=packed_seq_params,
            ),
        )

    def _finish_mhc_attention_graph_split_replay(
        self,
        graph_outputs,
        *,
        attn_state,
        padding_mask: Optional[Tensor] = None,
        input_ids: Optional[Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ):
        """Run the eager mHC write and MLP after the attention graph."""
        attention_output_with_bias = (
            graph_outputs[0],
            graph_outputs[1] if len(graph_outputs) > 1 else None,
        )
        nvtx_range_push(suffix="self_attn_bda")
        with self.bias_dropout_add_exec_handler():
            hidden_states = self._write_residual_connection(
                self.residual_write_self_attn,
                attention_output_with_bias,
                attn_state,
                is_mlp=False,
                recompute_context=None,
            )
        nvtx_range_pop(suffix="self_attn_bda")
        output = self._forward_mlp(
            hidden_states,
            padding_mask=padding_mask,
            input_ids=input_ids,
            packed_seq_params=packed_seq_params,
        )
        return output, None

    def _pack_mlp_cuda_graph_state(self, outputs, residual, mlp_state):
        # The read state already carries the residual, H_res and H_post.
        assert len(mlp_state) == 3, "mHC partial MoE graphs require the MLP read state."
        return list(outputs) + list(mlp_state)

    def _unpack_mlp_cuda_graph_state(self, outputs):
        h_post, h_res, residual = outputs.pop(), outputs.pop(), outputs.pop()
        return residual, (residual, h_res, h_post)
