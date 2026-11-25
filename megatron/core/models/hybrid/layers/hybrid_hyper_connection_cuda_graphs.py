# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Transformer Engine CUDA graphs for the HybridStack mHC wrapper."""

from functools import partial

import torch
from torch import Tensor

from megatron.core.extensions.te_cuda_graphs import TECudaGraphMixin
from megatron.core.models.hybrid.layers import utils as layer_utils
from megatron.core.transformer.enums import CudaGraphModule
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.transformer.transformer_layer_mixins import (
    _uses_mhc_attention_cuda_graph_split,
    _validate_mhc_te_graph_inputs,
)


class TECudaGraphHyperConnectionHybridLayerMixin(TECudaGraphMixin):
    """Capture the wrapper's TE scopes and replay them on the current microbatch."""

    def _validate_te_graph_inner_layer(self) -> None:
        """Check supported scopes after the wrapper has installed its inner layer."""
        layer = self.inner_layer
        if self._is_attention_graph_split() and not (
            isinstance(layer.mlp, IdentityOp) and isinstance(layer.cross_attention, IdentityOp)
        ):
            raise NotImplementedError(
                "mhc_recompute_attn_cuda_graph_split requires attention-only inner layers. "
                "Use separate attention and MLP layers in the Hybrid pattern."
            )
        if self._is_partial_moe_graph():
            if not isinstance(layer.self_attention, IdentityOp) or not isinstance(
                layer.cross_attention, IdentityOp
            ):
                raise NotImplementedError(
                    "Hybrid mHC partial MoE graphs require a MoE-only inner layer. "
                    "Use separate attention and MoE layers in the Hybrid pattern."
                )

    def get_layer_static_inputs(self, seq_length: int, micro_batch_size: int) -> dict:
        """Build static inputs with the wrapper's n-stream residual width.

        The attention graph split starts from the aggregated single-stream input.
        """
        if hasattr(self.inner_layer, "get_layer_static_inputs"):
            inputs = self.inner_layer.get_layer_static_inputs(seq_length, micro_batch_size)
        else:
            inputs = super().get_layer_static_inputs(seq_length, micro_batch_size)
        if self._is_attention_graph_split():
            # The eager MLP consumes these; the attention graph is captured without them.
            inputs.pop("input_ids", None)
            inputs.pop("padding_mask", None)
            return inputs
        hidden = inputs["hidden_states"]
        inputs["hidden_states"] = torch.ones(
            (*hidden.shape[:-1], self.config.mhc_num_residual_streams * self.config.hidden_size),
            dtype=hidden.dtype,
            device=hidden.device,
            requires_grad=hidden.requires_grad,
        )
        return inputs

    def _has_custom_inner_layer(self) -> bool:
        """Whether the inner layer is a custom layer type, which owns its forward.

        The wrapper's TE graph paths call TransformerLayer internals and would bypass a
        custom forward, so custom layers stay eager.
        """
        return (
            layer_utils.get_custom_layer_type(getattr(self.inner_layer, "config", None)) is not None
        )

    def is_graphable_for_scope(self, modules):
        """Graph the wrapper when its inner layer covers one of the requested scopes."""
        # Custom layer types run eagerly under mHC, whatever the scopes.
        if self._has_custom_inner_layer():
            return False
        # If cuda_graph_modules is not set, every layer is graphed.
        if not modules:
            return True
        # The wrapper owns capture, but the concrete inner layer determines
        # which configured scopes apply. A MoE-only wrapper has no attention
        # work to graph when only attn is selected.
        return isinstance(self.inner_layer, TECudaGraphMixin) and (
            self.inner_layer.is_graphable_for_scope(modules)
        )

    def _is_attention_graph_split(self) -> bool:
        """Whether this wrapper captures only its inner input layernorm and self-attention."""
        return (
            _uses_mhc_attention_cuda_graph_split(self.config)
            and not self._has_custom_inner_layer()
            and isinstance(self.inner_layer, TransformerLayer)
            and not isinstance(self.inner_layer.self_attention, IdentityOp)
        )

    def _packed_sequences(self) -> bool:
        return self.config.sequence_packing_scheduler is not None

    def _is_partial_moe_graph(self) -> bool:
        return (
            not self._has_custom_inner_layer()
            and isinstance(self.inner_layer, TransformerLayer)
            and self.inner_layer.is_moe_layer
            and CudaGraphModule.moe_router in self.config.cuda_graph_modules
        )

    def _te_cuda_graph_capture_forward(self, hidden_states: Tensor, *args, **kwargs):
        """Capture the wrapper, or its deterministic MoE prefix, as tensor outputs."""
        if self._is_attention_graph_split():
            self.inner_layer._reconstruct_packed_graph_inputs(kwargs)
            _validate_mhc_te_graph_inputs(
                kwargs, attention_graph_split=True, packed_sequences=self._packed_sequences()
            )
            # The graph starts from the aggregate and ends before the mHC bias-dropout-add.
            (output, bias), _, _ = self.inner_layer._forward_self_attention_output_with_bias(
                hidden_states,
                attention_mask=kwargs.get("attention_mask"),
                rotary_pos_emb=kwargs.get("rotary_pos_emb"),
                packed_seq_params=kwargs.get("packed_seq_params"),
            )
            return (output,) if bias is None else (output, bias)
        _validate_mhc_te_graph_inputs(kwargs)
        if self._is_partial_moe_graph():
            aggregated, h_res, h_post, residual = self.hyper_connection(
                hidden_states, return_residual=True
            )
            outputs = self.inner_layer._te_cuda_graph_capture_forward(aggregated, *args, **kwargs)
            # The residual must cross the graph boundary as an output so that
            # its gradient participates in the captured backward graph.
            return (*outputs, h_res, h_post, residual)
        hidden_states, context = self.forward(hidden_states, *args, **kwargs)
        if context is not None:
            raise NotImplementedError("Hybrid mHC TE graphs do not support cross-attention.")
        return (hidden_states,)

    def _te_cuda_graph_prepare_replay(self, hidden_states: Tensor, *args, **kwargs):
        """Bind the captured work and the eager steps that restore the wrapper's return."""
        if self._is_attention_graph_split():
            return self._prepare_attention_graph_split_replay(hidden_states, *args, **kwargs)
        _validate_mhc_te_graph_inputs(kwargs)
        return (
            partial(self._te_cuda_graph_replay_graph, hidden_states, *args, **kwargs),
            self._te_cuda_graph_finish_replay,
        )

    def _te_cuda_graph_finish_replay(self, outputs):
        """Restore the normal wrapper return contract from the graph outputs."""
        outputs = list(outputs)
        if not self._is_partial_moe_graph():
            return outputs[0], None
        residual, h_post, h_res = outputs.pop(), outputs.pop(), outputs.pop()
        # The ordinary inner layer's single-stream residual is not the mHC
        # residual; its raw expert output feeds the wrapper's BDA instead.
        self.inner_layer._unpack_mlp_cuda_graph_state(outputs)
        # The non-split TE graph rejects padding_mask, so the eager experts run unmasked.
        output_with_bias = self.inner_layer.mlp._te_cuda_graph_resume(outputs, None)
        hidden_states = self.hyper_connection.fused_h_res_h_post_bda(
            h_res,
            residual,
            h_post,
            output_with_bias,
            dropout_prob=self.inner_layer.hidden_dropout,
            training=self.training,
            fused=self.inner_layer.config.bias_dropout_fusion,
        )
        return hidden_states, None

    def _prepare_attention_graph_split_replay(self, hidden_states: Tensor, *args, **kwargs):
        """Run the eager mHC aggregate and bind the attention graph and eager mHC BDA."""
        if args:
            raise TypeError(
                "mhc_recompute_attn_cuda_graph_split takes the layer inputs after "
                "hidden_states as keyword arguments."
            )
        mhc_recompute_manager = kwargs.pop("mhc_recompute_manager", None)
        _validate_mhc_te_graph_inputs(
            kwargs, attention_graph_split=True, packed_sequences=self._packed_sequences()
        )
        # The attention-only inner layer has no MLP to consume these.
        kwargs.pop("padding_mask", None)
        kwargs.pop("input_ids", None)
        # The graph takes scheduler-packed metadata as fixed-capacity offsets.
        kwargs = self.inner_layer._decompose_packed_graph_inputs(kwargs)
        # TE copies the aggregate into the graph's static input and keeps that copy for the
        # graph's backward, so the aggregate is not checkpointed.
        aggregated, h_res, h_post, residual = self.hyper_connection(
            hidden_states, return_residual=True
        )
        return (
            partial(self._te_cuda_graph_replay_graph, aggregated, **kwargs),
            partial(
                self._finish_attention_graph_split_replay,
                h_res=h_res,
                residual=residual,
                h_post=h_post,
                mhc_recompute_manager=mhc_recompute_manager,
            ),
        )

    def _finish_attention_graph_split_replay(
        self, outputs, *, h_res, residual, h_post, mhc_recompute_manager
    ):
        """Run the eager mHC bias-dropout-add after the attention graph."""
        output_with_bias = (outputs[0], outputs[1] if len(outputs) > 1 else None)
        hidden_states = self._apply_mhc_post(
            h_res,
            residual,
            h_post,
            output_with_bias,
            dropout_prob=self.inner_layer.hidden_dropout,
            bias_dropout_fusion=self.inner_layer.config.bias_dropout_fusion,
            mhc_recompute_manager=mhc_recompute_manager,
        )
        return hidden_states, None

    def _get_te_cuda_graph_replay_args(self, *args, **kwargs):
        if isinstance(self.inner_layer, TransformerLayer):
            # Preserve TransformerLayer's None-mask normalization and TE version
            # handling, but select the graph/microbatch on this outer wrapper.
            graph_args, graph_kwargs = self.inner_layer._get_te_cuda_graph_replay_args(
                *args, **kwargs
            )
            graph_kwargs["is_first_microbatch"] = (
                getattr(self, "te_cuda_graph_current_microbatch", 0) == 0
            )
            return graph_args, graph_kwargs
        return super()._get_te_cuda_graph_replay_args(*args, **kwargs)

    def _get_submodules_under_cudagraphs(self):
        if self._is_attention_graph_split():
            # The hyper connection runs eagerly and triggers its own parameter hooks.
            return self.inner_layer._get_submodules_under_cudagraphs()
        if self._is_partial_moe_graph():
            return [self.hyper_connection, *self.inner_layer._get_submodules_under_cudagraphs()]
        return [self]
