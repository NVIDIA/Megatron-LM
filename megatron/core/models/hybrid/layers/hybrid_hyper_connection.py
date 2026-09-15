# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from typing import Optional, Tuple

import torch
from torch import Tensor

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
from megatron.core.transformer.transformer_layer import (
    TransformerLayer,
    _validate_mhc_te_graph_inputs,
)


class HyperConnectionHybridLayer(GraphableMegatronModule):
    """Layer-boundary mHC wrapper for HybridStack layers.

    Hybrid layers already own their local residual paths. Each wrapped layer is
    treated as one function by aggregating n streams to its input, running the
    existing layer, and feeding only the layer delta back through mHC expansion.

    This wrapper nests the inner layer under inner_layer. Checkpoints cannot
    switch between mHC-enabled and ordinary HybridStacks without key migration.
    """

    def __init__(self, config: TransformerConfig, layer: MegatronModule) -> None:
        super().__init__(config=config)
        self.inner_layer = layer
        self.layer_number = layer.layer_number
        if config.cuda_graph_impl == 'transformer_engine' and self._is_partial_moe_graph():
            if not isinstance(layer.self_attention, IdentityOp) or not isinstance(
                layer.cross_attention, IdentityOp
            ):
                raise NotImplementedError(
                    'Hybrid mHC partial MoE graphs require a MoE-only inner layer. '
                    'Use separate attention and MoE layers in the Hybrid pattern.'
                )
        self.hyper_connection = HyperConnectionModule(config=config, layer_number=self.layer_number)
        if config.params_dtype is not None:
            convert_module_to_dtype_except_fp32_marked(self.hyper_connection, config.params_dtype)
        if hasattr(layer, 'tp_group'):
            self.tp_group = layer.tp_group

    def create_mcore_cudagraph_manager(self, config: TransformerConfig) -> None:
        """Keep the existing local-graph ownership on the inner layer.

        This wrapper owns TE training graphs only. In particular, inheriting the
        graphable protocol must not create a second local manager around the
        inner layer's existing manager.
        """

    def get_layer_static_inputs(self, seq_length: int, micro_batch_size: int) -> dict:
        """Build static inputs with the wrapper's n-stream residual width."""
        if hasattr(self.inner_layer, 'get_layer_static_inputs'):
            inputs = self.inner_layer.get_layer_static_inputs(seq_length, micro_batch_size)
        else:
            inputs = super().get_layer_static_inputs(seq_length, micro_batch_size)
        hidden = inputs['hidden_states']
        inputs['hidden_states'] = torch.ones(
            (*hidden.shape[:-1], self.config.mhc_num_residual_streams * self.config.hidden_size),
            dtype=hidden.dtype,
            device=hidden.device,
            requires_grad=hidden.requires_grad,
        )
        return inputs

    def _is_partial_moe_graph(self) -> bool:
        return (
            isinstance(self.inner_layer, TransformerLayer)
            and self.inner_layer.is_moe_layer
            and CudaGraphModule.moe_router in self.config.cuda_graph_modules
        )

    def _te_cuda_graph_capture(self, *args, **kwargs):
        """Capture the wrapper, or its deterministic MoE prefix, as tensor outputs."""
        _validate_mhc_te_graph_inputs(kwargs)
        if self._is_partial_moe_graph():
            hidden_states = args[0] if args else kwargs.pop('hidden_states')
            aggregated, h_res, h_post, residual = self.hyper_connection(
                hidden_states, return_residual=True
            )
            inner_kwargs = dict(kwargs)
            inner_kwargs.pop('hidden_states', None)
            outputs = self.inner_layer._te_cuda_graph_capture(aggregated, **inner_kwargs)
            # The residual must cross the graph boundary as an output so that
            # its gradient participates in the captured backward graph.
            return (*outputs, h_res, h_post, residual)
        hidden_states, context = self.forward(*args, **kwargs)
        if context is not None:
            raise NotImplementedError('Hybrid mHC TE graphs do not support cross-attention.')
        return (hidden_states,)

    def _te_cuda_graph_replay(self, *args, **kwargs):
        """Replay captured work and restore the normal wrapper return contract."""
        _validate_mhc_te_graph_inputs(kwargs)
        outputs = list(super()._te_cuda_graph_replay(*args, **kwargs))
        if not self._is_partial_moe_graph():
            return outputs[0], None
        residual, h_post, h_res = outputs.pop(), outputs.pop(), outputs.pop()
        # The ordinary inner layer's single-stream residual is not the mHC
        # residual; its raw expert output feeds the wrapper's BDA instead.
        self.inner_layer._unpack_mlp_cuda_graph_state(outputs)
        output_with_bias = self.inner_layer._resume_moe_experts_after_partial_cudagraph(outputs)
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

    def _get_te_cuda_graph_replay_args(self, *args, **kwargs):
        if isinstance(self.inner_layer, TransformerLayer):
            # Preserve TransformerLayer's None-mask normalization and TE version
            # handling, but select the graph/microbatch on this outer wrapper.
            graph_args, graph_kwargs = self.inner_layer._get_te_cuda_graph_replay_args(
                *args, **kwargs
            )
            graph_kwargs['is_first_microbatch'] = getattr(self, 'current_microbatch', 0) == 0
            return graph_args, graph_kwargs
        return super()._get_te_cuda_graph_replay_args(*args, **kwargs)

    def _get_submodules_under_cudagraphs(self):
        if self._is_partial_moe_graph():
            return [self.hyper_connection, *self.inner_layer._get_submodules_under_cudagraphs()]
        return [self]

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
    ) -> Tuple[Tensor, Optional[Tensor]]:
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
        )
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
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Run the wrapped hybrid layer through one layer-boundary mHC update."""
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
            )
            if self.config.fp32_residual_connection and aggregated.dtype != layer_output.dtype:
                aggregated = aggregated.to(layer_output.dtype)
            layer_output_with_bias = (layer_output - aggregated, None)
            dropout_prob = 0.0
            bias_dropout_fusion = False
        else:
            layer_output_with_bias, context, dropout_prob, bias_dropout_fusion = fast_path_result

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
