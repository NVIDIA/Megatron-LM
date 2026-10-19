# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from dataclasses import replace
from typing import Optional, Tuple

from torch import Tensor

from megatron.core.inference.utils import InferenceMode
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.hyper_connection import HyperConnectionModule
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.layer_interface import (
    LayerAuxInputs,
    LayerAuxOutputs,
    bind_layer_interface,
)
from megatron.core.transformer.module import (
    MegatronModule,
    convert_module_to_dtype_except_fp32_marked,
)
from megatron.core.transformer.transformer_layer import TransformerLayer


class HyperConnectionHybridLayer(MegatronModule):
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
        self.hyper_connection = HyperConnectionModule(config=config, layer_number=self.layer_number)
        if config.params_dtype is not None:
            convert_module_to_dtype_except_fp32_marked(self.hyper_connection, config.params_dtype)
        if hasattr(layer, 'tp_group'):
            self.tp_group = layer.tp_group

    def mamba_state_shapes_per_request(self) -> Optional[Tuple[Tuple[int], Tuple[int]]]:
        """Delegate Mamba inference state shape requests to the wrapped layer."""
        if hasattr(self.inner_layer, 'mamba_state_shapes_per_request'):
            return self.inner_layer.mamba_state_shapes_per_request()
        mixer = getattr(self.inner_layer, 'self_attention', None)
        if mixer is not None and hasattr(mixer, 'mamba_state_shapes_per_request'):
            return mixer.mamba_state_shapes_per_request()
        return None

    def _call_inner_transformer_layer_without_local_bda(
        self, hidden_states: Tensor, aux_inputs: LayerAuxInputs
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
                layer._forward_self_attention_output_with_bias(hidden_states, aux_inputs)
            )
            output_with_bias = layer._group_offload_output_with_bias(
                output_with_bias, attn_norm_manager, forced_released_tensors=[residual]
            )
            return output_with_bias, None, layer.hidden_dropout, layer.config.bias_dropout_fusion

        output_with_bias, residual = layer._forward_mlp_output_with_bias(hidden_states, aux_inputs)
        mhc_recompute_manager = aux_inputs.mhc_recompute_manager
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

    def layer_aux_inputs_forward(self, hidden_states, aux_inputs, *, _call=None):
        """Validate shared inputs and expose the raw context as named auxiliary outputs."""
        if aux_inputs.residual_stream_recompute_context is not None:
            raise TypeError(
                "HyperConnectionHybridLayer does not support residual_stream recomputation; "
                "select 'mhc' in recompute_modules and configure mhc_recompute_layer_num instead."
            )
        call = self._call_with_bound_args if _call is None else _call

        def call_with_aux_outputs(*args, **kwargs):
            output, context = call(*args, **kwargs)
            return output, LayerAuxOutputs(context=context)

        return super().layer_aux_inputs_forward(
            hidden_states, aux_inputs, _call=call_with_aux_outputs
        )

    @bind_layer_interface
    def forward(
        self, hidden_states: Tensor, aux_inputs: LayerAuxInputs
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Run the wrapped hybrid layer through one layer-boundary mHC update."""
        mhc_recompute_manager = aux_inputs.mhc_recompute_manager
        aggregated, h_res, h_post, residual = self.hyper_connection(
            hidden_states, mhc_recompute_manager=mhc_recompute_manager, return_residual=True
        )
        fast_path_result = self._call_inner_transformer_layer_without_local_bda(
            aggregated, aux_inputs
        )

        if fast_path_result is None:
            layer_output, aux_outputs = self.inner_layer(
                aggregated, replace(aux_inputs, mhc_recompute_manager=None)
            )
            assert aux_outputs.bias is None, "mHC requires a complete residual layer"
            context = aux_outputs.context
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
