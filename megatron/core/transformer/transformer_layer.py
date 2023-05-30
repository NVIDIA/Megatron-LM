# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import torch

from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.enums import AttnType, AttnMaskType
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.mlp import MLP
from megatron.core.utils import make_viewless_tensor
from megatron.core.transformer.custom_layers.transformer_engine import TELayerNorm

class TransformerLayer(MegatronModule):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    def __init__(
        self, config: TransformerConfig, layer_number: int = 1, self_attn_mask_type=AttnMaskType.padding,
    ):
        super().__init__(config=config)
        self.config: TransformerConfig = config

        self.layer_number = layer_number
        self.self_attn_mask_type = self_attn_mask_type

        # Layernorm on the input data.
        # TODO: add pytorch only layernorm
        self.input_layernorm = TELayerNorm(
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
            persist_layer_norm=self.config.persist_layer_norm,
            sequence_parallel=self.config.sequence_parallel,
            zero_centered_gamma=self.config.layernorm_zero_centered_gamma,
        )

        # Self attention.
        self.self_attention = SelfAttention(
            config=self.config,
            layer_number=layer_number,
            attn_mask_type=self_attn_mask_type,
        )

        # Layernorm on the attention output
        self.post_self_attn_layernorm = TELayerNorm(
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
            persist_layer_norm=self.config.persist_layer_norm,
            sequence_parallel=self.config.sequence_parallel,
            zero_centered_gamma=self.config.layernorm_zero_centered_gamma,
        )

        # MLP
        self.mlp = MLP(config=self.config)

        # @jcasper how should we handle nvfuser?
        # Set bias+dropout+add fusion grad_enable execution handler.
        # TORCH_MAJOR = int(torch.__version__.split('.')[0])
        # TORCH_MINOR = int(torch.__version__.split('.')[1])
        # use_nvfuser = TORCH_MAJOR > 1 or (TORCH_MAJOR == 1 and TORCH_MINOR >= 10)
        # self.bias_dropout_add_exec_handler = nullcontext if use_nvfuser else torch.enable_grad
        self.bias_dropout_add_exec_handler = torch.enable_grad

        self.bias_dropout_add_func = get_bias_dropout_add(
            self.training,
            self.config.bias_dropout_fusion
        )

    # TODO: decide how to do inference_params
    def forward(
        self, hidden_states, attention_mask, encoder_output=None, enc_dec_attn_mask=None, inference_params=None
    ):
        # hidden_states: [s, b, h]

        # Layer norm at the beginning of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)
        # Self attention.
        attention_output_with_bias = self.self_attention(
            layernorm_output, attention_mask, inference_params=inference_params
        )

        # Residual connection.
        if self.config.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        # bias_dropout_add fusion returning fp32 instead of bf16
        with self.bias_dropout_add_exec_handler():
            layernorm_input = self.bias_dropout_add_func(
                attention_output_with_bias, residual, self.config.hidden_dropout
            )

        # Layer norm post the self attention.
        layernorm_output = self.post_self_attn_layernorm(layernorm_input)

        # MLP.
        mlp_output_with_bias = self.mlp(layernorm_output)

        # Second residual connection.
        if self.config.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        with self.bias_dropout_add_exec_handler():
            output = self.bias_dropout_add_func(
                mlp_output_with_bias, residual, self.config.hidden_dropout
            )

        # Jit compiled function creates 'view' tensor. This tensor
        # potentially gets saved in the MPU checkpoint function context,
        # which rejects view tensors. While making a viewless tensor here
        # won't result in memory savings (like the data loader, or
        # p2p_communication), it serves to document the origin of this
        # 'view' tensor.
        output = make_viewless_tensor(inp=output, requires_grad=output.requires_grad, keep_graph=True)

        return output
