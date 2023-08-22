# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import torch

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.spec_utils import (
    TransformerLayerSpec, build_module
)
from megatron.core.utils import make_viewless_tensor


class TransformerLayer(MegatronModule):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    def __init__(
        self,
        config: TransformerConfig,
        spec: TransformerLayerSpec,
        layer_number: int = 1,
        self_attn_mask_type=AttnMaskType.padding,
    ):
        super().__init__(config=config)
        self.config: TransformerConfig = config

        self.layer_number = layer_number
        self.self_attn_mask_type = self_attn_mask_type

        ## [Module 1: Input Layernorm] Optional Layernorm on the input data
        # TODO: add pytorch only layernorm
        self.input_layernorm = build_module(
            spec.input_layernorm,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
            persist_layer_norm=self.config.persist_layer_norm,
            sequence_parallel=self.config.sequence_parallel,
            zero_centered_gamma=self.config.layernorm_zero_centered_gamma,
            normalization=self.config.normalization,
        )

        ## [Module 2: SelfAttention]
        self.self_attention = build_module(
            spec.self_attention,
            config=self.config,
            spec=spec.self_attention,
            layer_number=layer_number,
        )

        ## [Module 3: BiasDropoutFusion]
        self.self_attn_bda = build_module(spec.self_attn_bda)

        ## [Module 4: Post SelfAttention] Optional Layernorm after self-attn
        self.post_self_attn_layernorm = build_module(
            spec.post_self_attn_layernorm,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
            persist_layer_norm=self.config.persist_layer_norm,
            sequence_parallel=self.config.sequence_parallel,
            zero_centered_gamma=self.config.layernorm_zero_centered_gamma,
            normalization=self.config.normalization,
        )

        ## [Module 5: CrossAttention]
        self.cross_attention = build_module(
            spec.cross_attention,
            config=self.config,
            spec=spec.cross_attention,
            layer_number=layer_number,
        )

        ## [Module 6: BiasDropoutFusion]
        self.cross_attn_bda = build_module(spec.cross_attn_bda)

        ## [Module 7: Post Cross Attention] Optional Layernorm after cross-attn
        self.post_cross_attn_layernorm = build_module(
            spec.post_cross_attn_layernorm,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
            persist_layer_norm=self.config.persist_layer_norm,
            sequence_parallel=self.config.sequence_parallel,
            zero_centered_gamma=self.config.layernorm_zero_centered_gamma,
            normalization=self.config.normalization,
        )

        ## [Module 8: MLP block]
        self.ln_mlp = build_module(spec.ln_mlp, config=self.config)

        ## [Module 9: BiasDropoutFusion]
        self.mlp_bda = build_module(spec.mlp_bda)

        ## [Module 10: Post MLP] Optional Layernorm after MLP
        self.post_mlp_layernorm = build_module(
            spec.post_mlp_layernorm,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
            persist_layer_norm=self.config.persist_layer_norm,
            sequence_parallel=self.config.sequence_parallel,
            zero_centered_gamma=self.config.layernorm_zero_centered_gamma,
            normalization=self.config.normalization,
        )

        # @jcasper how should we handle nvfuser?
        # Set bias+dropout+add fusion grad_enable execution handler.
        # TORCH_MAJOR = int(torch.__version__.split('.')[0])
        # TORCH_MINOR = int(torch.__version__.split('.')[1])
        # use_nvfuser = TORCH_MAJOR > 1 or (TORCH_MAJOR == 1 and TORCH_MINOR >= 10)
        # self.bias_dropout_add_exec_handler = nullcontext if use_nvfuser else torch.enable_grad
        self.bias_dropout_add_exec_handler = torch.enable_grad

        self.bias_dropout_add_func = get_bias_dropout_add(
            self.training, self.config.bias_dropout_fusion
        )


    # TODO: decide how to do inference_params
    def forward(
        self,
        hidden_states,
        attention_mask,
        context=None,
        context_mask=None,
        inference_params=None,
        rotary_pos_emb=None,
    ):
        # hidden_states: [s, b, h]

        # Optional Input Layer norm
        input_layernorm_output = self.input_layernorm(hidden_states)

        # Residual connection.
        residual = input_layernorm_output

        # Self attention.
        attention_output_with_bias = self.self_attention(
            input_layernorm_output,
            attention_mask=attention_mask,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
        )

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.self_attn_bda(
                self.training, self.config.bias_dropout_fusion
            )(attention_output_with_bias, residual, self.config.hidden_dropout)

        # Optional Layer norm after self-attention
        post_self_attn_layernorm_output = self.post_self_attn_layernorm(hidden_states)

        # Residual connection.
        residual = post_self_attn_layernorm_output

        # Cross attention.
        attention_output_with_bias = self.cross_attention(
            post_self_attn_layernorm_output,
            attention_mask=attention_mask,
            context=context,
            inference_params=inference_params,
        )

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.cross_attn_bda(
                self.training, self.config.bias_dropout_fusion
            )(attention_output_with_bias, residual, self.config.hidden_dropout)

        # Optional Layer norm post the cross-attention.
        post_cross_attn_layernorm_output = self.post_cross_attn_layernorm(hidden_states)

        # Residual connection.
        residual = post_cross_attn_layernorm_output

        # MLP.
        ln_mlp_output_with_bias = self.ln_mlp(post_cross_attn_layernorm_output)

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.mlp_bda(
                self.training, self.config.bias_dropout_fusion
            )(ln_mlp_output_with_bias, residual, self.config.hidden_dropout)

        # Optional Layer norm post MLP
        output = self.post_mlp_layernorm(hidden_states)

        # Jit compiled function creates 'view' tensor. This tensor
        # potentially gets saved in the MPU checkpoint function context,
        # which rejects view tensors. While making a viewless tensor here
        # won't result in memory savings (like the data loader, or
        # p2p_communication), it serves to document the origin of this
        # 'view' tensor.
        output = make_viewless_tensor(
            inp=output, requires_grad=output.requires_grad, keep_graph=True
        )

        return output
