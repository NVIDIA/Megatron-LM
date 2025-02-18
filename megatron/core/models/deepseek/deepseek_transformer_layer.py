"""tranformer layer for aiak."""

from dataclasses import dataclass
from typing import Union, Literal, Optional

import torch
from torch import Tensor

from megatron.core import InferenceParams, tensor_parallel
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.utils import make_viewless_tensor
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.fusions.fused_cross_entropy import fused_vocab_parallel_cross_entropy
from megatron.core.transformer.transformer_config import TransformerConfig

@dataclass
class DeepSeekTransformerLayerSubmodules(TransformerLayerSubmodules):
    """
    DeepSeekTransformerLayerSubmodules:
    Args:
        dense_mlp (Union[ModuleSpec, type], optional): Specification or type of the dense MLP module.
            Defaults to IdentityOp.
        moe_mlp (Union[ModuleSpec, type], optional): Specification or type of the MLP module for MoE.
            Defaults to IdentityOp.
    """
    dense_mlp: Union[ModuleSpec, type] = IdentityOp
    moe_mlp: Union[ModuleSpec, type] = IdentityOp

class DeepSeekTransformerLayer(TransformerLayer):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: DeepSeekTransformerLayerSubmodules,
        layer_number: int = 1,
        hidden_dropout: float = None,
        **kwargs,
    ):
        """
            Initializes the DeepSeekTransformerLayerSubmodules module.
        
        Args:
            config (TransformerConfig): The configuration of the model.
            submodules (DeepSeekTransformerLayerSubmodules): The submodules of the model.
            layer_number (int, optional): The number of the current layer. Defaults to 1.
            hidden_dropout (float, optional): The dropout probability for the hidden state. Defaults to None.
            **kwargs (dict, optional): Additional keyword arguments passed to the parent class. Defaults to {}.
        
        Raises:
            ValueError: If the `submodules` argument is not a valid type.
        """
        super(DeepSeekTransformerLayer, self).__init__(
            config=config, submodules=submodules, layer_number=layer_number, hidden_dropout=hidden_dropout, **kwargs)
        
        # Parse config.moe_layer_freq to determine the pattern of expert/dense layers.
        # 0 stands for dense layers, 1 stands for expert layers.
        # For integer N: Creates a pattern with one expert layer every N layers.
        # For string pattern: Evaluates the str directly (e.g. "[1,0,1]" for alternating expert/dense).
        if (isinstance(config.moe_layer_freq, int) and self.layer_number % config.moe_layer_freq == 0) or
           (isinstance(config.moe_layer_freq, list) and config.moe_layer_freq[self.layer_number]):
           self.mlp = build_module(submodules.moe_mlp, config=self.config)
        else:
            self.mlp = build_module(submodules.dense_mlp, config=self.config)

        if hasattr(self.mlp, 'set_layer_number'):
            self.mlp.set_layer_number(self.layer_number)

    def forward(
        self,
        hidden_states,
        attention_mask=non_persistent_ckpt_type,
        attn_mask_type=None,
        context=None,
        context_mask=None,
        rotary_pos_emb=None,
        inference_params=None,
        packed_seq_params=None,
    ):
        """
        forward method
        """
        # hidden_states: [s, b, h]

        # Residual connection.
        residual = hidden_states

        # Optional Input Layer norm
        input_layernorm_output = self.input_layernorm(hidden_states)

        # Self attention.
        attention_output_with_bias = self.self_attention(
            input_layernorm_output,
            attention_mask=attention_mask,
            inference_params=inference_params,
            packed_seq_params=packed_seq_params,
        )

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.self_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias, residual, self.hidden_dropout
            )

        # Residual connection.
        residual = hidden_states

        # Optional Layer norm after self-attention
        pre_cross_attn_layernorm_output = self.pre_cross_attn_layernorm(hidden_states)

        # Cross attention.
        attention_output_with_bias = self.cross_attention(
            pre_cross_attn_layernorm_output,
            attention_mask=context_mask,
            attn_mask_type=attn_mask_type,
            key_value_states=context,
            inference_params=inference_params,
        )

        if isinstance(attention_output_with_bias, dict) and "context" in attention_output_with_bias:
            context = attention_output_with_bias["context"]

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.cross_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias, residual, self.hidden_dropout
            )

        # Residual connection.
        residual = hidden_states

        # Optional Layer norm post the cross-attention.
        pre_mlp_layernorm_output = self.pre_mlp_layernorm(hidden_states)

        # MLP.
        mlp_output_with_bias = self.mlp(pre_mlp_layernorm_output)

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.mlp_bda(self.training, self.config.bias_dropout_fusion)(
                mlp_output_with_bias, residual, self.hidden_dropout
            )

        # Jit compiled function creates 'view' tensor. This tensor
        # potentially gets saved in the MPU checkpoint function context,
        # which rejects view tensors. While making a viewless tensor here
        # won't result in memory savings (like the data loader, or
        # p2p_communication), it serves to document the origin of this
        # 'view' tensor.
        output = make_viewless_tensor(
            inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
        )

        # CUDA graph requires returned values to be Tensors
        if self.config.external_cuda_graph and self.training:
            return output
        return output, context
