# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
from megatron.core.export.trtllm.trtllm_layers import TRTLLMLayers

# Map the most common mcore layers to TRTLLM layers
# pylint: disable=line-too-long
DEFAULT_CONVERSION_DICT = {
    # INPUT
    'embedding.word_embeddings.weight': TRTLLMLayers.vocab_embedding,
    'embedding.position_embeddings.weight': TRTLLMLayers.position_embedding,
    # ATTENTION
    'decoder.layers.input_layernorm.weight': TRTLLMLayers.input_layernorm_weight,
    'decoder.layers.input_layernorm.bias': TRTLLMLayers.input_layernorm_bias,
    'decoder.layers.self_attention.linear_qkv.weight': TRTLLMLayers.attention_qkv_weight,
    'decoder.layers.self_attention.linear_qkv.bias': TRTLLMLayers.attention_qkv_bias,
    'decoder.layers.self_attention.linear_proj.weight': TRTLLMLayers.attention_dense_weight,
    'decoder.layers.self_attention.linear_proj.bias': TRTLLMLayers.attention_dense_bias,
    # MLP
    'decoder.layers.pre_mlp_layernorm.weight': TRTLLMLayers.post_layernorm_weight,
    'decoder.layers.pre_mlp_layernorm.bias': TRTLLMLayers.post_layernorm_bias,
    'decoder.layers.mlp.linear_fc1.weight': TRTLLMLayers.mlp_fc_weight,
    'decoder.layers.mlp.linear_fc1.bias': TRTLLMLayers.mlp_fc_bias,
    'decoder.layers.mlp.linear_fc2.weight': TRTLLMLayers.mlp_projection_weight,
    'decoder.layers.mlp.linear_fc2.bias': TRTLLMLayers.mlp_projection_bias,
    # EXPERTS
    'decoder.layers.mlp.experts.experts.linear_fc1.weight': TRTLLMLayers.mlp_fc_weight_mixture_of_experts,
    'decoder.layers.mlp.experts.experts.linear_fc2.weight': TRTLLMLayers.mlp_projection_weight_mixture_of_experts,
    'decoder.layers.mlp.router.weight': TRTLLMLayers.mlp_router_weight,
    # FINAL LAYER NORM
    'decoder.final_layernorm.weight': TRTLLMLayers.final_layernorm_weight,
    'decoder.final_layernorm.bias': TRTLLMLayers.final_layernorm_bias,
    # OUTPUT LAYER
    'output_layer.weight': TRTLLMLayers.lm_head,
    # TRANSFORMER ENGINE LAYER NORM
    # ATTENTION
    'decoder.layers.self_attention.linear_qkv.layer_norm_weight': TRTLLMLayers.input_layernorm_weight,
    'decoder.layers.self_attention.linear_qkv.layer_norm_bias': TRTLLMLayers.input_layernorm_bias,
    # MLP
    'decoder.layers.mlp.linear_fc1.layer_norm_weight': TRTLLMLayers.post_layernorm_weight,
    'decoder.layers.mlp.linear_fc1.layer_norm_bias': TRTLLMLayers.post_layernorm_bias,
}
