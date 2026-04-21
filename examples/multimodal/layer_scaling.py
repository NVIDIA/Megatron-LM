# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
from functools import partial

import torch

from megatron.core.transformer.transformer_layer import TransformerLayer

def _bias_dropout_add_func_layer_scaling(ls, x_with_bias, residual, prob, training):
    x, bias = x_with_bias  # unpack
    residual = residual if residual.dtype == x.dtype else residual.to(x.dtype)
    if bias is not None:
        x = x + bias
        out = torch.nn.functional.dropout(x, p=prob, training=training)
        out = residual + out * ls
        return out
    else:
        out = torch.nn.functional.dropout(x, p=prob, training=training)
        out = residual + out * ls
        return out


def bias_dropout_add_unfused_layer_scaling(ls, training):
    """Bias-dropout-add as in Megatron but with added LayerScaling handling."""

    def _bias_dropout_add(x_with_bias, residual, prob):
        return _bias_dropout_add_func_layer_scaling(ls, x_with_bias, residual, prob, training)

    return _bias_dropout_add


def get_bias_dropout_add_layer_scaling(ls, training, fused):
    """Bias-dropout-add as in Megatron but with added LayerScaling handling."""
    assert not fused, "Fused bias-dropout-add not implemented for LayerScaling."
    return bias_dropout_add_unfused_layer_scaling(ls, training)


# Add LayerScaling to our default TransformerLayer.
class LayerScalingTransformerLayer(TransformerLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ls1 = torch.nn.Parameter(torch.ones(self.config.hidden_size))
        self.ls2 = torch.nn.Parameter(torch.ones(self.config.hidden_size))

        self.self_attn_bda = partial(self.self_attn_bda, self.ls1)
        self.mlp_bda = partial(self.mlp_bda, self.ls2)
