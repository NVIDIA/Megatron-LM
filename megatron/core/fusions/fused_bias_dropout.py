# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import torch
from typing import Tuple, Optional

def _bias_dropout_add_func(x, bias, residual, prob, training):
    # type: (Tensor, Optional[Tensor], Tensor, float, bool) -> Tensor
    # NOTE: Previously, the argument `bias` used to be passed as
    # `bias.expand_as(residual)` when the `bias_dropout_func` is called from the
    # transformer layer but broadcasting should automatically take care of that.
    # Also, looking at broadcasting semantics, `expand_as` and broadcasting
    # seem to be identical performance-wise (both just change the view).
    if bias is not None:
        x = x + bias
    out = torch.nn.functional.dropout(x, p=prob, training=training)
    out = residual + out
    return out

def get_bias_dropout_add(training, fused):

    def unfused_bias_dropout_add(x_with_bias, residual, prob):
        x, bias = x_with_bias # unpack
        return _bias_dropout_add_func(x, bias, residual, prob, training)

    @torch.jit.script
    def bias_dropout_add_fused_train(
        x_with_bias: Tuple[torch.Tensor, Optional[torch.Tensor]],
        residual: torch.Tensor,
        prob: float
    ) -> torch.Tensor:
        x, bias = x_with_bias # unpack
        return _bias_dropout_add_func(x, bias, residual, prob, True)

    @torch.jit.script
    def bias_dropout_add_fused_inference(
        x_with_bias: Tuple[torch.Tensor, Optional[torch.Tensor]],
        residual: torch.Tensor,
        prob: float
    ) -> torch.Tensor:
        x, bias = x_with_bias # unpack
        return _bias_dropout_add_func(x, bias, residual, prob, False)

    if fused:
        # jit scripting for a nn.module (with dropout) is not
        # triggering the fusion kernel. For now, we use two
        # different nn.functional routines to account for varying
        # dropout semantics during training and inference phases.
        if training:
            return bias_dropout_add_fused_train
        else:
            return bias_dropout_add_fused_inference
    else:
        return unfused_bias_dropout_add
