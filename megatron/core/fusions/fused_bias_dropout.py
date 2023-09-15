# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from typing import Optional, Tuple

import torch


def _bias_dropout_add_func(x, bias, residual, prob, training):
    # type: (Tensor, Optional[Tensor], Tensor, float, bool) -> Tensor
    # NOTE: Previously, the argument `bias` used to be passed as
    # `bias.expand_as(residual)` when the `bias_dropout_func` is called from the
    # transformer layer but broadcasting should automatically take care of that.
    # Also, looking at broadcasting semantics, `expand_as` and broadcasting
    # seem to be identical performance-wise (both just change the view).

    # If we want to train mixed precision, then the output of this function
    # should be half precision. However, in AMP O1, the input (residual) is
    # in fp32, and it will up-cast the result to fp32, causing pipeline parallel
    # GPU communication to hang. Therefore, we need to cast residual to the same
    # dtype as x.
    residual = residual if residual.dtype == x.dtype else residual.to(x.dtype)
    if bias is not None:
        x = x + bias
    out = torch.nn.functional.dropout(x, p=prob, training=training)
    out = residual + out
    return out


@torch.jit.script
def bias_dropout_add_fused_train(
    x_with_bias: Tuple[torch.Tensor, Optional[torch.Tensor]], residual: torch.Tensor, prob: float,
) -> torch.Tensor:
    x, bias = x_with_bias  # unpack
    return _bias_dropout_add_func(x, bias, residual, prob, True)


@torch.jit.script
def bias_dropout_add_fused_inference(
    x_with_bias: Tuple[torch.Tensor, Optional[torch.Tensor]], residual: torch.Tensor, prob: float,
) -> torch.Tensor:
    x, bias = x_with_bias  # unpack
    return _bias_dropout_add_func(x, bias, residual, prob, False)


def get_bias_dropout_add(training, fused):
    def unfused_bias_dropout_add(x_with_bias, residual, prob):
        x, bias = x_with_bias  # unpack
        return _bias_dropout_add_func(x, bias, residual, prob, training)

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
