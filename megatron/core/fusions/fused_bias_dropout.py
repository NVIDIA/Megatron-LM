# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
from typing import Optional, Tuple

import torch

from megatron.core.jit import jit_fuser

# pylint: disable=missing-function-docstring


def _bias_dropout_add_func(x_with_bias, residual, prob, training):
    # type: (Tuple[Tensor, Optional[Tensor]], Tensor, float, bool) -> Tensor
    # NOTE: Previously, the argument `bias` used to be passed as
    # `bias.expand_as(residual)` when the `bias_dropout_func` is called from the
    # transformer layer but broadcasting should automatically take care of that.
    # Also, looking at broadcasting semantics, `expand_as` and broadcasting
    # seem to be identical performance-wise (both just change the view).

    x, bias = x_with_bias  # unpack

    # Run in-place if in eval mode and inputs do not require gradients
    inplace = (
        not training
        and not x.requires_grad
        and not residual.requires_grad
        and (bias is None or not bias.requires_grad)
    )

    # If we want to train mixed precision, then the output of this function
    # should be half precision. However, in AMP O1, the input (residual) is
    # in fp32, and it will up-cast the result to fp32, causing pipeline parallel
    # GPU communication to hang. Therefore, we need to cast residual to the same
    # dtype as x.
    residual = residual if residual.dtype == x.dtype else residual.to(x.dtype)

    # The Dropout operation, Residual Addition and the tensor returning can be
    # done generically outside the if statement, but that stops fusing of Bias
    # Addition-Dropout-Residual Addition operation. So doing it together inside
    # the conditional branch to improve performance
    if bias is not None:
        if inplace:
            x.add_(bias)
        else:
            x = x + bias
        out = torch.nn.functional.dropout(x, p=prob, training=training, inplace=inplace)
        if inplace:
            out.add_(residual)
        else:
            out = residual + out
        return out
    else:
        out = torch.nn.functional.dropout(x, p=prob, training=training, inplace=inplace)
        if inplace:
            out.add_(residual)
        else:
            out = residual + out
        return out


def bias_dropout_add_unfused(training):
    def _bias_dropout_add(x_with_bias, residual, prob):
        return _bias_dropout_add_func(x_with_bias, residual, prob, training)

    return _bias_dropout_add


@jit_fuser
def bias_dropout_add_fused_train(
    x_with_bias: Tuple[torch.Tensor, Optional[torch.Tensor]], residual: torch.Tensor, prob: float
) -> torch.Tensor:
    return _bias_dropout_add_func(x_with_bias, residual, prob, True)


@jit_fuser
def bias_dropout_add_fused_inference(
    x_with_bias: Tuple[torch.Tensor, Optional[torch.Tensor]], residual: torch.Tensor, prob: float
) -> torch.Tensor:
    return _bias_dropout_add_func(x_with_bias, residual, prob, False)


def get_bias_dropout_add(training, fused):
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
        return bias_dropout_add_unfused(training)
