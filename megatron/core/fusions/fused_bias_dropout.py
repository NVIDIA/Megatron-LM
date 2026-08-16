# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
import math
from typing import Optional, Tuple

import torch
from torch import nn

from megatron.core.jit import jit_fuser

# pylint: disable=missing-function-docstring


def _logit(x: float, residual_eps: float) -> float:
    x = min(max(x, residual_eps), 1.0 - residual_eps)
    return math.log(x / (1.0 - x))


class ResidualForgetGate(nn.Module):
    """Produce a bounded learnable residual retention scalar."""

    def __init__(
        self,
        gamma_init: float = 0.999,
        max_forget: float = 0.02,
        residual_eps: float = 1.0e-6,
    ) -> None:
        super().__init__()
        if not (0.0 < residual_eps < 0.5):
            raise ValueError("residual_eps must be in (0, 0.5).")
        if not (0.0 < max_forget < 1.0):
            raise ValueError("max_forget must be in (0, 1).")
        min_gamma = 1.0 - max_forget
        if not (min_gamma < gamma_init < 1.0):
            raise ValueError("gamma_init must satisfy 1 - max_forget < gamma_init < 1.")

        self.max_forget = max_forget
        self.gamma_init = gamma_init
        self.residual_eps = residual_eps
        self.forget_logit = nn.Parameter(torch.empty((), dtype=torch.float32))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the learnable logit to the configured retention value."""
        forget_ratio = (1.0 - self.gamma_init) / self.max_forget
        nn.init.constant_(self.forget_logit, -_logit(forget_ratio, self.residual_eps))

    def gamma(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Return the bounded residual retention scalar."""
        forget_rate = self.max_forget * torch.sigmoid(-self.forget_logit)
        return (1.0 - forget_rate).to(dtype)

    def forward(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Return the residual retention scalar."""
        return self.gamma(dtype)


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

    # For fp32 residual connections: upcast x (and bias) to residual's dtype so that
    # the addition and output remain in fp32, preserving numerical precision in the
    # residual stream across layers. When fp32_residual_connection is enabled,
    # pipeline parallel communication dtype should be set to fp32 accordingly.
    if x.dtype != residual.dtype:
        x = x.to(residual.dtype)
        if bias is not None:
            bias = bias.to(residual.dtype)

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


def _scaled_bias_dropout_add_func(
    x_with_bias, residual, prob, training, branch_scale, residual_scale
):
    # type: (Tuple[Tensor, Optional[Tensor]], Tensor, float, bool, float, Optional[Tensor]) -> Tensor
    x, bias = x_with_bias

    if x.dtype != residual.dtype:
        x = x.to(residual.dtype)
        if bias is not None:
            bias = bias.to(residual.dtype)

    if bias is not None:
        x = x + bias
    branch = torch.nn.functional.dropout(x, p=prob, training=training)
    if residual_scale is None:
        return residual + branch_scale * branch
    return residual_scale * residual + branch_scale * branch


def bias_dropout_add_unfused(training):
    def _bias_dropout_add(x_with_bias, residual, prob):
        return _bias_dropout_add_func(x_with_bias, residual, prob, training)

    return _bias_dropout_add


def scaled_bias_dropout_add_unfused(training, branch_scale, residual_scale):
    def _bias_dropout_add(x_with_bias, residual, prob):
        return _scaled_bias_dropout_add_func(
            x_with_bias, residual, prob, training, branch_scale, residual_scale
        )

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


@jit_fuser
def scaled_bias_dropout_add_fused_train(
    x_with_bias: Tuple[torch.Tensor, Optional[torch.Tensor]],
    residual: torch.Tensor,
    prob: float,
    branch_scale: float,
    residual_scale: Optional[torch.Tensor],
) -> torch.Tensor:
    return _scaled_bias_dropout_add_func(
        x_with_bias, residual, prob, True, branch_scale, residual_scale
    )


@jit_fuser
def scaled_bias_dropout_add_fused_inference(
    x_with_bias: Tuple[torch.Tensor, Optional[torch.Tensor]],
    residual: torch.Tensor,
    prob: float,
    branch_scale: float,
    residual_scale: Optional[torch.Tensor],
) -> torch.Tensor:
    return _scaled_bias_dropout_add_func(
        x_with_bias, residual, prob, False, branch_scale, residual_scale
    )


def get_bias_dropout_add(training, fused, branch_scale=None, residual_scale=None):
    if branch_scale is not None:
        if fused:
            scaled_func = (
                scaled_bias_dropout_add_fused_train
                if training
                else scaled_bias_dropout_add_fused_inference
            )

            def _scaled_bias_dropout_add(x_with_bias, residual, prob):
                return scaled_func(
                    x_with_bias, residual, prob, branch_scale, residual_scale
                )

            return _scaled_bias_dropout_add
        return scaled_bias_dropout_add_unfused(training, branch_scale, residual_scale)

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
