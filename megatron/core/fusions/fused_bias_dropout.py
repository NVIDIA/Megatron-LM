# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Optional, Tuple

import torch
from torch import Tensor

from megatron.core.fusions.fused_bias_grad import bias_with_main_grad
from megatron.core.jit import jit_fuser

if TYPE_CHECKING:
    from megatron.core.tensor_parallel.random import MHCCheckpointManager

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


def get_bias_dropout_add(
    training: bool,
    fused: bool,
    mhc_recompute_manager: Optional['MHCCheckpointManager'] = None,
    *,
    gradient_accumulation_fusion: bool = False,
) -> Callable[..., Tensor]:
    """
    Get the bias-dropout-add function.

    Args:
        training: Whether in training mode.
        fused: Whether to use fused implementation.
        mhc_recompute_manager: Optional MHCCheckpointManager for checkpoint management.
            When provided, the returned function will wrap the BDA operation with
            CheckpointWithoutOutput for memory-efficient recomputation.
        gradient_accumulation_fusion: Enable direct bias accumulation when the wrapper
            supplies an FP32 gradient buffer.

    Returns:
        A callable that performs bias-dropout-add operation.
    """
    if mhc_recompute_manager is not None:
        bda = _get_checkpointed_bda(training, fused, mhc_recompute_manager)
    elif fused:
        bda = bias_dropout_add_fused_train if training else bias_dropout_add_fused_inference
    else:
        bda = bias_dropout_add_unfused(training)

    if not (training and fused and gradient_accumulation_fusion):
        return bda

    def _bda_with_main_grad(x_with_bias, residual, prob):
        if isinstance(x_with_bias, tuple):
            x, bias = x_with_bias
            if bias is not None:
                dtype = residual.dtype if x.dtype != residual.dtype else None
                x_with_bias = (x, bias_with_main_grad(bias, x.shape, dtype))
        return bda(x_with_bias, residual, prob)

    return _bda_with_main_grad


def configure_bias_dropout_add(
    factory: Callable[..., Any], gradient_accumulation_fusion: bool
) -> Callable[..., Any]:
    """Enable FP32 bias accumulation for the built-in BDA spec."""
    if factory is get_bias_dropout_add and gradient_accumulation_fusion:
        return partial(factory, gradient_accumulation_fusion=True)
    return factory


def _get_checkpointed_bda(training, fused, mhc_recompute_manager: 'MHCCheckpointManager'):
    """
    Create a checkpointed bias-dropout-add function.

    This function handles:
    1. Tuple unpacking for x_with_bias (required because save_for_backward can't save tuples)
    2. Non-tensor arguments like dropout probability (handled by CheckpointWithoutOutput)
    3. Auto-registration to the MHCCheckpointManager

    Args:
        training: Whether in training mode.
        fused: Whether to use fused implementation.
        mhc_recompute_manager: MHCCheckpointManager for checkpoint management.

    Returns:
        A callable that performs checkpointed bias-dropout-add operation.
    """
    from megatron.core.tensor_parallel.random import CheckpointWithoutOutput

    # Get the underlying BDA function
    if fused:
        if training:
            bda_func = bias_dropout_add_fused_train
        else:
            bda_func = bias_dropout_add_fused_inference
    else:
        bda_func = bias_dropout_add_unfused(training)

    def _checkpointed_bda(x_with_bias, residual, prob):
        """
        Checkpointed BDA that handles tuple unpacking internally.

        Args:
            x_with_bias: Either a tuple (x, bias) or a single tensor x.
            residual: Residual tensor.
            prob: Dropout probability.

        Returns:
            Output tensor after bias-dropout-add.
        """
        # Create checkpoint with manager
        ckpt = CheckpointWithoutOutput(ckpt_manager=mhc_recompute_manager)

        # Handle case where x_with_bias might be a single tensor (e.g., from IdentityOp)
        if isinstance(x_with_bias, tuple):
            x, bias = x_with_bias
        else:
            x = x_with_bias
            bias = None

        # Wrapper function that re-packs the tuple for the actual BDA function
        def _bda_wrapper(output, bias, res, dropout):
            return bda_func((output, bias), res, dropout)

        # Call checkpoint with unpacked arguments
        result = ckpt.checkpoint(_bda_wrapper, x, bias, residual, prob)

        # No-op when manager is set - manager handles all discarding uniformly
        ckpt.discard_output_and_register_recompute(result)

        return result

    return _checkpointed_bda
