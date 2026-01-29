# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
from typing import TYPE_CHECKING, Optional, Tuple

import torch

from megatron.core.jit import jit_fuser

if TYPE_CHECKING:
    from megatron.core.tensor_parallel.random import MHCBlockRecomputeManager

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
        if prob != 0.0:
            out = torch.nn.functional.dropout(x, p=prob, training=training, inplace=inplace)
        else:
            out = x
        if inplace:
            out.add_(residual)
        else:
            out = residual + out
        return out
    else:
        if prob != 0.0:
            out = torch.nn.functional.dropout(x, p=prob, training=training, inplace=inplace)
        else:
            out = x
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


def get_bias_dropout_add(training, fused, mhc_recompute_manager: Optional['MHCBlockRecomputeManager'] = None):
    """
    Get the bias-dropout-add function.
    
    Args:
        training: Whether in training mode.
        fused: Whether to use fused implementation.
        mhc_recompute_manager: Optional MHCBlockRecomputeManager for checkpoint management.
            When provided, the returned function will wrap the BDA operation with
            CheckpointWithoutOutput for memory-efficient recomputation.
    
    Returns:
        A callable that performs bias-dropout-add operation.
    """
    if mhc_recompute_manager is not None:
        # Return a checkpointed version that handles tuple unpacking internally
        return _get_checkpointed_bda(training, fused, mhc_recompute_manager)
    
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


def _get_checkpointed_bda(training, fused, mhc_recompute_manager: 'MHCBlockRecomputeManager'):
    """
    Create a checkpointed bias-dropout-add function.
    
    This function handles:
    1. Tuple unpacking for x_with_bias (required because save_for_backward can't save tuples)
    2. Non-tensor arguments like dropout probability (handled by CheckpointWithoutOutput)
    3. Auto-registration to the MHCBlockRecomputeManager
    
    Args:
        training: Whether in training mode.
        fused: Whether to use fused implementation.
        mhc_recompute_manager: MHCBlockRecomputeManager for checkpoint management.
    
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
