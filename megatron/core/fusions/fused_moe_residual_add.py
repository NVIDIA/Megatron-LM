# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Fused MoE output residual add: ``residual + (expert_output + shared_expert_output)``.

Replaces the two elementwise adds that follow the MoE token combine (the shared-expert add in
``MoELayer.postprocess`` and the residual add of the bias-dropout-add without bias or dropout)
with one kernel that reads the three tensors once. The routed and shared expert outputs are
summed and rounded to the activation dtype before the residual is added, so the result is
bit-identical to the two unfused adds.
"""

from unittest.mock import MagicMock

import torch

from megatron.core.utils import null_decorator

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False

if not HAVE_TRITON:
    triton = MagicMock()
    triton.jit = null_decorator
    tl = MagicMock()

_BLOCK_SIZE = 4096
_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


@triton.jit
def _moe_residual_add_kernel(
    expert_output_ptr,
    shared_expert_output_ptr,
    residual_ptr,
    output_ptr,
    num_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    expert_output = tl.load(expert_output_ptr + offsets, mask=mask).to(tl.float32)
    shared_expert_output = tl.load(shared_expert_output_ptr + offsets, mask=mask).to(tl.float32)
    residual = tl.load(residual_ptr + offsets, mask=mask).to(tl.float32)
    # Round the MoE output to the activation dtype before the residual add, as the unfused
    # adds do.
    dtype = output_ptr.dtype.element_ty
    moe_output = (expert_output + shared_expert_output).to(dtype).to(tl.float32)
    tl.store(output_ptr + offsets, (residual + moe_output).to(dtype), mask=mask)


class _FusedMoEResidualAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, expert_output, shared_expert_output, residual):
        """Return ``residual + (expert_output + shared_expert_output)``."""
        output = torch.empty_like(residual)
        num_elements = output.numel()
        grid = (triton.cdiv(num_elements, _BLOCK_SIZE),)
        _moe_residual_add_kernel[grid](
            expert_output,
            shared_expert_output,
            residual,
            output,
            num_elements,
            BLOCK_SIZE=_BLOCK_SIZE,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Pass the output gradient to all three inputs, which enter the sum with unit weight."""
        return grad_output, grad_output, grad_output


def can_use_fused_moe_residual_add(
    expert_output: torch.Tensor, shared_expert_output: torch.Tensor, residual: torch.Tensor
) -> bool:
    """Return whether ``fused_moe_residual_add`` supports the given tensors.

    The tensors must be contiguous CUDA tensors on the same device with the same shape and the
    same BF16, FP16 or FP32 dtype, and Triton must be available.
    """
    return (
        HAVE_TRITON
        and residual.is_cuda
        and expert_output.device == shared_expert_output.device == residual.device
        and residual.dtype in _SUPPORTED_DTYPES
        and expert_output.dtype == shared_expert_output.dtype == residual.dtype
        and expert_output.shape == shared_expert_output.shape == residual.shape
        and expert_output.is_contiguous()
        and shared_expert_output.is_contiguous()
        and residual.is_contiguous()
    )


def fused_moe_residual_add(
    expert_output: torch.Tensor, shared_expert_output: torch.Tensor, residual: torch.Tensor
) -> torch.Tensor:
    """Compute ``residual + (expert_output + shared_expert_output)`` in one kernel.

    The result is bit-identical to the two unfused adds, and the output gradient is passed to all
    three inputs. The inputs must satisfy ``can_use_fused_moe_residual_add``.

    Args:
        expert_output (torch.Tensor): Combined routed-expert output.
        shared_expert_output (torch.Tensor): Shared-expert output.
        residual (torch.Tensor): Residual stream input of the MLP.

    Returns:
        torch.Tensor: The MLP output added to the residual stream.
    """
    return _FusedMoEResidualAdd.apply(expert_output, shared_expert_output, residual)
