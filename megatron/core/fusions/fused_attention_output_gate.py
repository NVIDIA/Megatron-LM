# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Fused elementwise sigmoid attention output gate with the scale cast before the multiply.

``core_attn_out * sigmoid(gate.float()).to(core_attn_out.dtype)`` is the elementwise MLA output
gate (``apply_attention_output_gate(..., 'elementwise', cast_mode='before')``). Eagerly it is
four kernels in forward (two casts, the sigmoid, the multiply) and six in backward, and autograd
keeps the FP32 sigmoid output and the cast scale alive until backward. This module runs the
forward and the vector-Jacobian product as one Triton kernel each: it reads ``core_attn_out`` and
``gate`` once, saves only those two inputs and recomputes the sigmoid in backward.

The kernels replay the eager composition operation by operation with the same intermediate
roundings: the sigmoid is ``1 / (1 + expf(-x))`` in FP32 with libdevice's ``expf`` and an
IEEE-rounded division (what PyTorch's CUDA sigmoid kernel computes), the scale is rounded to the
activation dtype before the multiply, and the backward reproduces ``MulBackward``, the cast's
backward and ``SigmoidBackward`` (``grad * (1 - s) * s``) in the same order. Triton compiles
libdevice with flush-to-zero, which would turn the subnormal sigmoid outputs of gate logits near
-88 into zeros, so the division is emitted as an explicit non-FTZ ``div.rn.f32``. Outputs and
gradients are therefore bit-identical to the eager helper
(``apply_attention_output_gate(..., 'elementwise', cast_mode='before')``).

The inputs must be contiguous CUDA tensors of the same shape; the backward is first-order only
(no double backward).
"""

from unittest.mock import MagicMock

import torch

from megatron.core.utils import null_decorator

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False

if not HAVE_TRITON:
    triton = MagicMock()
    triton.jit = null_decorator
    tl = MagicMock()
    libdevice = MagicMock()

_BLOCK_SIZE = 4096
SUPPORTED_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


@triton.jit
def _sigmoid_fp32(x, one):
    """PyTorch's CUDA sigmoid: ``1 / (1 + expf(-x))`` in FP32 with an IEEE-rounded division.

    ``libdevice.exp`` is ``__nv_expf``; the quotient uses an explicit ``div.rn.f32`` because
    Triton's libdevice build (``__nv_fdiv_rn``) is flush-to-zero and would drop subnormal results.
    """
    denominator = one + libdevice.exp(-x)
    return tl.inline_asm_elementwise(
        "div.rn.f32 $0, $1, $2;",
        "=r,r,r",
        [one, denominator],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _gate_before_cast_fwd_kernel(
    core_attn_out_ptr, gate_ptr, output_ptr, num_elements, BLOCK_SIZE: tl.constexpr
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    out_dtype = output_ptr.dtype.element_ty
    core_attn_out = tl.load(core_attn_out_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    one = tl.full([BLOCK_SIZE], 1.0, tl.float32)
    # cast_mode='before': round the FP32 sigmoid to the activation dtype, then multiply in FP32
    # and round the product once, as the eager BF16/FP16 multiply does.
    scale = _sigmoid_fp32(gate, one).to(out_dtype).to(tl.float32)
    tl.store(output_ptr + offsets, (core_attn_out * scale).to(out_dtype), mask=mask)


@triton.jit
def _gate_before_cast_bwd_kernel(
    grad_output_ptr,
    core_attn_out_ptr,
    gate_ptr,
    grad_core_attn_out_ptr,
    grad_gate_ptr,
    num_elements,
    NEED_GRAD_CORE_ATTN_OUT: tl.constexpr,
    NEED_GRAD_GATE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    out_dtype = core_attn_out_ptr.dtype.element_ty
    grad_output = tl.load(grad_output_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    one = tl.full([BLOCK_SIZE], 1.0, tl.float32)
    sigmoid_out = _sigmoid_fp32(gate, one)
    if NEED_GRAD_CORE_ATTN_OUT:
        # MulBackward for core_attn_out: grad_output times the scale cast as in forward.
        scale = sigmoid_out.to(out_dtype).to(tl.float32)
        tl.store(grad_core_attn_out_ptr + offsets, (grad_output * scale).to(out_dtype), mask=mask)
    if NEED_GRAD_GATE:
        core_attn_out = tl.load(core_attn_out_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        # MulBackward for the scale (rounded to the activation dtype), the cast's backward (back
        # to FP32) and SigmoidBackward: grad * (1 - s) * s, then the cast to the gate dtype.
        grad_scale = (grad_output * core_attn_out).to(out_dtype).to(tl.float32)
        grad_gate = (grad_scale * (one - sigmoid_out)) * sigmoid_out
        tl.store(grad_gate_ptr + offsets, grad_gate.to(grad_gate_ptr.dtype.element_ty), mask=mask)


class _FusedElementwiseGateBeforeCast(torch.autograd.Function):
    """Autograd wrapper of the two kernels; saves only the two contiguous inputs."""

    @staticmethod
    def forward(ctx, core_attn_out: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """Return ``core_attn_out * sigmoid(gate.float()).to(core_attn_out.dtype)``."""
        output = torch.empty_like(core_attn_out)
        num_elements = output.numel()
        if num_elements:
            grid = (triton.cdiv(num_elements, _BLOCK_SIZE),)
            _gate_before_cast_fwd_kernel[grid](
                core_attn_out, gate, output, num_elements, BLOCK_SIZE=_BLOCK_SIZE
            )
        ctx.save_for_backward(core_attn_out, gate)
        return output

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output: torch.Tensor):
        """Recompute the sigmoid and produce both input gradients in one kernel."""
        core_attn_out, gate = ctx.saved_tensors
        need_grad_core_attn_out, need_grad_gate = ctx.needs_input_grad
        grad_core_attn_out = torch.empty_like(core_attn_out) if need_grad_core_attn_out else None
        grad_gate = torch.empty_like(gate) if need_grad_gate else None
        num_elements = core_attn_out.numel()
        if (need_grad_core_attn_out or need_grad_gate) and num_elements:
            grad_output = grad_output.contiguous()
            grid = (triton.cdiv(num_elements, _BLOCK_SIZE),)
            # Unused output pointers receive a valid placeholder; the constexpr flags compile
            # their stores out.
            _gate_before_cast_bwd_kernel[grid](
                grad_output,
                core_attn_out,
                gate,
                grad_core_attn_out if grad_core_attn_out is not None else grad_output,
                grad_gate if grad_gate is not None else grad_output,
                num_elements,
                NEED_GRAD_CORE_ATTN_OUT=need_grad_core_attn_out,
                NEED_GRAD_GATE=need_grad_gate,
                BLOCK_SIZE=_BLOCK_SIZE,
            )
        return grad_core_attn_out, grad_gate


def can_use_fused_elementwise_attention_output_gate(
    core_attn_out: torch.Tensor, gate: torch.Tensor
) -> bool:
    """Return whether ``fused_elementwise_attention_output_gate_before_cast`` supports the inputs.

    Both tensors must be contiguous CUDA tensors on the same device with the same shape and BF16,
    FP16 or FP32 dtypes (they may differ from each other), and Triton must be available.
    Contiguity is required so that the autograd graph keeps the inputs themselves and no copies.
    """
    return (
        HAVE_TRITON
        and core_attn_out.is_cuda
        and gate.is_cuda
        and core_attn_out.device == gate.device
        and core_attn_out.dtype in SUPPORTED_DTYPES
        and gate.dtype in SUPPORTED_DTYPES
        and core_attn_out.shape == gate.shape
        and core_attn_out.is_contiguous()
        and gate.is_contiguous()
    )


def fused_elementwise_attention_output_gate_before_cast(
    core_attn_out: torch.Tensor, gate: torch.Tensor
) -> torch.Tensor:
    """Compute ``core_attn_out * sigmoid(gate.float()).to(core_attn_out.dtype)`` in one kernel.

    Forward values and the gradients of both inputs are bit-identical to the eager composition;
    the autograd graph keeps only ``core_attn_out`` and ``gate``. The inputs must satisfy
    ``can_use_fused_elementwise_attention_output_gate``.

    Args:
        core_attn_out (torch.Tensor): Core attention output, flattened local channels last.
        gate (torch.Tensor): Gate logits with the same shape as ``core_attn_out``.

    Returns:
        torch.Tensor: The gated attention output in ``core_attn_out``'s dtype.
    """
    return _FusedElementwiseGateBeforeCast.apply(core_attn_out, gate)
