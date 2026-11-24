# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""FP32 bias-gradient reduction and accumulation."""

from typing import Any

import torch
from torch import Tensor

from megatron.core.jit import jit_fuser


@jit_fuser
def accumulate_bias_grad(grad_output: Tensor, main_grad: Tensor, overwrite: bool = False) -> None:
    """Reduce in FP32 and update the wrapper-owned buffer.

    Keeping the reduction and mutation in one compiled region permits fusion
    without materializing a full-size FP32 copy of grad_output. The compiler
    chooses the reduction decomposition; this does not promise one kernel for
    every shape. Calls targeting the same buffer must be stream ordered.
    Sharded FSDP requests overwrite for its temporary full-gradient buffer;
    FSDP itself accumulates the reduced shards across microbatches.
    """
    reduce_dims = tuple(range(grad_output.dim() - 1))
    reduced = grad_output.sum(dim=reduce_dims, dtype=torch.float32)
    if overwrite:
        main_grad.copy_(reduced)
    else:
        main_grad.add_(reduced)


def try_accumulate_bias_grad(bias: Tensor, grad_output: Tensor, main_grad: Tensor | None) -> bool:
    """Write one bias contribution through the owning wrapper's FP32 buffer."""
    fsdp_bias = hasattr(bias, "__fsdp_param__")
    if fsdp_bias:
        main_grad = bias.get_main_grad()
    if main_grad is None or main_grad.dtype != torch.float32:
        return False

    # Sharded FSDP reuses its temporary full gradient buffer on each backward.
    overwrite = getattr(bias, "overwrite_main_grad", False) and not getattr(
        bias, "bias_grad_added_to_main_grad", False
    )
    accumulate_bias_grad(grad_output, main_grad, overwrite)
    if fsdp_bias:
        bias.bias_grad_added_to_main_grad = True
    return True


class _BiasWithMainGrad(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any, bias: Tensor, shape: torch.Size, dtype: torch.dtype, main_grad: Tensor | None
    ) -> Tensor:
        """Cast and broadcast the bias while preserving its wrapper metadata."""
        ctx.bias = bias
        ctx.main_grad = main_grad
        ctx.differentiable_bias_grad = False
        # Cast the small vector before expanding. In particular, FP32 residual
        # connections must deliver FP32 per-position gradients to backward.
        return bias.to(dtype).expand(shape)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Tensor, None, None, None]:
        """Reduce the per-position gradient and hand it to the owning wrapper."""
        # Double backward can revisit this node with grad mode disabled.
        ctx.differentiable_bias_grad |= torch.is_grad_enabled()
        if not ctx.differentiable_bias_grad and try_accumulate_bias_grad(
            ctx.bias, grad_output, ctx.main_grad
        ):
            # The wrapper still needs an autograd edge and any ordinary .grad.
            grad_bias = torch.zeros_like(ctx.bias)
        else:
            grad_bias = grad_output.sum(dim=tuple(range(grad_output.dim() - 1)))
        return grad_bias, None, None, None


def bias_with_main_grad(
    bias: Tensor | None, shape: torch.Size, dtype: torch.dtype | None = None
) -> Tensor | None:
    """Expose a broadcast view so the consumer hands back per-position gradients."""
    if (
        not isinstance(bias, torch.Tensor)
        or not bias.requires_grad
        or not torch.is_grad_enabled()
        or len(shape) < 2
        or bias.shape != (shape[-1],)
        or not hasattr(bias, "grad_added_to_main_grad")
    ):
        return bias

    main_grad = None
    if not hasattr(bias, "__fsdp_param__"):
        main_grad = getattr(bias, "main_grad", None)
        if main_grad is None or main_grad.dtype != torch.float32:
            return bias
    return _BiasWithMainGrad.apply(bias, shape, bias.dtype if dtype is None else dtype, main_grad)
