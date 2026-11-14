"""Shared MoE utilities: _AllToAll and MoEAuxLossAutoScaler.

Extracted from models/*/moe.py (Level 0 Option C — pure extraction, no behavior change).
All three models (qwen3_moe, qwen3_5, deepseek_v3) had identical _AllToAll
implementations and functionally identical MoEAuxLossAutoScaler implementations.
The qwen3_moe version is used as the canonical form (adds docstring and named
intermediate variable for clarity).

Note: this is Megatron Lite's own MoEAuxLossAutoScaler, kept deliberately separate
from MC's `megatron.core.transformer.moe.moe_utils.MoEAuxLossAutoScaler`.
`runtime/backends/lite/runtime.py` calls `set_loss_scale` on this class to apply
the 1/num_microbatches aux-loss gradient scale.
"""

from __future__ import annotations

import torch  # pyright: ignore[reportMissingImports]
import torch.distributed as dist  # pyright: ignore[reportMissingImports]

__all__ = ["MoEAuxLossAutoScaler", "_AllToAll"]


class MoEAuxLossAutoScaler(torch.autograd.Function):
    """Piggyback aux_loss onto autograd so main_loss.backward() triggers it."""

    main_loss_backward_scale: torch.Tensor | None = None

    @staticmethod
    def forward(ctx, output: torch.Tensor, aux_loss: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(aux_loss)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        (aux_loss,) = ctx.saved_tensors
        scale = (
            MoEAuxLossAutoScaler.main_loss_backward_scale.to(aux_loss.device)
            if MoEAuxLossAutoScaler.main_loss_backward_scale is not None
            else torch.ones(1, device=aux_loss.device)
        )
        scaled_aux_loss_grad = torch.ones_like(aux_loss) * scale
        return grad_output, scaled_aux_loss_grad

    @staticmethod
    def set_loss_scale(scale: torch.Tensor) -> None:
        MoEAuxLossAutoScaler.main_loss_backward_scale = scale


class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, input_splits, output_splits, group):
        ctx.input_splits = input_splits
        ctx.output_splits = output_splits
        ctx.group = group
        input_tensor = input_tensor.contiguous()
        output = input_tensor.new_empty(
            [sum(output_splits)] + list(input_tensor.shape[1:])
        )
        dist.all_to_all_single(
            output,
            input_tensor,
            output_split_sizes=output_splits,
            input_split_sizes=input_splits,
            group=group,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        grad_input = grad_output.new_empty(
            [sum(ctx.input_splits)] + list(grad_output.shape[1:])
        )
        dist.all_to_all_single(
            grad_input,
            grad_output,
            output_split_sizes=ctx.input_splits,
            input_split_sizes=ctx.output_splits,
            group=ctx.group,
        )
        return grad_input, None, None, None
