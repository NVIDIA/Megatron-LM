from typing import Optional

import torch
from torch import nn

from megatron.core.tensor_parallel.mappings import _reduce


class _LayerScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, parallel_input: bool) -> torch.Tensor:
        ctx.save_for_backward(x, weight)
        ctx.parallel_input = parallel_input
        return weight*x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
        x, weight = ctx.saved_tensors
        if not ctx.parallel_input:
            return grad_output*weight, grad_output*x, None

        x_parallel = x
        grad_output_parallel = grad_output
        grad_x_parallel = grad_output_parallel*weight

        grad_weight_parallel = grad_output_parallel*weight
        grad_weight_parallel_reduced = torch.sum(grad_weight_parallel.view(-1, weight.size(-1)), dim=0)
        grad_weight = _reduce(grad_weight_parallel_reduced)

        return grad_x_parallel, grad_weight, None


class LayerScale(nn.Module):
    def __init__(self, hidden_size: Optional[int] = None, initial_value: float = 1.0, device=None, dtype=None,
                 sequence_parallel: bool = False):
        super().__init__()
        if hidden_size is None:
            self.weight = torch.nn.Parameter(torch.empty(1, device=device, dtype=dtype))
        else:
            self.weight = torch.nn.Parameter(torch.empty(hidden_size, device=device, dtype=dtype))
        self.sequence_parallel = sequence_parallel
        self.initial_value = initial_value
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.weight, self.initial_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        if x_dtype != self.weight.dtype:
            x = x.to(self.weight.dtype)
        y = _LayerScale.apply(x, self.weight, self.sequence_parallel)
        if x_dtype != self.weight.dtype:
            return y.to(x_dtype)
        return y
