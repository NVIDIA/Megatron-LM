# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

from typing import Tuple, Union

import torch

from megatron.core.transformer import TransformerConfig
from megatron.core.utils import prepare_input_tensors_for_wgrad_compute


class SyncedLinearAutograd(torch.autograd.Function):
    @staticmethod
    def forward(input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return input.matmul(weight.t())

    @staticmethod
    def setup_context(ctx, inputs: Tuple[torch.Tensor], output: torch.Tensor) -> None:
        input, weight = inputs
        ctx.save_for_backward(input, weight)

    @staticmethod
    def backward(ctx, grad_output) -> Tuple[Union[torch.Tensor, None], ...]:
        input, weight = ctx.saved_tensors
        grad_input = grad_weight = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.matmul(weight)
        if ctx.needs_input_grad[1]:
            grad_output, input = prepare_input_tensors_for_wgrad_compute(grad_output, input)
            grad_weight = grad_output.t().matmul(input)
            with torch.no_grad():
                torch.distributed.all_reduce(grad_weight, op=torch.distributed.ReduceOp.AVG)

        return grad_input, grad_weight


class SyncedLinear(torch.nn.Module):

    def __init__(self, input_size: int, output_size: int, init_method: callable, config: TransformerConfig, broadcast_weights: bool = True, **kwargs) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(init_method(torch.empty(
            output_size,
            input_size,
            device=torch.cuda.current_device(),
            dtype=config.params_dtype,
        )))
        if broadcast_weights:
            with torch.no_grad():
                torch.distributed.broadcast(self.weight, src=0)
    
    def forward(self, input: torch.Tensor) -> Tuple[Union[torch.Tensor, None]]:
        return SyncedLinearAutograd.apply(input, self.weight), None
