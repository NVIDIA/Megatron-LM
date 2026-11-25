# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Contiguous CP row transport; projected sources gather, gradients reduce-scatter."""
import torch


class GatherRows(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        out = x.new_empty((x.shape[0] * group.size(), *x.shape[1:]))
        torch.distributed.all_gather_into_tensor(out, x.contiguous(), group=group)
        return out

    @staticmethod
    def backward(ctx, grad):
        out = grad.new_empty((grad.shape[0] // ctx.group.size(), *grad.shape[1:]))
        torch.distributed.reduce_scatter_tensor(out, grad.contiguous(), group=ctx.group)
        return out, None


def gather_rows(x, group):
    return x if group is None or group.size() == 1 else GatherRows.apply(x, group)
