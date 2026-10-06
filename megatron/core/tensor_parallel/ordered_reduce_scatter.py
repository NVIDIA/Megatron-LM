# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Sequence-parallel reduction with a fixed FP32 source-rank addition order."""

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _sum_sources(inp, out, count: tl.constexpr, ranks: tl.constexpr, block: tl.constexpr):
        offsets = tl.program_id(0) * block + tl.arange(0, block)
        mask = offsets < count
        result = tl.load(inp + offsets, mask=mask, other=0).to(tl.float32)
        for rank in tl.static_range(1, ranks):
            result = result + tl.load(inp + rank * count + offsets, mask=mask, other=0).to(
                tl.float32
            )
        tl.store(out + offsets, result, mask=mask)


class _OrderedReduceScatter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_: torch.Tensor, group: torch.distributed.ProcessGroup):
        """Reduce sequence shards in a fixed source-rank order using FP32 additions."""
        ctx.group = group
        size = group.size()
        if size == 1:
            return input_
        received = torch.empty_like(input_, memory_format=torch.contiguous_format)
        torch.distributed.all_to_all_single(received, input_.contiguous(), group=group)
        shape = (input_.shape[0] // size, *input_.shape[1:])
        if triton is not None and input_.is_cuda:
            output = torch.empty(shape, dtype=input_.dtype, device=input_.device)
            count = output.numel()
            if count:
                # One kernel keeps intermediate FP32 sums in registers. Disable
                # fusion and preserve the validated source-rank addition order.
                _sum_sources[(triton.cdiv(count, 1024),)](
                    received, output, count, size, 1024, enable_fp_fusion=False, num_warps=4
                )
            return output
        parts = received.reshape(size, -1)
        # NCCL's reduction tree may change with message size. FP32 NCCL SUM
        # alone still differs at rare BF16 rounding ties. Communicate in the
        # original dtype, then add sources in the same order for every token.
        total = parts[0].to(torch.float32, copy=True)
        for rank in range(1, size):
            total.add_(parts[rank].float())
        return total.reshape(shape).to(input_.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Gather each rank's output gradient into the original unsharded shape."""
        size = ctx.group.size()
        if size == 1:
            return grad_output, None
        shape = (grad_output.shape[0] * size, *grad_output.shape[1:])
        grad_input = torch.empty(shape, dtype=grad_output.dtype, device=grad_output.device)
        torch.distributed.all_gather_into_tensor(
            grad_input, grad_output.contiguous(), group=ctx.group
        )
        return grad_input, None


def ordered_reduce_scatter_to_sequence_parallel_region(
    input_: torch.Tensor, *, group: torch.distributed.ProcessGroup
) -> torch.Tensor:
    """Sum TP partials in source-rank order and shard the leading dimension.

    Communication retains the input dtype; local additions use FP32. Backward
    gathers sequence shards without a reduction. This synchronous path requires
    extra scratch storage and does not overlap communication with the GEMM.

    Args:
        input_: Equal-sized FP16, BF16, or FP32 partial output on every TP rank.
        group: Explicit tensor-parallel process group.

    Returns:
        This rank's contiguous sequence shard, in the input dtype.
    """
    if input_.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("ordered TP reduction requires FP16, BF16, or FP32 inputs")
    if input_.ndim == 0 or input_.shape[0] % group.size():
        raise ValueError("the leading dimension must be divisible by the TP group size")
    return _OrderedReduceScatter.apply(input_, group)
