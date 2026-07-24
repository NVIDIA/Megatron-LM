# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.


from typing import Any, Optional

import torch


class _ReduceScatterWithFP32AccumulationWorkHandle:
    """Work handle to return to user when using reduce_scatter_with_fp32_accumulation with
    async_op=True."""

    def __init__(
        self,
        all_to_all_handle: Any,
        all_to_all_output_tensor: torch.Tensor,
        output_tensor: torch.Tensor,
        world_size: int,
    ):
        """Initialize WorkHandle object."""
        self.all_to_all_handle = all_to_all_handle
        self.all_to_all_output_tensor = all_to_all_output_tensor
        self.output_tensor = output_tensor
        self.world_size = world_size

    def wait(self):
        """Wait until communication (and associated computation) is completed."""
        # Wait for communication to complete if needed.
        if self.all_to_all_handle is not None:
            self.all_to_all_handle.wait()

        # Accumulate into a fp32 sum.
        output_tensor_in_fp32 = torch.sum(
            self.all_to_all_output_tensor.view((self.world_size, -1)), dim=0, dtype=torch.float32
        )
        assert output_tensor_in_fp32.dtype == torch.float32

        # Copy downcasted sum into output_tensor.
        self.output_tensor.copy_(output_tensor_in_fp32)


def reduce_scatter_with_fp32_accumulation(
    output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    op: torch.distributed.ReduceOp,
    group: torch.distributed.ProcessGroup,
    async_op: bool,
    output_buffer: Optional[torch.Tensor] = None,
):
    """Reduce-scatter with FP32 accumulation.

    Collects input_tensor in lower precision using an all-to-all, then locally accumulates in FP32
    precision, then downcasts final sum back into right location in input_tensor.


    Args:
        output_tensor (torch.Tensor): Output tensor with reduce-scattered output (only the shard).
        input_tensor (torch.Tensor): Input tensor that needs to be reduce-scattered.
        op (torch.distributed.ReduceOp): Only torch.distributed.ReduceOp.SUM is supported.
        group (torch.distributed.ProcessGroup): Process group to use for reduce-scatter.
        async_op (bool): Only False is supported right now.
        output_buffer (Optional[torch.Tensor]): Caller-owned scratch buffer to use as the
            all-to-all output, in lieu of allocating a fresh tensor. Must have the same dtype
            as input_tensor and at least input_tensor.numel() elements. The caller is
            responsible for ensuring that any previous call that used this buffer has had its
            returned work handle waited on before passing the buffer here again, otherwise the
            in-flight all-to-all would race with the new one. When None, the function
            allocates a fresh tensor each call (the safe default).
    """
    # Make sure arguments conform to the implementation.
    assert op == torch.distributed.ReduceOp.SUM

    # Get world_size.
    if group is None:
        world_size = torch.distributed.get_world_size()
    else:
        world_size = group.size()

    # Make sure input_tensor size is divisible by world size.
    assert input_tensor.numel() % world_size == 0

    # Call all_to_all (every rank should have their respective gradient shards collected from
    # all ranks). The all-to-all collective cannot be performed in-place, so we need a
    # separate output tensor sized like input_tensor. If a shared scratch buffer is provided,
    # slice into it; otherwise allocate fresh.
    if output_buffer is not None:
        assert output_buffer.dtype == input_tensor.dtype, (
            f"output_buffer dtype {output_buffer.dtype} does not match input_tensor dtype "
            f"{input_tensor.dtype}"
        )
        assert output_buffer.numel() >= input_tensor.numel(), (
            f"output_buffer has {output_buffer.numel()} elements but input_tensor has "
            f"{input_tensor.numel()} elements"
        )
        all_to_all_output_tensor = output_buffer[: input_tensor.numel()].view(input_tensor.shape)
    else:
        all_to_all_output_tensor = torch.empty_like(input_tensor)
    all_to_all_handle = torch.distributed.all_to_all_single(
        output=all_to_all_output_tensor, input=input_tensor, group=group, async_op=async_op
    )

    # Create a work handle to finish communication and reduction.
    reduce_scatter_handle = _ReduceScatterWithFP32AccumulationWorkHandle(
        all_to_all_handle, all_to_all_output_tensor, output_tensor, world_size
    )
    if async_op:
        # Return work handle; consumers can call .wait() to ensure communication and associated
        # reduction complete.
        return reduce_scatter_handle
    else:
        # Wait on work handle.
        reduce_scatter_handle.wait()
