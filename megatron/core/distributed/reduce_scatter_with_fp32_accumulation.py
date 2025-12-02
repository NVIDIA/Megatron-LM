# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.


from typing import Any

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
    # all ranks). We also create a tensor for the all-to-all output (the all-to-all collective
    # cannot be performed in-place).
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
