# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
from typing import List, Optional

import torch
from torch.distributed import ProcessGroup

from megatron.core import parallel_state


def is_pipeline_first_stage(pp_group: ProcessGroup):
    """Check if the current process is the first stage of the pipeline"""
    if pp_group is None:
        # set ignore_virtual=True since vpp is not used in inference
        return parallel_state.is_pipeline_first_stage(ignore_virtual=True)
    else:
        return pp_group.rank() == 0


def is_pipeline_last_stage(pp_group: ProcessGroup):
    """Check if the current process is the last stage of the pipeline"""
    if pp_group is None:
        # set ignore_virtual=True since vpp is not used in inference
        return parallel_state.is_pipeline_last_stage(ignore_virtual=True)
    else:
        return pp_group.rank() == pp_group.size() - 1


def _is_cuda(tensor):
    """Check if a tensor is not none and is cuda."""
    assert tensor is not None
    assert tensor.is_cuda


def broadcast_from_last_pipeline_stage(
    size: List[int],
    dtype: torch.dtype,
    tensor: Optional[torch.Tensor] = None,
    pp_group: Optional[ProcessGroup] = None,
):
    """Broadcast a tensor from last pipeline stage to all ranks.

    Args:
        size: Expected tensor size
        dtype: Expected tensor dtype
        tensor: Tensor to broadcast (only on last stage)
        pp_group: Custom process group (if None, uses global state)
    """
    # Use custom process group or fall back to global state
    if pp_group is None:
        pp_group = parallel_state.get_pipeline_model_parallel_group()
        last_rank = parallel_state.get_pipeline_model_parallel_last_rank()

        # add ignore_virtual=True since vpp is not used in inference
        is_last_stage = parallel_state.is_pipeline_last_stage(ignore_virtual=True)
    else:
        # Lists of ProcessGroups are used for multimodal inference but not supported here
        assert isinstance(
            pp_group, ProcessGroup
        ), "pp_group must be a single ProcessGroup, not a list of ProcessGroups"
        last_rank = torch.distributed.get_process_group_ranks(pp_group)[pp_group.size() - 1]
        is_last_stage = pp_group.rank() == pp_group.size() - 1

    if is_last_stage:
        assert size == list(
            tensor.shape
        ), f"Expected tensor of shape {size} but got {list(tensor.shape)}"
        assert dtype == tensor.dtype, f"Expected tensor of type {dtype} but got {tensor.dtype}"
        _is_cuda(tensor)
        assert tensor.is_contiguous()
    else:
        tensor = torch.empty(size, dtype=dtype, device=torch.cuda.current_device())

    # Broadcast the tensor
    torch.distributed.broadcast(tensor, src=last_rank, group=pp_group)
    return tensor


def recv_from_prev_pipeline_rank_(
    recv_buffer: torch.Tensor = None, pp_group: Optional[ProcessGroup] = None
):
    """Receive from previous pipeline stage and update the input buffer inplace.

    Args:
        recv_buffer: Buffer to receive data into
        pp_group: Custom process group (if None, uses global state)
    """
    # Determine previous rank
    if pp_group is None:
        prev_rank = parallel_state.get_pipeline_model_parallel_prev_rank()
    else:
        # Lists of ProcessGroups are used for multimodal inference but not supported here
        assert isinstance(
            pp_group, ProcessGroup
        ), "pp_group must be a single ProcessGroup, not a list of ProcessGroups"
        prev_rank = torch.distributed.get_process_group_ranks(pp_group)[
            (pp_group.rank() - 1) % pp_group.size()
        ]

    # Create receive operation
    recv_prev_op = torch.distributed.P2POp(torch.distributed.irecv, recv_buffer, prev_rank)

    reqs = torch.distributed.batch_isend_irecv([recv_prev_op])
    for req in reqs:
        req.wait()
    # To protect against race condition when using batch_isend_irecv().
    torch.cuda.synchronize()


def send_to_next_pipeline_rank(
    tensor: torch.Tensor = None, pp_group: Optional[ProcessGroup] = None
):
    """Send output to the next pipeline stage.

    Args:
        tensor: Tensor to send
        pp_group: Custom process group (if None, uses global state)
    """
    # Determine next rank
    if pp_group is None:
        next_rank = parallel_state.get_pipeline_model_parallel_next_rank()
    else:
        # Lists of ProcessGroups are used for multimodal inference but not supported here
        assert isinstance(
            pp_group, ProcessGroup
        ), "pp_group must be a single ProcessGroup, not a list of ProcessGroups"
        next_rank = torch.distributed.get_process_group_ranks(pp_group)[
            (pp_group.rank() + 1) % pp_group.size()
        ]

    # Create send operation
    send_next_op = torch.distributed.P2POp(torch.distributed.isend, tensor, next_rank)

    reqs = torch.distributed.batch_isend_irecv([send_next_op])
    for req in reqs:
        req.wait()
    # To protect against race condition when using batch_isend_irecv().
    torch.cuda.synchronize()
