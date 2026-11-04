# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
from typing import List, Optional

import torch
from torch.distributed import ProcessGroup

from megatron.core.process_groups_config import resolve_process_groups


def _resolve_group(
    group: Optional[ProcessGroup], *, field: str, argument: str, owner: str
) -> ProcessGroup:
    """Return `group`, or the global `field` group when the caller of `owner` omitted `argument`.

    The global fallback is deprecated: it warns once per `owner` with a
    `ProcessGroupFallbackWarning` and reads the group from `megatron.core.parallel_state`.
    """
    if group is not None:
        return group
    pg_collection = resolve_process_groups(
        None,
        owner=owner,
        required=(field,),
        argument=argument,
        deprecated_in="0.21",
        removed_in="0.23",
    )
    group = vars(pg_collection)[field]
    # torch.distributed treats a None group as the whole world.
    if group is None:
        raise ValueError(
            f"{owner} was called without `{argument}`, and megatron.core.parallel_state has no "
            f"`{field}` group to fall back to. Pass the model's process group."
        )
    return group


def is_pipeline_first_stage(pp_group: Optional[ProcessGroup]):
    """Check if the current process is the first stage of the pipeline.

    Args:
        pp_group: Pipeline-parallel group of the model. If None, the global pipeline-parallel
            group is used; that fallback is deprecated.
    """
    pp_group = _resolve_group(
        pp_group, field="pp", argument="pp_group", owner="is_pipeline_first_stage"
    )
    return pp_group.rank() == 0


def is_pipeline_last_stage(pp_group: Optional[ProcessGroup]):
    """Check if the current process is the last stage of the pipeline.

    Args:
        pp_group: Pipeline-parallel group of the model. If None, the global pipeline-parallel
            group is used; that fallback is deprecated.
    """
    pp_group = _resolve_group(
        pp_group, field="pp", argument="pp_group", owner="is_pipeline_last_stage"
    )
    return pp_group.rank() == pp_group.size() - 1


def _is_cuda(tensor):
    """Check if a tensor is not none and is cuda."""
    assert tensor is not None
    assert tensor.is_cuda


def _is_cuda_contiguous(tensor):
    """Check if a tensor is not none, is cuda, and is contiguous."""
    _is_cuda(tensor)
    assert tensor.is_contiguous()


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
        pp_group: Pipeline-parallel group of the model. If None, the global pipeline-parallel
            group is used; that fallback is deprecated.
    """
    pp_group = _resolve_group(
        pp_group, field="pp", argument="pp_group", owner="broadcast_from_last_pipeline_stage"
    )
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
        _is_cuda_contiguous(tensor)
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
        pp_group: Pipeline-parallel group of the model. If None, the global pipeline-parallel
            group is used; that fallback is deprecated.
    """
    pp_group = _resolve_group(
        pp_group, field="pp", argument="pp_group", owner="recv_from_prev_pipeline_rank_"
    )
    # Lists of ProcessGroups are used for multimodal inference but not supported here
    assert isinstance(
        pp_group, ProcessGroup
    ), "pp_group must be a single ProcessGroup, not a list of ProcessGroups"
    # Determine previous rank
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
        pp_group: Pipeline-parallel group of the model. If None, the global pipeline-parallel
            group is used; that fallback is deprecated.
    """
    pp_group = _resolve_group(
        pp_group, field="pp", argument="pp_group", owner="send_to_next_pipeline_rank"
    )
    # Lists of ProcessGroups are used for multimodal inference but not supported here
    assert isinstance(
        pp_group, ProcessGroup
    ), "pp_group must be a single ProcessGroup, not a list of ProcessGroups"
    # Determine next rank
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


def broadcast_tensor(
    size, dtype, tensor=None, rank=0, data_parallel=False, mp_group: Optional[ProcessGroup] = None
):
    """Given size and type of a tensor on all ranks and the tensor value
    only on a specific rank, broadcast from that rank to all other ranks.

    Args:
        data_parallel (bool): Broadcast across a single data parallel model replica.
        mp_group: Model-parallel group of the model, used with `data_parallel`: the tensor is
            broadcast within it from its first rank, and `rank` is ignored. If None, the global
            model-parallel group is used; that fallback is deprecated.
    """
    group = None
    if data_parallel:
        group = _resolve_group(mp_group, field="mp", argument="mp_group", owner="broadcast_tensor")
        rank = torch.distributed.get_global_rank(group, 0)

    if torch.distributed.get_rank() == rank:
        _is_cuda_contiguous(tensor)
    else:
        tensor = torch.empty(size, dtype=dtype, device=torch.cuda.current_device())

    torch.distributed.broadcast(tensor, rank, group=group)

    return tensor


def broadcast_list(
    size,
    dtype,
    list_values=None,
    rank=0,
    data_parallel=False,
    mp_group: Optional[ProcessGroup] = None,
):
    """Broadcast a list of values with a given type.

    Args:
        data_parallel (bool): Broadcast across a single data parallel model replica.
        mp_group: Model-parallel group of the model, used with `data_parallel`: the values are
            broadcast within it from its first rank, and `rank` is ignored. If None, the global
            model-parallel group is used; that fallback is deprecated.
    """

    tensor = None

    if data_parallel:
        mp_group = _resolve_group(mp_group, field="mp", argument="mp_group", owner="broadcast_list")
        rank = torch.distributed.get_global_rank(mp_group, 0)
        if rank == torch.distributed.get_rank():
            tensor = torch.tensor(list_values, dtype=dtype, device=torch.cuda.current_device())
    else:
        if torch.distributed.get_rank() == rank:
            tensor = torch.tensor(list_values, dtype=dtype, device=torch.cuda.current_device())

    return broadcast_tensor(
        size, dtype, tensor=tensor, rank=rank, data_parallel=data_parallel, mp_group=mp_group
    )


def broadcast_int_list(
    size, int_list=None, rank=0, data_parallel=False, mp_group: Optional[ProcessGroup] = None
):
    """Broadcast a list of integer values.

    Args:
        data_parallel (bool): Broadcast across a single data parallel model replica.
        mp_group: Model-parallel group of the model, used with `data_parallel`; see
            `broadcast_list`.
    """

    return broadcast_list(
        size,
        torch.int64,
        list_values=int_list,
        rank=rank,
        data_parallel=data_parallel,
        mp_group=mp_group,
    )


def broadcast_float_list(
    size, float_list=None, rank=0, data_parallel=False, mp_group: Optional[ProcessGroup] = None
):
    """Broadcast a list of float values.

    Args:
        data_parallel (bool): Broadcast across a single data parallel model replica.
        mp_group: Model-parallel group of the model, used with `data_parallel`; see
            `broadcast_list`.
    """

    return broadcast_list(
        size,
        torch.float32,
        list_values=float_list,
        rank=rank,
        data_parallel=data_parallel,
        mp_group=mp_group,
    )
