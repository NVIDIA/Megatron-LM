# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Communications utilities."""


from megatron.core.device_utils import get_current_device, get_xla_model
import torch

from megatron.core import parallel_state
from megatron.core import mpu

xm = get_xla_model()

# TODO: use functions from megatron/p2p
def recv_from_prev_pipeline_rank_(recv_buffer=None):
    """Receive from previous pipeline stage and update the
    input buffer inplace."""
    if not mpu.is_pipeline_first_stage():
        assert recv_buffer is not None
        if xm:
            xm.mark_step()
            device = recv_buffer.device
            recv_buffer_orig = recv_buffer
            recv_buffer = recv_buffer.cpu()
        recv_prev_op = torch.distributed.P2POp(
            torch.distributed.irecv, recv_buffer,
            mpu.get_pipeline_model_parallel_prev_rank(),
            group=mpu.get_default_process_group())
        reqs = torch.distributed.batch_isend_irecv([recv_prev_op])
        for req in reqs:
            req.wait()
        if xm:
            recv_buffer_orig.data = recv_buffer.to(device=device)

        # To protect against race condition when using batch_isend_irecv().
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elif xm:
            xm.mark_step()



# TODO: use functions from megatron/p2p
def send_to_next_pipeline_rank(tensor=None):
    """Send output to the next pipeline stage."""
    if not mpu.is_pipeline_last_stage():
        assert tensor is not None
        if xm:
            xm.mark_step()
            tensor = tensor.cpu()
        send_next_op = torch.distributed.P2POp(
            torch.distributed.isend, tensor,
            mpu.get_pipeline_model_parallel_next_rank(),
            group=mpu.get_default_process_group())
        reqs = torch.distributed.batch_isend_irecv([send_next_op])
        for req in reqs:
            req.wait()
        # To protect against race condition when using batch_isend_irecv().
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elif xm:
            xm.mark_step()



def _is_device(tensor):
    """Check if a tensor is not none and is on a device"""
    assert tensor is not None
    assert tensor.is_cuda or tensor.is_xla



def _is_device_contiguous(tensor):
    """Check if a tensor is not none, is on a device, and is contiguous."""
    _is_device(tensor)
    assert tensor.is_contiguous()



def broadcast_from_last_pipeline_stage(size, dtype, tensor=None):
    """Broadcast a tensor from last pipeline stage to all ranks."""

    is_last_stage = mpu.is_pipeline_last_stage()
    # If first stage and last state are the same, then there is no
    # pipeline parallelism and no need to communicate.
    if mpu.is_pipeline_first_stage() and is_last_stage:
        return tensor

    if is_last_stage:
        _is_device_contiguous(tensor)
    else:
        tensor = torch.empty(size,
                             dtype=dtype,
                             device=get_current_device())
    # Get the group and corresponding source rank.
    src = mpu.get_pipeline_model_parallel_last_rank()
    
    xm = get_xla_model()
    if xm:
        groups = mpu.get_pipeline_model_parallel_groups()
        xm.collective_broadcast([tensor],
                         src,
                         groups=groups, pin_layout=False)
    else:
        group = mpu.get_pipeline_model_parallel_group()
        torch.distributed.broadcast(tensor, src, group)

    return tensor


def _send_and_recv_from_last_to_first_pipeline_stage(tensor=None):
    is_last_stage = mpu.is_pipeline_last_stage()
    is_first_stage = mpu.is_pipeline_first_stage()

    if is_last_stage or is_first_stage:
        if is_first_stage:
            recv_prev_op = torch.distributed.P2POp(
                torch.distributed.irecv, tensor,
                mpu.get_pipeline_model_parallel_last_rank())
            reqs = torch.distributed.batch_isend_irecv([recv_prev_op])
        elif is_last_stage:
            send_next_op = torch.distributed.P2POp(
                torch.distributed.isend, tensor,
                mpu.get_pipeline_model_parallel_first_rank())
            reqs = torch.distributed.batch_isend_irecv([send_next_op])

        for req in reqs:
            req.wait()
        # To protect against race condition when using batch_isend_irecv().
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        return tensor


def broadcast_from_last_to_first_pipeline_stage(size, dtype, tensor=None):
    """Broadcast tensor values from last stage into the first stage."""

    is_last_stage = mpu.is_pipeline_last_stage()
    is_first_stage = mpu.is_pipeline_first_stage()
    # If first stage and last state are the same, then there is no
    # pipeline parallelism and no need to communicate.
    if is_first_stage and is_last_stage:
        return tensor
    # Only first and last stage pipeline stages need to be involved.
    if is_last_stage or is_first_stage:
        if is_last_stage:
            _is_device_contiguous(tensor)
        else:
            tensor = torch.empty(size,
                                 dtype=dtype,
                                 device=get_current_device())
        src = mpu.get_pipeline_model_parallel_last_rank()
        
        # Broadcast from last stage into the first stage.
        xm = get_xla_model()
        if xm:
            groups = mpu.get_embedding_groups()
            xm.collective_broadcast([tensor],
                            src,
                            groups=groups, pin_layout=False)
        else:
            group = mpu.get_embedding_group()
            torch.distributed.broadcast(tensor, src, group)
    else:
        tensor = None

    return tensor



def copy_from_last_to_first_pipeline_stage(size, dtype, tensor=None):
    """Copy tensor values from last stage into the first stage.
    Note that the input tensor is updated in place."""

    is_last_stage = mpu.is_pipeline_last_stage()
    is_first_stage = mpu.is_pipeline_first_stage()
    # If first stage and last state are the same, then there is no
    # pipeline parallelism and no need to communicate.
    if is_first_stage and is_last_stage:
        return
    # Only first and last stage pipeline stages need to be involved.
    if is_last_stage or is_first_stage:
        _is_device(tensor)
        is_contiguous = tensor.is_contiguous()
        if is_contiguous:
            tensor_ = tensor
        else:
            if is_last_stage:
                tensor_ = tensor.contiguous()
            else:
                tensor_ = torch.empty(size,
                                      dtype=dtype,
                                      device=get_current_device())
        src = mpu.get_pipeline_model_parallel_last_rank()

        # Broadcast from last stage into the first stage.
        xm = get_xla_model()
        if xm:
            groups = mpu.get_embedding_groups()
            xm.collective_broadcast([tensor_],
                            src,
                            groups=groups, pin_layout=False)
        else:
            group = mpu.get_embedding_group()
            torch.distributed.broadcast(tensor_, src, group)
        # Update the first stage tensor
        if is_first_stage and not is_contiguous:
            tensor[...] = tensor_



def broadcast_tensor(size, dtype, tensor=None, rank=0, data_parallel=False):
    """Given size and type of a tensor on all ranks and the tensor value
    only on a specific rank, broadcast from that rank to all other ranks.

    Args:
        data_parallel (bool): Broadcast across a single data parallel model replica.
    """
    if data_parallel:
        rank = parallel_state.get_model_parallel_src_rank()

    if torch.distributed.get_rank() == rank:
        _is_device_contiguous(tensor)
    else:
        tensor = torch.empty(size,
                             dtype=dtype,
                             device=get_current_device())

    group = None
    if data_parallel:
        group = parallel_state.get_model_parallel_group()

    torch.distributed.broadcast(tensor, rank, group=group)

    return tensor



def broadcast_list(size, dtype, list_values=None, rank=0, data_parallel=False):
    """Broadcast a list of values with a given type.

    Args:
        data_parallel (bool): Broadcast across a single data parallel model replica.
    """

    tensor = None

    if data_parallel:
        if parallel_state.get_model_parallel_src_rank() == torch.distributed.get_rank():
            tensor = torch.tensor(list_values, dtype=dtype,
                                  device=get_current_device())

        rank = parallel_state.get_model_parallel_src_rank()
    else:
        if torch.distributed.get_rank() == rank:
            tensor = torch.tensor(list_values, dtype=dtype,
                                  device=get_current_device())

    return broadcast_tensor(size, dtype, tensor=tensor, rank=rank, data_parallel=data_parallel)



def broadcast_int_list(size, int_list=None, rank=0, data_parallel=False):
    """Broadcast a list of integer values.

    Args:
        data_parallel (bool): Broadcast across a single data parallel model replica.
    """

    return broadcast_list(size, torch.int64, list_values=int_list, rank=rank, data_parallel=data_parallel)



def broadcast_float_list(size, float_list=None, rank=0, data_parallel=False):
    """Broadcast a list of float values.

    Args:
        data_parallel (bool): Broadcast across a single data parallel model replica.
    """

    return broadcast_list(size, torch.float32, list_values=float_list,
                          rank=rank, data_parallel=data_parallel)
