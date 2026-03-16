# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.distributed as dist

from .copy_services.base import CopyService
from .transforms import ReshardTransform, _ensure_sendable
from .utils import ReshardPlan

logger = logging.getLogger(__name__)


def _is_mxfp8_tensor(param):
    """Check if param is a TE MXFP8Tensor (fp8_param=true)."""
    return (
        hasattr(param, 'quantize_')
        and hasattr(param, 'dequantize')
        and hasattr(param, '_rowwise_data')
    )


def execute_reshard_plan(
    plan: ReshardPlan,
    src_module: torch.nn.Module,
    dst_module: torch.nn.Module,
    service: CopyService,
    group=None,
    transform: Optional[ReshardTransform] = None,
) -> None:
    """
    Execute a reshard plan (from centralized controller).
    A communication service must be provided to abstract transport.
    Expected service API: submit_send(tensor, dest_rank), submit_recv(tensor, src_rank), run().

    Supports None for src_module and/or dst_module to allow ranks in non-collocated mode:
    - src_module=None: Rank only receives data (destination-only)
    - dst_module=None: Rank only sends data (source-only)
    - Both provided: Rank participates in both send and recv (collocated mode)

    When *transform* is provided, parameters for which
    ``transform.should_transform(param_name)`` returns True use the
    transform's prepare_send / prepare_recv / finalize_recv methods instead
    of the default slice-and-copy logic.
    """

    # Extract parameters from models if present
    src_params = {}
    dst_params = {}
    if src_module is not None:
        src_params = {name: p for name, p in src_module.named_parameters(recurse=True)}
    if dst_module is not None:
        dst_params = {name: p for name, p in dst_module.named_parameters(recurse=True)}

    submit_send_with_id = getattr(service, "submit_send_with_id", None)
    submit_recv_with_id = getattr(service, "submit_recv_with_id", None)

    # Submit sends (only if we have source model)
    for op in plan.send_ops:
        if transform is not None and transform.should_transform(op.param_name):
            src_param = src_params.get(op.param_name)
            if src_param is not None:
                tensors = transform.prepare_send(op.param_name, op.my_slice, src_param)
                for t in tensors:
                    buf = t.contiguous()
                    if submit_send_with_id is not None and op.task_id is not None:
                        submit_send_with_id(op.task_id, buf, op.peer_rank)
                    else:
                        service.submit_send(buf, op.peer_rank)
        else:
            src_param = src_params.get(op.param_name)
            if src_param is not None:
                src_view = _ensure_sendable(src_param)[op.my_slice].contiguous()
                if submit_send_with_id is not None and op.task_id is not None:
                    submit_send_with_id(op.task_id, src_view, op.peer_rank)
                else:
                    service.submit_send(src_view, op.peer_rank)

    # Submit recvs (only if we have destination model)
    # Writebacks: each entry is either
    #   ('default', recv_buffer, dst_param, dst_slice)  or
    #   ('transform', param_name, dst_slice, [recv_buffers])
    recv_writebacks: list = []

    for op in plan.recv_ops:
        if transform is not None and transform.should_transform(op.param_name):
            recv_bufs = transform.prepare_recv(op.param_name, op.my_slice)
            for buf in recv_bufs:
                if submit_recv_with_id is not None and op.task_id is not None:
                    submit_recv_with_id(op.task_id, buf, op.peer_rank)
                else:
                    service.submit_recv(buf, op.peer_rank)
            recv_writebacks.append(('transform', op.param_name, op.my_slice, recv_bufs))
        else:
            dst_param = dst_params.get(op.param_name)
            if dst_param is not None:
                dst_slice_view = dst_param.data[op.my_slice]
                recv_buffer = torch.empty_like(dst_slice_view.contiguous())
                if submit_recv_with_id is not None and op.task_id is not None:
                    submit_recv_with_id(op.task_id, recv_buffer, op.peer_rank)
                else:
                    service.submit_recv(recv_buffer, op.peer_rank)
                recv_writebacks.append(('default', recv_buffer, dst_param, op.my_slice))

    # Execute
    logger.info(f"Executing {len(plan.send_ops)} sends + {len(plan.recv_ops)} recvs")
    service.run()
    torch.cuda.synchronize()
    dist.barrier(group=group)

    # Write back received buffers into their destination parameter slices.
    #
    # For quantized destination params (fp8_param=true on receiver),
    # accumulate ALL BF16 slices per-param before calling quantize_() once.
    # This avoids corrupting MXFP8 per-block scales from partial-slice updates.
    pending_quantized: dict[int, tuple[torch.nn.Parameter, torch.Tensor, list]] = {}

    for wb in recv_writebacks:
        with torch.no_grad():
            if wb[0] == 'transform':
                _, param_name, dst_slice, recv_bufs = wb
                transform.finalize_recv(param_name, dst_slice, recv_bufs)
            else:
                _, recv_buffer, dst_param, dst_slice = wb
                if _is_mxfp8_tensor(dst_param):
                    # Accumulate BF16 slices for deferred quantization
                    param_id = id(dst_param)
                    if param_id not in pending_quantized:
                        full_bf16 = dst_param.dequantize().clone()
                        pending_quantized[param_id] = (dst_param, full_bf16, [])
                    pending_quantized[param_id][2].append((dst_slice, recv_buffer))
                    pending_quantized[param_id][1][dst_slice].copy_(recv_buffer)
                else:
                    dst_param.data[dst_slice].copy_(recv_buffer)

    # Finalize deferred quantized param updates
    for param_id, (dst_param, full_bf16, slices) in pending_quantized.items():
        with torch.no_grad():
            dst_param.quantize_(full_bf16)

    # Ensure all writeback copies are visible to subsequent CUDA ops (e.g. CUDA
    # graph warmup).  The synchronize() above fires *before* the writeback loop,
    # so without this second sync the .copy_() kernels are still async when
    # execute_reshard_plan returns — creating a race with callers that immediately
    # inspect or capture (via CUDA graphs) the destination parameters.
    torch.cuda.synchronize()

    logger.info("Reshard complete")
