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

    # Cache dequantized BF16 views of MXFP8 source params so that multiple
    # send ops for the same param reuse one dequant instead of repeating it.
    _sendable_cache: dict[str, torch.Tensor] = {}

    def _get_sendable(param_name: str, param: torch.nn.Parameter) -> torch.Tensor:
        if param_name not in _sendable_cache:
            _sendable_cache[param_name] = _ensure_sendable(param)
        return _sendable_cache[param_name]

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
                sendable = _get_sendable(op.param_name, src_param)
                src_view = sendable[op.my_slice]
                # Only copy if the slice is non-contiguous.
                if not src_view.is_contiguous():
                    src_view = src_view.contiguous()
                if submit_send_with_id is not None and op.task_id is not None:
                    submit_send_with_id(op.task_id, src_view, op.peer_rank)
                else:
                    service.submit_send(src_view, op.peer_rank)

    # Free the dequant cache — slices have been submitted and the service
    # holds its own references to the contiguous buffers it needs.
    _sendable_cache.clear()

    # Submit recvs (only if we have destination model)
    # Writebacks: each entry is one of:
    #   ('direct',)                                       — recv'd in-place, no writeback
    #   ('default', recv_buffer, dst_param, dst_slice)    — copy recv_buffer → dst_param
    #   ('transform', param_name, dst_slice, [recv_bufs]) — transform.finalize_recv
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
                # Try to recv directly into the destination parameter slice to
                # avoid allocating a separate buffer + a writeback copy.  This
                # is safe when the slice view is already contiguous AND the
                # parameter is a plain tensor (not quantized — quantized params
                # need deferred accumulation).
                dst_slice_view = dst_param.data[op.my_slice]
                if dst_slice_view.is_contiguous() and not _is_mxfp8_tensor(dst_param):
                    # Recv directly into destination — no writeback needed.
                    if submit_recv_with_id is not None and op.task_id is not None:
                        submit_recv_with_id(op.task_id, dst_slice_view, op.peer_rank)
                    else:
                        service.submit_recv(dst_slice_view, op.peer_rank)
                    recv_writebacks.append(('direct',))
                else:
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
    #
    # Since refit overwrites every slice of each param, we allocate a fresh
    # BF16 buffer (torch.empty) instead of dequantizing the existing MXFP8
    # weights — this avoids a full-model-sized dequantize+clone.
    pending_quantized: dict[int, tuple[torch.nn.Parameter, torch.Tensor, list]] = {}

    for wb in recv_writebacks:
        with torch.no_grad():
            if wb[0] == 'direct':
                # Already written in-place during recv — nothing to do.
                pass
            elif wb[0] == 'transform':
                _, param_name, dst_slice, recv_bufs = wb
                transform.finalize_recv(param_name, dst_slice, recv_bufs)
            else:
                _, recv_buffer, dst_param, dst_slice = wb
                if _is_mxfp8_tensor(dst_param):
                    # Accumulate BF16 slices for deferred quantization
                    param_id = id(dst_param)
                    if param_id not in pending_quantized:
                        # Allocate empty BF16 buffer — no need to dequantize
                        # existing weights since all slices will be overwritten.
                        full_bf16 = torch.empty(
                            dst_param.shape, dtype=torch.bfloat16,
                            device=dst_param.device,
                        )
                        pending_quantized[param_id] = (dst_param, full_bf16, [])
                    pending_quantized[param_id][2].append((dst_slice, recv_buffer))
                    pending_quantized[param_id][1][dst_slice].copy_(recv_buffer)
                else:
                    dst_param.data[dst_slice].copy_(recv_buffer)

    # Free writeback list — recv_buffers are no longer needed after copy.
    recv_writebacks.clear()

    # Finalize deferred quantized param updates
    for param_id, (dst_param, full_bf16, slices) in pending_quantized.items():
        with torch.no_grad():
            dst_param.quantize_(full_bf16)
    # Free the BF16 accumulation buffers eagerly.
    pending_quantized.clear()

    # Ensure all writeback copies are visible to subsequent CUDA ops (e.g. CUDA
    # graph warmup).  The synchronize() above fires *before* the writeback loop,
    # so without this second sync the .copy_() kernels are still async when
    # execute_reshard_plan returns — creating a race with callers that immediately
    # inspect or capture (via CUDA graphs) the destination parameters.
    torch.cuda.synchronize()

    logger.info("Reshard complete")
