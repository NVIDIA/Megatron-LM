# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist

from megatron.core.fp8_utils import is_mxfp8tensor

from .copy_services.base import CopyService
from .transforms import ReshardTransform, _ensure_sendable
from .utils import ReshardPlan, get_refit_tensor_dict

logger = logging.getLogger(__name__)


@dataclass
class _Writeback:
    """Tagged-union for what to do with a received tensor after service.run().

    Exactly one of the three kinds applies; the other fields are unused for
    that kind.  ``direct`` means the data landed in its final destination
    during recv and there's nothing to copy.  ``copy`` copies a staging
    ``recv_buffer`` into a slice of ``dst_param`` (deferring to MXFP8
    accumulation when the dest is quantized).  ``transform`` hands the
    received buffers to a ``ReshardTransform.finalize_recv`` call.
    """

    kind: str  # 'direct' | 'copy' | 'transform'
    recv_buffer: Optional[torch.Tensor] = None
    dst_param: Optional[torch.Tensor] = None
    dst_slice: Optional[tuple] = None
    param_name: Optional[str] = None
    recv_bufs: Optional[list[torch.Tensor]] = None


def _get_mxfp8_accumulator(
    pending: dict[int, tuple], dst_param: torch.Tensor
) -> tuple[torch.Tensor, list]:
    """Get or lazily allocate the BF16 accumulation buffer for an MXFP8 dest param.

    All slices for the same dst_param land in this buffer; ``quantize_`` is
    called once after all slices have been written.  Allocates empty (not
    dequantized) because every slice will be overwritten.
    """
    param_id = id(dst_param)
    entry = pending.get(param_id)
    if entry is None:
        full_bf16 = torch.empty(dst_param.shape, dtype=torch.bfloat16, device=dst_param.device)
        entry = (dst_param, full_bf16, [])
        pending[param_id] = entry
    return entry[1], entry[2]


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
    Expected service API: submit_send(tensor, dest_rank, task_id),
    submit_recv(tensor, src_rank, task_id), run().

    Supports None for src_module and/or dst_module to allow ranks in non-collocated mode:
    - src_module=None: Rank only receives data (destination-only)
    - dst_module=None: Rank only sends data (source-only)
    - Both provided: Rank participates in both send and recv (collocated mode)

    When *transform* is provided, parameters for which
    ``transform.should_transform(param_name)`` returns True use the
    transform's prepare_send / prepare_recv / finalize_recv methods instead
    of the default slice-and-copy logic.
    """
    # Refit tensors (parameters + persistent buffers) are cached on each module
    # so the named_modules() walk happens once per model, not per refit.
    src_params = get_refit_tensor_dict(src_module) if src_module is not None else {}
    dst_params = get_refit_tensor_dict(dst_module) if dst_module is not None else {}

    # Dequantized BF16 views of MXFP8 source params are reused across multiple
    # send ops for the same param.  Issue all dequants on a side stream and
    # record per-param events so each send op only waits on its own dequant
    # (later dequants can overlap with earlier sends' slicing on default stream).
    sendable_cache: dict[str, torch.Tensor] = {}
    sendable_events: dict[str, torch.cuda.Event] = {}

    mxfp8_param_names: set[str] = set()
    for op in plan.send_ops:
        if transform is not None and transform.should_transform(op.param_name):
            continue
        src_param = src_params.get(op.param_name)
        if src_param is not None and is_mxfp8tensor(src_param):
            mxfp8_param_names.add(op.param_name)

    if mxfp8_param_names:
        prefetch_stream = torch.cuda.Stream()
        with torch.cuda.stream(prefetch_stream):
            for param_name in mxfp8_param_names:
                sendable_cache[param_name] = _ensure_sendable(src_params[param_name])
                ev = torch.cuda.Event()
                ev.record()
                sendable_events[param_name] = ev

    def get_sendable(param_name: str, param: torch.nn.Parameter) -> torch.Tensor:
        if param_name not in sendable_cache:
            sendable_cache[param_name] = _ensure_sendable(param)
        return sendable_cache[param_name]

    for op in plan.send_ops:
        src_param = src_params.get(op.param_name)
        if src_param is None:
            continue
        if transform is not None and transform.should_transform(op.param_name):
            tensors = transform.prepare_send(op.param_name, op.my_slice, src_param)
            for t in tensors:
                service.submit_send(t.contiguous(), op.peer_rank, task_id=op.task_id)
        else:
            ev = sendable_events.get(op.param_name)
            if ev is not None:
                torch.cuda.current_stream().wait_event(ev)
            sendable = get_sendable(op.param_name, src_param)
            src_view = sendable[op.my_slice]
            if not src_view.is_contiguous():
                src_view = src_view.contiguous()
            service.submit_send(src_view, op.peer_rank, task_id=op.task_id)

    sendable_cache.clear()
    sendable_events.clear()

    writebacks: list[_Writeback] = []
    # Maps id(dst_param) -> (dst_param, full_bf16, slices) for MXFP8 dests that
    # need deferred quantize_() after all slices are written.
    pending_quantized: dict[int, tuple[torch.nn.Parameter, torch.Tensor, list]] = {}

    for op in plan.recv_ops:
        if transform is not None and transform.should_transform(op.param_name):
            recv_bufs = transform.prepare_recv(op.param_name, op.my_slice)
            for buf in recv_bufs:
                service.submit_recv(buf, op.peer_rank, task_id=op.task_id)
            writebacks.append(
                _Writeback(
                    kind='transform',
                    param_name=op.param_name,
                    dst_slice=op.my_slice,
                    recv_bufs=recv_bufs,
                )
            )
            continue

        dst_param = dst_params.get(op.param_name)
        if dst_param is None:
            continue

        dst_slice_view = dst_param.data[op.my_slice]
        dst_is_mxfp8 = is_mxfp8tensor(dst_param)

        if not dst_is_mxfp8 and dst_slice_view.is_contiguous():
            # Plain tensor: recv straight into the destination slice.
            service.submit_recv(dst_slice_view, op.peer_rank, task_id=op.task_id)
            writebacks.append(_Writeback(kind='direct'))
            continue

        if dst_is_mxfp8:
            full_bf16, _slices = _get_mxfp8_accumulator(pending_quantized, dst_param)
            accum_view = full_bf16[op.my_slice]
            if accum_view.is_contiguous():
                # Recv straight into the BF16 accumulator slice.
                service.submit_recv(accum_view, op.peer_rank, task_id=op.task_id)
                writebacks.append(_Writeback(kind='direct'))
                continue

        # Fallback: stage into a temporary BF16 buffer.
        recv_buffer = torch.empty_like(dst_slice_view.contiguous())
        service.submit_recv(recv_buffer, op.peer_rank, task_id=op.task_id)
        writebacks.append(
            _Writeback(
                kind='copy', recv_buffer=recv_buffer, dst_param=dst_param, dst_slice=op.my_slice
            )
        )

    logger.info(f"Executing {len(plan.send_ops)} sends + {len(plan.recv_ops)} recvs")
    service.run()
    torch.cuda.synchronize()
    dist.barrier(group=group)

    # Writebacks: ``direct`` already landed in place; ``transform`` hands off to
    # the transform; ``copy`` copies the staging buffer into the destination
    # slice (deferring MXFP8 accumulation to one quantize_() per param).
    for i in range(len(writebacks)):
        wb = writebacks[i]
        writebacks[i] = None  # Drop reference eagerly so recv buffers can free.
        with torch.no_grad():
            if wb.kind == 'direct':
                continue
            if wb.kind == 'transform':
                transform.finalize_recv(wb.param_name, wb.dst_slice, wb.recv_bufs)
                continue
            # 'copy'
            if is_mxfp8tensor(wb.dst_param):
                full_bf16, slices = _get_mxfp8_accumulator(pending_quantized, wb.dst_param)
                slices.append((wb.dst_slice, wb.recv_buffer))
                full_bf16[wb.dst_slice].copy_(wb.recv_buffer)
            else:
                wb.dst_param.data[wb.dst_slice].copy_(wb.recv_buffer)
    writebacks.clear()

    had_mxfp8_staging = bool(pending_quantized)
    for _param_id, (dst_param, full_bf16, _slices) in pending_quantized.items():
        with torch.no_grad():
            dst_param.quantize_(full_bf16)
    pending_quantized.clear()

    # Second sync: the writeback loop's .copy_() kernels are still async when
    # execute_reshard_plan returns; without this CUDA-graph capture or callers
    # that read params immediately race against the writes.
    torch.cuda.synchronize()

    # MXFP8 destinations allocate a full-model-sized BF16 staging buffer that
    # can dwarf the rest of the working set.  Reclaim it back to the driver
    # when present; skip the (expensive) empty_cache walk otherwise.
    if had_mxfp8_staging:
        torch.cuda.empty_cache()

    logger.info("Reshard complete")
