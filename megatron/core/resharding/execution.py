# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist

from megatron.core.fp8_utils import is_mxfp8tensor
from megatron.core.transformer.module import MegatronModule

from .copy_services.base import CopyService
from .transforms import ReshardTransform, _ensure_sendable
from .utils import ReshardPlan, TransferOp, get_refit_tensor_dict

logger = logging.getLogger(__name__)


def refresh_module_caches(
    dst_module: torch.nn.Module | list[torch.nn.Module] | tuple[torch.nn.Module, ...] | None,
) -> None:
    """Refresh parameter-derived caches in the destination module(s).

    Lists and tuples let external refit callers, including Megatron Bridge,
    pass virtual-pipeline model chunks directly. ``None`` is a no-op for
    send-only ranks. The ``dst_module`` parameter name remains stable for
    callers that use keyword arguments.
    """
    if dst_module is None:
        return
    roots = dst_module if isinstance(dst_module, (list, tuple)) else (dst_module,)
    for root in roots:
        for module in root.modules():
            if isinstance(module, MegatronModule):
                module.refresh_cache()


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


def _validate_execution_batches(plan: ReshardPlan) -> None:
    """Validate locally checkable invariants of a batched reshard plan."""
    if plan.num_batches < 1:
        raise ValueError(f"ReshardPlan.num_batches must be positive, got {plan.num_batches}")

    recv_batch_by_param: dict[str, int] = {}
    for op in (*plan.send_ops, *plan.recv_ops):
        if not 0 <= op.batch_id < plan.num_batches:
            raise ValueError(
                f"Transfer task_id={op.task_id} has batch_id={op.batch_id}, but the plan has "
                f"{plan.num_batches} batches"
            )
        if op.is_send:
            continue
        previous_batch = recv_batch_by_param.setdefault(op.param_name, op.batch_id)
        if previous_batch != op.batch_id:
            raise ValueError(
                f"Receive operations for {op.param_name!r} span batches "
                f"{previous_batch} and {op.batch_id}; complete parameters must stay together"
            )


def _execute_batch(
    send_ops: list[TransferOp],
    recv_ops: list[TransferOp],
    src_params: dict[str, torch.Tensor],
    dst_params: dict[str, torch.Tensor],
    service: CopyService,
    transform: Optional[ReshardTransform],
    prefetch_stream: Optional[torch.cuda.Stream],
) -> bool:
    """Submit, execute, and finalize one memory-bounded operation batch."""
    # Reuse one logical source materialization within this batch. CopyService
    # retains submitted views until run(), so the cache cannot be released any
    # earlier; scoping it here prevents a model-sized BF16 cache.
    sendable_cache: dict[str, torch.Tensor] = {}
    sendable_events: dict[str, torch.cuda.Event] = {}

    mxfp8_param_names: set[str] = set()
    for op in send_ops:
        if transform is not None and transform.should_transform(op.param_name):
            continue
        src_param = src_params.get(op.param_name)
        if src_param is not None and is_mxfp8tensor(src_param):
            mxfp8_param_names.add(op.param_name)
    if mxfp8_param_names:
        assert prefetch_stream is not None
        with torch.cuda.stream(prefetch_stream):
            for param_name in mxfp8_param_names:
                sendable_cache[param_name] = _ensure_sendable(src_params[param_name])
                event = torch.cuda.Event()
                event.record()
                sendable_events[param_name] = event

    def get_sendable(param_name: str, param: torch.Tensor) -> torch.Tensor:
        if param_name not in sendable_cache:
            sendable_cache[param_name] = _ensure_sendable(param)
        return sendable_cache[param_name]

    for op in send_ops:
        src_param = src_params.get(op.param_name)
        if src_param is None:
            continue
        if transform is not None and transform.should_transform(op.param_name):
            tensors = transform.prepare_send(op.param_name, op.my_slice, src_param)
            for tensor in tensors:
                service.submit_send(tensor.contiguous(), op.peer_rank, task_id=op.task_id)
        else:
            event = sendable_events.get(op.param_name)
            if event is not None:
                torch.cuda.current_stream().wait_event(event)
            src_view = get_sendable(op.param_name, src_param)[op.my_slice]
            if not src_view.is_contiguous():
                src_view = src_view.contiguous()
            service.submit_send(src_view, op.peer_rank, task_id=op.task_id)

    writebacks: list[_Writeback] = []
    # Maps id(dst_param) -> (dst_param, full_bf16, slices) for MXFP8 dests that
    # need deferred quantize_() after all slices are written.
    pending_quantized: dict[int, tuple[torch.nn.Parameter, torch.Tensor, list]] = {}

    for op in recv_ops:
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

        # Try to recv directly into the destination parameter slice to avoid
        # allocating a separate buffer + a writeback copy. This is safe when
        # the slice view is already contiguous and the parameter is plain.
        dst_slice_view = dst_param.data[op.my_slice]
        dst_is_mxfp8 = is_mxfp8tensor(dst_param)

        if not dst_is_mxfp8 and dst_slice_view.is_contiguous():
            service.submit_recv(dst_slice_view, op.peer_rank, task_id=op.task_id)
            writebacks.append(_Writeback(kind='direct'))
            continue

        if dst_is_mxfp8:
            # Receive directly into the full BF16 accumulation buffer when its
            # slice is contiguous, avoiding a second per-slice allocation.
            full_bf16, _slices = _get_mxfp8_accumulator(pending_quantized, dst_param)
            accum_view = full_bf16[op.my_slice]
            if accum_view.is_contiguous():
                service.submit_recv(accum_view, op.peer_rank, task_id=op.task_id)
                writebacks.append(_Writeback(kind='direct'))
                continue

        recv_buffer = torch.empty_like(dst_slice_view.contiguous())
        service.submit_recv(recv_buffer, op.peer_rank, task_id=op.task_id)
        writebacks.append(
            _Writeback(
                kind='copy', recv_buffer=recv_buffer, dst_param=dst_param, dst_slice=op.my_slice
            )
        )

    service.run()
    sendable_cache.clear()
    sendable_events.clear()

    # Complete every destination parameter before releasing this batch. The
    # planner keeps all slices of a logical parameter in the same batch, which
    # is required for MXFP8 block-scale correctness.
    for i in range(len(writebacks)):
        wb = writebacks[i]
        writebacks[i] = None
        with torch.no_grad():
            if wb.kind == 'direct':
                continue
            if wb.kind == 'transform':
                assert transform is not None
                transform.finalize_recv(wb.param_name, wb.dst_slice, wb.recv_bufs)
                continue
            if is_mxfp8tensor(wb.dst_param):
                full_bf16, slices = _get_mxfp8_accumulator(pending_quantized, wb.dst_param)
                slices.append((wb.dst_slice, wb.recv_buffer))
                full_bf16[wb.dst_slice].copy_(wb.recv_buffer)
            else:
                wb.dst_param.data[wb.dst_slice].copy_(wb.recv_buffer)
    writebacks.clear()

    had_mxfp8_staging = bool(pending_quantized)
    for dst_param, full_bf16, _slices in pending_quantized.values():
        with torch.no_grad():
            dst_param.quantize_(full_bf16)
    pending_quantized.clear()
    return had_mxfp8_staging


def execute_reshard_plan(
    plan: ReshardPlan,
    src_module: torch.nn.Module,
    dst_module: torch.nn.Module,
    service: CopyService,
    group=None,
    transform: Optional[ReshardTransform] = None,
) -> None:
    """
    Execute a reshard plan (built locally on each rank).
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
    service.set_plan(plan, transform=transform)

    # Extract parameters and persistent buffers from models if present.
    # Persistent buffers carry training state (e.g. MoE router expert_bias)
    # and must be refit alongside parameters.  Cached on each module so the
    # named_modules() walk happens once per model, not per refit.
    src_params = get_refit_tensor_dict(src_module) if src_module is not None else {}
    dst_params = get_refit_tensor_dict(dst_module) if dst_module is not None else {}

    if service.execute_plan(plan, src_params, dst_params, transform=transform):
        logger.info("Executing native reshard plan")
        torch.cuda.synchronize()
        if service.requires_process_group_barrier:
            dist.barrier(group=group)
        refresh_module_caches(dst_module)
        torch.cuda.synchronize()
        logger.info("Reshard complete")
        return

    batches: list[tuple[int | None, list[TransferOp], list[TransferOp]]]
    if service.supports_incremental_runs:
        _validate_execution_batches(plan)
        send_ops_by_batch: list[list[TransferOp]] = [[] for _ in range(plan.num_batches)]
        recv_ops_by_batch: list[list[TransferOp]] = [[] for _ in range(plan.num_batches)]
        for op in plan.send_ops:
            send_ops_by_batch[op.batch_id].append(op)
        for op in plan.recv_ops:
            recv_ops_by_batch[op.batch_id].append(op)
        batches = [
            (batch_id, send_ops_by_batch[batch_id], recv_ops_by_batch[batch_id])
            for batch_id in range(plan.num_batches)
        ]
    else:
        # NIXL registers the complete receive address map once and requires the
        # same map on later refits, so preserve its model-wide submission.
        batches = [(None, plan.send_ops, plan.recv_ops)]

    prefetch_stream = (
        torch.cuda.Stream()
        if any(
            op.param_name in src_params and is_mxfp8tensor(src_params[op.param_name])
            for op in plan.send_ops
        )
        else None
    )

    had_mxfp8_staging = False
    for batch_id, send_ops, recv_ops in batches:
        batch_label = "all" if batch_id is None else f"{batch_id + 1}/{plan.num_batches}"
        logger.info(
            "Executing reshard batch %s: %d sends + %d recvs",
            batch_label,
            len(send_ops),
            len(recv_ops),
        )
        had_mxfp8_staging |= _execute_batch(
            send_ops, recv_ops, src_params, dst_params, service, transform, prefetch_stream
        )

    torch.cuda.synchronize()
    if service.requires_process_group_barrier:
        dist.barrier(group=group)

    refresh_module_caches(dst_module)

    # Cache refresh may enqueue parameter-derived copies after the batch-completion
    # synchronize above. Ensure those are visible before a caller inspects weights
    # or begins CUDA graph capture.
    torch.cuda.synchronize()

    # Release transient BF16 recv/accumulation buffers back to the CUDA driver.
    # Without this the caching allocator retains the peak allocation, which can
    # be significant for MXFP8 destinations (one bounded batch in BF16).
    # Skip the (expensive) empty_cache walk when no MXFP8 staging happened.
    if had_mxfp8_staging:
        torch.cuda.empty_cache()

    logger.info("Reshard complete")
