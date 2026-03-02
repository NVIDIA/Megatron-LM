# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
from __future__ import annotations

import logging
from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist

from .copy_services.base import CopyService
from .utils import ReshardPlan

logger = logging.getLogger(__name__)


class ReshardTransform:
    """Hook for custom send/recv/writeback during reshard execution.

    Implementations override the four methods below.  When an instance is
    passed to ``execute_reshard_plan``, each ``TransferOp`` is checked via
    ``should_transform(param_name)``; if True the transform methods are used
    instead of the default send/recv/writeback logic.

    The transform may change the wire format (e.g. send MXFP8 data+scale
    instead of BF16) **or** keep the same wire format and only post-process
    on the receive side (e.g. receive BF16, convert to MXFP8 in
    ``finalize_recv``).  The only constraint is that ``prepare_send`` and
    ``prepare_recv`` must return the same number of tensors for a given
    parameter so that send/recv pairs match.
    """

    def should_transform(self, param_name: str) -> bool:
        """Return True if *param_name* should use the transform path."""
        return False

    def prepare_send(
        self,
        param_name: str,
        src_slice: tuple[slice, ...],
        src_param: torch.nn.Parameter,
    ) -> list[torch.Tensor]:
        """Produce tensor(s) to send for *param_name*.

        May return multiple tensors (e.g. data + scale when converting to
        MXFP8 on the sender side).  The default implementation sends the
        BF16 slice unchanged (single tensor).
        """
        raise NotImplementedError

    def prepare_recv(
        self,
        param_name: str,
        dst_slice: tuple[slice, ...],
    ) -> list[torch.Tensor]:
        """Allocate receive buffer(s).  Count must match ``prepare_send`` output."""
        raise NotImplementedError

    def finalize_recv(
        self,
        param_name: str,
        dst_slice: tuple[slice, ...],
        recv_buffers: list[torch.Tensor],
    ) -> None:
        """Write received data into final destination (e.g. persistent buffers).

        This is where receiver-side format conversion can happen (e.g.
        converting a BF16 recv buffer to MXFP8 before writing into
        persistent storage).
        """
        raise NotImplementedError


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
                src_view = src_param.data[op.my_slice].contiguous()
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

    # Write back received buffers into their destination parameter slices
    for wb in recv_writebacks:
        with torch.no_grad():
            if wb[0] == 'transform':
                _, param_name, dst_slice, recv_bufs = wb
                transform.finalize_recv(param_name, dst_slice, recv_bufs)
            else:
                _, recv_buffer, dst_param, dst_slice = wb
                dst_param.data[dst_slice].copy_(recv_buffer)

    logger.info("Reshard complete")


# ---------------------------------------------------------------------------
# MXFP8 transform helpers
# ---------------------------------------------------------------------------

def _scale_slice_from_data_slice(
    data_slice: tuple[slice, ...],
    block_size: int = 32,
) -> tuple[slice, ...]:
    """Convert an MXFP8 data slice to the corresponding scale slice.

    In MXFP8, each group of ``block_size`` elements along the last (K)
    dimension shares a single scale value.  All dimensions except the last
    are passed through unchanged; the last ``slice`` has its start/stop
    divided by ``block_size``.
    """
    adjusted = list(data_slice)
    last = adjusted[-1]
    if isinstance(last, slice) and last.start is not None:
        assert last.start % block_size == 0 and last.stop % block_size == 0, (
            f"MXFP8 data slice last dim ({last}) must be aligned to block_size={block_size}"
        )
        adjusted[-1] = slice(last.start // block_size, last.stop // block_size)
    return tuple(adjusted)


class MXFP8ReshardTransform(ReshardTransform):
    """MXFP8 format-conversion transform for reshard.

    Writes received weight data directly into persistent ``MXFP8Tensor``
    buffers so that CUDA-graph device-pointer captures remain valid across
    refits.

    Two modes are supported, controlled by *convert_on_send*:

    ``convert_on_send=False`` (default — **receiver-side conversion**):
        The sender transmits plain BF16 (one tensor per op, identical to the
        default reshard path).  The receiver allocates a BF16 receive buffer,
        then ``finalize_recv`` converts BF16 → MXFP8 and writes into the
        persistent buffers.  Because the wire format is unchanged the sender
        does **not** need a transform — only the receiver creates one.  This
        is the simplest mode and avoids any sender/receiver coordination.

    ``convert_on_send=True`` (**sender-side conversion**):
        The sender converts each BF16 slice to MXFP8 and sends **two**
        tensors (data + scale) per op.  The receiver allocates matching
        MXFP8 buffers and ``finalize_recv`` copies them directly.  Both
        sender and receiver must use the transform so that tensor counts
        match.  This mode halves wire bandwidth (~1 byte/elem vs 2).

        **Caveat**: CopyService backends that match local (same-rank)
        transfers by ``task_id`` (Gloo, NVSHMEM) will break if multiple
        tensors share the same ``task_id``.  This mode is therefore only
        safe for non-colocated setups where sender and receiver are on
        different ranks.  A future fix could generate unique sub-IDs.

    Args:
        convertible_params: set of fully-qualified parameter names that
            should use this transform.
        persistent_buffers: dict mapping parameter names (without
            *buffer_key_prefix*) to ``MXFP8Tensor`` objects that hold the
            receiver's persistent data/scale storage.  Empty on the sender
            when using ``convert_on_send=True``.
        buffer_key_prefix: prefix to strip from ``param_name`` when looking
            up entries in *persistent_buffers* (e.g. ``"decoder."``).
        convert_on_send: if True, convert BF16 → MXFP8 on the sender and
            transmit two tensors (data + scale).  If False (default),
            transmit BF16 and convert on the receiver in ``finalize_recv``.
    """

    def __init__(
        self,
        convertible_params: set[str],
        persistent_buffers: dict,
        buffer_key_prefix: str = "",
        convert_on_send: bool = False,
    ):
        self.convertible_params = convertible_params
        self.persistent_buffers = persistent_buffers
        self.buffer_key_prefix = buffer_key_prefix
        self.convert_on_send = convert_on_send

    def should_transform(self, param_name: str) -> bool:
        return param_name in self.convertible_params

    # -- send ----------------------------------------------------------------

    def prepare_send(self, param_name, src_slice, src_param):
        if self.convert_on_send:
            from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor

            bf16_data = src_param.data[src_slice].contiguous().to(torch.bfloat16)
            mxfp8 = MXFP8Tensor.from_bf16(bf16_data)
            return [mxfp8.data.contiguous(), mxfp8.scale.contiguous()]
        else:
            # BF16 on the wire — same as the default reshard path.
            return [src_param.data[src_slice].contiguous()]

    # -- recv ----------------------------------------------------------------

    def prepare_recv(self, param_name, dst_slice):
        buf_key = param_name.removeprefix(self.buffer_key_prefix)
        buf = self.persistent_buffers[buf_key]

        if self.convert_on_send:
            # Receive MXFP8 data + scale (2 buffers).
            scale_slice = _scale_slice_from_data_slice(dst_slice)
            return [
                torch.empty_like(buf.data[dst_slice].contiguous()),
                torch.empty_like(buf.scale[scale_slice].contiguous()),
            ]
        else:
            # Receive BF16 (1 buffer, same shape as the MXFP8 data slice).
            shape = buf.data[dst_slice].shape
            return [torch.empty(shape, dtype=torch.bfloat16, device=buf.data.device)]

    def finalize_recv(self, param_name, dst_slice, recv_buffers):
        buf_key = param_name.removeprefix(self.buffer_key_prefix)
        buf = self.persistent_buffers[buf_key]
        scale_slice = _scale_slice_from_data_slice(dst_slice)

        if self.convert_on_send:
            # Already MXFP8 — direct copy.
            buf.data[dst_slice].copy_(recv_buffers[0])
            buf.scale[scale_slice].copy_(recv_buffers[1])
        else:
            # Convert received BF16 → MXFP8, then copy into persistent buffers.
            from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor

            mxfp8 = MXFP8Tensor.from_bf16(recv_buffers[0])
            buf.data[dst_slice].copy_(mxfp8.data)
            buf.scale[scale_slice].copy_(mxfp8.scale)
