# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
from __future__ import annotations

"""
Reshard transforms for custom send/recv/writeback during weight transfer.

- ReshardTransform: base class for pluggable format conversion hooks.
- MXFP8ReshardTransform: writes received BF16 data into persistent FlashInfer
  MXFP8Tensor buffers so CUDA-graph device-pointer captures remain valid.
"""

import torch

from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor


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
        self, param_name: str, src_slice: tuple[slice, ...], src_param: torch.nn.Parameter
    ) -> list[torch.Tensor]:
        """Produce tensor(s) to send for *param_name*.

        May return multiple tensors (e.g. data + scale when converting to
        MXFP8 on the sender side).  The default implementation sends the
        BF16 slice unchanged (single tensor).
        """
        raise NotImplementedError

    def prepare_recv(self, param_name: str, dst_slice: tuple[slice, ...]) -> list[torch.Tensor]:
        """Allocate receive buffer(s).  Count must match ``prepare_send`` output."""
        raise NotImplementedError

    def finalize_recv(
        self, param_name: str, dst_slice: tuple[slice, ...], recv_buffers: list[torch.Tensor]
    ) -> None:
        """Write received data into final destination (e.g. persistent buffers).

        This is where receiver-side format conversion can happen (e.g.
        converting a BF16 recv buffer to MXFP8 before writing into
        persistent storage).
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# MXFP8 transform helpers
# ---------------------------------------------------------------------------


def _scale_slice_from_data_slice(
    data_slice: tuple[slice, ...], block_size: int = 32
) -> tuple[slice, ...]:
    """Convert an MXFP8 data slice to the corresponding scale slice.

    In MXFP8, each group of ``block_size`` elements along the last (K)
    dimension shares a single scale value.  All dimensions except the last
    are passed through unchanged; the last ``slice`` has its start/stop
    divided by ``block_size``.  Integer index on the last dim is converted
    to scale index as idx // block_size.
    """
    adjusted = list(data_slice)
    last = adjusted[-1]
    if isinstance(last, slice):
        if last.start is not None and last.start % block_size != 0:
            raise AssertionError(
                f"MXFP8 data slice last dim ({last}) must be aligned to block_size={block_size}"
            )
        if last.stop is not None and last.stop % block_size != 0:
            raise AssertionError(
                f"MXFP8 data slice last dim ({last}) must be aligned to block_size={block_size}"
            )
        scale_start = (last.start // block_size) if last.start is not None else None
        scale_stop = (last.stop // block_size) if last.stop is not None else None
        # Scale has one value per block; do not use last.step (would index scale wrong).
        adjusted[-1] = slice(scale_start, scale_stop)
    elif isinstance(last, int):
        adjusted[-1] = last // block_size
    return tuple(adjusted)


def _ensure_sendable(param: torch.Tensor) -> torch.Tensor:
    """Return a standard-dtype tensor suitable for wire transmission.

    Quantized parameter types (e.g., Transformer Engine MXFP8Tensor) are
    dequantized to their original precision (usually BF16).  Standard
    parameters are returned via ``.data`` (unwrapped from autograd).
    """
    try:
        from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Tensor as _TEMXFP8

        if isinstance(param, _TEMXFP8):
            return param.dequantize()
    except ImportError:
        pass
    return param.data


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
        # Accumulation buffers for 1D-scale params that arrive in partial slices.
        # The 1D swizzled FlashInfer scale can't be updated partially; we collect
        # all BF16 slices here and quantize the full weight once it's assembled.
        # Maps buf_key -> (full-size BF16 accumulation tensor, elements written so far).
        self._pending_1d: dict = {}

    def should_transform(self, param_name: str) -> bool:
        return param_name in self.convertible_params

    # -- send ----------------------------------------------------------------

    def prepare_send(self, param_name, src_slice, src_param):
        src_data = _ensure_sendable(src_param)
        if self.convert_on_send:

            bf16_data = src_data[src_slice].contiguous().to(torch.bfloat16)
            mxfp8 = MXFP8Tensor.from_bf16(bf16_data)
            return [mxfp8.data.contiguous(), mxfp8.scale.contiguous()]
        else:
            # BF16 on the wire — same as the default reshard path.
            return [src_data[src_slice].contiguous()]

    # -- recv ----------------------------------------------------------------

    def prepare_recv(self, param_name, dst_slice):
        buf_key = param_name.removeprefix(self.buffer_key_prefix)
        buf = self.persistent_buffers[buf_key]

        if self.convert_on_send:
            # Receive MXFP8 data + scale (2 buffers).
            if buf.scale.ndim == 1:
                # 1D swizzled scale can't be partially reconstructed from sender-quantized
                # slices.  Use convert_on_send=False for models with 1D-scale params.
                raise NotImplementedError(
                    f"convert_on_send=True is not supported for parameters with 1D swizzled "
                    f"scale (param={param_name!r}).  Use convert_on_send=False instead, which "
                    f"receives BF16 and quantizes the full weight on the receiver."
                )
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

        if self.convert_on_send:
            # Already MXFP8 on the wire — copy data and 2D scale slices directly.
            # (1D scale is rejected at prepare_recv time, so only 2D reaches here.)
            buf.data[dst_slice].copy_(recv_buffers[0])
            scale_slice = _scale_slice_from_data_slice(dst_slice)
            buf.scale[scale_slice].copy_(recv_buffers[1])
        elif buf.scale.ndim == 1:
            # 1D swizzled scale (FlashInfer format) encodes scale values across the
            # full weight tensor; partial updates would corrupt the swizzle layout.
            # Accumulate BF16 slices and quantize once all slices are assembled.
            if buf_key not in self._pending_1d:
                # Use zeros so that any un-filled slice produces zeros rather than garbage.
                self._pending_1d[buf_key] = [
                    torch.zeros_like(buf.data, dtype=torch.bfloat16),
                    0,  # elements written so far
                ]
            accum, written = self._pending_1d[buf_key]
            accum[dst_slice].copy_(recv_buffers[0])
            written += recv_buffers[0].numel()
            if written >= buf.data.numel():
                if written != buf.data.numel():
                    raise AssertionError(
                        f"1D-scale param {param_name!r}: received {written} elements, "
                        f"expected {buf.data.numel()} (duplicate or missing slices?)"
                    )
                mxfp8 = MXFP8Tensor.from_bf16(accum)
                buf.data.copy_(mxfp8.data)
                buf.scale.copy_(mxfp8.scale)
                del self._pending_1d[buf_key]
            else:
                self._pending_1d[buf_key][1] = written
        else:
            # 2D scale: each scale row covers exactly one data row, so partial
            # row-wise updates are independent and can be applied immediately.
            mxfp8 = MXFP8Tensor.from_bf16(recv_buffers[0])
            buf.data[dst_slice].copy_(mxfp8.data)
            scale_slice = _scale_slice_from_data_slice(dst_slice)
            buf.scale[scale_slice].copy_(mxfp8.scale)
