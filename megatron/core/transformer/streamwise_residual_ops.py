# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Full-width streamwise residual operations.

The pure PyTorch functions are the numerical reference and CPU fallback. CUDA
training can use dedicated Triton kernels that consume the padded controller
logits directly. Those kernels form sigmoid factors in registers and compute
controller gradients from per-program partials, avoiding both dense slot maps
and a second activation-reading GEMM/BMM in backward.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl

    HAVE_STREAMWISE_TRITON = True
except ImportError:
    HAVE_STREAMWISE_TRITON = False


_STREAMWISE_MIN_BATCH = 256
_STREAMWISE_MAX_STREAMS = 16
_STREAMWISE_MAX_REDUCTION_BLOCK = 16_384


def _flatten_leading(tensor: Tensor) -> tuple[Tensor, torch.Size]:
    """Flatten all leading dimensions for the two-dimensional Triton kernels."""

    leading = tensor.shape[:-1]
    return tensor.reshape(-1, tensor.shape[-1]), leading


def _unflatten_leading(tensor: Tensor, leading: torch.Size) -> Tensor:
    """Restore leading dimensions after a Triton kernel launch."""

    return tensor.reshape(*leading, tensor.shape[-1])


def _streamwise_block_config(batch: int, stream_width: int) -> tuple[int, int, int]:
    """Use the block geometry tuned for the existing small-map MHC kernels."""

    if stream_width >= 512:
        return 32, 256, 4
    block_batch = 64 if batch >= 4096 else 32
    return block_batch, min(128, stream_width), 4


def _num_gradient_partials(batch: int, stream_width: int) -> int:
    block_batch, block_width, _ = _streamwise_block_config(batch, stream_width)
    return math.ceil(batch / block_batch) * math.ceil(stream_width / block_width)


def _validate_logits(logits: Tensor, num_streams: int, *, tensor_name: str) -> None:
    if num_streams <= 0:
        raise ValueError(f"num_streams must be positive, got {num_streams}.")
    if logits.ndim != 1 or logits.numel() < num_streams:
        raise ValueError(
            f"{tensor_name} must be one-dimensional with at least {num_streams} values, "
            f"got shape {tuple(logits.shape)}."
        )
    if not logits.is_floating_point():
        raise TypeError(f"{tensor_name} must use a floating-point dtype, got {logits.dtype}.")


def _validate_raw_read_inputs(hidden_states: Tensor, read_logits: Tensor, num_streams: int) -> int:
    _validate_logits(read_logits, num_streams, tensor_name="read_logits")
    if hidden_states.ndim == 0 or hidden_states.shape[-1] == 0:
        raise ValueError("hidden_states must have a non-empty hidden dimension.")
    if hidden_states.shape[-1] % num_streams != 0:
        raise ValueError(
            f"hidden_states hidden size {hidden_states.shape[-1]} is not divisible by "
            f"num_streams={num_streams}."
        )
    if read_logits.device != hidden_states.device:
        raise ValueError("read_logits and hidden_states must be on the same device.")
    return hidden_states.shape[-1] // num_streams


def _validate_raw_write_inputs(
    residual_stream: Tensor,
    branch_update: Tensor,
    write_logits: Tensor,
    num_streams: int,
    retention_logits: Tensor | None,
) -> int:
    stream_width = _validate_raw_read_inputs(residual_stream, write_logits, num_streams)
    expected_update_shape = (*residual_stream.shape[:-1], stream_width)
    if branch_update.shape != expected_update_shape:
        raise ValueError(
            "branch_update must match the residual leading dimensions and full-stream width, "
            f"expected {expected_update_shape}, got {tuple(branch_update.shape)}."
        )
    if branch_update.device != residual_stream.device:
        raise ValueError("branch_update and residual_stream must be on the same device.")
    if branch_update.dtype != residual_stream.dtype:
        raise ValueError("branch_update and residual_stream must have the same dtype.")
    if retention_logits is not None:
        _validate_logits(retention_logits, num_streams, tensor_name="retention_logits")
        if retention_logits.device != residual_stream.device:
            raise ValueError("retention_logits and residual_stream must be on the same device.")
    return stream_width


def _can_use_streamwise_triton(
    tensor: Tensor, logits: Tensor, num_streams: int, stream_width: int
) -> bool:
    """Return whether a direct raw-logit streamwise Triton kernel is supported."""

    if not HAVE_STREAMWISE_TRITON or tensor.ndim < 2:
        return False
    batch = math.prod(tensor.shape[:-1])
    if (
        batch < _STREAMWISE_MIN_BATCH
        or not 1 <= num_streams <= _STREAMWISE_MAX_STREAMS
        or stream_width < 64
    ):
        return False
    partials = _num_gradient_partials(batch, stream_width)
    reduction_block = triton.next_power_of_2(partials)
    return (
        tensor.is_cuda
        and logits.is_cuda
        and tensor.is_contiguous()
        and logits.is_contiguous()
        and tensor.dtype in (torch.bfloat16, torch.float16)
        and logits.dtype in (torch.bfloat16, torch.float16, torch.float32)
        and logits.ndim == 1
        and logits.numel() >= num_streams
        and tensor.shape[-1] == num_streams * stream_width
        and reduction_block <= _STREAMWISE_MAX_REDUCTION_BLOCK
    )


if HAVE_STREAMWISE_TRITON:

    @triton.jit
    def _streamwise_read_fwd_kernel(
        X,  # noqa: ANN001
        READ_LOGITS,  # noqa: ANN001
        OUT,  # noqa: ANN001
        BATCH: tl.constexpr,
        STREAM_WIDTH: tl.constexpr,
        NUM_STREAMS: tl.constexpr,
        BLOCK_BATCH: tl.constexpr,
        BLOCK_WIDTH: tl.constexpr,
    ) -> None:
        """Read full-width streams while forming sigmoid factors in registers."""

        pid_batch = tl.program_id(0)
        pid_width = tl.program_id(1)
        offsets_batch = pid_batch * BLOCK_BATCH + tl.arange(0, BLOCK_BATCH)
        offsets_width = pid_width * BLOCK_WIDTH + tl.arange(0, BLOCK_WIDTH)
        mask = (offsets_batch[:, None] < BATCH) & (offsets_width[None, :] < STREAM_WIDTH)
        output = tl.zeros((BLOCK_BATCH, BLOCK_WIDTH), tl.float32)
        for stream in tl.static_range(0, NUM_STREAMS):
            value = tl.load(
                X
                + offsets_batch[:, None] * (NUM_STREAMS * STREAM_WIDTH)
                + stream * STREAM_WIDTH
                + offsets_width[None, :],
                mask=mask,
                other=0.0,
            )
            factor = tl.sigmoid(tl.load(READ_LOGITS + stream).to(tl.float32))
            factor = factor.to(value.dtype).to(tl.float32)
            output += factor * value.to(tl.float32)
        tl.store(
            OUT + offsets_batch[:, None] * STREAM_WIDTH + offsets_width[None, :], output, mask=mask
        )

    @triton.jit
    def _streamwise_read_bwd_kernel(
        X,  # noqa: ANN001
        GRAD_OUT,  # noqa: ANN001
        READ_LOGITS,  # noqa: ANN001
        GRAD_X,  # noqa: ANN001
        GRAD_LOGIT_PARTIALS,  # noqa: ANN001
        BATCH: tl.constexpr,
        STREAM_WIDTH: tl.constexpr,
        NUM_STREAMS: tl.constexpr,
        NUM_WIDTH_BLOCKS: tl.constexpr,
        NUM_PARTIALS: tl.constexpr,
        BLOCK_BATCH: tl.constexpr,
        BLOCK_WIDTH: tl.constexpr,
    ) -> None:
        """Compute activation gradients and one logit-gradient partial per tile."""

        pid_batch = tl.program_id(0)
        pid_width = tl.program_id(1)
        partial_index = pid_batch * NUM_WIDTH_BLOCKS + pid_width
        offsets_batch = pid_batch * BLOCK_BATCH + tl.arange(0, BLOCK_BATCH)
        offsets_width = pid_width * BLOCK_WIDTH + tl.arange(0, BLOCK_WIDTH)
        mask = (offsets_batch[:, None] < BATCH) & (offsets_width[None, :] < STREAM_WIDTH)
        grad_output = tl.load(
            GRAD_OUT + offsets_batch[:, None] * STREAM_WIDTH + offsets_width[None, :],
            mask=mask,
            other=0.0,
        )
        grad_output_fp32 = grad_output.to(tl.float32)

        for stream in tl.static_range(0, NUM_STREAMS):
            offset = (
                offsets_batch[:, None] * (NUM_STREAMS * STREAM_WIDTH)
                + stream * STREAM_WIDTH
                + offsets_width[None, :]
            )
            value = tl.load(X + offset, mask=mask, other=0.0)
            sigmoid = tl.sigmoid(tl.load(READ_LOGITS + stream).to(tl.float32))
            factor = sigmoid.to(value.dtype).to(tl.float32)
            tl.store(GRAD_X + offset, factor * grad_output_fp32, mask=mask)

            derivative = sigmoid * (1.0 - sigmoid)
            partial = tl.sum(value.to(tl.float32) * grad_output_fp32) * derivative
            tl.store(GRAD_LOGIT_PARTIALS + stream * NUM_PARTIALS + partial_index, partial)

    @triton.jit
    def _streamwise_write_fwd_kernel(
        RESIDUAL,  # noqa: ANN001
        UPDATE,  # noqa: ANN001
        WRITE_LOGITS,  # noqa: ANN001
        RETENTION_LOGITS,  # noqa: ANN001
        OUT,  # noqa: ANN001
        MAX_FORGET: tl.constexpr,
        BATCH: tl.constexpr,
        STREAM_WIDTH: tl.constexpr,
        NUM_STREAMS: tl.constexpr,
        BLOCK_BATCH: tl.constexpr,
        BLOCK_WIDTH: tl.constexpr,
        HAS_RETENTION: tl.constexpr,
    ) -> None:
        """Write one update to every stream with optional learned retained carry."""

        pid_batch = tl.program_id(0)
        pid_width = tl.program_id(1)
        offsets_batch = pid_batch * BLOCK_BATCH + tl.arange(0, BLOCK_BATCH)
        offsets_width = pid_width * BLOCK_WIDTH + tl.arange(0, BLOCK_WIDTH)
        mask = (offsets_batch[:, None] < BATCH) & (offsets_width[None, :] < STREAM_WIDTH)
        update = tl.load(
            UPDATE + offsets_batch[:, None] * STREAM_WIDTH + offsets_width[None, :],
            mask=mask,
            other=0.0,
        )
        update_fp32 = update.to(tl.float32)

        for stream in tl.static_range(0, NUM_STREAMS):
            offset = (
                offsets_batch[:, None] * (NUM_STREAMS * STREAM_WIDTH)
                + stream * STREAM_WIDTH
                + offsets_width[None, :]
            )
            residual = tl.load(RESIDUAL + offset, mask=mask, other=0.0)
            write = 2.0 * tl.sigmoid(tl.load(WRITE_LOGITS + stream).to(tl.float32))
            write = write.to(update.dtype).to(tl.float32)
            output = write * update_fp32
            if HAS_RETENTION:
                forget = tl.sigmoid(-tl.load(RETENTION_LOGITS + stream).to(tl.float32))
                retention = (1.0 - MAX_FORGET * forget).to(residual.dtype).to(tl.float32)
                output += retention * residual.to(tl.float32)
            else:
                output += residual.to(tl.float32)
            tl.store(OUT + offset, output, mask=mask)

    @triton.jit
    def _streamwise_write_bwd_kernel(
        RESIDUAL,  # noqa: ANN001
        UPDATE,  # noqa: ANN001
        GRAD_OUT,  # noqa: ANN001
        WRITE_LOGITS,  # noqa: ANN001
        RETENTION_LOGITS,  # noqa: ANN001
        GRAD_RESIDUAL,  # noqa: ANN001
        GRAD_UPDATE,  # noqa: ANN001
        GRAD_WRITE_PARTIALS,  # noqa: ANN001
        GRAD_RETENTION_PARTIALS,  # noqa: ANN001
        MAX_FORGET: tl.constexpr,
        BATCH: tl.constexpr,
        STREAM_WIDTH: tl.constexpr,
        NUM_STREAMS: tl.constexpr,
        NUM_WIDTH_BLOCKS: tl.constexpr,
        NUM_PARTIALS: tl.constexpr,
        BLOCK_BATCH: tl.constexpr,
        BLOCK_WIDTH: tl.constexpr,
        HAS_RETENTION: tl.constexpr,
    ) -> None:
        """Compute writeback activation gradients and controller partials in one pass."""

        pid_batch = tl.program_id(0)
        pid_width = tl.program_id(1)
        partial_index = pid_batch * NUM_WIDTH_BLOCKS + pid_width
        offsets_batch = pid_batch * BLOCK_BATCH + tl.arange(0, BLOCK_BATCH)
        offsets_width = pid_width * BLOCK_WIDTH + tl.arange(0, BLOCK_WIDTH)
        mask = (offsets_batch[:, None] < BATCH) & (offsets_width[None, :] < STREAM_WIDTH)
        update = tl.load(
            UPDATE + offsets_batch[:, None] * STREAM_WIDTH + offsets_width[None, :],
            mask=mask,
            other=0.0,
        )
        update_fp32 = update.to(tl.float32)
        grad_update = tl.zeros((BLOCK_BATCH, BLOCK_WIDTH), tl.float32)

        for stream in tl.static_range(0, NUM_STREAMS):
            offset = (
                offsets_batch[:, None] * (NUM_STREAMS * STREAM_WIDTH)
                + stream * STREAM_WIDTH
                + offsets_width[None, :]
            )
            grad_output = tl.load(GRAD_OUT + offset, mask=mask, other=0.0)
            grad_output_fp32 = grad_output.to(tl.float32)
            write_sigmoid = tl.sigmoid(tl.load(WRITE_LOGITS + stream).to(tl.float32))
            write = (2.0 * write_sigmoid).to(update.dtype).to(tl.float32)
            grad_update += write * grad_output_fp32

            write_derivative = 2.0 * write_sigmoid * (1.0 - write_sigmoid)
            write_partial = tl.sum(grad_output_fp32 * update_fp32) * write_derivative
            tl.store(GRAD_WRITE_PARTIALS + stream * NUM_PARTIALS + partial_index, write_partial)

            if HAS_RETENTION:
                residual = tl.load(RESIDUAL + offset, mask=mask, other=0.0)
                forget = tl.sigmoid(-tl.load(RETENTION_LOGITS + stream).to(tl.float32))
                retention = (1.0 - MAX_FORGET * forget).to(grad_output.dtype).to(tl.float32)
                tl.store(GRAD_RESIDUAL + offset, retention * grad_output_fp32, mask=mask)
                retention_derivative = MAX_FORGET * forget * (1.0 - forget)
                retention_partial = (
                    tl.sum(residual.to(tl.float32) * grad_output_fp32) * retention_derivative
                )
                tl.store(
                    GRAD_RETENTION_PARTIALS + stream * NUM_PARTIALS + partial_index,
                    retention_partial,
                )

        tl.store(
            GRAD_UPDATE + offsets_batch[:, None] * STREAM_WIDTH + offsets_width[None, :],
            grad_update,
            mask=mask,
        )

    @triton.jit
    def _reduce_streamwise_partials_kernel(
        PARTIALS,  # noqa: ANN001
        SECOND_PARTIALS,  # noqa: ANN001
        GRAD_LOGITS,  # noqa: ANN001
        SECOND_GRAD_LOGITS,  # noqa: ANN001
        NUM_PARTIALS: tl.constexpr,
        BLOCK_PARTIALS: tl.constexpr,
        HAS_SECOND: tl.constexpr,
    ) -> None:
        """Deterministically reduce one contiguous partial vector per stream."""

        stream = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_PARTIALS)
        mask = offsets < NUM_PARTIALS
        values = tl.load(PARTIALS + stream * NUM_PARTIALS + offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        tl.store(GRAD_LOGITS + stream, tl.sum(values))
        if HAS_SECOND:
            second_values = tl.load(
                SECOND_PARTIALS + stream * NUM_PARTIALS + offsets, mask=mask, other=0.0
            ).to(tl.float32)
            tl.store(SECOND_GRAD_LOGITS + stream, tl.sum(second_values))


def _view_streams(tensor: Tensor, num_streams: int, *, tensor_name: str) -> Tensor:
    """View the final dimension as ``num_streams`` contiguous full-width streams."""

    if num_streams <= 0:
        raise ValueError(f"num_streams must be positive, got {num_streams}.")
    if tensor.shape[-1] % num_streams != 0:
        raise ValueError(
            f"{tensor_name} hidden size {tensor.shape[-1]} is not divisible by "
            f"num_streams={num_streams}."
        )
    stream_width = tensor.shape[-1] // num_streams
    return tensor.reshape(*tensor.shape[:-1], num_streams, stream_width)


def _validate_factors(factors: Tensor, *, factor_name: str) -> int:
    """Validate a vector containing one scalar controller per full-width stream."""

    if factors.ndim != 1 or factors.numel() == 0:
        raise ValueError(
            f"{factor_name} must be a non-empty one-dimensional tensor, got "
            f"shape {tuple(factors.shape)}."
        )
    return factors.numel()


def _broadcast_factors(factors: Tensor, streams: Tensor) -> Tensor:
    """Cast and view stream factors for broadcasting over leading and hidden dimensions."""

    shape = (1,) * (streams.ndim - 2) + (factors.numel(), 1)
    return factors.to(device=streams.device, dtype=streams.dtype).view(shape)


def streamwise_read(hidden_states: Tensor, read_factors: Tensor) -> Tensor:
    """Read ``K`` full-width streams into one branch-width activation.

    Given ``hidden_states[..., k, :] = X_k`` and one scalar ``c_k`` per stream,
    this computes ``sum_k c_k X_k``. Native autograd supplies activation and
    factor gradients without constructing a masked slot-mixing matrix.
    """

    num_streams = _validate_factors(read_factors, factor_name="read_factors")
    streams = _view_streams(hidden_states, num_streams, tensor_name="hidden_states")
    factors = read_factors.to(device=hidden_states.device, dtype=hidden_states.dtype)
    return torch.matmul(factors, streams)


def streamwise_writeback(
    residual_stream: Tensor,
    branch_update: Tensor,
    write_factors: Tensor,
    *,
    retention_factors: Tensor | None = None,
) -> Tensor:
    """Write one branch update independently into ``K`` full-width streams.

    For stream ``k``, this computes ``Y_k = gamma_k X_k + w_k U``. Omitting
    ``retention_factors`` gives identity carry, ``gamma_k = 1``. The returned
    tensor owns the only full-width output allocation; no expanded update tensor
    or masked slot map is materialized.
    """

    num_streams = _validate_factors(write_factors, factor_name="write_factors")
    streams = _view_streams(residual_stream, num_streams, tensor_name="residual_stream")
    expected_update_shape = (*residual_stream.shape[:-1], streams.shape[-1])
    if branch_update.shape != expected_update_shape:
        raise ValueError(
            "branch_update must match the residual leading dimensions and full-stream width, "
            f"expected {expected_update_shape}, got {tuple(branch_update.shape)}."
        )
    if branch_update.device != residual_stream.device:
        raise ValueError("branch_update and residual_stream must be on the same device.")
    if branch_update.dtype != residual_stream.dtype:
        raise ValueError("branch_update and residual_stream must have the same dtype.")

    write = _broadcast_factors(write_factors, streams)
    update = branch_update.unsqueeze(-2)
    if retention_factors is None:
        output = torch.addcmul(streams, update, write)
    else:
        retention_streams = _validate_factors(retention_factors, factor_name="retention_factors")
        if retention_streams != num_streams:
            raise ValueError(
                "retention_factors and write_factors must describe the same number of streams, "
                f"got {retention_streams} and {num_streams}."
            )
        output = streams * _broadcast_factors(retention_factors, streams)
        output.addcmul_(update, write)

    return output.flatten(-2)


def _reduce_streamwise_partials(
    partials: Tensor,
    grad_logits: Tensor,
    *,
    second_partials: Tensor | None = None,
    second_grad_logits: Tensor | None = None,
) -> None:
    """Reduce tile-local FP32 controller gradients without rereading activations."""

    num_streams, num_partials = partials.shape
    reduction_block = triton.next_power_of_2(num_partials)
    has_second = second_partials is not None
    if has_second != (second_grad_logits is not None):
        raise ValueError("Second partials and their destination must be provided together.")
    second_partials_arg = partials if second_partials is None else second_partials
    second_grad_logits_arg = grad_logits if second_grad_logits is None else second_grad_logits
    _reduce_streamwise_partials_kernel[(num_streams,)](
        partials,
        second_partials_arg,
        grad_logits,
        second_grad_logits_arg,
        num_partials,
        reduction_block,
        has_second,
        num_warps=8 if reduction_block >= 2048 else 4,
    )


def _streamwise_sigmoid_read_triton(
    hidden_states: Tensor, read_logits: Tensor, num_streams: int
) -> Tensor:
    hidden_flat, leading = _flatten_leading(hidden_states)
    batch = hidden_flat.shape[0]
    stream_width = hidden_flat.shape[1] // num_streams
    output = torch.empty((batch, stream_width), device=hidden_flat.device, dtype=hidden_flat.dtype)
    block_batch, block_width, num_warps = _streamwise_block_config(batch, stream_width)
    grid = (math.ceil(batch / block_batch), math.ceil(stream_width / block_width))
    _streamwise_read_fwd_kernel[grid](
        hidden_flat,
        read_logits,
        output,
        batch,
        stream_width,
        num_streams,
        block_batch,
        block_width,
        num_warps=num_warps,
    )
    return _unflatten_leading(output, leading)


def _streamwise_sigmoid_read_backward_triton(
    hidden_states: Tensor, grad_output: Tensor, read_logits: Tensor, num_streams: int
) -> tuple[Tensor, Tensor]:
    hidden_flat, leading = _flatten_leading(hidden_states)
    grad_output_flat, _ = _flatten_leading(grad_output)
    batch = hidden_flat.shape[0]
    stream_width = hidden_flat.shape[1] // num_streams
    block_batch, block_width, num_warps = _streamwise_block_config(batch, stream_width)
    num_width_blocks = math.ceil(stream_width / block_width)
    num_partials = math.ceil(batch / block_batch) * num_width_blocks
    grad_hidden = torch.empty_like(hidden_flat)
    partials = torch.empty(
        (num_streams, num_partials), device=hidden_flat.device, dtype=torch.float32
    )
    grid = (math.ceil(batch / block_batch), num_width_blocks)
    _streamwise_read_bwd_kernel[grid](
        hidden_flat,
        grad_output_flat,
        read_logits,
        grad_hidden,
        partials,
        batch,
        stream_width,
        num_streams,
        num_width_blocks,
        num_partials,
        block_batch,
        block_width,
        num_warps=num_warps,
    )
    grad_logits = torch.zeros_like(read_logits)
    _reduce_streamwise_partials(partials, grad_logits)
    return _unflatten_leading(grad_hidden, leading), grad_logits


def _streamwise_sigmoid_write_triton(
    residual_stream: Tensor,
    branch_update: Tensor,
    write_logits: Tensor,
    num_streams: int,
    retention_logits: Tensor | None,
    retention_max_forget: float,
) -> Tensor:
    residual_flat, leading = _flatten_leading(residual_stream)
    update_flat, _ = _flatten_leading(branch_update)
    batch = residual_flat.shape[0]
    stream_width = residual_flat.shape[1] // num_streams
    output = torch.empty_like(residual_flat)
    block_batch, block_width, num_warps = _streamwise_block_config(batch, stream_width)
    grid = (math.ceil(batch / block_batch), math.ceil(stream_width / block_width))
    retention_arg = write_logits if retention_logits is None else retention_logits
    _streamwise_write_fwd_kernel[grid](
        residual_flat,
        update_flat,
        write_logits,
        retention_arg,
        output,
        retention_max_forget,
        batch,
        stream_width,
        num_streams,
        block_batch,
        block_width,
        retention_logits is not None,
        num_warps=num_warps,
    )
    return _unflatten_leading(output, leading)


def _streamwise_sigmoid_write_backward_triton(
    residual_stream: Tensor | None,
    branch_update: Tensor,
    grad_output: Tensor,
    write_logits: Tensor,
    num_streams: int,
    retention_logits: Tensor | None,
    retention_max_forget: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
    grad_output_flat, leading = _flatten_leading(grad_output)
    update_flat, _ = _flatten_leading(branch_update)
    residual_flat = (
        grad_output_flat if residual_stream is None else _flatten_leading(residual_stream)[0]
    )
    batch = grad_output_flat.shape[0]
    stream_width = grad_output_flat.shape[1] // num_streams
    block_batch, block_width, num_warps = _streamwise_block_config(batch, stream_width)
    num_width_blocks = math.ceil(stream_width / block_width)
    num_partials = math.ceil(batch / block_batch) * num_width_blocks
    grad_residual = (
        torch.empty_like(grad_output_flat) if retention_logits is not None else grad_output_flat
    )
    grad_update = torch.empty_like(update_flat)
    write_partials = torch.empty(
        (num_streams, num_partials), device=grad_output.device, dtype=torch.float32
    )
    retention_partials = (
        torch.empty_like(write_partials) if retention_logits is not None else write_partials
    )
    retention_arg = write_logits if retention_logits is None else retention_logits
    grid = (math.ceil(batch / block_batch), num_width_blocks)
    _streamwise_write_bwd_kernel[grid](
        residual_flat,
        update_flat,
        grad_output_flat,
        write_logits,
        retention_arg,
        grad_residual,
        grad_update,
        write_partials,
        retention_partials,
        retention_max_forget,
        batch,
        stream_width,
        num_streams,
        num_width_blocks,
        num_partials,
        block_batch,
        block_width,
        retention_logits is not None,
        num_warps=num_warps,
    )
    grad_write_logits = torch.zeros_like(write_logits)
    grad_retention_logits = (
        torch.zeros_like(retention_logits) if retention_logits is not None else None
    )
    _reduce_streamwise_partials(
        write_partials,
        grad_write_logits,
        second_partials=retention_partials if retention_logits is not None else None,
        second_grad_logits=grad_retention_logits,
    )
    return (
        _unflatten_leading(grad_residual, leading),
        _unflatten_leading(grad_update, leading),
        grad_write_logits,
        grad_retention_logits,
    )


class _StreamwiseSigmoidRead(torch.autograd.Function):
    """Autograd wrapper for direct raw-logit streamwise Triton read."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx, hidden_states: Tensor, read_logits: Tensor, num_streams: int
    ) -> Tensor:
        """Run the fused streamwise read and save tensors for backward."""

        ctx.save_for_backward(hidden_states, read_logits)
        ctx.num_streams = num_streams
        return _streamwise_sigmoid_read_triton(hidden_states, read_logits, num_streams)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor | None, ...]:  # type: ignore[override]
        """Compute activation and read-controller gradients."""

        hidden_states, read_logits = ctx.saved_tensors
        grad_hidden, grad_logits = _streamwise_sigmoid_read_backward_triton(
            hidden_states, grad_output.contiguous(), read_logits, ctx.num_streams
        )
        return grad_hidden, grad_logits, None


class _StreamwiseSigmoidWriteback(torch.autograd.Function):
    """Autograd wrapper for direct raw-logit streamwise Triton writeback."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        residual_stream: Tensor,
        branch_update: Tensor,
        write_logits: Tensor,
        retention_logits: Tensor | None,
        num_streams: int,
        retention_max_forget: float,
    ) -> Tensor:
        """Run fused streamwise writeback and save tensors for backward."""

        ctx.has_retention = retention_logits is not None
        ctx.num_streams = num_streams
        ctx.retention_max_forget = retention_max_forget
        if retention_logits is None:
            ctx.save_for_backward(branch_update, write_logits)
        else:
            ctx.save_for_backward(residual_stream, branch_update, write_logits, retention_logits)
        return _streamwise_sigmoid_write_triton(
            residual_stream,
            branch_update,
            write_logits,
            num_streams,
            retention_logits,
            retention_max_forget,
        )

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor | None, ...]:  # type: ignore[override]
        """Compute residual, update, map, and retention gradients."""

        if ctx.has_retention:
            residual_stream, branch_update, write_logits, retention_logits = ctx.saved_tensors
        else:
            branch_update, write_logits = ctx.saved_tensors
            residual_stream = None
            retention_logits = None
        gradients = _streamwise_sigmoid_write_backward_triton(
            residual_stream,
            branch_update,
            grad_output.contiguous(),
            write_logits,
            ctx.num_streams,
            retention_logits,
            ctx.retention_max_forget,
        )
        grad_residual, grad_update, grad_write_logits, grad_retention_logits = gradients
        return grad_residual, grad_update, grad_write_logits, grad_retention_logits, None, None


def streamwise_sigmoid_read(hidden_states: Tensor, read_logits: Tensor, num_streams: int) -> Tensor:
    """Read full-width streams from raw padded logits with fused CUDA dispatch."""

    stream_width = _validate_raw_read_inputs(hidden_states, read_logits, num_streams)
    if _can_use_streamwise_triton(hidden_states, read_logits, num_streams, stream_width):
        return _StreamwiseSigmoidRead.apply(hidden_states, read_logits, num_streams)

    read_factors = torch.sigmoid(read_logits[:num_streams].float())
    return streamwise_read(hidden_states, read_factors)


def streamwise_sigmoid_writeback(
    residual_stream: Tensor,
    branch_update: Tensor,
    write_logits: Tensor,
    num_streams: int,
    *,
    retention_logits: Tensor | None = None,
    retention_max_forget: float = 0.0,
) -> Tensor:
    """Write full-width streams from raw padded logits with fused CUDA dispatch."""

    stream_width = _validate_raw_write_inputs(
        residual_stream, branch_update, write_logits, num_streams, retention_logits
    )
    if retention_logits is not None and not 0.0 < retention_max_forget <= 1.0:
        raise ValueError(
            "retention_max_forget must be in (0, 1] when retention is enabled, got "
            f"{retention_max_forget}."
        )
    supports_triton = _can_use_streamwise_triton(
        residual_stream, write_logits, num_streams, stream_width
    ) and _can_use_streamwise_triton(
        residual_stream,
        write_logits if retention_logits is None else retention_logits,
        num_streams,
        stream_width,
    )
    if supports_triton and branch_update.is_contiguous():
        return _StreamwiseSigmoidWriteback.apply(
            residual_stream,
            branch_update,
            write_logits,
            retention_logits,
            num_streams,
            retention_max_forget,
        )

    write_factors = 2.0 * torch.sigmoid(write_logits[:num_streams].float())
    retention_factors = None
    if retention_logits is not None:
        active_retention_logits = retention_logits[:num_streams].float()
        retention_factors = 1.0 - retention_max_forget * torch.sigmoid(-active_retention_logits)
    return streamwise_writeback(
        residual_stream, branch_update, write_factors, retention_factors=retention_factors
    )
