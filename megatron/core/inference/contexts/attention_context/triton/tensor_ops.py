# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from typing import Optional

import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore


@triton.jit
def _tensor_get_slice_after_kernel(
    INPUT_TENSOR,
    OUTPUT_TENSOR,
    POS_ON_DEVICE,
    INPUT_BATCH_SIZE: tl.constexpr,
    OUTPUT_BATCH_SIZE: tl.constexpr,
    ROW_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Kernel to copy rows from INPUT_TENSOR[pos_on_device:] into OUTPUT_TENSOR."""

    pid = tl.program_id(0)
    pos_on_device = tl.load(POS_ON_DEVICE)
    copy_size = INPUT_BATCH_SIZE - pos_on_device

    if pid < copy_size and pid < OUTPUT_BATCH_SIZE:
        input_idx = pos_on_device + pid

        if input_idx < INPUT_BATCH_SIZE:
            row_offsets = tl.arange(0, BLOCK_SIZE)
            row_mask = row_offsets < ROW_SIZE

            input_ptr = INPUT_TENSOR + input_idx * ROW_SIZE + row_offsets
            output_ptr = OUTPUT_TENSOR + pid * ROW_SIZE + row_offsets

            input_data = tl.load(input_ptr, mask=row_mask, other=0.0)
            tl.store(output_ptr, input_data, mask=row_mask)


@triton.jit
def _tensor_merge_kernel(
    TENSOR_A,
    TENSOR_B,
    OUTPUT_TENSOR,
    POS_ON_DEVICE,
    TENSOR_B_BATCH_SIZE: tl.constexpr,
    ROW_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    OUTPUT_BATCH_SIZE: tl.constexpr,
    IS_INPLACE: tl.constexpr,
):
    """
    Kernel to merge rows from tensor_a and tensor_b into output_tensor.

    - output[:pos_on_device] = tensor_a[:pos_on_device]
    - output[pos_on_device:pos_on_device + tensor_b_batch] = tensor_b[:tensor_b_batch]
    """

    pid = tl.program_id(0)
    pos_on_device = tl.load(POS_ON_DEVICE)

    if pid < pos_on_device:
        if not IS_INPLACE:
            row_offsets = tl.arange(0, BLOCK_SIZE)
            row_mask = row_offsets < ROW_SIZE

            tensor_a_ptr = TENSOR_A + pid * ROW_SIZE + row_offsets
            output_ptr = OUTPUT_TENSOR + pid * ROW_SIZE + row_offsets

            tensor_a_data = tl.load(tensor_a_ptr, mask=row_mask, other=0.0)
            tl.store(output_ptr, tensor_a_data, mask=row_mask)

    elif pid < pos_on_device + TENSOR_B_BATCH_SIZE and pid < OUTPUT_BATCH_SIZE:
        tensor_b_idx = pid - pos_on_device

        if tensor_b_idx < TENSOR_B_BATCH_SIZE:
            row_offsets = tl.arange(0, BLOCK_SIZE)
            row_mask = row_offsets < ROW_SIZE

            tensor_b_ptr = TENSOR_B + tensor_b_idx * ROW_SIZE + row_offsets
            output_ptr = OUTPUT_TENSOR + pid * ROW_SIZE + row_offsets

            tensor_b_data = tl.load(tensor_b_ptr, mask=row_mask, other=0.0)
            tl.store(output_ptr, tensor_b_data, mask=row_mask)


@triton.jit
def _tensor_masked_update_kernel_2d(
    STATES_PTR,
    IDX_PTR,
    NEW_STATES_PTR,
    stride_state_b,
    stride_state_d0,
    stride_new_b,
    stride_new_d0,
    ROW_SIZE,
    BLOCK_SIZE: tl.constexpr,
):
    """Kernel to update values in a 2D states tensor using a mask."""
    pid_batch = tl.program_id(0).to(tl.int64)
    pid_row_chunk = tl.program_id(1).to(tl.int64)

    target_idx = tl.load(IDX_PTR + pid_batch)
    if target_idx == -1:
        return

    row_start_offset = pid_row_chunk * BLOCK_SIZE
    row_offsets = row_start_offset + tl.arange(0, BLOCK_SIZE)
    mask = row_offsets < ROW_SIZE

    # 2D Calculation: base + batch * stride0 + col * stride1
    dst_ptr = (
        STATES_PTR
        + (target_idx.to(tl.int64) * stride_state_b)
        + (row_offsets.to(tl.int64) * stride_state_d0)
    )
    src_ptr = (
        NEW_STATES_PTR
        + (pid_batch * stride_new_b.to(tl.int64))
        + (row_offsets.to(tl.int64) * stride_new_d0)
    )

    val = tl.load(src_ptr, mask=mask)
    tl.store(dst_ptr, val, mask=mask)


@triton.jit
def _tensor_masked_update_kernel_3d(
    STATES_PTR,
    IDX_PTR,
    NEW_STATES_PTR,
    stride_state_b,
    stride_state_d0,
    stride_state_d1,
    stride_new_b,
    stride_new_d0,
    stride_new_d1,
    SIZE_D0,
    SIZE_D1,  # Dimensions of the non-batch axes
    ROW_SIZE,  # Total elements per batch item (D0 * D1)
    BLOCK_SIZE: tl.constexpr,
):
    """Kernel to update values in a 3D states tensor using a mask."""
    pid_batch = tl.program_id(0).to(tl.int64)
    pid_row_chunk = tl.program_id(1).to(tl.int64)

    target_idx = tl.load(IDX_PTR + pid_batch)
    if target_idx == -1:
        return

    # Linear index within the "row" (flattened 3D volume)
    row_start_offset = pid_row_chunk * BLOCK_SIZE
    flat_offsets = row_start_offset + tl.arange(0, BLOCK_SIZE)
    mask = flat_offsets < ROW_SIZE

    # Reconstruct 3D coordinates from linear index
    # Given shape (batch, D0, D1)
    # idx_d1 = flat_idx % D1
    # idx_d0 = flat_idx // D1
    idx_d1 = flat_offsets % SIZE_D1.to(tl.int64)
    idx_d0 = flat_offsets // SIZE_D1.to(tl.int64)

    # Calculate pointers using specific strides
    dst_offset = (
        (target_idx.to(tl.int64) * stride_state_b.to(tl.int64))
        + (idx_d0 * stride_state_d0)
        + (idx_d1 * stride_state_d1)
    )

    src_offset = (
        (pid_batch * stride_new_b.to(tl.int64))
        + (idx_d0 * stride_new_d0)
        + (idx_d1 * stride_new_d1)
    )

    dst_ptr = STATES_PTR + dst_offset
    src_ptr = NEW_STATES_PTR + src_offset

    val = tl.load(src_ptr, mask=mask)
    tl.store(dst_ptr, val, mask=mask)


@triton.jit
def _tensor_masked_update_kernel_4d(
    STATES_PTR,
    IDX_PTR,
    NEW_STATES_PTR,
    stride_state_b,
    stride_state_d0,
    stride_state_d1,
    stride_state_d2,
    stride_new_b,
    stride_new_d0,
    stride_new_d1,
    stride_new_d2,
    SIZE_D0,
    SIZE_D1,
    SIZE_D2,  # Dimensions (C, H, W)
    ROW_SIZE,  # Total elements (C * H * W)
    BLOCK_SIZE: tl.constexpr,
):
    """Kernel to update values in a 4D states tensor using a mask."""
    pid_batch = tl.program_id(0).to(tl.int64)
    pid_row_chunk = tl.program_id(1).to(tl.int64)

    target_idx = tl.load(IDX_PTR + pid_batch)
    if target_idx == -1:
        return

    # Linear index
    row_start_offset = pid_row_chunk * BLOCK_SIZE
    flat_offsets = row_start_offset + tl.arange(0, BLOCK_SIZE)
    mask = flat_offsets < ROW_SIZE

    # Reconstruct 4D coordinates from linear index
    # Given shape (batch, D0, D1, D2)
    # idx_d2 = flat % D2
    # temp   = flat // D2
    # idx_d1 = temp % D1
    # idx_d0 = temp // D1

    idx_d2 = flat_offsets % SIZE_D2.to(tl.int64)
    temp = flat_offsets // SIZE_D2.to(tl.int64)
    idx_d1 = temp % SIZE_D1.to(tl.int64)
    idx_d0 = temp // SIZE_D1.to(tl.int64)

    # Calculate pointers using specific strides
    dst_offset = (
        (target_idx.to(tl.int64) * stride_state_b.to(tl.int64))
        + (idx_d0 * stride_state_d0)
        + (idx_d1 * stride_state_d1)
        + (idx_d2 * stride_state_d2)
    )

    src_offset = (
        (pid_batch * stride_new_b.to(tl.int64))
        + (idx_d0 * stride_new_d0)
        + (idx_d1 * stride_new_d1)
        + (idx_d2 * stride_new_d2)
    )

    dst_ptr = STATES_PTR + dst_offset
    src_ptr = NEW_STATES_PTR + src_offset

    val = tl.load(src_ptr, mask=mask)
    tl.store(dst_ptr, val, mask=mask)


def _compute_row_size(tensor):
    if tensor.ndim == 1:
        return 1

    row_size = 1
    for dim in tensor.shape[1:]:
        row_size *= dim
    return row_size


def tensor_get_slice_after(input_tensor, output_tensor, pos_on_device, check_bounds: bool = False):
    """
    Copy from input_tensor[pos_on_device:] to output_tensor[:copy_size].
    """

    assert (
        input_tensor.device == output_tensor.device
    ), "Input and output tensors must be on the same device"
    assert (
        input_tensor.dtype == output_tensor.dtype
    ), "Input and output tensors must have the same dtype"
    assert (
        input_tensor.is_contiguous() and output_tensor.is_contiguous()
    ), "Input and output tensors must be contiguous"

    if check_bounds:
        assert (
            input_tensor.ndim == output_tensor.ndim
        ), "Input and output tensors must have the same number of dimensions"

        for i in range(1, input_tensor.ndim):
            assert (
                input_tensor.shape[i] == output_tensor.shape[i]
            ), f"Dimension {i} must match between input and output tensors"

        pos_on_device_val = pos_on_device[0].item()
        assert (
            0 <= pos_on_device_val <= input_tensor.shape[0]
        ), "pos_on_device must be between 0 and input_tensor.shape[0]"

        copy_size = input_tensor.shape[0] - pos_on_device_val
        assert (
            copy_size <= output_tensor.shape[0]
        ), f"Copy size ({copy_size}) exceeds output_tensor batch size ({output_tensor.shape[0]})"

    input_batch_size = input_tensor.shape[0]
    output_batch_size = output_tensor.shape[0]

    row_size = _compute_row_size(input_tensor)
    block_size = triton.next_power_of_2(row_size)

    grid = (input_batch_size,) if input_batch_size > 0 else (1,)

    if input_batch_size > 0:
        _tensor_get_slice_after_kernel[grid](
            input_tensor,
            output_tensor,
            POS_ON_DEVICE=pos_on_device,
            INPUT_BATCH_SIZE=input_batch_size,
            OUTPUT_BATCH_SIZE=output_batch_size,
            ROW_SIZE=row_size,
            BLOCK_SIZE=block_size,
        )


def tensor_merge(
    tensor_a: torch.Tensor,
    tensor_b: torch.Tensor,
    pos_on_device: torch.Tensor,
    output_tensor: Optional[torch.Tensor] = None,
    check_bounds: bool = False,
):
    """
    Merge tensor_a and tensor_b.

    If output_tensor is None, the operation is performed in-place on tensor_a.
    """

    is_inplace = False
    if output_tensor is None:
        output_tensor = tensor_a
        is_inplace = True

    assert (
        tensor_a.device == tensor_b.device == output_tensor.device
    ), "All tensors must be on the same device"
    assert (
        tensor_a.dtype == tensor_b.dtype == output_tensor.dtype
    ), "All tensors must have the same dtype"
    assert (
        tensor_a.is_contiguous() and tensor_b.is_contiguous() and output_tensor.is_contiguous()
    ), "All tensors must be contiguous"

    if check_bounds:
        assert (
            tensor_a.ndim == tensor_b.ndim == output_tensor.ndim
        ), "All tensors must have the same number of dimensions"

        for i in range(1, tensor_a.ndim):
            assert (
                tensor_a.shape[i] == tensor_b.shape[i] == output_tensor.shape[i]
            ), f"Dimension {i} must match across all tensors"

        assert (
            output_tensor.shape[0] >= tensor_a.shape[0]
        ), "output_tensor batch size must be >= tensor_a batch size"

        pos_on_device_val = pos_on_device[0].item()
        assert (
            0 <= pos_on_device_val <= tensor_a.shape[0]
        ), "pos_on_device must be between 0 and tensor_a batch size"

    tensor_b_batch_size = tensor_b.shape[0]
    output_batch_size = output_tensor.shape[0]

    row_size = _compute_row_size(tensor_a)
    block_size = triton.next_power_of_2(row_size)

    grid = (output_batch_size,)

    _tensor_merge_kernel[grid](
        tensor_a,
        tensor_b,
        output_tensor,
        POS_ON_DEVICE=pos_on_device,
        TENSOR_B_BATCH_SIZE=tensor_b_batch_size,
        ROW_SIZE=row_size,
        BLOCK_SIZE=block_size,
        OUTPUT_BATCH_SIZE=output_batch_size,
        IS_INPLACE=is_inplace,
    )


def tensor_masked_update(states: torch.Tensor, idx: torch.Tensor, new_states: torch.Tensor):
    """
    Update `states` to `new_states` at `idx`, but ignore any -1 values in `idx`.
    Works for 2D, 3D, or 4D tensors.

    Args:
        states: (N, ...) - Destination tensor (2D, 3D, or 4D)
        idx: (B,) - Indices to update. -1 means skip.
        new_states: (B, ...) - Source tensor. Must match states shape[1:]
    """
    assert states.is_cuda and idx.is_cuda and new_states.is_cuda
    assert idx.ndim == 1
    assert states.shape[1:] == new_states.shape[1:], "State dimensions must match"

    ndim = states.ndim
    assert ndim in [2, 3, 4], "Only 2D, 3D, and 4D tensors are supported"

    n_updates = idx.shape[0]

    row_size = 1
    for dim in states.shape[1:]:
        row_size *= dim

    BLOCK_SIZE = 1024
    grid = lambda meta: (n_updates, triton.cdiv(row_size, meta["BLOCK_SIZE"]))

    if ndim == 2:
        _tensor_masked_update_kernel_2d[grid](
            STATES_PTR=states,
            IDX_PTR=idx,
            NEW_STATES_PTR=new_states,
            stride_state_b=states.stride(0),
            stride_state_d0=states.stride(1),
            stride_new_b=new_states.stride(0),
            stride_new_d0=new_states.stride(1),
            ROW_SIZE=row_size,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    elif ndim == 3:
        # Shapes: (N, D0, D1)
        _tensor_masked_update_kernel_3d[grid](
            STATES_PTR=states,
            IDX_PTR=idx,
            NEW_STATES_PTR=new_states,
            # Strides
            stride_state_b=states.stride(0),
            stride_state_d0=states.stride(1),
            stride_state_d1=states.stride(2),
            stride_new_b=new_states.stride(0),
            stride_new_d0=new_states.stride(1),
            stride_new_d1=new_states.stride(2),
            # Dims
            SIZE_D0=states.shape[1],
            SIZE_D1=states.shape[2],
            ROW_SIZE=row_size,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    elif ndim == 4:
        # Shapes: (N, D0, D1, D2)
        _tensor_masked_update_kernel_4d[grid](
            STATES_PTR=states,
            IDX_PTR=idx,
            NEW_STATES_PTR=new_states,
            # Strides
            stride_state_b=states.stride(0),
            stride_state_d0=states.stride(1),
            stride_state_d1=states.stride(2),
            stride_state_d2=states.stride(3),
            stride_new_b=new_states.stride(0),
            stride_new_d0=new_states.stride(1),
            stride_new_d1=new_states.stride(2),
            stride_new_d2=new_states.stride(3),
            # Dims
            SIZE_D0=states.shape[1],
            SIZE_D1=states.shape[2],
            SIZE_D2=states.shape[3],
            ROW_SIZE=row_size,
            BLOCK_SIZE=BLOCK_SIZE,
        )
