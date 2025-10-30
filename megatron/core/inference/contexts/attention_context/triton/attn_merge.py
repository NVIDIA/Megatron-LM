import triton
import triton.language as tl


@triton.jit
def _attn_merge_kernel(
    # Input tensors
    DECODE_TENSOR,
    PREFILL_TENSOR,
    # Output tensor
    OUTPUT_TENSOR,
    # Tensor metadata
    DEVICE_DC,
    PREFILL_BATCH_SIZE: tl.constexpr,
    ROW_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    OUTPUT_BATCH_SIZE: tl.constexpr,
    PF_USEFUL_FROM_BEGINNING: tl.constexpr,
):
    """
    Triton kernel to merge decode and prefill tensors into output tensor.
    One block processes one row.

    Logic:
    - output[:device_dc] = decode_tensor[:device_dc]
    - If pf_useful_from_beginning:
        output[device_dc:device_dc+prefill_batch_size] = prefill_tensor[0:prefill_batch_size]
    - Else:
        output[device_dc:device_dc+prefill_batch_size] = prefill_tensor[device_dc:device_dc+prefill_batch_size]
    - output[device_dc+prefill_batch_size:] remains unchanged
    """

    # Get program ID - each program handles one batch element (one row)
    pid = tl.program_id(0)
    device_dc = tl.load(DEVICE_DC)
    if pid < device_dc:
        # Copy from decode_tensor[:device_dc] to output[:device_dc]
        # Copy entire row at once
        row_offsets = tl.arange(0, BLOCK_SIZE)
        row_mask = row_offsets < ROW_SIZE

        decode_ptr = DECODE_TENSOR + pid * ROW_SIZE + row_offsets
        output_ptr = OUTPUT_TENSOR + pid * ROW_SIZE + row_offsets

        # Load entire row from decode tensor
        decode_data = tl.load(decode_ptr, mask=row_mask, other=0.0)

        # Store entire row to output tensor
        tl.store(output_ptr, decode_data, mask=row_mask)

    elif pid < device_dc + PREFILL_BATCH_SIZE and pid < OUTPUT_BATCH_SIZE:
        # Copy from prefill_tensor to output[device_dc:]
        if PF_USEFUL_FROM_BEGINNING:
            # Mode 1: Copy prefill_tensor from beginning
            # output[device_dc:device_dc+prefill_batch] = prefill_tensor[0:prefill_batch]
            prefill_idx = pid - device_dc
        else:
            # Mode 2: Copy prefill_tensor from decode_count to end
            # output[device_dc:device_dc+prefill_batch] = prefill_tensor[device_dc:device_dc+prefill_batch]
            prefill_idx = pid

        # Bounds check for prefill tensor
        if prefill_idx < PREFILL_BATCH_SIZE:
            row_offsets = tl.arange(0, BLOCK_SIZE)
            row_mask = row_offsets < ROW_SIZE

            prefill_ptr = PREFILL_TENSOR + prefill_idx * ROW_SIZE + row_offsets
            output_ptr = OUTPUT_TENSOR + pid * ROW_SIZE + row_offsets

            # Load entire row from prefill tensor
            prefill_data = tl.load(prefill_ptr, mask=row_mask, other=0.0)

            # Store entire row to output tensor
            tl.store(output_ptr, prefill_data, mask=row_mask)

    # For pid >= DEVICE_DC + PREFILL_BATCH_SIZE, do nothing (leave unchanged)


def attn_merge_triton(
    decode_tensor,
    prefill_tensor,
    output_tensor,
    device_dc,
    pf_useful_from_beginning: bool,
    check_bounds: bool = False,
) -> None:
    """
    Merge decode and prefill tensors into output tensor using Triton.
    One block processes one row.

    Args:
        decode_tensor: Tensor containing decode data, shape [decode_batch, ...]
        prefill_tensor: Tensor containing prefill data, shape [prefill_batch, ...]
        output_tensor: Output tensor to write to, shape [output_batch, ...]
        device_dc: Tensor with decode count in first element, shape [2] (e.g., [dc_count, pf_count])
        pf_useful_from_beginning: If True, copy prefill from beginning; else from device_dc
        check_bounds: If True, perform bounds checking on inputs

    The operation performed is:
        output[:device_dc] = decode_tensor[:device_dc]
        If pf_useful_from_beginning:
            output[device_dc:device_dc+prefill_batch] = prefill_tensor[0:prefill_batch]
        Else:
            output[device_dc:device_dc+prefill_batch] = prefill_tensor[device_dc:device_dc+prefill_batch]
        output[device_dc+prefill_batch:] remains unchanged
    """

    # Validate inputs
    assert decode_tensor.device == prefill_tensor.device == output_tensor.device, \
        "All tensors must be on the same device"
    assert decode_tensor.dtype == prefill_tensor.dtype == output_tensor.dtype, \
        "All tensors must have the same dtype"
    assert decode_tensor.is_contiguous() and prefill_tensor.is_contiguous() and output_tensor.is_contiguous(), \
        "All tensors must be contiguous"

    if check_bounds:
        # Check tensor shapes are compatible
        assert decode_tensor.ndim == prefill_tensor.ndim == output_tensor.ndim, \
            "All tensors must have the same number of dimensions"

        for i in range(1, decode_tensor.ndim):
            assert decode_tensor.shape[i] == prefill_tensor.shape[i] == output_tensor.shape[i], \
                f"Dimension {i} must match across all tensors"

        assert output_tensor.shape[0] >= decode_tensor.shape[0], \
            f"Output tensor batch size must be >= decode_tensor.shape[0]"

        # Check device_dc is valid
        device_dc_val = device_dc[0].item()
        assert 0 <= device_dc_val <= decode_tensor.shape[0], \
            f"device_dc must be between 0 and decode_tensor.shape[0]"

        if not pf_useful_from_beginning:
            assert prefill_tensor.shape[0] >= device_dc_val, \
                f"prefill_tensor.shape[0] must be >= device_dc when pf_useful_from_beginning=False"

    # Get tensor dimensions
    prefill_batch_size = prefill_tensor.shape[0]
    output_batch_size = output_tensor.shape[0]
    # Calculate row size (everything after batch dimension)
    if decode_tensor.ndim == 1:
        row_size = 1
    else:
        row_size = 1
        for dim in decode_tensor.shape[1:]:
            row_size *= dim

    # Calculate block size (round up to next power of 2 for efficiency)
    block_size = triton.next_power_of_2(row_size)

    # One block per row - grid size is just the number of rows to process
    num_rows_to_process = output_batch_size
    grid = (num_rows_to_process,)

    # Launch kernel
    _attn_merge_kernel[grid](
        decode_tensor,
        prefill_tensor,
        output_tensor,
        DEVICE_DC=device_dc,
        PREFILL_BATCH_SIZE=prefill_batch_size,
        ROW_SIZE=row_size,
        BLOCK_SIZE=block_size,
        OUTPUT_BATCH_SIZE=output_batch_size,
        PF_USEFUL_FROM_BEGINNING=pf_useful_from_beginning,
    )
