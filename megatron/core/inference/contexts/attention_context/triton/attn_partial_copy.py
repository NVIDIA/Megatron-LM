import triton
import triton.language as tl


@triton.jit
def _attn_partial_copy_kernel(
    # Input tensors
    INPUT_TENSOR,
    # Output tensor
    OUTPUT_TENSOR,
    # Tensor metadata
    DEVICE_DC,
    INPUT_BATCH_SIZE: tl.constexpr,
    OUTPUT_BATCH_SIZE: tl.constexpr,
    ROW_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for partial copying from input tensor to output tensor.
    One block processes one row.

    Logic:
    - Copy input_tensor[device_dc:] to output_tensor[0:input_batch_size-device_dc]
    - Remaining output tensor elements are left unchanged
    """

    # Get program ID - each program handles one batch element (one row)
    pid = tl.program_id(0)
    device_dc = tl.load(DEVICE_DC)

    # Calculate copy size (how many rows to copy from input starting at device_dc)
    copy_size = INPUT_BATCH_SIZE - device_dc

    # Only process if this program is within the copy range
    if pid < copy_size and pid < OUTPUT_BATCH_SIZE:
        # Calculate source index in input tensor (starting from device_dc)
        input_idx = device_dc + pid

        # Bounds check for input tensor
        if input_idx < INPUT_BATCH_SIZE:
            row_offsets = tl.arange(0, BLOCK_SIZE)
            row_mask = row_offsets < ROW_SIZE

            # Source pointer in input tensor
            input_ptr = INPUT_TENSOR + input_idx * ROW_SIZE + row_offsets
            # Destination pointer in output tensor (starting from beginning)
            output_ptr = OUTPUT_TENSOR + pid * ROW_SIZE + row_offsets

            # Load entire row from input tensor
            input_data = tl.load(input_ptr, mask=row_mask, other=0.0)

            # Store entire row to output tensor
            tl.store(output_ptr, input_data, mask=row_mask)


def attn_partial_copy_triton(
    input_tensor, output_tensor, device_dc, check_bounds: bool = False
) -> None:
    """
    Partial copy from input tensor to output tensor using Triton.
    Copies from device_dc to the end of input tensor to the beginning of output tensor.

    Args:
        input_tensor: Input tensor to copy from, shape [input_batch, ...]
        output_tensor: Output tensor to write to, shape [output_batch, ...]
        device_dc: Tensor with decode count in first element, shape [2] (e.g., [dc_count, pf_count])
        check_bounds: Whether to perform bounds checking

    The operation performed is:
        copy_size = input_tensor.shape[0] - device_dc
        output_tensor[0:copy_size] = input_tensor[device_dc:device_dc+copy_size]
        output_tensor[copy_size:] remains unchanged
    """

    # Validate inputs
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
        # Check tensor shapes are compatible
        assert (
            input_tensor.ndim == output_tensor.ndim
        ), "Input and output tensors must have the same number of dimensions"

        for i in range(1, input_tensor.ndim):
            assert (
                input_tensor.shape[i] == output_tensor.shape[i]
            ), f"Dimension {i} must match between input and output tensors"

        # Check device_dc is valid
        device_dc_val = device_dc[0].item()
        assert (
            0 <= device_dc_val <= input_tensor.shape[0]
        ), f"device_dc must be between 0 and input_tensor.shape[0]"

        # Check that we have enough space in output tensor
        copy_size = input_tensor.shape[0] - device_dc_val
        assert (
            copy_size <= output_tensor.shape[0]
        ), f"Copy size ({copy_size}) exceeds output_tensor.shape[0] ({output_tensor.shape[0]})"

    # Get tensor dimensions
    input_batch_size = input_tensor.shape[0]
    output_batch_size = output_tensor.shape[0]

    # Calculate row size (everything after batch dimension)
    if input_tensor.ndim == 1:
        row_size = 1
    else:
        row_size = 1
        for dim in input_tensor.shape[1:]:
            row_size *= dim

    # Calculate block size (round up to next power of 2 for efficiency)
    block_size = triton.next_power_of_2(row_size)

    # Calculate how many rows we need to copy
    copy_size = input_batch_size

    # One block per row - grid size is the number of rows to copy
    grid = (copy_size,) if copy_size > 0 else (1,)

    # Launch kernel only if there's something to copy
    if copy_size > 0:
        _attn_partial_copy_kernel[grid](
            input_tensor,
            output_tensor,
            DEVICE_DC=device_dc,
            INPUT_BATCH_SIZE=input_batch_size,
            OUTPUT_BATCH_SIZE=output_batch_size,
            ROW_SIZE=row_size,
            BLOCK_SIZE=block_size,
        )
