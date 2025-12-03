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
):
    """
    Kernel to merge rows from tensor_a and tensor_b into output_tensor.

    - output[:pos_on_device] = tensor_a[:pos_on_device]
    - output[pos_on_device:pos_on_device + tensor_b_batch] = tensor_b[:tensor_b_batch]
    """

    pid = tl.program_id(0)
    pos_on_device = tl.load(POS_ON_DEVICE)

    if pid < pos_on_device:
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


def tensor_merge(tensor_a, tensor_b, output_tensor, pos_on_device, check_bounds: bool = False):
    """
    Merge tensor_a and tensor_b into output_tensor.

    The operation performed is:
        output[:pos_on_device] = tensor_a[:pos_on_device]
        output[pos_on_device:pos_on_device + tensor_b_batch] = tensor_b[:tensor_b_batch]
    Remaining rows in output_tensor are left unchanged.
    """

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
    )
