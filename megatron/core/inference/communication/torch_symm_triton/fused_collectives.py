# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import torch

from .barrier import symm_mem_sync
from .kernel_config import SYMM_MEM_KERNEL_CONFIG
from .multimem_asm import add_v8_bf16_from_u32, asm_rsqrt, ld_128, st_128
from .utils import sync_threads

try:
    import triton
    import triton.language as tl
except ImportError:
    from unittest.mock import MagicMock

    from megatron.core.utils import null_decorator

    triton = MagicMock()
    tl = MagicMock()
    triton.jit = null_decorator


@triton.jit
def unpack_bf16x2(x, mask):
    """
    Unpack x, which is in bf16x2 packed format stored in uint32,
    into two float32 tensors representing the high and low bf16 values.

    Args:
        x: tl.uint32 tensor containing packed bf16x2 values.
        mask: boolean mask tensor, 1 denotes that x is valid.
    Returns:
        x_hi: float32 tensor containing the high bf16 values.
        x_lo: float32 tensor containing the low bf16 values.
    """
    x = x * mask
    x_hi = (x >> 16).cast(tl.uint16).cast(tl.bfloat16, bitcast=True).cast(tl.float32)
    x_lo = x.cast(tl.uint16).cast(tl.bfloat16, bitcast=True).cast(tl.float32)
    return x_hi, x_lo


@triton.jit
def sum_sq(x, y, z, w, mask):
    """
    First computes the squared sum of 8 bf16 values
    packed in x, y, z, w. Then does an SM-wide
    reduction to get the total sqaured sum.
    Args:
        x, y, z, w: tl.uint32 tensors containing packed bf16x2 values.
        mask: boolean mask tensor, 1 denotes that x,y,z,w are valid.
    Returns:
        sq_sum: float32 scalar, the total squared sum.
    """
    x_hi, x_lo = unpack_bf16x2(x, mask)
    y_hi, y_lo = unpack_bf16x2(y, mask)
    z_hi, z_lo = unpack_bf16x2(z, mask)
    w_hi, w_lo = unpack_bf16x2(w, mask)
    # thread local sum
    sq_sum = (
        x_hi * x_hi
        + x_lo * x_lo
        + y_hi * y_hi
        + y_lo * y_lo
        + z_hi * z_hi
        + z_lo * z_lo
        + w_hi * w_hi
        + w_lo * w_lo
    )
    # sm-wide reduction
    sq_sum = tl.sum(sq_sum)
    return sq_sum


@triton.jit
def apply_norm(x, y, z, w, wx, wy, wz, ww, rrms, mask):
    """
    Apply RMS norm to the input bf16x2 tensors x,y,z,w using
    the rms norm weights wx,wy,wz,ww and the reciprocal
    root mean square rrms.
    """
    # todo: try converting to pure ASM code
    x_hi, x_lo = unpack_bf16x2(x, mask)
    y_hi, y_lo = unpack_bf16x2(y, mask)
    z_hi, z_lo = unpack_bf16x2(z, mask)
    w_hi, w_lo = unpack_bf16x2(w, mask)
    wx_hi, wx_lo = unpack_bf16x2(wx, mask)
    wy_hi, wy_lo = unpack_bf16x2(wy, mask)
    wz_hi, wz_lo = unpack_bf16x2(wz, mask)
    ww_hi, ww_lo = unpack_bf16x2(ww, mask)

    x_hi = (x_hi * rrms * wx_hi).cast(tl.bfloat16).cast(tl.uint16, bitcast=True).cast(
        tl.uint32
    ) << 16
    x_lo = (x_lo * rrms * wx_lo).cast(tl.bfloat16).cast(tl.uint16, bitcast=True).cast(tl.uint32)
    y_hi = (y_hi * rrms * wy_hi).cast(tl.bfloat16).cast(tl.uint16, bitcast=True).cast(
        tl.uint32
    ) << 16
    y_lo = (y_lo * rrms * wy_lo).cast(tl.bfloat16).cast(tl.uint16, bitcast=True).cast(tl.uint32)
    z_hi = (z_hi * rrms * wz_hi).cast(tl.bfloat16).cast(tl.uint16, bitcast=True).cast(
        tl.uint32
    ) << 16
    z_lo = (z_lo * rrms * wz_lo).cast(tl.bfloat16).cast(tl.uint16, bitcast=True).cast(tl.uint32)
    w_hi = (w_hi * rrms * ww_hi).cast(tl.bfloat16).cast(tl.uint16, bitcast=True).cast(
        tl.uint32
    ) << 16
    w_lo = (w_lo * rrms * ww_lo).cast(tl.bfloat16).cast(tl.uint16, bitcast=True).cast(tl.uint32)
    # pack back to bf16x2, to be used by nvls multicast store.
    x = x_hi | x_lo
    y = y_hi | y_lo
    z = z_hi | z_lo
    w = w_hi | w_lo
    return x, y, z, w


@triton.jit
def _multimem_reduce_scatter_residual_add_kernel(
    residual_output_ptr,
    residual_input_ptr,
    rms_norm_weights_ptr,
    multicast_ptr,  # points to symmetric memory buffer
    signal_pad_ptrs,
    num_tokens,
    eps,
    HIDDEN_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUMEL_PER_THREAD: tl.constexpr,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
):
    symm_mem_sync(
        signal_pad_ptrs,
        None,
        RANK,
        WORLD_SIZE,
        hasPreviousMemAccess=False,
        hasSubsequentMemAccess=False,
    )
    sync_threads()

    pid = tl.program_id(axis=0)
    tid = tl.arange(0, BLOCK_SIZE)

    tokens_per_rank = tl.cdiv(num_tokens, WORLD_SIZE)
    numel_per_token = tl.cdiv(HIDDEN_SIZE, NUMEL_PER_THREAD)
    numel_per_rank = tokens_per_rank * numel_per_token

    # each program handles 1 token at a time
    for token_offset in range(pid, tokens_per_rank, tl.num_programs(axis=0)):
        # Step 1: - reduce-scatter + residual add for this token + collect sq sum
        program_offset = token_offset * numel_per_token
        sq_sum_ = 0.0
        for thread_offset in range(0, numel_per_token, BLOCK_SIZE):
            offsets = program_offset + thread_offset + tid

            # BLOCK_SIZE is one token's row rounded up to a power of 2, capped at 1024.
            #   if BLOCK_SIZE > numel_per_token -   deactivates the extra threads.
            #   if BLOCK_SIZE < numel_per_token -   deactivates threads of the last window
            #                                       [thread_offset, thread_offset + BLOCK_SIZE)
            #                                       that fall past the end of the row
            mask = thread_offset + tid < numel_per_token
            multicast_ptrs = (
                multicast_ptr.to(tl.pointer_type(tl.uint64)) + (RANK * numel_per_rank + offsets) * 2
            )
            res_out_ptrs = residual_output_ptr.to(tl.pointer_type(tl.uint64)) + offsets * 2
            res_in_ptrs = residual_input_ptr.to(tl.pointer_type(tl.uint64)) + offsets * 2
            # reduce-scatter
            x, y, z, w = ld_128(multicast_ptrs, mask=mask, multicast_op=True)
            # load residual
            rx, ry, rz, rw = ld_128(res_in_ptrs, mask=mask, multicast_op=False)
            # add residual
            x, y, z, w = add_v8_bf16_from_u32(x, y, z, w, rx, ry, rz, rw)
            # store residual
            st_128(res_out_ptrs, x, y, z, w, mask=mask, multicast_op=False)
            # update squared sum for computing the norm later
            sq_sum_ += sum_sq(x, y, z, w, mask=mask)

        # sum_sq is now the sum of squares for this token
        # it is a SM-wide reduction, so no need to sync_threads()
        mean_sq = sq_sum_ / HIDDEN_SIZE
        rrms = asm_rsqrt(mean_sq, eps)

        # Step 2 - apply-rms-norm + all-gather
        for thread_offset in range(0, numel_per_token, BLOCK_SIZE):
            offsets = program_offset + thread_offset + tid
            # first offset is a token offset
            # second offset is a hidden-dim offset (in units of 128-bit)
            mask = thread_offset + tid < numel_per_token

            multicast_ptrs = (
                multicast_ptr.to(tl.pointer_type(tl.uint64)) + (RANK * numel_per_rank + offsets) * 2
            )
            res_out_ptrs = residual_output_ptr.to(tl.pointer_type(tl.uint64)) + offsets * 2

            rms_norm_weights_ptrs = (
                rms_norm_weights_ptr.to(tl.pointer_type(tl.uint64)) + (thread_offset + tid) * 2
            )

            rx, ry, rz, rw = ld_128(res_out_ptrs, mask=mask, multicast_op=False)
            wx, wy, wz, ww = ld_128(rms_norm_weights_ptrs, mask=mask, multicast_op=False)
            nx, ny, nz, nw = apply_norm(rx, ry, rz, rw, wx, wy, wz, ww, rrms, mask)
            st_128(multicast_ptrs, nx, ny, nz, nw, mask=mask, multicast_op=True)

    sync_threads()
    symm_mem_sync(
        signal_pad_ptrs,
        None,
        RANK,
        WORLD_SIZE,
        hasPreviousMemAccess=True,
        hasSubsequentMemAccess=True,
    )


def fused_multimem_rs_add_norm_ag(
    residual_output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    symm_mem_hdl,
    residual_input_tensor: torch.Tensor,
    rms_norm_weights: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """
    Calls a multicast reduce-scatter + residual add + rms norm + all-gather
    triton kernel. Writes out the output of the residual add to residual_output_tensor.
    The output of the full kernel is written in-place to the symmetric memory buffer.
    input_tensor must be a symmetric memory buffer.
    Args:
        residual_output_tensor: torch.Tensor to write the output of the residual add,
            this rank's num_tokens // world_size tokens.
        input_tensor: torch.Tensor, symmetric memory buffer to read the input from:
            num_tokens tokens of hidden size H, with num_tokens divisible by the world size
            and H by 8 (rows of whole 128-bit lanes).
        symm_mem_hdl: _SymmetricMemory handle for the symmetric memory buffer.
        residual_input_tensor: torch.Tensor, the residual input to be added, shaped like
            residual_output_tensor (it may be the same tensor).
        rms_norm_weights: torch.Tensor, the H weights for rms norm.
        eps: float, epsilon value for rms norm.
    Returns:
        residual_output_tensor: torch.Tensor, the output of the full fused operation.
    """
    # Do not autotune the number of blocks with Triton: every rank must launch the same grid
    # of at most max_num_blocks blocks, or the barrier overflows the signal pad or hangs.
    # See kernel_config.py.
    WARP_SIZE = SYMM_MEM_KERNEL_CONFIG.warp_size
    MAX_NUM_BLOCKS = SYMM_MEM_KERNEL_CONFIG.max_num_blocks
    MAX_BLOCK_SIZE = SYMM_MEM_KERNEL_CONFIG.max_block_size
    BYTES_PER_THREAD = 16

    assert input_tensor.dtype == torch.bfloat16, "Only bfloat16 is supported for now."
    assert residual_output_tensor.dtype == torch.bfloat16, "Only bfloat16 is supported for now."
    assert residual_input_tensor.dtype == torch.bfloat16, "Only bfloat16 is supported for now."
    assert (
        rms_norm_weights.dtype == torch.bfloat16
    ), f"The RMSNorm weights must be bfloat16, got {rms_norm_weights.dtype}."
    # The kernel addresses every tensor through a raw pointer as dense bf16 rows.
    for name, tensor in (
        ("input tensor", input_tensor),
        ("residual output", residual_output_tensor),
        ("residual input", residual_input_tensor),
        ("RMSNorm weights", rms_norm_weights),
    ):
        assert tensor.is_contiguous(), f"The {name} must be contiguous."

    # this evaluates to 128 for bf16.
    # each thread will process 128 bits (8 bf16 values) at a time.
    numel_per_thread = BYTES_PER_THREAD // residual_input_tensor.element_size()

    hsize = input_tensor.size(-1)
    num_tokens = input_tensor.numel() // hsize

    # The kernel gives each rank cdiv(num_tokens, world_size) rows of cdiv(hsize, 8) 128-bit
    # lanes and reads the weights as hsize bf16 values. Anything else makes it read and write
    # past the residual, the weights or this rank's region of every rank's buffer.
    assert (
        hsize % numel_per_thread == 0
    ), f"The hidden size ({hsize}) must be a multiple of {numel_per_thread} (128-bit rows)."
    assert num_tokens % symm_mem_hdl.world_size == 0, (
        f"The number of tokens ({num_tokens}) must be divisible by the world size "
        f"({symm_mem_hdl.world_size})."
    )
    assert (
        rms_norm_weights.numel() == hsize
    ), f"Expected {hsize} RMSNorm weights, got {rms_norm_weights.numel()}."
    tokens_per_rank = num_tokens // symm_mem_hdl.world_size
    for name, residual in (("output", residual_output_tensor), ("input", residual_input_tensor)):
        assert residual.numel() == tokens_per_rank * hsize, (
            f"The residual {name} must hold this rank's {tokens_per_rank} tokens of hidden size "
            f"{hsize}, got {residual.numel()} elements."
        )

    # One token per block, as in the variable-count collectives: each block processes its
    # tokens one after another, so sizing blocks from the whole shard would run several
    # tokens in series with most threads idle.
    numel_per_token = triton.cdiv(hsize, numel_per_thread)
    block_size = min(triton.next_power_of_2(numel_per_token), MAX_BLOCK_SIZE)
    num_warps = max(1, block_size // WARP_SIZE)
    num_blocks = min(triton.cdiv(num_tokens, symm_mem_hdl.world_size), MAX_NUM_BLOCKS)
    SYMM_MEM_KERNEL_CONFIG.check_num_blocks(num_blocks)

    _multimem_reduce_scatter_residual_add_kernel[(num_blocks, 1, 1)](
        residual_output_tensor.data_ptr(),
        residual_input_tensor.data_ptr(),
        rms_norm_weights.data_ptr(),
        symm_mem_hdl.multicast_ptr,
        symm_mem_hdl.signal_pad_ptrs_dev,
        num_tokens,
        eps=eps,
        HIDDEN_SIZE=hsize,
        BLOCK_SIZE=block_size,
        NUMEL_PER_THREAD=numel_per_thread,
        RANK=symm_mem_hdl.rank,
        WORLD_SIZE=symm_mem_hdl.world_size,
        num_warps=num_warps,
    )

    return residual_output_tensor
