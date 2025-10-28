import torch
import triton 
import triton.language as tl
from .asm_utils import multimem_ld_reduce_128, multimem_st_128, ld_128, st_128, add_v8_bf16_from_u32, asm_rsqrt

from .triton_barrier import blockwise_barrier
from .triton_utils import sync_threads


@triton.jit
def unpack_bf16x2(x, mask):
    # todo: try converting to pure ASM code
    x = x * mask
    x_hi = (x >> 16).cast(tl.uint16).cast(tl.bfloat16, bitcast=True).cast(tl.float32)
    x_lo = x.cast(tl.uint16).cast(tl.bfloat16, bitcast=True).cast(tl.float32)
    return x_hi, x_lo

@triton.jit
def sum_sq(x, y, z, w, mask):
    x_hi, x_lo = unpack_bf16x2(x, mask)
    y_hi, y_lo = unpack_bf16x2(y, mask)
    z_hi, z_lo = unpack_bf16x2(z, mask)
    w_hi, w_lo = unpack_bf16x2(w, mask)
    # calculate sum of squares
    # tl.sum -> SM-wide reduction
    sq_sum = x_hi * x_hi + x_lo * x_lo + \
             y_hi * y_hi + y_lo * y_lo + \
             z_hi * z_hi + z_lo * z_lo + \
             w_hi * w_hi + w_lo * w_lo
    sq_sum = tl.sum(sq_sum)
    return sq_sum 

@triton.jit
def apply_norm(x, y, z, w, wx, wy, wz, ww, rrms, mask):
    # todo: try converting to pure ASM code
    x_hi, x_lo = unpack_bf16x2(x, mask)
    y_hi, y_lo = unpack_bf16x2(y, mask)
    z_hi, z_lo = unpack_bf16x2(z, mask)
    w_hi, w_lo = unpack_bf16x2(w, mask)
    wx_hi, wx_lo = unpack_bf16x2(wx, mask)
    wy_hi, wy_lo = unpack_bf16x2(wy, mask)
    wz_hi, wz_lo = unpack_bf16x2(wz, mask)
    ww_hi, ww_lo = unpack_bf16x2(ww, mask)

    x_hi = (x_hi * rrms * wx_hi).cast(tl.bfloat16).cast(tl.uint16, bitcast=True).cast(tl.uint32) << 16
    x_lo = (x_lo * rrms * wx_lo).cast(tl.bfloat16).cast(tl.uint16, bitcast=True).cast(tl.uint32)
    y_hi = (y_hi * rrms * wy_hi).cast(tl.bfloat16).cast(tl.uint16, bitcast=True).cast(tl.uint32) << 16
    y_lo = (y_lo * rrms * wy_lo).cast(tl.bfloat16).cast(tl.uint16, bitcast=True).cast(tl.uint32)
    z_hi = (z_hi * rrms * wz_hi).cast(tl.bfloat16).cast(tl.uint16, bitcast=True).cast(tl.uint32) << 16
    z_lo = (z_lo * rrms * wz_lo).cast(tl.bfloat16).cast(tl.uint16, bitcast=True).cast(tl.uint32)
    w_hi = (w_hi * rrms * ww_hi).cast(tl.bfloat16).cast(tl.uint16, bitcast=True).cast(tl.uint32) << 16
    w_lo = (w_lo * rrms * ww_lo).cast(tl.bfloat16).cast(tl.uint16, bitcast=True).cast(tl.uint32)
    # pack back to bf16x2, to be used by nvls multicast store.
    x = x_hi | x_lo
    y = y_hi | y_lo
    z = z_hi | z_lo
    w = w_hi | w_lo
    return x, y, z, w

@triton.jit
def multimem_reduce_scatter_residual_add_kernel(
    residual_output_ptr,
    residual_input_ptr,
    rms_norm_weights_ptr, 
    multicast_ptr, # points to symmetric memory buffer
    signal_pad_ptrs,
    num_tokens,
    eps,
    HIDDEN_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUMEL_PER_THREAD: tl.constexpr,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
):
    blockwise_barrier(signal_pad_ptrs, None, RANK, WORLD_SIZE, sem="relaxed")
    sync_threads()

    pid = tl.program_id(axis=0)
    tid = tl.arange(0, BLOCK_SIZE)


    tokens_per_rank = tl.cdiv(num_tokens, WORLD_SIZE)
    numel_per_token = tl.cdiv(HIDDEN_SIZE, NUMEL_PER_THREAD)
    numel_per_rank = tokens_per_rank * numel_per_token    

    # each program handles 1 token at a time
    program_offset = pid * numel_per_token
    thread_mask = tid < numel_per_token

    for token_offset in range(pid, tokens_per_rank, tl.num_programs(axis=0)):    
        # Step 1: - reduce-scatter + residual add for this token + collect sq sum
        program_offset = token_offset * numel_per_token
        sq_sum_ = 0.0
        for thread_offset in range(0, numel_per_token, BLOCK_SIZE):
            offsets = program_offset + thread_offset + tid 
            mask = (offsets < numel_per_rank) & (thread_mask)
            multicast_ptrs = (
                multicast_ptr.to(tl.pointer_type(tl.uint64))
                + (RANK * numel_per_rank + offsets) * 2
            )
            res_out_ptrs = (
                residual_output_ptr.to(tl.pointer_type(tl.uint64))
                + offsets * 2
            )
            res_in_ptrs = (
                residual_input_ptr.to(tl.pointer_type(tl.uint64))
                + offsets * 2
            )
            # reduce-scatter
            (x, y, z, w) = multimem_ld_reduce_128(multicast_ptrs, mask=mask)
            # load residual
            (rx, ry, rz, rw) = ld_128(res_in_ptrs, mask=mask)
            # add residual
            (x, y, z, w) = add_v8_bf16_from_u32(x, y, z, w, rx, ry, rz, rw)
            # store residual
            st_128(res_out_ptrs, x, y, z, w, mask=mask)
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
            mask = (offsets < numel_per_rank) & (thread_mask)

            multicast_ptrs = (
                multicast_ptr.to(tl.pointer_type(tl.uint64))
                + (RANK * numel_per_rank + offsets) * 2
            )
            res_out_ptrs = (
                residual_output_ptr.to(tl.pointer_type(tl.uint64))
                + offsets * 2
            )
            
            rms_norm_weights_ptrs = (
                rms_norm_weights_ptr.to(tl.pointer_type(tl.uint64))
                + (thread_offset + tid) * 2
            )

            (rx, ry, rz, rw) = ld_128(res_out_ptrs, mask=mask)
            (wx, wy, wz, ww) = ld_128(rms_norm_weights_ptrs, mask=mask)
            (nx, ny, nz, nw) = apply_norm(rx, ry, rz, rw, wx, wy, wz, ww, rrms, mask)
            multimem_st_128(multicast_ptrs, nx, ny, nz, nw, mask=mask)

    sync_threads()
    blockwise_barrier(signal_pad_ptrs, None, RANK, WORLD_SIZE, sem="acq_rel")
    
        

def multimem_fused_rs_res_rms_ag(residual_output_tensor: torch.Tensor, 
                            input_tensor: torch.Tensor, 
                            symm_mem_hdl,
                            residual_input_tensor: torch.Tensor,
                            rms_norm_weights: torch.Tensor,
                            eps: float) -> torch.Tensor:
    """
    Input tensor must be a symmetric memory buffer.
    symm_mem_hdl is the symmemtric memory handle to input tensor.
    Output tensor can be a regular tensor. 
    Residual ip/op tensor can be a regular tensor.
    we will overwrite the input_tensor in place.
    """
    WARP_SIZE = 32
    MAX_NUM_BLOCKS = 128
    MAX_BLOCK_SIZE = 1024
    BYTES_PER_THREAD = 16

    assert input_tensor.dtype == torch.bfloat16, "Only bfloat16 is supported for now."
    assert residual_output_tensor.dtype == torch.bfloat16, "Only bfloat16 is supported for now."
    assert residual_input_tensor.dtype == torch.bfloat16, "Only bfloat16 is supported for now."
    
    # this evaluates to 128 for bf16.
    # each thread will process 128 bits (8 bf16 values) at a time.
    numel_per_thread = BYTES_PER_THREAD // residual_input_tensor.element_size()

    assert (
        input_tensor.numel() % numel_per_thread == 0
    ), "The number of elements must be 128-bit aligned."

    num_threads = triton.cdiv(
        input_tensor.numel() // numel_per_thread, symm_mem_hdl.world_size
    )
    
    if num_threads < MAX_BLOCK_SIZE:
        block_size = 1
        while block_size < num_threads:
            block_size *= 2
        num_warps = block_size // WARP_SIZE
        num_blocks = 1
    else:
        block_size = MAX_BLOCK_SIZE
        num_warps = MAX_BLOCK_SIZE // WARP_SIZE
        num_blocks = min(
            triton.cdiv(num_threads, MAX_BLOCK_SIZE),
            MAX_NUM_BLOCKS,
        )

    hsize = input_tensor.size(-1)
    kernel = multimem_reduce_scatter_residual_add_kernel[(num_blocks, 1, 1)](
        residual_output_tensor.data_ptr(),
        residual_input_tensor.data_ptr(),
        rms_norm_weights.data_ptr(),
        symm_mem_hdl.multicast_ptr,
        symm_mem_hdl.signal_pad_ptrs_dev,
        input_tensor.numel() // hsize, 
        eps=eps,
        HIDDEN_SIZE=hsize,
        BLOCK_SIZE=block_size,
        NUMEL_PER_THREAD=numel_per_thread,
        RANK=symm_mem_hdl.rank,
        WORLD_SIZE=symm_mem_hdl.world_size,
        num_warps=num_warps,
    )
    

    return residual_output_tensor