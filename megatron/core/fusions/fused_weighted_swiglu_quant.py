import torch

import triton
import triton.language as tl


SCALE_MIN_THRES = 1e-10
FP8_MAX_VALUE = {torch.float8_e4m3fn: 448.0, torch.float8_e5m2: 57344.0}


@triton.heuristics({"BLOCK_SN": lambda args: args["BLOCK_N"] // args["block_size"]})
@triton.jit
def fused_weighted_swiglu_quant_kernel(
    inp_ptr,
    w_ptr,
    out_data_ptr,
    out_scale_ptr,
    M,
    H,
    SN,
    block_size: tl.constexpr,
    fp8_max,
    inp_stride_0,
    inp_stride_1,
    w_stride_0,
    w_stride_1,
    out_data_stride_0,
    out_data_stride_1,
    out_scale_stride_0,
    out_scale_stride_1,
    force_pow_2_scales: tl.constexpr,
    SCALE_MIN_THRES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_SN: tl.constexpr,
):
    pid_dim0 = tl.program_id(0)
    pid_dim1 = tl.program_id(1)

    # split a and b, a: first BLOCK_N cols, b: next BLOCK_N cols
    inp_block_ptr_a = tl.make_block_ptr(
        base=inp_ptr,
        shape=(M, 2 * H),
        strides=(inp_stride_0, inp_stride_1),
        offsets=(pid_dim0 * BLOCK_M, pid_dim1 * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    inp_block_ptr_b = tl.make_block_ptr(
        base=inp_ptr,
        shape=(M, 2 * H),
        strides=(inp_stride_0, inp_stride_1),
        offsets=(pid_dim0 * BLOCK_M, pid_dim1 * BLOCK_N + H),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    w_block_ptr = tl.make_block_ptr(
        base=w_ptr,
        shape=(M, 1),
        strides=(w_stride_0, w_stride_1),
        offsets=(pid_dim0 * BLOCK_M, 0),
        block_shape=(BLOCK_M, 1),
        order=(1, 0),
    )

    a = tl.load(inp_block_ptr_a, boundary_check=(0, 1)).to(tl.float32)
    b = tl.load(inp_block_ptr_b, boundary_check=(0, 1)).to(tl.float32)
    w = tl.load(w_block_ptr, boundary_check=(0, 1)).to(tl.float32)

    # weighted swiglu = silu * b * w, silu(a) = a * sigmoid(a)
    sig = tl.sigmoid(a)
    silu = a * sig
    data = silu * b * w

    # Scale
    data = tl.reshape(data, (BLOCK_M, BLOCK_SN, block_size))
    amax = tl.max(tl.abs(data), axis=2)
    amax = tl.maximum(amax, SCALE_MIN_THRES)
    scale = tl.fdiv(amax, fp8_max)
    if force_pow_2_scales:
        # scale = tl.exp2(tl.ceil(tl.log2(scale)))
        s_bits = tl.cast(scale, tl.uint32, bitcast=True)
        scale_exp = ((s_bits >> 23) & 0xFF) + tl.cast((s_bits & 0x7FFFFF) != 0, tl.uint32)
        scale = tl.cast(scale_exp << 23, tl.float32, bitcast=True)
    scale = tl.reshape(scale, (BLOCK_M, BLOCK_SN, 1))

    # Quantize
    data_q = tl.fdiv(data, scale)

    data_q = data_q.to(out_data_ptr.type.element_ty)
    data_q = tl.reshape(data_q, (BLOCK_M, BLOCK_N))
    scale = scale.to(out_scale_ptr.type.element_ty)
    scale = tl.reshape(scale, (BLOCK_M, BLOCK_SN))

    out_data_block_ptr = tl.make_block_ptr(
        base=out_data_ptr,
        shape=(M, H),
        strides=(out_data_stride_0, out_data_stride_1),
        offsets=(pid_dim0 * BLOCK_M, pid_dim1 * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    scale_block_ptr = tl.make_block_ptr(
        base=out_scale_ptr,
        shape=(M, SN),
        strides=(out_scale_stride_0, out_scale_stride_1),
        offsets=(pid_dim0 * BLOCK_M, pid_dim1 * BLOCK_SN),
        block_shape=(BLOCK_M, BLOCK_SN),
        order=(1, 0),
    )
    tl.store(out_data_block_ptr, data_q, boundary_check=(0, 1))
    tl.store(scale_block_ptr, scale, boundary_check=(0, 1))


def fused_weighted_swiglu_quant(
    x: torch.Tensor, weights: torch.Tensor, block_size: int = 128, fp8type=torch.float8_e4m3fn
):
    assert x.dim() == 2 and weights.dim() == 2 and weights.shape[1] == 1, "weights's shape mismatch"
    M, N2 = x.shape
    if N2 % 2 != 0:
        raise ValueError("Last dim of input must be even (2*H)")
    H = N2 // 2

    SN = (H + block_size - 1) // block_size
    out_data = torch.empty((M, H), dtype=fp8type, device=x.device)
    out_scale = torch.empty((M, SN), dtype=torch.float32, device=x.device)

    BLOCK_M = 32
    BLOCK_N = block_size
    assert (H % BLOCK_N) == 0, "H must be divisible by BLOCK_N for this fixed setting"
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(H, BLOCK_N))
    fused_weighted_swiglu_quant_kernel[grid](
        x,
        weights,
        out_data,
        out_scale,
        M,
        H,
        SN,
        block_size,
        FP8_MAX_VALUE[fp8type],
        x.stride(0),
        x.stride(1),
        weights.stride(0),
        weights.stride(1),
        out_data.stride(0),
        out_data.stride(1),
        out_scale.stride(0),
        out_scale.stride(1),
        force_pow_2_scales=True,
        SCALE_MIN_THRES=SCALE_MIN_THRES,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    return out_data, out_scale


@triton.heuristics({"BLOCK_SN": lambda args: args["BLOCK_N"] // args["block_size"]})
@triton.jit
def fused_weighted_swiglu_quant_back_kernel(
    gradout_ptr,
    inp_ptr,
    w_ptr,
    out_input_grad_q_ptr,
    out_input_grad_scale_ptr,
    out_wgrad_ptr,
    M,
    H,
    SN,
    g_stride_0,
    g_stride_1,
    inp_stride_0,
    inp_stride_1,
    w_stride_0,
    w_stride_1,
    out_input_grad_q_stride_0,
    out_input_grad_q_stride_1,
    out_input_grad_scale_stride_0,
    out_input_grad_scale_stride_1,
    gw_stride_0: tl.constexpr,
    block_size: tl.constexpr,
    fp8_max: tl.constexpr,
    force_pow_2_scales: tl.constexpr,
    SCALE_MIN_THRES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_SN: tl.constexpr,
):
    pid_dim0 = tl.program_id(0)
    pid_dim1 = tl.program_id(1)

    a_ptr = tl.make_block_ptr(
        base=inp_ptr,
        shape=(M, 2 * H),
        strides=(inp_stride_0, inp_stride_1),
        offsets=(pid_dim0 * BLOCK_M, pid_dim1 * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    b_ptr = tl.make_block_ptr(
        base=inp_ptr,
        shape=(M, 2 * H),
        strides=(inp_stride_0, inp_stride_1),
        offsets=(pid_dim0 * BLOCK_M, pid_dim1 * BLOCK_N + H),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    gradout_ptr_blk = tl.make_block_ptr(
        base=gradout_ptr,
        shape=(M, H),
        strides=(g_stride_0, g_stride_1),
        offsets=(pid_dim0 * BLOCK_M, pid_dim1 * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    w_ptr_blk = tl.make_block_ptr(
        base=w_ptr,
        shape=(M, 1),
        strides=(w_stride_0, w_stride_1),
        offsets=(pid_dim0 * BLOCK_M, 0),
        block_shape=(BLOCK_M, 1),
        order=(1, 0),
    )

    a = tl.load(a_ptr, boundary_check=(0, 1)).to(tl.float32)
    b = tl.load(b_ptr, boundary_check=(0, 1)).to(tl.float32)
    g = tl.load(gradout_ptr_blk, boundary_check=(0, 1)).to(tl.float32)
    w = tl.load(w_ptr_blk, boundary_check=(0, 1)).to(tl.float32)

    sig = tl.sigmoid(a)
    silu = a * sig
    dsilu = sig * (1.0 + a * (1.0 - sig))
    g_eff = g * w
    dy1 = g_eff * dsilu * b
    dy2 = g_eff * silu

    # grad_w
    contrib = (silu * b) * g
    part_sum = tl.sum(contrib, axis=1)
    rows = pid_dim0 * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_rows = rows < M
    tl.atomic_add(out_wgrad_ptr + rows * gw_stride_0, part_sum, mask=mask_rows)

    # Quantize
    dy1 = tl.reshape(dy1, (BLOCK_M, BLOCK_SN, block_size))
    max1 = tl.max(tl.abs(dy1), axis=2)
    max1 = tl.maximum(max1, SCALE_MIN_THRES)
    scale1 = tl.fdiv(max1, fp8_max)
    if force_pow_2_scales:
        # scale1 = tl.exp2(tl.ceil(tl.log2(scale1)))
        s_bits = tl.cast(scale1, tl.uint32, bitcast=True)
        scale_exp1 = ((s_bits >> 23) & 0xFF) + tl.cast((s_bits & 0x7FFFFF) != 0, tl.uint32)
        scale1 = tl.cast(scale_exp1 << 23, tl.float32, bitcast=True)

    dy1_q = tl.fdiv(dy1, tl.reshape(scale1, (BLOCK_M, BLOCK_SN, 1)))

    dy2 = tl.reshape(dy2, (BLOCK_M, BLOCK_SN, block_size))
    max2 = tl.max(tl.abs(dy2), axis=2)
    max2 = tl.maximum(max2, SCALE_MIN_THRES)
    scale2 = tl.fdiv(max2, fp8_max)
    if force_pow_2_scales:
        # scale2 = tl.exp2(tl.ceil(tl.log2(scale2)))
        s_bits = tl.cast(scale2, tl.uint32, bitcast=True)
        scale_exp2 = ((s_bits >> 23) & 0xFF) + tl.cast((s_bits & 0x7FFFFF) != 0, tl.uint32)
        scale2 = tl.cast(scale_exp2 << 23, tl.float32, bitcast=True)

    dy2_q = tl.fdiv(dy2, tl.reshape(scale2, (BLOCK_M, BLOCK_SN, 1)))

    dy1_q = dy1_q.to(out_input_grad_q_ptr.type.element_ty)
    dy1_q = tl.reshape(dy1_q, (BLOCK_M, BLOCK_N))
    dy2_q = dy2_q.to(out_input_grad_q_ptr.type.element_ty)
    dy2_q = tl.reshape(dy2_q, (BLOCK_M, BLOCK_N))
    scale1 = scale1.to(out_input_grad_scale_ptr.type.element_ty)
    scale2 = scale2.to(out_input_grad_scale_ptr.type.element_ty)

    gy1_ptr = tl.make_block_ptr(
        base=out_input_grad_q_ptr,
        shape=(M, 2 * H),
        strides=(out_input_grad_q_stride_0, out_input_grad_q_stride_1),
        offsets=(pid_dim0 * BLOCK_M, pid_dim1 * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    gy2_ptr = tl.make_block_ptr(
        base=out_input_grad_q_ptr,
        shape=(M, 2 * H),
        strides=(out_input_grad_q_stride_0, out_input_grad_q_stride_1),
        offsets=(pid_dim0 * BLOCK_M, H + pid_dim1 * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    tl.store(gy1_ptr, dy1_q, boundary_check=(0, 1))
    tl.store(gy2_ptr, dy2_q, boundary_check=(0, 1))

    s1_ptr = tl.make_block_ptr(
        base=out_input_grad_scale_ptr,
        shape=(M, 2 * SN),
        strides=(out_input_grad_scale_stride_0, out_input_grad_scale_stride_1),
        offsets=(pid_dim0 * BLOCK_M, pid_dim1 * BLOCK_SN),
        block_shape=(BLOCK_M, BLOCK_SN),
        order=(1, 0),
    )
    s2_ptr = tl.make_block_ptr(
        base=out_input_grad_scale_ptr,
        shape=(M, 2 * SN),
        strides=(out_input_grad_scale_stride_0, out_input_grad_scale_stride_1),
        offsets=(pid_dim0 * BLOCK_M, SN + pid_dim1 * BLOCK_SN),
        block_shape=(BLOCK_M, BLOCK_SN),
        order=(1, 0),
    )
    tl.store(s1_ptr, scale1, boundary_check=(0, 1))
    tl.store(s2_ptr, scale2, boundary_check=(0, 1))


def fused_weighted_swiglu_quant_back(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    weights: torch.Tensor,
    block_size: int = 128,
    fp8type=torch.float8_e4m3fn,
):
    assert (
        input.shape[:-1] == grad_output.shape[:-1] and input.shape[-1] == 2 * grad_output.shape[-1]
    ), "shape mismatch"
    assert weights.shape[0] == grad_output.shape[0], "shape mismatch"

    device = grad_output.device
    M, H = grad_output.shape
    SN = (H + block_size - 1) // block_size

    out_input_grad_q = torch.empty((M, 2 * H), device=device, dtype=fp8type)
    out_input_grad_scale = torch.empty((M, 2 * SN), device=device, dtype=torch.float32)
    out_wgrad = torch.zeros((M,), device=device, dtype=torch.float32)

    BLOCK_M = 32
    BLOCK_N = block_size
    assert (H % BLOCK_N) == 0, "H must be divisible by BLOCK_N for this fixed setting"
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(H, BLOCK_N))
    fused_weighted_swiglu_quant_back_kernel[grid](
        grad_output,
        input,
        weights,
        out_input_grad_q,
        out_input_grad_scale,
        out_wgrad,
        M,
        H,
        SN,
        grad_output.stride(0),
        grad_output.stride(1),
        input.stride(0),
        input.stride(1),
        weights.stride(0),
        weights.stride(1),
        out_input_grad_q.stride(0),
        out_input_grad_q.stride(1),
        out_input_grad_scale.stride(0),
        out_input_grad_scale.stride(1),
        1,
        block_size,
        FP8_MAX_VALUE[fp8type],
        force_pow_2_scales=True,
        SCALE_MIN_THRES=SCALE_MIN_THRES,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )

    out_wgrad = out_wgrad.reshape(M, 1).to(weights.dtype)
    return out_input_grad_q, out_input_grad_scale, out_wgrad
