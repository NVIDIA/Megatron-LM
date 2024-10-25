# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import torch
import triton
import triton.language as tl


@triton.jit
def update_inference_params_faster_kernel(
    q_ptr,
    k_ptr,  # B, H, D
    v_ptr,
    kv_win_ptr,
    w_win_ptr,
    lens_ptr,
    # meta
    bs,
    win_pos,
    nheads: tl.constexpr,
    head_dim: tl.constexpr,
    win_sz: tl.constexpr,
    extra_val: tl.constexpr,  # float=5.0,
    eps: tl.constexpr,  # float=torch.finfo(torch.bfloat16).eps
    BLOCK_SIZE: tl.constexpr,
):
    """Fast kernel for DMC state update

    Passes less arguments to be resolved at run-time,
    which speeds up the execution roughly 2x.

    Limitations
     * fixed strides of tensors
     * static nheads / head_dim etc.
     * the first boundary needs to be explicitly set to zero outside this fun
    """
    pid_m = tl.program_id(axis=0)
    pid_batch = tl.program_id(axis=1)
    pid_head = tl.program_id(axis=2)

    # strides
    q_stride_batch = nheads * head_dim
    q_stride_nheads = head_dim
    k_stride_batch = nheads * head_dim
    k_stride_nheads = head_dim
    v_stride_batch = nheads * head_dim
    v_stride_nheads = head_dim

    q_ptr = q_ptr + pid_batch * q_stride_batch + pid_head * q_stride_nheads
    k_ptr = k_ptr + pid_batch * k_stride_batch + pid_head * k_stride_nheads
    v_ptr = v_ptr + pid_batch * v_stride_batch + pid_head * v_stride_nheads

    k_sum_offset = 0
    v_sum_offset = bs * nheads * head_dim
    k_win_offset = (win_pos + 1) * (2 * bs * nheads * head_dim)
    v_win_offset = (win_pos + 1) * (2 * bs * nheads * head_dim) + (bs * nheads * head_dim)
    kv_stride_batch = nheads * head_dim
    kv_stride_nheads = head_dim

    k_win_ptr = kv_win_ptr + k_win_offset + pid_batch * kv_stride_batch + pid_head * kv_stride_nheads
    v_win_ptr = kv_win_ptr + v_win_offset + pid_batch * kv_stride_batch + pid_head * kv_stride_nheads
    k_sum_ptr = kv_win_ptr + k_sum_offset + pid_batch * kv_stride_batch + pid_head * kv_stride_nheads
    v_sum_ptr = kv_win_ptr + v_sum_offset + pid_batch * kv_stride_batch + pid_head * kv_stride_nheads

    w_sum_offset = 0
    w_win_offset = (win_pos + 1) * (bs * nheads)
    w_stride_batch = nheads
    w_stride_nheads = 1

    w_sum_ptr = w_win_ptr + w_sum_offset + pid_batch * w_stride_batch + pid_head * w_stride_nheads
    w_win_ptr = w_sum_ptr + w_win_offset

    lens_head_offset = bs * nheads

    lens_ptr = lens_ptr + pid_batch * w_stride_batch + pid_head * w_stride_nheads
    lens_head_ptr = lens_ptr + lens_head_offset

    dim_offsets = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Load data
    k = tl.load(k_ptr + dim_offsets, mask=dim_offsets < head_dim, other=0.0)
    v = tl.load(v_ptr + dim_offsets, mask=dim_offsets < head_dim, other=0.0)
    k_sum = tl.load(k_sum_ptr + dim_offsets, mask=dim_offsets < head_dim, other=0.0)
    v_sum = tl.load(v_sum_ptr + dim_offsets, mask=dim_offsets < head_dim, other=0.0)
    k_win = tl.load(k_win_ptr + dim_offsets, mask=dim_offsets < head_dim, other=0.0)
    v_win = tl.load(v_win_ptr + dim_offsets, mask=dim_offsets < head_dim, other=0.0)
    lens = tl.load(lens_ptr)
    lens_head = tl.load(lens_head_ptr)
    w_win = tl.load(w_win_ptr)
    w_sum = tl.load(w_sum_ptr)

    k = tl.where(dim_offsets < head_dim - 1, k, 0.0)

    logits = tl.load(k_ptr + head_dim - 1)

    if logits.dtype is tl.bfloat16:
        bounds = (logits - extra_val - 0.00785 >= 0.0).to(tl.int16)
    else:
        bounds = (logits - extra_val > 0.0).to(tl.int16)
    # bounds = (logits - extra_val > 0.0).to(tl.int16)

    # Force "append" for the first element (set boundary variable to 0)
    # bounds = bounds * (seq_start >= 1)

    w_logits = tl.load(q_ptr + head_dim - 1)
    weights = tl.sigmoid(w_logits.to(tl.float32) + extra_val) + eps

    k_weighted = k * weights
    v_weighted = v * weights
    lens_head = lens_head * bounds + 1
    win_overflow = lens_head > 12
    k_oldest = k_win
    v_oldest = v_win
    w_oldest = w_win
    k_sum_new = k_sum * bounds - win_overflow * k_oldest + k_weighted
    v_sum_new = v_sum * bounds - win_overflow * v_oldest + v_weighted
    w_sum_new = w_sum * bounds - win_overflow * w_oldest + weights

    # Trying to write just 0 raises "FloatAttr does not match expected type of the constant"
    tl.store(q_ptr + head_dim - 1, weights * 0)

    tl.store(k_ptr + dim_offsets, k_sum_new / w_sum_new, mask=dim_offsets < head_dim)
    tl.store(v_ptr + dim_offsets, v_sum_new / w_sum_new, mask=dim_offsets < head_dim)

    tl.store(k_sum_ptr + dim_offsets, k_sum_new, mask=dim_offsets < head_dim)
    tl.store(v_sum_ptr + dim_offsets, v_sum_new, mask=dim_offsets < head_dim)
    tl.store(w_sum_ptr, w_sum_new)

    tl.store(k_win_ptr + dim_offsets, k_weighted, mask=dim_offsets < head_dim)
    tl.store(v_win_ptr + dim_offsets, v_weighted, mask=dim_offsets < head_dim)
    tl.store(w_win_ptr, weights)

    tl.store(lens_ptr, lens + 1 - bounds)
    tl.store(lens_head_ptr, lens_head)


@triton.jit
def update_inference_params_faster_no_window_kernel(
    q_ptr,
    k_ptr,  # B, H, D
    v_ptr,
    kv_win_ptr,
    w_win_ptr,
    lens_ptr,
    # meta
    bs,
    nheads: tl.constexpr,
    head_dim: tl.constexpr,
    extra_val: tl.constexpr,  # float=5.0,
    eps: tl.constexpr,  # float=torch.finfo(torch.bfloat16).eps
    BLOCK_SIZE: tl.constexpr,
):
    """Fast kernel for DMC state update

    Passes less arguments to be resolved at run-time,
    which speeds up the execution roughly 2x.

    Limitations
     * fixed strides of tensors
     * static nheads / head_dim etc.
     * the first boundary needs to be explicitly set to zero outside this fun
    """
    pid_m = tl.program_id(axis=0)
    pid_batch = tl.program_id(axis=1)
    pid_head = tl.program_id(axis=2)

    # strides
    q_stride_batch = nheads * head_dim
    q_stride_nheads = head_dim
    k_stride_batch = nheads * head_dim
    k_stride_nheads = head_dim
    v_stride_batch = nheads * head_dim
    v_stride_nheads = head_dim

    q_ptr = q_ptr + pid_batch * q_stride_batch + pid_head * q_stride_nheads
    k_ptr = k_ptr + pid_batch * k_stride_batch + pid_head * k_stride_nheads
    v_ptr = v_ptr + pid_batch * v_stride_batch + pid_head * v_stride_nheads

    k_sum_offset = 0
    v_sum_offset = bs * nheads * head_dim
    kv_stride_batch = nheads * head_dim
    kv_stride_nheads = head_dim

    k_sum_ptr = kv_win_ptr + k_sum_offset + pid_batch * kv_stride_batch + pid_head * kv_stride_nheads
    v_sum_ptr = kv_win_ptr + v_sum_offset + pid_batch * kv_stride_batch + pid_head * kv_stride_nheads

    w_sum_offset = 0
    w_stride_batch = nheads
    w_stride_nheads = 1

    w_sum_ptr = w_win_ptr + w_sum_offset + pid_batch * w_stride_batch + pid_head * w_stride_nheads

    lens_head_offset = bs * nheads

    lens_ptr = lens_ptr + pid_batch * w_stride_batch + pid_head * w_stride_nheads
    lens_head_ptr = lens_ptr + lens_head_offset

    dim_offsets = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Load data
    k = tl.load(k_ptr + dim_offsets, mask=dim_offsets < head_dim, other=0.0)
    v = tl.load(v_ptr + dim_offsets, mask=dim_offsets < head_dim, other=0.0)
    k_sum = tl.load(k_sum_ptr + dim_offsets, mask=dim_offsets < head_dim, other=0.0)
    v_sum = tl.load(v_sum_ptr + dim_offsets, mask=dim_offsets < head_dim, other=0.0)
    lens = tl.load(lens_ptr)
    lens_head = tl.load(lens_head_ptr)
    w_sum = tl.load(w_sum_ptr)

    k = tl.where(dim_offsets < head_dim - 1, k, 0.0)

    logits = tl.load(k_ptr + head_dim - 1)

    if logits.dtype is tl.bfloat16:
        bounds = (logits - extra_val - 0.00785 >= 0.0).to(tl.int16)
    else:
        bounds = (logits - extra_val > 0.0).to(tl.int16)
    # bounds = (logits - extra_val > 0.0).to(tl.int16)

    # Force "append" for the first element (set boundary variable to 0)
    # bounds = bounds * (seq_start >= 1)

    w_logits = tl.load(q_ptr + head_dim - 1)
    weights = tl.sigmoid(w_logits.to(tl.float32) + extra_val) + eps

    k_weighted = k * weights
    v_weighted = v * weights
    lens_head = lens_head * bounds + 1
    k_sum_new = k_sum * bounds + k_weighted
    v_sum_new = v_sum * bounds + v_weighted
    w_sum_new = w_sum * bounds + weights

    # Trying to write just 0 raises "FloatAttr does not match expected type of the constant"
    tl.store(q_ptr + head_dim - 1, weights * 0)

    tl.store(k_ptr + dim_offsets, k_sum_new / w_sum_new, mask=dim_offsets < head_dim)
    tl.store(v_ptr + dim_offsets, v_sum_new / w_sum_new, mask=dim_offsets < head_dim)

    tl.store(k_sum_ptr + dim_offsets, k_sum_new, mask=dim_offsets < head_dim)
    tl.store(v_sum_ptr + dim_offsets, v_sum_new, mask=dim_offsets < head_dim)
    tl.store(w_sum_ptr, w_sum_new)

    tl.store(lens_ptr, lens + 1 - bounds)
    tl.store(lens_head_ptr, lens_head)


def update_inference_params_triton_faster(
    q,
    k,
    v,
    kv_win,
    w_win,
    lens,
    win_ptr=0,
    win_sz=12,
    seq_start=0,
    extra_val=5.0,
    eps=torch.finfo(torch.bfloat16).eps,
):
    _, _, batch, nheads, dim = kv_win.size()
    grid = lambda META: (triton.cdiv(dim, META["BLOCK_SIZE"]), batch, nheads)
    BLOCK_SIZE = min(triton.next_power_of_2(dim), 256)

    if seq_start == 0:
        k[..., -1] = -5  # Explicitly sets boundary variable to 0 (append)

    if win_sz == 0:
        update_inference_params_faster_no_window_kernel[grid](
            q,
            k,
            v,
            kv_win,
            w_win,
            lens,
            # vars
            batch,
            # win_ptr,
            # consts
            nheads,
            dim,
            # win_sz,
            extra_val,
            eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    else:
        update_inference_params_faster_kernel[grid](
            q,
            k,
            v,
            kv_win,
            w_win,
            lens,
            # vars
            batch,
            win_ptr,
            # consts
            nheads,
            dim,
            win_sz,
            extra_val,
            eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )
