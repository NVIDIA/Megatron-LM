# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Optional Triton kernels for the minimum-activation DSA-GQA backend."""

from __future__ import annotations

from typing import Optional, Tuple
from unittest.mock import MagicMock

import torch

try:
    import triton
    import triton.language as tl
    from triton.runtime.errors import OutOfResources as _TritonOutOfResources

    HAVE_TRITON = True
    _TRITON_RESOURCE_ERRORS = (_TritonOutOfResources,)
except ImportError:
    HAVE_TRITON = False
    _TRITON_RESOURCE_ERRORS = ()


def _null_decorator(fn=None, *args, **kwargs):
    if callable(fn):
        return fn

    def inner(func):
        return func

    return inner


if not HAVE_TRITON:
    triton = MagicMock()
    triton.jit = _null_decorator
    triton.heuristics = _null_decorator
    triton.autotune = _null_decorator
    triton.Config = lambda *args, **kwargs: None
    tl = MagicMock()

if not hasattr(triton, "autotune"):
    triton.autotune = _null_decorator
if not hasattr(triton, "Config"):
    triton.Config = lambda *args, **kwargs: None


_MAX_TRITON_SUPPORT_TOPK = 2048
_VALUE_DTYPE_FP32 = 0
_VALUE_DTYPE_FP16 = 1
_VALUE_DTYPE_BF16 = 2
_HADAMARD_CACHE = {}
_MIN_MEMORY_TRITON_ENABLED = True


def set_min_memory_triton_enabled(enabled: bool) -> bool:
    """Enable or disable optional Triton dispatch for the active min-memory backend call."""
    global _MIN_MEMORY_TRITON_ENABLED
    previous = _MIN_MEMORY_TRITON_ENABLED
    _MIN_MEMORY_TRITON_ENABLED = bool(enabled)
    return previous


def _triton_disabled() -> bool:
    return not _MIN_MEMORY_TRITON_ENABLED


def _next_power_of_2(value: int) -> int:
    if HAVE_TRITON:
        return int(triton.next_power_of_2(value))
    return 1 << (int(value) - 1).bit_length()


def _supported_tensor(tensor: torch.Tensor) -> bool:
    return HAVE_TRITON and tensor.is_cuda and tensor.dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    )


def _supported_index_tensor(tensor: torch.Tensor) -> bool:
    return HAVE_TRITON and tensor.is_cuda and tensor.dtype in (torch.int32, torch.int64)


def _hadamard_matrix(dim: int, device: torch.device) -> Optional[torch.Tensor]:
    if not HAVE_TRITON or dim < 16 or dim > 256 or dim & (dim - 1):
        return None
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    key = (device.type, device_index, dim)
    matrix = _HADAMARD_CACHE.get(key)
    if matrix is not None and matrix.device == device:
        return matrix

    matrix = torch.ones((1, 1), device=device, dtype=torch.float32)
    while matrix.size(0) < dim:
        matrix = torch.cat(
            (
                torch.cat((matrix, matrix), dim=1),
                torch.cat((matrix, -matrix), dim=1),
            ),
            dim=0,
        )
    matrix = (matrix * (dim**-0.5)).to(dtype=torch.bfloat16)
    _HADAMARD_CACHE[key] = matrix
    return matrix


def _can_use_sparse_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    topk_indices: torch.Tensor,
) -> bool:
    if _triton_disabled():
        return False
    if not (
        _supported_tensor(query)
        and _supported_tensor(key)
        and _supported_tensor(value)
        and _supported_index_tensor(topk_indices)
    ):
        return False
    head_dim = query.size(-1)
    value_dim = value.size(-1)
    topk = topk_indices.size(-1)
    return head_dim <= 256 and value_dim <= 256 and topk <= _MAX_TRITON_SUPPORT_TOPK


def _can_use_index_scores(
    q_index: torch.Tensor,
    weights: torch.Tensor,
    k_index: torch.Tensor,
    topk: int,
) -> bool:
    if _triton_disabled():
        return False
    if not (_supported_tensor(q_index) and _supported_tensor(weights) and _supported_tensor(k_index)):
        return False
    index_heads = q_index.size(2)
    index_head_dim = q_index.size(3)
    key_block = k_index.size(0)
    return (
        index_heads <= 64
        and index_head_dim <= 256
        and key_block <= 2048
        and topk <= _MAX_TRITON_SUPPORT_TOPK
    )


def _can_use_selected_index_scores(
    q_index: torch.Tensor,
    weights: torch.Tensor,
    selected_k_index: torch.Tensor,
    topk_indices: torch.Tensor,
) -> bool:
    if _triton_disabled():
        return False
    if not (
        _supported_tensor(q_index)
        and _supported_tensor(weights)
        and _supported_tensor(selected_k_index)
        and _supported_index_tensor(topk_indices)
    ):
        return False
    index_heads = q_index.size(2)
    index_head_dim = q_index.size(3)
    topk = topk_indices.size(-1)
    return index_heads <= 64 and index_head_dim <= 256 and topk <= _MAX_TRITON_SUPPORT_TOPK


def _value_dtype_tag(dtype: torch.dtype) -> int:
    if dtype == torch.float16:
        return _VALUE_DTYPE_FP16
    if dtype == torch.bfloat16:
        return _VALUE_DTYPE_BF16
    return _VALUE_DTYPE_FP32


def _sparse_attention_autotune_configs():
    return [
        triton.Config({"BLOCK_K": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_K": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_K": 128}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_K": 256}, num_warps=8, num_stages=4),
    ]


def _topk_tiled_autotune_configs():
    return [
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 256}, num_warps=8, num_stages=2),
    ]


def _selected_index_scores_autotune_configs():
    return [
        triton.Config({"BLOCK_K": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_K": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_K": 128}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_K": 256}, num_warps=8, num_stages=4),
    ]


def _simplified_selected_scores_autotune_configs():
    return [
        triton.Config({"BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_K": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_K": 128}, num_warps=8, num_stages=3),
    ]


def _simplified_score_block_autotune_configs():
    return [
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_warps=8, num_stages=3),
    ]


def _selected_index_scores_bwd_dot_autotune_configs():
    return [
        triton.Config({"BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_K": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_K": 128}, num_warps=8, num_stages=4),
    ]


def _indexer_loss_autotune_configs():
    return [
        triton.Config({}, num_warps=4, num_stages=3),
        triton.Config({}, num_warps=8, num_stages=4),
    ]


def _teacher_scores_autotune_configs():
    return [
        triton.Config({"BLOCK_K": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_K": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_K": 256}, num_warps=8, num_stages=4),
    ]


def _linear_wgrad_autotune_configs():
    return [
        triton.Config({"BLOCK_O": 16, "BLOCK_I": 32, "BLOCK_N": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_O": 16, "BLOCK_I": 64, "BLOCK_N": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_O": 32, "BLOCK_I": 32, "BLOCK_N": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_O": 32, "BLOCK_I": 64, "BLOCK_N": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_O": 32, "BLOCK_I": 64, "BLOCK_N": 64}, num_warps=8, num_stages=4),
    ]


def _selected_k_linear_autotune_configs():
    return [
        triton.Config({"BLOCK_N": 16, "BLOCK_D": 32, "BLOCK_H": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_N": 32, "BLOCK_D": 32, "BLOCK_H": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_N": 32, "BLOCK_D": 32, "BLOCK_H": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_N": 32, "BLOCK_D": 64, "BLOCK_H": 64}, num_warps=8, num_stages=3),
    ]


def _selected_k_score_autotune_configs():
    return [
        triton.Config({"BLOCK_N": 16, "BLOCK_H": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_N": 16, "BLOCK_H": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_N": 32, "BLOCK_H": 64}, num_warps=8, num_stages=3),
    ]


def _gathered_linear_wgrad_autotune_configs():
    return [
        triton.Config({"BLOCK_O": 16, "BLOCK_I": 32, "BLOCK_N": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_O": 16, "BLOCK_I": 64, "BLOCK_N": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_O": 32, "BLOCK_I": 32, "BLOCK_N": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_O": 32, "BLOCK_I": 64, "BLOCK_N": 64}, num_warps=8, num_stages=4),
    ]


def _scatter_selected_grad_autotune_configs():
    return [
        triton.Config({"BLOCK_N": 64, "BLOCK_D": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_N": 128, "BLOCK_D": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_N": 128, "BLOCK_D": 64}, num_warps=8, num_stages=4),
    ]


def _k_ln_backward_autotune_configs():
    return [
        triton.Config({"BLOCK_D": 32, "BLOCK_N": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_D": 64, "BLOCK_N": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_D": 64, "BLOCK_N": 64}, num_warps=8, num_stages=4),
    ]


@triton.jit
def _dsa_topk_index_block_kernel(
    q_ptr,
    weights_ptr,
    k_ptr,
    out_scores_ptr,
    out_indices_ptr,
    q_start,
    k_start,
    query_len: tl.constexpr,
    key_len: tl.constexpr,
    topk: tl.constexpr,
    q_stride_m: tl.constexpr,
    q_stride_b: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_stride_d: tl.constexpr,
    w_stride_m: tl.constexpr,
    w_stride_b: tl.constexpr,
    w_stride_h: tl.constexpr,
    k_stride_s: tl.constexpr,
    k_stride_b: tl.constexpr,
    k_stride_d: tl.constexpr,
    out_score_stride_b: tl.constexpr,
    out_score_stride_m: tl.constexpr,
    out_score_stride_k: tl.constexpr,
    out_index_stride_b: tl.constexpr,
    out_index_stride_m: tl.constexpr,
    out_index_stride_k: tl.constexpr,
    INDEX_HEADS: tl.constexpr,
    INDEX_HEAD_DIM: tl.constexpr,
    APPLY_RELU: tl.constexpr,
    SCORE_SCALE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    query_idx = tl.program_id(1)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    key_mask = offs_n < key_len
    score = tl.full((BLOCK_N,), 0.0, dtype=tl.float32)

    for head_idx in tl.range(0, INDEX_HEADS):
        q = tl.load(
            q_ptr
            + query_idx * q_stride_m
            + batch_idx * q_stride_b
            + head_idx * q_stride_h
            + offs_d * q_stride_d,
            mask=offs_d < INDEX_HEAD_DIM,
            other=0.0,
        ).to(tl.float32)
        k = tl.load(
            k_ptr
            + offs_n[:, None] * k_stride_s
            + batch_idx * k_stride_b
            + offs_d[None, :] * k_stride_d,
            mask=key_mask[:, None] & (offs_d[None, :] < INDEX_HEAD_DIM),
            other=0.0,
        ).to(tl.float32)
        dot = tl.sum(k * q[None, :], axis=1)
        if APPLY_RELU:
            dot = tl.maximum(dot, 0.0)
        weight = tl.load(
            weights_ptr
            + query_idx * w_stride_m
            + batch_idx * w_stride_b
            + head_idx * w_stride_h
        ).to(tl.float32)
        score += dot * weight

    score *= SCORE_SCALE

    query_position = q_start + query_idx
    key_position = k_start + offs_n
    score = tl.where(key_mask & (key_position <= query_position), score, -float("inf"))

    work = score
    for topk_idx in tl.range(0, topk):
        max_score = tl.max(work, axis=0)
        is_max = (work == max_score) & key_mask
        selected_rel = tl.min(tl.where(is_max, offs_n, BLOCK_N), axis=0)
        selected_rel = tl.minimum(selected_rel, key_len - 1)
        first_invalid = tl.min(
            tl.where(key_mask & (key_position > query_position), offs_n, BLOCK_N),
            axis=0,
        )
        selected_rel = tl.where(
            (max_score == -float("inf")) & (first_invalid < BLOCK_N),
            first_invalid,
            selected_rel,
        )
        tl.store(
            out_scores_ptr
            + batch_idx * out_score_stride_b
            + query_idx * out_score_stride_m
            + topk_idx * out_score_stride_k,
            max_score,
        )
        tl.store(
            out_indices_ptr
            + batch_idx * out_index_stride_b
            + query_idx * out_index_stride_m
            + topk_idx * out_index_stride_k,
            selected_rel + k_start,
        )
        work = tl.where(offs_n == selected_rel, -float("inf"), work)


@triton.autotune(
    configs=_linear_wgrad_autotune_configs(),
    key=[
        "total_rows",
        "out_features",
        "in_features",
        "USE_BF16_OPERANDS",
        "USE_FP16_OPERANDS",
    ],
)
@triton.jit
def _dsa_linear_wgrad_kernel(
    grad_output_ptr,
    input_ptr,
    out_delta_ptr,
    total_rows: tl.constexpr,
    out_features: tl.constexpr,
    in_features: tl.constexpr,
    go_stride_n: tl.constexpr,
    go_stride_o: tl.constexpr,
    in_stride_n: tl.constexpr,
    in_stride_i: tl.constexpr,
    out_stride_o: tl.constexpr,
    out_stride_i: tl.constexpr,
    USE_BF16_OPERANDS: tl.constexpr,
    USE_FP16_OPERANDS: tl.constexpr,
    BLOCK_O: tl.constexpr,
    BLOCK_I: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    out_block = tl.program_id(0)
    in_block = tl.program_id(1)
    offs_o = out_block * BLOCK_O + tl.arange(0, BLOCK_O)
    offs_i = in_block * BLOCK_I + tl.arange(0, BLOCK_I)
    offs_n = tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_O, BLOCK_I), dtype=tl.float32)
    for row_start in tl.range(0, total_rows, BLOCK_N):
        rows = row_start + offs_n
        grad_output = tl.load(
            grad_output_ptr + rows[:, None] * go_stride_n + offs_o[None, :] * go_stride_o,
            mask=(rows[:, None] < total_rows) & (offs_o[None, :] < out_features),
            other=0.0,
        )
        input = tl.load(
            input_ptr + rows[:, None] * in_stride_n + offs_i[None, :] * in_stride_i,
            mask=(rows[:, None] < total_rows) & (offs_i[None, :] < in_features),
            other=0.0,
        )
        if USE_BF16_OPERANDS:
            grad_output = grad_output.to(tl.bfloat16)
            input = input.to(tl.bfloat16)
        elif USE_FP16_OPERANDS:
            grad_output = grad_output.to(tl.float16)
            input = input.to(tl.float16)
        else:
            grad_output = grad_output.to(tl.float32)
            input = input.to(tl.float32)
        acc += tl.dot(
            tl.trans(grad_output),
            input,
            input_precision="tf32",
            out_dtype=tl.float32,
        )

    tl.store(
        out_delta_ptr + offs_o[:, None] * out_stride_o + offs_i[None, :] * out_stride_i,
        acc,
        mask=(offs_o[:, None] < out_features) & (offs_i[None, :] < in_features),
    )


@triton.autotune(
    configs=_selected_k_linear_autotune_configs(),
    key=[
        "total_rows",
        "query_len",
        "topk",
        "hidden_size",
        "out_features",
        "APPLY_INPUT_NORM",
        "USE_BF16_OPERANDS",
        "USE_FP16_OPERANDS",
    ],
)
@triton.jit
def _dsa_selected_k_linear_kernel(
    hidden_ptr,
    topk_indices_ptr,
    weight_ptr,
    input_norm_weight_ptr,
    input_norm_rstd_ptr,
    out_ptr,
    total_rows: tl.constexpr,
    query_len: tl.constexpr,
    topk: tl.constexpr,
    hidden_size: tl.constexpr,
    out_features: tl.constexpr,
    hidden_stride_s: tl.constexpr,
    hidden_stride_b: tl.constexpr,
    hidden_stride_h: tl.constexpr,
    ti_stride_b: tl.constexpr,
    ti_stride_m: tl.constexpr,
    ti_stride_k: tl.constexpr,
    weight_stride_o: tl.constexpr,
    weight_stride_i: tl.constexpr,
    inw_stride_h: tl.constexpr,
    inr_stride_s: tl.constexpr,
    inr_stride_b: tl.constexpr,
    out_stride_b: tl.constexpr,
    out_stride_m: tl.constexpr,
    out_stride_k: tl.constexpr,
    out_stride_d: tl.constexpr,
    USE_BF16_OPERANDS: tl.constexpr,
    USE_FP16_OPERANDS: tl.constexpr,
    APPLY_INPUT_NORM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row_block = tl.program_id(0)
    out_block = tl.program_id(1)
    rows = row_block * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = out_block * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_h = tl.arange(0, BLOCK_H)
    row_mask = rows < total_rows

    rows_per_batch = query_len * topk
    batch_idx = rows // rows_per_batch
    rem = rows - batch_idx * rows_per_batch
    query_idx = rem // topk
    support_idx = rem - query_idx * topk
    selected = tl.load(
        topk_indices_ptr
        + batch_idx * ti_stride_b
        + query_idx * ti_stride_m
        + support_idx * ti_stride_k,
        mask=row_mask,
        other=0,
    )
    if APPLY_INPUT_NORM:
        input_rstd = tl.load(
            input_norm_rstd_ptr
            + selected * inr_stride_s
            + batch_idx * inr_stride_b,
            mask=row_mask,
            other=0.0,
        ).to(tl.float32)

    acc = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    for hidden_start in tl.range(0, hidden_size, BLOCK_H):
        hidden_offsets = hidden_start + offs_h
        hidden = tl.load(
            hidden_ptr
            + selected[:, None] * hidden_stride_s
            + batch_idx[:, None] * hidden_stride_b
            + hidden_offsets[None, :] * hidden_stride_h,
            mask=row_mask[:, None] & (hidden_offsets[None, :] < hidden_size),
            other=0.0,
        )
        if APPLY_INPUT_NORM:
            input_norm_weight = tl.load(
                input_norm_weight_ptr + hidden_offsets * inw_stride_h,
                mask=hidden_offsets < hidden_size,
                other=0.0,
            ).to(tl.float32)
            hidden = hidden.to(tl.float32) * input_rstd[:, None] * input_norm_weight[None, :]
        weight = tl.load(
            weight_ptr
            + offs_d[None, :] * weight_stride_o
            + hidden_offsets[:, None] * weight_stride_i,
            mask=(hidden_offsets[:, None] < hidden_size) & (offs_d[None, :] < out_features),
            other=0.0,
        )
        if USE_BF16_OPERANDS:
            hidden = hidden.to(tl.bfloat16)
            weight = weight.to(tl.bfloat16)
        elif USE_FP16_OPERANDS:
            hidden = hidden.to(tl.float16)
            weight = weight.to(tl.float16)
        else:
            hidden = hidden.to(tl.float32)
            weight = weight.to(tl.float32)
        acc += tl.dot(hidden, weight, input_precision="tf32", out_dtype=tl.float32)

    tl.store(
        out_ptr
        + batch_idx[:, None] * out_stride_b
        + query_idx[:, None] * out_stride_m
        + support_idx[:, None] * out_stride_k
        + offs_d[None, :] * out_stride_d,
        acc,
        mask=row_mask[:, None] & (offs_d[None, :] < out_features),
    )


@triton.autotune(
    configs=_selected_k_score_autotune_configs(),
    key=[
        "total_rows",
        "query_len",
        "topk",
        "hidden_size",
        "out_features",
        "INDEX_HEADS",
        "INDEX_ROTARY_DIM",
        "USE_ROPE",
        "USE_HADAMARD",
        "HAS_BIAS",
        "APPLY_INPUT_NORM",
        "USE_BF16_OPERANDS",
        "USE_FP16_OPERANDS",
        "STORE_K_LINEAR",
    ],
)
@triton.jit
def _dsa_selected_k_project_score_kernel(
    hidden_ptr,
    topk_indices_ptr,
    linear_k_weight_ptr,
    input_norm_weight_ptr,
    input_norm_rstd_ptr,
    k_norm_weight_ptr,
    k_norm_bias_ptr,
    q_ptr,
    weights_ptr,
    inv_freq_ptr,
    hadamard_ptr,
    out_scores_ptr,
    out_k_linear_ptr,
    total_rows: tl.constexpr,
    query_len: tl.constexpr,
    topk: tl.constexpr,
    hidden_size: tl.constexpr,
    out_features: tl.constexpr,
    q_start,
    eps: tl.constexpr,
    mscale: tl.constexpr,
    interpolation_scale: tl.constexpr,
    hidden_stride_s: tl.constexpr,
    hidden_stride_b: tl.constexpr,
    hidden_stride_h: tl.constexpr,
    ti_stride_b: tl.constexpr,
    ti_stride_m: tl.constexpr,
    ti_stride_k: tl.constexpr,
    lkw_stride_o: tl.constexpr,
    lkw_stride_i: tl.constexpr,
    inw_stride_h: tl.constexpr,
    inr_stride_s: tl.constexpr,
    inr_stride_b: tl.constexpr,
    knw_stride_d: tl.constexpr,
    knb_stride_d: tl.constexpr,
    q_stride_m: tl.constexpr,
    q_stride_b: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_stride_d: tl.constexpr,
    w_stride_m: tl.constexpr,
    w_stride_b: tl.constexpr,
    w_stride_h: tl.constexpr,
    if_stride_d: tl.constexpr,
    hm_stride_i: tl.constexpr,
    hm_stride_o: tl.constexpr,
    os_stride_b: tl.constexpr,
    os_stride_m: tl.constexpr,
    os_stride_k: tl.constexpr,
    okl_stride_b: tl.constexpr,
    okl_stride_m: tl.constexpr,
    okl_stride_k: tl.constexpr,
    okl_stride_d: tl.constexpr,
    INDEX_HEADS: tl.constexpr,
    INDEX_ROTARY_DIM: tl.constexpr,
    ROTARY_INTERLEAVED: tl.constexpr,
    USE_ROPE: tl.constexpr,
    USE_HADAMARD: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    APPLY_INPUT_NORM: tl.constexpr,
    USE_BF16_OPERANDS: tl.constexpr,
    USE_FP16_OPERANDS: tl.constexpr,
    STORE_K_LINEAR: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row_block = tl.program_id(0)
    rows = row_block * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    offs_h = tl.arange(0, BLOCK_H)
    row_mask = rows < total_rows
    rows_per_batch = query_len * topk
    batch_idx = rows // rows_per_batch
    rem = rows - batch_idx * rows_per_batch
    query_idx = rem // topk
    support_idx = rem - query_idx * topk
    batch_idx_hidden = tl.broadcast_to(
        tl.expand_dims(batch_idx, 1), (BLOCK_N, BLOCK_H)
    )
    row_mask_hidden = tl.broadcast_to(
        tl.expand_dims(row_mask, 1), (BLOCK_N, BLOCK_H)
    )
    offs_d_weight = tl.broadcast_to(
        tl.expand_dims(offs_d, 0), (BLOCK_H, BLOCK_D)
    )
    batch_idx_feature = tl.broadcast_to(
        tl.expand_dims(batch_idx, 1), (BLOCK_N, BLOCK_D)
    )
    row_mask_feature = tl.broadcast_to(
        tl.expand_dims(row_mask, 1), (BLOCK_N, BLOCK_D)
    )
    offs_d_feature = tl.broadcast_to(
        tl.expand_dims(offs_d, 0), (BLOCK_N, BLOCK_D)
    )
    feature_mask = offs_d < out_features
    feature_mask_block = offs_d_feature < out_features
    selected = tl.load(
        topk_indices_ptr
        + batch_idx * ti_stride_b
        + query_idx * ti_stride_m
        + support_idx * ti_stride_k,
        mask=row_mask,
        other=0,
    )
    if APPLY_INPUT_NORM:
        input_rstd = tl.load(
            input_norm_rstd_ptr
            + selected * inr_stride_s
            + batch_idx * inr_stride_b,
            mask=row_mask,
            other=0.0,
        ).to(tl.float32)

    k_linear = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    for hidden_start in tl.range(0, hidden_size, BLOCK_H):
        hidden_offsets = hidden_start + offs_h
        hidden = tl.load(
            hidden_ptr
            + selected[:, None] * hidden_stride_s
            + batch_idx_hidden * hidden_stride_b
            + hidden_offsets[None, :] * hidden_stride_h,
            mask=row_mask_hidden & (hidden_offsets[None, :] < hidden_size),
            other=0.0,
        )
        if APPLY_INPUT_NORM:
            input_norm_weight = tl.load(
                input_norm_weight_ptr + hidden_offsets * inw_stride_h,
                mask=hidden_offsets < hidden_size,
                other=0.0,
            ).to(tl.float32)
            hidden = hidden.to(tl.float32) * input_rstd[:, None] * input_norm_weight[None, :]
        linear_k_weight = tl.load(
            linear_k_weight_ptr
            + offs_d_weight * lkw_stride_o
            + hidden_offsets[:, None] * lkw_stride_i,
            mask=(hidden_offsets[:, None] < hidden_size)
            & (offs_d_weight < out_features),
            other=0.0,
        )
        if USE_BF16_OPERANDS:
            hidden = hidden.to(tl.bfloat16)
            linear_k_weight = linear_k_weight.to(tl.bfloat16)
        elif USE_FP16_OPERANDS:
            hidden = hidden.to(tl.float16)
            linear_k_weight = linear_k_weight.to(tl.float16)
        else:
            hidden = hidden.to(tl.float32)
            linear_k_weight = linear_k_weight.to(tl.float32)
        k_linear += tl.dot(hidden, linear_k_weight, input_precision="tf32", out_dtype=tl.float32)

    if USE_BF16_OPERANDS:
        k_linear = k_linear.to(tl.bfloat16).to(tl.float32)
    elif USE_FP16_OPERANDS:
        k_linear = k_linear.to(tl.float16).to(tl.float32)

    if STORE_K_LINEAR:
        tl.store(
            out_k_linear_ptr
            + batch_idx_feature * okl_stride_b
            + query_idx[:, None] * okl_stride_m
            + support_idx[:, None] * okl_stride_k
            + offs_d_feature * okl_stride_d,
            k_linear,
            mask=row_mask_feature & feature_mask_block,
        )

    inv_features = 1.0 / out_features
    mean = tl.sum(tl.where(feature_mask_block, k_linear, 0.0), axis=1) * inv_features
    centered = tl.where(feature_mask_block, k_linear - mean[:, None], 0.0)
    variance = tl.sum(centered * centered, axis=1) * inv_features
    rstd = tl.rsqrt(variance + eps)
    norm_weight = tl.load(
        k_norm_weight_ptr + offs_d * knw_stride_d,
        mask=feature_mask,
        other=0.0,
    ).to(tl.float32)
    k_index = centered * rstd[:, None] * norm_weight[None, :]
    if HAS_BIAS:
        norm_bias = tl.load(
            k_norm_bias_ptr + offs_d * knb_stride_d,
            mask=feature_mask,
            other=0.0,
        ).to(tl.float32)
        k_index += norm_bias[None, :]
    if USE_BF16_OPERANDS:
        k_index = k_index.to(tl.bfloat16).to(tl.float32)
    elif USE_FP16_OPERANDS:
        k_index = k_index.to(tl.float16).to(tl.float32)

    if USE_ROPE:
        nope = out_features - INDEX_ROTARY_DIM
        rotary_half = INDEX_ROTARY_DIM // 2
        rotary_pos = offs_d - nope
        is_rotary = (offs_d >= nope) & (offs_d < out_features)
        if ROTARY_INTERLEAVED:
            is_first = (rotary_pos % 2) == 0
            partner_pos = tl.where(is_first, rotary_pos + 1, rotary_pos - 1)
            inv_idx = rotary_pos // 2
        else:
            is_first = rotary_pos < rotary_half
            partner_pos = tl.where(is_first, rotary_pos + rotary_half, rotary_pos - rotary_half)
            inv_idx = tl.where(is_first, rotary_pos, rotary_pos - rotary_half)
        partner_d = nope + partner_pos
        partner_linear = tl.load(
            out_k_linear_ptr
            + batch_idx_feature * okl_stride_b
            + query_idx[:, None] * okl_stride_m
            + support_idx[:, None] * okl_stride_k
            + partner_d[None, :] * okl_stride_d,
            mask=row_mask_feature
            & is_rotary[None, :]
            & (partner_d[None, :] >= 0)
            & (partner_d[None, :] < out_features),
            other=0.0,
        ).to(tl.float32)
        partner_centered = partner_linear - mean[:, None]
        partner_weight = tl.load(
            k_norm_weight_ptr + partner_d * knw_stride_d,
            mask=is_rotary & (partner_d >= 0) & (partner_d < out_features),
            other=0.0,
        ).to(tl.float32)
        partner_value = partner_centered * rstd[:, None] * partner_weight[None, :]
        if HAS_BIAS:
            partner_bias = tl.load(
                k_norm_bias_ptr + partner_d * knb_stride_d,
                mask=is_rotary & (partner_d >= 0) & (partner_d < out_features),
                other=0.0,
            ).to(tl.float32)
            partner_value += partner_bias[None, :]
        inv_freq = tl.load(
            inv_freq_ptr + inv_idx * if_stride_d,
            mask=is_rotary & (inv_idx >= 0) & (inv_idx < rotary_half),
            other=0.0,
        ).to(tl.float32)
        pos = selected.to(tl.float32) * interpolation_scale
        freqs = pos[:, None] * inv_freq[None, :]
        cos = tl.cos(freqs) * mscale
        sin = tl.sin(freqs) * mscale
        rotated_half = tl.where(is_first[None, :], -partner_value, partner_value)
        k_rope = k_index * cos + rotated_half * sin
        k_index = tl.where(is_rotary[None, :], k_rope, k_index)
        if USE_BF16_OPERANDS:
            k_index = k_index.to(tl.bfloat16).to(tl.float32)
        elif USE_FP16_OPERANDS:
            k_index = k_index.to(tl.float16).to(tl.float32)

    if USE_HADAMARD:
        hadamard = tl.load(
            hadamard_ptr + offs_d[:, None] * hm_stride_i + offs_d[None, :] * hm_stride_o,
            mask=feature_mask[:, None] & feature_mask[None, :],
            other=0.0,
        )
        k_index = tl.dot(
            k_index.to(tl.bfloat16),
            hadamard,
            input_precision="tf32",
            out_dtype=tl.float32,
        )
        k_index = k_index.to(tl.bfloat16).to(tl.float32)

    score = tl.full((BLOCK_N,), 0.0, dtype=tl.float32)
    for head_idx in tl.range(0, INDEX_HEADS):
        q = tl.load(
            q_ptr
            + query_idx[:, None] * q_stride_m
            + batch_idx_feature * q_stride_b
            + head_idx * q_stride_h
            + offs_d_feature * q_stride_d,
            mask=row_mask_feature & feature_mask_block,
            other=0.0,
        ).to(tl.float32)
        dot = tl.sum(q * k_index, axis=1)
        dot = tl.maximum(dot, 0.0)
        weight = tl.load(
            weights_ptr
            + query_idx * w_stride_m
            + batch_idx * w_stride_b
            + head_idx * w_stride_h,
            mask=row_mask,
            other=0.0,
        ).to(tl.float32)
        score += dot * weight

    valid = row_mask & (selected <= q_start + query_idx)
    score = tl.where(valid, score, -float("inf"))
    tl.store(
        out_scores_ptr
        + batch_idx * os_stride_b
        + query_idx * os_stride_m
        + support_idx * os_stride_k,
        score,
        mask=row_mask,
    )


@triton.autotune(
    configs=_k_ln_backward_autotune_configs(),
    key=[
        "total_rows",
        "out_features",
        "query_len",
        "topk",
        "HAS_WEIGHT_GRAD",
        "HAS_BIAS_GRAD",
    ],
    reset_to_zero=["partial_weight_ptr", "partial_bias_ptr"],
)
@triton.jit
def _dsa_k_ln_backward_kernel(
    grad_k_norm_ptr,
    k_linear_ptr,
    k_norm_weight_ptr,
    grad_k_linear_ptr,
    partial_weight_ptr,
    partial_bias_ptr,
    total_rows: tl.constexpr,
    query_len: tl.constexpr,
    topk: tl.constexpr,
    out_features: tl.constexpr,
    eps: tl.constexpr,
    gkn_stride_b: tl.constexpr,
    gkn_stride_m: tl.constexpr,
    gkn_stride_k: tl.constexpr,
    gkn_stride_d: tl.constexpr,
    kl_stride_b: tl.constexpr,
    kl_stride_m: tl.constexpr,
    kl_stride_k: tl.constexpr,
    kl_stride_d: tl.constexpr,
    weight_stride_d: tl.constexpr,
    gkl_stride_b: tl.constexpr,
    gkl_stride_m: tl.constexpr,
    gkl_stride_k: tl.constexpr,
    gkl_stride_d: tl.constexpr,
    pw_stride_n: tl.constexpr,
    pw_stride_d: tl.constexpr,
    pb_stride_n: tl.constexpr,
    pb_stride_d: tl.constexpr,
    HAS_WEIGHT_GRAD: tl.constexpr,
    HAS_BIAS_GRAD: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_D_FULL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row_block = tl.program_id(0)
    d_block = tl.program_id(1)
    rows = row_block * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_d_full = tl.arange(0, BLOCK_D_FULL)
    row_mask = rows < total_rows
    rows_per_batch = query_len * topk
    batch_idx = rows // rows_per_batch
    rem = rows - batch_idx * rows_per_batch
    query_idx = rem // topk
    support_idx = rem - query_idx * topk
    inv_features = 1.0 / out_features

    k_full = tl.load(
        k_linear_ptr
        + batch_idx[:, None] * kl_stride_b
        + query_idx[:, None] * kl_stride_m
        + support_idx[:, None] * kl_stride_k
        + offs_d_full[None, :] * kl_stride_d,
        mask=row_mask[:, None] & (offs_d_full[None, :] < out_features),
        other=0.0,
    ).to(tl.float32)
    mean = tl.sum(k_full, axis=1) * inv_features
    centered_full = tl.where(
        offs_d_full[None, :] < out_features,
        k_full - mean[:, None],
        0.0,
    )
    variance = tl.sum(centered_full * centered_full, axis=1) * inv_features
    rstd = tl.rsqrt(variance + eps)
    normalized_full = centered_full * rstd[:, None]

    grad_full = tl.load(
        grad_k_norm_ptr
        + batch_idx[:, None] * gkn_stride_b
        + query_idx[:, None] * gkn_stride_m
        + support_idx[:, None] * gkn_stride_k
        + offs_d_full[None, :] * gkn_stride_d,
        mask=row_mask[:, None] & (offs_d_full[None, :] < out_features),
        other=0.0,
    ).to(tl.float32)
    norm_weight_full = tl.load(
        k_norm_weight_ptr + offs_d_full * weight_stride_d,
        mask=offs_d_full < out_features,
        other=0.0,
    ).to(tl.float32)
    grad_normalized_full = grad_full * norm_weight_full[None, :]
    mean_grad = tl.sum(grad_normalized_full, axis=1) * inv_features
    mean_grad_norm = tl.sum(grad_normalized_full * normalized_full, axis=1) * inv_features

    k = tl.load(
        k_linear_ptr
        + batch_idx[:, None] * kl_stride_b
        + query_idx[:, None] * kl_stride_m
        + support_idx[:, None] * kl_stride_k
        + offs_d[None, :] * kl_stride_d,
        mask=row_mask[:, None] & (offs_d[None, :] < out_features),
        other=0.0,
    ).to(tl.float32)
    grad = tl.load(
        grad_k_norm_ptr
        + batch_idx[:, None] * gkn_stride_b
        + query_idx[:, None] * gkn_stride_m
        + support_idx[:, None] * gkn_stride_k
        + offs_d[None, :] * gkn_stride_d,
        mask=row_mask[:, None] & (offs_d[None, :] < out_features),
        other=0.0,
    ).to(tl.float32)
    norm_weight = tl.load(
        k_norm_weight_ptr + offs_d * weight_stride_d,
        mask=offs_d < out_features,
        other=0.0,
    ).to(tl.float32)
    normalized = (k - mean[:, None]) * rstd[:, None]
    grad_normalized = grad * norm_weight[None, :]
    grad_k_linear = (
        grad_normalized - mean_grad[:, None] - normalized * mean_grad_norm[:, None]
    ) * rstd[:, None]
    valid = row_mask[:, None] & (offs_d[None, :] < out_features)
    grad_k_linear = tl.where(valid, grad_k_linear, 0.0)
    tl.store(
        grad_k_linear_ptr
        + batch_idx[:, None] * gkl_stride_b
        + query_idx[:, None] * gkl_stride_m
        + support_idx[:, None] * gkl_stride_k
        + offs_d[None, :] * gkl_stride_d,
        grad_k_linear,
        mask=valid,
    )

    grad = tl.where(valid, grad, 0.0)
    normalized = tl.where(valid, normalized, 0.0)
    if HAS_WEIGHT_GRAD:
        tl.store(
            partial_weight_ptr + row_block * pw_stride_n + offs_d * pw_stride_d,
            tl.sum(grad * normalized, axis=0),
            mask=offs_d < out_features,
        )
    if HAS_BIAS_GRAD:
        tl.store(
            partial_bias_ptr + row_block * pb_stride_n + offs_d * pb_stride_d,
            tl.sum(grad, axis=0),
            mask=offs_d < out_features,
        )


@triton.autotune(
    configs=_gathered_linear_wgrad_autotune_configs(),
    key=[
        "total_rows",
        "out_features",
        "hidden_size",
        "query_len",
        "topk",
        "USE_BF16_OPERANDS",
        "USE_FP16_OPERANDS",
    ],
)
@triton.jit
def _dsa_gathered_linear_wgrad_kernel(
    grad_output_ptr,
    hidden_ptr,
    topk_indices_ptr,
    out_delta_ptr,
    total_rows: tl.constexpr,
    query_len: tl.constexpr,
    topk: tl.constexpr,
    out_features: tl.constexpr,
    hidden_size: tl.constexpr,
    go_stride_b: tl.constexpr,
    go_stride_m: tl.constexpr,
    go_stride_k: tl.constexpr,
    go_stride_o: tl.constexpr,
    hidden_stride_s: tl.constexpr,
    hidden_stride_b: tl.constexpr,
    hidden_stride_h: tl.constexpr,
    ti_stride_b: tl.constexpr,
    ti_stride_m: tl.constexpr,
    ti_stride_k: tl.constexpr,
    out_stride_o: tl.constexpr,
    out_stride_i: tl.constexpr,
    USE_BF16_OPERANDS: tl.constexpr,
    USE_FP16_OPERANDS: tl.constexpr,
    BLOCK_O: tl.constexpr,
    BLOCK_I: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    out_block = tl.program_id(0)
    in_block = tl.program_id(1)
    offs_o = out_block * BLOCK_O + tl.arange(0, BLOCK_O)
    offs_i = in_block * BLOCK_I + tl.arange(0, BLOCK_I)
    offs_n = tl.arange(0, BLOCK_N)
    rows_per_batch = query_len * topk

    acc = tl.zeros((BLOCK_O, BLOCK_I), dtype=tl.float32)
    for row_start in tl.range(0, total_rows, BLOCK_N):
        rows = row_start + offs_n
        row_mask = rows < total_rows
        batch_idx = rows // rows_per_batch
        rem = rows - batch_idx * rows_per_batch
        query_idx = rem // topk
        support_idx = rem - query_idx * topk
        selected = tl.load(
            topk_indices_ptr
            + batch_idx * ti_stride_b
            + query_idx * ti_stride_m
            + support_idx * ti_stride_k,
            mask=row_mask,
            other=0,
        )
        grad_output = tl.load(
            grad_output_ptr
            + batch_idx[:, None] * go_stride_b
            + query_idx[:, None] * go_stride_m
            + support_idx[:, None] * go_stride_k
            + offs_o[None, :] * go_stride_o,
            mask=row_mask[:, None] & (offs_o[None, :] < out_features),
            other=0.0,
        )
        hidden = tl.load(
            hidden_ptr
            + selected[:, None] * hidden_stride_s
            + batch_idx[:, None] * hidden_stride_b
            + offs_i[None, :] * hidden_stride_h,
            mask=row_mask[:, None] & (offs_i[None, :] < hidden_size),
            other=0.0,
        )
        if USE_BF16_OPERANDS:
            grad_output = grad_output.to(tl.bfloat16)
            hidden = hidden.to(tl.bfloat16)
        elif USE_FP16_OPERANDS:
            grad_output = grad_output.to(tl.float16)
            hidden = hidden.to(tl.float16)
        else:
            grad_output = grad_output.to(tl.float32)
            hidden = hidden.to(tl.float32)
        acc += tl.dot(
            tl.trans(grad_output),
            hidden,
            input_precision="tf32",
            out_dtype=tl.float32,
        )

    tl.store(
        out_delta_ptr + offs_o[:, None] * out_stride_o + offs_i[None, :] * out_stride_i,
        acc,
        mask=(offs_o[:, None] < out_features) & (offs_i[None, :] < hidden_size),
    )


@triton.jit
def _dsa_simplified_input_norm_stats_kernel(
    hidden_ptr,
    rstd_ptr,
    total_rows: tl.constexpr,
    batch_size: tl.constexpr,
    hidden_size: tl.constexpr,
    hidden_stride_s: tl.constexpr,
    hidden_stride_b: tl.constexpr,
    hidden_stride_h: tl.constexpr,
    stats_stride_s: tl.constexpr,
    stats_stride_b: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    sequence_idx = row // batch_size
    batch_idx = row - sequence_idx * batch_size
    offs_h = tl.arange(0, BLOCK_H)
    mask = (row < total_rows) & (offs_h < hidden_size)
    hidden = tl.load(
        hidden_ptr
        + sequence_idx * hidden_stride_s
        + batch_idx * hidden_stride_b
        + offs_h * hidden_stride_h,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    inv_hidden = 1.0 / hidden_size
    variance = tl.sum(hidden * hidden, axis=0) * inv_hidden
    rstd = tl.rsqrt(variance + eps)
    stats_offset = sequence_idx * stats_stride_s + batch_idx * stats_stride_b
    tl.store(rstd_ptr + stats_offset, rstd, mask=row < total_rows)


@triton.autotune(
    configs=_gathered_linear_wgrad_autotune_configs(),
    key=[
        "total_rows",
        "out_features",
        "hidden_size",
        "query_len",
        "topk",
        "USE_BF16_OPERANDS",
        "USE_FP16_OPERANDS",
    ],
)
@triton.jit
def _dsa_simplified_gathered_linear_wgrad_kernel(
    grad_output_ptr,
    hidden_ptr,
    topk_indices_ptr,
    norm_weight_ptr,
    rstd_ptr,
    out_delta_ptr,
    total_rows: tl.constexpr,
    query_len: tl.constexpr,
    topk: tl.constexpr,
    out_features: tl.constexpr,
    hidden_size: tl.constexpr,
    go_stride_b: tl.constexpr,
    go_stride_m: tl.constexpr,
    go_stride_k: tl.constexpr,
    go_stride_o: tl.constexpr,
    hidden_stride_s: tl.constexpr,
    hidden_stride_b: tl.constexpr,
    hidden_stride_h: tl.constexpr,
    ti_stride_b: tl.constexpr,
    ti_stride_m: tl.constexpr,
    ti_stride_k: tl.constexpr,
    nw_stride_h: tl.constexpr,
    stats_stride_s: tl.constexpr,
    stats_stride_b: tl.constexpr,
    out_stride_o: tl.constexpr,
    out_stride_i: tl.constexpr,
    USE_BF16_OPERANDS: tl.constexpr,
    USE_FP16_OPERANDS: tl.constexpr,
    BLOCK_O: tl.constexpr,
    BLOCK_I: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    out_block = tl.program_id(0)
    in_block = tl.program_id(1)
    offs_o = out_block * BLOCK_O + tl.arange(0, BLOCK_O)
    offs_i = in_block * BLOCK_I + tl.arange(0, BLOCK_I)
    offs_n = tl.arange(0, BLOCK_N)
    rows_per_batch = query_len * topk

    acc = tl.zeros((BLOCK_O, BLOCK_I), dtype=tl.float32)
    for row_start in tl.range(0, total_rows, BLOCK_N):
        rows = row_start + offs_n
        row_mask = rows < total_rows
        batch_idx = rows // rows_per_batch
        rem = rows - batch_idx * rows_per_batch
        query_idx = rem // topk
        support_idx = rem - query_idx * topk
        selected = tl.load(
            topk_indices_ptr
            + batch_idx * ti_stride_b
            + query_idx * ti_stride_m
            + support_idx * ti_stride_k,
            mask=row_mask,
            other=0,
        )
        grad_output = tl.load(
            grad_output_ptr
            + batch_idx[:, None] * go_stride_b
            + query_idx[:, None] * go_stride_m
            + support_idx[:, None] * go_stride_k
            + offs_o[None, :] * go_stride_o,
            mask=row_mask[:, None] & (offs_o[None, :] < out_features),
            other=0.0,
        )
        hidden = tl.load(
            hidden_ptr
            + selected[:, None] * hidden_stride_s
            + batch_idx[:, None] * hidden_stride_b
            + offs_i[None, :] * hidden_stride_h,
            mask=row_mask[:, None] & (offs_i[None, :] < hidden_size),
            other=0.0,
        ).to(tl.float32)
        stats_offset = selected * stats_stride_s + batch_idx * stats_stride_b
        rstd = tl.load(rstd_ptr + stats_offset, mask=row_mask, other=0.0).to(tl.float32)
        norm_weight = tl.load(
            norm_weight_ptr + offs_i * nw_stride_h,
            mask=offs_i < hidden_size,
            other=0.0,
        ).to(tl.float32)
        hidden = hidden * rstd[:, None] * norm_weight[None, :]
        if USE_BF16_OPERANDS:
            grad_output = grad_output.to(tl.bfloat16)
            hidden = hidden.to(tl.bfloat16)
        elif USE_FP16_OPERANDS:
            grad_output = grad_output.to(tl.float16)
            hidden = hidden.to(tl.float16)
        else:
            grad_output = grad_output.to(tl.float32)
            hidden = hidden.to(tl.float32)
        acc += tl.dot(
            tl.trans(grad_output),
            hidden,
            input_precision="tf32",
            out_dtype=tl.float32,
        )

    tl.store(
        out_delta_ptr + offs_o[:, None] * out_stride_o + offs_i[None, :] * out_stride_i,
        acc,
        mask=(offs_o[:, None] < out_features) & (offs_i[None, :] < hidden_size),
    )


@triton.autotune(
    configs=_scatter_selected_grad_autotune_configs(),
    key=["total_rows", "sequence_length", "query_len", "topk", "out_features"],
    reset_to_zero=["out_ptr"],
)
@triton.jit
def _dsa_scatter_selected_grad_to_sequence_kernel(
    grad_output_ptr,
    topk_indices_ptr,
    out_ptr,
    total_rows: tl.constexpr,
    sequence_length: tl.constexpr,
    query_len: tl.constexpr,
    topk: tl.constexpr,
    out_features: tl.constexpr,
    go_stride_b: tl.constexpr,
    go_stride_m: tl.constexpr,
    go_stride_k: tl.constexpr,
    go_stride_o: tl.constexpr,
    ti_stride_b: tl.constexpr,
    ti_stride_m: tl.constexpr,
    ti_stride_k: tl.constexpr,
    out_stride_s: tl.constexpr,
    out_stride_b: tl.constexpr,
    out_stride_o: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row_block = tl.program_id(0)
    dim_block = tl.program_id(1)
    rows = row_block * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = dim_block * BLOCK_D + tl.arange(0, BLOCK_D)
    row_mask = rows < total_rows
    rows_per_batch = query_len * topk
    batch_idx = rows // rows_per_batch
    rem = rows - batch_idx * rows_per_batch
    query_idx = rem // topk
    support_idx = rem - query_idx * topk
    selected = tl.load(
        topk_indices_ptr
        + batch_idx * ti_stride_b
        + query_idx * ti_stride_m
        + support_idx * ti_stride_k,
        mask=row_mask,
        other=-1,
    )
    dim_mask = offs_d < out_features
    valid = row_mask & (selected >= 0) & (selected < sequence_length)
    safe_selected = tl.where(valid, selected, 0)
    grad = tl.load(
        grad_output_ptr
        + batch_idx[:, None] * go_stride_b
        + query_idx[:, None] * go_stride_m
        + support_idx[:, None] * go_stride_k
        + offs_d[None, :] * go_stride_o,
        mask=valid[:, None] & dim_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    tl.atomic_add(
        out_ptr
        + safe_selected[:, None] * out_stride_s
        + batch_idx[:, None] * out_stride_b
        + offs_d[None, :] * out_stride_o,
        grad,
        mask=valid[:, None] & dim_mask[None, :],
    )
@triton.autotune(
    configs=_topk_tiled_autotune_configs(),
    key=["query_len", "key_len", "topk", "INDEX_HEADS", "INDEX_HEAD_DIM"],
)
@triton.jit
def _dsa_topk_index_block_tiled_kernel(
    q_ptr,
    weights_ptr,
    k_ptr,
    out_scores_ptr,
    out_indices_ptr,
    q_start,
    k_start,
    query_len: tl.constexpr,
    key_len: tl.constexpr,
    topk: tl.constexpr,
    q_stride_m: tl.constexpr,
    q_stride_b: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_stride_d: tl.constexpr,
    w_stride_m: tl.constexpr,
    w_stride_b: tl.constexpr,
    w_stride_h: tl.constexpr,
    k_stride_s: tl.constexpr,
    k_stride_b: tl.constexpr,
    k_stride_d: tl.constexpr,
    out_score_stride_b: tl.constexpr,
    out_score_stride_m: tl.constexpr,
    out_score_stride_k: tl.constexpr,
    out_index_stride_b: tl.constexpr,
    out_index_stride_m: tl.constexpr,
    out_index_stride_k: tl.constexpr,
    INDEX_HEADS: tl.constexpr,
    INDEX_HEAD_DIM: tl.constexpr,
    APPLY_RELU: tl.constexpr,
    SCORE_SCALE: tl.constexpr,
    DOT_INPUT_PRECISION: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_D_TILE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    query_block = tl.program_id(1)
    offs_m = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_dt = tl.arange(0, BLOCK_D_TILE)
    query_mask = offs_m < query_len
    key_mask = offs_n < key_len
    score = tl.full((BLOCK_M, BLOCK_N), 0.0, dtype=tl.float32)

    for head_idx in tl.range(0, INDEX_HEADS):
        dot = tl.full((BLOCK_M, BLOCK_N), 0.0, dtype=tl.float32)
        for d_start in tl.range(0, BLOCK_D, BLOCK_D_TILE):
            offs_d = d_start + offs_dt
            q = tl.load(
                q_ptr
                + offs_m[:, None] * q_stride_m
                + batch_idx * q_stride_b
                + head_idx * q_stride_h
                + offs_d[None, :] * q_stride_d,
                mask=query_mask[:, None] & (offs_d[None, :] < INDEX_HEAD_DIM),
                other=0.0,
            )
            k = tl.load(
                k_ptr
                + offs_n[:, None] * k_stride_s
                + batch_idx * k_stride_b
                + offs_d[None, :] * k_stride_d,
                mask=key_mask[:, None] & (offs_d[None, :] < INDEX_HEAD_DIM),
                other=0.0,
            )
            dot += tl.dot(
                q,
                tl.trans(k),
                input_precision=DOT_INPUT_PRECISION,
                out_dtype=tl.float32,
            )
        if APPLY_RELU:
            dot = tl.maximum(dot, 0.0)
        weight = tl.load(
            weights_ptr
            + offs_m * w_stride_m
            + batch_idx * w_stride_b
            + head_idx * w_stride_h,
            mask=query_mask,
            other=0.0,
        ).to(tl.float32)
        score += dot * weight[:, None]

    score *= SCORE_SCALE

    query_position = q_start + offs_m
    key_position = k_start + offs_n
    valid = query_mask[:, None] & key_mask[None, :] & (key_position[None, :] <= query_position[:, None])
    work = tl.where(valid, score, -float("inf"))

    if topk == key_len:
        tl.store(
            out_scores_ptr
            + batch_idx * out_score_stride_b
            + offs_m[:, None] * out_score_stride_m
            + offs_n[None, :] * out_score_stride_k,
            work,
            mask=query_mask[:, None] & key_mask[None, :],
        )
        tl.store(
            out_indices_ptr
            + batch_idx * out_index_stride_b
            + offs_m[:, None] * out_index_stride_m
            + offs_n[None, :] * out_index_stride_k,
            offs_n[None, :] + k_start,
            mask=query_mask[:, None] & key_mask[None, :],
        )
    else:
        for topk_idx in tl.range(0, topk):
            max_score = tl.max(work, axis=1)
            is_max = (work == max_score[:, None]) & key_mask[None, :]
            selected_rel = tl.min(tl.where(is_max, offs_n[None, :], BLOCK_N), axis=1)
            selected_rel = tl.minimum(selected_rel, key_len - 1)
            first_invalid = tl.min(
                tl.where(
                    key_mask[None, :] & (key_position[None, :] > query_position[:, None]),
                    offs_n[None, :],
                    BLOCK_N,
                ),
                axis=1,
            )
            selected_rel = tl.where(
                (max_score == -float("inf")) & (first_invalid < BLOCK_N),
                first_invalid,
                selected_rel,
            )
            tl.store(
                out_scores_ptr
                + batch_idx * out_score_stride_b
                + offs_m * out_score_stride_m
                + topk_idx * out_score_stride_k,
                max_score,
                mask=query_mask,
            )
            tl.store(
                out_indices_ptr
                + batch_idx * out_index_stride_b
                + offs_m * out_index_stride_m
                + topk_idx * out_index_stride_k,
                selected_rel + k_start,
                mask=query_mask,
            )
            work = tl.where(offs_n[None, :] == selected_rel[:, None], -float("inf"), work)


@triton.autotune(
    configs=_simplified_selected_scores_autotune_configs(),
    key=["query_len", "topk", "HEAD_DIM"],
)
@triton.jit
def _dsa_simplified_selected_scores_kernel(
    q_ptr,
    key_ptr,
    topk_indices_ptr,
    out_scores_ptr,
    q_start,
    query_len: tl.constexpr,
    topk: tl.constexpr,
    q_stride_m: tl.constexpr,
    q_stride_b: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_stride_d: tl.constexpr,
    k_stride_s: tl.constexpr,
    k_stride_b: tl.constexpr,
    k_stride_g: tl.constexpr,
    k_stride_d: tl.constexpr,
    ti_stride_b: tl.constexpr,
    ti_stride_m: tl.constexpr,
    ti_stride_k: tl.constexpr,
    out_stride_b: tl.constexpr,
    out_stride_m: tl.constexpr,
    out_stride_k: tl.constexpr,
    SCORE_SCALE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    query_idx = tl.program_id(1)
    support_block = tl.program_id(2)
    offs_k = support_block * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, BLOCK_D)
    support_mask = offs_k < topk
    selected = tl.load(
        topk_indices_ptr
        + batch_idx * ti_stride_b
        + query_idx * ti_stride_m
        + offs_k * ti_stride_k,
        mask=support_mask,
        other=0,
    )
    valid = support_mask & (selected <= q_start + query_idx)
    q = tl.load(
        q_ptr
        + query_idx * q_stride_m
        + batch_idx * q_stride_b
        + offs_d * q_stride_d,
        mask=offs_d < HEAD_DIM,
        other=0.0,
    ).to(tl.float32)
    key = tl.load(
        key_ptr
        + selected[:, None] * k_stride_s
        + batch_idx * k_stride_b
        + offs_d[None, :] * k_stride_d,
        mask=valid[:, None] & (offs_d[None, :] < HEAD_DIM),
        other=0.0,
    ).to(tl.float32)
    scores = tl.sum(key * q[None, :], axis=1) * SCORE_SCALE
    scores = tl.where(valid, scores, -float("inf"))
    tl.store(
        out_scores_ptr
        + batch_idx * out_stride_b
        + query_idx * out_stride_m
        + offs_k * out_stride_k,
        scores,
        mask=support_mask,
    )


@triton.autotune(
    configs=_simplified_selected_scores_autotune_configs(),
    key=["query_len", "topk", "HEAD_DIM"],
)
@triton.jit
def _dsa_simplified_selected_scores_backward_kernel(
    key_ptr,
    topk_indices_ptr,
    grad_scores_ptr,
    grad_q_ptr,
    q_start,
    query_len: tl.constexpr,
    topk: tl.constexpr,
    k_stride_s: tl.constexpr,
    k_stride_b: tl.constexpr,
    k_stride_g: tl.constexpr,
    k_stride_d: tl.constexpr,
    ti_stride_b: tl.constexpr,
    ti_stride_m: tl.constexpr,
    ti_stride_k: tl.constexpr,
    gs_stride_b: tl.constexpr,
    gs_stride_m: tl.constexpr,
    gs_stride_k: tl.constexpr,
    gq_stride_m: tl.constexpr,
    gq_stride_b: tl.constexpr,
    gq_stride_h: tl.constexpr,
    gq_stride_d: tl.constexpr,
    SCORE_SCALE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    query_idx = tl.program_id(1)
    offs_d = tl.arange(0, BLOCK_D)
    grad_q = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for support_start in range(0, topk, BLOCK_K):
        offs_k = support_start + tl.arange(0, BLOCK_K)
        support_mask = offs_k < topk
        selected = tl.load(
            topk_indices_ptr
            + batch_idx * ti_stride_b
            + query_idx * ti_stride_m
            + offs_k * ti_stride_k,
            mask=support_mask,
            other=0,
        )
        valid = support_mask & (selected <= q_start + query_idx)
        grad_scores = tl.load(
            grad_scores_ptr
            + batch_idx * gs_stride_b
            + query_idx * gs_stride_m
            + offs_k * gs_stride_k,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        key = tl.load(
            key_ptr
            + selected[:, None] * k_stride_s
            + batch_idx * k_stride_b
            + offs_d[None, :] * k_stride_d,
            mask=valid[:, None] & (offs_d[None, :] < HEAD_DIM),
            other=0.0,
        ).to(tl.float32)
        grad_q += tl.sum(key * grad_scores[:, None], axis=0)
    tl.store(
        grad_q_ptr
        + query_idx * gq_stride_m
        + batch_idx * gq_stride_b
        + offs_d * gq_stride_d,
        grad_q * SCORE_SCALE,
        mask=offs_d < HEAD_DIM,
    )


@triton.autotune(
    configs=_simplified_selected_scores_autotune_configs(),
    key=["query_len", "topk", "HEAD_DIM"],
    reset_to_zero=["grad_q_ptr"],
)
@triton.jit
def _dsa_simplified_selected_scores_backward_qk_kernel(
    q_ptr,
    selected_k_ptr,
    topk_indices_ptr,
    grad_scores_ptr,
    grad_q_ptr,
    grad_selected_k_ptr,
    q_start,
    query_len: tl.constexpr,
    topk: tl.constexpr,
    q_stride_m: tl.constexpr,
    q_stride_b: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_stride_d: tl.constexpr,
    sk_stride_b: tl.constexpr,
    sk_stride_m: tl.constexpr,
    sk_stride_k: tl.constexpr,
    sk_stride_d: tl.constexpr,
    ti_stride_b: tl.constexpr,
    ti_stride_m: tl.constexpr,
    ti_stride_k: tl.constexpr,
    gs_stride_b: tl.constexpr,
    gs_stride_m: tl.constexpr,
    gs_stride_k: tl.constexpr,
    gq_stride_m: tl.constexpr,
    gq_stride_b: tl.constexpr,
    gq_stride_h: tl.constexpr,
    gq_stride_d: tl.constexpr,
    gsk_stride_b: tl.constexpr,
    gsk_stride_m: tl.constexpr,
    gsk_stride_k: tl.constexpr,
    gsk_stride_d: tl.constexpr,
    SCORE_SCALE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    query_idx = tl.program_id(1)
    support_block = tl.program_id(2)
    offs_k = support_block * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, BLOCK_D)
    support_mask = offs_k < topk
    selected_positions = tl.load(
        topk_indices_ptr
        + batch_idx * ti_stride_b
        + query_idx * ti_stride_m
        + offs_k * ti_stride_k,
        mask=support_mask,
        other=0,
    )
    valid = support_mask & (selected_positions <= q_start + query_idx)
    grad_scores = tl.load(
        grad_scores_ptr
        + batch_idx * gs_stride_b
        + query_idx * gs_stride_m
        + offs_k * gs_stride_k,
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    q = tl.load(
        q_ptr
        + query_idx * q_stride_m
        + batch_idx * q_stride_b
        + offs_d * q_stride_d,
        mask=offs_d < HEAD_DIM,
        other=0.0,
    ).to(tl.float32)
    selected_k = tl.load(
        selected_k_ptr
        + batch_idx * sk_stride_b
        + query_idx * sk_stride_m
        + offs_k[:, None] * sk_stride_k
        + offs_d[None, :] * sk_stride_d,
        mask=valid[:, None] & (offs_d[None, :] < HEAD_DIM),
        other=0.0,
    ).to(tl.float32)
    scaled_grad = grad_scores * SCORE_SCALE
    grad_q = tl.sum(selected_k * scaled_grad[:, None], axis=0)
    tl.atomic_add(
        grad_q_ptr
        + query_idx * gq_stride_m
        + batch_idx * gq_stride_b
        + offs_d * gq_stride_d,
        grad_q,
        sem="relaxed",
        mask=offs_d < HEAD_DIM,
    )
    tl.store(
        grad_selected_k_ptr
        + batch_idx * gsk_stride_b
        + query_idx * gsk_stride_m
        + offs_k[:, None] * gsk_stride_k
        + offs_d[None, :] * gsk_stride_d,
        scaled_grad[:, None] * q[None, :],
        mask=support_mask[:, None] & (offs_d[None, :] < HEAD_DIM),
    )


@triton.autotune(
    configs=_simplified_score_block_autotune_configs(),
    key=["query_len", "key_len", "HEAD_DIM"],
)
@triton.jit
def _dsa_simplified_score_block_kernel(
    q_ptr,
    key_ptr,
    out_scores_ptr,
    q_start,
    k_start,
    query_len: tl.constexpr,
    key_len: tl.constexpr,
    q_stride_m: tl.constexpr,
    q_stride_b: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_stride_d: tl.constexpr,
    k_stride_s: tl.constexpr,
    k_stride_b: tl.constexpr,
    k_stride_g: tl.constexpr,
    k_stride_d: tl.constexpr,
    out_stride_b: tl.constexpr,
    out_stride_m: tl.constexpr,
    out_stride_k: tl.constexpr,
    SCORE_SCALE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    offs_m = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.program_id(2) * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    q_mask = offs_m < query_len
    k_mask = offs_n < key_len
    q = tl.load(
        q_ptr
        + offs_m[:, None] * q_stride_m
        + batch_idx * q_stride_b
        + offs_d[None, :] * q_stride_d,
        mask=q_mask[:, None] & (offs_d[None, :] < HEAD_DIM),
        other=0.0,
    )
    key = tl.load(
        key_ptr
        + offs_n[:, None] * k_stride_s
        + batch_idx * k_stride_b
        + offs_d[None, :] * k_stride_d,
        mask=k_mask[:, None] & (offs_d[None, :] < HEAD_DIM),
        other=0.0,
    )
    scores = tl.dot(q, tl.trans(key), input_precision="ieee", out_dtype=tl.float32)
    scores *= SCORE_SCALE
    valid = (
        q_mask[:, None]
        & k_mask[None, :]
        & (k_start + offs_n[None, :] <= q_start + offs_m[:, None])
    )
    scores = tl.where(valid, scores, -float("inf"))
    tl.store(
        out_scores_ptr
        + batch_idx * out_stride_b
        + offs_m[:, None] * out_stride_m
        + offs_n[None, :] * out_stride_k,
        scores,
        mask=q_mask[:, None] & k_mask[None, :],
    )


@triton.autotune(
    configs=_selected_index_scores_autotune_configs(),
    key=["query_len", "topk", "INDEX_HEADS", "INDEX_HEAD_DIM"],
)
@triton.jit
def _dsa_selected_index_scores_kernel(
    q_ptr,
    weights_ptr,
    selected_k_ptr,
    topk_indices_ptr,
    out_scores_ptr,
    q_start,
    query_len: tl.constexpr,
    topk: tl.constexpr,
    q_stride_m: tl.constexpr,
    q_stride_b: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_stride_d: tl.constexpr,
    w_stride_m: tl.constexpr,
    w_stride_b: tl.constexpr,
    w_stride_h: tl.constexpr,
    sk_stride_b: tl.constexpr,
    sk_stride_m: tl.constexpr,
    sk_stride_k: tl.constexpr,
    sk_stride_d: tl.constexpr,
    ti_stride_b: tl.constexpr,
    ti_stride_m: tl.constexpr,
    ti_stride_k: tl.constexpr,
    out_stride_b: tl.constexpr,
    out_stride_m: tl.constexpr,
    out_stride_k: tl.constexpr,
    INDEX_HEADS: tl.constexpr,
    INDEX_HEAD_DIM: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    query_idx = tl.program_id(1)
    k_block = tl.program_id(2)
    offs_k = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, BLOCK_D)
    support_mask = offs_k < topk
    score = tl.full((BLOCK_K,), 0.0, dtype=tl.float32)

    for head_idx in tl.range(0, INDEX_HEADS):
        q = tl.load(
            q_ptr
            + query_idx * q_stride_m
            + batch_idx * q_stride_b
            + head_idx * q_stride_h
            + offs_d * q_stride_d,
            mask=offs_d < INDEX_HEAD_DIM,
            other=0.0,
        ).to(tl.float32)
        selected_k = tl.load(
            selected_k_ptr
            + batch_idx * sk_stride_b
            + query_idx * sk_stride_m
            + offs_k[:, None] * sk_stride_k
            + offs_d[None, :] * sk_stride_d,
            mask=support_mask[:, None] & (offs_d[None, :] < INDEX_HEAD_DIM),
            other=0.0,
        ).to(tl.float32)
        dot = tl.sum(selected_k * q[None, :], axis=1)
        dot = tl.maximum(dot, 0.0)
        weight = tl.load(
            weights_ptr
            + query_idx * w_stride_m
            + batch_idx * w_stride_b
            + head_idx * w_stride_h
        ).to(tl.float32)
        score += dot * weight

    selected_positions = tl.load(
        topk_indices_ptr
        + batch_idx * ti_stride_b
        + query_idx * ti_stride_m
        + offs_k * ti_stride_k,
        mask=support_mask,
        other=0,
    )
    valid = support_mask & (selected_positions <= q_start + query_idx)
    score = tl.where(valid, score, -float("inf"))
    tl.store(
        out_scores_ptr
        + batch_idx * out_stride_b
        + query_idx * out_stride_m
        + offs_k * out_stride_k,
        score,
        mask=support_mask,
    )


@triton.autotune(
    configs=_indexer_loss_autotune_configs(),
    key=["query_len", "topk", "INDEX_HEADS", "INDEX_HEAD_DIM", "BLOCK_K", "BLOCK_D"],
)
@triton.jit
def _dsa_selected_index_kl_loss_kernel(
    q_ptr,
    weights_ptr,
    selected_k_ptr,
    topk_indices_ptr,
    teacher_ptr,
    partial_loss_ptr,
    q_start,
    query_len: tl.constexpr,
    topk: tl.constexpr,
    loss_scale: tl.constexpr,
    q_stride_m: tl.constexpr,
    q_stride_b: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_stride_d: tl.constexpr,
    w_stride_m: tl.constexpr,
    w_stride_b: tl.constexpr,
    w_stride_h: tl.constexpr,
    sk_stride_b: tl.constexpr,
    sk_stride_m: tl.constexpr,
    sk_stride_k: tl.constexpr,
    sk_stride_d: tl.constexpr,
    ti_stride_b: tl.constexpr,
    ti_stride_m: tl.constexpr,
    ti_stride_k: tl.constexpr,
    teacher_stride_b: tl.constexpr,
    teacher_stride_m: tl.constexpr,
    teacher_stride_k: tl.constexpr,
    out_stride_b: tl.constexpr,
    out_stride_m: tl.constexpr,
    INDEX_HEADS: tl.constexpr,
    INDEX_HEAD_DIM: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    query_idx = tl.program_id(1)
    offs_k = tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, BLOCK_D)
    support_mask = offs_k < topk
    score = tl.full((BLOCK_K,), 0.0, dtype=tl.float32)

    for head_idx in tl.range(0, INDEX_HEADS):
        q = tl.load(
            q_ptr
            + query_idx * q_stride_m
            + batch_idx * q_stride_b
            + head_idx * q_stride_h
            + offs_d * q_stride_d,
            mask=offs_d < INDEX_HEAD_DIM,
            other=0.0,
        ).to(tl.float32)
        selected_k = tl.load(
            selected_k_ptr
            + batch_idx * sk_stride_b
            + query_idx * sk_stride_m
            + offs_k[:, None] * sk_stride_k
            + offs_d[None, :] * sk_stride_d,
            mask=support_mask[:, None] & (offs_d[None, :] < INDEX_HEAD_DIM),
            other=0.0,
        ).to(tl.float32)
        dot = tl.sum(selected_k * q[None, :], axis=1)
        dot = tl.maximum(dot, 0.0)
        weight = tl.load(
            weights_ptr
            + query_idx * w_stride_m
            + batch_idx * w_stride_b
            + head_idx * w_stride_h
        ).to(tl.float32)
        score += dot * weight

    selected_positions = tl.load(
        topk_indices_ptr
        + batch_idx * ti_stride_b
        + query_idx * ti_stride_m
        + offs_k * ti_stride_k,
        mask=support_mask,
        other=0,
    )
    valid = support_mask & (selected_positions <= q_start + query_idx)
    score = tl.where(valid, score, -float("inf"))
    row_max = tl.max(score, axis=0)
    exp_scores = tl.exp(score - row_max)
    exp_scores = tl.where(valid, exp_scores, 0.0)
    student = exp_scores / tl.sum(exp_scores, axis=0)
    teacher = tl.load(
        teacher_ptr
        + batch_idx * teacher_stride_b
        + query_idx * teacher_stride_m
        + offs_k * teacher_stride_k,
        mask=support_mask,
        other=0.0,
    ).to(tl.float32)
    kl = teacher * (tl.log(teacher + 1.0e-10) - tl.log(student + 1.0e-10))
    kl = tl.where(valid, kl, 0.0)
    tl.store(
        partial_loss_ptr + batch_idx * out_stride_b + query_idx * out_stride_m,
        tl.sum(kl, axis=0) * loss_scale,
    )


@triton.autotune(
    configs=_selected_index_scores_autotune_configs(),
    key=["query_len", "topk", "INDEX_HEADS", "INDEX_HEAD_DIM"],
    reset_to_zero=["grad_q_ptr", "grad_weights_ptr"],
)
@triton.jit
def _dsa_selected_index_scores_backward_kernel(
    q_ptr,
    weights_ptr,
    selected_k_ptr,
    topk_indices_ptr,
    grad_scores_ptr,
    grad_q_ptr,
    grad_weights_ptr,
    grad_selected_k_ptr,
    q_start,
    query_len: tl.constexpr,
    topk: tl.constexpr,
    q_stride_m: tl.constexpr,
    q_stride_b: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_stride_d: tl.constexpr,
    w_stride_m: tl.constexpr,
    w_stride_b: tl.constexpr,
    w_stride_h: tl.constexpr,
    sk_stride_b: tl.constexpr,
    sk_stride_m: tl.constexpr,
    sk_stride_k: tl.constexpr,
    sk_stride_d: tl.constexpr,
    ti_stride_b: tl.constexpr,
    ti_stride_m: tl.constexpr,
    ti_stride_k: tl.constexpr,
    gs_stride_b: tl.constexpr,
    gs_stride_m: tl.constexpr,
    gs_stride_k: tl.constexpr,
    gq_stride_m: tl.constexpr,
    gq_stride_b: tl.constexpr,
    gq_stride_h: tl.constexpr,
    gq_stride_d: tl.constexpr,
    gw_stride_m: tl.constexpr,
    gw_stride_b: tl.constexpr,
    gw_stride_h: tl.constexpr,
    gsk_stride_b: tl.constexpr,
    gsk_stride_m: tl.constexpr,
    gsk_stride_k: tl.constexpr,
    gsk_stride_d: tl.constexpr,
    INDEX_HEADS: tl.constexpr,
    INDEX_HEAD_DIM: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    query_idx = tl.program_id(1)
    k_block = tl.program_id(2)
    offs_k = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, BLOCK_D)
    support_mask = offs_k < topk
    selected_positions = tl.load(
        topk_indices_ptr
        + batch_idx * ti_stride_b
        + query_idx * ti_stride_m
        + offs_k * ti_stride_k,
        mask=support_mask,
        other=0,
    )
    valid = support_mask & (selected_positions <= q_start + query_idx)
    grad_scores = tl.load(
        grad_scores_ptr
        + batch_idx * gs_stride_b
        + query_idx * gs_stride_m
        + offs_k * gs_stride_k,
        mask=support_mask,
        other=0.0,
    ).to(tl.float32)
    grad_selected_k = tl.zeros((BLOCK_K, BLOCK_D), dtype=tl.float32)

    for head_idx in tl.range(0, INDEX_HEADS):
        q = tl.load(
            q_ptr
            + query_idx * q_stride_m
            + batch_idx * q_stride_b
            + head_idx * q_stride_h
            + offs_d * q_stride_d,
            mask=offs_d < INDEX_HEAD_DIM,
            other=0.0,
        ).to(tl.float32)
        selected_k = tl.load(
            selected_k_ptr
            + batch_idx * sk_stride_b
            + query_idx * sk_stride_m
            + offs_k[:, None] * sk_stride_k
            + offs_d[None, :] * sk_stride_d,
            mask=support_mask[:, None] & (offs_d[None, :] < INDEX_HEAD_DIM),
            other=0.0,
        ).to(tl.float32)
        dot = tl.sum(selected_k * q[None, :], axis=1)
        relu_dot = tl.maximum(dot, 0.0)
        active = valid & (dot > 0.0)
        weight = tl.load(
            weights_ptr
            + query_idx * w_stride_m
            + batch_idx * w_stride_b
            + head_idx * w_stride_h
        ).to(tl.float32)
        coeff = tl.where(active, grad_scores * weight, 0.0)
        grad_q = tl.sum(selected_k * coeff[:, None], axis=0)
        grad_weight = tl.sum(tl.where(valid, grad_scores * relu_dot, 0.0), axis=0)
        grad_selected_k += coeff[:, None] * q[None, :]
        tl.atomic_add(
            grad_q_ptr
            + query_idx * gq_stride_m
            + batch_idx * gq_stride_b
            + head_idx * gq_stride_h
            + offs_d * gq_stride_d,
            grad_q,
            sem="relaxed",
            mask=offs_d < INDEX_HEAD_DIM,
        )
        tl.atomic_add(
            grad_weights_ptr
            + query_idx * gw_stride_m
            + batch_idx * gw_stride_b
            + head_idx * gw_stride_h,
            grad_weight,
            sem="relaxed",
        )

    tl.store(
        grad_selected_k_ptr
        + batch_idx * gsk_stride_b
        + query_idx * gsk_stride_m
        + offs_k[:, None] * gsk_stride_k
        + offs_d[None, :] * gsk_stride_d,
        grad_selected_k,
        mask=support_mask[:, None] & (offs_d[None, :] < INDEX_HEAD_DIM),
    )


@triton.autotune(
    configs=_selected_index_scores_bwd_dot_autotune_configs(),
    key=["query_len", "topk", "INDEX_HEADS", "INDEX_HEAD_DIM", "BLOCK_H", "BLOCK_D"],
    reset_to_zero=["grad_q_ptr", "grad_weights_ptr"],
)
@triton.jit
def _dsa_selected_index_scores_backward_dot_kernel(
    q_ptr,
    weights_ptr,
    selected_k_ptr,
    topk_indices_ptr,
    grad_scores_ptr,
    grad_q_ptr,
    grad_weights_ptr,
    grad_selected_k_ptr,
    q_start,
    query_len: tl.constexpr,
    topk: tl.constexpr,
    q_stride_m: tl.constexpr,
    q_stride_b: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_stride_d: tl.constexpr,
    w_stride_m: tl.constexpr,
    w_stride_b: tl.constexpr,
    w_stride_h: tl.constexpr,
    sk_stride_b: tl.constexpr,
    sk_stride_m: tl.constexpr,
    sk_stride_k: tl.constexpr,
    sk_stride_d: tl.constexpr,
    ti_stride_b: tl.constexpr,
    ti_stride_m: tl.constexpr,
    ti_stride_k: tl.constexpr,
    gs_stride_b: tl.constexpr,
    gs_stride_m: tl.constexpr,
    gs_stride_k: tl.constexpr,
    gq_stride_m: tl.constexpr,
    gq_stride_b: tl.constexpr,
    gq_stride_h: tl.constexpr,
    gq_stride_d: tl.constexpr,
    gw_stride_m: tl.constexpr,
    gw_stride_b: tl.constexpr,
    gw_stride_h: tl.constexpr,
    gsk_stride_b: tl.constexpr,
    gsk_stride_m: tl.constexpr,
    gsk_stride_k: tl.constexpr,
    gsk_stride_d: tl.constexpr,
    INDEX_HEADS: tl.constexpr,
    INDEX_HEAD_DIM: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    query_idx = tl.program_id(1)
    k_block = tl.program_id(2)
    offs_k = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_h = tl.arange(0, BLOCK_H)
    offs_d = tl.arange(0, BLOCK_D)
    support_mask = offs_k < topk
    head_mask = offs_h < INDEX_HEADS
    feature_mask = offs_d < INDEX_HEAD_DIM

    selected_positions = tl.load(
        topk_indices_ptr
        + batch_idx * ti_stride_b
        + query_idx * ti_stride_m
        + offs_k * ti_stride_k,
        mask=support_mask,
        other=0,
    )
    valid = support_mask & (selected_positions <= q_start + query_idx)
    grad_scores = tl.load(
        grad_scores_ptr
        + batch_idx * gs_stride_b
        + query_idx * gs_stride_m
        + offs_k * gs_stride_k,
        mask=support_mask,
        other=0.0,
    ).to(tl.float32)
    q = tl.load(
        q_ptr
        + query_idx * q_stride_m
        + batch_idx * q_stride_b
        + offs_h[:, None] * q_stride_h
        + offs_d[None, :] * q_stride_d,
        mask=head_mask[:, None] & feature_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    selected_k = tl.load(
        selected_k_ptr
        + batch_idx * sk_stride_b
        + query_idx * sk_stride_m
        + offs_k[:, None] * sk_stride_k
        + offs_d[None, :] * sk_stride_d,
        mask=support_mask[:, None] & feature_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    scores = tl.dot(
        selected_k,
        tl.trans(q),
        input_precision="ieee",
        out_dtype=tl.float32,
    )
    relu_scores = tl.maximum(scores, 0.0)
    active = valid[:, None] & head_mask[None, :] & (scores > 0.0)
    weights = tl.load(
        weights_ptr
        + query_idx * w_stride_m
        + batch_idx * w_stride_b
        + offs_h * w_stride_h,
        mask=head_mask,
        other=0.0,
    ).to(tl.float32)
    coeff = tl.where(active, grad_scores[:, None] * weights[None, :], 0.0)

    grad_q = tl.dot(
        tl.trans(coeff),
        selected_k,
        input_precision="ieee",
        out_dtype=tl.float32,
    )
    grad_weights = tl.sum(
        tl.where(valid[:, None] & head_mask[None, :], grad_scores[:, None] * relu_scores, 0.0),
        axis=0,
    )
    grad_selected_k = tl.dot(
        coeff,
        q,
        input_precision="ieee",
        out_dtype=tl.float32,
    )

    tl.atomic_add(
        grad_q_ptr
        + query_idx * gq_stride_m
        + batch_idx * gq_stride_b
        + offs_h[:, None] * gq_stride_h
        + offs_d[None, :] * gq_stride_d,
        grad_q,
        sem="relaxed",
        mask=head_mask[:, None] & feature_mask[None, :],
    )
    tl.atomic_add(
        grad_weights_ptr
        + query_idx * gw_stride_m
        + batch_idx * gw_stride_b
        + offs_h * gw_stride_h,
        grad_weights,
        sem="relaxed",
        mask=head_mask,
    )
    tl.store(
        grad_selected_k_ptr
        + batch_idx * gsk_stride_b
        + query_idx * gsk_stride_m
        + offs_k[:, None] * gsk_stride_k
        + offs_d[None, :] * gsk_stride_d,
        grad_selected_k,
        mask=support_mask[:, None] & feature_mask[None, :],
    )
@triton.jit
def _dsa_indexer_loss_grad_kernel(
    selected_scores_ptr,
    teacher_ptr,
    scale_ptr,
    grad_scores_ptr,
    query_len: tl.constexpr,
    topk: tl.constexpr,
    ss_stride_b: tl.constexpr,
    ss_stride_m: tl.constexpr,
    ss_stride_k: tl.constexpr,
    teacher_stride_b: tl.constexpr,
    teacher_stride_m: tl.constexpr,
    teacher_stride_k: tl.constexpr,
    gs_stride_b: tl.constexpr,
    gs_stride_m: tl.constexpr,
    gs_stride_k: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    query_idx = tl.program_id(1)
    offs_k = tl.arange(0, BLOCK_K)
    mask = offs_k < topk
    selected_scores = tl.load(
        selected_scores_ptr
        + batch_idx * ss_stride_b
        + query_idx * ss_stride_m
        + offs_k * ss_stride_k,
        mask=mask,
        other=-float("inf"),
    ).to(tl.float32)
    teacher = tl.load(
        teacher_ptr
        + batch_idx * teacher_stride_b
        + query_idx * teacher_stride_m
        + offs_k * teacher_stride_k,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    row_max = tl.max(selected_scores, axis=0)
    exp_scores = tl.exp(selected_scores - row_max)
    exp_scores = tl.where(mask, exp_scores, 0.0)
    student = exp_scores / tl.sum(exp_scores, axis=0)
    teacher_over_student = teacher * student / (student + 1.0e-10)
    teacher_over_student = tl.where(mask, teacher_over_student, 0.0)
    scale = tl.load(scale_ptr).to(tl.float32)
    grad_scores = (
        student * tl.sum(teacher_over_student, axis=0) - teacher_over_student
    ) * scale
    tl.store(
        grad_scores_ptr
        + batch_idx * gs_stride_b
        + query_idx * gs_stride_m
        + offs_k * gs_stride_k,
        grad_scores,
        mask=mask,
    )


@triton.autotune(
    configs=_sparse_attention_autotune_configs(),
    key=["topk", "HEAD_DIM", "VALUE_DIM", "VALUE_DTYPE"],
)
@triton.jit
def _dsa_sparse_attention_forward_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    topk_indices_ptr,
    output_ptr,
    softmax_scale: tl.constexpr,
    q_start,
    query_len: tl.constexpr,
    topk: tl.constexpr,
    repeat_factor: tl.constexpr,
    q_stride_m: tl.constexpr,
    q_stride_b: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_stride_d: tl.constexpr,
    k_stride_s: tl.constexpr,
    k_stride_b: tl.constexpr,
    k_stride_g: tl.constexpr,
    k_stride_d: tl.constexpr,
    v_stride_s: tl.constexpr,
    v_stride_b: tl.constexpr,
    v_stride_g: tl.constexpr,
    v_stride_d: tl.constexpr,
    ti_stride_b: tl.constexpr,
    ti_stride_m: tl.constexpr,
    ti_stride_k: tl.constexpr,
    out_stride_m: tl.constexpr,
    out_stride_b: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_d: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_K: tl.constexpr,
    VALUE_DTYPE: tl.constexpr,
):
    query_idx = tl.program_id(0)
    batch_idx = tl.program_id(1)
    head_idx = tl.program_id(2)
    group_idx = head_idx // repeat_factor
    offs_d = tl.arange(0, BLOCK_D)
    offs_dv = tl.arange(0, BLOCK_DV)
    # Materialize loop-invariant broadcasts in the parent region. Some Triton
    # versions otherwise reuse an op from the first loop in the sibling loop.
    offs_d_block = tl.broadcast_to(tl.expand_dims(offs_d, 0), (BLOCK_K, BLOCK_D))
    offs_d_mask = offs_d_block < HEAD_DIM
    q = tl.load(
        query_ptr
        + query_idx * q_stride_m
        + batch_idx * q_stride_b
        + head_idx * q_stride_h
        + offs_d * q_stride_d,
        mask=offs_d < HEAD_DIM,
        other=0.0,
    ).to(tl.float32)
    q_block = tl.broadcast_to(tl.expand_dims(q, 0), (BLOCK_K, BLOCK_D))
    running_max = tl.full((), -float("inf"), dtype=tl.float32)
    running_sum = tl.full((), 0.0, dtype=tl.float32)

    for support_start in range(0, topk, BLOCK_K):
        offs_k = support_start + tl.arange(0, BLOCK_K)
        support_mask = offs_k < topk
        selected = tl.load(
            topk_indices_ptr
            + batch_idx * ti_stride_b
            + query_idx * ti_stride_m
            + offs_k * ti_stride_k,
            mask=support_mask,
            other=0,
        )
        valid = support_mask & (selected <= q_start + query_idx)
        k = tl.load(
            key_ptr
            + selected[:, None] * k_stride_s
            + batch_idx * k_stride_b
            + group_idx * k_stride_g
            + offs_d_block * k_stride_d,
            mask=valid[:, None] & offs_d_mask,
            other=0.0,
        ).to(tl.float32)
        scores = tl.sum(k * q_block, axis=1) * softmax_scale
        scores = tl.where(valid, scores, -float("inf"))
        block_max = tl.max(scores, axis=0)
        new_max = tl.maximum(running_max, block_max)
        old_scale = tl.exp(running_max - new_max)
        block_probs = tl.exp(scores - new_max)
        block_sum = tl.sum(block_probs, axis=0)
        running_sum = running_sum * old_scale + block_sum
        running_max = new_max

    out_acc = tl.zeros((BLOCK_DV,), dtype=tl.float32)
    for support_start in range(0, topk, BLOCK_K):
        offs_k = support_start + tl.arange(0, BLOCK_K)
        support_mask = offs_k < topk
        selected = tl.load(
            topk_indices_ptr
            + batch_idx * ti_stride_b
            + query_idx * ti_stride_m
            + offs_k * ti_stride_k,
            mask=support_mask,
            other=0,
        )
        valid = support_mask & (selected <= q_start + query_idx)
        k = tl.load(
            key_ptr
            + selected[:, None] * k_stride_s
            + batch_idx * k_stride_b
            + group_idx * k_stride_g
            + offs_d_block * k_stride_d,
            mask=valid[:, None] & offs_d_mask,
            other=0.0,
        ).to(tl.float32)
        scores = tl.sum(k * q_block, axis=1) * softmax_scale
        scores = tl.where(valid, scores, -float("inf"))
        probs = tl.exp(scores - running_max) / running_sum
        v = tl.load(
            value_ptr
            + selected[:, None] * v_stride_s
            + batch_idx * v_stride_b
            + group_idx * v_stride_g
            + offs_dv[None, :] * v_stride_d,
            mask=valid[:, None] & (offs_dv[None, :] < VALUE_DIM),
            other=0.0,
        )
        if VALUE_DTYPE == 1:
            probs_for_value = probs.to(tl.float16)
        elif VALUE_DTYPE == 2:
            probs_for_value = probs.to(tl.bfloat16)
        else:
            probs_for_value = probs
        dot_rows = tl.arange(0, 16)
        probs_for_dot = tl.where(dot_rows[:, None] == 0, probs_for_value[None, :], 0.0)
        if VALUE_DTYPE == 1:
            probs_for_dot = probs_for_dot.to(tl.float16)
        elif VALUE_DTYPE == 2:
            probs_for_dot = probs_for_dot.to(tl.bfloat16)
        value_acc = tl.dot(probs_for_dot, v, out_dtype=tl.float32)
        out_acc += tl.sum(value_acc, axis=0)

    tl.store(
        output_ptr
        + query_idx * out_stride_m
        + batch_idx * out_stride_b
        + head_idx * out_stride_h
        + offs_dv * out_stride_d,
        out_acc,
        mask=offs_dv < VALUE_DIM,
    )


@triton.autotune(
    configs=_sparse_attention_autotune_configs(),
    key=["topk", "HEAD_DIM", "VALUE_DIM", "VALUE_DTYPE"],
    restore_value=["grad_key_ptr", "grad_value_ptr"],
)
@triton.jit
def _dsa_sparse_attention_backward_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    topk_indices_ptr,
    grad_output_ptr,
    grad_query_ptr,
    grad_key_ptr,
    grad_value_ptr,
    softmax_scale: tl.constexpr,
    q_start,
    query_len: tl.constexpr,
    topk: tl.constexpr,
    repeat_factor: tl.constexpr,
    q_stride_m: tl.constexpr,
    q_stride_b: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_stride_d: tl.constexpr,
    k_stride_s: tl.constexpr,
    k_stride_b: tl.constexpr,
    k_stride_g: tl.constexpr,
    k_stride_d: tl.constexpr,
    v_stride_s: tl.constexpr,
    v_stride_b: tl.constexpr,
    v_stride_g: tl.constexpr,
    v_stride_d: tl.constexpr,
    ti_stride_b: tl.constexpr,
    ti_stride_m: tl.constexpr,
    ti_stride_k: tl.constexpr,
    go_stride_m: tl.constexpr,
    go_stride_b: tl.constexpr,
    go_stride_h: tl.constexpr,
    go_stride_d: tl.constexpr,
    gq_stride_m: tl.constexpr,
    gq_stride_b: tl.constexpr,
    gq_stride_h: tl.constexpr,
    gq_stride_d: tl.constexpr,
    gk_stride_s: tl.constexpr,
    gk_stride_b: tl.constexpr,
    gk_stride_g: tl.constexpr,
    gk_stride_d: tl.constexpr,
    gv_stride_s: tl.constexpr,
    gv_stride_b: tl.constexpr,
    gv_stride_g: tl.constexpr,
    gv_stride_d: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_K: tl.constexpr,
    VALUE_DTYPE: tl.constexpr,
):
    query_idx = tl.program_id(0)
    batch_idx = tl.program_id(1)
    head_idx = tl.program_id(2)
    group_idx = head_idx // repeat_factor
    offs_d = tl.arange(0, BLOCK_D)
    offs_dv = tl.arange(0, BLOCK_DV)
    offs_d_block = tl.broadcast_to(tl.expand_dims(offs_d, 0), (BLOCK_K, BLOCK_D))
    offs_d_mask = offs_d_block < HEAD_DIM
    offs_dv_block = tl.broadcast_to(tl.expand_dims(offs_dv, 0), (BLOCK_K, BLOCK_DV))
    offs_dv_mask = offs_dv_block < VALUE_DIM
    q = tl.load(
        query_ptr
        + query_idx * q_stride_m
        + batch_idx * q_stride_b
        + head_idx * q_stride_h
        + offs_d * q_stride_d,
        mask=offs_d < HEAD_DIM,
        other=0.0,
    ).to(tl.float32)
    q_block = tl.broadcast_to(tl.expand_dims(q, 0), (BLOCK_K, BLOCK_D))
    grad_out = tl.load(
        grad_output_ptr
        + query_idx * go_stride_m
        + batch_idx * go_stride_b
        + head_idx * go_stride_h
        + offs_dv * go_stride_d,
        mask=offs_dv < VALUE_DIM,
        other=0.0,
    )
    if VALUE_DTYPE == 1:
        grad_out_for_value = grad_out.to(tl.float16)
    elif VALUE_DTYPE == 2:
        grad_out_for_value = grad_out.to(tl.bfloat16)
    else:
        grad_out_for_value = grad_out
    grad_out_block = tl.broadcast_to(
        tl.expand_dims(grad_out_for_value, 0), (BLOCK_K, BLOCK_DV)
    )
    running_max = tl.full((), -float("inf"), dtype=tl.float32)
    running_sum = tl.full((), 0.0, dtype=tl.float32)
    dprob_acc = tl.full((), 0.0, dtype=tl.float32)

    for support_start in range(0, topk, BLOCK_K):
        offs_k = support_start + tl.arange(0, BLOCK_K)
        support_mask = offs_k < topk
        selected = tl.load(
            topk_indices_ptr
            + batch_idx * ti_stride_b
            + query_idx * ti_stride_m
            + offs_k * ti_stride_k,
            mask=support_mask,
            other=0,
        )
        valid = support_mask & (selected <= q_start + query_idx)
        k = tl.load(
            key_ptr
            + selected[:, None] * k_stride_s
            + batch_idx * k_stride_b
            + group_idx * k_stride_g
            + offs_d_block * k_stride_d,
            mask=valid[:, None] & offs_d_mask,
            other=0.0,
        ).to(tl.float32)
        scores = tl.sum(k * q_block, axis=1) * softmax_scale
        scores = tl.where(valid, scores, -float("inf"))
        block_max = tl.max(scores, axis=0)
        new_max = tl.maximum(running_max, block_max)
        old_scale = tl.exp(running_max - new_max)
        block_probs = tl.exp(scores - new_max)
        block_sum = tl.sum(block_probs, axis=0)
        v = tl.load(
            value_ptr
            + selected[:, None] * v_stride_s
            + batch_idx * v_stride_b
            + group_idx * v_stride_g
            + offs_dv_block * v_stride_d,
            mask=valid[:, None] & offs_dv_mask,
            other=0.0,
        )
        dprob = tl.sum((v * grad_out_block).to(tl.float32), axis=1)
        block_dprob = tl.sum(dprob * block_probs, axis=0)
        dprob_acc = dprob_acc * old_scale + block_dprob
        running_sum = running_sum * old_scale + block_sum
        running_max = new_max

    delta = dprob_acc / running_sum
    grad_q = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for support_start in range(0, topk, BLOCK_K):
        offs_k = support_start + tl.arange(0, BLOCK_K)
        support_mask = offs_k < topk
        selected = tl.load(
            topk_indices_ptr
            + batch_idx * ti_stride_b
            + query_idx * ti_stride_m
            + offs_k * ti_stride_k,
            mask=support_mask,
            other=0,
        )
        valid = support_mask & (selected <= q_start + query_idx)
        k = tl.load(
            key_ptr
            + selected[:, None] * k_stride_s
            + batch_idx * k_stride_b
            + group_idx * k_stride_g
            + offs_d_block * k_stride_d,
            mask=valid[:, None] & offs_d_mask,
            other=0.0,
        ).to(tl.float32)
        scores = tl.sum(k * q_block, axis=1) * softmax_scale
        scores = tl.where(valid, scores, -float("inf"))
        probs = tl.exp(scores - running_max) / running_sum
        v = tl.load(
            value_ptr
            + selected[:, None] * v_stride_s
            + batch_idx * v_stride_b
            + group_idx * v_stride_g
            + offs_dv_block * v_stride_d,
            mask=valid[:, None] & offs_dv_mask,
            other=0.0,
        )
        dprob = tl.sum((v * grad_out_block).to(tl.float32), axis=1)
        dscores = probs * (dprob - delta)
        dscores = tl.where(valid, dscores, 0.0)
        grad_q += tl.sum(k * dscores[:, None], axis=0) * softmax_scale

        grad_k = dscores[:, None] * q_block * softmax_scale
        tl.atomic_add(
            grad_key_ptr
            + selected[:, None] * gk_stride_s
            + batch_idx * gk_stride_b
            + group_idx * gk_stride_g
            + offs_d_block * gk_stride_d,
            grad_k,
            sem="relaxed",
            mask=valid[:, None] & offs_d_mask,
        )

        if VALUE_DTYPE == 1:
            probs_for_value = probs.to(tl.float16)
        elif VALUE_DTYPE == 2:
            probs_for_value = probs.to(tl.bfloat16)
        else:
            probs_for_value = probs
        grad_v = (probs_for_value[:, None] * grad_out_block).to(tl.float32)
        tl.atomic_add(
            grad_value_ptr
            + selected[:, None] * gv_stride_s
            + batch_idx * gv_stride_b
            + group_idx * gv_stride_g
            + offs_dv_block * gv_stride_d,
            grad_v,
            sem="relaxed",
            mask=valid[:, None] & offs_dv_mask,
        )

    tl.store(
        grad_query_ptr
        + query_idx * gq_stride_m
        + batch_idx * gq_stride_b
        + head_idx * gq_stride_h
        + offs_d * gq_stride_d,
        grad_q,
        mask=offs_d < HEAD_DIM,
    )


@triton.autotune(
    configs=_sparse_attention_autotune_configs(),
    key=["topk", "HEAD_DIM", "VALUE_DIM", "VALUE_DTYPE"],
    restore_value=["grad_key_ptr", "grad_value_ptr"],
)
@triton.jit
def _dsa_sparse_attention_backward_pair_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    topk_indices_ptr,
    grad_output_ptr,
    grad_query_ptr,
    grad_key_ptr,
    grad_value_ptr,
    softmax_scale: tl.constexpr,
    q_start,
    query_len: tl.constexpr,
    topk: tl.constexpr,
    repeat_factor: tl.constexpr,
    pairs_per_group: tl.constexpr,
    q_stride_m: tl.constexpr,
    q_stride_b: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_stride_d: tl.constexpr,
    k_stride_s: tl.constexpr,
    k_stride_b: tl.constexpr,
    k_stride_g: tl.constexpr,
    k_stride_d: tl.constexpr,
    v_stride_s: tl.constexpr,
    v_stride_b: tl.constexpr,
    v_stride_g: tl.constexpr,
    v_stride_d: tl.constexpr,
    ti_stride_b: tl.constexpr,
    ti_stride_m: tl.constexpr,
    ti_stride_k: tl.constexpr,
    go_stride_m: tl.constexpr,
    go_stride_b: tl.constexpr,
    go_stride_h: tl.constexpr,
    go_stride_d: tl.constexpr,
    gq_stride_m: tl.constexpr,
    gq_stride_b: tl.constexpr,
    gq_stride_h: tl.constexpr,
    gq_stride_d: tl.constexpr,
    gk_stride_s: tl.constexpr,
    gk_stride_b: tl.constexpr,
    gk_stride_g: tl.constexpr,
    gk_stride_d: tl.constexpr,
    gv_stride_s: tl.constexpr,
    gv_stride_b: tl.constexpr,
    gv_stride_g: tl.constexpr,
    gv_stride_d: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_K: tl.constexpr,
    VALUE_DTYPE: tl.constexpr,
):
    query_idx = tl.program_id(0)
    batch_idx = tl.program_id(1)
    pair_program_idx = tl.program_id(2)
    group_idx = pair_program_idx // pairs_per_group
    pair_idx = pair_program_idx - group_idx * pairs_per_group
    head0 = group_idx * repeat_factor + pair_idx * 2
    head1 = head0 + 1
    offs_d = tl.arange(0, BLOCK_D)
    offs_dv = tl.arange(0, BLOCK_DV)
    offs_d_block = tl.broadcast_to(tl.expand_dims(offs_d, 0), (BLOCK_K, BLOCK_D))
    offs_d_mask = offs_d_block < HEAD_DIM
    offs_dv_block = tl.broadcast_to(tl.expand_dims(offs_dv, 0), (BLOCK_K, BLOCK_DV))
    offs_dv_mask = offs_dv_block < VALUE_DIM

    q0 = tl.load(
        query_ptr
        + query_idx * q_stride_m
        + batch_idx * q_stride_b
        + head0 * q_stride_h
        + offs_d * q_stride_d,
        mask=offs_d < HEAD_DIM,
        other=0.0,
    ).to(tl.float32)
    q1 = tl.load(
        query_ptr
        + query_idx * q_stride_m
        + batch_idx * q_stride_b
        + head1 * q_stride_h
        + offs_d * q_stride_d,
        mask=offs_d < HEAD_DIM,
        other=0.0,
    ).to(tl.float32)
    q0_block = tl.broadcast_to(tl.expand_dims(q0, 0), (BLOCK_K, BLOCK_D))
    q1_block = tl.broadcast_to(tl.expand_dims(q1, 0), (BLOCK_K, BLOCK_D))
    grad_out0 = tl.load(
        grad_output_ptr
        + query_idx * go_stride_m
        + batch_idx * go_stride_b
        + head0 * go_stride_h
        + offs_dv * go_stride_d,
        mask=offs_dv < VALUE_DIM,
        other=0.0,
    )
    grad_out1 = tl.load(
        grad_output_ptr
        + query_idx * go_stride_m
        + batch_idx * go_stride_b
        + head1 * go_stride_h
        + offs_dv * go_stride_d,
        mask=offs_dv < VALUE_DIM,
        other=0.0,
    )
    if VALUE_DTYPE == 1:
        grad_out0_for_value = grad_out0.to(tl.float16)
        grad_out1_for_value = grad_out1.to(tl.float16)
    elif VALUE_DTYPE == 2:
        grad_out0_for_value = grad_out0.to(tl.bfloat16)
        grad_out1_for_value = grad_out1.to(tl.bfloat16)
    else:
        grad_out0_for_value = grad_out0
        grad_out1_for_value = grad_out1
    grad_out0_block = tl.broadcast_to(
        tl.expand_dims(grad_out0_for_value, 0), (BLOCK_K, BLOCK_DV)
    )
    grad_out1_block = tl.broadcast_to(
        tl.expand_dims(grad_out1_for_value, 0), (BLOCK_K, BLOCK_DV)
    )

    running_max0 = tl.full((), -float("inf"), dtype=tl.float32)
    running_max1 = tl.full((), -float("inf"), dtype=tl.float32)
    running_sum0 = tl.full((), 0.0, dtype=tl.float32)
    running_sum1 = tl.full((), 0.0, dtype=tl.float32)
    dprob_acc0 = tl.full((), 0.0, dtype=tl.float32)
    dprob_acc1 = tl.full((), 0.0, dtype=tl.float32)

    for support_start in range(0, topk, BLOCK_K):
        offs_k = support_start + tl.arange(0, BLOCK_K)
        support_mask = offs_k < topk
        selected = tl.load(
            topk_indices_ptr
            + batch_idx * ti_stride_b
            + query_idx * ti_stride_m
            + offs_k * ti_stride_k,
            mask=support_mask,
            other=0,
        )
        valid = support_mask & (selected <= q_start + query_idx)
        k = tl.load(
            key_ptr
            + selected[:, None] * k_stride_s
            + batch_idx * k_stride_b
            + group_idx * k_stride_g
            + offs_d_block * k_stride_d,
            mask=valid[:, None] & offs_d_mask,
            other=0.0,
        ).to(tl.float32)
        score0 = tl.sum(k * q0_block, axis=1) * softmax_scale
        score1 = tl.sum(k * q1_block, axis=1) * softmax_scale
        score0 = tl.where(valid, score0, -float("inf"))
        score1 = tl.where(valid, score1, -float("inf"))
        block_max0 = tl.max(score0, axis=0)
        block_max1 = tl.max(score1, axis=0)
        new_max0 = tl.maximum(running_max0, block_max0)
        new_max1 = tl.maximum(running_max1, block_max1)
        old_scale0 = tl.exp(running_max0 - new_max0)
        old_scale1 = tl.exp(running_max1 - new_max1)
        block_probs0 = tl.exp(score0 - new_max0)
        block_probs1 = tl.exp(score1 - new_max1)
        block_probs0 = tl.where(valid, block_probs0, 0.0)
        block_probs1 = tl.where(valid, block_probs1, 0.0)
        v = tl.load(
            value_ptr
            + selected[:, None] * v_stride_s
            + batch_idx * v_stride_b
            + group_idx * v_stride_g
            + offs_dv_block * v_stride_d,
            mask=valid[:, None] & offs_dv_mask,
            other=0.0,
        )
        dprob0 = tl.sum((v * grad_out0_block).to(tl.float32), axis=1)
        dprob1 = tl.sum((v * grad_out1_block).to(tl.float32), axis=1)
        dprob_acc0 = dprob_acc0 * old_scale0 + tl.sum(dprob0 * block_probs0, axis=0)
        dprob_acc1 = dprob_acc1 * old_scale1 + tl.sum(dprob1 * block_probs1, axis=0)
        running_sum0 = running_sum0 * old_scale0 + tl.sum(block_probs0, axis=0)
        running_sum1 = running_sum1 * old_scale1 + tl.sum(block_probs1, axis=0)
        running_max0 = new_max0
        running_max1 = new_max1

    delta0 = dprob_acc0 / running_sum0
    delta1 = dprob_acc1 / running_sum1
    grad_q0 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    grad_q1 = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for support_start in range(0, topk, BLOCK_K):
        offs_k = support_start + tl.arange(0, BLOCK_K)
        support_mask = offs_k < topk
        selected = tl.load(
            topk_indices_ptr
            + batch_idx * ti_stride_b
            + query_idx * ti_stride_m
            + offs_k * ti_stride_k,
            mask=support_mask,
            other=0,
        )
        valid = support_mask & (selected <= q_start + query_idx)
        k = tl.load(
            key_ptr
            + selected[:, None] * k_stride_s
            + batch_idx * k_stride_b
            + group_idx * k_stride_g
            + offs_d_block * k_stride_d,
            mask=valid[:, None] & offs_d_mask,
            other=0.0,
        ).to(tl.float32)
        score0 = tl.sum(k * q0_block, axis=1) * softmax_scale
        score1 = tl.sum(k * q1_block, axis=1) * softmax_scale
        score0 = tl.where(valid, score0, -float("inf"))
        score1 = tl.where(valid, score1, -float("inf"))
        prob0 = tl.exp(score0 - running_max0) / running_sum0
        prob1 = tl.exp(score1 - running_max1) / running_sum1
        prob0 = tl.where(valid, prob0, 0.0)
        prob1 = tl.where(valid, prob1, 0.0)
        v = tl.load(
            value_ptr
            + selected[:, None] * v_stride_s
            + batch_idx * v_stride_b
            + group_idx * v_stride_g
            + offs_dv_block * v_stride_d,
            mask=valid[:, None] & offs_dv_mask,
            other=0.0,
        )
        dprob0 = tl.sum((v * grad_out0_block).to(tl.float32), axis=1)
        dprob1 = tl.sum((v * grad_out1_block).to(tl.float32), axis=1)
        dscores0 = tl.where(valid, prob0 * (dprob0 - delta0), 0.0)
        dscores1 = tl.where(valid, prob1 * (dprob1 - delta1), 0.0)
        grad_q0 += tl.sum(k * dscores0[:, None], axis=0) * softmax_scale
        grad_q1 += tl.sum(k * dscores1[:, None], axis=0) * softmax_scale

        grad_k = dscores0[:, None] * q0_block + dscores1[:, None] * q1_block
        grad_k = grad_k * softmax_scale
        tl.atomic_add(
            grad_key_ptr
            + selected[:, None] * gk_stride_s
            + batch_idx * gk_stride_b
            + group_idx * gk_stride_g
            + offs_d_block * gk_stride_d,
            grad_k,
            sem="relaxed",
            mask=valid[:, None] & offs_d_mask,
        )

        if VALUE_DTYPE == 1:
            prob0_for_value = prob0.to(tl.float16)
            prob1_for_value = prob1.to(tl.float16)
        elif VALUE_DTYPE == 2:
            prob0_for_value = prob0.to(tl.bfloat16)
            prob1_for_value = prob1.to(tl.bfloat16)
        else:
            prob0_for_value = prob0
            prob1_for_value = prob1
        grad_v = (
            prob0_for_value[:, None] * grad_out0_block
            + prob1_for_value[:, None] * grad_out1_block
        ).to(tl.float32)
        tl.atomic_add(
            grad_value_ptr
            + selected[:, None] * gv_stride_s
            + batch_idx * gv_stride_b
            + group_idx * gv_stride_g
            + offs_dv_block * gv_stride_d,
            grad_v,
            sem="relaxed",
            mask=valid[:, None] & offs_dv_mask,
        )

    tl.store(
        grad_query_ptr
        + query_idx * gq_stride_m
        + batch_idx * gq_stride_b
        + head0 * gq_stride_h
        + offs_d * gq_stride_d,
        grad_q0,
        mask=offs_d < HEAD_DIM,
    )
    tl.store(
        grad_query_ptr
        + query_idx * gq_stride_m
        + batch_idx * gq_stride_b
        + head1 * gq_stride_h
        + offs_d * gq_stride_d,
        grad_q1,
        mask=offs_d < HEAD_DIM,
    )
@triton.autotune(
    configs=_teacher_scores_autotune_configs(),
    key=["query_len", "topk", "HEAD_DIM"],
    reset_to_zero=["teacher_ptr"],
)
@triton.jit
def _dsa_teacher_scores_kernel(
    query_ptr,
    key_ptr,
    topk_indices_ptr,
    teacher_ptr,
    softmax_scale: tl.constexpr,
    q_start,
    query_len: tl.constexpr,
    topk: tl.constexpr,
    repeat_factor: tl.constexpr,
    q_stride_m: tl.constexpr,
    q_stride_b: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_stride_d: tl.constexpr,
    k_stride_s: tl.constexpr,
    k_stride_b: tl.constexpr,
    k_stride_g: tl.constexpr,
    k_stride_d: tl.constexpr,
    ti_stride_b: tl.constexpr,
    ti_stride_m: tl.constexpr,
    ti_stride_k: tl.constexpr,
    teacher_stride_b: tl.constexpr,
    teacher_stride_m: tl.constexpr,
    teacher_stride_k: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    query_idx = tl.program_id(0)
    batch_idx = tl.program_id(1)
    head_idx = tl.program_id(2)
    group_idx = head_idx // repeat_factor
    offs_d = tl.arange(0, BLOCK_D)
    offs_d_block = tl.broadcast_to(tl.expand_dims(offs_d, 0), (BLOCK_K, BLOCK_D))
    offs_d_mask = offs_d_block < HEAD_DIM
    q = tl.load(
        query_ptr
        + query_idx * q_stride_m
        + batch_idx * q_stride_b
        + head_idx * q_stride_h
        + offs_d * q_stride_d,
        mask=offs_d < HEAD_DIM,
        other=0.0,
    ).to(tl.float32)
    q_block = tl.broadcast_to(tl.expand_dims(q, 0), (BLOCK_K, BLOCK_D))
    running_max = tl.full((), -float("inf"), dtype=tl.float32)
    running_sum = tl.full((), 0.0, dtype=tl.float32)

    for support_start in range(0, topk, BLOCK_K):
        offs_k = support_start + tl.arange(0, BLOCK_K)
        support_mask = offs_k < topk
        selected = tl.load(
            topk_indices_ptr
            + batch_idx * ti_stride_b
            + query_idx * ti_stride_m
            + offs_k * ti_stride_k,
            mask=support_mask,
            other=0,
        )
        valid = support_mask & (selected <= q_start + query_idx)
        k = tl.load(
            key_ptr
            + selected[:, None] * k_stride_s
            + batch_idx * k_stride_b
            + group_idx * k_stride_g
            + offs_d_block * k_stride_d,
            mask=valid[:, None] & offs_d_mask,
            other=0.0,
        ).to(tl.float32)
        scores = tl.sum(k * q_block, axis=1) * softmax_scale
        scores = tl.where(valid, scores, -float("inf"))
        block_max = tl.max(scores, axis=0)
        new_max = tl.maximum(running_max, block_max)
        old_scale = tl.exp(running_max - new_max)
        block_sum = tl.sum(tl.exp(scores - new_max), axis=0)
        running_sum = running_sum * old_scale + block_sum
        running_max = new_max

    for support_start in range(0, topk, BLOCK_K):
        offs_k = support_start + tl.arange(0, BLOCK_K)
        support_mask = offs_k < topk
        selected = tl.load(
            topk_indices_ptr
            + batch_idx * ti_stride_b
            + query_idx * ti_stride_m
            + offs_k * ti_stride_k,
            mask=support_mask,
            other=0,
        )
        valid = support_mask & (selected <= q_start + query_idx)
        k = tl.load(
            key_ptr
            + selected[:, None] * k_stride_s
            + batch_idx * k_stride_b
            + group_idx * k_stride_g
            + offs_d_block * k_stride_d,
            mask=valid[:, None] & offs_d_mask,
            other=0.0,
        ).to(tl.float32)
        scores = tl.sum(k * q_block, axis=1) * softmax_scale
        scores = tl.where(valid, scores, -float("inf"))
        probs = tl.exp(scores - running_max) / running_sum
        tl.atomic_add(
            teacher_ptr
            + batch_idx * teacher_stride_b
            + query_idx * teacher_stride_m
            + offs_k * teacher_stride_k,
            probs,
            sem="relaxed",
            mask=support_mask,
        )


def _triton_sparse_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    topk_indices: torch.Tensor,
    softmax_scale: float,
    q_start: int,
) -> torch.Tensor:
    query_len, batch_size, num_heads, head_dim = query.shape
    num_groups = key.size(2)
    repeat_factor = num_heads // num_groups
    value_dim = value.size(-1)
    output = value.new_empty((query_len, batch_size, num_heads, value_dim))
    block_d = _next_power_of_2(head_dim)
    block_dv = max(32, _next_power_of_2(value_dim))
    grid = (query_len, batch_size, num_heads)
    _dsa_sparse_attention_forward_kernel[grid](
        query,
        key,
        value,
        topk_indices,
        output,
        softmax_scale,
        q_start,
        query_len,
        topk_indices.size(-1),
        repeat_factor,
        *query.stride(),
        *key.stride(),
        *value.stride(),
        *topk_indices.stride(),
        *output.stride(),
        HEAD_DIM=head_dim,
        VALUE_DIM=value_dim,
        BLOCK_D=block_d,
        BLOCK_DV=block_dv,
        VALUE_DTYPE=_value_dtype_tag(value.dtype),
    )
    return output


def _triton_sparse_attention_backward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    topk_indices: torch.Tensor,
    grad_output: torch.Tensor,
    grad_query: torch.Tensor,
    grad_key_accum: torch.Tensor,
    grad_value_accum: torch.Tensor,
    softmax_scale: float,
    q_start: int,
) -> None:
    query_len, batch_size, num_heads, head_dim = query.shape
    num_groups = key.size(2)
    repeat_factor = num_heads // num_groups
    value_dim = value.size(-1)
    block_d = _next_power_of_2(head_dim)
    block_dv = max(32, _next_power_of_2(value_dim))
    topk = topk_indices.size(-1)
    if (
        repeat_factor >= 2
        and repeat_factor % 2 == 0
        and block_d <= 256
        and block_dv <= 256
    ):
        pairs_per_group = repeat_factor // 2
        grid = (query_len, batch_size, num_groups * pairs_per_group)
        try:
            _dsa_sparse_attention_backward_pair_kernel[grid](
                query,
                key,
                value,
                topk_indices,
                grad_output,
                grad_query,
                grad_key_accum,
                grad_value_accum,
                softmax_scale,
                q_start,
                query_len,
                topk,
                repeat_factor,
                pairs_per_group,
                *query.stride(),
                *key.stride(),
                *value.stride(),
                *topk_indices.stride(),
                *grad_output.stride(),
                *grad_query.stride(),
                *grad_key_accum.stride(),
                *grad_value_accum.stride(),
                HEAD_DIM=head_dim,
                VALUE_DIM=value_dim,
                BLOCK_D=block_d,
                BLOCK_DV=block_dv,
                VALUE_DTYPE=_value_dtype_tag(value.dtype),
            )
            return
        except _TRITON_RESOURCE_ERRORS:
            pass
    grid = (query_len, batch_size, num_heads)
    _dsa_sparse_attention_backward_kernel[grid](
        query,
        key,
        value,
        topk_indices,
        grad_output,
        grad_query,
        grad_key_accum,
        grad_value_accum,
        softmax_scale,
        q_start,
        query_len,
        topk,
        repeat_factor,
        *query.stride(),
        *key.stride(),
        *value.stride(),
        *topk_indices.stride(),
        *grad_output.stride(),
        *grad_query.stride(),
        *grad_key_accum.stride(),
        *grad_value_accum.stride(),
        HEAD_DIM=head_dim,
        VALUE_DIM=value_dim,
        BLOCK_D=block_d,
        BLOCK_DV=block_dv,
        VALUE_DTYPE=_value_dtype_tag(value.dtype),
    )


def triton_sparse_attention_backward_supported(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    topk_indices: torch.Tensor,
    grad_output: torch.Tensor,
    grad_query: torch.Tensor,
) -> bool:
    return (
        _can_use_sparse_attention(query, key, value, topk_indices)
        and _supported_tensor(grad_output)
        and _supported_tensor(grad_query)
    )


def triton_sparse_attention_backward_path(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    topk_indices: torch.Tensor,
) -> str:
    if not _can_use_sparse_attention(query, key, value, topk_indices):
        return "unsupported"
    num_heads = query.size(2)
    num_groups = key.size(2)
    repeat_factor = num_heads // num_groups
    head_dim = query.size(-1)
    value_dim = value.size(-1)
    block_d = _next_power_of_2(head_dim)
    block_dv = max(32, _next_power_of_2(value_dim))
    if (
        repeat_factor >= 2
        and repeat_factor % 2 == 0
        and block_d <= 256
        and block_dv <= 256
    ):
        return "pair"
    return "row"


def triton_sparse_attention_backward_accumulate(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    topk_indices: torch.Tensor,
    grad_output: torch.Tensor,
    grad_query: torch.Tensor,
    grad_key_accum: torch.Tensor,
    grad_value_accum: torch.Tensor,
    softmax_scale: float,
    q_start: int,
) -> bool:
    if not triton_sparse_attention_backward_supported(
        query, key, value, topk_indices, grad_output, grad_query
    ):
        return False
    if not (
        grad_key_accum.is_cuda
        and grad_value_accum.is_cuda
        and grad_key_accum.dtype == torch.float32
        and grad_value_accum.dtype == torch.float32
    ):
        return False
    _triton_sparse_attention_backward(
        query,
        key,
        value,
        topk_indices,
        grad_output,
        grad_query,
        grad_key_accum,
        grad_value_accum,
        softmax_scale,
        q_start,
    )
    return True


def triton_linear_wgrad(
    grad_output: torch.Tensor,
    input_tensor: torch.Tensor,
    grad_weight: torch.Tensor,
) -> bool:
    if _triton_disabled():
        return False
    if not (
        _supported_tensor(grad_output)
        and _supported_tensor(input_tensor)
        and grad_weight.is_cuda
        and grad_weight.dtype == torch.float32
    ):
        return False
    if grad_output.size(-1) != grad_weight.size(0):
        return False
    if input_tensor.size(-1) != grad_weight.size(1):
        return False

    grad_output_2d = grad_output.reshape(-1, grad_output.size(-1))
    input_2d = input_tensor.reshape(-1, input_tensor.size(-1))
    if grad_output_2d.size(0) != input_2d.size(0):
        return False
    if grad_output_2d.size(0) == 0:
        return True

    out_delta = torch.empty_like(grad_weight, dtype=torch.float32)
    grid = lambda meta: (
        triton.cdiv(grad_weight.size(0), meta["BLOCK_O"]),
        triton.cdiv(grad_weight.size(1), meta["BLOCK_I"]),
    )
    _dsa_linear_wgrad_kernel[grid](
        grad_output_2d,
        input_2d,
        out_delta,
        grad_output_2d.size(0),
        grad_weight.size(0),
        grad_weight.size(1),
        *grad_output_2d.stride(),
        *input_2d.stride(),
        *out_delta.stride(),
        USE_BF16_OPERANDS=input_tensor.dtype == torch.bfloat16,
        USE_FP16_OPERANDS=input_tensor.dtype == torch.float16,
    )
    grad_weight.add_(out_delta)
    return True


def triton_selected_k_linear(
    hidden_states: torch.Tensor,
    topk_indices: torch.Tensor,
    linear_k_weight: torch.Tensor,
    input_norm_weight: Optional[torch.Tensor] = None,
    input_norm_stats: Optional[torch.Tensor] = None,
    input_norm_zero_centered_gamma: bool = False,
) -> Optional[torch.Tensor]:
    if _triton_disabled():
        return None
    if torch.is_grad_enabled() and (
        hidden_states.requires_grad or linear_k_weight.requires_grad
    ):
        return None
    if not (
        _supported_tensor(hidden_states)
        and _supported_tensor(linear_k_weight)
        and _supported_index_tensor(topk_indices)
    ):
        return None
    if linear_k_weight.dtype != hidden_states.dtype:
        return None
    apply_input_norm = input_norm_weight is not None or input_norm_stats is not None
    if apply_input_norm:
        if input_norm_weight is None or input_norm_stats is None:
            return None
        if not _supported_tensor(input_norm_weight):
            return None
        if (
            input_norm_weight.device != hidden_states.device
            or input_norm_stats.device != hidden_states.device
        ):
            return None
        if input_norm_weight.numel() != hidden_states.size(-1):
            return None
        if input_norm_stats.shape != hidden_states.shape[:2]:
            return None
        if not input_norm_stats.is_cuda or input_norm_stats.dtype != torch.float32:
            return None
    if hidden_states.dim() != 3 or topk_indices.dim() != 3 or linear_k_weight.dim() != 2:
        return None
    if hidden_states.size(1) != topk_indices.size(0):
        return None
    out_features, hidden_size = linear_k_weight.shape
    if hidden_states.size(2) != hidden_size:
        return None
    if out_features > 256 or out_features < 16 or hidden_size < 16:
        return None
    total_rows = topk_indices.numel()
    if total_rows == 0:
        return hidden_states.new_empty(
            (*topk_indices.shape, out_features), dtype=hidden_states.dtype
        )

    # Match PyTorch/TE zero-centered-gamma semantics: gamma + 1 is formed in
    # the norm parameter dtype before normalization proceeds in FP32.
    effective_input_norm_weight = None
    if apply_input_norm:
        effective_input_norm_weight = input_norm_weight.detach()
        if input_norm_zero_centered_gamma:
            effective_input_norm_weight = effective_input_norm_weight + 1.0

    output = hidden_states.new_empty(
        (*topk_indices.shape, out_features), dtype=hidden_states.dtype
    )
    query_len = topk_indices.size(1)
    topk = topk_indices.size(2)
    grid = lambda meta: (
        triton.cdiv(total_rows, meta["BLOCK_N"]),
        triton.cdiv(out_features, meta["BLOCK_D"]),
    )
    try:
        _dsa_selected_k_linear_kernel[grid](
            hidden_states,
            topk_indices,
            linear_k_weight,
            effective_input_norm_weight if apply_input_norm else linear_k_weight,
            input_norm_stats if apply_input_norm else hidden_states,
            output,
            total_rows,
            query_len,
            topk,
            hidden_size,
            out_features,
            *hidden_states.stride(),
            *topk_indices.stride(),
            *linear_k_weight.stride(),
            (
                effective_input_norm_weight.stride(0)
                if apply_input_norm
                else linear_k_weight.stride(-1)
            ),
            *(
                input_norm_stats.stride()
                if apply_input_norm
                else hidden_states.stride()[:2]
            ),
            *output.stride(),
            USE_BF16_OPERANDS=hidden_states.dtype == torch.bfloat16,
            USE_FP16_OPERANDS=hidden_states.dtype == torch.float16,
            APPLY_INPUT_NORM=apply_input_norm,
        )
    except _TRITON_RESOURCE_ERRORS:
        return None
    return output


def triton_selected_index_scores_from_hidden(
    hidden_states: torch.Tensor,
    topk_indices: torch.Tensor,
    linear_k_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    k_norm_bias: torch.Tensor,
    q_index: torch.Tensor,
    weights: torch.Tensor,
    inv_freq: torch.Tensor,
    q_start: int,
    k_norm_eps: float,
    index_rotary_dim: int,
    rotary_interleaved: bool,
    use_indexer_rope: bool,
    use_hadamard: bool,
    has_k_norm_bias: bool,
    mscale: float,
    interpolation_scale: float,
    return_k_linear: bool = False,
    input_norm_weight: Optional[torch.Tensor] = None,
    input_norm_stats: Optional[torch.Tensor] = None,
    input_norm_zero_centered_gamma: bool = False,
) -> Optional[Tuple[torch.Tensor, Optional[torch.Tensor]]]:
    if _triton_disabled():
        return None
    if torch.is_grad_enabled() and (
        hidden_states.requires_grad
        or linear_k_weight.requires_grad
        or k_norm_weight.requires_grad
        or k_norm_bias.requires_grad
    ):
        return None
    if not (
        _supported_tensor(hidden_states)
        and _supported_tensor(linear_k_weight)
        and _supported_tensor(k_norm_weight)
        and _supported_tensor(q_index)
        and _supported_tensor(weights)
        and _supported_index_tensor(topk_indices)
    ):
        return None
    if has_k_norm_bias and not _supported_tensor(k_norm_bias):
        return None
    if use_indexer_rope and not _supported_tensor(inv_freq):
        return None
    if linear_k_weight.dtype != hidden_states.dtype:
        return None
    apply_input_norm = input_norm_weight is not None or input_norm_stats is not None
    if apply_input_norm:
        if input_norm_weight is None or input_norm_stats is None:
            return None
        if not _supported_tensor(input_norm_weight):
            return None
        if (
            input_norm_weight.device != hidden_states.device
            or input_norm_stats.device != hidden_states.device
        ):
            return None
        if input_norm_weight.numel() != hidden_states.size(-1):
            return None
        if input_norm_stats.shape != hidden_states.shape[:2]:
            return None
        if not input_norm_stats.is_cuda or input_norm_stats.dtype != torch.float32:
            return None
    if hidden_states.dim() != 3 or topk_indices.dim() != 3 or linear_k_weight.dim() != 2:
        return None
    if hidden_states.size(1) != topk_indices.size(0):
        return None
    if q_index.dim() != 4 or weights.dim() != 3:
        return None
    batch_size, query_len, topk = topk_indices.shape
    index_heads = q_index.size(2)
    out_features = q_index.size(3)
    if q_index.shape[:2] != (query_len, batch_size):
        return None
    if weights.shape != (query_len, batch_size, index_heads):
        return None
    if tuple(linear_k_weight.shape) != (out_features, hidden_states.size(2)):
        return None
    if k_norm_weight.numel() != out_features:
        return None
    if has_k_norm_bias and k_norm_bias.numel() != out_features:
        return None
    hidden_size = hidden_states.size(2)
    if out_features > 256 or out_features < 16 or hidden_size < 16:
        return None
    if max(16, _next_power_of_2(out_features)) != out_features:
        return None
    if topk > _MAX_TRITON_SUPPORT_TOPK:
        return None
    if index_rotary_dim < 0 or index_rotary_dim > out_features or index_rotary_dim % 2 != 0:
        return None
    if use_indexer_rope and inv_freq.numel() < index_rotary_dim // 2:
        return None
    hadamard = None
    if use_hadamard:
        if out_features > 128:
            return None
        hadamard = _hadamard_matrix(out_features, hidden_states.device)
        if hadamard is None:
            return None
    else:
        hadamard = k_norm_weight.new_empty((1, 1), dtype=torch.bfloat16)
    if not use_indexer_rope:
        inv_freq = k_norm_weight.new_empty((1,), dtype=torch.float32)

    # Keep gamma formation in the source parameter dtype, as in the exact
    # tiled PyTorch normalization path, before the kernel promotes it to FP32.
    effective_input_norm_weight = None
    if apply_input_norm:
        effective_input_norm_weight = input_norm_weight.detach()
        if input_norm_zero_centered_gamma:
            effective_input_norm_weight = effective_input_norm_weight + 1.0

    total_rows = topk_indices.numel()
    scores = torch.empty((batch_size, query_len, topk), device=hidden_states.device, dtype=torch.float32)
    store_k_linear = return_k_linear or use_indexer_rope
    k_linear = (
        hidden_states.new_empty((*topk_indices.shape, out_features), dtype=hidden_states.dtype)
        if store_k_linear
        else hidden_states.new_empty((1, 1, 1, out_features), dtype=hidden_states.dtype)
    )
    if total_rows == 0:
        return scores, k_linear if return_k_linear else None

    grid = lambda meta: (triton.cdiv(total_rows, meta["BLOCK_N"]),)
    try:
        _dsa_selected_k_project_score_kernel[grid](
            hidden_states,
            topk_indices,
            linear_k_weight,
            effective_input_norm_weight if apply_input_norm else linear_k_weight,
            input_norm_stats if apply_input_norm else hidden_states,
            k_norm_weight,
            k_norm_bias,
            q_index,
            weights,
            inv_freq,
            hadamard,
            scores,
            k_linear,
            total_rows,
            query_len,
            topk,
            hidden_size,
            out_features,
            q_start,
            k_norm_eps,
            float(mscale),
            float(interpolation_scale),
            *hidden_states.stride(),
            *topk_indices.stride(),
            *linear_k_weight.stride(),
            (
                effective_input_norm_weight.stride(0)
                if apply_input_norm
                else linear_k_weight.stride(-1)
            ),
            *(
                input_norm_stats.stride()
                if apply_input_norm
                else hidden_states.stride()[:2]
            ),
            *k_norm_weight.stride(),
            *(k_norm_bias.stride() if has_k_norm_bias else (1,)),
            *q_index.stride(),
            *weights.stride(),
            *inv_freq.stride(),
            *hadamard.stride(),
            *scores.stride(),
            *k_linear.stride(),
            INDEX_HEADS=index_heads,
            INDEX_ROTARY_DIM=index_rotary_dim,
            ROTARY_INTERLEAVED=rotary_interleaved,
            USE_ROPE=use_indexer_rope,
            USE_HADAMARD=use_hadamard,
            HAS_BIAS=has_k_norm_bias,
            APPLY_INPUT_NORM=apply_input_norm,
            USE_BF16_OPERANDS=hidden_states.dtype == torch.bfloat16,
            USE_FP16_OPERANDS=hidden_states.dtype == torch.float16,
            STORE_K_LINEAR=store_k_linear,
            BLOCK_D=out_features,
        )
    except _TRITON_RESOURCE_ERRORS:
        return None
    return scores, k_linear if return_k_linear else None


def triton_k_ln_backward_prepare(
    grad_k_norm: torch.Tensor,
    k_linear: torch.Tensor,
    k_norm_weight: torch.Tensor,
    k_norm_eps: float,
    grad_k_norm_weight: Optional[torch.Tensor],
    grad_k_norm_bias: Optional[torch.Tensor],
    scratch_dtype: torch.dtype,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    if _triton_disabled():
        return None
    if not (_supported_tensor(grad_k_norm) and _supported_tensor(k_linear)):
        return None
    if grad_k_norm.shape != k_linear.shape or grad_k_norm.dim() != 4:
        return None
    if not _supported_tensor(k_norm_weight) or k_norm_weight.numel() != grad_k_norm.size(-1):
        return None
    if scratch_dtype not in (torch.float32, torch.float16, torch.bfloat16):
        return None
    if grad_k_norm_weight is not None and (
        not grad_k_norm_weight.is_cuda or grad_k_norm_weight.dtype != torch.float32
    ):
        return None
    if grad_k_norm_bias is not None and (
        not grad_k_norm_bias.is_cuda or grad_k_norm_bias.dtype != torch.float32
    ):
        return None

    batch_size, query_len, topk, out_features = grad_k_norm.shape
    if out_features > 256 or out_features < 16:
        return None
    total_rows = batch_size * query_len * topk
    if total_rows == 0:
        grad_k_linear = torch.empty_like(k_linear, dtype=scratch_dtype)
        empty_partials = grad_k_norm.new_zeros((0, out_features), dtype=torch.float32)
        return grad_k_linear, empty_partials, empty_partials

    block_d_full = max(16, _next_power_of_2(out_features))
    min_block_n = 32
    num_row_blocks = triton.cdiv(total_rows, min_block_n)
    partial_weight = torch.zeros(
        (num_row_blocks, out_features), device=grad_k_norm.device, dtype=torch.float32
    )
    partial_bias = torch.zeros(
        (num_row_blocks, out_features), device=grad_k_norm.device, dtype=torch.float32
    )
    grad_k_linear = torch.empty_like(k_linear, dtype=scratch_dtype)
    ln_grid = lambda meta: (
        triton.cdiv(total_rows, meta["BLOCK_N"]),
        triton.cdiv(out_features, meta["BLOCK_D"]),
    )
    try:
        _dsa_k_ln_backward_kernel[ln_grid](
            grad_k_norm,
            k_linear,
            k_norm_weight,
            grad_k_linear,
            partial_weight,
            partial_bias,
            total_rows,
            query_len,
            topk,
            out_features,
            k_norm_eps,
            *grad_k_norm.stride(),
            *k_linear.stride(),
            *k_norm_weight.stride(),
            *grad_k_linear.stride(),
            *partial_weight.stride(),
            *partial_bias.stride(),
            HAS_WEIGHT_GRAD=grad_k_norm_weight is not None,
            HAS_BIAS_GRAD=grad_k_norm_bias is not None,
            BLOCK_D_FULL=block_d_full,
        )
    except _TRITON_RESOURCE_ERRORS:
        return None
    return grad_k_linear, partial_weight, partial_bias


def triton_k_ln_param_reduce(
    partial_weight: torch.Tensor,
    partial_bias: torch.Tensor,
    grad_k_norm_weight: Optional[torch.Tensor],
    grad_k_norm_bias: Optional[torch.Tensor],
) -> bool:
    if grad_k_norm_weight is not None:
        grad_k_norm_weight.add_(partial_weight.sum(dim=0))
    if grad_k_norm_bias is not None:
        grad_k_norm_bias.add_(partial_bias.sum(dim=0))
    return True


def triton_gathered_linear_wgrad(
    grad_output: torch.Tensor,
    hidden_states: torch.Tensor,
    topk_indices: torch.Tensor,
    grad_weight: torch.Tensor,
) -> bool:
    if _triton_disabled():
        return False
    if not (
        _supported_tensor(grad_output)
        and _supported_tensor(hidden_states)
        and _supported_index_tensor(topk_indices)
        and grad_weight.is_cuda
        and grad_weight.dtype == torch.float32
    ):
        return False
    if grad_output.dim() != 4 or hidden_states.dim() != 3 or topk_indices.dim() != 3:
        return False
    if grad_output.shape[:3] != topk_indices.shape:
        return False
    if hidden_states.size(1) != topk_indices.size(0):
        return False

    batch_size, query_len, topk, out_features = grad_output.shape
    hidden_size = hidden_states.size(2)
    if tuple(grad_weight.shape) != (out_features, hidden_size):
        return False
    if out_features > 256 or out_features < 16 or hidden_size < 16:
        return False
    total_rows = batch_size * query_len * topk
    if total_rows == 0:
        return True

    out_delta = torch.empty_like(grad_weight, dtype=torch.float32)
    wgrad_grid = lambda meta: (
        triton.cdiv(out_features, meta["BLOCK_O"]),
        triton.cdiv(hidden_size, meta["BLOCK_I"]),
    )
    try:
        _dsa_gathered_linear_wgrad_kernel[wgrad_grid](
            grad_output,
            hidden_states,
            topk_indices,
            out_delta,
            total_rows,
            query_len,
            topk,
            out_features,
            hidden_size,
            *grad_output.stride(),
            *hidden_states.stride(),
            *topk_indices.stride(),
            *out_delta.stride(),
            USE_BF16_OPERANDS=hidden_states.dtype == torch.bfloat16,
            USE_FP16_OPERANDS=hidden_states.dtype == torch.float16,
        )
    except _TRITON_RESOURCE_ERRORS:
        return False
    grad_weight.add_(out_delta)
    return True


def triton_simplified_input_norm_stats(
    hidden_states: torch.Tensor,
    eps: float,
    normalization: str,
) -> Optional[torch.Tensor]:
    """Return FP32 RMSNorm statistics for the simplified learned-K WGRAD fast path."""
    if _triton_disabled() or not _supported_tensor(hidden_states) or hidden_states.dim() != 3:
        return None
    # Native LayerNorm and this explicit Triton reduction need not produce the
    # same model-dtype activation. Let LayerNorm use the exact forward
    # recomputation path before its WGRAD GEMM instead.
    if normalization != "RMSNorm":
        return None
    sequence_length, batch_size, hidden_size = hidden_states.shape
    if hidden_size < 1 or hidden_size > 65536:
        return None
    rstd = torch.empty(
        (sequence_length, batch_size), device=hidden_states.device, dtype=torch.float32
    )
    total_rows = sequence_length * batch_size
    if total_rows == 0:
        return rstd
    block_h = _next_power_of_2(hidden_size)
    try:
        _dsa_simplified_input_norm_stats_kernel[(total_rows,)](
            hidden_states,
            rstd,
            total_rows,
            batch_size,
            hidden_size,
            *hidden_states.stride(),
            *rstd.stride(),
            eps=float(eps),
            BLOCK_H=block_h,
            num_warps=8 if block_h >= 2048 else 4,
        )
    except _TRITON_RESOURCE_ERRORS:
        return None
    return rstd


def triton_simplified_gathered_linear_wgrad(
    grad_output: torch.Tensor,
    hidden_states: torch.Tensor,
    topk_indices: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: Optional[torch.Tensor],
    norm_stats: torch.Tensor,
    normalization: str,
    zero_centered_gamma: bool,
    grad_weight: torch.Tensor,
) -> bool:
    """Accumulate simplified learned-K WGRAD from normalized selected input rows."""
    if _triton_disabled() or normalization != "RMSNorm":
        return False
    del norm_bias
    rstd = norm_stats
    if not (
        _supported_tensor(grad_output)
        and _supported_tensor(hidden_states)
        and _supported_index_tensor(topk_indices)
        and _supported_tensor(norm_weight)
        and rstd.is_cuda
        and rstd.dtype == torch.float32
        and grad_weight.is_cuda
        and grad_weight.dtype == torch.float32
    ):
        return False
    if grad_output.dim() != 4 or hidden_states.dim() != 3 or topk_indices.dim() != 3:
        return False
    if grad_output.shape[:3] != topk_indices.shape:
        return False
    if hidden_states.size(1) != topk_indices.size(0):
        return False
    if rstd.shape != hidden_states.shape[:2]:
        return False

    batch_size, query_len, topk, out_features = grad_output.shape
    hidden_size = hidden_states.size(2)
    if tuple(grad_weight.shape) != (out_features, hidden_size):
        return False
    if norm_weight.numel() != hidden_size or norm_weight.device != hidden_states.device:
        return False
    if out_features > 256 or out_features < 16 or hidden_size < 16:
        return False
    total_rows = batch_size * query_len * topk
    if total_rows == 0:
        return True

    # Form the effective gamma in the parameter dtype. This agrees with the
    # forward normalization even when norm and activation dtypes differ.
    effective_norm_weight = norm_weight.detach()
    if zero_centered_gamma:
        effective_norm_weight = effective_norm_weight + 1.0

    out_delta = torch.empty_like(grad_weight, dtype=torch.float32)
    wgrad_grid = lambda meta: (
        triton.cdiv(out_features, meta["BLOCK_O"]),
        triton.cdiv(hidden_size, meta["BLOCK_I"]),
    )
    try:
        _dsa_simplified_gathered_linear_wgrad_kernel[wgrad_grid](
            grad_output,
            hidden_states,
            topk_indices,
            effective_norm_weight,
            rstd,
            out_delta,
            total_rows,
            query_len,
            topk,
            out_features,
            hidden_size,
            *grad_output.stride(),
            *hidden_states.stride(),
            *topk_indices.stride(),
            effective_norm_weight.stride(0),
            *rstd.stride(),
            *out_delta.stride(),
            USE_BF16_OPERANDS=hidden_states.dtype == torch.bfloat16,
            USE_FP16_OPERANDS=hidden_states.dtype == torch.float16,
        )
    except _TRITON_RESOURCE_ERRORS:
        return False
    grad_weight.add_(out_delta)
    return True


def triton_scatter_selected_grad_to_sequence(
    grad_output: torch.Tensor,
    topk_indices: torch.Tensor,
    sequence_length: int,
) -> Optional[torch.Tensor]:
    if _triton_disabled():
        return None
    if not (_supported_tensor(grad_output) and _supported_index_tensor(topk_indices)):
        return None
    if grad_output.dim() != 4 or topk_indices.dim() != 3:
        return None
    if grad_output.shape[:3] != topk_indices.shape:
        return None
    if sequence_length <= 0:
        return None

    batch_size, query_len, topk, out_features = grad_output.shape
    if out_features > 256 or out_features < 16:
        return None
    total_rows = batch_size * query_len * topk
    out = torch.zeros(
        (sequence_length, batch_size, out_features),
        device=grad_output.device,
        dtype=torch.float32,
    )
    if total_rows == 0:
        return out

    scatter_grid = lambda meta: (
        triton.cdiv(total_rows, meta["BLOCK_N"]),
        triton.cdiv(out_features, meta["BLOCK_D"]),
    )
    try:
        _dsa_scatter_selected_grad_to_sequence_kernel[scatter_grid](
            grad_output,
            topk_indices,
            out,
            total_rows,
            sequence_length,
            query_len,
            topk,
            out_features,
            *grad_output.stride(),
            *topk_indices.stride(),
            *out.stride(),
        )
    except _TRITON_RESOURCE_ERRORS:
        return None
    return out


def _merge_topk_tensors(
    running_scores: Optional[torch.Tensor],
    running_indices: Optional[torch.Tensor],
    block_scores: torch.Tensor,
    block_indices: torch.Tensor,
    topk: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if running_scores is None or running_indices is None:
        return block_scores, block_indices
    merged_scores = torch.cat((running_scores, block_scores), dim=-1)
    merged_indices = torch.cat((running_indices, block_indices), dim=-1)
    keep = merged_scores.topk(min(topk, merged_scores.size(-1)), dim=-1).indices
    return torch.gather(merged_scores, -1, keep), torch.gather(merged_indices, -1, keep)


def _triton_topk_index_block_tiled_once(
    q_index: torch.Tensor,
    weights: torch.Tensor,
    k_index: torch.Tensor,
    topk: int,
    q_start: int,
    k_start: int,
    apply_relu: bool = True,
    score_scale: float = 1.0,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    if not _can_use_index_scores(q_index, weights, k_index, topk):
        return None
    query_len, batch_size, index_heads, index_head_dim = q_index.shape
    key_len = k_index.size(0)
    block_topk = min(topk, key_len)
    block_d = max(32, _next_power_of_2(index_head_dim))
    block_d_tile = min(block_d, 64 if q_index.dtype == torch.float32 else 128)
    scores = torch.empty((batch_size, query_len, block_topk), device=q_index.device, dtype=torch.float32)
    indices = torch.empty((batch_size, query_len, block_topk), device=q_index.device, dtype=torch.long)
    grid = lambda meta: (batch_size, triton.cdiv(query_len, meta["BLOCK_M"]))
    try:
        _dsa_topk_index_block_tiled_kernel[grid](
            q_index,
            weights,
            k_index,
            scores,
            indices,
            q_start,
            k_start,
            query_len,
            key_len,
            block_topk,
            *q_index.stride(),
            *weights.stride(),
            *k_index.stride(),
            *scores.stride(),
            *indices.stride(),
            INDEX_HEADS=index_heads,
            INDEX_HEAD_DIM=index_head_dim,
            APPLY_RELU=apply_relu,
            SCORE_SCALE=float(score_scale),
            DOT_INPUT_PRECISION="tf32" if apply_relu else "ieee",
            BLOCK_D=block_d,
            BLOCK_D_TILE=block_d_tile,
        )
    except _TRITON_RESOURCE_ERRORS:
        return None
    return scores, indices


def _triton_topk_index_block_query_once(
    q_index: torch.Tensor,
    weights: torch.Tensor,
    k_index: torch.Tensor,
    topk: int,
    q_start: int,
    k_start: int,
    apply_relu: bool = True,
    score_scale: float = 1.0,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    if not _can_use_index_scores(q_index, weights, k_index, topk):
        return None
    query_len, batch_size, index_heads, index_head_dim = q_index.shape
    key_len = k_index.size(0)
    block_topk = min(topk, key_len)
    block_n = _next_power_of_2(key_len)
    block_d = _next_power_of_2(index_head_dim)
    scores = torch.empty((batch_size, query_len, block_topk), device=q_index.device, dtype=torch.float32)
    indices = torch.empty((batch_size, query_len, block_topk), device=q_index.device, dtype=torch.long)
    grid = (batch_size, query_len)
    _dsa_topk_index_block_kernel[grid](
        q_index,
        weights,
        k_index,
        scores,
        indices,
        q_start,
        k_start,
        query_len,
        key_len,
        block_topk,
        *q_index.stride(),
        *weights.stride(),
        *k_index.stride(),
        *scores.stride(),
        *indices.stride(),
        INDEX_HEADS=index_heads,
        INDEX_HEAD_DIM=index_head_dim,
        APPLY_RELU=apply_relu,
        SCORE_SCALE=float(score_scale),
        BLOCK_N=block_n,
        BLOCK_D=block_d,
    )
    return scores, indices


def triton_topk_index_block(
    q_index: torch.Tensor,
    weights: torch.Tensor,
    k_index: torch.Tensor,
    topk: int,
    q_start: int,
    k_start: int,
    apply_relu: bool = True,
    score_scale: float = 1.0,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    if not _can_use_index_scores(q_index, weights, k_index, topk):
        return None
    key_len = k_index.size(0)
    block_topk = min(topk, key_len)
    # Keep the Tensor Core tiled kernel in a compile-friendly key width and merge sub-block top-k
    # results exactly the same way the outer streamed router merges key chunks.
    sub_block_n = 256
    if key_len <= sub_block_n:
        tiled = _triton_topk_index_block_tiled_once(
            q_index, weights, k_index, topk, q_start, k_start, apply_relu, score_scale
        )
        if tiled is not None:
            if block_topk == key_len:
                block_scores, block_indices = tiled
                keep = block_scores.topk(block_topk, dim=-1).indices
                return torch.gather(block_scores, -1, keep), torch.gather(block_indices, -1, keep)
            return tiled
        return _triton_topk_index_block_query_once(
            q_index, weights, k_index, topk, q_start, k_start, apply_relu, score_scale
        )

    running_scores = None
    running_indices = None
    for sub_start in range(0, key_len, sub_block_n):
        sub_end = min(sub_start + sub_block_n, key_len)
        sub_topk = min(block_topk, sub_end - sub_start)
        sub_scores_indices = _triton_topk_index_block_tiled_once(
            q_index,
            weights,
            k_index[sub_start:sub_end],
            sub_topk,
            q_start,
            k_start + sub_start,
            apply_relu,
            score_scale,
        )
        if sub_scores_indices is None:
            return _triton_topk_index_block_query_once(
                q_index,
                weights,
                k_index,
                topk,
                q_start,
                k_start,
                apply_relu,
                score_scale,
            )
        sub_scores, sub_indices = sub_scores_indices
        running_scores, running_indices = _merge_topk_tensors(
            running_scores, running_indices, sub_scores, sub_indices, block_topk
        )
    return running_scores, running_indices


def _can_use_simplified_scores(
    q_index: torch.Tensor,
    key: torch.Tensor,
    topk_indices: Optional[torch.Tensor] = None,
) -> bool:
    if _triton_disabled() or not (_supported_tensor(q_index) and _supported_tensor(key)):
        return False
    if q_index.dim() != 4 or key.dim() != 4:
        return False
    if (
        q_index.device != key.device
        or q_index.dtype != key.dtype
        or q_index.size(1) != key.size(1)
        or q_index.size(2) != 1
        or key.size(2) != 1
        or q_index.size(-1) != key.size(-1)
    ):
        return False
    if q_index.size(-1) > 256:
        return False
    return topk_indices is None or (
        _supported_index_tensor(topk_indices)
        and topk_indices.shape[:2] == (q_index.size(1), q_index.size(0))
        and topk_indices.size(-1) <= _MAX_TRITON_SUPPORT_TOPK
    )


def triton_simplified_selected_index_scores(
    q_index: torch.Tensor,
    key: torch.Tensor,
    topk_indices: torch.Tensor,
    score_scale: float,
    q_start: int,
) -> Optional[torch.Tensor]:
    """Score selected main-attention keys for the one-head simplified indexer."""
    if not _can_use_simplified_scores(q_index, key, topk_indices):
        return None
    query_len, batch_size, _, head_dim = q_index.shape
    topk = topk_indices.size(-1)
    scores = torch.empty(
        (batch_size, query_len, topk), device=q_index.device, dtype=torch.float32
    )
    block_d = max(16, _next_power_of_2(head_dim))
    grid = lambda meta: (batch_size, query_len, triton.cdiv(topk, meta["BLOCK_K"]))
    try:
        _dsa_simplified_selected_scores_kernel[grid](
            q_index,
            key,
            topk_indices,
            scores,
            q_start,
            query_len,
            topk,
            *q_index.stride(),
            *key.stride(),
            *topk_indices.stride(),
            *scores.stride(),
            SCORE_SCALE=float(score_scale),
            HEAD_DIM=head_dim,
            BLOCK_D=block_d,
        )
    except _TRITON_RESOURCE_ERRORS:
        return None
    return scores


def triton_simplified_selected_index_scores_backward(
    key: torch.Tensor,
    topk_indices: torch.Tensor,
    grad_scores: torch.Tensor,
    score_scale: float,
    q_start: int,
) -> Optional[torch.Tensor]:
    """Return FP32 dQ without producing a gradient for the detached main K."""
    query_len = topk_indices.size(1)
    batch_size = topk_indices.size(0)
    if (
        _triton_disabled()
        or not _supported_tensor(key)
        or key.dim() != 4
        or key.size(2) != 1
        or key.size(-1) > 256
        or topk_indices.dim() != 3
        or topk_indices.size(0) != key.size(1)
        or not _supported_index_tensor(topk_indices)
        or topk_indices.size(-1) > _MAX_TRITON_SUPPORT_TOPK
    ):
        return None
    if not _supported_tensor(grad_scores) or grad_scores.shape != topk_indices.shape:
        return None
    topk = topk_indices.size(-1)
    head_dim = key.size(-1)
    grad_q = torch.empty(
        (query_len, batch_size, 1, head_dim), device=key.device, dtype=torch.float32
    )
    block_d = max(16, _next_power_of_2(head_dim))
    grid = (batch_size, query_len)
    grad_scores = grad_scores.contiguous()
    try:
        _dsa_simplified_selected_scores_backward_kernel[grid](
            key,
            topk_indices,
            grad_scores,
            grad_q,
            q_start,
            query_len,
            topk,
            *key.stride(),
            *topk_indices.stride(),
            *grad_scores.stride(),
            *grad_q.stride(),
            SCORE_SCALE=float(score_scale),
            HEAD_DIM=head_dim,
            BLOCK_D=block_d,
        )
    except _TRITON_RESOURCE_ERRORS:
        return None
    return grad_q


def triton_simplified_selected_index_scores_backward_qk(
    q_index: torch.Tensor,
    selected_k_index: torch.Tensor,
    topk_indices: torch.Tensor,
    grad_scores: torch.Tensor,
    score_scale: float,
    q_start: int,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    if _triton_disabled() or not (
        _supported_tensor(q_index)
        and _supported_tensor(selected_k_index)
        and _supported_tensor(grad_scores)
        and _supported_index_tensor(topk_indices)
    ):
        return None
    if q_index.dim() != 4 or selected_k_index.dim() != 4 or topk_indices.dim() != 3:
        return None
    query_len, batch_size, index_heads, head_dim = q_index.shape
    if index_heads != 1 or head_dim > 256:
        return None
    if selected_k_index.shape != (*topk_indices.shape, head_dim):
        return None
    if grad_scores.shape != topk_indices.shape or topk_indices.size(-1) > _MAX_TRITON_SUPPORT_TOPK:
        return None
    if q_index.dtype != selected_k_index.dtype or q_index.device != selected_k_index.device:
        return None
    topk = topk_indices.size(-1)
    grad_q = torch.zeros_like(q_index, dtype=torch.float32)
    grad_selected_k = torch.empty_like(selected_k_index, dtype=torch.float32)
    grad_scores = grad_scores.contiguous()
    block_d = max(16, _next_power_of_2(head_dim))
    grid = lambda meta: (batch_size, query_len, triton.cdiv(topk, meta["BLOCK_K"]))
    try:
        _dsa_simplified_selected_scores_backward_qk_kernel[grid](
            q_index,
            selected_k_index,
            topk_indices,
            grad_scores,
            grad_q,
            grad_selected_k,
            q_start,
            query_len,
            topk,
            *q_index.stride(),
            *selected_k_index.stride(),
            *topk_indices.stride(),
            *grad_scores.stride(),
            *grad_q.stride(),
            *grad_selected_k.stride(),
            SCORE_SCALE=float(score_scale),
            HEAD_DIM=head_dim,
            BLOCK_D=block_d,
        )
    except _TRITON_RESOURCE_ERRORS:
        return None
    return grad_q, grad_selected_k


def triton_simplified_index_scores_block(
    q_index: torch.Tensor,
    key_block: torch.Tensor,
    score_scale: float,
    q_start: int,
    k_start: int,
) -> Optional[torch.Tensor]:
    """Return a causal FP32 score tile using BF16/FP16 Tensor Core operands."""
    if not _can_use_simplified_scores(q_index, key_block):
        return None
    query_len, batch_size, _, head_dim = q_index.shape
    key_len = key_block.size(0)
    scores = torch.empty(
        (batch_size, query_len, key_len), device=q_index.device, dtype=torch.float32
    )
    block_d = max(16, _next_power_of_2(head_dim))
    grid = lambda meta: (
        batch_size,
        triton.cdiv(query_len, meta["BLOCK_M"]),
        triton.cdiv(key_len, meta["BLOCK_N"]),
    )
    try:
        _dsa_simplified_score_block_kernel[grid](
            q_index,
            key_block,
            scores,
            q_start,
            k_start,
            query_len,
            key_len,
            *q_index.stride(),
            *key_block.stride(),
            *scores.stride(),
            SCORE_SCALE=float(score_scale),
            HEAD_DIM=head_dim,
            BLOCK_D=block_d,
        )
    except _TRITON_RESOURCE_ERRORS:
        return None
    return scores


def _triton_selected_index_scores_forward(
    q_index: torch.Tensor,
    weights: torch.Tensor,
    selected_k_index: torch.Tensor,
    topk_indices: torch.Tensor,
    q_start: int,
) -> Optional[torch.Tensor]:
    if not _can_use_selected_index_scores(q_index, weights, selected_k_index, topk_indices):
        return None
    batch_size, query_len, topk, index_head_dim = selected_k_index.shape
    index_heads = q_index.size(2)
    block_d = max(32, _next_power_of_2(index_head_dim))
    scores = torch.empty((batch_size, query_len, topk), device=q_index.device, dtype=torch.float32)
    grid = lambda meta: (batch_size, query_len, triton.cdiv(topk, meta["BLOCK_K"]))
    _dsa_selected_index_scores_kernel[grid](
        q_index,
        weights,
        selected_k_index,
        topk_indices,
        scores,
        q_start,
        query_len,
        topk,
        *q_index.stride(),
        *weights.stride(),
        *selected_k_index.stride(),
        *topk_indices.stride(),
        *scores.stride(),
        INDEX_HEADS=index_heads,
        INDEX_HEAD_DIM=index_head_dim,
        BLOCK_D=block_d,
    )
    return scores


def _triton_selected_index_kl_loss(
    q_index: torch.Tensor,
    weights: torch.Tensor,
    selected_k_index: torch.Tensor,
    topk_indices: torch.Tensor,
    teacher: torch.Tensor,
    loss_scale: float,
    q_start: int,
) -> torch.Tensor:
    batch_size, query_len, topk, index_head_dim = selected_k_index.shape
    index_heads = q_index.size(2)
    block_d = max(32, _next_power_of_2(index_head_dim))
    block_k = max(16, _next_power_of_2(topk))
    partial = torch.empty((batch_size, query_len), device=q_index.device, dtype=torch.float32)
    grid = (batch_size, query_len)
    _dsa_selected_index_kl_loss_kernel[grid](
        q_index,
        weights,
        selected_k_index,
        topk_indices,
        teacher,
        partial,
        q_start,
        query_len,
        topk,
        float(loss_scale),
        *q_index.stride(),
        *weights.stride(),
        *selected_k_index.stride(),
        *topk_indices.stride(),
        *teacher.stride(),
        *partial.stride(),
        INDEX_HEADS=index_heads,
        INDEX_HEAD_DIM=index_head_dim,
        BLOCK_K=block_k,
        BLOCK_D=block_d,
    )
    return partial.sum()


def _triton_selected_index_scores_backward(
    q_index: torch.Tensor,
    weights: torch.Tensor,
    selected_k_index: torch.Tensor,
    topk_indices: torch.Tensor,
    grad_scores: torch.Tensor,
    q_start: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, query_len, topk, index_head_dim = selected_k_index.shape
    index_heads = q_index.size(2)
    block_d = max(32, _next_power_of_2(index_head_dim))
    grad_q = torch.zeros(q_index.shape, device=q_index.device, dtype=torch.float32)
    grad_weights = torch.zeros(weights.shape, device=weights.device, dtype=torch.float32)
    grad_selected_k = torch.empty(
        selected_k_index.shape, device=selected_k_index.device, dtype=torch.float32
    )
    grid = lambda meta: (batch_size, query_len, triton.cdiv(topk, meta["BLOCK_K"]))
    _dsa_selected_index_scores_backward_kernel[grid](
        q_index,
        weights,
        selected_k_index,
        topk_indices,
        grad_scores,
        grad_q,
        grad_weights,
        grad_selected_k,
        q_start,
        query_len,
        topk,
        *q_index.stride(),
        *weights.stride(),
        *selected_k_index.stride(),
        *topk_indices.stride(),
        *grad_scores.stride(),
        *grad_q.stride(),
        *grad_weights.stride(),
        *grad_selected_k.stride(),
        INDEX_HEADS=index_heads,
        INDEX_HEAD_DIM=index_head_dim,
        BLOCK_D=block_d,
    )
    return grad_q, grad_weights, grad_selected_k


def _triton_selected_index_scores_backward_dot(
    q_index: torch.Tensor,
    weights: torch.Tensor,
    selected_k_index: torch.Tensor,
    topk_indices: torch.Tensor,
    grad_scores: torch.Tensor,
    q_start: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, query_len, topk, index_head_dim = selected_k_index.shape
    index_heads = q_index.size(2)
    block_h = max(16, _next_power_of_2(index_heads))
    block_d = max(16, _next_power_of_2(index_head_dim))
    grad_q = torch.zeros(q_index.shape, device=q_index.device, dtype=torch.float32)
    grad_weights = torch.zeros(weights.shape, device=weights.device, dtype=torch.float32)
    grad_selected_k = torch.empty(
        selected_k_index.shape, device=selected_k_index.device, dtype=torch.float32
    )
    grid = lambda meta: (batch_size, query_len, triton.cdiv(topk, meta["BLOCK_K"]))
    _dsa_selected_index_scores_backward_dot_kernel[grid](
        q_index,
        weights,
        selected_k_index,
        topk_indices,
        grad_scores,
        grad_q,
        grad_weights,
        grad_selected_k,
        q_start,
        query_len,
        topk,
        *q_index.stride(),
        *weights.stride(),
        *selected_k_index.stride(),
        *topk_indices.stride(),
        *grad_scores.stride(),
        *grad_q.stride(),
        *grad_weights.stride(),
        *grad_selected_k.stride(),
        INDEX_HEADS=index_heads,
        INDEX_HEAD_DIM=index_head_dim,
        BLOCK_H=block_h,
        BLOCK_D=block_d,
    )
    return grad_q, grad_weights, grad_selected_k


class _SelectedIndexScoresTritonFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q_index: torch.Tensor,
        weights: torch.Tensor,
        selected_k_index: torch.Tensor,
        topk_indices: torch.Tensor,
        q_start: int,
    ) -> torch.Tensor:
        ctx.save_for_backward(q_index, weights, selected_k_index, topk_indices)
        ctx.q_start = q_start
        return _triton_selected_index_scores_forward(
            q_index, weights, selected_k_index, topk_indices, q_start
        )

    @staticmethod
    def backward(ctx, grad_scores: torch.Tensor):
        q_index, weights, selected_k_index, topk_indices = ctx.saved_tensors
        grad_q, grad_weights, grad_selected_k = _triton_selected_index_scores_backward(
            q_index,
            weights,
            selected_k_index,
            topk_indices,
            grad_scores.contiguous(),
            ctx.q_start,
        )
        return (
            grad_q.to(dtype=q_index.dtype),
            grad_weights.to(dtype=weights.dtype),
            grad_selected_k.to(dtype=selected_k_index.dtype),
            None,
            None,
        )


def triton_selected_index_scores(
    q_index: torch.Tensor,
    weights: torch.Tensor,
    selected_k_index: torch.Tensor,
    topk_indices: torch.Tensor,
    q_start: int,
) -> Optional[torch.Tensor]:
    if not _can_use_selected_index_scores(q_index, weights, selected_k_index, topk_indices):
        return None
    if torch.is_grad_enabled() and (
        q_index.requires_grad or weights.requires_grad or selected_k_index.requires_grad
    ):
        return _SelectedIndexScoresTritonFn.apply(
            q_index, weights, selected_k_index, topk_indices, q_start
        )
    return _triton_selected_index_scores_forward(
        q_index, weights, selected_k_index, topk_indices, q_start
    )


def triton_selected_index_kl_loss(
    q_index: torch.Tensor,
    weights: torch.Tensor,
    selected_k_index: torch.Tensor,
    topk_indices: torch.Tensor,
    teacher: torch.Tensor,
    loss_scale: float,
    q_start: int,
) -> Optional[torch.Tensor]:
    if _triton_disabled():
        return None
    if not _can_use_selected_index_scores(q_index, weights, selected_k_index, topk_indices):
        return None
    if not _supported_tensor(teacher) or teacher.shape != topk_indices.shape:
        return None
    topk = topk_indices.size(-1)
    index_head_dim = selected_k_index.size(-1)
    block_d = max(32, _next_power_of_2(index_head_dim))
    block_k = max(16, _next_power_of_2(topk))
    if topk > 512 or block_k * block_d > 32768:
        return None
    try:
        return _triton_selected_index_kl_loss(
            q_index,
            weights,
            selected_k_index,
            topk_indices,
            teacher.contiguous(),
            loss_scale,
            q_start,
        )
    except _TRITON_RESOURCE_ERRORS:
        return None


def triton_selected_index_scores_backward(
    q_index: torch.Tensor,
    weights: torch.Tensor,
    selected_k_index: torch.Tensor,
    topk_indices: torch.Tensor,
    grad_scores: torch.Tensor,
    q_start: int,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    if not _can_use_selected_index_scores(q_index, weights, selected_k_index, topk_indices):
        return None
    if not _supported_tensor(grad_scores):
        return None
    try:
        return _triton_selected_index_scores_backward_dot(
            q_index,
            weights,
            selected_k_index,
            topk_indices,
            grad_scores.contiguous(),
            q_start,
        )
    except _TRITON_RESOURCE_ERRORS:
        pass
    return _triton_selected_index_scores_backward(
        q_index, weights, selected_k_index, topk_indices, grad_scores.contiguous(), q_start
    )


def triton_indexer_loss_grad(
    selected_scores: torch.Tensor,
    teacher: torch.Tensor,
    scale: torch.Tensor,
) -> Optional[torch.Tensor]:
    if _triton_disabled():
        return None
    if not (_supported_tensor(selected_scores) and _supported_tensor(teacher)):
        return None
    if not (scale.is_cuda and scale.numel() == 1):
        return None
    batch_size, query_len, topk = selected_scores.shape
    if topk > _MAX_TRITON_SUPPORT_TOPK:
        return None
    block_k = _next_power_of_2(topk)
    grad_scores = torch.empty_like(selected_scores, dtype=torch.float32)
    grid = (batch_size, query_len)
    _dsa_indexer_loss_grad_kernel[grid](
        selected_scores,
        teacher,
        scale,
        grad_scores,
        query_len,
        topk,
        *selected_scores.stride(),
        *teacher.stride(),
        *grad_scores.stride(),
        BLOCK_K=block_k,
    )
    return grad_scores


def triton_sparse_attention_tile(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    topk_indices: torch.Tensor,
    softmax_scale: float,
    q_start: int,
) -> Optional[torch.Tensor]:
    if not _can_use_sparse_attention(query, key, value, topk_indices):
        return None
    return _triton_sparse_attention_forward(query, key, value, topk_indices, softmax_scale, q_start)


def triton_teacher_scores_tile(
    query: torch.Tensor,
    key: torch.Tensor,
    topk_indices: torch.Tensor,
    softmax_scale: float,
    q_start: int,
) -> Optional[torch.Tensor]:
    if not _can_use_sparse_attention(query, key, key, topk_indices):
        return None
    query_len, batch_size, num_heads, head_dim = query.shape
    num_groups = key.size(2)
    repeat_factor = num_heads // num_groups
    block_d = _next_power_of_2(head_dim)
    teacher = torch.zeros(
        (batch_size, query_len, topk_indices.size(-1)),
        device=query.device,
        dtype=torch.float32,
    )
    grid = (query_len, batch_size, num_heads)
    _dsa_teacher_scores_kernel[grid](
        query,
        key,
        topk_indices,
        teacher,
        softmax_scale,
        q_start,
        query_len,
        topk_indices.size(-1),
        repeat_factor,
        *query.stride(),
        *key.stride(),
        *topk_indices.stride(),
        *teacher.stride(),
        HEAD_DIM=head_dim,
        BLOCK_D=block_d,
    )
    return teacher
