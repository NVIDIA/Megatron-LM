# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

import triton  # type: ignore
import triton.language as tl  # type: ignore

# NOTE: tl.pointer_type() is not available in Triton 3.3.0


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 1024}, num_stages=3, num_warps=32),
        triton.Config({"BLOCK_SIZE_M": 2048}, num_stages=3, num_warps=32),
    ],
    key=["num_tokens"],
)
@triton.jit
def get_num_valid_tokens(
    num_tokens: tl.int64,
    ignore_index: tl.int64,
    labels_ptr,  #: tl.pointer_type(tl.int64),
    stride_labels: tl.int64,
    num_valid_tokens_ptr,  #: tl.pointer_type(tl.int64),
    BLOCK_SIZE_M: tl.constexpr,
):
    """
    Calculate the number of valid tokens in the labels tensor.
    """
    num_pid_m: tl.int64 = tl.cdiv(num_tokens, BLOCK_SIZE_M)

    num_valid_tokens: tl.int64 = tl.zeros((), dtype=tl.int64)
    for m in range(0, num_pid_m):
        offs_am = m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

        labels = tl.load(
            labels_ptr + offs_am * stride_labels, mask=offs_am < num_tokens, other=ignore_index
        )

        valid_labels_mask = labels != ignore_index
        num_valid_tokens += (tl.sum(valid_labels_mask.to(tl.int32), axis=0)).to(tl.int64)
    tl.store(num_valid_tokens_ptr, num_valid_tokens)


@triton.autotune(
    configs=[triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64})],
    key=["num_tokens", "num_splits"],
)
@triton.jit
def forward_dp_epilogue(
    num_tokens: tl.int64,
    num_splits: tl.int64,  # TODO: maybe this could be a constexpr
    ignore_index: tl.int64,
    labels_ptr,  #: tl.pointer_type(tl.int64),
    stride_labels: tl.int64,
    num_valid_tokens_ptr,  #: tl.pointer_type(tl.int64),
    max_ptr,  #: tl.pointer_type(tl.float32),
    stride_max_m: tl.int64,
    stride_max_n: tl.int64,
    accu_ptr,  #: tl.pointer_type(tl.float32),
    stride_accu_m: tl.int64,
    stride_accu_n: tl.int64,
    global_max_ptr,  #: tl.pointer_type(tl.float32),
    stride_global_max: tl.int64,
    global_accu_ptr,  #: tl.pointer_type(tl.float32),
    stride_global_accu: tl.int64,
    global_logprobs_ptr,  #: tl.pointer_type(tl.float32),
    stride_global_logprobs: tl.int64,
    global_logprobs_scalar_ptr,  #: tl.pointer_type(tl.float32),
    REDUCTION: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    forward epilogue in dp
    """
    pid_m = tl.program_id(axis=0)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    global_max = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    global_accu = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

    for pid_n in range(0, tl.cdiv(num_splits, BLOCK_SIZE_N)):
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

        _max = tl.load(
            max_ptr + offs_m[:, None] * stride_max_m + offs_n[None, :] * stride_max_n,
            mask=(offs_m[:, None] < num_tokens) & (offs_n[None, :] < num_splits),
            other=0.0,
        )
        _accu = tl.load(
            accu_ptr + offs_m[:, None] * stride_accu_m + offs_n[None, :] * stride_accu_n,
            mask=(offs_m[:, None] < num_tokens) & (offs_n[None, :] < num_splits),
            other=0.0,
        )

        # local reduction
        _max_old = global_max
        _local_max = tl.max(_max, axis=1, return_indices=False)
        global_max = tl.maximum(global_max, _local_max)

        _scale = tl.exp(_max - global_max[:, None])
        _coeff = tl.exp(_max_old - global_max)
        global_accu = _coeff * global_accu + tl.sum(_scale * _accu, axis=1)

    # store maximum
    tl.store(global_max_ptr + offs_m * stride_global_max, global_max, mask=offs_m < num_tokens)
    # store accumulate
    tl.store(global_accu_ptr + offs_m * stride_global_accu, global_accu, mask=offs_m < num_tokens)
    # update logprobs
    labels = tl.load(
        labels_ptr + offs_m * stride_labels, mask=offs_m < num_tokens, other=ignore_index
    )
    global_logprobs_ptrs = global_logprobs_ptr + offs_m * stride_global_logprobs
    global_logprobs = tl.load(global_logprobs_ptrs, mask=offs_m < num_tokens)
    global_logprobs = global_max + tl.log(global_accu) - global_logprobs
    label_mask = labels != ignore_index
    global_logprobs = tl.where(label_mask, global_logprobs, 0.0)

    if REDUCTION == 0:  # no-reduction
        tl.store(global_logprobs_ptrs, global_logprobs, mask=offs_m < num_tokens)
    elif REDUCTION == 1:  # sum
        global_logprobs_scalar = tl.sum(global_logprobs, axis=0)
        tl.atomic_add(global_logprobs_scalar_ptr, global_logprobs_scalar)
    elif REDUCTION == 2:  # mean
        num_valid_tokens = tl.load(num_valid_tokens_ptr)
        global_logprobs_scalar = tl.fdiv(
            tl.sum(global_logprobs, axis=0), num_valid_tokens.to(tl.float32)
        )
        tl.atomic_add(global_logprobs_scalar_ptr, global_logprobs_scalar)


@triton.autotune(
    configs=[triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64})],
    key=["num_tokens", "num_splits"],
)
@triton.jit
def forward_tp_epilogue(
    num_tokens: tl.int64,
    num_splits: tl.int64,
    reduced_max_ptr,  #: tl.pointer_type(tl.float32),
    stride_reduced_max_m: tl.int64,
    stride_reduced_max_n: tl.int64,
    original_max_ptr,  #: tl.pointer_type(tl.float32),
    stride_original_max_m: tl.int64,
    stride_original_max_n: tl.int64,
    accu_ptr,  #: tl.pointer_type(tl.float32),
    stride_accu_m: tl.int64,
    stride_accu_n: tl.int64,
    global_max_ptr,  #: tl.pointer_type(tl.float32),
    stride_global_max: tl.int64,
    global_accu_ptr,  #: tl.pointer_type(tl.float32),
    stride_global_accu: tl.int64,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    forward epilogue in tp
    """
    pid_m = tl.program_id(axis=0)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

    global_max = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    global_accu = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

    for pid_n in range(0, tl.cdiv(num_splits, BLOCK_SIZE_N)):
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

        _reduced_max = tl.load(
            reduced_max_ptr
            + offs_m[:, None] * stride_reduced_max_m
            + offs_n[None, :] * stride_reduced_max_n,
            mask=(offs_m[:, None] < num_tokens) & (offs_n[None, :] < num_splits),
            other=0.0,
        )
        _original_max = tl.load(
            original_max_ptr
            + offs_m[:, None] * stride_original_max_m
            + offs_n[None, :] * stride_original_max_n,
            mask=(offs_m[:, None] < num_tokens) & (offs_n[None, :] < num_splits),
            other=0.0,
        )
        _accu = tl.load(
            accu_ptr + offs_m[:, None] * stride_accu_m + offs_n[None, :] * stride_accu_n,
            mask=(offs_m[:, None] < num_tokens) & (offs_n[None, :] < num_splits),
            other=0.0,
        )

        # local reduction
        _max_old = global_max
        _local_max = tl.max(_reduced_max, axis=1)
        global_max = tl.maximum(global_max, _local_max)

        # update accumulate
        _coeff = tl.exp(_max_old - global_max)
        _scale = tl.exp(_original_max - global_max[:, None])
        global_accu = _coeff * global_accu + tl.sum(_scale * _accu, axis=1)

    # store
    tl.store(global_max_ptr + offs_m * stride_global_max, global_max, mask=offs_m < num_tokens)
    tl.store(global_accu_ptr + offs_m * stride_global_accu, global_accu, mask=offs_m < num_tokens)


@triton.autotune(configs=[triton.Config({"BLOCK_SIZE_M": 16})], key=["num_tokens"])
@triton.jit
def forward_tp_epilogue_update_logprobs(
    num_tokens: tl.int64,
    ignore_index: tl.int64,
    num_valid_tokens_ptr,  #: tl.pointer_type(tl.int64),
    labels_ptr,  #: tl.pointer_type(tl.int64),
    stride_labels: tl.int64,
    logprobs_ptr,  #: tl.pointer_type(tl.float32),
    stride_logprobs: tl.int64,
    maximum_ptr,  #: tl.pointer_type(tl.float32),
    stride_maximum: tl.int64,
    accumulate_ptr,  #: tl.pointer_type(tl.float32),
    stride_accumulate: tl.int64,
    logprobs_scalar_ptr,  #: tl.pointer_type(tl.float32),
    REDUCTION: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    """
    update logprobs in tp
    """
    pid_m = tl.program_id(axis=0)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

    logprobs = tl.load(logprobs_ptr + offs_m * stride_logprobs, mask=offs_m < num_tokens)
    maximum = tl.load(maximum_ptr + offs_m * stride_maximum, mask=offs_m < num_tokens)
    accumulate = tl.load(accumulate_ptr + offs_m * stride_accumulate, mask=offs_m < num_tokens)

    labels = tl.load(
        labels_ptr + offs_m * stride_labels, mask=offs_m < num_tokens, other=ignore_index
    )
    label_mask = labels != ignore_index

    logprobs = maximum + tl.log(accumulate) - logprobs
    logprobs = tl.where(label_mask, logprobs, 0.0)

    if REDUCTION == 0:  # no-reduction
        tl.store(logprobs_ptr + offs_m * stride_logprobs, logprobs, mask=offs_m < num_tokens)
    elif REDUCTION == 1:  # sum
        logprobs_scalar = tl.sum(logprobs, axis=0)
        tl.atomic_add(logprobs_scalar_ptr, logprobs_scalar)
    elif REDUCTION == 2:  # mean
        num_valid_tokens = tl.load(num_valid_tokens_ptr)
        logprobs_scalar = tl.fdiv(tl.sum(logprobs, axis=0), num_valid_tokens.to(tl.float32))
        tl.atomic_add(logprobs_scalar_ptr, logprobs_scalar)
