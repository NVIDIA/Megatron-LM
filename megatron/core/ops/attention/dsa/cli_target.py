# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from https://github.com/021ai/ZJ-Megatron-LM at
# a06c53757f6bc636db3a88f5dfca99f5b3ee6030 (BSD-3-Clause, repository LICENSE).

"""Minimal TileLang target accumulation used by full-recompute CLI training."""

import threading
from collections import OrderedDict
from dataclasses import dataclass

import torch

from megatron.core.transformer.experimental_attention_variant.ops.tilelang_utils import (
    HAVE_TILELANG,
    T,
    _get_cached_kernel,
    require_tilelang,
    tilelang,
    tilelang_jit,
)

require_tilelang()

_KERNEL_CACHE = OrderedDict()
_KERNEL_CACHE_LOCK = threading.Lock()
_PASS_CONFIGS = {tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True} if HAVE_TILELANG else {}


@dataclass
class GroupedCLITargetAccumulator:
    """Sufficient statistics for teacher targets in one Top-K reuse group."""

    target_sum: torch.Tensor
    teacher_entropy_sum: torch.Tensor
    reuse_count: int
    topk_indices: torch.Tensor

    def detach(self) -> "GroupedCLITargetAccumulator":
        """Detach persistent no-grad state while preserving reuse metadata."""
        return GroupedCLITargetAccumulator(
            self.target_sum.detach(),
            self.teacher_entropy_sum.detach(),
            self.reuse_count,
            self.topk_indices,
        )


@tilelang_jit(pass_configs=_PASS_CONFIGS)
def _target_accumulation_kernel(  # pragma: no cover
    topk: int, accumulate: bool, block_i: int = 256, threads: int = 256
):
    """Build the normalized teacher-target accumulation kernel."""
    require_tilelang()
    assert topk % block_i == 0

    seq_len = T.dynamic("seq_len")
    accum_dtype = T.float32
    num_tiles = tilelang.cdiv(topk, block_i)
    ln_2 = 0.6931471805599453
    eps = 1.0e-10

    @T.prim_func
    def main(
        Target: T.Tensor([seq_len, topk], accum_dtype),  # type: ignore
        ValidMask: T.Tensor([seq_len, topk], "bool"),  # type: ignore
        TargetSum: T.Tensor([seq_len, topk], accum_dtype),  # type: ignore
        EntropyRows: T.Tensor([seq_len], accum_dtype),  # type: ignore
    ):
        with T.Kernel(seq_len, threads=threads) as row:
            target = T.alloc_fragment([1, block_i], accum_dtype)
            entropy_terms = T.alloc_fragment([1, block_i], accum_dtype)
            entropy_sum = T.alloc_fragment([1], accum_dtype)
            entropy_tile_sum = T.alloc_fragment([1], accum_dtype)
            target_mass = T.alloc_fragment([1], accum_dtype)
            target_tile_sum = T.alloc_fragment([1], accum_dtype)
            valid = T.alloc_fragment([block_i], "bool")
            T.fill(entropy_sum, 0)
            T.fill(target_mass, 0)

            for tile in T.serial(num_tiles):
                for item in T.Parallel(block_i):
                    valid[item] = ValidMask[row, tile * block_i + item]
                    target[0, item] = T.if_then_else(
                        valid[item], Target[row, tile * block_i + item], 0
                    )
                T.reduce_sum(target, target_tile_sum, dim=1, clear=True)
                target_mass[0] += target_tile_sum[0]

            for tile in T.serial(num_tiles):
                for item in T.Parallel(block_i):
                    valid[item] = ValidMask[row, tile * block_i + item]
                    target[0, item] = T.if_then_else(
                        valid[item] and target_mass[0] > 0,
                        Target[row, tile * block_i + item] / target_mass[0],
                        0,
                    )
                    if accumulate:
                        TargetSum[row, tile * block_i + item] += target[0, item]
                    else:
                        TargetSum[row, tile * block_i + item] = target[0, item]
                    entropy_terms[0, item] = T.if_then_else(
                        valid[item] and target[0, item] > 0,
                        target[0, item] * T.log2(T.max(target[0, item], eps)) * ln_2,
                        0,
                    )
                T.reduce_sum(entropy_terms, entropy_tile_sum, dim=1, clear=True)
                entropy_sum[0] += entropy_tile_sum[0]
            EntropyRows[row] = entropy_sum[0]

    return main


def _accumulate_target(
    target: torch.Tensor, valid_mask: torch.Tensor, target_sum: torch.Tensor, *, accumulate: bool
) -> torch.Tensor:
    key = (valid_mask.size(1), accumulate)
    kernel = _get_cached_kernel(
        _KERNEL_CACHE,
        _KERNEL_CACHE_LOCK,
        key,
        lambda: _target_accumulation_kernel(topk=key[0], accumulate=key[1]),
    )
    entropy_rows = torch.empty(target.size(0), dtype=torch.float32, device=target.device)
    kernel(target, valid_mask, target_sum, entropy_rows)
    return entropy_rows.sum()


def accumulate_cli_attention_target(
    attention_target: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_length: torch.Tensor | None,
    query_valid_rows: torch.Tensor | None,
    chunk_rows: int,
    accumulator: GroupedCLITargetAccumulator | None = None,
) -> GroupedCLITargetAccumulator:
    """Normalize and accumulate one reuse-layer teacher target without student work."""
    if (
        not attention_target.is_cuda
        or attention_target.dtype != torch.float32
        or not attention_target.is_contiguous()
        or attention_target.shape != topk_indices.shape
    ):
        raise ValueError(
            "CLI attention target must be contiguous CUDA FP32 and match Top-K indices"
        )
    if (
        topk_indices.ndim != 3
        or topk_indices.device != attention_target.device
        or not topk_indices.is_contiguous()
    ):
        raise ValueError("CLI Top-K indices must be contiguous [B,S,K] on the target device")
    if chunk_rows < 1:
        raise ValueError("CLI chunk_rows must be positive")
    if topk_length is not None and (
        topk_length.shape != topk_indices.shape[:-1]
        or topk_length.device != topk_indices.device
        or topk_length.dtype != torch.int32
    ):
        raise ValueError("CLI Top-K lengths must be int32 [B,S] on the target device")
    if query_valid_rows is not None and (
        query_valid_rows.shape != topk_indices.shape[:-1]
        or query_valid_rows.device != topk_indices.device
        or query_valid_rows.dtype != torch.bool
    ):
        raise ValueError("CLI valid-query mask must be boolean [B,S] on the target device")
    if topk_indices.dtype != torch.int32 or not topk_indices.is_cuda:
        raise ValueError("CLI Top-K indices must be CUDA int32")
    if topk_indices.size(-1) % 256 != 0:
        raise ValueError("CLI Top-K must be divisible by 256")
    if accumulator is None:
        accumulator = GroupedCLITargetAccumulator(
            target_sum=torch.empty_like(attention_target),
            teacher_entropy_sum=torch.zeros(
                (), dtype=torch.float32, device=attention_target.device
            ),
            reuse_count=0,
            topk_indices=topk_indices.detach(),
        )
    elif (
        accumulator.target_sum.shape != attention_target.shape
        or accumulator.target_sum.dtype != torch.float32
        or accumulator.target_sum.device != attention_target.device
        or not accumulator.target_sum.is_contiguous()
        or accumulator.teacher_entropy_sum.dtype != torch.float32
        or accumulator.teacher_entropy_sum.device != attention_target.device
        or accumulator.teacher_entropy_sum.numel() != 1
        or accumulator.reuse_count <= 0
    ):
        raise ValueError("CLI target accumulator is invalid for this reuse group")

    positions = torch.arange(topk_indices.size(-1), device=topk_indices.device)
    chunk_rows = int(chunk_rows)
    initialize = accumulator.reuse_count == 0
    for batch_index in range(topk_indices.size(0)):
        for start in range(0, topk_indices.size(1), chunk_rows):
            end = min(start + chunk_rows, topk_indices.size(1))
            indices = topk_indices[batch_index, start:end]
            valid = indices >= 0
            if topk_length is not None:
                valid &= positions < topk_length[batch_index, start:end].unsqueeze(-1)
            if query_valid_rows is not None:
                valid &= query_valid_rows[batch_index, start:end].unsqueeze(-1)
            entropy = _accumulate_target(
                attention_target[batch_index, start:end],
                valid.contiguous(),
                accumulator.target_sum[batch_index, start:end],
                accumulate=not initialize,
            )
            accumulator.teacher_entropy_sum.add_(entropy)
    accumulator.reuse_count += 1
    return accumulator


__all__ = ["GroupedCLITargetAccumulator", "accumulate_cli_attention_target"]
