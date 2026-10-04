# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Indexer top-k selection primitives.

The operand quantizers and row-order kernels shared by every indexer top-k selector: the
selectors must consume byte-identical operands and return rows in a deterministic order.
"""

from __future__ import annotations

from megatron.lite.primitive.kernels.indexer_topk.order import compact_valid_topk_, sort_topk_rows_
from megatron.lite.primitive.kernels.indexer_topk.quant import (
    fold_indexer_weights,
    quantize_indexer_fp8_rows,
)

__all__ = [
    "compact_valid_topk_",
    "fold_indexer_weights",
    "quantize_indexer_fp8_rows",
    "sort_topk_rows_",
]
