# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Indexer top-k selection primitives.

The operand quantizers and row-order kernels shared by every indexer top-k selector: the
selectors must consume byte-identical operands and return rows in a deterministic order. The
query layouts that describe the rows of a selection call, the head counts the selectors of a
layer score with, the tuning policy seam of the selection plan, and the matched-precision
reference selector. Also the configuration and error types of the selectors and of their
external plugins, which Megatron Lite loads from explicit paths only. The per-module bindings
that run the selectors live in ``megatron.lite.primitive.modules.attention.indexer_topk``.
"""

from __future__ import annotations

from megatron.lite.primitive.kernels.indexer_topk.config import (
    ExactTopKConfig,
    IndexerTopKBackend,
    IndexerTopKConfig,
    IndexerTopKConfigError,
    IndexerTopKFormat,
    IndexerTopKPluginError,
    IndexerTopKPrecision,
    IndexerTopKRuntimeError,
    IndexerTopKTuning,
    LiteTopKPluginConfig,
    LiteTopKPluginSettings,
    ResolvedIndexerTopKTuning,
    normalize_indexer_topk_config,
    resolve_indexer_topk_tuning,
)
from megatron.lite.primitive.kernels.indexer_topk.engine import (
    IndexerTopKStats,
    release_indexer_topk_workspaces,
)
from megatron.lite.primitive.kernels.indexer_topk.heads import IndexerHeads
from megatron.lite.primitive.kernels.indexer_topk.layout import (
    IndexerGeometry,
    QueryLayout,
    QuerySegment,
)
from megatron.lite.primitive.kernels.indexer_topk.order import compact_valid_topk_, sort_topk_rows_
from megatron.lite.primitive.kernels.indexer_topk.quant import (
    fold_indexer_weights,
    quantize_indexer_fp8_rows,
)
from megatron.lite.primitive.kernels.indexer_topk.reference import plan_score_rows, reference_topk

__all__ = [
    "ExactTopKConfig",
    "IndexerGeometry",
    "IndexerHeads",
    "IndexerTopKBackend",
    "IndexerTopKConfig",
    "IndexerTopKConfigError",
    "IndexerTopKFormat",
    "IndexerTopKPluginError",
    "IndexerTopKPrecision",
    "IndexerTopKRuntimeError",
    "IndexerTopKStats",
    "IndexerTopKTuning",
    "LiteTopKPluginConfig",
    "LiteTopKPluginSettings",
    "QueryLayout",
    "QuerySegment",
    "ResolvedIndexerTopKTuning",
    "compact_valid_topk_",
    "fold_indexer_weights",
    "normalize_indexer_topk_config",
    "plan_score_rows",
    "quantize_indexer_fp8_rows",
    "reference_topk",
    "release_indexer_topk_workspaces",
    "resolve_indexer_topk_tuning",
    "sort_topk_rows_",
]
