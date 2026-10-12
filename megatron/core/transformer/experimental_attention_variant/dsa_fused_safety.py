# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Shared safety diagnostics for cuDNN fused DSA indexer calls."""

from __future__ import annotations

import logging

# Verified defect in the fused indexer kernel package (measured on GB200 with
# cudnn-frontend 1.26.0; no known-good version demonstrated yet): a fused top-k
# call with more than this many query rows is silently corrupted from row 32768
# on unless it is the process's FIRST fused call. The predecessor's shape is
# irrelevant (a bit-identical predecessor also triggers it) and calls at or
# below the limit are immune to process history. See the WORKSPACE NOTE in
# tests/unit_tests/transformer/test_cp_balanced_indexer_layout.py for the
# controlled matrix and reproducer, and cudnn-frontend PR #410 for a candidate
# upstream fix.
FUSED_INDEXER_MAX_SAFE_ROWS = 32768

_ROW_LIMIT_WARNED = False


def warn_fused_indexer_row_limit_once(total_q: int, *, logger: logging.Logger) -> None:
    """Warn once per process before an above-limit fused indexer invocation.

    Existing DSA paths retain their behavior because the defect has only been
    verified on GB200 with cudnn-frontend 1.26.0. New callers that necessarily
    issue multiple above-limit calls may enforce a stricter policy before they
    reach this diagnostic boundary.
    """
    global _ROW_LIMIT_WARNED
    if total_q <= FUSED_INDEXER_MAX_SAFE_ROWS or _ROW_LIMIT_WARNED:
        return
    _ROW_LIMIT_WARNED = True
    logger.warning(
        "CORRECTNESS WARNING: fused indexer top-k call with %d query rows exceeds "
        "%d, the verified-safe limit of the current fused kernel package. On the "
        "verified stack (GB200, cudnn-frontend 1.26.0) such a call following ANY "
        "prior fused call silently returns incorrect top-k indices for rows >= "
        "32768. Upgrade to a backend verified against the reproducer (see the "
        "WORKSPACE NOTE in tests/unit_tests/transformer/"
        "test_cp_balanced_indexer_layout.py and cudnn-frontend PR #410), reduce "
        "rows per call (higher CP degree / smaller pack capacity), or disable the "
        "fused implementation. Proceeding with the fused call.",
        total_q,
        FUSED_INDEXER_MAX_SAFE_ROWS,
    )
