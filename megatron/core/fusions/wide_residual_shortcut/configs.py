# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Measured heuristics and configs of the fused seams: the qualified device, the admitted shapes,
and each kernel's schedule selection. Plain Python (no GPU
imports), so the op module can consult it eagerly."""

from typing import NamedTuple

import torch


def _cdiv(a, b):
    return -(-a // b)


def _next_power_of_2(x):
    return 1 << (x - 1).bit_length()


# ---------------------------------------------------------------- device and shapes

SMS, SM_SHARED_BYTES = 152, 232_000  # the qualified GB300


def qualified_device(properties):
    """The measured GB300 residency is not portable to every SM103 device."""
    return (
        "GB300" in properties.name
        and properties.major == 10
        and properties.minor == 3
        and properties.multi_processor_count == 152
    )


# ---------------------------------------------------------------- forward


class _ForwardConfig(NamedTuple):
    """Row-kernel schedule: ceil(H / WIDTH) segments of VECTOR columns per thread.

    Reader emits trail the norm by PIPELINE_DEPTH segments, the first OFFLOAD_SEGMENTS wait in
    shared memory, REDUCE_VECTOR is the sum-of-squares layout (0: the mixing one). PARTS CTAs
    replicate a row and each stores a slice; SPLIT gives each of two readers a CTA; CLUSTER CTAs
    each own a column slice and exchange partial sums (sm_90+, needs LOADS_FIRST, PARTS == 1 and
    WIDTH dividing H / CLUSTER). LOADS_FIRST issues the row's loads before any store.
    """

    WIDTH: int = 128
    PIPELINE_DEPTH: int = 0
    OFFLOAD_SEGMENTS: int = 0
    PAIRED_REDUCE: bool = False
    REDUCE_VECTOR: int = 16
    NUM_WARPS: int = 4
    MAXNREG: int | None = None
    STORE_LATE: bool = False
    PARTS: int = 1
    SPLIT: bool = False
    LOADS_FIRST: bool = False
    CLUSTER: int = 1
    VECTOR: int = 4


# Batch-invariant inference uses this row count's schedule at every row count, so a row's
# arithmetic does not depend on the batch (a decode-cluster schedule, MXFP8-aligned).
_BATCH_INVARIANT_ROWS = 32


def _replicated(cfg, replicas, readers, h):
    """Spread ``replicas`` CTAs per row over split readers and column parts."""
    split = replicas > 1 and readers == 2
    # Whole 32-column groups keep both data and MXFP8 scale ownership exhaustive.
    parts = min(replicas // (1 + split), (h // 32) & -(h // 32)) if h % 32 == 0 else 1
    return cfg._replace(PARTS=parts, SPLIT=split)


def _inference_config(m, h, streams, readers, bf16_residual, outputs):
    """GB300 inference tuned over 1–8192 rows, H=2048–16384 and 1–6 residual streams.

    Small batches can use multiple CTAs per row. Once the grid fills, row traffic and reader
    formats determine the schedule. Above 1024 rows all tuned widths use the prefill rule.
    """
    if m <= 128 or (m <= 1024 and h > 12288):
        cfg = _ForwardConfig(4096, 4, 0, False, 0, 16, 128, True, LOADS_FIRST=True)
        if m > 128:
            if readers == 2 and any(fmt != "bf16" for fmt in outputs):
                return cfg._replace(PIPELINE_DEPTH=2, PAIRED_REDUCE=True, MAXNREG=None)
            return cfg
        if readers == 2 and 32 <= m <= 64:
            # Without a suitable cluster, dedicate a CTA to each reader of a short batch.
            cfg = cfg._replace(PIPELINE_DEPTH=1, SPLIT=True)
        if m * readers <= 192:
            # Each cluster reduces one row per reader. Limit the total grid to 512 CTAs.
            warps = 4 * readers
            for width in (512, 1024, 2048) if h <= 4096 else (2048, 1024):
                ctas, tail = divmod(h, width)
                if not tail and 1 < ctas <= 8 and width >= 128 * warps:
                    if m * readers * ctas > 512:
                        return cfg
                    return _ForwardConfig(
                        width, 0, 0, False, 0, warps, None, True, 1, readers == 2, True, ctas
                    )
            if m <= 64 and not cfg.SPLIT:
                # Replicate a row when the grid is small and its width cannot tile a cluster.
                return _replicated(cfg, 4 if m <= 32 else 2, readers, h)
        return cfg
    row_bytes = streams * h * (2 if bf16_residual else 4)
    if m <= 1024 and 4096 < h <= 9216 and (m <= 256 or readers == 2 or row_bytes > 98304):
        return _ForwardConfig(2048, 2, 1, readers == 2, 0, 8, 128, LOADS_FIRST=True)

    vector = 4
    if row_bytes <= 24576:
        if bf16_residual and m >= 512:
            fields, vector = (1024, 2, 0, False, 0, 4, None, False), 8
        else:
            fields = (4096, 1, 0, True, 0, 8, None, True)
    elif row_bytes <= 73728:
        if h <= 6144:
            if row_bytes <= 36864:
                fields = (1024, 4, 1, True, 16, 8, 80, True)
            else:
                fields = (2048, 2, 1, True, 0, 8, 128, False)
        elif readers == 1:
            fields, vector = (2048, 2, 1, True, 0, 8, 128, False), 8
        else:
            fields, vector = (2048, 2, 2, True, 0, 8, 128, False), 8
    elif h > 12288:
        fields = (4096, 2, 0, True, 0, 16, None, True)
    elif row_bytes <= 98304:
        fields = (2048, 2, 1, True, 0, 8, 128, False)
    elif "both" in outputs:
        fields = (4096, 1, 0, False, 0, 16, 128, True)
    else:
        fields = (4096, 4, 0, False, 0, 16, 128, True)
    return _ForwardConfig(*fields, LOADS_FIRST=True, VECTOR=vector)


def _training_config(h, streams, readers, bf16_residual):
    """Training decision tree (rows >= 2048, hidden 2048-16384, 1-6 streams).

    Fit to a cold-L2 GB300 sweep of 578 cells (hidden 2048-16384, 1-6 streams, 2K-16K rows, both
    residual dtypes, one and two readers with and without a shortcut add, every output format):
    each leaf holds the schedule with the lowest mean slowdown against its cells' best. The
    residual row's bytes decide first; no leaf depends on the readers' output formats. Mean
    slowdown 1.3% against the per-cell best of the sweep's schedules.
    """
    row_bytes = streams * h * (2 if bf16_residual else 4)
    vector = 4
    if row_bytes <= 12288:
        fields, vector = (1024, 2, 0, False, 0, 4, None, False), 8
    elif row_bytes <= 24576:
        if not bf16_residual:
            fields = (1024, 4, 0, False, 4, 8, 64, False)
        elif readers == 1:
            fields, vector = (1024, 2, 0, False, 0, 4, None, False), 8
        elif h <= 4096:
            fields, vector = (2048, 4, 1, False, 0, 4, None, False), 8
        else:
            fields, vector = (2048, 2, 1, True, 0, 8, 128, False), 8
    elif row_bytes <= 73728:
        if h <= 4096:
            fields = (1024, 4, 1, True, 16, 8, 80, True)
        elif bf16_residual:
            fields, vector = (2048, 2, 2, True, 0, 8, 128, False), 8
        else:
            fields = (2048, 2, 1, False, 0, 8, 128, False)
    elif row_bytes <= 98304:
        fields = (2048, 2, 1, True, 0, 8, 128, False)
    else:
        fields = (4096, 4, 0, False, 0, 16, 128, True)
    return _ForwardConfig(*fields, LOADS_FIRST=True, VECTOR=vector)


def choose_forward_config(
    m, h, streams, readers=1, *, outputs=None, inference=False, updates=None, bf16_residual=False
):
    """Choose the row-kernel schedule. Readers default to BF16. The update count can be
    supplied by callers benchmarking the seam and does not affect the schedule."""
    outputs = ("bf16",) * readers if outputs is None else tuple(outputs)
    if 2048 <= h <= 16384 and not inference and m >= 2048:
        cfg = _training_config(h, streams, readers, bf16_residual)
    elif 2048 <= h <= 16384 and inference:
        cfg = _inference_config(m, h, streams, readers, bf16_residual, outputs)
    else:
        cfg = _choose(m, h, streams, readers, inference)
    # Pairing the two readers' reductions needs two readers.
    return cfg._replace(PAIRED_REDUCE=False) if cfg.PAIRED_REDUCE and readers == 1 else cfg


def _choose(m, h, streams, readers, inference):
    """Fallback for short training batches and untuned widths."""
    if inference and m <= 1024:
        if h < 3072:
            profile = "row" if readers == 1 else "paired"
        else:
            profile = "pipelined" if readers == 1 else "wide"
    else:
        if h * streams <= 10240:
            profile = "compact" if h * streams <= 5120 else "paired"
        elif h <= 13312:
            profile = "pipelined" if m <= 128 else "streamed"
        else:
            profile = "streamed" if readers == 1 else "wide"
    # Untuned fallback profiles.
    width = max(32, _next_power_of_2(h))
    tile = min(width, 4096)
    cfg = {
        "compact": lambda: _ForwardConfig(tile, REDUCE_VECTOR=0),
        "row": lambda: _ForwardConfig(width, 0, 0, False, 16, 8, 128, True),
        "paired": lambda: _ForwardConfig(tile, 2, 0, readers == 2, 0, 8, 80),
        "pipelined": lambda: _ForwardConfig(tile, 4, 1, readers == 2, 0, 16, 64, True),
        "streamed": lambda: _ForwardConfig(tile, 2, 1, False, 0, 8, 128, True),
        "wide": lambda: _ForwardConfig(min(width, 8192), 0, 0, readers == 2, 0, 16),
    }[profile]()
    replicas = min(8, 128 // _next_power_of_2(m)) if inference and m <= 64 else 1
    return _replicated(cfg, replicas, readers, h)


# ---------------------------------------------------------------- backward


class _BackwardConfig(NamedTuple):
    """Launch grid (whole width-spanning CTA groups), register cap, and the shared bytes a CTA
    may add without lowering how many share an SM."""

    TARGET_CTAS: int
    MAXNREG: int
    SPARE_SHARED: int


def get_backward_knobs(m, h, streams, readers=1, *, grid="auto", bf16=False, second=False):
    """Select the launch knobs for one shape; an explicit grid overrides the auto grid.

    Each CTA stages one row tile in shared memory (the predecessor streams, the update and,
    below six streams, every reader gradient), and the grid keeps as many CTAs per SM as that
    allows, at most four. Four leave 128 registers each, enough only for one reader or one
    stream; three take 168 and fewer take them all. From an 8K-row sweep of widths 2048-10240,
    1-6 streams, both residual dtypes and all reader variants.
    """
    staged = 2 * (readers + second) if streams < 6 else 0
    shared = 4096 * ((2 if bf16 else 4) * streams + staged + 2) + 3000
    per_sm = min(SM_SHARED_BYTES // shared, 4 if readers == 1 or streams == 1 else 3)
    if grid == "auto":
        participants = _cdiv(h, 256)
        grid = min(_cdiv(m, 16), max(1, per_sm * SMS // participants)) * participants
    maxnreg = {4: 128, 3: 168}.get(per_sm, 255)
    return _BackwardConfig(grid, maxnreg, SM_SHARED_BYTES // per_sm - shared)


def choose_backward_layout(spare, h, readers, *, bf16, mxfp8, paired_shares):
    """``(stash_shared, tma_store, paired)`` for a launch with ``spare`` shared bytes per CTA.

    MXFP8 du stashes an even row tile (8 KB per CTA) in shared memory where that keeps residency,
    else in global memory. BF16 residuals whose extra shared bytes (beside the stash) keep
    residency: two readers send dR and any BF16 du through an 8 KB ring and TMA bulk stores
    (whole 256-column tiles); one reader pairs a tile's column halves, half zero leaving its
    controller shares (``paired_shares`` bytes) for half one (FP32 measured ~1% slower).
    """
    stash_shared = mxfp8 and spare >= 8192
    ring = 8192 if readers == 2 else paired_shares
    resident = bf16 and ring + 8192 * stash_shared <= spare
    return stash_shared, resident and readers == 2 and h % 256 == 0, resident and readers == 1


# ---------------------------------------------------------------- replay

# Column replay schedules per residual dtype: (VEC columns per thread, THREADS lanes across a
# 32-row tile, WARPS warps down it, MAXNREG registers, AHEAD transitions of update prefetch),
# from an 8K-token Nano/Super/Ultra sweep.
_MX_SCHEDULES = {torch.float32: (4, 8, 2, 128, 1), torch.bfloat16: (4, 8, 2, 128, 1)}
