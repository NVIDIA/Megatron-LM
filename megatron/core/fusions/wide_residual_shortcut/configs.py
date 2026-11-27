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


# Nano/Super/Ultra (rows, hidden): the shapes the tuned tables and tests anchor on.
FLAGSHIPS = ((16384, 3072), (8192, 4096), (8192, 10240))


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


def _bf16_config(*fields, vector):
    return _ForwardConfig(*fields, LOADS_FIRST=True, VECTOR=vector)


# GB300 inference configs for a BF16 residual stream at 512-1024 rows (3 streams), which needs
# more rows in flight. Key: (hidden, readers, updates, MXFP8 handoff).
# fmt: off
_FORWARD_CONFIGS_BF16 = {
    (3072, 1, 1, False): _bf16_config(4096, 1, 0, False, 0, 8, None, True, vector=4),
    (3072, 1, 1, True): _bf16_config(4096, 2, 0, False, 0, 8, 80, False, vector=4),
    (3072, 2, 1, False): _bf16_config(4096, 2, 0, True, 0, 8, 80, False, vector=4),
    (3072, 2, 1, True): _bf16_config(2048, 0, 0, False, 0, 4, 168, True, vector=8),
    (3072, 2, 2, False): _bf16_config(4096, 4, 0, False, 0, 8, None, True, vector=4),
    (3072, 2, 2, True): _bf16_config(4096, 1, 0, True, 0, 8, None, True, vector=4),
    (4096, 1, 1, False): _bf16_config(2048, 0, 0, False, 0, 4, 128, False, vector=8),
    (4096, 1, 1, True): _bf16_config(2048, 0, 0, False, 0, 4, 128, False, vector=8),
    (4096, 2, 1, False): _bf16_config(4096, 2, 0, True, 0, 8, 80, False, vector=4),
    (4096, 2, 1, True): _bf16_config(4096, 2, 0, True, 0, 8, 80, False, vector=8),
    (4096, 2, 2, False): _bf16_config(2048, 0, 0, True, 0, 4, 128, False, vector=4),
    (4096, 2, 2, True): _bf16_config(2048, 0, 0, False, 0, 4, 168, True, vector=8),
    (10240, 1, 1, False): _bf16_config(4096, 2, 0, False, 0, 8, 128, True, vector=8),
    (10240, 1, 1, True): _bf16_config(4096, 2, 0, False, 0, 8, 128, True, vector=8),
    (10240, 2, 1, False): _bf16_config(4096, 1, 1, False, 0, 8, 128, False, vector=4),
    (10240, 2, 1, True): _bf16_config(2048, 2, 1, True, 0, 8, 128, False, vector=8),
    (10240, 2, 2, False): _bf16_config(2048, 2, 1, True, 0, 8, 128, False, vector=8),
    (10240, 2, 2, True): _bf16_config(2048, 2, 1, True, 0, 8, 128, False, vector=4),
}
# fmt: on

# Batch-invariant inference uses this row count's schedule at every row count, so a row's
# arithmetic does not depend on the batch (a decode-cluster schedule, MXFP8-aligned).
_BATCH_INVARIANT_ROWS = 32


def _inference_leaf(m, h, streams, readers, mxfp8):
    """Inference decision tree: ``(leading config fields, CTAs per row)``.

    Fit to ~10K timed GB300 configs (hidden 3072-16384, 1-6 streams, 1/2 readers, BF16/MXFP8,
    rows 32-1024) minimizing the mean slowdown against the best of 16 recipes x CTAs per row.
    """
    if h <= 5632:
        if m <= 88:
            if h <= 3584:
                return (4096, 2, 0, True, 0, 8, 128, True), 2
            return (4096, 1, 0, False, 0, 16, 128, True), 2
        if h <= 3584:
            if streams <= 3:
                return (4096, 1, 0, True, 0, 8, None, True), 1
            return (4096, 2, 0, True, 0, 8, 128, True), 1
        if readers == 1:
            if streams <= 4:
                return (4096, 4, 1, False, 0, 16, 64, True), 1
            return (4096, 2, 0, True, 0, 8, 128, True), 1
        if streams <= 4:
            return (2048, 2, 0, True, 0, 16, None, False), 1
        return (2048, 2, 1, True, 0, 8, 128, False), 1
    if m <= 40:
        return (4096, 4, 0, False, 0, 16, 128, True), 4
    if m <= 72:
        if streams <= 4:
            return (4096, 2, 1, False, 0, 16, 128, True), 2
        return (4096, 1, 0, True, 0, 8, None, True), 2
    if h <= 9728:
        if m <= 128:
            return (4096, 4, 0, False, 0, 16, 128, True), 1
        return (2048, 2, 1, True, 0, 8, 128, False), 1
    if readers == 1:
        return (4096, 1, 0, False, 0, 16, 128, True), 1
    if not mxfp8:
        return (4096, 4, 0, False, 0, 16, 128, True), 1
    return (4096, 2, 0, True, 0, 16, None, True), 1


def _replicated(cfg, replicas, readers, h):
    """Spread ``replicas`` CTAs per row over split readers and column parts."""
    split = replicas > 1 and readers == 2
    # Whole 32-column groups keep both data and MXFP8 scale ownership exhaustive.
    parts = min(replicas // (1 + split), (h // 32) & -(h // 32)) if h % 32 == 0 else 1
    return cfg._replace(PARTS=parts, SPLIT=split)


def _inference_config(m, h, streams, readers, mxfp8, updates):
    if h * streams >= 8192 and m <= (64 if readers == 1 else 32):
        # Decode rows: a cluster per row (per reader with two readers), one slice per CTA.
        warps = 4 if readers == 1 else 8
        for columns in (512, 1024, 2048) if h <= 4096 else (2048, 1024):
            if h % columns == 0 and h // columns <= 8 and columns >= 128 * warps:
                split = readers == 2
                return _ForwardConfig(
                    columns, 0, 0, False, 0, warps, None, True, 1, split, True, h // columns
                )
    fields, replicas = _inference_leaf(m, h, streams, readers, mxfp8)
    if readers == 2 and updates == 1:
        # One update read twice: regions where the shortcut add's leaves measured slower.
        if mxfp8 and 3584 < h <= 5632 and m > 88 and (streams == 3 or (streams == 4 and h > 4096)):
            fields, replicas = (4096, 2, 0, True, 0, 16, None, True), 1
        elif h == 10240 and 160 < m <= 256:
            fields, replicas = (2048, 2, 1, True, 0, 8, 128, False), 1
    return _replicated(_ForwardConfig(*fields, LOADS_FIRST=True), replicas, readers, h)


def _prefill_config(h, streams, readers, bf16_residual, outputs):
    """Prefill decision tree (inference rows > 1024, hidden 2048-16384, 1-6 streams).

    Prefill writes no tape for the backward, so it gets its own tree, fit like
    ``_training_config`` to a cold-L2 GB300 sweep of 584 inference cells (hidden 2048-16384,
    1-6 streams, 1.5K-16K rows, both residual dtypes, one and two readers with and without a
    shortcut add, every output format) over 32 schedules. Mean slowdown 1.3% against the per-cell
    best (the tables it replaced: 6.8%).
    """
    row_bytes = streams * h * (2 if bf16_residual else 4)
    vector = 4
    if row_bytes <= 24576:
        if bf16_residual:
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
    """Choose the row-kernel schedule; ``updates`` None means one per reader and ``outputs`` (the
    readers' formats, ``OUTPUT_FORMATS``) None means BF16 readers."""
    outputs = ("bf16",) * readers if outputs is None else tuple(outputs)
    if 2048 <= h <= 16384 and not inference and m >= 2048:
        cfg = _training_config(h, streams, readers, bf16_residual)
    elif 2048 <= h <= 16384 and inference and m > 1024:
        cfg = _prefill_config(h, streams, readers, bf16_residual, outputs)
    else:
        # Otherwise schedules depend only on MXFP8 presence (reader zero's, where tuned).
        mxfp8 = any(f != "bf16" for f in outputs)
        cfg = _choose(m, h, streams, readers, mxfp8, inference, updates, bf16_residual)
    # Pairing the two readers' reductions needs two readers.
    return cfg._replace(PAIRED_REDUCE=False) if cfg.PAIRED_REDUCE and readers == 1 else cfg


def _choose(m, h, streams, readers, mxfp8, inference, updates, bf16_residual):
    """Decode, short training rows and untuned widths (training and prefill rows otherwise take
    ``_training_config`` and ``_prefill_config``)."""
    tuned = 3072 <= h <= 16384
    if inference and m >= 512 and bf16_residual and streams == 3:
        cfg = _FORWARD_CONFIGS_BF16.get((h, readers, updates or readers, mxfp8))
        if cfg is not None:
            return cfg
    if inference and m <= 1024:
        if tuned:
            return _inference_config(m, h, streams, readers, mxfp8, updates)
        if h <= 5120:
            profile = "row" if readers == 1 else "paired"
        elif h <= 13312:
            profile = "streamed" if mxfp8 else "pipelined"
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
