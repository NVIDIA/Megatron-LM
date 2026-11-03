# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Forward shortcut scheduling and matched folded residual readers.

Read the config selector and host entry points first, then the arithmetic stages
and _forward. Mixing, coefficient preparation, and reader stores live here;
kernel_utils supplies shared primitives. Inference skips the saved state needed by backward.
"""

from functools import lru_cache
from typing import NamedTuple

import torch
import triton
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import BlockedLayout, SwizzledSharedLayout
from triton.language.extra.cuda import gdc_wait

from .kernel_utils import (
    SavedWrite,
    _emit_mxfp8,
    _tensor,
    allocate_operand,
    operand_buffers,
    validate_num_streams,
)


class _WriteArgs(NamedTuple):
    """Tensors the residual writer reads and writes.

    Attributes:
        residual: Previous FP32 or BF16 residual, ``[M, STREAMS * H]``.
        update: First BF16 branch update, ``[M, H]``.
        other_update: Second BF16 branch update of a shortcut add, or None for a single update.
        output: Current residual written by the kernel in the residual's dtype,
            ``[M, STREAMS * H]``.
        summed_update: BF16 sum of a shortcut add's updates saved for backward, or None.
    """

    residual: object
    update: object
    other_update: object
    output: object
    summed_update: object


class _ReadArgs(NamedTuple):
    """Per-reader normalization inputs and outputs.

    Attributes:
        gammas: BF16 RMSNorm weights, one ``[H]`` tensor per reader.
        rstd: FP32 per-row reciprocal RMS saved for backward per reader, or Nones.
        outputs: Reader output data (BF16, or MXFP8 rowwise bytes) per reader.
        scales: MXFP8 rowwise E8M0 scales per reader, or None for BF16 readers.
    """

    gammas: tuple
    rstd: tuple
    outputs: tuple
    scales: tuple


class _ControllerArgs(NamedTuple):
    """FP32 or BF16 controller logits, one entry per stream.

    Attributes:
        write: Write-gate logits.
        retain: Retention logits, or None for fixed identity retention.
        read_a: Reader-0 mixing logits.
        read_b: Reader-1 mixing logits (reader 0's again with one reader).
    """

    write: object
    retain: object
    read_a: object
    read_b: object


class _Geometry(NamedTuple):
    """Compile-time problem shape.

    Attributes:
        H: Hidden width.
        STREAMS: Number of residual streams.
        HAS_RETENTION: Whether retention coefficients are loaded at runtime.
    """

    H: int
    STREAMS: int
    HAS_RETENTION: bool


class _ForwardConfig(NamedTuple):
    """Compile-time schedule of the row forward kernel.

    Attributes:
        WIDTH: Columns per segment (a power of two); a row is processed in ceil(H / WIDTH)
            segments.
        PIPELINE_DEPTH: Segments by which reader emits trail their normalization.
        OFFLOAD_SEGMENTS: Leading segments whose mixed values wait in shared memory instead of
            registers until the norm is known.
        PAIRED_REDUCE: Reduce the two readers' sums of squares together.
        REDUCE_VECTOR: Per-thread width of the final sum-of-squares layout (0 keeps the
            mixing layout).
        NUM_WARPS: Warps per CTA.
        MAXNREG: Register cap per thread, or None.
        STORE_LATE: Store each stream's residual after its reader mixing instead of before.
        PARTS: CTAs per row (replicas); each recomputes the row's norm and stores its slice.
        SPLIT: One CTA per reader (two readers only).
        LOADS_FIRST: Issue every segment's update and residual loads before any mix or store.
        CLUSTER: CTAs per row in an sm_90+ cluster; each mixes H / CLUSTER columns and the
            cluster exchanges partial sums of squares (one cluster per reader with SPLIT).
            Needs LOADS_FIRST, PARTS == 1 and WIDTH dividing H / CLUSTER.
        VECTOR: Contiguous columns per thread (4, or 8 for 16-byte BF16 residual accesses);
            WIDTH must give every thread at least VECTOR columns.
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


# GB300 training/prefill configs for the production widths (3 streams, rows >= 2048): one
# config per shape, chosen from a paired GB300 sweep at 2K/4K/8K/16K rows (1.06x over the
# previous presets and profiles; Ultra two-reader 1.11-1.16x).
# Key: (hidden size, readers, MXFP8 handoff).
# Stores use the default cache policy.
# fmt: off
_FORWARD_CONFIGS = {
    (3072, 1, False): _ForwardConfig(4096, 0, 1, False, 1, 16, 64, False),
    (3072, 1, True): _ForwardConfig(4096, 2, 0, False, 0, 8, 128, True, LOADS_FIRST=True),
    (3072, 2, False): _ForwardConfig(2048, 2, 1, True, 0, 8, 128, False, LOADS_FIRST=True),
    (3072, 2, True): _ForwardConfig(2048, 2, 1, True, 0, 8, 128, False, LOADS_FIRST=True),
    (4096, 1, False): _ForwardConfig(4096, 1, 1, False, 0, 8, 128, False),
    (4096, 1, True): _ForwardConfig(4096, 4, 1, False, 0, 16, 64, True, LOADS_FIRST=True),
    (4096, 2, False): _ForwardConfig(2048, 2, 1, True, 0, 8, 128, False, LOADS_FIRST=True),
    (4096, 2, True): _ForwardConfig(2048, 2, 1, True, 0, 8, 128, False, LOADS_FIRST=True),
    (10240, 1, False): _ForwardConfig(4096, 1, 0, False, 0, 16, 128, True, LOADS_FIRST=True),
    (10240, 1, True): _ForwardConfig(4096, 2, 1, False, 0, 16, 128, True, LOADS_FIRST=True),
    (10240, 2, False): _ForwardConfig(4096, 1, 0, False, 0, 16, 128, True, LOADS_FIRST=True),
    (10240, 2, True): _ForwardConfig(4096, 4, 0, False, 0, 16, 128, True, LOADS_FIRST=True),
}


# fmt: on

# GB300 configs for a BF16 residual stream at the production widths (3 streams), which moves
# half the residual bytes and needs more rows in flight than the FP32 schedules keep. Chosen
# per shape from a sweep of the 16 recipes x LOADS_FIRST x VECTOR (4 or 8 BF16 columns per
# thread) at 8K rows, confirmed at 4K/16K: 1.04-1.64x over the FP32 table on BF16 streams.
# Key: (hidden size, readers, updates, MXFP8 handoff); one update read twice is its own shape.
# Training/prefill from 2048 rows and inference from 512 rows use them.
# fmt: off
_FORWARD_CONFIGS_BF16 = {
    (3072, 1, 1, False): _ForwardConfig(4096, 1, 0, False, 0, 8, None, True, LOADS_FIRST=True, VECTOR=4),
    (3072, 1, 1, True): _ForwardConfig(4096, 2, 0, False, 0, 8, 80, False, LOADS_FIRST=True, VECTOR=4),
    (3072, 2, 1, False): _ForwardConfig(4096, 2, 0, True, 0, 8, 80, False, LOADS_FIRST=True, VECTOR=4),
    (3072, 2, 1, True): _ForwardConfig(2048, 0, 0, False, 0, 4, 168, True, LOADS_FIRST=True, VECTOR=8),
    (3072, 2, 2, False): _ForwardConfig(4096, 4, 0, False, 0, 8, None, True, LOADS_FIRST=True, VECTOR=4),
    (3072, 2, 2, True): _ForwardConfig(4096, 1, 0, True, 0, 8, None, True, LOADS_FIRST=True, VECTOR=4),
    (4096, 1, 1, False): _ForwardConfig(2048, 0, 0, False, 0, 4, 128, False, LOADS_FIRST=True, VECTOR=8),
    (4096, 1, 1, True): _ForwardConfig(2048, 0, 0, False, 0, 4, 128, False, LOADS_FIRST=True, VECTOR=8),
    (4096, 2, 1, False): _ForwardConfig(4096, 2, 0, True, 0, 8, 80, False, LOADS_FIRST=True, VECTOR=4),
    (4096, 2, 1, True): _ForwardConfig(4096, 2, 0, True, 0, 8, 80, False, LOADS_FIRST=True, VECTOR=8),
    (4096, 2, 2, False): _ForwardConfig(2048, 0, 0, True, 0, 4, 128, False, LOADS_FIRST=True, VECTOR=4),
    (4096, 2, 2, True): _ForwardConfig(2048, 0, 0, False, 0, 4, 168, True, LOADS_FIRST=True, VECTOR=8),
    (10240, 1, 1, False): _ForwardConfig(4096, 2, 0, False, 0, 8, 128, True, LOADS_FIRST=True, VECTOR=8),
    (10240, 1, 1, True): _ForwardConfig(4096, 2, 0, False, 0, 8, 128, True, LOADS_FIRST=True, VECTOR=8),
    (10240, 2, 1, False): _ForwardConfig(4096, 1, 1, False, 0, 8, 128, False, LOADS_FIRST=True, VECTOR=4),
    (10240, 2, 1, True): _ForwardConfig(2048, 2, 1, True, 0, 8, 128, False, LOADS_FIRST=True, VECTOR=8),
    (10240, 2, 2, False): _ForwardConfig(2048, 2, 1, True, 0, 8, 128, False, LOADS_FIRST=True, VECTOR=8),
    (10240, 2, 2, True): _ForwardConfig(2048, 2, 1, True, 0, 8, 128, False, LOADS_FIRST=True, VECTOR=4),
}
# fmt: on

# BF16 residual streams of other shapes from 2048 rows: the single most robust config of the
# same sweep (mean 1.13x of the per-shape best vs 1.23x for the FP32 tree's leaves).
_BF16_STREAM_CONFIG = _ForwardConfig(4096, 1, 1, False, 0, 8, 128, False, LOADS_FIRST=True)


def _inference_leaf(m, h, streams, readers, mxfp8):
    """Return the inference decision tree's leaf for a row-kernel launch.

    16 leaves fit to about 10K timed GB300 configs: 123 shapes (hidden 3072-16384, 1-6
    streams, 1/2 readers, BF16/MXFP8) at rows 32-1024 plus Nano/Super/Ultra every 16-64 rows,
    cold inputs and distinct outputs per launch, minimizing the mean per-row slowdown against
    the best of 16 recipes x CTAs per row. On 20 held-out shapes it is 1.21x faster in total
    than the previous heuristic and within 5% of the per-row best.

    Args:
        m: Rows.
        h: Hidden width.
        streams: Residual streams.
        readers: Reader count (1 or 2).
        mxfp8: Whether reader 0 is handed off as MXFP8.

    Returns:
        ``(fields, replicas)``: the leading ``_ForwardConfig`` fields (WIDTH through
        STORE_LATE) and the CTAs per row.
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


def _single_write_leaf(m, h, streams, mxfp8):
    """Return the inference leaf for one update read twice where it differs from the tree's.

    Elsewhere the two-update (shortcut add) leaves serve one update as well. Fit to a GB300
    sweep of the 16 recipes for one update and two readers (Nano/Super/Ultra, BF16/MXFP8, rows
    32-1024), where these regions were up to 1.21x slower on the shortcut add's leaves, then
    restricted to the widths and stream counts where no point of a 240-point check (hidden
    4096-16384, 1-6 streams, the leaf vs the shortcut add's) was slower: 35 points at
    1.05-1.30x (MXFP8, 3584 < H <= 5632) and 20 at 1.00-1.47x (H = 10240, 161-256 rows).
    Wider rows spill the second leaf's registers, and widths with a partial 2048-column
    segment (9984) lost up to 15%. On 28 further points (hidden 3840/4864/10240, 3-5
    streams) the leaves were 1.00-1.34x faster.

    Args:
        m: Rows.
        h: Hidden width.
        streams: Residual streams.
        mxfp8: Whether reader 0 is handed off as MXFP8.

    Returns:
        The leading ``_ForwardConfig`` fields, or None to use ``_inference_leaf``.
    """
    if mxfp8 and 3584 < h <= 5632 and m > 88 and (streams == 3 or (streams == 4 and h > 4096)):
        return (4096, 2, 0, True, 0, 16, None, True)
    if h == 10240 and 160 < m <= 256:
        return (2048, 2, 1, True, 0, 8, 128, False)
    return None


def _inference_config(m, h, streams, readers, mxfp8, updates=None):
    """Return the LOADS_FIRST row config for an inference launch.

    Decode rows (<= 64 with one reader, <= 32 with two) of residuals with at least 8K columns use a cluster per row
    (per reader with two readers) with one 512-2048 column slice per CTA: fit to a GB300 sweep of cluster
    size x slice x warps at rows 1-64, 1.24x over the tree on 480 decode points (hidden
    3072-16384, 1-6 streams, 1/2 readers, BF16/MXFP8). Longer rows use ``_inference_leaf``.

    Args:
        m: Rows.
        h: Hidden width.
        streams: Residual streams.
        readers: Reader count (1 or 2).
        mxfp8: Whether reader 0 is handed off as MXFP8.
        updates: Branch updates written (2 for a shortcut add); None means one per reader.

    Returns:
        The ``_ForwardConfig`` to launch.
    """
    if h * streams >= 8192 and m <= (64 if readers == 1 else 32):
        # Decode rows: a cluster per row (per reader with two readers), one column slice per CTA.
        warps = 4 if readers == 1 else 8
        for columns in (512, 1024, 2048) if h <= 4096 else (2048, 1024):
            if h % columns == 0 and h // columns <= 8 and columns >= 128 * warps:
                split = readers == 2
                return _ForwardConfig(
                    columns, 0, 0, False, 0, warps, None, True, 1, split, True, h // columns
                )
    fields, replicas = _inference_leaf(m, h, streams, readers, mxfp8)
    if readers == 2 and updates == 1:
        single = _single_write_leaf(m, h, streams, mxfp8)
        fields, replicas = (single, 1) if single is not None else (fields, replicas)
    split = replicas > 1 and readers == 2
    # Whole 32-column groups keep both data and MXFP8 scale ownership exhaustive.
    parts = min(replicas // (1 + split), (h // 32) & -(h // 32)) if h % 32 == 0 else 1
    cfg = _ForwardConfig(*fields, parts, split, True)
    return cfg._replace(
        PAIRED_REDUCE=cfg.PAIRED_REDUCE and readers == 2
    )  # pairing needs two readers


def _training_leaf(h, streams, readers, mxfp8):
    """Return the training decision tree's leaf for rows >= 2048 outside ``_FORWARD_CONFIGS``.

    6 leaves fit to a GB300 training-mode sweep of 107 shapes (hidden 3072-16384, 1-6 streams,
    1/2 readers, BF16/MXFP8) at 2K-16K rows, minimizing the mean per-row slowdown against the
    best of 17 recipes x LOADS_FIRST and the profile tree. Shape-wise 5-fold cross-validation:
    1.14x faster in total than the profile tree, within 4% of the per-row best. The best config
    did not depend on the row count, so the tree has no row split.

    Args:
        h: Hidden width.
        streams: Residual streams.
        readers: Reader count (1 or 2).
        mxfp8: Whether reader 0 is handed off as MXFP8.

    Returns:
        ``(fields, loads_first)``: the leading ``_ForwardConfig`` fields (WIDTH through
        STORE_LATE) and LOADS_FIRST.
    """
    if h <= 9728:
        if streams <= 1:
            return (4096, 2, 0, True, 0, 8, 80, False), True
        return (2048, 2, 1, True, 0, 8, 128, False), True
    if h <= 15872:
        if streams <= 1:
            return (2048, 2, 1, True, 0, 8, 128, False), True
        return (4096, 2, 1, False, 0, 16, 128, True), True
    if readers == 1:
        return (4096, 4, 0, False, 0, 16, 128, True), True
    return (4096, 2, 0, True, 0, 16, None, True), True


def _training_config(h, streams, readers, mxfp8):
    """Return the row config for a training or long-prefill launch from ``_training_leaf``.

    Args:
        h: Hidden width.
        streams: Residual streams.
        readers: Reader count (1 or 2).
        mxfp8: Whether reader 0 is handed off as MXFP8.

    Returns:
        The ``_ForwardConfig`` to launch, one CTA per row.
    """
    fields, loads_first = _training_leaf(h, streams, readers, mxfp8)
    cfg = _ForwardConfig(*fields, LOADS_FIRST=loads_first)
    return cfg._replace(
        PAIRED_REDUCE=cfg.PAIRED_REDUCE and readers == 2
    )  # pairing needs two readers


@lru_cache
def _programmatic_launch(device):
    """Return whether a device supports PDL launches and CTA clusters (sm_90+).

    Args:
        device: CUDA device.

    Returns:
        True on compute capability 9.0 and newer; older devices keep an ordinary launch.
    """
    return torch.cuda.get_device_capability(device)[0] >= 9


def choose_forward_config(
    m, h, streams, readers=1, *, mxfp8=False, inference=False, updates=None, bf16_residual=False
):
    """Choose the GB300 row forward kernel schedule.

    Inference rows up to 1024 (hidden 3072-16384, up to 6 streams) use ``_inference_config``;
    one update read twice refines a few of its leaves (``_single_write_leaf``). Training and
    long prefill use ``_FORWARD_CONFIGS`` for Nano/Super/Ultra and ``_training_config`` for
    other covered shapes from 2048 rows; two readers share these configs between one and two
    updates (one update was within 1.4% of the best of 32 recipes at Nano/Super/Ultra
    2K-16K rows). Everything else falls back
    to profile heuristics fitted from decision trees over a 13K-trial config sweep.

    Args:
        m: Rows.
        h: Hidden width.
        streams: Residual streams.
        readers: Reader count (1 or 2).
        mxfp8: Whether reader 0 is handed off as MXFP8.
        inference: Whether the launch skips backward state.
        updates: Branch updates written (2 for a shortcut add); None means one per reader.
        bf16_residual: Whether the residual stream is BF16 (FP32 otherwise).

    Returns:
        The ``_ForwardConfig`` to launch.
    """
    if bf16_residual:
        key = (h, readers, updates or readers, mxfp8)
        if streams == 3 and key in _FORWARD_CONFIGS_BF16 and m >= (512 if inference else 2048):
            return _FORWARD_CONFIGS_BF16[key]
        if not inference and m >= 2048 and 3072 <= h <= 16384 and streams <= 6:
            cfg = _BF16_STREAM_CONFIG
            return cfg._replace(PAIRED_REDUCE=cfg.PAIRED_REDUCE and readers == 2)
    if inference and m <= 1024 and 3072 <= h <= 16384 and streams <= 6:
        return _inference_config(m, h, streams, readers, mxfp8, updates)
    width = max(32, triton.next_power_of_2(h))
    tile = min(width, 4096)
    if inference and m <= 1024:
        if h <= 5120:
            profile = "row" if readers == 1 else "paired"
        elif h <= 13312:
            profile = "streamed" if mxfp8 else "pipelined"
        else:
            profile = "pipelined" if readers == 1 else "wide"
    else:
        # Training and long prefill: tuned production-width configs, else the profile tree.
        if streams == 3 and m >= 2048 and (h, readers, mxfp8) in _FORWARD_CONFIGS:
            return _FORWARD_CONFIGS[h, readers, mxfp8]
        if m >= 2048 and 3072 <= h <= 16384 and streams <= 6:
            return _training_config(h, streams, readers, mxfp8)
        if h * streams <= 10240:
            profile = "compact" if h * streams <= 5125 else "paired"
        elif h <= 13312:
            profile = "pipelined" if m <= 128 else "streamed"
        else:
            profile = "streamed" if readers == 1 else "wide"
    if profile == "compact":
        cfg = _ForwardConfig(tile, REDUCE_VECTOR=0)
    elif profile == "row":
        cfg = _ForwardConfig(width, 0, 0, False, 16, 8, 128, True)
    elif profile == "paired":
        cfg = _ForwardConfig(tile, 2, 0, readers == 2, 0, 8, 80)
    elif profile == "pipelined":
        cfg = _ForwardConfig(tile, 4, 1, readers == 2, 0, 16, 64, True)
    elif profile == "streamed":
        cfg = _ForwardConfig(tile, 2, 1, False, 0, 8, 128, True)
    else:
        cfg = _ForwardConfig(min(width, 8192), 0, 0, readers == 2, 0, 16)
    replicas = min(8, 128 // triton.next_power_of_2(m)) if inference and m <= 64 else 1
    split = replicas > 1 and readers == 2
    # Whole 32-column groups keep both data and MXFP8 scale ownership exhaustive.
    parts = min(replicas // (1 + split), (h // 32) & -(h // 32)) if h % 32 == 0 else 1
    return cfg._replace(PARTS=parts, SPLIT=split)


@torch.no_grad()
def forward(
    residual,
    updates,
    write,
    retain,
    reads,
    gammas,
    *,
    mxfp8_handoff=False,
    num_streams=3,
    epsilon=1.0e-5,
    inference=False,
):
    """Write the current residual and the folded RMSNorm reader operands.

    One or two readers (``reads``/``gammas``) consume the write of one update, or of a
    shortcut add of two updates. Inference skips backward state
    (coefficients, the shortcut add's BF16 update sum and reader statistics), returning
    saved=None. A shortcut add rounds its BF16 update sum before writing the residual.
    The residual is FP32 or BF16 (the current residual keeps its dtype) and controller logits
    are FP32 or BF16; every value is widened to FP32 for the arithmetic, a BF16 residual is
    rounded only when stored, and readers fold the unrounded FP32 values.
    Operands belong to forward consumers, not the tape. The exclusive MXFP8
    handoff quantizes reader zero; every other reader returns BF16.
    Widths divisible by four use packet loads and require a 16-byte-aligned FP32 residual base.

    retain=None means fixed identity retention (t=1), not zero logits (t=0.95).
    This static mode removes retention arithmetic and its parameter gradient.
    A supplied tensor always keeps the learned path, regardless of its values.

    Args:
        residual: Previous FP32 or BF16 residual, ``[M, num_streams * H]``.
        updates: One BF16 branch update, or two summed by a shortcut add, each ``[M, H]``.
        write: FP32 or BF16 write-gate logits.
        retain: FP32 or BF16 retention logits, or None for fixed identity retention.
        reads: FP32 or BF16 mixing logits, one tensor per reader.
        gammas: BF16 RMSNorm weights, one ``[H]`` tensor per reader.
        mxfp8_handoff: Quantize reader 0 to rowwise MXFP8.
        num_streams: Residual streams.
        epsilon: RMSNorm epsilon.
        inference: Skip backward state and use the inference schedules.

    Returns:
        ``(current residual, saved write, consumer operands)``; saved write is None in
        inference.

    Raises:
        ValueError: If the inputs' counts, shapes, devices or MXFP8 alignment are invalid.
    """
    validate_num_streams(num_streams)
    n = len(reads)
    if n not in (1, 2) or len(gammas) != n or len(updates) not in (1, 2):
        raise ValueError("one or two readers write one update or a shortcut add of two")
    shortcut_add = len(updates) == 2
    if not residual.is_cuda or updates[0].ndim != 2:
        raise ValueError("expected CUDA [M,H] updates")
    m, h = updates[0].shape
    if m <= 0 or h <= 0:
        raise ValueError("forward requires positive rows and width")
    device = residual.device
    if residual.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("the residual must be FP32 or BF16")
    _tensor(residual, (m, num_streams * h), residual.dtype, device, "residual")
    for u in updates:
        _tensor(u, (m, h), torch.bfloat16, device, "update")
    for controller in (write, *((retain,) if retain is not None else ()), *reads):
        if controller.ndim != 1 or controller.numel() < num_streams:
            raise ValueError(f"controllers require at least {num_streams} entries")
        if controller.dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("controller logits must be FP32 or BF16")
        _tensor(controller, controller.shape, controller.dtype, device, "controller")
    for gamma in gammas:
        _tensor(gamma, (h,), torch.bfloat16, device, "gamma")
    config = choose_forward_config(
        m,
        h,
        num_streams,
        n,
        mxfp8=mxfp8_handoff,
        inference=inference,
        updates=len(updates),
        bf16_residual=residual.dtype == torch.bfloat16,
    )
    c = None if inference else torch.empty((8 * num_streams,), device=device, dtype=torch.float32)

    packets = h % 4 == 0 and residual.dtype == torch.float32
    if packets and type(residual).__name__ not in ("FakeTensor", "FunctionalTensor"):
        assert residual.data_ptr() % 16 == 0, "residual base must be 16-byte aligned"
    if mxfp8_handoff and (m % 32 or h % 32):
        raise ValueError("MXFP8 handoff requires rows and hidden width divisible by 32")

    output_bf16 = (False, True)[:n] if mxfp8_handoff else (True,) * n
    current = torch.empty_like(residual)
    summed = torch.empty_like(updates[0]) if shortcut_add and not inference else None
    stats = tuple(
        None if inference else torch.empty(m, dtype=torch.float32, device=device) for _ in range(n)
    )
    qs = []
    for output_is_bf16 in output_bf16:
        q = allocate_operand((m, h), device, output_is_bf16)
        # Inference fills padded rows in-kernel; partial scale columns still need clearing.
        if not output_is_bf16 and (h % 128 or (m % 128 and not inference)):
            q._rowwise_scale_inv.zero_()
        qs.append(q)

    saved = (
        None
        if inference
        else SavedWrite(
            summed if summed is not None else updates[0].detach(),
            c,
            tuple(gamma.detach() for gamma in gammas),
            stats,
            tuple(0 if v is None else v.numel() for v in (write, retain, *reads)),
            num_streams,
            retain is not None,
            len(updates),
        )
    )
    # A runtime unit retention coefficient preserves Ultra register occupancy.
    # Short rows duplicate reductions for occupancy; stores have one owner.
    dependent = inference and _programmatic_launch(device)
    grid = (m, 2 if config.SPLIT else 1, config.PARTS)
    write_args = _WriteArgs(
        residual, updates[0], updates[1] if shortcut_add else None, current, summed
    )
    kwargs = dict(
        read_tensors=_ReadArgs(gammas, stats, *operand_buffers(qs, columnwise=False)),
        GEOMETRY=_Geometry(h, num_streams, retain is not None or h == 10240),
        EPSILON=epsilon,
        controllers=_ControllerArgs(write, retain, reads[0], reads[-1]),
        saved_coefficients=c,
        PAD_ROWS=inference and mxfp8_handoff and bool(m % 128),
        DEPENDENT=dependent,
        enable_fp_fusion=False,
        # Inference launches as a programmatic dependent: the grid may start while the
        # previous kernel drains, then waits for it (griddepcontrol.wait) before any access.
        launch_pdl=dependent,
    )
    if config.CLUSTER > 1:
        cluster = dict(
            CONFIG=config,
            EXCHANGE=_exchange_asm(config.CLUSTER, 1 if config.SPLIT else n),
            num_warps=config.NUM_WARPS,
            maxnreg=config.MAXNREG,
        )
        if _programmatic_launch(device) and _as_cluster(
            write_args, grid, config.CLUSTER, kwargs | cluster
        ):
            _forward[grid](write_args, **kwargs, **cluster)
            return current, saved, tuple(qs)
        config = config._replace(CLUSTER=1)  # same slices, one CTA per row
    _forward[grid](
        write_args, CONFIG=config, num_warps=config.NUM_WARPS, maxnreg=config.MAXNREG, **kwargs
    )
    return current, saved, tuple(qs)


def _as_cluster(write_args, grid, ctas, kwargs):
    """Make ``_forward``'s compiled kernel for these arguments launch as clusters of CTAs.

    ``_forward`` compiles as a single-CTA kernel; Triton's launcher runs gridX * num_ctas CTAs
    in clusters of num_ctas along x, taking num_ctas from the packed (warps, ctas, shared)
    metadata. An exchange without its cluster peers would never complete, so an unexpected
    metadata form makes the caller launch without a cluster.

    Args:
        write_args: ``_WriteArgs`` of the launch.
        grid: Launch grid; x is the row count.
        ctas: CTAs per cluster.
        kwargs: The launch's remaining keyword arguments.

    Returns:
        True if the kernel will launch as clusters of ``ctas`` CTAs, False otherwise.
    """
    kernel = _forward.warmup(write_args, grid=grid, **kwargs)
    meta = kernel.packed_metadata
    if not (isinstance(meta, tuple) and len(meta) == 3 and meta[1] in (1, ctas)):
        return False
    kernel.packed_metadata = (meta[0], ctas, meta[2])
    return True


def _exchange_asm(ctas, partials):
    """Return the PTX for a cluster's partial-sum exchange (sm_90+).

    Gluon (Triton 3.7.1) cannot address a peer CTA's shared memory, so the exchange is PTX
    with its own static shared barrier and buffer. Receive buffer: ``xbuf[k * ctas + rank]``
    holds reader k's partial from CTA ``rank``.

    Args:
        ctas: CTAs per cluster.
        partials: Partial sums per CTA (readers normalized in the CTA).

    Returns:
        ``(init, exchange)``. ``init`` runs once the loads are in flight. ``exchange`` maps
        this CTA's per-reader partial sums to their sums over the cluster, added in rank order
        so every CTA computes identical bits.
    """

    def ptx(*lines):
        """Join ``(instruction, comment)`` pairs into one commented PTX line each."""
        return "\n".join(f"{code:<64} // {why}" for code, why in lines)

    init = ptx(
        (".shared .align 8 .b64 xbar;", "barrier: completes when every peer's partials arrive"),
        (
            f".shared .align 16 .f32 xbuf[{ctas * partials}];",
            "receive buffer, one slot per reader/rank",
        ),
        ("{", "scope the registers below to this block"),
        (".reg .pred lead;", "thread 0 sets up the barrier for the whole CTA"),
        (".reg .u32 tid, mbar;", "thread index; shared address of xbar"),
        ("mov.u32 tid, %tid.x;", "read this thread's index in the CTA"),
        ("setp.eq.u32 lead, tid, 0;", "lead = thread 0"),
        ("mov.u32 mbar, xbar;", "shared-window address of this CTA's barrier"),
        ("@lead mbarrier.init.shared::cta.b64 [mbar], 1;", "one arrival (the next line) per phase"),
        (
            "@lead mbarrier.arrive.expect_tx.shared::cta.b64 _, [mbar], "
            f"{(ctas - 1) * partials * 4};",
            "arrive, and expect this many bytes from the peers",
        ),
        ("fence.mbarrier_init.release.cluster;", "publish the initialized barrier to peer CTAs"),
        ("barrier.cluster.arrive.relaxed.aligned;", "signal ready; the wait is in exchange"),
        ("mov.u32 $0, 0;", "dummy result: inline_asm_elementwise needs an output"),
        ("}", "end of scope"),
    )
    lines = [
        ("{", "scope the registers below to this block"),
        (".reg .pred lead, send, own, done;", "thread 0; sends to a peer; own rank; barrier done"),
        (
            ".reg .u32 tid, rank, buf, mbar, slot, peer_mbar, peer_slot;",
            "indices and shared addresses",
        ),
        (".reg .f32 v;", "one partial while summing"),
        ("mov.u32 tid, %tid.x;", "read this thread's index in the CTA"),
        ("mov.u32 rank, %cluster_ctarank;", "this CTA's rank in the cluster (its column slice)"),
        ("setp.eq.u32 lead, tid, 0;", "lead = thread 0, the only sender"),
        ("mov.u32 buf, xbuf;", "shared-window address of the receive buffer"),
        ("mov.u32 mbar, xbar;", "shared-window address of the barrier"),
        (
            "mad.lo.u32 slot, rank, 4, buf;",
            "&xbuf[rank]: where peers keep this CTA's reader-0 partial",
        ),
        ("barrier.cluster.wait.aligned;", "wait until every peer has initialized its barrier"),
    ]
    for peer in range(ctas):
        lines += [
            (
                f"setp.ne.and.u32 send, rank, {peer}, lead;",
                f"send = thread 0 and rank {peer} is a peer",
            ),
            (
                f"@send mapa.shared::cluster.u32 peer_mbar, mbar, {peer};",
                f"rank {peer}'s barrier address",
            ),
            (
                f"@send mapa.shared::cluster.u32 peer_slot, slot, {peer};",
                f"rank {peer}'s xbuf[rank]",
            ),
        ]
        for k in range(partials):
            if k:
                lines.append(
                    (
                        f"@send add.u32 peer_slot, peer_slot, {ctas * 4};",
                        f"advance to reader {k}'s slots",
                    )
                )
            lines.append(
                (
                    "@send st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 "
                    f"[peer_slot], ${partials + k}, [peer_mbar];",
                    f"store reader {k}'s partial; count its 4 bytes on rank {peer}'s barrier",
                )
            )
    lines += [
        ("WAIT_${:uid}:", "retry point (unique label per inlined copy)"),
        (
            "mbarrier.try_wait.parity.shared::cta.b64 done, [mbar], 0;",
            "done once all peers' bytes arrived",
        ),
        ("@!done bra WAIT_${:uid};", "not yet: try again"),
    ]
    for k in range(partials):
        for peer in range(ctas):
            lines += [
                (
                    f"setp.eq.u32 own, rank, {peer};",
                    f"own = rank {peer} is this CTA (its partial is in a register)",
                ),
                (f"@own mov.f32 v, ${partials + k};", f"this CTA's reader-{k} partial"),
                (
                    f"@!own ld.shared.f32 v, [buf + {(k * ctas + peer) * 4}];",
                    f"rank {peer}'s reader-{k} partial",
                ),
                (
                    f"add.rn.f32 ${k}, ${k}, v;" if peer else f"mov.f32 ${k}, v;",
                    "sum in rank order: identical bits in every CTA",
                ),
            ]
    lines.append(("}", "end of scope"))
    return init, ptx(*lines)


# Forward arithmetic and store ownership.


@g.jit
def _load_coefficient(C, INDEX: gl.constexpr):
    """Return one entry of the coefficient table as a scalar.

    Args:
        C: Coefficient table from ``_prepare_coefficients``.
        INDEX: Table index.

    Returns:
        The coefficient.
    """
    indices = gl.full(
        (1,), INDEX, gl.int32, gl.constexpr(BlockedLayout([1], [32], [gl.num_warps()], [0]))
    )
    return gl.sum(C.gather(indices, 0), 0)


@g.jit
def _load_residual(write_tensors, row, stream, cols, GEOMETRY: gl.constexpr, CONFIG: gl.constexpr):
    """Load one stream's residual fragment (FP32 in 4-wide packets when H allows, or BF16).

    Args:
        write_tensors: ``_WriteArgs``.
        row: Row index.
        stream: Stream index.
        cols: Column indices of the fragment.
        GEOMETRY: ``_Geometry``.
        CONFIG: ``_ForwardConfig``.

    Returns:
        The residual values, zero outside the row.
    """
    H: gl.constexpr = GEOMETRY.H
    NUM_STREAMS: gl.constexpr = GEOMETRY.STREAMS
    mask = cols < H
    residual = write_tensors.residual + row * (NUM_STREAMS * H) + stream * H + cols

    # Explicitly vector-load four contiguous FP32 values with inline PTX when H allows it.
    if write_tensors.residual.dtype.element_ty != gl.float32:
        # Stays packed BF16 (two per register) until _mix_fragment widens it at first use.
        old = gl.load(residual, mask, 0)
    elif H % 4 == 0:
        if H % CONFIG.WIDTH:
            # H and tile widths are multiples of four: each packet is wholly valid or masked.
            old = gl.inline_asm_elementwise(
                """
                { // Begin masked packet-load register scope.
                    .reg .pred p; // Declare the packet validity predicate.
                    setp.ne.u32 p, $8, 0; // Check whether the four-value packet is valid.
                    mov.b32 $0, 0; // Zero the first value for a masked packet.
                    mov.b32 $1, 0; // Zero the second value for a masked packet.
                    mov.b32 $2, 0; // Zero the third value for a masked packet.
                    mov.b32 $3, 0; // Zero the fourth value for a masked packet.
                    @p ld.global.nc.v4.f32 {$0,$1,$2,$3},[$4]; // Load four valid FP32 values.
                } // End masked packet-load register scope.
                """,
                "=f,=f,=f,=f,l,l,l,l,r,r,r,r",
                (residual.to(gl.uint64), mask.to(gl.int32)),
                dtype=gl.float32,
                is_pure=False,
                pack=4,
            )
        else:
            old = gl.inline_asm_elementwise(
                """
                { // Begin packet-load scope.
                    ld.global.nc.v4.f32 {$0,$1,$2,$3},[$4]; // Load four contiguous FP32 values.
                } // End packet-load scope.
                """,
                "=f,=f,=f,=f,l,l,l,l",
                (residual.to(gl.uint64),),
                dtype=gl.float32,
                is_pure=False,
                pack=4,
            )
    else:
        old = gl.load(residual, mask, 0)
    return old


@g.jit
def _load_fragment(write_tensors, row, cols, GEOMETRY: gl.constexpr, CONFIG: gl.constexpr):
    """Issue one fragment's update loads, plus every stream's residual load with LOADS_FIRST.

    Args:
        write_tensors: ``_WriteArgs``.
        row: Row index.
        cols: Column indices of the fragment.
        GEOMETRY: ``_Geometry``.
        CONFIG: ``_ForwardConfig``.

    Returns:
        ``(update, other, olds)``: the BF16 updates (``other`` repeats ``update`` without a
        shortcut add) and, with LOADS_FIRST, each stream's residual values (else an empty
        tuple).
    """
    H: gl.constexpr = GEOMETRY.H
    mask = cols < H
    update = gl.load(write_tensors.update + (row * H + cols), mask, 0)
    other = update  # unused without a shortcut add
    if write_tensors.other_update is not None:
        other = gl.load(write_tensors.other_update + (row * H + cols), mask, 0)
    NUM_STREAMS: gl.constexpr = GEOMETRY.STREAMS
    olds = ()
    if CONFIG.LOADS_FIRST:
        for stream in gl.static_range(NUM_STREAMS):
            olds += (_load_residual(write_tensors, row, stream, cols, GEOMETRY, CONFIG),)
    return update, other, olds


@g.jit
def _mix_fragment(
    write_tensors,
    coefficients,
    row,
    cols,
    GEOMETRY: gl.constexpr,
    TWO_READERS: gl.constexpr,
    CONFIG: gl.constexpr,
    OWNER,
    LOADED=None,
):
    """Write one fragment's residual streams and return its folded reader inputs.

    Preserves the native writer's FMA/RN contraction pattern, which checkpoint replay
    reproduces bit for bit.

    Args:
        write_tensors: ``_WriteArgs``.
        coefficients: Coefficient table from ``_prepare_coefficients``.
        row: Row index.
        cols: Column indices of the fragment.
        GEOMETRY: ``_Geometry``.
        TWO_READERS: Whether there is a second reader.
        CONFIG: ``_ForwardConfig``.
        OWNER: Mask of the columns this CTA stores.
        LOADED: Loads already issued by ``_load_fragment``, or None to load here.

    Returns:
        The two readers' pre-norm inputs (the second is zero with one reader), zero outside the row.
    """
    H: gl.constexpr = GEOMETRY.H
    NUM_STREAMS: gl.constexpr = GEOMETRY.STREAMS
    HAS_RETENTION: gl.constexpr = GEOMETRY.HAS_RETENTION
    mask = cols < H
    if LOADED is None:
        LOADED = _load_fragment(write_tensors, row, cols, GEOMETRY, CONFIG)
    update, other, olds = LOADED
    update = update.to(gl.float32)
    if write_tensors.other_update is not None:  # shortcut add
        other = other.to(gl.float32)
        update = (update + other).to(gl.bfloat16).to(gl.float32)
        if write_tensors.summed_update is not None:
            gl.store(write_tensors.summed_update + row * H + cols, update, mask)
    x0 = gl.full(mask.shape, 0, gl.float32, mask.type.layout)
    x1 = gl.full(mask.shape, 0, gl.float32, mask.type.layout)
    update_mix0 = 0.0
    update_mix1 = 0.0
    for stream in gl.static_range(NUM_STREAMS):
        owner = OWNER & (not CONFIG.SPLIT or gl.program_id(1) == (stream == NUM_STREAMS - 1))
        if CONFIG.LOADS_FIRST:
            old = olds[stream]
        else:
            old = _load_residual(write_tensors, row, stream, cols, GEOMETRY, CONFIG)
        old = old.to(gl.float32)  # exact for BF16; a no-op for FP32

        write = _load_coefficient(coefficients, stream)
        retain = _load_coefficient(coefficients, NUM_STREAMS + stream) if HAS_RETENTION else 1.0

        separate = write * update + retain * old
        # Preserve the native writer's row/stream contraction pattern;
        # checkpoint replay uses exactly these RN/FMA boundaries.
        contracted = (stream == 0) | ((stream == 1) & (row % 32 < 12))
        value = gl.where(contracted, gl.fma(write, update, retain * old), separate)

        stored = value.to(write_tensors.output.dtype.element_ty)  # RN for a BF16 residual
        if not CONFIG.STORE_LATE:
            gl.store(
                write_tensors.output + row * (NUM_STREAMS * H) + stream * H + cols,
                stored,
                mask & owner,
            )
        beta0 = _load_coefficient(coefficients, 2 * NUM_STREAMS + stream)
        x0 = gl.fma(beta0 * retain, old, x0)
        update_mix0 = gl.fma(beta0, write, update_mix0)
        if TWO_READERS:
            beta1 = _load_coefficient(coefficients, 3 * NUM_STREAMS + stream)
            x1 = gl.fma(beta1 * retain, old, x1)
            update_mix1 = gl.fma(beta1, write, update_mix1)
        if CONFIG.STORE_LATE:
            gl.store(
                write_tensors.output + row * (NUM_STREAMS * H) + stream * H + cols,
                stored,
                mask & owner,
            )
    x0 = gl.fma(update_mix0, update, x0)
    if TWO_READERS:
        x1 = gl.fma(update_mix1, update, x1)
    return gl.where(mask, x0, 0.0), gl.where(mask, x1, 0.0)


@g.jit
def _owns_columns(cols, H: gl.constexpr, partition: gl.constexpr):
    """Return which columns this CTA stores when a row spans several CTAs.

    Args:
        cols: Column (or 32-column group) indices.
        H: Width of the indexed axis.
        partition: ``(PARTS, SPLIT, CLUSTER)`` of the launch.

    Returns:
        A mask of owned indices, or True when one CTA owns the whole row.
    """
    parts: gl.constexpr = partition[0]
    split: gl.constexpr = partition[1]
    if partition[2] > 1:
        owned = cols // (H // partition[2]) == gl.program_id(0) % partition[2]
    elif split and parts == 2:
        owned = (cols < H // 2) == (gl.program_id(2) == 0)
    elif parts == 1:
        owned = True
    else:
        owned = cols // (H // parts) == gl.program_id(2)
    return owned


@g.jit
def _emit(x, OPERANDS, indices, rstd, H: gl.constexpr, partition: gl.constexpr, rows):
    """Normalize one reader fragment and store it as BF16 or rowwise MXFP8.

    Args:
        x: Pre-norm reader input fragment.
        OPERANDS: ``(weight, output, scales)``; scales is None for BF16 outputs.
        indices: ``(row, cols)`` of the fragment.
        rstd: Reciprocal RMS of the row.
        H: Hidden width.
        partition: ``(PARTS, SPLIT, CLUSTER)`` of the launch.
        rows: Row count for padding MXFP8 scales of the last 128-row tile, or None.
    """
    weight, output, scales = OPERANDS
    row, cols = indices
    mask = (cols < H) & _owns_columns(cols, H, partition)
    y = (x * rstd) * weight
    # None specializes this branch at compile time; there is no runtime format check.
    if scales is None:
        gl.store(output + row * H + cols, y.to(gl.bfloat16), mask)
    else:
        grouped = gl.reshape(y, (x.numel // 32, 32))
        group = gl.min(gl.reshape(cols, (x.numel // 32, 32)), 1) // 32
        _emit_mxfp8(
            grouped,
            (output + row * H + cols, mask),
            (scales, (group < H // 32) & _owns_columns(group, H // 32, partition)),
            (row, group, H),
            rows,
        )


@g.jit
def _prefetch_l2(pointers, cols, H: gl.constexpr):
    """Prefetch the 128-byte lines under ``pointers`` (columns ``cols`` < H) into L2."""
    LINE: gl.constexpr = 1024 // pointers.dtype.element_ty.primitive_bitwidth  # elements
    first = (cols < H) & (cols % LINE == 0)  # one prefetch per line
    gl.inline_asm_elementwise(
        """
        { // Begin predicated L2 prefetch scope.
            .reg .pred p; // Declare the line-leader predicate.
            setp.ne.u32 p, $2, 0; // Prefetch only from the first column of each line.
            @p prefetch.global.L2 [$1]; // Hint the line into L2; no data is returned.
            mov.u32 $0, 0; // Dummy result keeps the side effect in program order.
        } // End predicated L2 prefetch scope.
        """,
        "=r,l,r",
        [pointers.to(gl.uint64), first.to(gl.int32)],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )


@g.jit
def _prepare_coefficients(
    controllers, C, row, TWO_READERS: gl.constexpr, STREAMS: gl.constexpr, RETENTION: gl.constexpr
):
    """Compute the write, retain and read coefficients from the controller logits.

    Row zero also saves the coefficients and their sigmoid derivative terms for backward.

    Args:
        controllers: ``_ControllerArgs``.
        C: FP32 save buffer of ``8 * STREAMS`` entries, or None in inference.
        row: Row index (only row zero saves).
        TWO_READERS: Whether there is a second reader.
        STREAMS: Residual streams.
        RETENTION: Whether retention logits are supplied.

    Returns:
        The coefficient table: write, retain, read_a and read_b per stream.
    """
    L: gl.constexpr = BlockedLayout([1], [32], [gl.num_warps()], [0])
    i = gl.arange(0, triton.next_power_of_2(4 * STREAMS), layout=L)
    controller, stream = i // STREAMS, i % STREAMS
    # One masked load per controller: each may be FP32 or BF16 and is widened exactly.
    sources = (
        controllers.write,
        controllers.retain if RETENTION else controllers.write,
        controllers.read_a,
        controllers.read_b,
    )
    logit = gl.zeros(i.shape, gl.float32, layout=L)
    for index in gl.static_range(4):
        selected = (controller == index) & (i < 4 * STREAMS)
        logit += gl.load(sources[index] + stream, selected, 0).to(gl.float32)
    value = 1.0 / (1.0 + gl.exp(gl.where(controller == 1, logit, -logit)))
    value = gl.where(
        ((controller != 1) | RETENTION) & ((controller != 3) | TWO_READERS), value, 0.0
    )
    factor = gl.where(controller == 0, 2.0, gl.where(controller == 1, 0.1, 1.0))
    coefficient = gl.where(controller == 1, gl.fma(-0.1, value, 1.0), factor * value)
    if C is not None:
        gl.store(C + i, coefficient, (row == 0) & (i < 4 * STREAMS))
        gl.store(
            C + i + 4 * STREAMS, (factor * value) * (1 - value), (row == 0) & (i < 4 * STREAMS)
        )
    return coefficient


@g.jit
def _forward(
    write_tensors,
    saved_coefficients,
    read_tensors,
    controllers,
    GEOMETRY: gl.constexpr,
    CONFIG: gl.constexpr,
    EPSILON: gl.constexpr = 1.0e-5,
    PAD_ROWS: gl.constexpr = False,
    DEPENDENT: gl.constexpr = False,
    EXCHANGE: gl.constexpr = None,
):
    """Write residuals and folded reader outputs for one row (or one slice of it) per CTA.

    Args:
        write_tensors: ``_WriteArgs``.
        saved_coefficients: Coefficient save buffer, or None in inference.
        read_tensors: ``_ReadArgs``.
        controllers: ``_ControllerArgs``.
        GEOMETRY: ``_Geometry``.
        CONFIG: ``_ForwardConfig``.
        EPSILON: RMSNorm epsilon.
        PAD_ROWS: Zero the MXFP8 scales of padded rows in the last 128-row tile.
        DEPENDENT: Launched as a programmatic dependent (inference): compute the coefficients
            from the controller logits and prefetch the residual row and gammas into L2, then
            wait for the preceding grid before reading any of its outputs.
        EXCHANGE: ``_exchange_asm`` PTX for a cluster launch, or None.
    """
    # Before gdc_wait (DEPENDENT) the kernel reads only parameters, never preceding-grid outputs.
    H: gl.constexpr = GEOMETRY.H
    gammas = read_tensors.gammas
    TWO_READERS: gl.constexpr = len(gammas) == 2  # Split CTAs still mix both updates.
    partition: gl.constexpr = (CONFIG.PARTS, CONFIG.SPLIT, CONFIG.CLUSTER)
    K: gl.constexpr = CONFIG.CLUSTER
    if CONFIG.SPLIT:
        reader = gl.program_id(1)
        gammas = (gl.where(reader == 0, gammas[0], gammas[1]),)
        read = gl.where(reader == 0, controllers.read_a, controllers.read_b)
        controllers = _ControllerArgs(
            controllers.write, controllers.retain, read, controllers.read_b
        )
    row = gl.program_id(0) // K if K > 1 else gl.program_id(0)
    lead = gl.program_id(0) % K == 0 if K > 1 else True  # one owner of per-row statistics
    # Each CTA uses local coefficients; only row zero saves them for backward (in a cluster,
    # only its lead CTA: the others pass a nonzero row).
    coefficients = _prepare_coefficients(
        controllers,
        saved_coefficients,
        gl.where(lead, row, 1) if K > 1 else row,
        TWO_READERS,
        GEOMETRY.STREAMS,
        controllers.retain is not None,
    )
    width: gl.constexpr = CONFIG.WIDTH
    paired: gl.constexpr = CONFIG.PAIRED_REDUCE and not CONFIG.SPLIT and K == 1
    NUM_SEGMENTS: gl.constexpr = triton.cdiv(H // K, width)

    cols = gl.arange(
        0, width, layout=gl.constexpr(BlockedLayout([CONFIG.VECTOR], [32], [gl.num_warps()], [0]))
    )
    if K > 1:
        cols += (gl.program_id(0) % K) * (H // K)  # this CTA's column slice
    if DEPENDENT:
        # Decode rows are latency bound on cold loads: start the residual row and gamma
        # fetches into L2 before waiting for the preceding grid. A prefetch is only a hint and
        # L2 is coherent, so it never observes a stale value; the real loads follow the wait.
        STREAMS: gl.constexpr = GEOMETRY.STREAMS
        for segment in gl.static_range(NUM_SEGMENTS):
            cc = cols + segment * width
            for stream in gl.static_range(STREAMS):
                _prefetch_l2(write_tensors.residual + row * (STREAMS * H) + stream * H + cc, cc, H)
            for reader in gl.static_range(len(gammas)):
                _prefetch_l2(gammas[reader] + cc, cc, H)
        gdc_wait()  # PDL: wait for the preceding grid before reading its outputs
    mixed = ()
    totals = (0.0,) * len(gammas)
    # Short rows are latency bound: issue the whole row's loads before any dependent store.
    loaded = ()
    if CONFIG.LOADS_FIRST:
        for segment in gl.static_range(NUM_SEGMENTS):
            loaded += (
                _load_fragment(write_tensors, row, cols + segment * width, GEOMETRY, CONFIG),
            )
    if EXCHANGE is not None:
        # With the loads in flight, arm this CTA's exchange barrier and signal the cluster.
        one: gl.constexpr = BlockedLayout([1], [32], [gl.num_warps()], [0])
        gl.inline_asm_elementwise(
            EXCHANGE[0],
            "=r,r",
            [gl.full([1], 0, gl.int32, layout=one)],
            dtype=gl.int32,
            is_pure=False,
            pack=1,
        )

    # Each CTA processes one row, padding the final segment when needed.
    for segment in gl.static_range(NUM_SEGMENTS):
        cc = cols + segment * width
        owner = _owns_columns(cc, H, partition)
        # shortcut_add, residual read, residual write
        # a, b will be inputs to the rmsnorm
        a, b = _mix_fragment(
            write_tensors,
            coefficients,
            row,
            cc,
            GEOMETRY,
            TWO_READERS,
            CONFIG,
            OWNER=owner,
            LOADED=loaded[segment] if CONFIG.LOADS_FIRST else None,
        )
        next_totals = ()
        for reader in gl.static_range(len(gammas)):
            x = a if reader == 0 else b
            fragment = x
            if segment < CONFIG.OFFLOAD_SEGMENTS:
                fragment = gl.allocate_shared_memory(
                    gl.float32, [width], gl.constexpr(SwizzledSharedLayout(1, 1, 1, [0])), x
                )
            # Group square sums to balance reduction work and register pressure.
            squares = gl.reshape(x * x, (width // 16, 16))
            next_totals += (totals[reader] + gl.sum(squares, 1),)
            mixed += (fragment,)
        totals = next_totals

    if EXCHANGE is not None:
        # Sum the cluster's partial squares in rank order (identical bits in every CTA).
        partials = ()
        for reader in gl.static_range(len(gammas)):
            partials += (gl.zeros([1], gl.float32, layout=one) + gl.sum(totals[reader], 0),)
        if len(partials) == 2:
            totals = gl.inline_asm_elementwise(
                EXCHANGE[1],
                "=f,=f,f,f",
                list(partials),
                dtype=(gl.float32, gl.float32),
                is_pure=False,
                pack=1,
            )
        else:
            totals = (
                gl.inline_asm_elementwise(
                    EXCHANGE[1], "=f,f", list(partials), dtype=gl.float32, is_pure=False, pack=1
                ),
            )
    queue = ()
    # The final reader also drains the queued stages.
    for reader_index in gl.static_range(len(gammas)):
        for segment_index in gl.static_range(
            NUM_SEGMENTS + (CONFIG.PIPELINE_DEPTH if reader_index == len(gammas) - 1 else 0)
        ):
            if segment_index < NUM_SEGMENTS:
                # Loading gamma early slightly improves performance
                weight = gl.load(
                    gammas[reader_index] + cols + segment_index * width,
                    (cols + segment_index * width < H)
                    & _owns_columns(cols + segment_index * width, H, partition),
                    0,
                ).to(gl.float32)

                if paired and reader_index == 0 and segment_index == 0:
                    if read_tensors.scales[0] is not None:
                        norm_layout: gl.constexpr = BlockedLayout([1], [32], [gl.num_warps()], [0])
                        totals = (
                            gl.convert_layout(totals[0], norm_layout),
                            gl.convert_layout(totals[1], norm_layout),
                        )
                    norm_total = gl.convert_layout(
                        gl.join(totals[0], totals[1]),
                        gl.constexpr(BlockedLayout([1, 2], [32, 1], [gl.num_warps(), 1], [0, 1])),
                    )
                elif not paired and segment_index == 0:
                    norm_total = totals[reader_index]
                    # Tuning this layout can reduce register pressure and improve latency hiding.
                    if CONFIG.REDUCE_VECTOR and K == 1:
                        norm_total = gl.convert_layout(
                            norm_total,
                            gl.constexpr(
                                BlockedLayout([CONFIG.REDUCE_VECTOR], [32], [gl.num_warps()], [0])
                            ),
                        )
                else:
                    norm_total = None

                if norm_total is not None:
                    # Reduce, normalize, and save statistics for backward.
                    rstd = gl.rsqrt(gl.sum(norm_total, 0) / H + EPSILON)
                    norms = gl.split(rstd) if paired else (rstd,)
                    if read_tensors.rstd[0] is not None:
                        for reader in gl.static_range(len(norms)):
                            gl.store(
                                read_tensors.rstd[reader if paired else reader_index] + row,
                                norms[reader],
                                mask=lead if K > 1 else None,
                            )
                prefetch_rstd = norms[reader_index if paired else 0]
                # load the activation segment for the right SEGMENT and reader
                x = mixed[segment_index * len(gammas) + reader_index]
                # if segmented, load from shared mem
                if segment_index < CONFIG.OFFLOAD_SEGMENTS:
                    x = x.load(cols.type.layout)
                queue += ((x, weight, prefetch_rstd),)

            if reader_index * NUM_SEGMENTS + segment_index >= CONFIG.PIPELINE_DEPTH:
                x, weight, rstd = queue[0]
                output_stage = reader_index * NUM_SEGMENTS + segment_index - CONFIG.PIPELINE_DEPTH
                for target in gl.static_range(len(read_tensors.outputs)):
                    if target == (
                        gl.program_id(1) if CONFIG.SPLIT else output_stage // NUM_SEGMENTS
                    ):
                        _emit(
                            x,
                            (weight, read_tensors.outputs[target], read_tensors.scales[target]),
                            (row, cols + (output_stage % NUM_SEGMENTS) * width),
                            rstd,
                            H,
                            partition,
                            gl.num_programs(0) // K if PAD_ROWS else None,
                        )
                queue = queue[1:]
