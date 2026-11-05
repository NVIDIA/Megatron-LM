# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Forward seam kernel: write the current residual and the folded RMSNorm reader operands.

``choose_forward_config`` selects a GB300-tuned schedule; inference skips backward state.
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
    _coefficient,
    _emit_mxfp8,
    _tensor,
    allocate_operand,
    operand_buffers,
    validate_num_streams,
)


class _WriteArgs(NamedTuple):
    """Residual write; ``other_update``/``summed_update`` belong to a shortcut add (else None)."""

    residual: object
    update: object
    other_update: object
    output: object
    summed_update: object


class _ReadArgs(NamedTuple):
    """Per reader: gamma, saved rstd (None in inference), output, MXFP8 scales (None: BF16)."""

    gammas: tuple
    rstd: tuple
    outputs: tuple
    scales: tuple


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


# GB300 training/prefill configs (3 streams, rows >= 2048) from a paired sweep at 2K-16K rows.
# Key: (hidden, readers, MXFP8 handoff).
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


def _bf16_config(*fields, vector):
    return _ForwardConfig(*fields, LOADS_FIRST=True, VECTOR=vector)


# GB300 configs for a BF16 residual stream (3 streams; training from 2048 rows, inference from
# 512), which needs more rows in flight. Key: (hidden, readers, updates, MXFP8 handoff).
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

# BF16 residual streams of other shapes from 2048 rows: the sweep's most robust config.
_BF16_STREAM_CONFIG = _ForwardConfig(4096, 1, 1, False, 0, 8, 128, False, LOADS_FIRST=True)

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


def _training_config(h, streams, readers):
    """Training tree for rows >= 2048 (no row dependence in a 107-shape GB300 sweep)."""
    half = (2048, 2, 1, True, 0, 8, 128, False)
    if h <= 9728:
        fields = (4096, 2, 0, True, 0, 8, 80, False) if streams <= 1 else half
    elif h <= 15872:
        fields = half if streams <= 1 else (4096, 2, 1, False, 0, 16, 128, True)
    elif readers == 1:
        fields = (4096, 4, 0, False, 0, 16, 128, True)
    else:
        fields = (4096, 2, 0, True, 0, 16, None, True)
    return _ForwardConfig(*fields, LOADS_FIRST=True)


@lru_cache
def _programmatic_launch(device):
    """PDL launches and CTA clusters need sm_90+."""
    return torch.cuda.get_device_capability(device)[0] >= 9


def choose_forward_config(
    m, h, streams, readers=1, *, mxfp8=False, inference=False, updates=None, bf16_residual=False
):
    """Choose the row-kernel schedule; ``updates`` None means one per reader."""
    cfg = _choose(m, h, streams, readers, mxfp8, inference, updates, bf16_residual)
    # Pairing the two readers' reductions needs two readers.
    return cfg._replace(PAIRED_REDUCE=False) if cfg.PAIRED_REDUCE and readers == 1 else cfg


def _choose(m, h, streams, readers, mxfp8, inference, updates, bf16_residual):
    tuned = 3072 <= h <= 16384 and streams <= 6
    if bf16_residual:
        cfg = _FORWARD_CONFIGS_BF16.get((h, readers, updates or readers, mxfp8))
        if streams == 3 and cfg is not None and m >= (512 if inference else 2048):
            return cfg
        if not inference and m >= 2048 and tuned:
            return _BF16_STREAM_CONFIG
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
        if streams == 3 and m >= 2048 and (h, readers, mxfp8) in _FORWARD_CONFIGS:
            return _FORWARD_CONFIGS[h, readers, mxfp8]
        if m >= 2048 and tuned:
            return _training_config(h, streams, readers)
        if h * streams <= 10240:
            profile = "compact" if h * streams <= 5120 else "paired"
        elif h <= 13312:
            profile = "pipelined" if m <= 128 else "streamed"
        else:
            profile = "streamed" if readers == 1 else "wide"
    # Untuned fallback profiles.
    width = max(32, triton.next_power_of_2(h))
    tile = min(width, 4096)
    cfg = {
        "compact": lambda: _ForwardConfig(tile, REDUCE_VECTOR=0),
        "row": lambda: _ForwardConfig(width, 0, 0, False, 16, 8, 128, True),
        "paired": lambda: _ForwardConfig(tile, 2, 0, readers == 2, 0, 8, 80),
        "pipelined": lambda: _ForwardConfig(tile, 4, 1, readers == 2, 0, 16, 64, True),
        "streamed": lambda: _ForwardConfig(tile, 2, 1, False, 0, 8, 128, True),
        "wide": lambda: _ForwardConfig(min(width, 8192), 0, 0, readers == 2, 0, 16),
    }[profile]()
    replicas = min(8, 128 // triton.next_power_of_2(m)) if inference and m <= 64 else 1
    return _replicated(cfg, replicas, readers, h)


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
    """Write the current residual and one or two RMSNorm readers of it.

    ``updates`` is one BF16 ``[M, H]`` update or a shortcut add of two (its BF16 sum is rounded
    before the write); the ``[M, S * H]`` residual keeps its dtype. All arithmetic is FP32 and
    readers fold unrounded values. ``retain`` None is a static identity retention (not zero
    logits). ``mxfp8_handoff`` quantizes reader zero to rowwise MXFP8; other readers are BF16.

    Returns ``(current residual, SavedWrite or None in inference, reader operands)``.
    """
    validate_num_streams(num_streams)
    n = len(reads)
    if n not in (1, 2) or len(gammas) != n or len(updates) not in (1, 2):
        raise ValueError("one or two readers write one update or a shortcut add of two")
    shortcut_add = len(updates) == 2
    if not residual.is_cuda or updates[0].ndim != 2:
        raise ValueError("expected CUDA [M,H] updates")
    m, h = updates[0].shape
    device = residual.device
    _tensor(residual, (m, num_streams * h), residual.dtype, device, "residual")
    for u in updates:
        _tensor(u, (m, h), torch.bfloat16, device, "update")
    for controller in (write, *((retain,) if retain is not None else ()), *reads):
        logits = controller.dtype in (torch.float32, torch.bfloat16)
        if not logits or controller.ndim != 1 or controller.numel() < num_streams:
            raise ValueError(f"controllers require at least {num_streams} FP32 or BF16 logits")
        _tensor(controller, controller.shape, controller.dtype, device, "controller")
    for gamma in gammas:
        _tensor(gamma, (h,), torch.bfloat16, device, "gamma")
    rows = m
    if inference:
        from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
            is_batch_invariant_mode_enabled,
        )

        rows = _BATCH_INVARIANT_ROWS if is_batch_invariant_mode_enabled() else m
    config = choose_forward_config(
        rows,
        h,
        num_streams,
        n,
        mxfp8=mxfp8_handoff,
        inference=inference,
        updates=len(updates),
        bf16_residual=residual.dtype == torch.bfloat16,
    )
    c = None if inference else torch.empty((8 * num_streams,), device=device, dtype=torch.float32)

    if type(residual).__name__ not in ("FakeTensor", "FunctionalTensor"):
        # FP32 residuals use 16-byte packet loads; the backward reads the saved residual and
        # updates through TMA (rows and H divisible by 32).
        packets = h % 4 == 0 and residual.dtype == torch.float32
        tma = not inference and m % 32 == 0 and h % 32 == 0
        aligned = ((residual,) if packets or tma else ()) + (tuple(updates) if tma else ())
        if any(t.data_ptr() % 16 for t in aligned):
            raise ValueError("FP32 packet or training TMA operands must be 16-byte aligned")
    if mxfp8_handoff and (m % 32 or h % 32):
        raise ValueError("MXFP8 handoff requires rows and hidden width divisible by 32")

    current = torch.empty_like(residual)
    summed = torch.empty_like(updates[0]) if shortcut_add and not inference else None
    stats = tuple(
        None if inference else torch.empty(m, dtype=torch.float32, device=device) for _ in range(n)
    )
    qs = []
    for reader in range(n):
        bf16 = not mxfp8_handoff or reader > 0
        qs.append(allocate_operand((m, h), device, bf16))
        # Inference pads rows in-kernel; partial scale columns still need clearing.
        if not bf16 and (h % 128 or (m % 128 and not inference)):
            qs[-1]._rowwise_scale_inv.zero_()

    saved = None
    if not inference:
        saved = SavedWrite(
            summed if summed is not None else updates[0].detach(),
            c,
            tuple(gamma.detach() for gamma in gammas),
            stats,
            tuple(0 if v is None else v.numel() for v in (write, retain, *reads)),
            num_streams,
            retain is not None,
            len(updates),
        )
    # Inference is a programmatic dependent launch: it reads only parameters and prefetches
    # into L2 before waiting on the previous kernel (griddepcontrol.wait).
    dependent = inference and _programmatic_launch(device)
    grid = (m, 2 if config.SPLIT else 1, config.PARTS)
    write_args = _WriteArgs(
        residual, updates[0], updates[1] if shortcut_add else None, current, summed
    )
    kwargs = dict(
        read_tensors=_ReadArgs(gammas, stats, *operand_buffers(qs, columnwise=False)),
        H=h,
        STREAMS=num_streams,
        # A runtime unit retention coefficient preserves Ultra register occupancy.
        HAS_RETENTION=retain is not None or h == 10240,
        EPSILON=epsilon,
        # Logits (write, retain, read_a, read_b); read_b repeats read_a with one reader.
        controllers=(write, retain, reads[0], reads[-1]),
        saved_coefficients=c,
        PAD_ROWS=inference and mxfp8_handoff and bool(m % 128),
        DEPENDENT=dependent,
        EXCHANGE=None,
        enable_fp_fusion=False,
        launch_pdl=dependent,
    )
    if config.CLUSTER > 1:
        kwargs["EXCHANGE"] = _exchange_asm(config.CLUSTER, 1 if config.SPLIT else n)
        launch = dict(CONFIG=config, num_warps=config.NUM_WARPS, maxnreg=config.MAXNREG)
        # The launch must repeat the warmup's keyword order to hit the patched kernel.
        if not (
            _programmatic_launch(device)
            and _as_cluster(write_args, grid, config.CLUSTER, kwargs | launch)
        ):
            # Same slices, one CTA per row.
            config, kwargs["EXCHANGE"] = config._replace(CLUSTER=1), None
    _forward[grid](
        write_args, **kwargs, CONFIG=config, num_warps=config.NUM_WARPS, maxnreg=config.MAXNREG
    )
    return current, saved, tuple(qs)


def _as_cluster(write_args, grid, ctas, kwargs):
    """Launch ``_forward`` as ``ctas``-CTA clusters via Triton 3.7's private packed metadata.

    An exchange without its peers would hang, so an unexpected form returns False (unclustered).
    """
    kernel = _forward.warmup(write_args, grid=grid, **kwargs)
    meta = kernel.packed_metadata
    if not (isinstance(meta, tuple) and len(meta) == 3 and meta[1] in (1, ctas)):
        return False
    kernel.packed_metadata = (meta[0], ctas, meta[2])
    return True


def _exchange_asm(ctas, partials):
    """PTX ``(init, exchange, constraints)``: peers ``st.async`` reader k's partial into
    ``xbuf[k * ctas + rank]`` (Gluon cannot address peer shared memory); every CTA sums the
    partials in rank order, so all compute identical bits."""
    init = [
        ".shared .align 8 .b64 xbar;",
        f".shared .align 16 .f32 xbuf[{ctas * partials}];",
        "{",
        ".reg .pred lead;",
        ".reg .u32 tid, mbar;",
        "mov.u32 tid, %tid.x;",
        "setp.eq.u32 lead, tid, 0;",
        "mov.u32 mbar, xbar;",
        "@lead mbarrier.init.shared::cta.b64 [mbar], 1;",
        f"@lead mbarrier.arrive.expect_tx.shared::cta.b64 _, [mbar], {(ctas - 1) * partials * 4};",
        "fence.mbarrier_init.release.cluster;",
        "barrier.cluster.arrive.relaxed.aligned;",
        "mov.u32 $0, 0;",
        "}",
    ]
    lines = [
        "{",
        ".reg .pred lead, send, own, done;",
        ".reg .u32 tid, rank, buf, mbar, slot, peer_mbar, peer_slot;",
        ".reg .f32 v;",
        "mov.u32 tid, %tid.x;",
        "mov.u32 rank, %cluster_ctarank;",
        "setp.eq.u32 lead, tid, 0;",
        "mov.u32 buf, xbuf;",
        "mov.u32 mbar, xbar;",
        "mad.lo.u32 slot, rank, 4, buf;",
        "barrier.cluster.wait.aligned;",
    ]
    for peer in range(ctas):
        lines += [
            f"setp.ne.and.u32 send, rank, {peer}, lead;",
            f"@send mapa.shared::cluster.u32 peer_mbar, mbar, {peer};",
            f"@send mapa.shared::cluster.u32 peer_slot, slot, {peer};",
        ]
        for k in range(partials):
            if k:
                lines.append(f"@send add.u32 peer_slot, peer_slot, {ctas * 4};")
            lines.append(
                "@send st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 "
                f"[peer_slot], ${partials + k}, [peer_mbar];"
            )
    lines += [
        "WAIT_${:uid}:",
        "mbarrier.try_wait.parity.shared::cta.b64 done, [mbar], 0;",
        "@!done bra WAIT_${:uid};",
    ]
    for k in range(partials):
        for peer in range(ctas):
            lines += [
                f"setp.eq.u32 own, rank, {peer};",
                f"@own mov.f32 v, ${partials + k};",
                f"@!own ld.shared.f32 v, [buf + {(k * ctas + peer) * 4}];",
                f"add.rn.f32 ${k}, ${k}, v;" if peer else f"mov.f32 ${k}, v;",
            ]
    lines.append("}")
    constraints = ",".join(["=f"] * partials + ["f"] * partials)
    return "\n".join(init), "\n".join(lines), constraints


@g.jit
def _load_residual(
    write_tensors, row, stream, cols, H: gl.constexpr, STREAMS: gl.constexpr, CONFIG: gl.constexpr
):
    """One stream's residual fragment: FP32 4-wide packets when H allows, else plain loads."""
    residual = write_tensors.residual + row * (STREAMS * H) + stream * H + cols
    if write_tensors.residual.dtype.element_ty != gl.float32 or H % 4:
        old = gl.load(residual, cols < H, 0)  # BF16 stays packed until _mix_fragment widens it
    else:
        # Packets are wholly valid or wholly masked; whole tiles pass a constant-true predicate.
        if H % CONFIG.WIDTH:
            valid = (cols < H).to(gl.int32)
        else:
            valid = gl.full(cols.shape, 1, gl.int32, cols.type.layout)
        old = gl.inline_asm_elementwise(
            "{ .reg .pred p; setp.ne.u32 p, $8, 0; mov.b32 $0, 0; mov.b32 $1, 0; mov.b32 $2, 0;"
            " mov.b32 $3, 0; @p ld.global.nc.v4.f32 {$0,$1,$2,$3},[$4]; }",
            "=f,=f,=f,=f,l,l,l,l,r,r,r,r",
            (residual.to(gl.uint64), valid),
            dtype=gl.float32,
            is_pure=False,
            pack=4,
        )
    return old


@g.jit
def _load_fragment(
    write_tensors, row, cols, H: gl.constexpr, STREAMS: gl.constexpr, CONFIG: gl.constexpr
):
    """``(update, other update or update again, per-stream residuals with LOADS_FIRST)``."""
    mask = cols < H
    update = gl.load(write_tensors.update + (row * H + cols), mask, 0)
    other = update  # unused without a shortcut add
    if write_tensors.other_update is not None:
        other = gl.load(write_tensors.other_update + (row * H + cols), mask, 0)
    olds = ()
    if CONFIG.LOADS_FIRST:
        for stream in gl.static_range(STREAMS):
            olds += (_load_residual(write_tensors, row, stream, cols, H, STREAMS, CONFIG),)
    return update, other, olds


@g.jit
def _mix_fragment(
    write_tensors,
    coefficients,
    row,
    cols,
    H: gl.constexpr,
    STREAMS: gl.constexpr,
    HAS_RETENTION: gl.constexpr,
    TWO_READERS: gl.constexpr,
    CONFIG: gl.constexpr,
    OWNER,
    LOADED=None,
):
    """Store one fragment's residual streams (columns in OWNER) and return both readers'
    pre-norm inputs (zero outside the row; the second is zero with one reader)."""
    mask = cols < H
    if LOADED is None:
        LOADED = _load_fragment(write_tensors, row, cols, H, STREAMS, CONFIG)
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
    for stream in gl.static_range(STREAMS):
        owner = OWNER & (not CONFIG.SPLIT or gl.program_id(1) == (stream == STREAMS - 1))
        if CONFIG.LOADS_FIRST:
            old = olds[stream]
        else:
            old = _load_residual(write_tensors, row, stream, cols, H, STREAMS, CONFIG)
        old = old.to(gl.float32)  # exact for BF16

        write = _coefficient(coefficients, stream)
        retain = _coefficient(coefficients, STREAMS + stream) if HAS_RETENTION else 1.0

        separate = write * update + retain * old
        # The native writer's row/stream FMA contraction pattern, which replay reproduces.
        contracted = (stream == 0) | ((stream == 1) & (row % 32 < 12))
        value = gl.where(contracted, gl.fma(write, update, retain * old), separate)

        stored = value.to(write_tensors.output.dtype.element_ty)  # RN for a BF16 residual
        if not CONFIG.STORE_LATE:
            gl.store(
                write_tensors.output + row * (STREAMS * H) + stream * H + cols, stored, mask & owner
            )
        beta0 = _coefficient(coefficients, 2 * STREAMS + stream)
        x0 = gl.fma(beta0 * retain, old, x0)
        update_mix0 = gl.fma(beta0, write, update_mix0)
        if TWO_READERS:
            beta1 = _coefficient(coefficients, 3 * STREAMS + stream)
            x1 = gl.fma(beta1 * retain, old, x1)
            update_mix1 = gl.fma(beta1, write, update_mix1)
        if CONFIG.STORE_LATE:
            gl.store(
                write_tensors.output + row * (STREAMS * H) + stream * H + cols, stored, mask & owner
            )
    x0 = gl.fma(update_mix0, update, x0)
    if TWO_READERS:
        x1 = gl.fma(update_mix1, update, x1)
    return gl.where(mask, x0, 0.0), gl.where(mask, x1, 0.0)


@g.jit
def _owns_columns(cols, H: gl.constexpr, CONFIG: gl.constexpr):
    """Columns (or 32-column groups) this CTA stores of its row's PARTS replicas.

    A CLUSTER CTA's columns already lie in its own slice (WIDTH divides H / CLUSTER).
    """
    if CONFIG.SPLIT and CONFIG.PARTS == 2:
        owned = (cols < H // 2) == (gl.program_id(2) == 0)
    else:
        owned = True if CONFIG.PARTS == 1 else cols // (H // CONFIG.PARTS) == gl.program_id(2)
    return owned


@g.jit
def _emit(x, OPERANDS, indices, rstd, H: gl.constexpr, CONFIG: gl.constexpr, rows):
    """Normalize one reader fragment; store BF16 or (with scales) rowwise MXFP8.

    ``rows`` pads the last 128-row tile's MXFP8 scales, or None.
    """
    weight, output, scales = OPERANDS
    row, cols = indices
    mask = (cols < H) & _owns_columns(cols, H, CONFIG)
    y = (x * rstd) * weight
    if scales is None:
        gl.store(output + row * H + cols, y.to(gl.bfloat16), mask)
    else:
        grouped = gl.reshape(y, (x.numel // 32, 32))
        group = gl.min(gl.reshape(cols, (x.numel // 32, 32)), 1) // 32
        _emit_mxfp8(
            grouped,
            (output + row * H + cols, mask),
            (scales, (group < H // 32) & _owns_columns(group, H // 32, CONFIG)),
            (row, group, H),
            rows,
        )


@g.jit
def _prefetch_l2(pointers, cols, H: gl.constexpr):
    """Prefetch the 128-byte lines under ``pointers`` (columns < H) into L2."""
    LINE: gl.constexpr = 1024 // pointers.dtype.element_ty.primitive_bitwidth  # elements
    first = (cols < H) & (cols % LINE == 0)  # one prefetch per line
    gl.inline_asm_elementwise(
        """
        {
            .reg .pred p;
            setp.ne.u32 p, $2, 0;
            @p prefetch.global.L2 [$1];
            mov.u32 $0, 0;
        }
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
    """Return [write, retain, read_a, read_b] x STREAMS coefficients from the logits.

    Row zero also saves them and their sigmoid derivative terms to ``C`` (None in inference).
    """
    L: gl.constexpr = BlockedLayout([1], [32], [gl.num_warps()], [0])
    i = gl.arange(0, triton.next_power_of_2(4 * STREAMS), layout=L)
    controller, stream = i // STREAMS, i % STREAMS
    # One masked load per controller: each may be FP32 or BF16 and is widened exactly.
    logit = gl.zeros(i.shape, gl.float32, layout=L)
    for index in gl.static_range(4):
        if index != 1 or RETENTION:
            selected = (controller == index) & (i < 4 * STREAMS)
            logit += gl.load(controllers[index] + stream, selected, 0).to(gl.float32)
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
    H: gl.constexpr,
    STREAMS: gl.constexpr,
    HAS_RETENTION: gl.constexpr,
    CONFIG: gl.constexpr,
    EPSILON: gl.constexpr = 1.0e-5,
    PAD_ROWS: gl.constexpr = False,
    DEPENDENT: gl.constexpr = False,
    EXCHANGE: gl.constexpr = None,
):
    """One row (or slice) per CTA. DEPENDENT (inference PDL) prefetches the row into L2 from
    parameters only before ``gdc_wait``; EXCHANGE is ``_exchange_asm``'s PTX, or None."""
    gammas = read_tensors.gammas
    TWO_READERS: gl.constexpr = len(gammas) == 2  # Split CTAs still mix both updates.
    K: gl.constexpr = CONFIG.CLUSTER
    if CONFIG.SPLIT:
        reader = gl.program_id(1)
        gammas = (gl.where(reader == 0, gammas[0], gammas[1]),)
        read = gl.where(reader == 0, controllers[2], controllers[3])
        controllers = (controllers[0], controllers[1], read, controllers[3])
    row = gl.program_id(0) // K
    lead = gl.program_id(0) % K == 0  # one owner of per-row statistics
    # Only row zero saves coefficients (in a cluster, only its lead CTA).
    coefficients = _prepare_coefficients(
        controllers,
        saved_coefficients,
        gl.where(lead, row, 1),
        TWO_READERS,
        STREAMS,
        controllers[1] is not None,
    )
    width: gl.constexpr = CONFIG.WIDTH
    paired: gl.constexpr = CONFIG.PAIRED_REDUCE and not CONFIG.SPLIT and K == 1
    NUM_SEGMENTS: gl.constexpr = triton.cdiv(H // K, width)

    cols = gl.arange(
        0, width, layout=gl.constexpr(BlockedLayout([CONFIG.VECTOR], [32], [gl.num_warps()], [0]))
    )
    cols += (gl.program_id(0) % K) * (H // K)  # this CTA's column slice
    if DEPENDENT:
        # Decode rows are latency bound: warm L2 before waiting. A prefetch is only a hint and
        # L2 is coherent, so it never observes a stale value; the real loads follow the wait.
        for segment in gl.static_range(NUM_SEGMENTS):
            cc = cols + segment * width
            for stream in gl.static_range(STREAMS):
                _prefetch_l2(write_tensors.residual + row * (STREAMS * H) + stream * H + cc, cc, H)
            for reader in gl.static_range(len(gammas)):
                _prefetch_l2(gammas[reader] + cc, cc, H)
        gdc_wait()
    mixed = ()
    totals = (0.0,) * len(gammas)
    loaded = ()
    if CONFIG.LOADS_FIRST:
        for segment in gl.static_range(NUM_SEGMENTS):
            loaded += (
                _load_fragment(write_tensors, row, cols + segment * width, H, STREAMS, CONFIG),
            )
    if EXCHANGE is not None:
        # With the loads in flight, arm this CTA's exchange barrier and signal the cluster.
        one: gl.constexpr = BlockedLayout([1], [32], [gl.num_warps()], [0])
        zero = gl.full([1], 0, gl.int32, layout=one)
        gl.inline_asm_elementwise(
            EXCHANGE[0], "=r,r", [zero], dtype=gl.int32, is_pure=False, pack=1
        )

    for segment in gl.static_range(NUM_SEGMENTS):
        cc = cols + segment * width
        owner = _owns_columns(cc, H, CONFIG)
        a, b = _mix_fragment(
            write_tensors,
            coefficients,
            row,
            cc,
            H,
            STREAMS,
            HAS_RETENTION,
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
        partials = ()
        for reader in gl.static_range(len(gammas)):
            partials += (gl.zeros([1], gl.float32, layout=one) + gl.sum(totals[reader], 0),)
        totals = gl.inline_asm_elementwise(
            EXCHANGE[1],
            EXCHANGE[2],
            list(partials),
            dtype=(gl.float32,) * len(partials),
            is_pure=False,
            pack=1,
        )
    queue = ()
    # The final reader also drains the queued stages.
    for reader_index in gl.static_range(len(gammas)):
        for segment_index in gl.static_range(
            NUM_SEGMENTS + (CONFIG.PIPELINE_DEPTH if reader_index == len(gammas) - 1 else 0)
        ):
            if segment_index < NUM_SEGMENTS:
                weight = gl.load(
                    gammas[reader_index] + cols + segment_index * width,
                    (cols + segment_index * width < H)
                    & _owns_columns(cols + segment_index * width, H, CONFIG),
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
                    rstd = gl.rsqrt(gl.sum(norm_total, 0) / H + EPSILON)
                    norms = gl.split(rstd) if paired else (rstd,)
                    if read_tensors.rstd[0] is not None:
                        for reader in gl.static_range(len(norms)):
                            gl.store(
                                read_tensors.rstd[reader if paired else reader_index] + row,
                                norms[reader],
                                mask=lead,
                            )
                prefetch_rstd = norms[reader_index if paired else 0]
                x = mixed[segment_index * len(gammas) + reader_index]
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
                            CONFIG,
                            gl.num_programs(0) // K if PAD_ROWS else None,
                        )
                queue = queue[1:]
