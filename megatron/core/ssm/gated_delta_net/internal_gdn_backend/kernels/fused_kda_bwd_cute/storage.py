"""Shared-memory and tensor-memory contracts for the fused KDA backward.

``TMEM_RANGES`` assigns the 512 tensor-memory columns to accumulators per
chunk phase and is checked for overlapping live ranges at import time.  Every
shared-memory buffer below lists its contents in chunk order; a buffer is only
reused for a new tensor after the phase that last reads the previous one.
"""

from dataclasses import dataclass

import cutlass
import cutlass.cute as cute

_BT = 64
_DK = 128
_DV = 128
_THREADS_PER_CTA = 384
_BUFFER_ALIGN_BYTES = 1024
_SM100_OPTIN_SMEM_BYTES = 232_448

# Phases of one chunk, in execution order.
PHASES = ("seed", "wy", "grad", "inv_mid", "inv_out", "intra", "epilogue")
_PHASE = {name: index for index, name in enumerate(PHASES)}

TMEM_COLUMNS = 512
TMEM_RANGES = (
    # name, first column, one-past-last column, first phase, last phase
    ("dh", 0, 128, "seed", "epilogue"),
    ("dq", 128, 192, "grad", "epilogue"),
    ("dk_plus", 192, 256, "grad", "epilogue"),
    # dAkk (64x64) during WY, then the dk_minus accumulator.
    ("dk_minus", 256, 320, "wy", "epilogue"),
    # Token-major slot A: X, v_new, then the inverse-chain squares.
    ("slot_a", 320, 384, "seed", "inv_out"),
    # Token-major slot B: dv_new, D, then dAqk.
    ("slot_b", 384, 448, "seed", "inv_mid"),
    # Intra-chunk partial ring (two channel-major slots; with a dS tape, which
    # leaves "dh" unused, two more slots at columns 0 and 64).
    ("ring", 320, 448, "intra", "intra"),
    # Channel-major gate column (lane = key channel, column = token).
    ("gate", 448, 512, "seed", "epilogue"),
)


def _validate_tmem_ranges() -> None:
    """Spatially overlapping ranges must have disjoint phase intervals."""

    for name, begin, end, first, last in TMEM_RANGES:
        assert 0 <= begin < end <= TMEM_COLUMNS, name
        assert _PHASE[first] <= _PHASE[last], name
    for index, lhs in enumerate(TMEM_RANGES):
        for rhs in TMEM_RANGES[index + 1 :]:
            columns = max(lhs[1], rhs[1]) < min(lhs[2], rhs[2])
            phases = max(_PHASE[lhs[3]], _PHASE[rhs[3]]) <= min(_PHASE[lhs[4]], _PHASE[rhs[4]])
            assert not (columns and phases), f"live TMEM aliases overlap: {lhs[0]} and {rhs[0]}"


_validate_tmem_ranges()


def _tile(dtype, elements):
    return cute.struct.Align[cute.struct.MemRange[dtype, elements], _BUFFER_ALIGN_BYTES]


@cute.struct
class SharedStorage:
    """Fixed buffers; reuse follows the phase order documented per field."""

    # One-shot role edges, indexed by the kernel's _EDGE_* constants.
    edges: cute.struct.MemRange[cutlass.Int64, 32]
    # Intra ring (two slots, four with a dS tape): full (MMA commit) and empty
    # (one arrival per consumer warp) barriers per slot.
    ring_full: cute.struct.MemRange[cutlass.Int64, 4]
    ring_empty: cute.struct.MemRange[cutlass.Int64, 4]
    # Persistent tape schedule: work items handed from the TMA warp to the
    # three consumer roles through two slots.
    item_full: cute.struct.MemRange[cutlass.Int64, 2]
    item_empty: cute.struct.MemRange[cutlass.Int64, 2]
    items: cute.struct.MemRange[cutlass.Int32, 2]
    # Two-CTA split: the other CTA arrives once dS for this CTA's next chunk is ready.
    ds_ready: cute.struct.MemRange[cutlass.Int64, 1]
    tmem_holding: cute.struct.MemRange[cutlass.Int32, 1]

    beta: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, _BT], 16]
    dbeta: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, _BT], 16]
    # Channel-major exchange between the two consumer warpgroups: per-channel
    # dg boundary partials (one row per warpgroup) and the reverse-cumsum carry.
    xchg: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, 2 * _DK], 16]
    suffix: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, _DK], 16]

    # TMA inputs; never aliased so the next chunk can be prefetched.
    sQ: _tile(cutlass.BFloat16, _BT * _DK)
    sK: _tile(cutlass.BFloat16, _BT * _DK)
    sDo: _tile(cutlass.BFloat16, _BT * _DV)
    sH: _tile(cutlass.BFloat16, _DK * _DV)
    sAkk: _tile(cutlass.BFloat16, _BT * _BT)
    # v (TMA) -> Delta = v - X -> dv staging.
    sV: _tile(cutlass.BFloat16, _BT * _DV)
    # Aqk (TMA) -> dAqk.
    sAqk: _tile(cutlass.BFloat16, _BT * _BT)
    # Akk * diag(beta) -> dAkk -> inverse middle term -> dL.
    sSq: _tile(cutlass.BFloat16, _BT * _BT)
    # BF16 dS snapshot -> two anchored intra half-tiles.
    sDS: _tile(cutlass.BFloat16, _DK * _DV)
    # k * Gamma -> -(k * beta * Gamma) -> anchored intra half-tile.
    sKA: _tile(cutlass.BFloat16, _BT * _DK)
    # k * 2^(G_L - G) -> D -> anchored intra half-tile.
    sKB: _tile(cutlass.BFloat16, _BT * _DK)
    # scale * q * Gamma -> anchored intra half-tile.
    sQA: _tile(cutlass.BFloat16, _BT * _DK)
    # dv_new -> v_new -> anchored intra half-tile.
    sT21: _tile(cutlass.BFloat16, _BT * _DV)


assert SharedStorage.size_in_bytes() <= _SM100_OPTIN_SMEM_BYTES, SharedStorage.size_in_bytes()


@dataclass(frozen=True)
class LayoutBudget:
    smem_bytes: int
    tmem_columns: int
    threads: int


def get_layout_budget() -> LayoutBudget:
    return LayoutBudget(SharedStorage.size_in_bytes(), TMEM_COLUMNS, _THREADS_PER_CTA)


def tmem_offset(name: str) -> int:
    for entry in TMEM_RANGES:
        if entry[0] == name:
            return entry[1]
    raise KeyError(name)


__all__ = [
    "LayoutBudget",
    "PHASES",
    "SharedStorage",
    "TMEM_COLUMNS",
    "TMEM_RANGES",
    "get_layout_budget",
    "tmem_offset",
]
