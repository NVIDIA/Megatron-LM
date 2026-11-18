# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Per-sequence zigzag context parallelism for DSv4 CSA (``attention_cp_layout='zigzag'``).

Layout
------
This is Megatron's standard THD zigzag split (``tex.thd_get_partitioned_indices``): every packed
sequence ``i`` (padded rows ``[cu[i], cu[i+1])``, length divisible by ``2*cp_size``) is cut into
``2*cp_size`` chunks of ``c_i`` rows and CP rank ``r`` holds chunk ``r`` and chunk
``2*cp_size-1-r`` of every sequence, in sequence order::

    local rows [lcu[i],       lcu[i] + c_i)  <-> pack rows [cu[i] + r*c_i,              +c_i)
    local rows [lcu[i] + c_i, lcu[i+1])      <-> pack rows [cu[i] + (2*cp_size-1-r)*c_i, +c_i)
    lcu = cu // cp_size,  c_i = (cu[i+1] - cu[i]) // (2*cp_size)

The rank therefore owns ``2*n_seq`` contiguous row *blocks* (block ``b = 2*i + h``), each inside
one sequence. The zigzag mirror point is the padded end of every sequence, so with the padding
tail represented as its own sequence (``thd_tail_padding_policy='append_dummy_seq'``) the
prefix-proportional kernels of the attention stack (indexer scoring / top-k, attend-to-all-
compressed layers) are balanced across CP ranks per real sequence. The data split, MTP, LM head
and loss all run on this layout (as for every other attention variant); nothing is
redistributed inside the model.

Geometry is data dependent (``n_seq`` and ``c_i`` come from ``cu_seqlens``), so every table is a
device tensor derived from ``cu_seqlens`` with static shapes (``2*n_seq`` blocks, where
``n_seq = cu_seqlens.numel() - 1`` is a shape, not a value) and no host synchronisation: the same
code runs eagerly and inside a captured CUDA graph, where ``cu_seqlens`` is a static input
buffer refreshed before replay.

Halo (boundary) rows
--------------------
Block ``b`` needs the ``d_window = max(csa_window_size, d_comp)`` rows preceding its chunk. With
``c_i >= d_window`` for every non-empty sequence (``validate_geometry`` / the data pipeline's
per-sequence padding floor, see ``dsv4_zigzag_min_padded_seq_len``) those rows lie entirely in
the preceding chunk, which is held by a fixed neighbour:

    block (i, 0) = chunk r            <- tail of chunk r-1       = rank r-1's block (i, 0)
    block (i, 1) = chunk 2*cp-1-r     <- tail of chunk 2*cp-2-r  = rank r+1's block (i, 1)
                                         (rank cp-1: its own block (i, 0))

so the exchange needs one point-to-point message per
neighbour in each direction (two neighbours for interior ranks, one for ranks 0 and cp-1), each
carrying ``n_seq`` tails of ``d_window`` rows (a fixed capacity of ``n_seq * d_window`` rows;
empty sequences contribute unused slots whose gradient is dropped). The halo buffer holds one
``d_window``-row slot per block::

    boundary_hidden = [slot 0 | slot 1 | ... | slot 2*n_seq-1],   slot b = rows before block b

Physical KV layout seen by the sparse kernels on a rank (the contiguous layout with the single
halo window replaced by the slot region)::

    kv_full_thd = cat(boundary_kv (2*n_seq*d_window rows), local (l_local rows), compressed)
    halo row m of block b        -> b*d_window + m
    local row j                  -> 2*n_seq*d_window + j
    compressed base              -> 2*n_seq*d_window + l_local

Compressor: block ``b`` (chunk ``k`` of sequence ``i``) owns the compressed groups whose last
token lies in the chunk, plus the groups that start within the ``d_comp`` rows before it (the
contiguous ownership rule evaluated per block), as one THD segment of the compaction buffer; the
per-rank compaction capacity is static (``compact_group_capacity_blocks``). The rank-major
all-gather is unchanged.

Indexer (ratio-4 layers): one fused call per layer with ``2*n_seq`` query segments (the blocks)
plus one empty synthetic segment, against a per-block copy of each sequence's compressed K
(truncated to the block's causally visible prefix); ``q_causal_offsets`` carries each block's
position inside its sequence.
"""

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Hashable, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist

from . import thd_layout_kernels

# =============================================================================
# Static helpers (no tensors)
# =============================================================================


def compressor_d_comp(ratio: int) -> int:
    """Compressor halo rows for ``ratio`` (matches the layout builders)."""
    return 8 if ratio == 4 else ratio


def boundary_window(csa_window_size: int, ratio: int) -> int:
    """``d_window`` of a layer: the rows an attention layer needs before each of its blocks."""
    d_comp = 8 if ratio == 4 else ratio if ratio > 1 else 0
    return max(int(csa_window_size), d_comp)


def max_boundary_window(csa_window_size: int, ratios: Sequence[int]) -> int:
    """Largest ``d_window`` over the model's layer kinds (the per-sequence floor uses it)."""
    return max([boundary_window(csa_window_size, int(r)) for r in ratios] + [int(csa_window_size)])


def dsv4_zigzag_min_padded_seq_len(
    cp_size: int, csa_window_size: int, ratios: Sequence[int]
) -> int:
    """Smallest padded length a non-empty packed sequence may have under this layout.

    Every half-chunk must cover the largest boundary window: ``c_i >= d_window`` with
    ``c_i = L_i / (2*cp_size)``, i.e. ``L_i >= 2*cp_size*d_window``.
    """
    return 2 * int(cp_size) * max_boundary_window(csa_window_size, ratios)


def validate_layer_padding_floor(
    floor: int, cp_size: int, csa_window_size: int, ratio: int
) -> None:
    """A layer's boundary window must fit the data pipeline's per-sequence padding floor.

    The floor (``TransformerConfig.dsv4_zigzag_min_padded_seq_len``) is derived from
    ``csa_compress_ratios``; a layer built with a ratio that list does not contain (a hybrid
    layer pattern that is not aligned with it) may need longer sequences than the pipeline
    guarantees.
    """
    needed = dsv4_zigzag_min_padded_seq_len(cp_size, csa_window_size, [int(ratio)])
    if needed > int(floor):
        raise ValueError(
            f"DSv4 attention_cp_layout='zigzag': a layer with compress ratio {ratio} needs "
            f"packed sequences of at least {needed} rows, but the data pipeline pads them to "
            f"{floor} (dsv4_zigzag_min_padded_seq_len, derived from csa_compress_ratios). List "
            "every compress ratio the model builds in csa_compress_ratios."
        )


def validate_hybrid_layer_pattern(main_pattern: Optional[str], mtp_pattern: Optional[str]) -> None:
    """Reject hybrid layer patterns that this layout does not cover.

    Only the DSv4 attention layers (W / C / H) implement the layout and its data conventions
    (the per-sequence padding floor, logical sequence boundaries in ``PackedSeqParams``); MLP
    and MoE layers act on single tokens. Every other layer type of a DSv4 model keeps requiring
    the contiguous layout.
    """
    from megatron.core.models.hybrid.layers.utils import Symbols

    supported = {Symbols.WINDOW, Symbols.CSA, Symbols.HCA, Symbols.MLP, Symbols.MOE, Symbols.PIPE}
    unsupported = sorted(set((main_pattern or "") + (mtp_pattern or "")) - supported)
    if unsupported:
        raise ValueError(
            "DSv4 attention_cp_layout='zigzag' supports hybrid layer patterns made of DSv4 "
            "attention (W / C / H), MLP and MoE layers only; hybrid_layer_pattern contains "
            f"{unsupported}. With other layer types DSv4 context parallelism requires "
            "cp_partition_mode='contiguous' (attention_cp_layout='contiguous')."
        )


def compact_group_capacity_blocks(l_local: int, n_seq_max: int, ratio: int) -> int:
    """Static per-rank compaction capacity (groups) for ``2*n_seq_max`` blocks.

    Block ``b`` (chunk ``k``, ``c`` rows) owns ``floor((k+1)c/ratio) - ceil(max(0, kc-d_comp)/ratio)
    <= (c + d_comp) / ratio`` groups, so the rank's total is at most
    ``floor((l_local + 2*n*d_comp) / ratio)``; rounded up to the compaction alignment (the
    contiguous formula ``(l_local + d_comp) // ratio`` is the one-block case).
    """
    ratio = int(ratio)
    d_comp = compressor_d_comp(ratio)
    alignment = 32 // math.gcd(32, ratio)
    cap = max(1, (int(l_local) + 2 * int(n_seq_max) * d_comp) // ratio)
    return ((cap + alignment - 1) // alignment) * alignment


def owner_rank_of_chunk(chunk: int, cp_size: int) -> int:
    """CP rank owning chunk ``chunk`` (``0 <= chunk < 2*cp_size``) of any sequence."""
    return chunk if chunk < cp_size else 2 * cp_size - 1 - chunk


# =============================================================================
# Host reference (pure Python ints; the kernels and the device tables follow these formulas)
# =============================================================================


def ref_blocks(
    cu: Sequence[int], cp_size: int, cp_rank: int
) -> List[Tuple[int, int, int, int, int]]:
    """``(seq, chunk, rows, local_offset, global_start)`` of every block, in local order."""
    out = []
    for i in range(len(cu) - 1):
        length = cu[i + 1] - cu[i]
        if length % (2 * cp_size):
            raise ValueError(f"sequence {i} length {length} is not divisible by {2 * cp_size}")
        c = length // (2 * cp_size)
        lcu = cu[i] // cp_size
        for h, k in enumerate((cp_rank, 2 * cp_size - 1 - cp_rank)):
            out.append((i, k, c, lcu + h * c, cu[i] + k * c))
    return out


def ref_local_to_global(cu: Sequence[int], cp_size: int, cp_rank: int) -> List[int]:
    """Pack row of every local row (the inverse of ``tex.thd_get_partitioned_indices``)."""
    rows = []
    for _, _, c, _, g0 in ref_blocks(cu, cp_size, cp_rank):
        rows.extend(range(g0, g0 + c))
    return rows


def ref_owned_groups(k: int, c: int, ratio: int) -> Tuple[int, int]:
    """``[first, stop)`` groups (sequence-local ids) compacted by chunk ``k`` of a sequence with
    ``c`` rows per chunk: groups whose last token lies in the chunk, plus the halo groups starting
    at most ``d_comp`` rows before it (the contiguous builders' rule)."""
    d_comp = compressor_d_comp(ratio)
    if c == 0:
        return 0, 0
    first = -(-max(0, k * c - d_comp) // ratio)
    stop = ((k + 1) * c) // ratio
    return first, max(first, stop)


def ref_group_owner_chunk(g: int, c: int, cp_size: int, ratio: int) -> int:
    """Chunk owning sequence-local group ``g`` (the chunk holding its last token)."""
    return min(((g + 1) * ratio - 1) // c, 2 * cp_size - 1)


def ref_compressor_layout(cu, cp_size, cp_rank, ratio, c_cap):
    """Host replica of the block compaction maps.

    Returns ``(compact_to_source, source_to_compact, group_ids, position_ids, local_cu, local_cuc,
    global_cuc)`` with the conceptual source buffer ``[slot tails (2*n_seq*d_comp) | local]``.
    """
    d_comp = compressor_d_comp(ratio)
    blocks = ref_blocks(cu, cp_size, cp_rank)
    n_blocks = len(blocks)
    l_local = cu[-1] // cp_size
    compact_len = c_cap * ratio
    compact_to_source = [-1] * compact_len
    group_ids = [-1] * c_cap
    local_cu, local_cuc = [0], [0]
    source_to_compact = [-1] * (n_blocks * d_comp + l_local)
    t = 0
    for b, (i, k, c, loff, g0) in enumerate(blocks):
        first, stop = ref_owned_groups(k, c, ratio)
        for g in range(first, stop):
            q = t // ratio
            if q < c_cap:
                group_ids[q] = g
            for tau in range(ratio):
                G = cu[i] + g * ratio + tau
                if G < g0:
                    src = b * d_comp + (G - (g0 - d_comp))
                else:
                    src = n_blocks * d_comp + loff + (G - g0)
                if t < compact_len:
                    compact_to_source[t] = src
                    source_to_compact[src] = t
                t += 1
        local_cu.append(min(t, compact_len))
        local_cuc.append(min(t // ratio, c_cap))
    position_ids = [g * ratio if g >= 0 else 0 for g in group_ids]
    global_cuc = [0]
    for i in range(len(cu) - 1):
        global_cuc.append(global_cuc[-1] + (cu[i + 1] - cu[i]) // ratio)
    return (
        compact_to_source,
        source_to_compact,
        group_ids,
        position_ids,
        local_cu,
        local_cuc,
        global_cuc,
    )


def ref_seq_to_rank_row(cu, cp_size, ratio, c_cap) -> List[int]:
    """Host replica of the sequence-major -> rank-major compressed row map."""
    n_seq = len(cu) - 1
    global_cuc = [0]
    for i in range(n_seq):
        global_cuc.append(global_cuc[-1] + (cu[i + 1] - cu[i]) // ratio)
    seq_major_rows = (cu[-1] // cp_size) * cp_size // ratio
    out = []
    for lr in range(seq_major_rows):
        if lr >= global_cuc[-1]:
            out.append(-1)
            continue
        i = max(s for s in range(n_seq) if lr >= global_cuc[s])
        g = lr - global_cuc[i]
        c = (cu[i + 1] - cu[i]) // (2 * cp_size)
        k = ref_group_owner_chunk(g, c, cp_size, ratio)
        owner = owner_rank_of_chunk(k, cp_size)
        prefix = 0
        for j, kk, cc, _, _ in ref_blocks(cu, cp_size, owner):
            if (j, kk) == (i, k):
                break
            f, s = ref_owned_groups(kk, cc, ratio)
            prefix += s - f
        first, _ = ref_owned_groups(k, c, ratio)
        out.append(owner * c_cap + prefix + (g - first))
    return out


# =============================================================================
# Device geometry
# =============================================================================


@dataclass(frozen=True, eq=False)
class SeqZigzagGeometry:
    """Per-rank, per-microbatch description of the per-sequence zigzag layout.

    All per-block tables are int32 device tensors of shape ``(n_blocks,)`` built from
    ``cu_seqlens`` without host synchronisation (``n_blocks = 2 * (cu_seqlens.numel() - 1)``);
    only shape-derived quantities are Python ints.
    """

    cp_size: int
    cp_rank: int
    l_local: int
    d_window: int
    n_seq: int
    cu_seqlens: torch.Tensor  # (n_seq + 1,) padded pack prefixes (global)
    chunk_rows: torch.Tensor  # (n_seq,)  c_i
    local_cu: torch.Tensor  # (n_seq + 1,) lcu = cu // cp_size
    block_rows: torch.Tensor  # (n_blocks,)
    block_local_offset: torch.Tensor  # (n_blocks,)
    block_global_start: torch.Tensor  # (n_blocks,)
    block_seq: torch.Tensor  # (n_blocks,)
    block_chunk: torch.Tensor  # (n_blocks,)
    # Row tables derived from the fields above, built once per geometry (a layer asks for them
    # several times). Geometries compare by identity (eq=False): the fields are device tensors.
    _memo: Dict[Hashable, torch.Tensor] = field(default_factory=dict, repr=False)

    def _memoized(self, key: Hashable, build: Callable[[], torch.Tensor]) -> torch.Tensor:
        value = self._memo.get(key)
        if value is None:
            value = build()
            self._memo[key] = value
        return value

    @property
    def n_blocks(self) -> int:
        """Number of row blocks (two per packed sequence)."""
        return 2 * self.n_seq

    @property
    def halo_rows(self) -> int:
        """Rows of the halo slot region (``n_blocks * d_window``)."""
        return self.n_blocks * self.d_window

    @property
    def compressed_base(self) -> int:
        """Physical offset of the gathered compressed rows inside ``kv_full_thd``."""
        return self.halo_rows + self.l_local

    def row_blocks(self) -> torch.Tensor:
        """Block id of every local row, ``(l_local,)`` int64 (empty blocks are never selected)."""
        return self._memoized("row_blocks", self._build_row_blocks)

    def _build_row_blocks(self) -> torch.Tensor:
        rows = torch.arange(self.l_local, device=self.cu_seqlens.device, dtype=torch.int32)
        return (
            torch.searchsorted(self.block_local_offset, rows, right=True)
            .sub_(1)
            .clamp_(0, self.n_blocks - 1)
            .to(torch.int64)
        )

    def _target(self, device) -> torch.device:
        return torch.device(device) if device is not None else self.cu_seqlens.device

    def global_rows(self, device=None, dtype=torch.int32) -> torch.Tensor:
        """Pack row of every local row, ``(l_local,)`` (shared tensor: do not modify in place)."""
        device = self._target(device)
        return self._memoized(
            ("global_rows", device, dtype),
            lambda: self._build_global_rows().to(dtype=dtype, device=device),
        )

    def _build_global_rows(self) -> torch.Tensor:
        blk = self.row_blocks()
        rows = torch.arange(self.l_local, device=self.cu_seqlens.device, dtype=torch.int32)
        return self.block_global_start[blk] + (rows - self.block_local_offset[blk])

    def halo_global_rows(self, device=None, dtype=torch.int32) -> torch.Tensor:
        """Pack row of every halo slot row (``d_window`` rows before each block), ``(halo_rows,)``.

        Rows before the pack start are negative (the position helpers map them to 0). Shared
        tensor: do not modify in place.
        """
        device = self._target(device)
        return self._memoized(
            ("halo_global_rows", device, dtype),
            lambda: self._build_halo_global_rows().to(dtype=dtype, device=device),
        )

    def _build_halo_global_rows(self) -> torch.Tensor:
        ar = torch.arange(self.d_window, device=self.cu_seqlens.device, dtype=torch.int32)
        return (self.block_global_start.unsqueeze(1) - self.d_window + ar.unsqueeze(0)).reshape(-1)

    def tail_indices(self, half: int, rows: int) -> torch.Tensor:
        """Local rows of the last ``rows`` rows of every block of ``half`` (0: chunk ``r``, 1: chunk
        ``2*cp-1-r``), ``(n_seq * rows,)`` int64, clamped into range: the entries of an empty
        sequence alias rows of other sequences, so its slots are never read and the exchange
        drops their gradient."""
        ends = self.local_cu[:-1] + (half + 1) * self.chunk_rows  # (n_seq,) end of the block
        ar = torch.arange(rows, device=self.cu_seqlens.device, dtype=torch.int32)
        idx = ends.unsqueeze(1) - rows + ar.unsqueeze(0)
        return idx.reshape(-1).clamp_(0, max(self.l_local - 1, 0)).to(torch.int64)

    def validity(self) -> torch.Tensor:
        """Device scalar: the pack satisfies the preconditions of the layout.

        Every packed length is a multiple of ``2 * cp_size`` (TE's zigzag split), every
        non-empty sequence has ``c_i >= d_window`` (the per-sequence padding floor), and the
        pack starts at row 0 and covers exactly this rank's rows
        (``cu[-1] == cp_size * l_local``)."""
        lengths = self.cu_seqlens[1:] - self.cu_seqlens[:-1]
        floor_ok = (self.chunk_rows == 0) | (self.chunk_rows >= self.d_window)
        split_ok = lengths == self.chunk_rows * (2 * self.cp_size)
        rows_ok = (self.cu_seqlens[0] == 0) & (self.cu_seqlens[-1] == self.cp_size * self.l_local)
        return (floor_ok & split_ok).all() & rows_ok


def make_geometry(
    cp_group: dist.ProcessGroup, cu_seqlens: torch.Tensor, l_local: int, d_window: int
) -> SeqZigzagGeometry:
    """Build this rank's geometry from the (global, padded) ``cu_seqlens`` of the pack."""
    cp_size, cp_rank = cp_group.size(), cp_group.rank()
    if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
        raise ValueError("per-sequence zigzag CP needs a 1-D cu_seqlens with >= 2 entries.")
    cu = cu_seqlens.to(torch.int32)
    n_seq = cu.numel() - 1
    lengths = cu[1:] - cu[:-1]
    chunk_rows = torch.div(lengths, 2 * cp_size, rounding_mode="floor")
    local_cu = torch.div(cu, cp_size, rounding_mode="floor")
    # Device-only arithmetic (no host -> device copies: the tables may be built inside a CUDA
    # graph capture): halves = [0, 1, 0, 1, ...], chunk = r for half 0 and 2*cp-1-r for half 1.
    halves = torch.arange(2 * n_seq, device=cu.device, dtype=torch.int32).remainder_(2)
    block_chunk = halves * (2 * cp_size - 1 - 2 * cp_rank) + cp_rank
    block_rows = chunk_rows.repeat_interleave(2)
    block_seq = torch.arange(n_seq, device=cu.device, dtype=torch.int32).repeat_interleave(2)
    block_local_offset = local_cu[:-1].repeat_interleave(2) + halves * block_rows
    block_global_start = cu[:-1].repeat_interleave(2) + block_chunk * block_rows
    return SeqZigzagGeometry(
        cp_size=cp_size,
        cp_rank=cp_rank,
        l_local=int(l_local),
        d_window=int(d_window),
        n_seq=int(n_seq),
        cu_seqlens=cu,
        chunk_rows=chunk_rows,
        local_cu=local_cu,
        block_rows=block_rows,
        block_local_offset=block_local_offset.contiguous(),
        block_global_start=block_global_start.contiguous(),
        block_seq=block_seq,
        block_chunk=block_chunk,
    )


def _layout_assert_message(geom: SeqZigzagGeometry) -> str:
    """Text of the device-side layout assertion (``torch._assert_async`` caps it at 254 chars)."""
    return (
        "DSv4 per-sequence zigzag CP: every packed sequence must be a multiple of 2*cp_size "
        f"rows and at least 2*cp_size*d_window={2 * geom.cp_size * geom.d_window} rows long "
        "(dsv4_zigzag_min_padded_seq_len); the pack must start at row 0 and fill "
        "cp_size*local rows."
    )


def validate_geometry(geom: SeqZigzagGeometry, host_check: bool) -> None:
    """Check the layout preconditions of ``SeqZigzagGeometry.validity``.

    ``host_check=True`` raises a Python error (one device sync; eager only); otherwise a
    device-side assertion is queued (CUDA-graph safe).
    """
    ok = geom.validity()
    if host_check:
        if not bool(ok.item()):
            raise RuntimeError(
                "per-sequence zigzag CP needs every packed sequence length to be a multiple "
                f"of 2*cp_size = {2 * geom.cp_size}, every non-empty one to be >= "
                f"2*cp_size*d_window = {2 * geom.cp_size * geom.d_window} rows (half-chunk >= "
                f"d_window={geom.d_window}), and the pack to start at row 0 and cover "
                f"cp_size*l_local = {geom.cp_size * geom.l_local} rows; got cu_seqlens "
                f"{geom.cu_seqlens.tolist()}. Enable the per-sequence padding floor "
                "(dsv4_zigzag_min_padded_seq_len) in the data pipeline."
            )
    else:
        torch._assert_async(ok, _layout_assert_message(geom))


# =============================================================================
# Boundary (halo) exchange: one window before each block, two neighbours
# =============================================================================


class _SeqZigzagBoundaryExchange(torch.autograd.Function):
    """Exchange the ``d_window`` rows preceding every block (see the module docstring)."""

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, geom: SeqZigzagGeometry, cp_group: dist.ProcessGroup):
        """Send the block tails to the neighbours and return the ``2 * n_seq`` halo slots."""
        cp_size, cp_rank, d_w, n_seq = geom.cp_size, geom.cp_rank, geom.d_window, geom.n_seq
        ctx.cp_group = cp_group
        ctx.geom = geom
        ctx.input_shape = tensor.shape
        tail0 = geom.tail_indices(0, d_w)  # (n_seq*d_w,) rows my block (i,0) hands to rank r+1
        tail1 = geom.tail_indices(1, d_w)  # (n_seq*d_w,) rows my block (i,1) hands to rank r-1
        ctx.save_for_backward(tail0, tail1)
        boundary = tensor.new_zeros((2 * n_seq, d_w) + tuple(tensor.shape[1:]))
        send0 = tensor.index_select(0, tail0).view((n_seq, d_w) + tuple(tensor.shape[1:]))
        send1 = tensor.index_select(0, tail1).view((n_seq, d_w) + tuple(tensor.shape[1:]))
        recv0 = torch.empty_like(send0)  # halo of my blocks (i,0) from rank r-1
        recv1 = torch.empty_like(send1)  # halo of my blocks (i,1) from rank r+1
        ops = []
        if cp_rank > 0:
            prev = dist.get_global_rank(cp_group, cp_rank - 1)
            ops.append(dist.P2POp(dist.irecv, recv0, prev, cp_group))
            ops.append(dist.P2POp(dist.isend, send1, prev, cp_group))
        if cp_rank + 1 < cp_size:
            nxt = dist.get_global_rank(cp_group, cp_rank + 1)
            ops.append(dist.P2POp(dist.irecv, recv1, nxt, cp_group))
            ops.append(dist.P2POp(dist.isend, send0, nxt, cp_group))
        if ops:
            for req in dist.batch_isend_irecv(ops):
                req.wait()
        if cp_rank > 0:
            boundary[0::2] = recv0
        if cp_rank + 1 < cp_size:
            boundary[1::2] = recv1
        else:
            # rank cp-1: block (i,1) is chunk cp, preceded by chunk cp-1 = own block (i,0)
            boundary[1::2] = send0
        return boundary.view((2 * n_seq * d_w,) + tuple(tensor.shape[1:]))

    @staticmethod
    def backward(ctx, grad_boundary: torch.Tensor):
        """Return the halo-slot gradients to the block tails they were copied from."""
        cp_group, geom = ctx.cp_group, ctx.geom
        cp_size, cp_rank, d_w, n_seq = geom.cp_size, geom.cp_rank, geom.d_window, geom.n_seq
        tail0, tail1 = ctx.saved_tensors
        grad_boundary = grad_boundary.contiguous().view(
            (2 * n_seq, d_w) + tuple(grad_boundary.shape[1:])
        )
        grad_slot0 = grad_boundary[0::2].contiguous()  # -> rank r-1's block (i,0) tails
        grad_slot1 = grad_boundary[1::2].contiguous()  # -> rank r+1's block (i,1) tails
        recv_tail0 = torch.empty_like(grad_slot0)  # grad of my block (i,0) tails, from rank r+1
        recv_tail1 = torch.empty_like(grad_slot1)  # grad of my block (i,1) tails, from rank r-1
        ops = []
        if cp_rank > 0:
            prev = dist.get_global_rank(cp_group, cp_rank - 1)
            ops.append(dist.P2POp(dist.isend, grad_slot0, prev, cp_group))
            ops.append(dist.P2POp(dist.irecv, recv_tail1, prev, cp_group))
        if cp_rank + 1 < cp_size:
            nxt = dist.get_global_rank(cp_group, cp_rank + 1)
            ops.append(dist.P2POp(dist.isend, grad_slot1, nxt, cp_group))
            ops.append(dist.P2POp(dist.irecv, recv_tail0, nxt, cp_group))
        if ops:
            for req in dist.batch_isend_irecv(ops):
                req.wait()
        grad_input = grad_boundary.new_zeros(ctx.input_shape)
        flat_shape = (n_seq * d_w,) + tuple(grad_boundary.shape[2:])
        # The tail indices of an empty sequence are clamped onto rows of other sequences. No
        # consumer addresses its slots, so their gradient is zero; drop it explicitly so that a
        # stray value can never be added to those rows (and every destination row receives at
        # most one non-zero contribution per scatter, which keeps index_add_ order independent).
        empty = (geom.chunk_rows == 0).view((n_seq,) + (1,) * (grad_boundary.dim() - 1))
        tail0_grad = recv_tail0 if cp_rank + 1 < cp_size else grad_slot1
        grad_input.index_add_(0, tail0, tail0_grad.masked_fill(empty, 0).view(flat_shape))
        if cp_rank > 0:
            grad_input.index_add_(0, tail1, recv_tail1.masked_fill(empty, 0).view(flat_shape))
        return grad_input, None, None


def exchange_cp_boundary_hidden_seq_zigzag(
    hidden_states: torch.Tensor, geom: SeqZigzagGeometry, cp_group: dist.ProcessGroup
) -> torch.Tensor:
    """Return ``(n_blocks * d_window, ...)``: the window before block 0, block 1, ..."""
    hidden_flat = hidden_states.reshape(hidden_states.shape[0], -1)
    boundary = _SeqZigzagBoundaryExchange.apply(hidden_flat, geom, cp_group)
    return boundary.reshape((geom.halo_rows,) + tuple(hidden_states.shape[1:]))


def compressor_boundary_rows(
    boundary_hidden: torch.Tensor, geom: SeqZigzagGeometry, d_comp: int
) -> torch.Tensor:
    """The ``d_comp`` rows preceding each block, ``(n_blocks * d_comp, ...)``: the tail of every
    halo slot (``Compressor._forward_thd_cp`` slices ``boundary[-boundary_rows:]`` and the block
    layout sets ``boundary_rows == n_blocks * d_comp``)."""
    d_w = geom.d_window
    if boundary_hidden.shape[0] != geom.halo_rows:
        raise RuntimeError(
            f"per-sequence zigzag compressor boundary expects {geom.halo_rows} rows, "
            f"got {boundary_hidden.shape[0]}."
        )
    if d_comp > d_w:
        raise RuntimeError("compressor halo exceeds the boundary window.")
    slots = boundary_hidden.view((geom.n_blocks, d_w) + tuple(boundary_hidden.shape[1:]))
    return slots[:, d_w - d_comp :].reshape(
        (geom.n_blocks * d_comp,) + tuple(boundary_hidden.shape[1:])
    )


# =============================================================================
# Compressor: block segments sharing one CPCompressorLayout
# =============================================================================


def build_cp_compressor_layout_seq_zigzag(
    cu_seqlens: torch.Tensor, geom: SeqZigzagGeometry, ratio: int, c_cap: Optional[int] = None
) -> thd_layout_kernels.CPCompressorLayout:
    """Block-segment ``CPCompressorLayout`` for this rank (see ``ref_compressor_layout``)."""
    ratio = int(ratio)
    if c_cap is None:
        c_cap = compact_group_capacity_blocks(geom.l_local, geom.n_seq, ratio)
    layout = thd_layout_kernels.build_cp_compressor_layout_blocks(
        cu_seqlens,
        geom.block_rows,
        geom.block_local_offset,
        geom.block_global_start,
        geom.block_seq,
        geom.chunk_rows,
        geom.l_local,
        geom.cp_size,
        ratio,
        int(c_cap),
    )
    return layout


def prepare_cp_compressor_input_seq_zigzag(
    hidden_local: torch.Tensor,
    compressor_boundary: torch.Tensor,
    layout: thd_layout_kernels.CPCompressorLayout,
) -> torch.Tensor:
    """``hidden_compact`` for the single-projection fallback: gather the conceptual source buffer
    ``[compressor_boundary | hidden_local | zero row]`` through the layout's ``compact_to_source``
    map; padding rows (``-1``) select the zero row, so one gather follows the concatenation."""
    zero_row = hidden_local.new_zeros((1,) + tuple(hidden_local.shape[1:]))
    source = torch.cat((compressor_boundary, hidden_local, zero_row), dim=0)
    idx = layout.compact_to_source.to(torch.int64)
    idx = idx.masked_fill(idx < 0, source.shape[0] - 1)
    return source.index_select(0, idx)


# =============================================================================
# Indexer: one fused call over the blocks against per-block K copies
# =============================================================================


def build_cp_indexer_layout_seq_zigzag(
    cu_seqlens_compressed: torch.Tensor, geom: SeqZigzagGeometry, ratio: int, seq_major_rows: int
) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor, int]:
    """Compact THD indexer layout over the blocks.

    Returns ``((cu_q, cu_k, q_causal_offsets), source_row_map, total_k_rows)``: ``cu_q`` has one
    segment per block plus the synthetic padding segment, ``cu_k`` the per-block K copies (each
    the block's causally visible prefix of its sequence's compressed rows, plus the compact
    wrapper's guard row; the synthetic segment absorbs the remaining fixed capacity), and
    ``source_row_map`` gathers that K buffer from the sequence-major K (``>= seq_major_rows``
    selects the appended zero row, as ``pack_thd_compact_k`` expects).
    """
    ratio = int(ratio)
    dev = geom.cu_seqlens.device
    n_blocks = geom.n_blocks
    cuc = cu_seqlens_compressed.to(torch.int32)
    total_k_rows = 2 * int(seq_major_rows) + n_blocks + 1
    zero = torch.zeros((1,), dtype=torch.int32, device=dev)
    # Q: the blocks in local order, then the (empty) synthetic segment.
    l_local = torch.full((1,), geom.l_local, dtype=torch.int32, device=dev)
    cu_q = torch.cat((geom.block_local_offset, l_local, l_local))
    # K per block: visible prefix of the sequence's compressed rows.
    seq_groups = (cuc[1:] - cuc[:-1])[geom.block_seq.long()]
    visible = torch.minimum(
        seq_groups,
        torch.div((geom.block_chunk + 1) * geom.block_rows, ratio, rounding_mode="floor"),
    ).clamp_min(0)
    kernel_lens = visible + 1  # one invisible guard row per real segment
    kernel_prefix = torch.cumsum(kernel_lens, dim=0, dtype=torch.int32)
    tail = torch.full((1,), total_k_rows, dtype=torch.int32, device=dev)
    cu_k = torch.cat((zero, kernel_prefix, tail))
    q_causal_offsets = torch.cat((geom.block_chunk * geom.block_rows, zero))
    # source rows of the K copies
    kernel_rows = torch.arange(total_k_rows, dtype=torch.int32, device=dev)
    seg = torch.searchsorted(cu_k[1:], kernel_rows, right=True).clamp_max(n_blocks).to(torch.int64)
    safe = seg.clamp_max(n_blocks - 1)
    pos = kernel_rows - cu_k[seg]
    valid = (seg < n_blocks) & (pos < visible[safe])
    src = cuc[:-1][geom.block_seq.long()][safe] + pos
    source_row_map = torch.where(valid, src, torch.full_like(src, int(seq_major_rows))).to(
        torch.int64
    )
    return (
        (cu_q.contiguous(), cu_k.contiguous(), q_causal_offsets.contiguous()),
        source_row_map,
        total_k_rows,
    )


# =============================================================================
# Final attention indices
# =============================================================================


def build_attention_indices_seq_zigzag(
    cu_seqlens: torch.Tensor,
    geom: SeqZigzagGeometry,
    window_size: int,
    ratio: int,
    compressed_width: int,
    compressed_topk: Optional[torch.Tensor] = None,
    cu_seqlens_compressed: Optional[torch.Tensor] = None,
    seq_to_rank_row: Optional[torch.Tensor] = None,
    for_indexer_loss: bool = False,
    compressed_rows: Optional[int] = None,
    cu_seqlens_unpadded: Optional[torch.Tensor] = None,
    output_alignment: int = 1,
):
    """Block-layout version of ``thd_layout_kernels.build_attention_indices`` (same outputs)."""
    return thd_layout_kernels.build_attention_indices(
        cu_seqlens,
        0,
        geom.l_local,
        geom.d_window,
        window_size,
        ratio,
        compressed_width,
        compressed_topk,
        cu_seqlens_compressed=cu_seqlens_compressed,
        seq_to_rank_row=seq_to_rank_row,
        for_indexer_loss=for_indexer_loss,
        compressed_base=geom.compressed_base,
        compressed_rows=compressed_rows,
        cu_seqlens_unpadded=cu_seqlens_unpadded,
        output_alignment=output_alignment,
        block_tables=(geom.block_local_offset, geom.block_global_start, geom.block_rows),
    )
