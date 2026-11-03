# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Pack-level zigzag ("two contiguous segments per rank") context parallelism for DSv4 CSA.

Motivation
----------
Under the contiguous CP split, rank ``r`` owns pack rows ``[r*l, (r+1)*l)``. Every causal,
prefix-proportional kernel of the DSv4 attention stack (indexer scoring / top-k in ratio-4 layers,
"attend to all compressed rows" in ratio-128 layers) therefore costs ``2*cp_size - 1`` times more
on the last CP rank than on the first one, and every other rank waits for it (MOK's per-layer EP
barrier, PP p2p, the DP grad reduce-scatter).

Layout
------
With ``c = l_local // 2`` and ``2*cp_size`` pack chunks of ``c`` rows, rank ``r`` owns chunk ``r``
(segment 0, light) and chunk ``2*cp_size - 1 - r`` (segment 1, heavy):

    local rows [0, c)      <-> pack rows [r*c, (r+1)*c)
    local rows [c, 2c)     <-> pack rows [(2*cp_size-1-r)*c, (2*cp_size-r)*c)

Each segment is exactly what a contiguous rank of a *virtual* ``2*cp_size`` CP world would hold,
so every contiguous mechanism of the CSA CP path (halo exchange, compressor group ownership,
rank-major compressed all-gather, index building, synthetic indexer layouts) is reused per
segment. Unlike a per-sequence zigzag this needs no per-sequence divisibility, no per-pack
metadata (geometry is static -> CUDA-graph friendly) and keeps the halo volume at two windows
per rank. The decoder input is converted contiguous -> pack-zigzag once per microbatch and the
decoder output back (point-to-point exchanges with at most two peers each way); data, MTP, LM
head and loss stay contiguous.

Physical KV layout seen by the sparse kernels on a rank::

    kv_full_thd = cat(halo0, local[:c], halo1, local[c:], gathered_compressed)
    block b base   = b * (d_window + c)
    compressed base = 2 * (d_window + c)

Compressor layout: the two segments are compacted as two blocks of ``c_cap`` groups each
(``c_cap = compact_group_capacity(c, ratio)``); rank ``r``'s all-gather slot therefore holds
``[block 0 groups | block 1 groups]`` and a sequence-major compressed row maps to
``owner_rank * 2*c_cap + block * c_cap + slot``. The per-block row maps come from the ordinary
contiguous builders evaluated in the virtual ``2*cp_size`` world and are re-based here.

Ported 2026-10-17 from the September prototype (tree Megatron-LM-zzcp-0917) onto the October
tree, whose compressor projects first and compacts the narrow KV/gate with
``thd_layout_kernels.CPCompressorLayout`` row maps.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.distributed as dist

from . import thd_layout_kernels

# =============================================================================
# Geometry
# =============================================================================


@dataclass(frozen=True)
class PackZigzagGeometry:
    """Static per-rank description of the pack-zigzag layout."""

    cp_size: int
    cp_rank: int
    l_local: int
    d_window: int

    def __post_init__(self):
        if self.l_local % 2 != 0:
            raise ValueError(
                f"pack-zigzag CP needs an even per-rank row count, got l_local={self.l_local}."
            )
        if self.chunk < self.d_window:
            raise RuntimeError(
                "pack-zigzag CP needs each half-chunk to cover the boundary window: "
                f"l_local/2={self.chunk} < D_window={self.d_window}."
            )

    @property
    def chunk(self) -> int:
        """Rows per segment (``c``)."""
        return self.l_local // 2

    @property
    def virtual_ranks(self) -> Tuple[int, int]:
        """Contiguous rank ids of the two segments in the virtual ``2*cp_size`` world."""
        return (self.cp_rank, 2 * self.cp_size - 1 - self.cp_rank)

    @property
    def seg_global_starts(self) -> Tuple[int, int]:
        """First pack row of each segment."""
        return tuple(v * self.chunk for v in self.virtual_ranks)

    @property
    def seg_local_offsets(self) -> Tuple[int, int]:
        """First local row of each segment."""
        return (0, self.chunk)

    def block_index_base(self, block: int) -> int:
        """Physical offset of block ``block``'s (halo, local) pair inside ``kv_full_thd``."""
        return block * (self.d_window + self.chunk)

    @property
    def compressed_base(self) -> int:
        """Physical offset of the gathered compressed rows inside ``kv_full_thd``."""
        return 2 * (self.d_window + self.chunk)

    def global_rows(self, device, dtype=torch.int32) -> torch.Tensor:
        """Pack row of every local row, shape ``(l_local,)``."""
        c = self.chunk
        ar = torch.arange(c, device=device, dtype=dtype)
        s0, s1 = self.seg_global_starts
        return torch.cat((ar + s0, ar + s1))

    def halo_global_rows(self, device, dtype=torch.int32) -> torch.Tensor:
        """Pack row of every halo row (``d_window`` rows before each segment), ``(2*d_window,)``.

        Rows before the pack start (rank 0, segment 0) are negative; the position helpers map
        them to position 0 and the halo tensor holds zeros there, as on the contiguous path.
        """
        ar = torch.arange(self.d_window, device=device, dtype=dtype)
        s0, s1 = self.seg_global_starts
        return torch.cat((ar + (s0 - self.d_window), ar + (s1 - self.d_window)))


def make_geometry(cp_group: dist.ProcessGroup, l_local: int, d_window: int) -> PackZigzagGeometry:
    """Build the geometry for this rank."""
    return PackZigzagGeometry(
        cp_size=cp_group.size(),
        cp_rank=cp_group.rank(),
        l_local=int(l_local),
        d_window=int(d_window),
    )


def virtual_owner_of_chunk(chunk: int, cp_size: int) -> int:
    """Real CP rank owning pack chunk ``chunk`` (``0 <= chunk < 2*cp_size``)."""
    return chunk if chunk < cp_size else 2 * cp_size - 1 - chunk


# =============================================================================
# Layout redistribution (contiguous <-> pack-zigzag): point-to-point, two peers each way
# =============================================================================


def _a2a_plan(cp_size: int, cp_rank: int, c: int, to_zigzag: bool):
    """Return (input_split_sizes, output_split_sizes, swap_send_halves, swap_recv_halves).

    Contiguous rank r holds pack chunks (2r, 2r+1) as its two local halves; zigzag rank r
    holds chunks (r, 2*cp_size-1-r). ``all_to_all_single`` orders the send buffer by
    destination rank and the receive buffer by source rank, so a half swap is needed when the
    natural chunk order disagrees with the rank order.
    """
    z = lambda k: virtual_owner_of_chunk(k, cp_size)
    in_splits = [0] * cp_size
    out_splits = [0] * cp_size
    if to_zigzag:
        d0, d1 = z(2 * cp_rank), z(2 * cp_rank + 1)
        s0, s1 = cp_rank // 2, (2 * cp_size - 1 - cp_rank) // 2
        in_splits[d0] += c
        in_splits[d1] += c
        out_splits[s0] += c
        out_splits[s1] += c
        return in_splits, out_splits, d0 > d1, s0 > s1
    d0, d1 = cp_rank // 2, (2 * cp_size - 1 - cp_rank) // 2
    s0, s1 = z(2 * cp_rank), z(2 * cp_rank + 1)
    in_splits[d0] += c
    in_splits[d1] += c
    out_splits[s0] += c
    out_splits[s1] += c
    return in_splits, out_splits, d0 > d1, s0 > s1


def _p2p_views(flat: torch.Tensor, out: torch.Tensor, c: int, rank: int, plan):
    """Per-peer (view, peer) send/recv lists realising the plan with point-to-point ops only.

    Every rank exchanges whole halves with at most two peers in each direction, so the layout
    change is a handful of send/recv pairs, not an all-to-all. Going through ``all_to_all`` /
    ``all_to_all_single`` costs two things we measured: a materialised half swap on the ranks
    whose chunk order disagrees with the rank order (512 MiB for the mHC multistream at 16K
    tokens/rank), and - because the NCCL backend posts zero-byte send/recv to *every* peer -
    a P2P connection (plus its buffers, ~0.33 GiB each on NVL72) to all CP peers: +5 GiB at
    CP16, +10 GiB at CP32 of device memory outside the allocator. Explicit P2P to the real
    peers avoids both. Chunks that stay on this rank are returned as ``self_pair`` (a copy).
    Returns None when the plan is not the usual "two halves to two peers" shape.
    """
    in_splits, out_splits, swap_send, swap_recv = plan
    if any(n not in (0, c) for n in in_splits) or any(n not in (0, c) for n in out_splits):
        return None
    send_halves = (flat[c:], flat[:c]) if swap_send else (flat[:c], flat[c:])
    recv_halves = (out[c:], out[:c]) if swap_recv else (out[:c], out[c:])
    sends, recvs, self_pair = [], [], None
    i = j = 0
    self_send = self_recv = None
    for r in range(len(in_splits)):
        if in_splits[r]:
            if r == rank:
                self_send = send_halves[i]
            else:
                sends.append((send_halves[i], r))
            i += 1
        if out_splits[r]:
            if r == rank:
                self_recv = recv_halves[j]
            else:
                recvs.append((recv_halves[j], r))
            j += 1
    if i != 2 or j != 2 or (self_send is None) != (self_recv is None):
        return None
    if self_send is not None:
        self_pair = (self_recv, self_send)
    return sends, recvs, self_pair


def _redistribute_rows(
    x: torch.Tensor, cp_group: dist.ProcessGroup, to_zigzag: bool
) -> torch.Tensor:
    """Permute dim 0 (tokens) of ``x`` between the contiguous and the pack-zigzag layout."""
    cp_size = cp_group.size()
    if cp_size == 1:
        return x
    s = x.shape[0]
    if s % 2 != 0:
        raise ValueError(f"pack-zigzag redistribution needs an even row count, got {s}.")
    c = s // 2
    rank = cp_group.rank()
    plan = _a2a_plan(cp_size, rank, c, to_zigzag)
    flat = x.reshape(s, -1)
    if not flat.is_contiguous():
        flat = flat.contiguous()
    out = torch.empty_like(flat)
    views = _p2p_views(flat, out, c, rank, plan)
    if views is not None:
        sends, recvs, self_pair = views
        ops = [
            dist.P2POp(dist.isend, t, dist.get_global_rank(cp_group, peer), group=cp_group)
            for t, peer in sends
        ] + [
            dist.P2POp(dist.irecv, t, dist.get_global_rank(cp_group, peer), group=cp_group)
            for t, peer in recvs
        ]
        if self_pair is not None:
            self_pair[0].copy_(self_pair[1])
        if ops:
            for req in dist.batch_isend_irecv(ops):
                req.wait()
        return out.reshape(x.shape)
    # Generic (copying) fallback.
    in_splits, out_splits, swap_send, swap_recv = plan
    send = torch.cat((flat[c:], flat[:c])) if swap_send else flat
    dist.all_to_all_single(out, send, out_splits, in_splits, group=cp_group)
    if swap_recv:
        out = torch.cat((out[c:], out[:c]))
    return out.reshape(x.shape)


class _PackZigzagRedistribute(torch.autograd.Function):
    """Differentiable layout permutation; the backward is the inverse permutation."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, cp_group: dist.ProcessGroup, to_zigzag: bool):
        ctx.cp_group = cp_group
        ctx.to_zigzag = to_zigzag
        return _redistribute_rows(x, cp_group, to_zigzag)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        return _redistribute_rows(grad.contiguous(), ctx.cp_group, not ctx.to_zigzag), None, None


def redistribute_pack_zigzag(
    x: torch.Tensor, cp_group: dist.ProcessGroup, to_zigzag: bool
) -> torch.Tensor:
    """Differentiable contiguous <-> pack-zigzag conversion of a ``[tokens, ...]`` tensor."""
    return _PackZigzagRedistribute.apply(x, cp_group, to_zigzag)


def redistribute_tokens_no_grad(
    x: torch.Tensor, cp_group: dist.ProcessGroup, to_zigzag: bool, token_dim: int = 0
) -> torch.Tensor:
    """Non-differentiable conversion for side inputs (input_ids, padding_mask, ...)."""
    if cp_group.size() == 1:
        return x
    with torch.no_grad():
        moved = x.movedim(token_dim, 0)
        orig_dtype = moved.dtype
        if orig_dtype == torch.bool:
            moved = moved.to(torch.uint8)
        out = _redistribute_rows(moved.contiguous(), cp_group, to_zigzag)
        if orig_dtype == torch.bool:
            out = out.to(torch.bool)
        return out.movedim(0, token_dim).contiguous()


# =============================================================================
# Boundary (halo) exchange: one window before each of the two segments
# =============================================================================


class _TwoSegmentBoundaryExchange(torch.autograd.Function):
    """Exchange the ``d_window`` rows preceding each segment.

    segment 0 (chunk r)                <- last rows of chunk r-1   = rank r-1's segment 0
    segment 1 (chunk 2*cp_size-1-r)    <- last rows of chunk 2*cp_size-2-r
                                          = rank r+1's segment 1 (r < cp_size-1)
                                          = own segment 0        (r == cp_size-1)
    """

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, geom: PackZigzagGeometry, cp_group: dist.ProcessGroup):
        cp_size, cp_rank, d_w, c = geom.cp_size, geom.cp_rank, geom.d_window, geom.chunk
        ctx.cp_group = cp_group
        ctx.geom = geom
        ctx.input_shape = tensor.shape
        boundary = tensor.new_zeros((2 * d_w,) + tuple(tensor.shape[1:]))
        seg0_tail = tensor[c - d_w : c].contiguous()
        seg1_tail = tensor[-d_w:].contiguous()
        ops = []
        if cp_rank > 0:
            ops.append(
                dist.P2POp(
                    dist.irecv,
                    boundary[:d_w],
                    dist.get_global_rank(cp_group, cp_rank - 1),
                    cp_group,
                )
            )
            # our segment-1 tail precedes rank r-1's segment 1
            ops.append(
                dist.P2POp(
                    dist.isend, seg1_tail, dist.get_global_rank(cp_group, cp_rank - 1), cp_group
                )
            )
        if cp_rank + 1 < cp_size:
            ops.append(
                dist.P2POp(
                    dist.irecv,
                    boundary[d_w:],
                    dist.get_global_rank(cp_group, cp_rank + 1),
                    cp_group,
                )
            )
            # our segment-0 tail precedes rank r+1's segment 0
            ops.append(
                dist.P2POp(
                    dist.isend, seg0_tail, dist.get_global_rank(cp_group, cp_rank + 1), cp_group
                )
            )
        else:
            boundary[d_w:] = seg0_tail
        if ops:
            for req in dist.batch_isend_irecv(ops):
                req.wait()
        return boundary

    @staticmethod
    def backward(ctx, grad_boundary: torch.Tensor):
        cp_group, geom = ctx.cp_group, ctx.geom
        cp_size, cp_rank, d_w, c = geom.cp_size, geom.cp_rank, geom.d_window, geom.chunk
        grad_boundary = grad_boundary.contiguous()
        grad_input = grad_boundary.new_zeros(ctx.input_shape)
        recv_seg0 = grad_boundary.new_empty((d_w,) + tuple(grad_boundary.shape[1:]))
        recv_seg1 = grad_boundary.new_empty((d_w,) + tuple(grad_boundary.shape[1:]))
        ops = []
        if cp_rank > 0:
            # halo0 was rank r-1's segment-0 tail -> its gradient goes back to r-1; our
            # segment-1 tail was rank r-1's halo1 -> that gradient comes back from r-1.
            ops.append(
                dist.P2POp(
                    dist.isend,
                    grad_boundary[:d_w].contiguous(),
                    dist.get_global_rank(cp_group, cp_rank - 1),
                    cp_group,
                )
            )
            ops.append(
                dist.P2POp(
                    dist.irecv, recv_seg1, dist.get_global_rank(cp_group, cp_rank - 1), cp_group
                )
            )
        if cp_rank + 1 < cp_size:
            ops.append(
                dist.P2POp(
                    dist.isend,
                    grad_boundary[d_w:].contiguous(),
                    dist.get_global_rank(cp_group, cp_rank + 1),
                    cp_group,
                )
            )
            ops.append(
                dist.P2POp(
                    dist.irecv, recv_seg0, dist.get_global_rank(cp_group, cp_rank + 1), cp_group
                )
            )
        if ops:
            for req in dist.batch_isend_irecv(ops):
                req.wait()
        if cp_rank + 1 < cp_size:
            grad_input[c - d_w : c] = recv_seg0
        else:
            # rank cp_size-1: halo1 was a local copy of the own segment-0 tail
            grad_input[c - d_w : c] = grad_boundary[d_w:]
        if cp_rank > 0:
            grad_input[-d_w:] = recv_seg1
        return grad_input, None, None


def exchange_cp_boundary_hidden_pack_zigzag(
    hidden_states: torch.Tensor, geom: PackZigzagGeometry, cp_group: dist.ProcessGroup
) -> torch.Tensor:
    """Return ``(2*d_window, ...)``: the window before segment 0, then the one before segment 1."""
    hidden_flat = hidden_states.view(hidden_states.shape[0], -1)
    boundary = _TwoSegmentBoundaryExchange.apply(hidden_flat, geom, cp_group)
    return boundary.reshape((2 * geom.d_window,) + tuple(hidden_states.shape[1:]))


# =============================================================================
# Compressor: two compaction blocks sharing one CPCompressorLayout
# =============================================================================


def compact_group_capacity(rows: int, ratio: int) -> int:
    """Fixed compaction capacity (groups) for a ``rows``-token contiguous block."""
    d_comp = 8 if ratio == 4 else ratio
    group_alignment = 32 // math.gcd(32, ratio)
    c_cap = max(1, (rows + d_comp) // ratio)
    return ((c_cap + group_alignment - 1) // group_alignment) * group_alignment


def compressor_d_comp(ratio: int) -> int:
    """Compressor halo rows for ``ratio`` (matches the layout builders)."""
    return 8 if ratio == 4 else ratio


def compressor_boundary_rows(
    boundary_hidden: torch.Tensor, geom: PackZigzagGeometry, d_comp: int
) -> torch.Tensor:
    """The ``d_comp`` rows preceding each segment, ``(2*d_comp, ...)``, from the two halo windows.

    ``Compressor._forward_thd_cp`` slices ``boundary_hidden[-layout.boundary_rows:]``; with the
    two-block layout ``boundary_rows == 2*d_comp`` and this tensor is exactly that slice.
    """
    d_w = geom.d_window
    if boundary_hidden.shape[0] != 2 * d_w:
        raise RuntimeError(
            f"pack-zigzag compressor boundary expects 2*d_window={2*d_w} rows, "
            f"got {boundary_hidden.shape[0]}."
        )
    if d_comp > d_w:
        raise RuntimeError("compressor halo exceeds the boundary window.")
    return torch.cat(
        (boundary_hidden[d_w - d_comp : d_w], boundary_hidden[2 * d_w - d_comp :]), dim=0
    )


def remap_virtual_seq_to_rank_row(
    virtual_rows: torch.Tensor, cp_size: int, c_cap: int
) -> torch.Tensor:
    """Map ``virtual_rank * c_cap + slot`` rows of the ``2*cp_size`` world to the physical
    rank-major buffer ``real_rank * 2*c_cap + block * c_cap + slot`` (``-1`` is kept)."""
    valid = virtual_rows >= 0
    v = torch.div(virtual_rows.clamp_min(0), c_cap, rounding_mode="floor")
    slot = virtual_rows.clamp_min(0) - v * c_cap
    is_tail = v >= cp_size
    real = torch.where(is_tail, 2 * cp_size - 1 - v, v)
    phys = real * (2 * c_cap) + is_tail.to(virtual_rows.dtype) * c_cap + slot
    return torch.where(valid, phys, virtual_rows).to(torch.int32)


def build_cp_compressor_layout_pack_zigzag(
    cu_seqlens: torch.Tensor, geom: PackZigzagGeometry, ratio: int
) -> thd_layout_kernels.CPCompressorLayout:
    """Two-block ``CPCompressorLayout`` for this rank's two segments.

    Each block is the contiguous layout of the matching virtual rank (``2*cp_size`` world,
    ``c`` rows). The maps are re-based onto the conceptual source buffer
    ``[boundary0[-d_comp:], boundary1[-d_comp:], local[:c], local[c:]]`` (``boundary_rows ==
    2*d_comp``) and onto the concatenated compact buffer ``[block 0 | block 1]``. The local
    pre-grouped THD prefixes describe that buffer as ``2*n_seq + 1`` segments: block 0's
    segments, one segment covering block 0's static padding tail (its rows are zero-filled by
    the compaction kernels, so they pool to finite values, and it keeps block 1's first segment
    a fresh segment start for the overlap window), then block 1's segments offset by
    ``compact_len`` / ``c_cap``. Block 1's padding tail stays outside all segments, exactly like
    the contiguous layout's tail, so ``Compressor._forward_thd`` hands the buffer to the fused
    compressor kernel unchanged (the unfused region remains the fallback).
    """
    ratio = int(ratio)
    c = geom.chunk
    cp = geom.cp_size
    nv = 2 * cp
    d_comp = compressor_d_comp(ratio)
    c_cap = compact_group_capacity(c, ratio)
    compact_len = c_cap * ratio
    lay = [
        thd_layout_kernels.build_cp_compressor_layout(cu_seqlens, s, c, nv, ratio)
        for s in geom.seg_global_starts
    ]
    for b, l in enumerate(lay):
        if l.compact_to_source.numel() != compact_len or l.source_to_compact.numel() != c + d_comp:
            raise RuntimeError(f"unexpected per-block compressor layout sizes for block {b}.")
        if l.boundary_rows != d_comp:
            raise RuntimeError("per-block compressor halo does not match d_comp.")

    def rebase_c2s(t: torch.Tensor, b: int) -> torch.Tensor:
        is_bnd = (t >= 0) & (t < d_comp)
        is_loc = t >= d_comp
        return torch.where(
            is_bnd, t + b * d_comp, torch.where(is_loc, t - d_comp + 2 * d_comp + b * c, t)
        )

    compact_to_source = torch.cat(
        (rebase_c2s(lay[0].compact_to_source, 0), rebase_c2s(lay[1].compact_to_source, 1))
    )
    s2c0 = lay[0].source_to_compact
    s2c1 = lay[1].source_to_compact
    s2c1 = torch.where(s2c1 >= 0, s2c1 + compact_len, s2c1)
    source_to_compact = torch.cat((s2c0[:d_comp], s2c1[:d_comp], s2c0[d_comp:], s2c1[d_comp:]))
    group_ids = torch.cat((lay[0].group_ids, lay[1].group_ids))
    position_ids = torch.cat((lay[0].position_ids, lay[1].position_ids))
    seq_to_rank_row = remap_virtual_seq_to_rank_row(lay[0].seq_to_rank_row, cp, c_cap)
    # THD prefixes over the concatenated compact buffer (see the docstring): block 0's segments,
    # block 0's padding tail as one segment, block 1's segments shifted by one block.
    dev = cu_seqlens.device
    pad_cu = torch.full((1,), compact_len, dtype=torch.int32, device=dev)
    pad_cuc = torch.full((1,), c_cap, dtype=torch.int32, device=dev)
    local_cu = torch.cat(
        (lay[0].local_cu_seqlens, pad_cu, lay[1].local_cu_seqlens[1:] + compact_len)
    )
    local_cuc = torch.cat(
        (
            lay[0].local_cu_seqlens_compressed,
            pad_cuc,
            lay[1].local_cu_seqlens_compressed[1:] + c_cap,
        )
    )
    return thd_layout_kernels.CPCompressorLayout(
        compact_to_source.contiguous(),
        source_to_compact.contiguous(),
        group_ids.contiguous(),
        position_ids.contiguous(),
        local_cu.contiguous(),
        local_cuc.contiguous(),
        lay[0].cu_seqlens_compressed,
        seq_to_rank_row.contiguous(),
        2 * d_comp,
        int(geom.seg_global_starts[0]),
        int(cp),
    )


def prepare_cp_compressor_input_pack_zigzag(
    hidden_local: torch.Tensor,
    boundary_hidden: torch.Tensor,
    cu_seqlens: torch.Tensor,
    geom: PackZigzagGeometry,
    ratio: int,
) -> torch.Tensor:
    """Two-block ``hidden_compact`` ``(2 * c_cap * ratio, ...)`` for the single-projection fallback
    (block 0 then block 1, each the contiguous compaction of its virtual rank)."""
    ratio = int(ratio)
    d_comp = compressor_d_comp(ratio)
    c = geom.chunk
    c_cap = compact_group_capacity(c, ratio)
    d_w = geom.d_window
    if boundary_hidden.shape[0] != 2 * d_w:
        raise RuntimeError(
            f"pack-zigzag compressor input expects 2*d_window={2*d_w} boundary rows, "
            f"got {boundary_hidden.shape[0]}."
        )
    compacts = []
    for b, (g_start, l_off) in enumerate(zip(geom.seg_global_starts, geom.seg_local_offsets)):
        hc, *_ = thd_layout_kernels.CompressorInputCompact.apply(
            hidden_local[l_off : l_off + c],
            boundary_hidden[b * d_w : (b + 1) * d_w],
            cu_seqlens,
            g_start,
            ratio,
            d_comp,
            c_cap,
            2 * geom.cp_size,
        )
        compacts.append(hc)
    return torch.cat(compacts, dim=0)


# =============================================================================
# Final attention indices: one kernel launch per block into shared output tensors
# =============================================================================


def _aligned(t: Optional[torch.Tensor]) -> bool:
    # the CuTe wrapper maps tensors with assumed_align=16: a row slice must start on a
    # 16-byte boundary to be passed directly; otherwise lower into a temporary and copy.
    return t is None or (t.data_ptr() % 16 == 0)


def build_attention_indices_pack_zigzag(
    cu_seqlens: torch.Tensor,
    geom: PackZigzagGeometry,
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
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Two-block version of ``thd_layout_kernels.build_attention_indices`` (same outputs)."""
    c = geom.chunk
    dev = cu_seqlens.device
    logical_width = int(window_size) + int(compressed_width)
    output_alignment = int(output_alignment)
    total_width = (
        (logical_width + output_alignment - 1) // output_alignment * output_alignment
        if logical_width > 0
        else 0
    )
    topk_idxs = torch.empty((geom.l_local, total_width), dtype=torch.int32, device=dev)
    topk_length = (
        None if for_indexer_loss else torch.empty((geom.l_local,), dtype=torch.int32, device=dev)
    )
    indexer_rows = (
        torch.empty_like(compressed_topk)
        if (for_indexer_loss and compressed_topk is not None)
        else None
    )
    q_padding_mask = (
        torch.empty((geom.l_local,), dtype=torch.bool, device=dev)
        if cu_seqlens_unpadded is not None
        else None
    )
    for b, (g_start, l_off) in enumerate(zip(geom.seg_global_starts, geom.seg_local_offsets)):
        sl = slice(l_off, l_off + c)
        topk_in = None if compressed_topk is None else compressed_topk[sl]
        if topk_in is not None and not (_aligned(topk_in) and topk_in.is_contiguous()):
            topk_in = topk_in.contiguous().clone()
        o_idx = topk_idxs[sl]
        o_len = None if topk_length is None else topk_length[sl]
        o_irm = None if indexer_rows is None else indexer_rows[sl]
        o_pm = None if q_padding_mask is None else q_padding_mask[sl]
        direct = _aligned(o_idx) and _aligned(o_len) and _aligned(o_irm) and _aligned(o_pm)
        res = thd_layout_kernels.build_attention_indices(
            cu_seqlens,
            g_start,
            c,
            geom.d_window,
            window_size,
            ratio,
            compressed_width,
            topk_in,
            cu_seqlens_compressed=cu_seqlens_compressed,
            seq_to_rank_row=seq_to_rank_row,
            for_indexer_loss=for_indexer_loss,
            compressed_base=geom.compressed_base,
            compressed_rows=compressed_rows,
            cu_seqlens_unpadded=cu_seqlens_unpadded,
            output_alignment=output_alignment,
            index_base=geom.block_index_base(b),
            out_topk_idxs=o_idx if direct else None,
            out_topk_length=o_len if direct else None,
            out_indexer_rows=o_irm if direct else None,
            out_q_padding_mask=o_pm if direct else None,
        )
        if not direct:
            o_idx.copy_(res[0])
            if o_len is not None:
                o_len.copy_(res[1])
            if o_irm is not None:
                o_irm.copy_(res[2])
            if o_pm is not None:
                o_pm.copy_(res[3])
    if for_indexer_loss:
        return topk_idxs, None, indexer_rows, q_padding_mask
    return topk_idxs, topk_length, None, q_padding_mask


def cat_kv_full_thd_pack_zigzag(
    boundary_kv: torch.Tensor,
    kv_local: torch.Tensor,
    compressed_kv: torch.Tensor,
    geom: PackZigzagGeometry,
) -> torch.Tensor:
    """``cat(halo0, local[:c], halo1, local[c:], compressed)`` matching the index bases."""
    d_w, c = geom.d_window, geom.chunk
    return torch.cat(
        (boundary_kv[:d_w], kv_local[:c], boundary_kv[d_w:], kv_local[c:], compressed_kv), dim=0
    )
