# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Pass linear-attention boundary states across context-parallel ranks with two all-gathers.

Chunkwise context parallelism for linear attention summarizes each rank's shard of a sequence as
an affine map of the recurrent state, ``S_out = M S_in + delta_S``, packed as ``[delta_S | M]``
with shape ``[..., K, V + K]`` (FLA's ``hm`` layout). A rank's incoming state is the composition
of the maps of the ranks before it in its sequence, applied to a zero state; the backward pass
does the same in reverse rank order with the adjoint maps.

FLA all-gathers every rank's summary over the whole CP group, so at CP 64 each rank receives 63
summaries (504 MiB for 64 heads of 128 x 128). Here the group is split into blocks of ``G``
consecutive ranks:

1. an all-gather inside the block gives each rank its block's summaries;
2. each block composes the part of its summaries that later blocks need into one summary, and an
   all-gather across the blocks, among the ranks at the same position in their blocks, gives each
   rank every block's composed summary;
3. each rank applies to a zero state the composed summaries of the earlier blocks in its sequence,
   then the summaries of the earlier ranks of its own block.

At CP 64 with ``G = 8`` a rank receives 14 summaries in two collectives. Packed sequences split
the ranks into chains (the ranks one sequence spans); :class:`RankChains` describes them for every
rank, computed on the host from the global sequence boundaries with FLA's arithmetic, so every
rank takes the same decisions without communication.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.distributed as dist

try:
    import triton
    from fla.ops.cp.chunk_delta_h import merge_fwd_bwd_kernel

    HAVE_FLA_MERGE = True
except ImportError:
    HAVE_FLA_MERGE = False


@dataclass(frozen=True)
class RankChains:
    """Per CP rank: ranks before/after it in its first/last local sequence, and its sequences.

    ``pre[r]`` and ``post[r]`` are FLA's ``pre_num_ranks`` and ``post_num_ranks`` of rank ``r``;
    ``num_seqs[r]`` is the number of sequences with tokens on rank ``r``.
    """

    pre: tuple
    post: tuple
    num_seqs: tuple


def rank_chains(cu_seqlens_cpu: torch.Tensor, cp_size: int) -> RankChains:
    """FLA's ``get_cp_cu_seqlens`` counts for every rank of an equal-split CP group."""
    cu = [int(c) for c in cu_seqlens_cpu.tolist()]
    part = cu[-1] // cp_size
    pre, post, num_seqs = [], [], []
    for rank in range(cp_size):
        start, end = part * rank, part * (rank + 1)
        first = bisect.bisect_right(cu[1:], start)
        last = bisect.bisect_left(cu[:-1], end)
        local = []
        for c in cu[first : last + 1]:
            c = min(max(c, start), end) - start
            if not local or local[-1] != c:
                local.append(c)
        pre.append(rank - cu[first] // part)
        post.append((cu[last] - 1) // part - rank)
        num_seqs.append(len(local) - 1)
    return RankChains(tuple(pre), tuple(post), tuple(num_seqs))


def block_size(cp_size: int) -> Optional[int]:
    """Ranks per block, or None when one all-gather over the group is as cheap.

    The smallest power of two at least ``sqrt(cp_size)`` that divides it; None for 8 ranks or
    fewer, where the second all-gather would not save anything.
    """
    if cp_size <= 8:
        return None
    size = 1 << math.ceil(math.log2(math.sqrt(cp_size)))
    return size if cp_size % size == 0 else None


def compose_affine_summaries(later: torch.Tensor, earlier: torch.Tensor) -> torch.Tensor:
    """Pack ``later ∘ earlier``: apply ``earlier`` first, then ``later``.

    With ``later = [d_l | M_l]`` and ``earlier = [d_e | M_e]`` the composition is
    ``[M_l d_e + d_l | M_l M_e]``, one batched ``M_l @ earlier`` plus ``d_l``.
    """
    key_dim = later.shape[-2]
    value_dim = later.shape[-1] - key_dim
    composed = torch.matmul(later[..., value_dim:], earlier)
    composed[..., :value_dim] += later[..., :value_dim]
    return composed


def apply_summaries(stack: torch.Tensor, ascending: bool) -> torch.Tensor:
    """Apply the summaries of ``stack`` ``[m, ..., K, V + K]`` to a zero state, in index order.

    ``ascending`` applies ``stack[0]`` first; otherwise ``stack[m - 1]`` first. Returns the state
    ``[..., K, V]``. CUDA FP32 stacks of ``[m, HV, K, V + K]`` with ``K, V >= 16`` run FLA's CP
    merge kernel (its ``tl.dot`` needs 16); others a PyTorch loop.
    """
    key_dim = stack.shape[-2]
    value_dim = stack.shape[-1] - key_dim
    count = stack.shape[0]
    out = stack.new_zeros(stack.shape[1:-1] + (value_dim,))
    if count == 0:
        return out
    if (
        HAVE_FLA_MERGE
        and stack.is_cuda
        and stack.dtype == torch.float32
        and stack.dim() == 4
        and min(key_dim, value_dim) >= 16
    ):
        stack = stack.contiguous()
        heads = stack.shape[1]

        def grid(meta):
            return (triton.cdiv(value_dim, meta["BV"]), heads)

        # FLA's CP merge visits ranks rank - num + idx (forward) or rank + num - idx (backward);
        # rebased onto this slice that is 0..count-1 or count-1..0.
        merge_fwd_bwd_kernel[grid](
            h=out,
            ag_hm=stack,
            pre_or_post_num_ranks=count,
            rank=count if ascending else -1,
            seq_offsets=None,
            init_offsets=None,
            h0_seq_ids=None,
            h0=None,
            HV=heads,
            K=key_dim,
            V=value_dim,
            BK=triton.next_power_of_2(key_dim),
            FORWARD=ascending,
            INTRACARD_MODE=False,
            NUM_SEQ_ENTRIES=0,
            STATE_V_FIRST=False,
        )
        return out
    order = range(count) if ascending else range(count - 1, -1, -1)
    for i in order:
        out = torch.matmul(stack[i][..., value_dim:], out) + stack[i][..., :value_dim]
    return out


_SUBGROUPS: dict = {}


def _block_subgroups(group: dist.ProcessGroup, size: int):
    """This rank's block group and cross-block group, created once per CP group and block size.

    Only the members of each new group take part in its creation, so every rank creates its block
    group first and its cross-block group second, as all its peers do.
    """
    ranks = tuple(dist.get_process_group_ranks(group))
    key = (ranks, size)
    if key not in _SUBGROUPS:
        me = group.rank()
        block, position = divmod(me, size)
        block_ranks = list(ranks[block * size : (block + 1) * size])
        cross_ranks = list(ranks[position::size])
        intra = dist.new_group(block_ranks, use_local_synchronization=True)
        inter = dist.new_group(cross_ranks, use_local_synchronization=True)
        _SUBGROUPS[key] = (intra, inter)
    return _SUBGROUPS[key]


def _default_all_gather(output: torch.Tensor, local: torch.Tensor, group) -> None:
    dist.all_gather_into_tensor(output, local, group=group)


def exchange_boundary_state(
    local_summary: torch.Tensor,
    *,
    group: dist.ProcessGroup,
    chains: RankChains,
    reverse: bool = False,
    size: Optional[int] = None,
    subgroups: Optional[Callable] = None,
    all_gather: Callable = _default_all_gather,
) -> Optional[torch.Tensor]:
    """This rank's incoming state: its upstream ranks' summaries applied to a zero state.

    Args:
        local_summary: This rank's ``[delta_S | M]`` summary of its outgoing sequence (its last
            local sequence, its first when ``reverse``), ``[..., K, V + K]``. Every rank passes a
            tensor of the same shape and dtype; it is read only when a later rank needs it.
        group: The CP group, ranks in causal order.
        chains: :func:`rank_chains` of the global sequence boundaries.
        reverse: The backward pass: states flow from the last rank towards the first.
        size: Ranks per block; :func:`block_size` by default. The group size must be a multiple.
        subgroups: ``(group, size) -> (block_group, cross_block_group)``, for tests.
        all_gather: ``(output, local, group)`` all-gather into ``output``, for tests.

    Returns:
        ``delta_S`` ``[..., K, V]`` of the composition, or None when no rank precedes this one in
        its sequence. Every rank of ``group`` must call this together.
    """
    n = group.size()
    size = size or block_size(n)
    if size is None or n % size:
        raise ValueError(f"no block size for a CP group of {n} ranks")
    blocks = n // size
    me = group.rank()
    # Logical indices run in the direction the states flow; physical ones are group ranks.
    phys = (lambda j: n - 1 - j) if reverse else (lambda j: j)
    up = chains.post if reverse else chains.pre
    down = chains.pre if reverse else chains.post

    def upstream(j):  # ranks before logical j in its incoming chain
        return up[phys(j)]

    def chain_start(j):  # logical start of logical j's outgoing chain
        p = phys(j)
        return j - (up[p] if chains.num_seqs[p] == 1 else 0)

    def block_needed(b):  # does a later block read logical block b's composed summary?
        last = (b + 1) * size - 1
        return down[phys(last)] >= 1

    if all(u == 0 for u in up):
        return None
    my = phys(me)  # logical index of this rank
    my_block = my // size
    block_group, cross_group = (subgroups or _block_subgroups)(group, size)

    # 1. Block all-gather, into a buffer with a spare slot that later holds the earlier blocks'
    #    state as a summary with M = 0: [spare | block] forwards, [block | spare] backwards, so the
    #    spare sits next to the block's first logical rank.
    key_dim = local_summary.shape[-2]
    value_dim = local_summary.shape[-1] - key_dim
    buffer = local_summary.new_zeros((size + 1,) + tuple(local_summary.shape))
    block_slots = buffer[:size] if reverse else buffer[1:]
    all_gather(block_slots, local_summary.contiguous(), block_group)
    phys_block = phys(my_block * size) // size

    def slot(j):  # buffer index of logical j's summary (j in this rank's block)
        within = phys(j) - phys_block * size
        return within if reverse else within + 1

    # 2. This block's composed summary, all-gathered across blocks when any block needs one.
    if any(block_needed(b) for b in range(blocks - 1)):
        last = my_block * size + size - 1
        composed = torch.zeros_like(local_summary)
        if block_needed(my_block):
            first = max(my_block * size, chain_start(last))
            composed = buffer[slot(first)]
            for j in range(first + 1, last + 1):
                composed = compose_affine_summaries(buffer[slot(j)], composed)
        cross = local_summary.new_empty((blocks,) + tuple(local_summary.shape))
        all_gather(cross, composed.contiguous(), cross_group)
    else:
        cross = None

    # 3. Apply the earlier blocks' composed summaries, then this block's earlier ranks.
    if upstream(my) == 0:
        return None
    start = my - upstream(my)
    block_start = my_block * size
    if start >= block_start:
        lo, hi = sorted((slot(start), slot(my - 1)))
        return apply_summaries(buffer[lo : hi + 1], ascending=not reverse)
    first_block = start // size
    # Logical blocks first_block..my_block-1 are physical blocks in one contiguous run.
    pb = sorted((phys(first_block * size) // size, phys((my_block - 1) * size) // size))
    earlier = apply_summaries(cross[pb[0] : pb[1] + 1], ascending=not reverse)
    spare = 0 if not reverse else size
    buffer[spare].zero_()
    buffer[spare][..., :value_dim] = earlier
    if my == block_start:
        return earlier
    lo, hi = sorted((spare, slot(my - 1)))
    return apply_summaries(buffer[lo : hi + 1], ascending=not reverse)
