# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Independent references for the vLLM backend's block alignment, shared by the ``moe_vllm``
oracle modules (``test_scatter_token_indices_kernel.py``, ``test_fill_expert_block_ids_kernel.py``,
``test_moe_align_block_size.py``).

Written from the contract of ``_moe_align_block_size_cuda_graphable``
(``megatron/core/inference/moe/vllm_fused_moe.py``) and of the grouped GEMM that consumes its
tables, not from the kernels:

* a *pair* is ``p = t * topk + j``, slot ``j`` of token ``t``; it is *live* when ``t < valid``;
* local expert ``e`` owns the live pairs with ``routing_map[t, j] == start + e``;
* its segment of the padded table is its pair count rounded up to ``BLOCK_SIZE_M`` (an expert
  without pairs gets no segment); the segments follow each other in expert order from 0;
* every block of a segment carries its expert's local id.

Everything here is plain PyTorch on the host: Python loops and ``torch.nonzero`` per expert, no
cumulative sums over device tensors, atomics or block arithmetic shared with the kernels.
"""

from typing import Dict, List, Sequence, Tuple

import torch

INT32_MIN = torch.iinfo(torch.int32).min


def live_pairs_per_expert(
    routing: torch.Tensor, valid: int, start: int, num_local: int
) -> List[torch.Tensor]:
    """Per local expert, the ascending pair indices of the first ``valid`` rows routed to it."""
    flat = routing[:valid].reshape(-1).to(torch.int64).cpu()
    return [torch.nonzero(flat == start + e).flatten() for e in range(num_local)]


def aligned_offsets(counts: Sequence[int], block_m: int) -> Tuple[List[int], List[int]]:
    """Exclusive and inclusive segment offsets: each count rounded up to ``block_m``, 0 stays 0."""
    exclusive, inclusive, position = [], [], 0
    for n in counts:
        exclusive.append(position)
        position += -(-int(n) // block_m) * block_m
        inclusive.append(position)
    return exclusive, inclusive


def canonical_segments(
    table: torch.Tensor, starts: Sequence[int], lengths: Sequence[int]
) -> torch.Tensor:
    """A host copy of ``table`` with each ``[start, start + length)`` slice sorted ascending.

    The order of pairs inside a segment is unspecified (atomic claims); sorting each live slice
    lets one ``assert_exact`` compare the whole table, segments and everything around them.
    """
    out = table.detach().to(torch.int64).cpu().clone()
    for s, n in zip(starts, lengths):
        if n:
            out[s : s + n] = out[s : s + n].sort().values
    return out


def expected_sorted_ids(
    pairs: Sequence[torch.Tensor], exclusive: Sequence[int], length: int, fill: int
) -> torch.Tensor:
    """The table with each expert's pairs (ascending) at its segment start, ``fill`` elsewhere."""
    out = torch.full((length,), fill, dtype=torch.int64)
    for p, s in zip(pairs, exclusive):
        out[s : s + p.numel()] = p
    return out


def expected_expert_ids(
    exclusive: Sequence[int], inclusive: Sequence[int], block_m: int, length: int, fill: int
) -> torch.Tensor:
    """Per block, the local expert whose segment holds it, ``fill`` for every other block."""
    out = torch.full((length,), fill, dtype=torch.int64)
    for e, (lo, hi) in enumerate(zip(exclusive, inclusive)):
        out[lo // block_m : hi // block_m] = e
    return out


def routing_from_counts(
    counts: Dict[int, int],
    tokens: int,
    topk: int,
    fillers: Sequence[int],
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    """A ``[tokens, topk]`` map in which global expert ``g`` holds exactly ``counts[g]`` slots.

    The other slots take ids drawn from ``fillers`` (``-1`` and other ranks' experts); slots are
    placed in a random order, so an expert may repeat within a token (the kernels treat every
    pair on its own).
    """
    ids = [g for g, c in sorted(counts.items()) for _ in range(c)]
    rest = tokens * topk - len(ids)
    if rest < 0:
        raise ValueError(f"{len(ids)} routed slots do not fit in {tokens} x {topk}")
    if rest:
        fill = torch.tensor(list(fillers), dtype=torch.int64)
        pick = torch.randint(0, len(fillers), (rest,), generator=generator, device=device).cpu()
        ids += fill[pick].tolist()
    flat = torch.tensor(ids, dtype=torch.int64, device=device)
    order = torch.randperm(flat.numel(), generator=generator, device=device)
    return flat[order].view(tokens, topk).contiguous()


def make_routing(
    kind: str,
    tokens: int,
    topk: int,
    num_experts: int,
    start: int,
    num_local: int,
    generator: torch.Generator,
    device: torch.device,
    block_m: int = 16,
) -> torch.Tensor:
    """Live rows of a routing map (global expert ids, ``-1`` for a masked slot).

    ``random``: ids uniform over ``[-1, num_experts)``. ``edges``: only the ids at and next to
    the local slice's edges (``start - 1``, ``start``, ``start + E_local - 1``,
    ``start + E_local``) and ``-1``. ``one_hot``: every slot on the last local expert.
    ``ends_empty``: no slot on the first or the last local expert. ``nonlocal``: nothing local.
    ``masked``: every slot ``-1``. ``block_counts``: local experts hold ``BM - 1``, ``BM``,
    ``BM + 1``, 0, 1, ``2 BM - 1``, ``2 BM`` and ``2 BM + 1`` slots in turn (as many as fit).
    """
    local = list(range(start, start + num_local))
    others = [g for g in range(num_experts) if g not in local]
    size = (tokens, topk)
    if kind == "random":
        return torch.randint(-1, num_experts, size, generator=generator, device=device)
    if kind == "edges":
        ids = sorted({-1, start, start + num_local - 1} | ({start - 1} if start else set()))
        ids += [start + num_local] if start + num_local < num_experts else []
        pick = torch.randint(0, len(ids), size, generator=generator, device=device)
        return torch.tensor(ids, dtype=torch.int64, device=device)[pick]
    if kind == "one_hot":
        return torch.full(size, local[-1], dtype=torch.int64, device=device)
    if kind == "ends_empty":
        inner = local[1:-1] or others or [-1]
        pool = torch.tensor(inner + others + [-1], dtype=torch.int64, device=device)
        return pool[torch.randint(0, pool.numel(), size, generator=generator, device=device)]
    if kind == "nonlocal":
        pool = torch.tensor(others + [-1], dtype=torch.int64, device=device)
        return pool[torch.randint(0, pool.numel(), size, generator=generator, device=device)]
    if kind == "masked":
        return torch.full(size, -1, dtype=torch.int64, device=device)
    if kind == "block_counts":
        cycle = [block_m - 1, block_m, block_m + 1, 0, 1, 2 * block_m - 1, 2 * block_m]
        cycle += [2 * block_m + 1]
        counts, room = {}, tokens * topk
        for i, g in enumerate(local):
            c = min(cycle[i % len(cycle)], room)
            counts[g], room = c, room - c
        return routing_from_counts(counts, tokens, topk, others + [-1], generator, device)
    raise ValueError(kind)


def full_routing(
    live: torch.Tensor,
    max_tokens: int,
    start: int,
    num_local: int,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> torch.Tensor:
    """``live`` followed by rows routed to local experts up to ``max_tokens`` rows.

    Rows past ``valid_tokens`` are garbage by contract; local ids there make any read of them
    claim a slot or add a term, so it shows in the tables or the sums.
    """
    valid, topk = live.shape
    pad = torch.randint(
        start,
        start + num_local,
        (max_tokens - valid, topk),
        generator=generator,
        device=live.device,
    )
    return torch.cat([live.to(torch.int64), pad]).to(dtype).contiguous()
