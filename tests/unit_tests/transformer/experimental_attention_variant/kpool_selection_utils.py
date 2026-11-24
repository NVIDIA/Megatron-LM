# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Shared inputs and checks for the KPool pool-selection backend tests (CPU and GPU)."""

from types import SimpleNamespace

import torch

from megatron.core.transformer.experimental_attention_variant import (
    dsa,
    dsa_cudnn_kernels,
    dsa_layout,
)

POOL = 4


def kpool_inputs(
    *, seq, cp_size, cp_rank, batch, heads, dim, device, seed=0, projection_dtype=torch.bfloat16
):
    """One CP rank's KPool scoring inputs: local queries, the gathered sequence's keys.

    ``projection_dtype`` is the head-weight projection's output dtype; HF scoring weights are
    only meaningful for BF16 (``dsa_indexer_hf_scoring`` requires it).
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    rows = seq // cp_size

    def randn(*shape):
        return torch.randn(shape, generator=generator).to(device)

    positions, _ = dsa_layout.get_cp_positions_from_layout(
        sq=rows, skv=seq, cp_size=cp_size, cp_rank=cp_rank, cp_comm_type="allgather", device=device
    )
    projection = randn(rows, batch, heads).to(projection_dtype)
    return SimpleNamespace(
        q=randn(rows, batch, heads, dim).to(torch.bfloat16),
        k=randn(seq, batch, dim).to(torch.bfloat16),
        gate=randn(seq, batch, dim).to(torch.bfloat16),
        ape=0.02 * randn(POOL, dim),
        projection=projection,
        # HF scoring: BF16 projection upcast, head-count scale only.
        weights_hf=projection.float() * heads**-0.5,
        # GLM-5.2 convention: both scales on the weights, in the projection's dtype.
        weights_scaled=projection * (heads**-0.5 * dim**-0.5),
        positions=positions,
        starts=torch.zeros_like(positions),
        ends=positions + 1,
    )


def select(x, *, topk, hf_scoring, pool_topk_fn=None):
    """``fused_qk_topk_kpool`` as DSAttention calls it on the causal CP path."""
    with torch.no_grad():
        _, tokens = dsa.fused_qk_topk_kpool(
            x.q,
            x.k,
            x.weights_hf if hf_scoring else x.weights_scaled,
            topk,
            POOL,
            x.gate,
            x.ape,
            varlen_starts=x.starts,
            varlen_ends=x.ends,
            use_relu=True,
            always_select_tail=True,
            return_index_scores=False,
            kpool_key_dtype_compression=True,
            hf_scoring=hf_scoring,
            pool_topk_fn=pool_topk_fn,
        )
    return tokens


def cudnn_pool_topk_fn(query_runs, *, deterministic=False, config=None, calls=None):
    """The pool selector DSAttention builds for ``dsa_indexer_scorer_backend='cudnn'``.

    ``calls``, when given, records per call whether the backend took it (not None).
    """

    def select_pools(q, k_pooled, weights, pool_topk, pool_size, hf_scoring):
        result = dsa_cudnn_kernels.run_kpool_pool_topk(
            q=q,
            k_pooled=k_pooled,
            weights=weights,
            pool_topk=pool_topk,
            pool_size=pool_size,
            query_runs=query_runs,
            hf_scoring=hf_scoring,
            deterministic=deterministic,
            config=config,
        )
        if calls is not None:
            calls.append(result is not None)
        return result

    return select_pools


def cudnn_packed_pool_topk_fn(
    cu_seqlens, max_seqlen, *, deterministic=False, config=None, calls=None, scores_out=None
):
    """The device-boundary pool selector DSAttention builds for packed THD without CP.

    ``calls`` records per call whether the backend took it; ``scores_out`` (a dict), when
    given, receives the selected pools' scores under ``"pool_scores"``.
    """

    def select_pools(q, k_pooled, weights, pool_topk, pool_size, hf_scoring):
        result = dsa_cudnn_kernels.run_kpool_pool_topk(
            q=q,
            k_pooled=k_pooled,
            weights=weights,
            pool_topk=pool_topk,
            pool_size=pool_size,
            query_runs=None,
            hf_scoring=hf_scoring,
            deterministic=deterministic,
            config=config,
            return_scores=scores_out is not None,
            packed_cu_seqlens=cu_seqlens,
            packed_max_seqlen=max_seqlen,
        )
        if calls is not None:
            calls.append(result is not None)
        if result is not None and scores_out is not None:
            result, scores_out["pool_scores"] = result
        return result

    return select_pools


def query_runs(seq, cp_size, cp_rank):
    """Host query runs of one CP rank's rows (all of them, no TP slicing)."""
    rows = seq // cp_size
    runs, reason = dsa_layout.nonpacked_query_position_runs(
        local_rows=rows, full_rows=rows, row_start=0, skv=seq, cp_size=cp_size, cp_rank=cp_rank
    )
    assert runs is not None, reason
    return runs


def pool_sets(tokens, topk):
    """Per row: the set of selected pools (from the expanded tokens) and the tail columns."""
    pools = tokens[..., :topk].reshape(*tokens.shape[:-1], topk // POOL, POOL)[..., 0]
    return [
        [set(int(p) // POOL for p in row.tolist() if p >= 0) for row in batch_rows]
        for batch_rows in pools
    ]


def assert_same_selection_up_to_boundary_ties(
    x, reference, candidate, *, topk, rel_tol, kpool_key_dtype_compression=True
):
    """The two selections agree, except pools whose exact score ties the selection boundary.

    For a row whose pool sets differ, the pools are scored in FP64 (HF formula); every pool
    only one side kept must score within ``rel_tol`` of that row's ``topk // POOL``-th best
    score. Tails (columns past ``topk``) must be identical. ``x`` needs ``q``, ``k``, ``gate``,
    ``ape``, ``weights_hf`` and ``ends``; ``kpool_key_dtype_compression`` is the pool
    compression profile the selections were made with. Returns the number of differing rows.

    The gap is relative to the boundary score itself. Head weights are signed, so that score
    can be close to zero and a rounding-level tie then shows a large relative gap: a failure
    reports the boundary score next to the row's largest ``|score|`` to tell the two apart.
    """
    assert reference.shape == candidate.shape
    assert torch.equal(reference[..., topk:], candidate[..., topk:])
    budget = topk // POOL
    k_pooled = dsa._kpool_compress_keys(
        x.k, x.gate, x.ape, POOL, kpool_key_dtype_compression=kpool_key_dtype_compression
    )
    ref_sets, new_sets = pool_sets(reference, topk), pool_sets(candidate, topk)
    differing = 0
    for b, (ref_rows, new_rows) in enumerate(zip(ref_sets, new_sets)):
        for row, (ref, new) in enumerate(zip(ref_rows, new_rows)):
            if ref == new:
                continue
            differing += 1
            assert len(ref) == len(new), f"batch {b} row {row}: valid pool counts differ"
            visible = int(x.ends[row]) // POOL
            assert len(ref) == budget < visible, f"batch {b} row {row}: a short row differs"
            q = x.q[row, b].double()
            keys = k_pooled[:visible, b].double()
            scores = torch.relu((q @ keys.T) * q.size(-1) ** -0.5)
            score = (x.weights_hf[row, b].double().unsqueeze(-1) * scores).sum(0)
            kth = float(torch.topk(score, budget).values[-1])
            for pool in ref ^ new:
                gap = abs(float(score[pool]) - kth) / max(abs(kth), 1e-30)
                assert gap <= rel_tol, (
                    f"batch {b} row {row} pool {pool}: gap {gap:.3e} relative to the boundary "
                    f"score {kth:.3e}; the row's largest |score| is {float(score.abs().max()):.3e}"
                )
    return differing


def packed_kpool_inputs(
    lengths, *, cp_size, cp_rank, batch, heads, dim, device, seed=0, pad_to=None
):
    """One CP rank's packed KPool inputs: local queries, the gathered packed keys and bounds.

    ``lengths`` are the documents' padded lengths (divisible by ``2 * cp_size`` under CP). The
    local rows are each document's front and back zigzag spans, as packed all-gather CP lays
    them out; ``starts``/``ends`` are their global packed key bounds (document start, position
    + 1), ``cu`` the global boundary table (padded to ``pad_to + 1`` entries when given, as
    static packed inputs are) and ``host_cu`` its compacted host copy.
    """
    host_cu = [0]
    for length in lengths:
        host_cu.append(host_cu[-1] + length)
    starts, ends = [], []
    for doc, length in enumerate(lengths):
        if cp_size == 1:
            positions = list(range(length))
        else:
            assert length % (2 * cp_size) == 0
            half = length // (2 * cp_size)
            positions = list(range(cp_rank * half, (cp_rank + 1) * half)) + list(
                range(length - (cp_rank + 1) * half, length - cp_rank * half)
            )
        starts += [host_cu[doc]] * len(positions)
        ends += [host_cu[doc] + position + 1 for position in positions]
    cu = list(host_cu)
    if pad_to is not None:
        cu += [cu[-1]] * (pad_to + 1 - len(cu))
    generator = torch.Generator(device="cpu").manual_seed(seed)

    def randn(*shape):
        return torch.randn(shape, generator=generator).to(device)

    rows, total = len(starts), host_cu[-1]
    projection = randn(rows, batch, heads).to(torch.bfloat16)
    return SimpleNamespace(
        q=randn(rows, batch, heads, dim).to(torch.bfloat16),
        k=randn(total, batch, dim).to(torch.bfloat16),
        gate=randn(total, batch, dim).to(torch.bfloat16),
        ape=0.02 * randn(POOL, dim),
        weights_hf=projection.float() * heads**-0.5,
        weights_scaled=projection * (heads**-0.5 * dim**-0.5),
        starts=torch.tensor(starts, dtype=torch.int64, device=device),
        ends=torch.tensor(ends, dtype=torch.int64, device=device),
        cu=torch.tensor(cu, dtype=torch.int32, device=device),
        host_cu=host_cu,
        rows=rows,
        cp_size=cp_size,
        cp_rank=cp_rank,
    )


def select_packed(x, *, topk, hf_scoring, pool_topk_fn=None):
    """``fused_qk_topk_kpool`` as DSAttention calls it on the packed causal path."""
    with torch.no_grad():
        _, tokens = dsa.fused_qk_topk_kpool(
            x.q,
            x.k,
            x.weights_hf if hf_scoring else x.weights_scaled,
            topk,
            POOL,
            x.gate,
            x.ape,
            varlen_starts=x.starts,
            varlen_ends=x.ends,
            cu_seqlens_kv=x.cu,
            use_relu=True,
            always_select_tail=True,
            return_index_scores=False,
            kpool_key_dtype_compression=True,
            hf_scoring=hf_scoring,
            pool_topk_fn=pool_topk_fn,
        )
    return tokens


def packed_query_runs(x, *, local_rows=None, row_start=0):
    """Keyed host query runs of a packed rank (``dsa_layout.packed_kpool_query_runs``)."""
    runs, reason = dsa_layout.packed_kpool_query_runs(
        host_cu_seqlens_q=x.host_cu,
        host_cu_seqlens_kv=x.host_cu,
        local_rows=x.rows if local_rows is None else local_rows,
        full_rows=x.rows,
        row_start=row_start,
        cp_size=x.cp_size,
        cp_rank=x.cp_rank,
        pool_size=POOL,
    )
    assert runs is not None, reason
    return runs


def assert_same_packed_selection_up_to_boundary_ties(x, reference, candidate, *, topk, rel_tol):
    """:func:`assert_same_selection_up_to_boundary_ties` on per-document packed pools.

    Pools are named by their first token; a differing row's pools are scored in FP64 (HF
    formula) over the row's visible pools of its own document. Returns the number of
    differing rows.
    """
    assert reference.shape == candidate.shape
    assert torch.equal(reference[..., topk:], candidate[..., topk:])
    budget = topk // POOL
    k_pooled, base = dsa._kpool_compress_keys_per_seg(
        x.k, x.gate, x.ape, POOL, x.cu, kpool_key_dtype_compression=True
    )
    base = base.tolist()
    starts, ends = x.starts.tolist(), x.ends.tolist()

    def pool_firsts(tokens):
        firsts = tokens[..., :topk].reshape(*tokens.shape[:-1], budget, POOL)[..., 0]
        return [[set(int(t) for t in row.tolist() if t >= 0) for row in b] for b in firsts]

    differing = 0
    for b, (ref_rows, new_rows) in enumerate(zip(pool_firsts(reference), pool_firsts(candidate))):
        for row, (ref, new) in enumerate(zip(ref_rows, new_rows)):
            if ref == new:
                continue
            differing += 1
            visible = [i for i, t in enumerate(base) if t >= starts[row] and t + POOL <= ends[row]]
            assert len(ref) == len(new) == budget < len(visible), f"row {row}: short row differs"
            q = x.q[row, b].double()
            keys = k_pooled[visible, b].double()
            scores = torch.relu((q @ keys.T) * q.size(-1) ** -0.5)
            score = (x.weights_hf[row, b].double().unsqueeze(-1) * scores).sum(0)
            kth = float(torch.topk(score, budget).values[-1])
            by_first = {base[i]: j for j, i in enumerate(visible)}
            for first in ref ^ new:
                gap = abs(float(score[by_first[first]]) - kth) / max(abs(kth), 1e-30)
                assert gap <= rel_tol, f"batch {b} row {row} pool at token {first}: gap {gap:.3e}"
    return differing
