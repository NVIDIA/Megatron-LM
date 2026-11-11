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
    """The pool selector DSAttention builds for ``dsa_indexer_kpool_scorer_backend='cudnn'``.

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
