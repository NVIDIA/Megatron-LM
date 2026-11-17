# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU tests of the trainable KPool indexer (sparse pool-level KL) against an independent oracle.

The oracle, ``kpool_trainable_reference``, is plain dense PyTorch written from the loss
semantics alone and imports nothing from ``dsa.py``. These tests check the phase-1 helpers
of ``dsa.py`` against it:

- ``_kpool_canonical_slots``: slot order, tail columns, no mutation;
- ``_kpool_sparse_pool_target``: the teacher, its tail-in-denominator rule, rows without pool
  mass (no complete pool, FP32 underflow), unused slots and the TP all-reduce of the head
  mass (through a fake TP group);
- ``_kpool_selected_pool_scores``: the student under HF and legacy scoring;
- ``KPoolSparseIndexerLoss``: value and gradients, an FP64 ``gradcheck`` with one row per chunk
  and with all rows in one chunk, the row-count divisor, empty inputs and deterministic
  accumulation of the pooled-key gradient (segment sums in a fixed order, no scan over a
  leading dimension);
- the whole chain (compression, canonical slots, teacher, loss) in both compression
  profiles, plain and packed THD;
- the per-layer tail mass ``p_E`` that ``DSAttention._kpool_sparse_indexer_loss`` records in
  its own tracker slots (the oracle's row mean, the indexer loss untouched) and its
  reduction and reporting in ``track_indexer_metrics``;
- ``DSAIndexer.kpool_pool_keys``: values, gradients to the gate and the APE (finite
  differences in the FP32 profile, an autograd oracle with a BF16 tolerance in the key-dtype
  profile) and pool-then-gather equality;
- ``fused_qk_topk_kpool(pooled_keys=..., return_selection=...)``;
- the configuration rules, and that frozen configurations are unchanged.

Everything runs on CPU. A missing phase-1 symbol fails its tests with a message that names
it; nothing is skipped for that reason.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from megatron.core import utils as core_utils
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.experimental_attention_variant import dsa as dsa_module
from megatron.core.transformer.transformer_config import MLATransformerConfig, TransformerConfig
from megatron.core.utils import init_method_normal
from tests.unit_tests.transformer.experimental_attention_variant import (
    kpool_trainable_reference as ref,
)
from tests.unit_tests.transformer.experimental_attention_variant.test_mtp_dsa_cross_depth_sharing import (
    _make_config as make_small_dsa_config,
)

POOL = 4
TOPK = 16  # four selected pools per row
SEQ = 24  # six complete pools on the last row
# Packed THD: a document shorter than a pool, then two documents that do not end on the pool
# grid. Six pool slots, four of them used.
PACKED_CU = (0, 3, 14, 24)
ALL_ROWS = 1 << 40  # chunk_bytes that puts every row in one chunk
ONE_ROW = 1  # chunk_bytes that forces one row per chunk
SCORING = {
    "hf": dict(hf_scoring=True, use_relu=True),
    "legacy_relu": dict(hf_scoring=False, use_relu=True),
    "legacy_linear": dict(hf_scoring=False, use_relu=False),
}


def _contract(name):
    """A phase-1 symbol of ``dsa.py``; fails (does not skip) while it is missing."""
    value = getattr(dsa_module, name, None)
    if value is None:
        pytest.fail(f"dsa.{name} is missing: the phase-1 contract symbol is not implemented")
    return value


def _generator(seed):
    return torch.Generator().manual_seed(seed)


def _randn(generator, *shape, dtype=torch.float32, scale=1.0):
    return (torch.randn(shape, generator=generator) * scale).to(dtype)


def _assert_close_scaled(actual, expected, rel, what):
    """Max absolute error within ``rel`` times the largest reference magnitude."""
    assert actual is not None, f"{what}: no gradient"
    assert actual.shape == expected.shape, f"{what}: shape {actual.shape} != {expected.shape}"
    if expected.numel() == 0:
        return
    actual, expected = actual.double(), expected.double()
    assert torch.isfinite(actual).all(), f"{what}: non-finite values"
    scale = float(expected.abs().max())
    error = float((actual - expected).abs().max())
    assert error <= rel * max(
        scale, 1e-30
    ), f"{what}: max error {error:.3e} exceeds {rel:g} x max |reference| {scale:.3e}"


def _selection(
    seed, *, seq_len=SEQ, batch=2, index_topk=TOPK, cu_seqlens=None, interleave_pads=True
):
    """An oracle selection in selector slot order and its canonical form."""
    base = ref.pool_token_base(seq_len, POOL, cu_seqlens)
    pool_ids = ref.random_pool_selection(
        _generator(seed),
        seq_len=seq_len,
        batch=batch,
        pool_size=POOL,
        index_topk=index_topk,
        base=base,
        cu_seqlens=cu_seqlens,
        interleave_pads=interleave_pads,
    )
    token_topk = ref.expand_selection(pool_ids, base, POOL, index_topk, cu_seqlens)
    pool_ids_c, token_topk_c = ref.canonical_slots(
        token_topk, pool_ids, index_topk=index_topk, pool_size=POOL
    )
    return SimpleNamespace(
        base=base,
        # The selector emits int32 token ids and int64 pool ids.
        pool_ids=pool_ids,
        token_topk=token_topk.to(torch.int32),
        pool_ids_c=pool_ids_c,
        token_topk_c=token_topk_c.to(torch.int32),
        cu_seqlens=cu_seqlens,
        index_topk=index_topk,
        seq_len=seq_len,
        batch=batch,
    )


def _attention(seed, seq_len, batch, heads, dim, *, latent, dtype=torch.float32):
    """Main-attention query ``[sq, b, np, hn]`` and key ``[sk, b, 1 or np, hn]``."""
    generator = _generator(seed)
    query = _randn(generator, seq_len, batch, heads, dim, dtype=dtype)
    key = _randn(generator, seq_len, batch, 1 if latent else heads, dim, dtype=dtype)
    return query, key


def _recency_attention(seq_len, batch, recency, *, dim=8, latent=True, seed=0):
    """Main attention whose head ``h`` prefers recent tokens with strength ``recency[h]``.

    Feature 0 of every key is its position scaled to ``[0, 1]`` and feature 0 of head ``h``'s
    query is ``recency[h]`` (softmax scale 1), so a large value puts most of that head's mass
    on the newest tokens, the tail.
    """
    generator = _generator(seed)
    heads = len(recency)
    key = _randn(generator, seq_len, batch, 1 if latent else heads, dim, scale=0.3)
    key[..., 0] = (torch.arange(seq_len, dtype=torch.float32) / (seq_len - 1)).view(-1, 1, 1)
    query = _randn(generator, seq_len, batch, heads, dim, scale=0.3)
    query[..., 0] = torch.tensor(recency, dtype=torch.float32)
    return query, key


class _FakeGroup:
    def __init__(self, size, rank):
        self._size, self._rank = size, rank

    def size(self):
        return self._size

    def rank(self):
        return self._rank


# ---------------------------------------------------------------------------------------
# _kpool_canonical_slots
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("interleave_pads", [False, True])
@pytest.mark.parametrize("packed", [False, True])
def test_canonical_slots_match_oracle(packed, interleave_pads):
    canonical = _contract("_kpool_canonical_slots")
    sel = _selection(
        3,
        batch=1 if packed else 2,
        cu_seqlens=PACKED_CU if packed else None,
        interleave_pads=interleave_pads,
    )
    pool_ids_c, token_topk_c = canonical(
        sel.token_topk, sel.pool_ids, index_topk=TOPK, pool_size=POOL
    )
    assert pool_ids_c.shape == sel.pool_ids.shape
    assert token_topk_c.shape == sel.token_topk.shape
    assert torch.equal(pool_ids_c.long(), sel.pool_ids_c)
    assert torch.equal(token_topk_c.long(), sel.token_topk_c.long())
    # Valid pools ascend, unused slots come last, the tail columns are untouched.
    valid = pool_ids_c >= 0
    assert torch.equal(valid, valid.sort(dim=-1, descending=True, stable=True).values)
    keyed = pool_ids_c.masked_fill(~valid, torch.iinfo(torch.int64).max)
    assert torch.all(keyed[..., 1:] >= keyed[..., :-1])
    assert torch.equal(token_topk_c[..., TOPK:].long(), sel.token_topk[..., TOPK:].long())


def test_canonical_slots_do_not_mutate_inputs_and_return_new_tensors():
    canonical = _contract("_kpool_canonical_slots")
    sel = _selection(4, interleave_pads=True)
    token_topk, pool_ids = sel.token_topk.clone(), sel.pool_ids.clone()
    pool_ids_c, token_topk_c = canonical(token_topk, pool_ids, index_topk=TOPK, pool_size=POOL)
    assert torch.equal(token_topk, sel.token_topk)
    assert torch.equal(pool_ids, sel.pool_ids)
    # Writing into the outputs never reaches the inputs.
    pool_ids_c.fill_(-7)
    token_topk_c.fill_(-7)
    assert torch.equal(token_topk, sel.token_topk)
    assert torch.equal(pool_ids, sel.pool_ids)


def test_canonical_slots_are_idempotent_and_keep_rows_without_pools():
    canonical = _contract("_kpool_canonical_slots")
    sel = _selection(5, interleave_pads=True)
    once = canonical(sel.token_topk, sel.pool_ids, index_topk=TOPK, pool_size=POOL)
    twice = canonical(once[1], once[0], index_topk=TOPK, pool_size=POOL)
    assert torch.equal(once[0], twice[0]) and torch.equal(once[1], twice[1])
    # The first P - 1 rows have no complete pool: every pool slot stays -1.
    assert torch.all(once[0][:, : POOL - 1] == -1)
    assert torch.all(once[1][:, : POOL - 1, :TOPK] == -1)


def test_canonical_slots_accept_zero_rows():
    canonical = _contract("_kpool_canonical_slots")
    token_topk = torch.empty(2, 0, TOPK + POOL - 1, dtype=torch.int32)
    pool_ids = torch.empty(2, 0, TOPK // POOL, dtype=torch.int64)
    pool_ids_c, token_topk_c = canonical(token_topk, pool_ids, index_topk=TOPK, pool_size=POOL)
    assert pool_ids_c.shape == pool_ids.shape and token_topk_c.shape == token_topk.shape


# ---------------------------------------------------------------------------------------
# _kpool_sparse_pool_target
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "chunk_bytes", [ONE_ROW, ALL_ROWS, None], ids=["one-row", "all-rows", "default"]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("latent", [True, False], ids=["latent-key", "per-head-keys"])
@pytest.mark.parametrize("packed", [False, True], ids=["plain", "packed"])
def test_sparse_pool_target_matches_oracle(packed, latent, dtype, chunk_bytes):
    target_fn = _contract("_kpool_sparse_pool_target")
    sel = _selection(11, batch=1 if packed else 2, cu_seqlens=PACKED_CU if packed else None)
    heads, dim = 4, 16
    query, key = _attention(12, SEQ, sel.batch, heads, dim, latent=latent, dtype=dtype)
    scale = dim**-0.5
    kwargs = {} if chunk_bytes is None else {"chunk_bytes": chunk_bytes}
    target, row_has_mass, tail_mass = target_fn(
        query,
        key,
        sel.token_topk_c,
        sel.pool_ids_c,
        softmax_scale=scale,
        index_topk=TOPK,
        pool_size=POOL,
        **kwargs,
    )
    expected = ref.sparse_pool_target(
        query, key, sel.pool_ids_c, sel.base, POOL, softmax_scale=scale, cu_seqlens=sel.cu_seqlens
    )
    assert target.dtype == torch.float32 and target.shape == sel.pool_ids_c.shape
    assert row_has_mass.dtype == torch.bool and row_has_mass.shape == sel.pool_ids_c.shape[:2]
    assert tail_mass.dtype == torch.float32 and tail_mass.shape == sel.pool_ids_c.shape[:2]
    assert torch.equal(row_has_mass, expected.row_has_mass)
    torch.testing.assert_close(target, expected.target.float(), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(tail_mass, expected.tail_mass.float(), rtol=1e-5, atol=1e-6)
    # Unused slots carry no target; a row with mass sums to one over its selected pools.
    assert torch.all(target[sel.pool_ids_c < 0] == 0)
    sums = target.sum(dim=-1)
    torch.testing.assert_close(sums[row_has_mass], torch.ones_like(sums[row_has_mass]))
    assert torch.all(target[~row_has_mass] == 0)


def test_sparse_pool_target_does_not_record_autograd_state():
    target_fn = _contract("_kpool_sparse_pool_target")
    sel = _selection(13)
    query, key = _attention(14, SEQ, 2, 2, 8, latent=True)
    query.requires_grad_()
    key.requires_grad_()
    with torch.enable_grad():
        outputs = target_fn(
            query,
            key,
            sel.token_topk_c,
            sel.pool_ids_c,
            softmax_scale=8**-0.5,
            index_topk=TOPK,
            pool_size=POOL,
        )
    assert all(not tensor.requires_grad for tensor in outputs)


def test_sparse_pool_target_counts_the_tail_only_in_the_denominator():
    """Head 0 spends most of its mass on the tail and so weighs less; the tail never enters
    the target. A per-head renormalization over the selected pools would weigh it fully."""
    target_fn = _contract("_kpool_sparse_pool_target")
    sel = _selection(15)
    query, key = _recency_attention(SEQ, sel.batch, [40.0, 0.0], latent=True, seed=16)
    kwargs = dict(softmax_scale=1.0, index_topk=TOPK, pool_size=POOL)
    target, row_has_mass, tail_mass = target_fn(
        query, key, sel.token_topk_c, sel.pool_ids_c, **kwargs
    )
    full = ref.sparse_pool_target(query, key, sel.pool_ids_c, sel.base, POOL, softmax_scale=1.0)
    per_head = ref.sparse_pool_target(
        query, key, sel.pool_ids_c, sel.base, POOL, softmax_scale=1.0, tail_in_denominator=False
    )
    torch.testing.assert_close(target, full.target.float(), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(tail_mass, full.tail_mass.float(), rtol=1e-5, atol=1e-6)
    # The contrast is real: the two definitions differ by much more than the tolerance.
    assert float((full.target - per_head.target).abs().max()) > 0.1
    # The tail comes from the token columns: without them, the target is the contrast.
    no_tail = sel.token_topk_c.clone()
    no_tail[..., TOPK:] = -1
    target_no_tail, _, tail_no_tail = target_fn(query, key, no_tail, sel.pool_ids_c, **kwargs)
    torch.testing.assert_close(target_no_tail, per_head.target.float(), rtol=1e-5, atol=1e-6)
    assert torch.all(tail_no_tail == 0)
    # Rows with a tail and a pool put mass on the tail; the target still sums to one.
    has_tail = (sel.token_topk_c[..., TOPK:] >= 0).any(dim=-1) & row_has_mass
    assert has_tail.any() and torch.all(tail_mass[has_tail] > 0)


def test_sparse_pool_target_rows_without_a_complete_pool_have_no_mass():
    target_fn = _contract("_kpool_sparse_pool_target")
    sel = _selection(17, batch=1, cu_seqlens=PACKED_CU)
    query, key = _attention(18, SEQ, 1, 2, 8, latent=True)
    target, row_has_mass, tail_mass = target_fn(
        query,
        key,
        sel.token_topk_c,
        sel.pool_ids_c,
        softmax_scale=8**-0.5,
        index_topk=TOPK,
        pool_size=POOL,
    )
    # The first P - 1 rows of each document attend only their tail.
    starts = ref.document_starts(SEQ, PACKED_CU)
    no_pool = torch.tensor([row - starts[row] < POOL - 1 for row in range(SEQ)])
    assert no_pool.sum() == 3 + 3 + 3
    assert not row_has_mass[0, no_pool].any()
    assert torch.all(target[0, no_pool] == 0)
    torch.testing.assert_close(tail_mass[0, no_pool], torch.ones(int(no_pool.sum())))
    assert row_has_mass[0, ~no_pool].all()


def test_sparse_pool_target_rows_whose_pool_mass_underflows_have_no_mass():
    """Row 9 sees pools 0 and 1 (tokens 0-7) and the tail 8-9. Its pool logits sit 200 nats
    below the tail's, so every pool mass underflows FP32 on both heads: no target, no row."""
    target_fn = _contract("_kpool_sparse_pool_target")
    seq_len, heads, dim, topk = 12, 2, 4, 8
    sel = _selection(19, seq_len=seq_len, batch=1, index_topk=topk)
    generator = _generator(20)
    key = _randn(generator, seq_len, 1, 1, dim)
    key[:, 0, 0, 0] = torch.where(torch.arange(seq_len) < 8, -200.0, 0.0)
    query = _randn(generator, seq_len, 1, heads, dim)
    query[..., 0] = 0.0
    query[9] = 0.0
    query[9, 0, :, 0] = 1.0
    target, row_has_mass, tail_mass = target_fn(
        query,
        key,
        sel.token_topk_c,
        sel.pool_ids_c,
        softmax_scale=1.0,
        index_topk=topk,
        pool_size=POOL,
    )
    assert sel.pool_ids_c[0, 9].tolist() == [0, 1]
    assert not row_has_mass[0, 9]
    assert torch.all(target[0, 9] == 0)
    torch.testing.assert_close(tail_mass[0, 9], torch.tensor(1.0))
    # The FP64 oracle keeps a tiny mass; the FP32 oracle underflows exactly like the helper.
    exact = ref.sparse_pool_target(query, key, sel.pool_ids_c, sel.base, POOL, softmax_scale=1.0)
    fp32 = ref.sparse_pool_target(
        query, key, sel.pool_ids_c, sel.base, POOL, softmax_scale=1.0, dtype=torch.float32
    )
    assert exact.row_has_mass[0, 9] and not fp32.row_has_mass[0, 9]
    others = torch.arange(seq_len) != 9
    assert torch.equal(row_has_mass[0, others], exact.row_has_mass[0, others])
    torch.testing.assert_close(
        target[0, others], exact.target[0, others].float(), rtol=1e-5, atol=1e-6
    )


@pytest.mark.parametrize("chunk_bytes", [ONE_ROW, ALL_ROWS], ids=["one-row", "all-rows"])
@pytest.mark.parametrize("latent", [True, False], ids=["latent-key", "per-head-keys"])
def test_sparse_pool_target_all_reduces_the_head_mass_over_tp(monkeypatch, latent, chunk_bytes):
    """Two TP ranks hold heads 0-1 and 2-3. A fake SUM all-reduce adds, in place, the buffer
    the other rank's call handed to its own all-reduce (recorded in a first pass), so the
    test does not depend on the buffer's layout. Rank 1's heads put far more mass on the
    tail, so the ranks' tail masses differ. Both ranks must hold the all-head target.
    """
    target_fn = _contract("_kpool_sparse_pool_target")
    sel = _selection(21)
    query, key = _recency_attention(SEQ, sel.batch, [0.0, 2.0, 30.0, 40.0], latent=latent, seed=22)
    rank_heads = ([0, 1], [2, 3])
    groups = (_FakeGroup(2, 0), _FakeGroup(2, 1))
    recorded = ([], [])  # per rank: the buffers its call all-reduced, before the reduction
    state = {}

    def fake_all_reduce(tensor, op=None, group=None, async_op=False):
        rank = state["rank"]
        assert group is groups[rank]
        assert op in (None, torch.distributed.ReduceOp.SUM)
        assert not async_op
        index = state["calls"]
        state["calls"] += 1
        if not state["reduce"]:
            recorded[rank].append(tensor.clone())
            return
        peer = recorded[1 - rank]
        assert index < len(peer), "the two ranks all-reduced a different number of times"
        assert peer[index].shape == tensor.shape and peer[index].dtype == tensor.dtype
        tensor.add_(peer[index])

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(
        torch.distributed, "get_world_size", lambda group=None: group.size() if group else 1
    )
    monkeypatch.setattr(dsa_module, "get_pg_size", lambda g=None: g.size() if g is not None else 1)

    def run(rank, reduce):
        state.update(rank=rank, reduce=reduce, calls=0)
        if not reduce:
            recorded[rank].clear()
        heads = rank_heads[rank]
        outputs = target_fn(
            query[:, :, heads],
            key if latent else key[:, :, heads],
            sel.token_topk_c,
            sel.pool_ids_c,
            softmax_scale=1.0,
            index_topk=TOPK,
            pool_size=POOL,
            tp_group=groups[rank],
            chunk_bytes=chunk_bytes,
        )
        if reduce:
            assert state["calls"] == len(recorded[1 - rank]) > 0
        return outputs

    # First pass: record what each rank all-reduces; second pass: reduce with the peer's.
    run(0, reduce=False)
    run(1, reduce=False)
    rank0 = run(0, reduce=True)
    rank1 = run(1, reduce=True)

    expected = ref.sparse_pool_target(query, key, sel.pool_ids_c, sel.base, POOL, softmax_scale=1.0)
    local_only = ref.sparse_pool_target(
        query, key, sel.pool_ids_c, sel.base, POOL, softmax_scale=1.0, heads=rank_heads[0]
    )
    assert float((expected.target - local_only.target).abs().max()) > 0.05
    for target, row_has_mass, tail_mass in (rank0, rank1):
        assert torch.equal(row_has_mass, expected.row_has_mass)
        torch.testing.assert_close(target, expected.target.float(), rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(tail_mass, expected.tail_mass.float(), rtol=1e-5, atol=1e-6)
    assert torch.equal(rank0[0], rank1[0])


# ---------------------------------------------------------------------------------------
# _kpool_selected_pool_scores
# ---------------------------------------------------------------------------------------


def _student_inputs(
    seed, scoring, *, seq_len=SEQ, batch=2, heads=4, dim=16, key_dtype=torch.bfloat16
):
    generator = _generator(seed)
    projection = _randn(generator, seq_len, batch, heads)
    if SCORING[scoring]["hf_scoring"]:
        # HF scoring: the BF16 projection upcast with the head-count scale only.
        weights = projection.bfloat16().float() * heads**-0.5
    else:
        weights = projection * (heads**-0.5 * dim**-0.5)
    return SimpleNamespace(
        q=_randn(generator, seq_len, batch, heads, dim, dtype=torch.bfloat16),
        weights=weights,
        k_pooled=_randn(generator, seq_len // POOL, batch, dim, dtype=key_dtype),
    )


@pytest.mark.parametrize("key_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("scoring", list(SCORING))
def test_selected_pool_scores_match_oracle(scoring, key_dtype):
    scores_fn = _contract("_kpool_selected_pool_scores")
    sel = _selection(31)
    x = _student_inputs(32, scoring, key_dtype=key_dtype)
    scores = scores_fn(x.q, x.weights, x.k_pooled, sel.pool_ids_c, **SCORING[scoring])
    expected = ref.selected_pool_scores(
        x.q, x.weights, x.k_pooled, sel.pool_ids_c, **SCORING[scoring]
    )
    valid = sel.pool_ids_c >= 0
    assert scores.dtype == torch.float32 and scores.shape == sel.pool_ids_c.shape
    assert torch.isfinite(scores).all()
    torch.testing.assert_close(scores[valid], expected[valid].float(), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("scoring", ["hf", "legacy_relu"])
def test_selected_pool_scores_are_the_selector_scores(scoring):
    """The student is the score the PyTorch selector ranks on, at the selected pools."""
    scores_fn = _contract("_kpool_selected_pool_scores")
    sel = _selection(33)
    x = _student_inputs(34, scoring)
    scores = scores_fn(x.q, x.weights, x.k_pooled, sel.pool_ids_c, **SCORING[scoring])
    if SCORING[scoring]["hf_scoring"]:
        dense = dsa_module._compute_kpool_hf_scores(x.q, x.k_pooled, x.weights)
    else:
        dense = dsa_module._compute_index_scores(x.q, x.weights, x.k_pooled, use_relu=True)
    selector = dense.gather(-1, sel.pool_ids_c.clamp_min(0))
    valid = sel.pool_ids_c >= 0
    torch.testing.assert_close(scores[valid], selector[valid], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("scoring", list(SCORING))
def test_selected_pool_scores_are_differentiable(scoring):
    scores_fn = _contract("_kpool_selected_pool_scores")
    sel = _selection(35)
    x = _student_inputs(36, scoring, key_dtype=torch.float32)
    q = x.q.float().requires_grad_()
    weights = x.weights.requires_grad_()
    k_pooled = x.k_pooled.requires_grad_()
    scores = scores_fn(q, weights, k_pooled, sel.pool_ids_c, **SCORING[scoring])
    assert scores.requires_grad
    # Only the selected slots are probed; callers mask the unused ones.
    probe = _randn(_generator(37), *scores.shape, dtype=torch.float64) * (sel.pool_ids_c >= 0)
    (scores.double() * probe).sum().backward()
    q_ref, w_ref, k_ref = (t.detach().double().requires_grad_() for t in (q, weights, k_pooled))
    expected = ref.selected_pool_scores(q_ref, w_ref, k_ref, sel.pool_ids_c, **SCORING[scoring])
    (expected * probe).sum().backward()
    _assert_close_scaled(q.grad, q_ref.grad, 1e-5, "dq")
    _assert_close_scaled(weights.grad, w_ref.grad, 1e-5, "dweights")
    _assert_close_scaled(k_pooled.grad, k_ref.grad, 1e-5, "dk_pooled")


# ---------------------------------------------------------------------------------------
# KPoolSparseIndexerLoss
# ---------------------------------------------------------------------------------------


def _loss_case(
    seed,
    scoring,
    *,
    dtype=torch.float32,
    key_dtype=None,
    seq_len=SEQ,
    batch=2,
    heads=4,
    dim=16,
    index_topk=TOPK,
    cu_seqlens=None,
):
    """Indexer inputs (leaves that require grad) and an oracle teacher for a selection."""
    sel = _selection(
        seed, seq_len=seq_len, batch=batch, index_topk=index_topk, cu_seqlens=cu_seqlens
    )
    generator = _generator(seed + 1)
    weights_scale = heads**-0.5 * (1.0 if SCORING[scoring]["hf_scoring"] else dim**-0.5)
    q = _randn(generator, seq_len, batch, heads, dim, dtype=dtype).requires_grad_()
    weights = _randn(generator, seq_len, batch, heads, dtype=dtype, scale=weights_scale)
    weights.requires_grad_()
    k_pooled = _randn(generator, seq_len // POOL, batch, dim, dtype=key_dtype or dtype)
    k_pooled.requires_grad_()
    query, key = _attention(seed + 2, seq_len, batch, 2, 8, latent=True)
    teacher = ref.sparse_pool_target(
        query, key, sel.pool_ids_c, sel.base, POOL, softmax_scale=8**-0.5, cu_seqlens=cu_seqlens
    )
    return SimpleNamespace(
        sel=sel,
        q=q,
        weights=weights,
        k_pooled=k_pooled,
        target=teacher.target.float(),
        row_has_mass=teacher.row_has_mass,
        rows=seq_len * batch,
        **SCORING[scoring],
    )


def _apply_loss(case, *, row_count, loss_coeff=0.37, chunk_bytes=ALL_ROWS, row_has_mass=None):
    loss_fn = _contract("KPoolSparseIndexerLoss")
    return loss_fn.apply(
        case.q,
        case.weights,
        case.k_pooled,
        case.sel.pool_ids_c,
        case.target,
        case.row_has_mass if row_has_mass is None else row_has_mass,
        row_count,
        loss_coeff,
        case.hf_scoring,
        case.use_relu,
        chunk_bytes,
    )


def _oracle_loss(case, *, row_count, loss_coeff=0.37, row_has_mass=None):
    """The oracle loss and its FP64 gradients for ``q``, ``weights`` and ``k_pooled``."""
    q, weights, k_pooled = (
        t.detach().double().requires_grad_() for t in (case.q, case.weights, case.k_pooled)
    )
    scores = ref.selected_pool_scores(
        q,
        weights,
        k_pooled,
        case.sel.pool_ids_c,
        hf_scoring=case.hf_scoring,
        use_relu=case.use_relu,
    )
    loss = ref.sparse_kl_loss(
        case.target.double(),
        scores,
        case.sel.pool_ids_c,
        case.row_has_mass if row_has_mass is None else row_has_mass,
        row_count=row_count,
        loss_coeff=loss_coeff,
    )
    loss.backward()
    return loss.detach(), (q.grad, weights.grad, k_pooled.grad)


@pytest.mark.parametrize("row_count_kind", ["int", "tensor"])
@pytest.mark.parametrize("chunk_bytes", [ONE_ROW, ALL_ROWS], ids=["one-row", "all-rows"])
@pytest.mark.parametrize("key_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("scoring", list(SCORING))
def test_sparse_indexer_loss_matches_oracle(scoring, key_dtype, chunk_bytes, row_count_kind):
    case = _loss_case(41, scoring, key_dtype=key_dtype)
    assert (~case.row_has_mass).any() and case.row_has_mass.any()
    rows = case.rows
    row_count = rows if row_count_kind == "int" else torch.tensor(float(rows))
    loss = _apply_loss(case, row_count=row_count, chunk_bytes=chunk_bytes)
    assert loss.dim() == 0 and loss.dtype == torch.float32
    expected, expected_grads = _oracle_loss(case, row_count=rows)
    torch.testing.assert_close(loss.double(), expected, rtol=1e-5, atol=1e-7)
    loss.backward()
    names = ("dq", "dweights", "dk_pooled")
    for name, tensor, grad in zip(names, (case.q, case.weights, case.k_pooled), expected_grads):
        assert tensor.grad is not None and tensor.grad.dtype == tensor.dtype, name
        rel = 2e-2 if tensor.dtype == torch.bfloat16 else 1e-4
        _assert_close_scaled(tensor.grad, grad, rel, name)


@pytest.mark.parametrize("per_token_loss", [False, True], ids=["mean", "per-token"])
@pytest.mark.parametrize("chunk_bytes", [ONE_ROW, ALL_ROWS], ids=["one-row", "all-rows"])
@pytest.mark.parametrize("scoring", list(SCORING))
def test_sparse_indexer_loss_fp64_gradcheck(scoring, chunk_bytes, per_token_loss):
    """Tiny FP64 shapes: 12 rows x 2 samples, three pool slots, two selected per row."""
    loss_fn = _contract("KPoolSparseIndexerLoss")
    case = _loss_case(
        43, scoring, dtype=torch.float64, seq_len=12, batch=2, heads=2, dim=4, index_topk=8
    )
    assert case.row_has_mass.any() and (case.sel.pool_ids_c < 0).any()
    # calculate_per_token_loss: the caller divides by 1 and supplies the sum semantics.
    row_count = 1 if per_token_loss else case.rows

    def loss_of(q, weights, k_pooled):
        return loss_fn.apply(
            q,
            weights,
            k_pooled,
            case.sel.pool_ids_c,
            case.target,
            case.row_has_mass,
            row_count,
            0.7,
            case.hf_scoring,
            case.use_relu,
            chunk_bytes,
        )

    # A gradcheck needs the Function to keep FP64 inputs in FP64 (an FP32 loss and FP32
    # internals make the numerical Jacobian noise-dominated); production BF16/FP32 inputs
    # still give an FP32 loss.
    loss_dtype = loss_of(case.q, case.weights, case.k_pooled).dtype
    assert loss_dtype == torch.float64, (
        f"KPoolSparseIndexerLoss returned a {loss_dtype} loss for FP64 inputs; the FP64 "
        "gradcheck requires FP64 arithmetic for FP64 inputs"
    )
    assert torch.autograd.gradcheck(
        loss_of,
        (case.q, case.weights, case.k_pooled),
        eps=1e-6,
        atol=1e-7,
        rtol=1e-5,
        nondet_tol=0.0,
    )


def test_sparse_indexer_loss_row_count_divides_the_row_sum():
    case = _loss_case(45, "hf")
    loss_sum = _apply_loss(case, row_count=1)
    grads_sum = torch.autograd.grad(loss_sum, (case.q, case.weights, case.k_pooled))
    loss_mean = _apply_loss(case, row_count=case.rows)
    grads_mean = torch.autograd.grad(loss_mean, (case.q, case.weights, case.k_pooled))
    torch.testing.assert_close(loss_mean * case.rows, loss_sum, rtol=1e-6, atol=0)
    for mean, total in zip(grads_mean, grads_sum):
        torch.testing.assert_close(mean * case.rows, total, rtol=1e-5, atol=1e-8)
    # Rows without mass add nothing but still count: dropping one row with mass from the
    # mask lowers the sum by exactly that row's term.
    row = case.row_has_mass.nonzero()[0].tolist()
    reduced_mask = case.row_has_mass.clone()
    reduced_mask[row[0], row[1]] = False
    loss_reduced = _apply_loss(case, row_count=1, row_has_mass=reduced_mask)
    expected_full, _ = _oracle_loss(case, row_count=1)
    expected_reduced, _ = _oracle_loss(case, row_count=1, row_has_mass=reduced_mask)
    torch.testing.assert_close(
        (loss_sum - loss_reduced).double(), expected_full - expected_reduced, rtol=1e-4, atol=2e-5
    )


@pytest.mark.parametrize("empty", ["no-rows", "no-pools", "no-mass"])
def test_sparse_indexer_loss_without_terms_returns_zero_loss_and_zero_grads(empty):
    loss_fn = _contract("KPoolSparseIndexerLoss")
    batch, heads, dim, budget = 2, 4, 8, TOPK // POOL
    rows = {"no-rows": 0, "no-pools": POOL - 1, "no-mass": SEQ}[empty]
    slots = {"no-rows": 6, "no-pools": 0, "no-mass": 6}[empty]
    generator = _generator(51)
    q = _randn(generator, rows, batch, heads, dim).requires_grad_()
    weights = _randn(generator, rows, batch, heads).requires_grad_()
    k_pooled = _randn(generator, slots, batch, dim, dtype=torch.bfloat16).requires_grad_()
    if empty == "no-mass":
        sel = _selection(52)
        pool_ids = sel.pool_ids_c
    else:
        pool_ids = torch.full((batch, rows, budget), -1, dtype=torch.int64)
    target = torch.zeros(batch, rows, budget)
    row_has_mass = torch.zeros(batch, rows, dtype=torch.bool)
    for hf_scoring in (True, False):
        for tensor in (q, weights, k_pooled):
            tensor.grad = None
        loss = loss_fn.apply(
            q,
            weights,
            k_pooled,
            pool_ids,
            target,
            row_has_mass,
            max(rows * batch, 1),
            0.5,
            hf_scoring,
            True,
            ALL_ROWS,
        )
        assert loss.dim() == 0 and loss.dtype == torch.float32
        assert float(loss) == 0.0
        loss.backward()
        for name, tensor in (("q", q), ("weights", weights), ("k_pooled", k_pooled)):
            assert tensor.grad is not None, f"{name}: no gradient on empty input"
            assert tensor.grad.shape == tensor.shape and tensor.grad.dtype == tensor.dtype
            assert torch.all(tensor.grad == 0), f"{name}: non-zero gradient on empty input"


@pytest.mark.parametrize("scoring", ["hf", "legacy_relu"])
def test_sparse_indexer_loss_unused_slots_give_no_gradient(scoring):
    """No row selects pool 0, while unused (-1) slots exist on many rows: an unused slot
    must not leak a gradient into pool 0's key (or any other) through a clamped index."""
    loss_fn = _contract("KPoolSparseIndexerLoss")
    case = _loss_case(53, scoring)
    sel = case.sel
    pool_ids = sel.pool_ids.masked_fill(sel.pool_ids == 0, -1)
    token_topk = ref.expand_selection(pool_ids, sel.base, POOL, TOPK)
    pool_ids_c, _ = ref.canonical_slots(token_topk, pool_ids, index_topk=TOPK, pool_size=POOL)
    assert (pool_ids_c < 0).any() and not (pool_ids_c == 0).any()
    query, key = _attention(54, SEQ, sel.batch, 2, 8, latent=True)
    teacher = ref.sparse_pool_target(query, key, pool_ids_c, sel.base, POOL, softmax_scale=0.5)
    loss = loss_fn.apply(
        case.q,
        case.weights,
        case.k_pooled,
        pool_ids_c,
        teacher.target.float(),
        teacher.row_has_mass,
        case.rows,
        0.37,
        case.hf_scoring,
        case.use_relu,
        ALL_ROWS,
    )
    loss.backward()
    assert torch.all(case.k_pooled.grad[0] == 0)
    assert case.k_pooled.grad[1:].abs().sum() > 0
    # Rows without any selected pool give no query or weight gradient.
    no_pool = ~(pool_ids_c >= 0).any(dim=-1)  # [b, sq]
    assert no_pool.any()
    assert torch.all(case.q.grad.transpose(0, 1)[no_pool] == 0)
    assert torch.all(case.weights.grad.transpose(0, 1)[no_pool] == 0)


def test_sparse_indexer_loss_accumulates_pooled_key_gradients_deterministically():
    """Many rows share few pools; the pooled-key gradient sums them without atomics."""
    case = _loss_case(55, "hf", seq_len=64, batch=2, heads=4, dim=16, key_dtype=torch.float32)
    _, expected = _oracle_loss(case, row_count=case.rows)

    def run():
        for tensor in (case.q, case.weights, case.k_pooled):
            tensor.grad = None
        _apply_loss(case, row_count=case.rows, chunk_bytes=4096).backward()
        return [t.grad.clone() for t in (case.q, case.weights, case.k_pooled)]

    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        deterministic = [run(), run()]
        torch.use_deterministic_algorithms(False)
        default = [run(), run()]
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
    names = ("dq", "dweights", "dk_pooled")
    for runs in (deterministic, default):
        for name, first, second in zip(names, *runs):
            assert torch.equal(first, second), f"{name} differs between two identical runs"
    for name, det, other, exp in zip(names, deterministic[0], default[0], expected):
        _assert_close_scaled(det, exp, 1e-4, f"{name} (deterministic algorithms)")
        _assert_close_scaled(other, exp, 1e-4, f"{name} (default algorithms)")


def test_sparse_indexer_loss_backward_scans_no_leading_dimension(monkeypatch):
    """The backward runs no cumulative scan over a leading dimension.

    PyTorch's CUDA scan over a leading dimension runs one serial thread per remaining
    column: an FP64 prefix over ``[slots, dim]`` made the backward take seconds per call at
    1M tokens. The gradients stay those of the oracle.
    """
    case = _loss_case(55, "hf", seq_len=64, batch=2, heads=4, dim=16, key_dtype=torch.float32)
    _, expected = _oracle_loss(case, row_count=case.rows)
    real_cumsum = torch.cumsum
    leading = []

    def spy(input, dim, *args, **kwargs):
        if input.ndim > 1 and dim % input.ndim != input.ndim - 1:
            leading.append((tuple(input.shape), dim))
        return real_cumsum(input, dim, *args, **kwargs)

    monkeypatch.setattr(torch, "cumsum", spy)
    _apply_loss(case, row_count=case.rows, chunk_bytes=4096).backward()
    monkeypatch.undo()
    assert not leading, f"cumsum over a leading dimension: {leading}"
    grads = (case.q.grad, case.weights.grad, case.k_pooled.grad)
    for name, got, exp in zip(("dq", "dweights", "dk_pooled"), grads, expected):
        _assert_close_scaled(got, exp, 1e-4, name)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_pooled_key_gradient_accumulation_sums_each_key_in_a_fixed_order(dtype):
    """``_kpool_accumulate_pool_key_grads`` adds every slot to its key, empty keys included.

    Duplicate ids, keys that no slot touches (zero-length segments), padding rows and a
    gradient buffer that already holds an earlier chunk; the result equals an FP64
    ``index_add_`` and repeats bitwise.
    """
    gen = torch.Generator().manual_seed(7)
    keys, dim, slots = 23, 5, 61
    flat_ids = torch.randint(0, keys - 6, (slots,), generator=gen)  # the last 6 keys stay empty
    flat_ids[:9] = 3  # one long run
    contributions = torch.randn(slots, dim, generator=gen, dtype=dtype)
    earlier = torch.randn(keys, dim, generator=gen, dtype=dtype)
    expected = earlier.double().index_add(0, flat_ids, contributions.double())

    def run():
        grad = earlier.clone()
        key_ids = dsa_module._kpool_accumulate_pool_key_grads(grad, flat_ids, contributions, None)
        assert torch.equal(key_ids, torch.arange(keys))
        reused = dsa_module._kpool_accumulate_pool_key_grads(
            torch.zeros_like(grad), flat_ids, contributions, key_ids
        )
        assert reused is key_ids
        return grad

    first, second = run(), run()
    assert torch.equal(first, second)
    assert first.dtype == dtype
    torch.testing.assert_close(first.double(), expected, rtol=1e-6, atol=1e-6)
    assert torch.equal(first[keys - 6 :], earlier[keys - 6 :])


def test_sparse_indexer_loss_saves_only_its_inputs():
    """The backward recomputes per chunk: no score or probability tensor is saved."""
    case = _loss_case(57, "hf")
    allowed = {
        tuple(t.shape)
        for t in (
            case.q,
            case.weights,
            case.k_pooled,
            case.sel.pool_ids_c,
            case.target,
            case.row_has_mass,
        )
    }
    saved = []

    def pack(tensor):
        saved.append((tuple(tensor.shape), tensor.numel()))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        loss = _apply_loss(case, row_count=case.rows)
    loss.backward()
    assert saved, "KPoolSparseIndexerLoss saved nothing through ctx.save_for_backward"
    # A scalar (or empty) bookkeeping tensor, such as the row count, is fine.
    unexpected = [shape for shape, numel in saved if shape not in allowed and numel > 1]
    assert not unexpected, f"saved tensors outside the contract: {unexpected}"


# ---------------------------------------------------------------------------------------
# The whole chain: compression, canonical slots, teacher, loss
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("packed", [False, True], ids=["plain", "packed"])
@pytest.mark.parametrize("scoring", ["hf", "legacy_relu"])
@pytest.mark.parametrize("profile", ["fp32", "key_dtype"])
def test_sparse_kpool_loss_chain_matches_oracle(profile, scoring, packed):
    canonical = _contract("_kpool_canonical_slots")
    target_fn = _contract("_kpool_sparse_pool_target")
    loss_fn = _contract("KPoolSparseIndexerLoss")
    cu = PACKED_CU if packed else None
    batch, heads, dim, main_heads, main_dim = (1 if packed else 2), 4, 16, 4, 24
    generator = _generator(61)
    weights_scale = heads**-0.5 * (1.0 if SCORING[scoring]["hf_scoring"] else dim**-0.5)
    leaves = dict(
        q=_randn(generator, SEQ, batch, heads, dim),
        weights=_randn(generator, SEQ, batch, heads, scale=weights_scale),
        k=_randn(generator, SEQ, batch, dim, dtype=torch.bfloat16),
        gate_score=_randn(generator, SEQ, batch, dim, dtype=torch.bfloat16),
        ape=_randn(generator, POOL, dim, scale=0.5),
    )
    query, key = _attention(62, SEQ, batch, main_heads, main_dim, latent=True, dtype=torch.bfloat16)
    scale = main_dim**-0.5
    sel = _selection(63, batch=batch, cu_seqlens=cu, interleave_pads=True)
    # Padded query rows (the last two) leave the loss through query_valid_rows.
    row_mask = torch.ones(batch, SEQ, dtype=torch.bool)
    if packed:
        row_mask[:, -2:] = False
    row_count = int(row_mask.sum())
    coeff = 0.25

    # Production.
    prod = {name: t.clone().requires_grad_() for name, t in leaves.items()}
    key_dtype = profile == "key_dtype"
    if packed:
        k_pooled, base = dsa_module._kpool_compress_keys_per_seg(
            prod["k"],
            prod["gate_score"],
            prod["ape"],
            POOL,
            torch.tensor(cu),
            kpool_key_dtype_compression=key_dtype,
        )
        assert torch.equal(base, sel.base)
    else:
        k_pooled = dsa_module._kpool_compress_keys(
            prod["k"], prod["gate_score"], prod["ape"], POOL, kpool_key_dtype_compression=key_dtype
        )
    pool_ids_c, token_topk_c = canonical(
        sel.token_topk, sel.pool_ids, index_topk=TOPK, pool_size=POOL
    )
    target, row_has_mass, _ = target_fn(
        query, key, token_topk_c, pool_ids_c, softmax_scale=scale, index_topk=TOPK, pool_size=POOL
    )
    loss = loss_fn.apply(
        prod["q"],
        prod["weights"],
        k_pooled,
        pool_ids_c,
        target,
        row_has_mass & row_mask,
        row_count,
        coeff,
        SCORING[scoring]["hf_scoring"],
        True,
        ALL_ROWS,
    )
    loss.backward()

    # Oracle.
    oracle = {name: t.clone().requires_grad_() for name, t in leaves.items()}
    expected = ref.sparse_indexer_loss(
        **oracle,
        query=query,
        key=key,
        pool_ids=sel.pool_ids_c,
        pool_size=POOL,
        softmax_scale=scale,
        profile=profile,
        loss_coeff=coeff,
        row_count=row_count,
        cu_seqlens=cu,
        row_mask=row_mask,
        **SCORING[scoring],
    )
    expected.loss.backward()

    valid_slots = sel.base >= 0
    torch.testing.assert_close(
        k_pooled[valid_slots].float(), expected.k_pooled[valid_slots].float(), rtol=1e-2, atol=1e-2
    )
    assert torch.equal(row_has_mass & row_mask, expected.row_has_mass)
    assert expected.row_has_mass.any() and (~expected.row_has_mass).any()
    torch.testing.assert_close(loss.double(), expected.loss.detach(), rtol=1e-4, atol=1e-7)
    for name in ("q", "weights"):
        _assert_close_scaled(prod[name].grad, oracle[name].grad, 1e-3, f"d{name}")
    for name in ("k", "gate_score", "ape"):
        _assert_close_scaled(prod[name].grad, oracle[name].grad, 3e-2, f"d{name}")


# ---------------------------------------------------------------------------------------
# The tail mass p_E: its own per-layer tracker slots, reduction and training-log reporting
# ---------------------------------------------------------------------------------------

_TAIL_KEYS = ("tail_mass_sums", "tail_mass_rows")


def _fake_kpool_layer(*, layer_number, num_layers, mtp_num_layers, softmax_scale, per_token):
    """The attributes ``DSAttention._kpool_sparse_indexer_loss`` reads from its layer."""
    return SimpleNamespace(
        indexer=SimpleNamespace(index_kpool=POOL),
        index_topk=TOPK,
        softmax_scale=softmax_scale,
        layer_number=layer_number,
        config=SimpleNamespace(
            calculate_per_token_loss=per_token,
            dsa_indexer_rotate_activation=False,
            dsa_indexer_hf_scoring=True,
            dsa_indexer_scoring_relu=True,
            num_layers=num_layers,
            mtp_num_layers=mtp_num_layers,
        ),
    )


@pytest.mark.parametrize("per_token", [False, True], ids=["mean", "per-token"])
@pytest.mark.parametrize("layout", ["plain", "packed-padded"])
@pytest.mark.parametrize(
    ("layer_number", "mtp"), [(3, False), (1, True)], ids=["decoder-layer", "mtp-layer"]
)
def test_sparse_indexer_loss_records_the_mean_tail_mass_apart_from_the_loss(
    monkeypatch, layout, per_token, layer_number, mtp
):
    """The layer records p_E's sum and row count over the loss's rows in its own slots.

    The rows are the loss's: rows with pool mass, and only the valid query rows of a padded
    pack. Their mean p_E is the oracle's. The indexer-loss slots (scaled and raw) are left
    bit for bit as they were, and an MTP layer records at ``layer_number + num_layers``.
    """
    helper = dsa_module.DSAIndexerLossLoggingHelper
    num_layers, mtp_num_layers = 4, 1
    packed = layout == "packed-padded"
    cu = PACKED_CU if packed else None
    batch, heads, dim, main_heads, main_dim = (1 if packed else 2), 4, 16, 4, 24
    generator = _generator(81)
    q = _randn(generator, SEQ, batch, heads, dim)
    weights = _randn(generator, SEQ, batch, heads, scale=heads**-0.5)
    k_pooled = _randn(generator, SEQ // POOL, batch, dim, dtype=torch.bfloat16)
    query, key = _attention(82, SEQ, batch, main_heads, main_dim, latent=True)
    scale = main_dim**-0.5
    sel = _selection(83, batch=batch, cu_seqlens=cu, interleave_pads=True)
    query_valid_rows = None
    if packed:
        query_valid_rows = torch.ones(batch, SEQ, dtype=torch.bool)
        query_valid_rows[:, -2:] = False

    seeded = {"values": torch.arange(1.0, 6.0), "raw_values": torch.arange(10.0, 15.0)}
    monkeypatch.setattr(
        helper, "tracker", {name: tensor.clone() for name, tensor in seeded.items()}
    )
    reduce_group, avg_group = object(), object()
    layer = _fake_kpool_layer(
        layer_number=layer_number + (num_layers if mtp else 0),
        num_layers=num_layers,
        mtp_num_layers=mtp_num_layers,
        softmax_scale=scale,
        per_token=per_token,
    )
    loss = dsa_module.DSAttention._kpool_sparse_indexer_loss(
        layer,
        q=q,
        weights=weights,
        k_pooled=k_pooled,
        topk_indices=sel.token_topk,
        pool_ids=sel.pool_ids,
        query=query,
        key=key,
        query_valid_rows=query_valid_rows,
        indexer_loss_coeff=0.25,
        tp_group=None,
        metric_num_layers=num_layers + mtp_num_layers,
        metric_reduce_group=reduce_group,
        metric_avg_group=avg_group,
    )
    assert torch.isfinite(loss) and float(loss) > 0

    tracker = helper.tracker
    for name, tensor in seeded.items():
        assert torch.equal(tracker[name], tensor), f"recording p_E changed the {name} slots"
    assert tracker["reduce_group"] is reduce_group and tracker["avg_group"] is avg_group

    teacher = ref.sparse_pool_target(
        query, key, sel.pool_ids_c, sel.base, POOL, softmax_scale=scale, cu_seqlens=cu
    )
    rows = teacher.row_has_mass
    if query_valid_rows is not None:
        rows = rows & query_valid_rows
    expected_sum = (teacher.tail_mass * rows).sum()
    expected_rows = rows.sum().double()
    slot = layer.layer_number - 1
    sums, counts = (tracker[name] for name in _TAIL_KEYS)
    assert sums.shape == counts.shape == (num_layers + mtp_num_layers,)
    others = torch.arange(sums.numel()) != slot
    assert torch.all(sums[others] == 0) and torch.all(counts[others] == 0)
    assert float(counts[slot]) == float(expected_rows)
    torch.testing.assert_close(sums[slot].double(), expected_sum, rtol=1e-5, atol=1e-6)
    expected_mean = expected_sum / expected_rows
    torch.testing.assert_close(
        (sums[slot] / counts[slot]).double(), expected_mean, rtol=1e-5, atol=1e-6
    )
    # The excluded rows matter: rows without a complete pool have p_E = 1, and the padded
    # rows have p_E too, so a mean over every row (or over the valid rows) differs.
    no_mass = ~teacher.row_has_mass
    assert no_mass.any() and torch.all(teacher.tail_mass[no_mass] == 1)
    every_row_mean = teacher.tail_mass.mean()
    assert abs(float(every_row_mean - expected_mean)) > 1e-3
    if query_valid_rows is not None:
        unpadded = teacher.row_has_mass
        unpadded_mean = (teacher.tail_mass * unpadded).sum() / unpadded.sum()
        assert abs(float(unpadded_mean - expected_mean)) > 1e-4

    # A second call (the next microbatch) adds to the same slots.
    dsa_module.DSAttention._kpool_sparse_indexer_loss(
        layer,
        q=q,
        weights=weights,
        k_pooled=k_pooled,
        topk_indices=sel.token_topk,
        pool_ids=sel.pool_ids,
        query=query,
        key=key,
        query_valid_rows=query_valid_rows,
        indexer_loss_coeff=0.25,
        tp_group=None,
        metric_num_layers=num_layers + mtp_num_layers,
        metric_reduce_group=reduce_group,
        metric_avg_group=avg_group,
    )
    assert float(counts[slot]) == 2 * float(expected_rows)
    torch.testing.assert_close(sums[slot].double(), 2 * expected_sum, rtol=1e-5, atol=1e-6)
    for name, tensor in seeded.items():
        assert torch.equal(tracker[name], tensor), f"recording p_E changed the {name} slots"


def _fixed_tracker(values, raw_values, sums, rows, **extra):
    """A tracker as ``initialize_dsa_metric_tracker`` leaves it, on the host."""
    tracker = {
        "values": torch.tensor(values),
        "raw_values": torch.tensor(raw_values),
        "tail_mass_sums": torch.tensor(sums),
        "tail_mass_rows": torch.tensor(rows),
        "agreed_size": len(values),
        "agreed_size_pp_group": extra.pop("pp_group"),
    }
    tracker.update(extra)
    return tracker


class _Writer:
    def __init__(self):
        self.scalars = {}

    def add_scalar(self, name, value, iteration):
        self.scalars[name] = (float(value), iteration)


@pytest.mark.parametrize("cp_group_key", ["avg_group", "reduce_group"])
def test_tail_mass_follows_the_indexer_loss_reduction_and_logs_its_row_mean(
    monkeypatch, cp_group_key
):
    """``track_indexer_metrics`` reduces the tail-mass sums and rows with the loss and logs
    their ratio: per layer, and averaged over the layers that recorded rows.

    Two CP ranks, as one sees it: this rank and a peer with other sums and row counts. The CP
    reduction (an average for the mean loss, a sum for the per-token loss) and the DP average
    scale the sums and the rows alike, so each logged value is the exact row mean over both
    ranks. The indexer loss logs as before; the loss scale does not touch the ratio.
    """
    helper = dsa_module.DSAIndexerLossLoggingHelper
    pp_group, dp_group, cp_group = object(), object(), object()
    # Layers 1 and 3 compute (the MTP layer 5 too); 2 and 4 are skip layers.
    local = dict(sums=[3.0, 0.0, 1.0, 0.0, 6.0], rows=[10.0, 0.0, 4.0, 0.0, 12.0])
    peer = dict(sums=[5.0, 0.0, 2.0, 0.0, 2.0], rows=[10.0, 0.0, 6.0, 0.0, 8.0])
    tracker = _fixed_tracker(
        [0.4, 0.0, 0.2, 0.0, 0.6],
        [4.0, 0.0, 2.0, 0.0, 6.0],
        local["sums"],
        local["rows"],
        pp_group=pp_group,
        tail_mass_recorded=True,
        raw_loss_recorded=True,
        **{cp_group_key: cp_group},
    )
    monkeypatch.setattr(helper, "tracker", tracker)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    peers = {
        id(tracker["values"]): torch.tensor([0.2, 0.0, 0.4, 0.0, 0.2]),
        id(tracker["raw_values"]): torch.tensor([2.0, 0.0, 4.0, 0.0, 2.0]),
        id(tracker["tail_mass_sums"]): torch.tensor(peer["sums"]),
        id(tracker["tail_mass_rows"]): torch.tensor(peer["rows"]),
    }
    calls = []

    def all_reduce(tensor, op=None, group=None):
        calls.append((id(tensor), group, op))
        if group is cp_group:
            total = tensor + peers[id(tensor)]
            tensor.copy_(total / 2 if op == torch.distributed.ReduceOp.AVG else total)

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    writer, total_loss_dict = _Writer(), {}
    helper.track_indexer_metrics(
        loss_scale=0.5,
        iteration=9,
        writer=writer,
        total_loss_dict=total_loss_dict,
        num_layers=5,
        num_indexer_layers=3,
        pp_group=pp_group,
        dp_group=dp_group,
    )

    # Every metric tensor takes the same path: CP, then PP, then the DP average.
    cp_op = torch.distributed.ReduceOp.AVG if cp_group_key == "avg_group" else None
    for key in ("values", "raw_values") + _TAIL_KEYS:
        path = [(group, op) for tensor_id, group, op in calls if tensor_id == id(tracker[key])]
        assert path == [
            (cp_group, cp_op),
            (pp_group, None),
            (dp_group, torch.distributed.ReduceOp.AVG),
        ], f"{key} took {path}"

    sums = torch.tensor(local["sums"]) + torch.tensor(peer["sums"])
    rows = torch.tensor(local["rows"]) + torch.tensor(peer["rows"])
    per_layer = {layer: float(sums[layer - 1] / rows[layer - 1]) for layer in (1, 3, 5)}
    average = sum(per_layer.values()) / len(per_layer)
    assert set(writer.scalars) == {
        "indexer loss",
        "indexer raw loss",
        "indexer tail mass",
        "indexer tail mass layer 1",
        "indexer tail mass layer 3",
        "indexer tail mass layer 5",
    }
    for layer, value in per_layer.items():
        assert writer.scalars[f"indexer tail mass layer {layer}"] == (
            pytest.approx(value, rel=1e-6),
            9,
        )
    assert writer.scalars["indexer tail mass"][0] == pytest.approx(average, rel=1e-6)
    torch.testing.assert_close(
        total_loss_dict["indexer tail mass"], torch.tensor(average), rtol=1e-6, atol=0
    )
    # The indexer loss: the CP-reduced values, scaled by the loss scale, over three layers.
    cp_values = torch.tensor([0.4, 0.0, 0.2, 0.0, 0.6]) + torch.tensor([0.2, 0.0, 0.4, 0.0, 0.2])
    if cp_group_key == "avg_group":
        cp_values = cp_values / 2
    torch.testing.assert_close(
        total_loss_dict["indexer loss"], (cp_values * 0.5).sum() / 3, rtol=1e-6, atol=0
    )
    for key in ("values", "raw_values") + _TAIL_KEYS:
        assert torch.count_nonzero(tracker[key]) == 0, f"{key} was not cleared after logging"


def test_tail_mass_is_not_reduced_or_logged_without_a_writer(monkeypatch):
    """Runs without a trainable KPool layer (token DSA, a frozen KPool) log nothing new.

    A fixed tracker owns the tail-mass storage on every PP rank, but nothing recorded into
    it: the PP-agreed decision is "disabled", so its tensors join no collective and no
    "indexer tail mass" entry appears.
    """
    helper = dsa_module.DSAIndexerLossLoggingHelper
    pp_group, dp_group = object(), object()
    tracker = _fixed_tracker([0.5, 0.0], [5.0, 0.0], [0.0, 0.0], [0.0, 0.0], pp_group=pp_group)
    monkeypatch.setattr(helper, "tracker", tracker)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    reduced = []
    monkeypatch.setattr(
        torch.distributed,
        "all_reduce",
        lambda tensor, op=None, group=None: reduced.append(id(tensor)),
    )
    writer, total_loss_dict = _Writer(), {}
    helper.track_indexer_metrics(
        loss_scale=1.0,
        iteration=1,
        writer=writer,
        total_loss_dict=total_loss_dict,
        num_layers=2,
        num_indexer_layers=1,
        pp_group=pp_group,
        dp_group=dp_group,
    )
    assert tracker["tail_mass_enabled"] is False
    assert id(tracker["tail_mass_sums"]) not in reduced
    assert id(tracker["tail_mass_rows"]) not in reduced
    assert set(writer.scalars) == {"indexer loss"}
    assert set(total_loss_dict) == {"indexer loss"}


# ---------------------------------------------------------------------------------------
# DSAIndexer: parameters and kpool_pool_keys
# ---------------------------------------------------------------------------------------


class _Linear(torch.nn.Linear):
    """Stands in for TELinear: parameters in ``params_dtype``, returns ``(output, bias)``."""

    def __init__(self, input_size, output_size, *, config, **_unused):
        super().__init__(input_size, output_size, bias=False, dtype=config.params_dtype)

    def forward(self, x):
        return super().forward(x), None


class _LayerNorm(torch.nn.LayerNorm):
    """Stands in for TENorm as the indexer's key norm."""

    def __init__(self, config, hidden_size, eps):
        super().__init__(hidden_size, eps=eps, dtype=config.params_dtype)


def _indexer_config(*, trainable, key_dtype_compression=False, params_dtype=torch.bfloat16):
    return MLATransformerConfig(
        num_layers=2,
        hidden_size=64,
        num_attention_heads=4,
        num_query_groups=4,
        kv_channels=8,
        multi_latent_attention=True,
        experimental_attention_variant="dsa",
        q_lora_rank=16,
        kv_lora_rank=16,
        qk_head_dim=8,
        qk_pos_emb_head_dim=0,
        v_head_dim=8,
        rope_type="rope",
        rotary_base=10000,
        apply_rope_fusion=False,
        dsa_indexer_n_heads=4,
        dsa_indexer_head_dim=8,
        dsa_indexer_topk=8,
        dsa_indexer_kpool=POOL,
        dsa_indexer_loss_coeff=0.1 if trainable else 0.0,
        dsa_indexer_use_sparse_loss=trainable,
        dsa_indexer_rotate_activation=False,
        dsa_indexer_scoring_relu=True,
        dsa_indexer_kpool_key_dtype_compression=key_dtype_compression,
        dsa_kernel_backend="none",
        attention_backend=AttnBackend.unfused,
        add_bias_linear=False,
        normalization="RMSNorm",
        bf16=params_dtype == torch.bfloat16,
        params_dtype=params_dtype,
        use_cpu_initialization=True,
        init_method=init_method_normal(0.02),
    )


def _build_indexer(config, *, seed=71):
    submodules = dsa_module.DSAIndexerSubmodules(
        linear_wq_b=_Linear, linear_wk=_Linear, k_norm=_LayerNorm, linear_weights_proj=_Linear
    )
    groups = SimpleNamespace(tp=_FakeGroup(1, 0), cp=_FakeGroup(1, 0))
    indexer = dsa_module.DSAIndexer(config, submodules, groups)
    generator = _generator(seed)
    with torch.no_grad():
        ape = indexer.index_kpool_compress_ape
        gate = indexer.index_kpool_compress_gate
        ape.copy_(_randn(generator, *ape.shape, scale=0.5))
        gate.copy_(_randn(generator, *gate.shape, dtype=gate.dtype, scale=0.3))
    return indexer


def test_trainable_kpool_parameters_require_grad_and_frozen_ones_do_not():
    frozen = _build_indexer(_indexer_config(trainable=False))
    assert all(not p.requires_grad for p in frozen.parameters())
    trainable = _build_indexer(_indexer_config(trainable=True))
    assert all(p.requires_grad for p in trainable.parameters())
    for indexer in (frozen, trainable):
        ape, gate = indexer.index_kpool_compress_ape, indexer.index_kpool_compress_gate
        assert ape.dtype == torch.float32 and getattr(ape, "keep_in_fp32", False)
        assert ape.shape == (POOL, 8) and gate.shape == (8, 64)
        assert gate.dtype == torch.bfloat16
        for param in (ape, gate):
            assert getattr(param, "average_gradients_across_tp_domain", False)
    assert getattr(trainable.index_kpool_compress_ape, "use_muon", None) is False


@pytest.mark.parametrize("packed", [False, True], ids=["plain", "packed"])
@pytest.mark.parametrize("key_dtype_compression", [False, True], ids=["fp32", "key-dtype"])
def test_kpool_pool_keys_equal_the_selector_compression(key_dtype_compression, packed):
    indexer = _build_indexer(
        _indexer_config(trainable=True, key_dtype_compression=key_dtype_compression)
    )
    pool_keys = getattr(indexer, "kpool_pool_keys", None)
    if pool_keys is None:
        pytest.fail("DSAIndexer.kpool_pool_keys is missing: phase-1 contract method")
    generator = _generator(72)
    k = _randn(generator, SEQ, 1 if packed else 2, 8, dtype=torch.bfloat16)
    gate = _randn(generator, *k.shape, dtype=torch.bfloat16)
    ape = indexer.index_kpool_compress_ape
    cu = torch.tensor(PACKED_CU, dtype=torch.int32) if packed else None
    with torch.no_grad():
        k_pooled, base = pool_keys(k, gate, cu_seqlens=cu)
        if packed:
            expected, expected_base = dsa_module._kpool_compress_keys_per_seg(
                k, gate, ape, POOL, cu, kpool_key_dtype_compression=key_dtype_compression
            )
        else:
            expected = dsa_module._kpool_compress_keys(
                k, gate, ape, POOL, kpool_key_dtype_compression=key_dtype_compression
            )
            expected_base = torch.arange(SEQ // POOL) * POOL
    assert not k_pooled.requires_grad
    assert torch.equal(k_pooled, expected)
    assert torch.equal(base.long(), expected_base.long())
    oracle_base = ref.pool_token_base(SEQ, POOL, PACKED_CU if packed else None)
    assert torch.equal(base.long(), oracle_base)
    profile = "key_dtype" if key_dtype_compression else "fp32"
    oracle = ref.compress_keys(k, gate, ape.detach(), oracle_base, POOL, profile=profile)
    valid = oracle_base >= 0
    # One BF16 rounding apart at most.
    torch.testing.assert_close(
        k_pooled[valid].float(), oracle[valid].float(), rtol=2**-7, atol=2**-7
    )


@pytest.mark.parametrize("key_dtype_compression", [False, True], ids=["fp32", "key-dtype"])
def test_kpool_pool_keys_pool_then_gather_equals_gather_then_pool(key_dtype_compression):
    """A zigzag CP rank's local tokens (chunks r and 2·cp - 1 - r, pool-aligned) compress to
    exactly the matching pools of the whole sequence."""
    indexer = _build_indexer(
        _indexer_config(trainable=True, key_dtype_compression=key_dtype_compression)
    )
    pool_keys = getattr(indexer, "kpool_pool_keys", None)
    if pool_keys is None:
        pytest.fail("DSAIndexer.kpool_pool_keys is missing: phase-1 contract method")
    seq_len, cp_size = 32, 2
    chunk = seq_len // (2 * cp_size)
    generator = _generator(73)
    k = _randn(generator, seq_len, 2, 8, dtype=torch.bfloat16)
    gate = _randn(generator, *k.shape, dtype=torch.bfloat16)
    with torch.no_grad():
        full, _ = pool_keys(k, gate)
        for rank in range(cp_size):
            parts = [rank, 2 * cp_size - 1 - rank]
            rows = torch.cat([torch.arange(p * chunk, (p + 1) * chunk) for p in parts])
            local, _ = pool_keys(k[rows], gate[rows])
            pools = torch.cat(
                [torch.arange(p * chunk // POOL, (p + 1) * chunk // POOL) for p in parts]
            )
            assert torch.equal(local, full[pools])


def _probe(shape, seed):
    """A BF16-exact probe, so the backward of the pooled key's BF16 cast is exact."""
    return _randn(_generator(seed), *shape).bfloat16().float()


def test_kpool_pool_keys_fp32_profile_gradients_match_finite_differences():
    indexer = _build_indexer(_indexer_config(trainable=True, params_dtype=torch.float32))
    pool_keys = getattr(indexer, "kpool_pool_keys", None)
    if pool_keys is None:
        pytest.fail("DSAIndexer.kpool_pool_keys is missing: phase-1 contract method")
    gate_param = indexer.index_kpool_compress_gate
    ape_param = indexer.index_kpool_compress_ape
    assert gate_param.requires_grad and ape_param.requires_grad
    generator = _generator(74)
    seq_len = 13  # three complete pools and one leftover token
    x = _randn(generator, seq_len, 2, 64)
    k = _randn(generator, seq_len, 2, 8).requires_grad_()
    gate_score = F.linear(x, gate_param)
    k_pooled, base = pool_keys(k, gate_score)
    probe = _probe(k_pooled.shape, 75)
    (k_pooled.float() * probe).sum().backward()
    assert gate_param.grad is not None and ape_param.grad is not None and k.grad is not None

    oracle_base = ref.pool_token_base(seq_len, POOL)
    assert torch.equal(base.long(), oracle_base)
    x64, probe64 = x.double(), probe.double()

    def objective(gate64, ape64, k64):
        pooled = ref.compress_keys(
            k64, F.linear(x64, gate64), ape64, oracle_base, POOL, profile="exact"
        )
        return (pooled * probe64).sum()

    point = [t.detach().double() for t in (gate_param, ape_param, k)]
    leaves = [t.clone().requires_grad_() for t in point]
    autograd = torch.autograd.grad(objective(*leaves), leaves)
    produced = (gate_param.grad, ape_param.grad, k.grad)
    for name, grad, reference in zip(("dgate", "dape", "dk"), produced, autograd):
        _assert_close_scaled(grad, reference, 1e-4, f"{name} vs FP64 autograd")
    # Central differences on every APE entry and on a sample of gate and key entries.
    eps = 1e-6
    samples = {
        0: [(i, j) for i in range(8) for j in range(0, 64, 9)],
        1: [(i, j) for i in range(POOL) for j in range(8)],
        2: [(t, b, d) for t in range(seq_len) for b in range(2) for d in (0, 5)],
    }
    for which, entries in samples.items():
        scale = float(produced[which].abs().max())
        for entry in entries:
            plus = [t.clone() for t in point]
            minus = [t.clone() for t in point]
            plus[which][entry] += eps
            minus[which][entry] -= eps
            with torch.no_grad():
                numeric = float((objective(*plus) - objective(*minus)) / (2 * eps))
            assert abs(float(produced[which][entry]) - numeric) <= 1e-4 * scale + 1e-7, (
                f"input {which} entry {entry}: autograd {float(produced[which][entry]):.6e} "
                f"vs finite difference {numeric:.6e}"
            )


def test_kpool_pool_keys_key_dtype_profile_gradients_match_the_autograd_oracle():
    indexer = _build_indexer(_indexer_config(trainable=True, key_dtype_compression=True))
    pool_keys = getattr(indexer, "kpool_pool_keys", None)
    if pool_keys is None:
        pytest.fail("DSAIndexer.kpool_pool_keys is missing: phase-1 contract method")
    gate_param = indexer.index_kpool_compress_gate
    ape_param = indexer.index_kpool_compress_ape
    generator = _generator(76)
    seq_len = 16
    x = _randn(generator, seq_len, 2, 64, dtype=torch.bfloat16)
    k = _randn(generator, seq_len, 2, 8, dtype=torch.bfloat16).requires_grad_()
    k_pooled, base = pool_keys(k, F.linear(x, gate_param))
    assert k_pooled.dtype == torch.bfloat16
    probe = _probe(k_pooled.shape, 77)
    (k_pooled.float() * probe).sum().backward()

    gate_ref = gate_param.detach().clone().requires_grad_()
    ape_ref = ape_param.detach().clone().requires_grad_()
    k_ref = k.detach().clone().requires_grad_()
    pooled = ref.compress_keys(
        k_ref,
        F.linear(x, gate_ref),
        ape_ref,
        ref.pool_token_base(seq_len, POOL),
        POOL,
        profile="key_dtype",
    )
    torch.testing.assert_close(k_pooled, pooled, rtol=2**-7, atol=2**-7)
    (pooled.float() * probe).sum().backward()
    for name, grad, reference in (
        ("dgate", gate_param.grad, gate_ref.grad),
        ("dape", ape_param.grad, ape_ref.grad),
        ("dk", k.grad, k_ref.grad),
    ):
        _assert_close_scaled(grad, reference, 2e-2, name)


def test_kpool_pool_keys_follow_the_caller_grad_mode():
    indexer = _build_indexer(_indexer_config(trainable=True))
    pool_keys = getattr(indexer, "kpool_pool_keys", None)
    if pool_keys is None:
        pytest.fail("DSAIndexer.kpool_pool_keys is missing: phase-1 contract method")
    generator = _generator(78)
    k = _randn(generator, SEQ, 2, 8, dtype=torch.bfloat16)
    gate = _randn(generator, SEQ, 2, 8, dtype=torch.bfloat16)
    with torch.no_grad():
        frozen, _ = pool_keys(k, gate)
    with torch.enable_grad():
        live, _ = pool_keys(k, gate)
    assert not frozen.requires_grad
    assert live.requires_grad  # through the APE
    assert torch.equal(frozen, live.detach())


# ---------------------------------------------------------------------------------------
# fused_qk_topk_kpool: pooled_keys and return_selection
# ---------------------------------------------------------------------------------------


def _selector_inputs(seed, *, batch, hf_scoring, cu_seqlens=None, heads=4, dim=16):
    generator = _generator(seed)
    projection = _randn(generator, SEQ, batch, heads, dtype=torch.bfloat16)
    starts = torch.tensor(ref.document_starts(SEQ, cu_seqlens), dtype=torch.int64)
    return SimpleNamespace(
        q=_randn(generator, SEQ, batch, heads, dim, dtype=torch.bfloat16),
        k=_randn(generator, SEQ, batch, dim, dtype=torch.bfloat16),
        gate=_randn(generator, SEQ, batch, dim, dtype=torch.bfloat16),
        ape=_randn(generator, POOL, dim, scale=0.5),
        weights=(
            projection.float() * heads**-0.5
            if hf_scoring
            else projection * (heads**-0.5 * dim**-0.5)
        ),
        starts=starts,
        ends=torch.arange(1, SEQ + 1, dtype=torch.int64),
        cu=torch.tensor(cu_seqlens, dtype=torch.int32) if cu_seqlens is not None else None,
    )


def _select(x, *, hf_scoring, key_dtype_compression=True, **kwargs):
    with torch.no_grad():
        return dsa_module.fused_qk_topk_kpool(
            x.q,
            x.k,
            x.weights,
            TOPK,
            POOL,
            x.gate,
            x.ape,
            varlen_starts=x.starts,
            varlen_ends=x.ends,
            cu_seqlens_kv=x.cu,
            use_relu=True,
            always_select_tail=True,
            return_index_scores=False,
            kpool_key_dtype_compression=key_dtype_compression,
            hf_scoring=hf_scoring,
            **kwargs,
        )


@pytest.mark.parametrize("packed", [False, True], ids=["plain", "packed"])
@pytest.mark.parametrize("hf_scoring", [False, True], ids=["legacy", "hf"])
def test_fused_qk_topk_kpool_returns_the_pool_selection(hf_scoring, packed):
    cu = PACKED_CU if packed else None
    x = _selector_inputs(81, batch=1 if packed else 2, hf_scoring=hf_scoring, cu_seqlens=cu)
    default = _select(x, hf_scoring=hf_scoring)
    assert len(default) == 2
    result = _select(x, hf_scoring=hf_scoring, return_selection=True)
    assert len(result) == 3
    _, tokens, pool_ids = result
    assert torch.equal(tokens, default[1])
    assert pool_ids.dtype == torch.int64 and pool_ids.shape == (x.q.size(1), SEQ, TOPK // POOL)
    base = ref.pool_token_base(SEQ, POOL, cu)
    # Selector slot order: pool slot j holds the token columns j·P .. j·P + P - 1.
    expanded = ref.expand_selection(pool_ids, base, POOL, TOPK, cu)
    assert torch.equal(expanded, tokens.long())
    starts = ref.document_starts(SEQ, cu)
    base_list = base.tolist()
    for b in range(pool_ids.size(0)):
        for row in range(SEQ):
            chosen = [p for p in pool_ids[b, row].tolist() if p >= 0]
            visible = ref.visible_pools(row, base_list, POOL, starts[row])
            assert len(set(chosen)) == len(chosen) == min(TOPK // POOL, len(visible))
            assert set(chosen) <= set(visible)


@pytest.mark.parametrize("packed", [False, True], ids=["plain", "packed"])
@pytest.mark.parametrize("hf_scoring", [False, True], ids=["legacy", "hf"])
@pytest.mark.parametrize("key_dtype_compression", [False, True], ids=["fp32", "key-dtype"])
def test_fused_qk_topk_kpool_with_pooled_keys_selects_like_internal_compression(
    key_dtype_compression, hf_scoring, packed
):
    cu = PACKED_CU if packed else None
    x = _selector_inputs(82, batch=1 if packed else 2, hf_scoring=hf_scoring, cu_seqlens=cu)
    if packed:
        k_pooled, base = dsa_module._kpool_compress_keys_per_seg(
            x.k, x.gate, x.ape, POOL, x.cu, kpool_key_dtype_compression=key_dtype_compression
        )
    else:
        k_pooled = dsa_module._kpool_compress_keys(
            x.k, x.gate, x.ape, POOL, kpool_key_dtype_compression=key_dtype_compression
        )
        base = torch.arange(SEQ // POOL, dtype=torch.int64) * POOL
    kwargs = dict(hf_scoring=hf_scoring, key_dtype_compression=key_dtype_compression)
    internal = _select(x, return_selection=True, **kwargs)
    pooled = _select(x, pooled_keys=(k_pooled, base), return_selection=True, **kwargs)
    assert torch.equal(internal[1], pooled[1])
    assert torch.equal(internal[2], pooled[2])


@pytest.mark.parametrize("hf_scoring", [False, True], ids=["legacy", "hf"])
def test_fused_qk_topk_kpool_selection_with_pooled_keys_records_no_autograd_state(hf_scoring):
    """Selection is discrete: grad-tracked pooled keys and queries leave no autograd state."""
    x = _selector_inputs(83, batch=2, hf_scoring=hf_scoring)
    gate = x.gate.float().requires_grad_()
    ape = x.ape.clone().requires_grad_()
    k = x.k.float().requires_grad_()
    with torch.enable_grad():
        k_pooled = dsa_module._kpool_compress_keys(k, gate, ape, POOL).float()
        base = torch.arange(SEQ // POOL, dtype=torch.int64) * POOL
        q = x.q.float().requires_grad_()
        weights = x.weights.float().requires_grad_()
        saved = []

        def pack(tensor):
            saved.append(tuple(tensor.shape))
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            outputs = dsa_module.fused_qk_topk_kpool(
                q,
                k,
                weights,
                TOPK,
                POOL,
                gate,
                ape,
                varlen_starts=x.starts,
                varlen_ends=x.ends,
                use_relu=True,
                always_select_tail=True,
                return_index_scores=False,
                hf_scoring=hf_scoring,
                pooled_keys=(k_pooled, base),
                return_selection=True,
            )
    assert saved == []
    assert all(not t.requires_grad for t in outputs if isinstance(t, torch.Tensor))
    assert q.grad is None and k.grad is None and gate.grad is None and ape.grad is None


# ---------------------------------------------------------------------------------------
# Configuration rules
# ---------------------------------------------------------------------------------------

_TRAINABLE = dict(dsa_indexer_loss_coeff=0.1, dsa_indexer_use_sparse_loss=True)
_FULL_RECOMPUTE = dict(
    recompute_granularity="full", recompute_method="uniform", recompute_num_layers=1
)
_CORE_ATTN_RECOMPUTE = dict(recompute_granularity="selective", recompute_modules=["core_attn"])
_FULL_RECOMPUTE_MESSAGE = r"(?is)recompute.*full|full.*recompute"


def _dsa_config(**overrides):
    values = dict(
        num_layers=2,
        hidden_size=128,
        num_attention_heads=4,
        experimental_attention_variant="dsa",
        add_bias_linear=False,
        dsa_indexer_topk=32,
        dsa_indexer_kpool=POOL,
        dsa_indexer_loss_coeff=0.0,
    )
    values.update(overrides)
    return MLATransformerConfig(**values)


def test_trainable_kpool_config_is_accepted():
    config = _dsa_config(**_TRAINABLE)
    assert config.dsa_indexer_kpool == POOL and config.dsa_indexer_loss_coeff == 0.1
    assert config.dsa_indexer_use_sparse_loss


_REJECTED = [
    pytest.param(
        {"dsa_indexer_loss_coeff": 0.1}, "indexer loss is not supported", id="dense-default"
    ),
    pytest.param(
        {"dsa_indexer_loss_coeff": 0.1, "dsa_indexer_use_sparse_loss": False},
        "indexer loss is not supported",
        id="dense-explicit",
    ),
    pytest.param(
        {**_TRAINABLE, "dsa_indexer_kpool_use_quantization": True},
        "dsa_indexer_kpool_use_quantization",
        id="fp8-kpool-scoring",
    ),
    pytest.param(
        {**_TRAINABLE, "cuda_graph_impl": "full_iteration"}, "full_iteration", id="full-iteration"
    ),
    pytest.param({**_TRAINABLE, **_FULL_RECOMPUTE}, _FULL_RECOMPUTE_MESSAGE, id="full-recompute"),
    pytest.param({**_TRAINABLE, **_CORE_ATTN_RECOMPUTE}, "core_attn", id="core-attn-recompute"),
]


@pytest.mark.parametrize(("overrides", "match"), _REJECTED)
def test_trainable_kpool_config_rejects_unsupported_combinations(overrides, match):
    with pytest.raises(ValueError, match=match):
        _dsa_config(**overrides)


@pytest.mark.parametrize("coeff", [None, 0.0])
@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({}, id="plain"),
        pytest.param({"dsa_indexer_kpool_use_quantization": True}, id="fp8-kpool-scoring"),
        pytest.param({"cuda_graph_impl": "full_iteration"}, id="full-iteration"),
        pytest.param(_FULL_RECOMPUTE, id="full-recompute"),
        pytest.param(_CORE_ATTN_RECOMPUTE, id="core-attn-recompute"),
        pytest.param({"dsa_indexer_use_sparse_loss": True}, id="sparse-flag"),
    ],
)
def test_frozen_kpool_configs_are_unchanged(overrides, coeff):
    config = _dsa_config(dsa_indexer_loss_coeff=coeff, **overrides)
    assert (config.dsa_indexer_loss_coeff or 0.0) == 0.0


def test_trainable_kpool_rejects_checkpoint_topk_reuse(monkeypatch):
    # Config validation only; neither a CUDA device nor the kernel dependencies are needed.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(core_utils, "_missing_cudnn_dsa_kernel_dependencies", lambda: [])
    fields = dict(
        dsa_kernel_backend="cudnn",
        attention_backend=AttnBackend.auto,
        dsa_indexer_kpool=POOL,
        dsa_indexer_topk=8,
        **_TRAINABLE,
    )
    make_small_dsa_config(**fields)  # control: accepted without the reuse
    with pytest.raises(ValueError, match="dsa_indexer_reuse_checkpoint_topk"):
        make_small_dsa_config(dsa_indexer_reuse_checkpoint_topk=True, **fields)


@pytest.mark.parametrize("loss", ["frozen", "trainable"])
def test_simplified_indexer_rejects_kpool(loss):
    fields = dict(
        num_layers=1,
        hidden_size=32,
        num_attention_heads=4,
        num_query_groups=1,
        kv_channels=8,
        experimental_attention_variant="dsa",
        dsa_indexer_mode="simplified",
        add_bias_linear=False,
        dsa_indexer_topk=8,
    )
    if loss == "frozen":
        fields.update(dsa_kernel_backend="reference", dsa_indexer_loss_coeff=0.0)
    else:
        fields.update(dsa_kernel_backend="min-memory-torch", **_TRAINABLE)
    TransformerConfig(**fields)  # control: accepted without KPool
    # The GQA-path rules of this block are assertions; the KPool rule may be either kind.
    with pytest.raises((ValueError, AssertionError), match="simplified"):
        TransformerConfig(dsa_indexer_kpool=POOL, **fields)


def _validator_config(**overrides):
    values = dict(
        dsa_indexer_kpool=POOL,
        dsa_indexer_topk=2048,
        dsa_indexer_loss_coeff=0.1,
        dsa_indexer_use_sparse_loss=True,
        dsa_indexer_kpool_use_quantization=False,
        cuda_graph_impl="none",
        recompute_granularity=None,
        recompute_modules=["core_attn"],
        dsa_indexer_reuse_checkpoint_topk=False,
        dsa_indexer_mode="standard",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_validate_dsa_kpool_config_accepts_the_trainable_sparse_loss():
    dsa_module._validate_dsa_kpool_config(_validator_config())


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        pytest.param(
            {"dsa_indexer_use_sparse_loss": False}, "indexer loss is not supported", id="dense"
        ),
        pytest.param(
            {"dsa_indexer_kpool_use_quantization": True},
            "dsa_indexer_kpool_use_quantization",
            id="fp8-kpool-scoring",
        ),
        pytest.param({"cuda_graph_impl": "full_iteration"}, "full_iteration", id="full-iteration"),
        pytest.param(_FULL_RECOMPUTE, _FULL_RECOMPUTE_MESSAGE, id="full-recompute"),
        pytest.param(_CORE_ATTN_RECOMPUTE, "core_attn", id="core-attn-recompute"),
        pytest.param(
            {"dsa_indexer_reuse_checkpoint_topk": True},
            "dsa_indexer_reuse_checkpoint_topk",
            id="checkpoint-topk-reuse",
        ),
        pytest.param({"dsa_indexer_mode": "simplified"}, "simplified", id="simplified"),
    ],
)
def test_validate_dsa_kpool_config_rejects_unsupported_combinations(overrides, match):
    with pytest.raises(ValueError, match=match):
        dsa_module._validate_dsa_kpool_config(_validator_config(**overrides))


@pytest.mark.parametrize("coeff", [None, 0.0])
def test_validate_dsa_kpool_config_keeps_frozen_configs(coeff):
    """Trainable-only rules do not apply to a frozen selector."""
    dsa_module._validate_dsa_kpool_config(
        _validator_config(
            dsa_indexer_loss_coeff=coeff,
            dsa_indexer_use_sparse_loss=False,
            dsa_indexer_kpool_use_quantization=True,
            cuda_graph_impl="full_iteration",
            **_FULL_RECOMPUTE,
        )
    )
