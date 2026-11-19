# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU tests of ``dsa_indexer_kpool_scorer_backend='cudnn'`` on packed THD input.

Packed KPool pools restart at every document (``_kpool_compress_keys_per_seg``). The cuDNN
selector scores each document span of the local query rows against that document's pooled
keys only, with positions counted from the document's first token, and maps the ids back to
the whole pooled-key axis. Fake cuDNN frontends (``dsa_cudnn_fe_fakes``) model the kernel's
causal contract; the tests check the host runs against the packed layout and the fused
selection against the PyTorch KPool scorer, up to ties at the selection boundary.
"""

import logging
from types import SimpleNamespace

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant import (
    dsa,
    dsa_cudnn_kernels,
    dsa_kernels,
    dsa_layout,
)
from tests.unit_tests.transformer.experimental_attention_variant import dsa_cudnn_fe_fakes as fakes
from tests.unit_tests.transformer.experimental_attention_variant import kpool_selection_utils as ks

POOL = ks.POOL

# (document lengths, cp_size): lengths divisible by 2 * cp_size under CP; documents shorter
# than a pool, lengths off the pool grid (CP1, CP3) and a pool-aligned pack.
PACKS = [
    ([30, 7, 64, 3], 1),
    ([64], 2),
    ([64, 36, 8, 20], 2),
    ([66, 18, 6, 102], 3),
    ([64, 40, 8, 24, 120], 4),
]


@pytest.fixture
def frontend(monkeypatch):
    """Install a fake frontend namespace and version for one test."""

    def install(namespace, version):
        if hasattr(namespace, "reset"):
            namespace.reset()
        monkeypatch.setattr(dsa_cudnn_kernels, "_ensure_dsa_namespace", lambda: None)
        monkeypatch.setattr(dsa_cudnn_kernels, "_cudnn_dsa", namespace)
        monkeypatch.setattr(dsa_cudnn_kernels, "_cudnn_frontend_version", lambda: version)
        monkeypatch.setattr(dsa_cudnn_kernels, "_KPOOL_DECLINES_LOGGED", set())
        return namespace

    return install


def _packed_inputs(lengths, cp_size, cp_rank, *, batch=1, pad_to=None):
    return ks.packed_kpool_inputs(
        lengths,
        cp_size=cp_size,
        cp_rank=cp_rank,
        batch=batch,
        heads=4,
        dim=16,
        device="cpu",
        pad_to=pad_to,
    )


def _runs(x, *, local_rows=None, row_start=0):
    return ks.packed_query_runs(x, local_rows=local_rows, row_start=row_start)


@pytest.mark.parametrize("lengths, cp_size", PACKS)
def test_packed_runs_match_the_packed_layout_and_the_pool_axis(lengths, cp_size):
    for cp_rank in sorted({0, cp_size // 2, cp_size - 1}):
        x = _packed_inputs(lengths, cp_size, cp_rank)
        runs = _runs(x)
        _, base = dsa._kpool_compress_keys_per_seg(
            x.k, x.gate, x.ape, POOL, x.cu, kpool_key_dtype_compression=True
        )
        assert runs[0][0] == 0 and runs[-1][1] == x.rows
        assert all(a[1] == b[0] for a, b in zip(runs, runs[1:]))
        for row_start, row_end, position, key_start, key_count in runs:
            doc_start = int(x.starts[row_start])
            assert x.starts[row_start:row_end].eq(doc_start).all()
            expected = torch.arange(position, position + row_end - row_start) + doc_start + 1
            assert torch.equal(x.ends[row_start:row_end], expected)
            # The run's key range is exactly its document's complete pools.
            doc_end = x.host_cu[x.host_cu.index(doc_start) + 1]
            assert key_count == (doc_end - doc_start) // POOL
            assert base[key_start : key_start + key_count].tolist() == [
                doc_start + j * POOL for j in range(key_count)
            ]


@pytest.mark.parametrize("tp_size", [2, 3])
def test_packed_runs_slice_sequence_parallel_rows(tp_size):
    lengths, cp_size, cp_rank = [64, 40, 8, 24, 120], 4, 1
    x = _packed_inputs(lengths, cp_size, cp_rank)
    full = _runs(x)
    local_rows = x.rows // tp_size
    for tp_rank in range(tp_size):
        row_start = tp_rank * local_rows
        runs = _runs(x, local_rows=local_rows, row_start=row_start)
        assert runs[0][0] == 0 and runs[-1][1] == local_rows

        def expand(rs, offset):
            return [
                (row + offset, position + row - start, key_start, key_count)
                for start, end, position, key_start, key_count in rs
                for row in range(start, end)
            ]

        assert expand(runs, row_start) == expand(full, 0)[row_start : row_start + local_rows]


@pytest.mark.parametrize(
    "change, words",
    [
        (dict(host_cu_seqlens_q=None), "host"),
        (dict(host_cu_seqlens_kv=[0, 16, 64]), "differ"),
        (dict(full_rows=60, local_rows=60), "rows"),
        (dict(host_cu_seqlens_q=[0, 30, 64], host_cu_seqlens_kv=[0, 30, 64]), "2 * cp_size"),
        (dict(row_start=8), "outside"),
    ],
)
def test_packed_runs_decline(change, words):
    facts = dict(
        host_cu_seqlens_q=[0, 32, 64],
        host_cu_seqlens_kv=[0, 32, 64],
        local_rows=32,
        full_rows=32,
        row_start=0,
        cp_size=2,
        cp_rank=0,
        pool_size=POOL,
    )
    facts.update(change)
    runs, reason = dsa_layout.packed_kpool_query_runs(**facts)
    assert runs is None and words in reason


@pytest.mark.parametrize(
    "namespace, version", [(fakes.Fe131DSA, (1, 31)), (fakes.BottomRightDefaultDSA, (1, 21))]
)
@pytest.mark.parametrize("lengths, cp_size", PACKS)
@pytest.mark.parametrize("hf_scoring", [True, False])
def test_packed_fused_selection_matches_torch_scorer(
    frontend, namespace, version, lengths, cp_size, hf_scoring
):
    frontend(namespace, version)
    topk = 16
    for cp_rank in sorted({0, cp_size - 1}):
        x = _packed_inputs(lengths, cp_size, cp_rank, batch=1, pad_to=len(lengths) + 2)
        reference = ks.select_packed(x, topk=topk, hf_scoring=hf_scoring)
        calls = []
        selector = ks.cudnn_pool_topk_fn(_runs(x), calls=calls)
        candidate = ks.select_packed(x, topk=topk, hf_scoring=hf_scoring, pool_topk_fn=selector)
        assert calls == [True]
        ks.assert_same_packed_selection_up_to_boundary_ties(
            x, reference, candidate, topk=topk, rel_tol=1e-4
        )


def test_packed_keyed_runs_return_global_ids_and_aligned_scores(frontend):
    frontend(fakes.Fe131DSA, (1, 31))
    lengths, cp_size, cp_rank = [64, 40, 8, 24, 120], 4, 2
    x = _packed_inputs(lengths, cp_size, cp_rank)
    k_pooled, base = dsa._kpool_compress_keys_per_seg(
        x.k, x.gate, x.ape, POOL, x.cu, kpool_key_dtype_compression=True
    )
    runs = _runs(x)
    kwargs = dict(
        q=x.q,
        k_pooled=k_pooled,
        weights=x.weights_hf,
        pool_topk=4,
        pool_size=POOL,
        query_runs=runs,
        hf_scoring=True,
    )
    ids = dsa_cudnn_kernels.run_kpool_pool_topk(**kwargs)
    ids_again, scores = dsa_cudnn_kernels.run_kpool_pool_topk(**kwargs, return_scores=True)
    assert torch.equal(ids, ids_again)
    assert torch.equal(scores.isfinite(), ids >= 0)
    for row_start, row_end, position, key_start, key_count in runs:
        chosen = ids[:, row_start:row_end]
        valid = chosen >= 0
        # Every id lies in the row's own document, among its causal pools.
        assert ((chosen[valid] >= key_start) & (chosen[valid] < key_start + key_count)).all()
        rows = torch.arange(row_end - row_start).view(1, -1, 1)
        visible = (position + rows + 1) // POOL
        assert (chosen - key_start < visible)[valid].all()
        assert (base[chosen[valid]] >= 0).all()


def _attention_stub(**config):
    defaults = dict(
        dsa_indexer_kpool_scorer_backend="cudnn",
        attention_backend="auto",
        dsa_kernel_backend="cudnn",
        dsa_indexer_scoring_relu=True,
        dsa_indexer_kpool=POOL,
    )
    defaults.update(config)
    return SimpleNamespace(config=SimpleNamespace(**defaults))


def _capturing(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)


def test_packed_selector_stays_on_the_torch_scorer_inside_a_graph_capture(monkeypatch, caplog):
    """A captured selector would replay the capture's pack runs for every later pack."""
    monkeypatch.setattr(dsa, "_KPOOL_SCORER_FALLBACK_REASONS_WARNED", set())
    facts = dict(
        has_mask=False,
        positions_from_position_ids=False,
        local_rows=64,
        full_rows=64,
        row_start=0,
        skv=128,
        cp_size=2,
        cp_rank=0,
        host_cu_seqlens_q=[0, 64, 128],
        host_cu_seqlens_kv=[0, 64, 128],
    )
    _capturing(monkeypatch)
    with caplog.at_level(logging.WARNING):
        assert (
            dsa.DSAttention._kpool_pool_topk_fn(_attention_stub(), packed_thd=True, **facts) is None
        )
    assert "CUDA graph capture" in caplog.text
    # Non-packed runs are the same for every microbatch, so they may be captured.
    facts.update(host_cu_seqlens_q=None, host_cu_seqlens_kv=None, cp_size=1)
    facts.update(local_rows=128, full_rows=128)
    assert dsa.DSAttention._kpool_pool_topk_fn(_attention_stub(), packed_thd=False, **facts)


def test_attention_builds_a_packed_selector_from_the_host_boundaries(monkeypatch, caplog):
    monkeypatch.setattr(dsa, "_KPOOL_SCORER_FALLBACK_REASONS_WARNED", set())
    seen = {}

    def fake_hook(
        config, q, k_pooled, weights, pool_topk, pool_size, query_runs, hf_scoring, use_relu
    ):
        seen.update(query_runs=query_runs)
        return None

    monkeypatch.setattr(dsa_kernels, "run_kpool_pool_topk", fake_hook)
    facts = dict(
        packed_thd=True,
        has_mask=False,
        # The packed forward ignores position_ids, so they do not keep the PyTorch scorer.
        positions_from_position_ids=True,
        local_rows=64,
        full_rows=64,
        row_start=0,
        skv=128,
        cp_size=2,
        cp_rank=1,
    )
    selector = dsa.DSAttention._kpool_pool_topk_fn(
        _attention_stub(), **facts, host_cu_seqlens_q=[0, 64, 128], host_cu_seqlens_kv=[0, 64, 128]
    )
    selector(None, None, None, 8, POOL, True)
    # Rank 1 of CP2 holds tokens 16-31 and 32-47 of each 64-token document (16 pools each).
    assert seen["query_runs"] == (
        (0, 16, 16, 0, 16),
        (16, 32, 32, 0, 16),
        (32, 48, 16, 16, 16),
        (48, 64, 32, 16, 16),
    )
    # Without host boundaries the PyTorch scorer stays, warned once.
    with caplog.at_level(logging.WARNING):
        assert dsa.DSAttention._kpool_pool_topk_fn(_attention_stub(), **facts) is None
    assert "packed THD boundaries are not on the host" in caplog.text


# Packed THD without CP selects from the pack's device boundaries instead of host runs.
CP1_PACKS = [[30, 7, 64, 3], [64], [5, 2, 41], [3]]


def _pooled(x):
    return dsa._kpool_compress_keys_per_seg(
        x.k, x.gate, x.ape, POOL, x.cu, kpool_key_dtype_compression=True
    )


@pytest.mark.parametrize(
    "namespace, version", [(fakes.Fe131DSA, (1, 31)), (fakes.BottomRightDefaultDSA, (1, 21))]
)
@pytest.mark.parametrize("lengths", CP1_PACKS)
@pytest.mark.parametrize("hf_scoring", [True, False])
def test_packed_device_selection_matches_torch_scorer_and_host_runs(
    frontend, namespace, version, lengths, hf_scoring
):
    frontend(namespace, version)
    topk = 16
    x = _packed_inputs(lengths, 1, 0, pad_to=len(lengths) + 2)
    reference = ks.select_packed(x, topk=topk, hf_scoring=hf_scoring)
    calls = []
    selector = ks.cudnn_packed_pool_topk_fn(x.cu, x.rows, calls=calls)
    candidate = ks.select_packed(x, topk=topk, hf_scoring=hf_scoring, pool_topk_fn=selector)
    assert calls == [True]
    ks.assert_same_packed_selection_up_to_boundary_ties(
        x, reference, candidate, topk=topk, rel_tol=1e-4
    )
    # The host runs' calls score the same rows against the same pools.
    host = ks.select_packed(
        x, topk=topk, hf_scoring=hf_scoring, pool_topk_fn=ks.cudnn_pool_topk_fn(_runs(x))
    )
    ks.assert_same_packed_selection_up_to_boundary_ties(x, host, candidate, topk=topk, rel_tol=1e-4)


def test_packed_device_selection_returns_global_ids_and_aligned_scores(frontend):
    frontend(fakes.Fe131DSA, (1, 31))
    x = _packed_inputs([30, 7, 64, 3], 1, 0, pad_to=6)
    k_pooled, base = _pooled(x)
    kwargs = dict(
        k_pooled=k_pooled, weights=x.weights_hf, pool_topk=4, pool_size=POOL, hf_scoring=True
    )
    ids, scores = dsa_cudnn_kernels.run_kpool_pool_topk(
        q=x.q,
        query_runs=None,
        return_scores=True,
        packed_cu_seqlens=x.cu,
        packed_max_seqlen=x.rows,
        **kwargs,
    )
    host_ids, host_scores = dsa_cudnn_kernels.run_kpool_pool_topk(
        q=x.q, query_runs=_runs(x), return_scores=True, **kwargs
    )

    def same_scores_per_row(a, b):
        # Equal-score pools may swap between calls of different widths (ReLU scores tie at
        # zero), and the slot order is the top-k kernel's; the fake frontend's FP32 sums depend
        # on how many keys a call scores.
        torch.testing.assert_close(a.sort(dim=-1).values, b.sort(dim=-1).values, rtol=1e-6, atol=0)

    same_scores_per_row(scores, host_scores)
    assert torch.equal(scores.isfinite(), ids >= 0)
    assert (base[ids[ids >= 0]] >= 0).all()
    # Every id lies in the row's own document, among its causal pools.
    doc = torch.searchsorted(x.cu[1:].long(), torch.arange(x.rows), right=True)
    first_pool = torch.cat((torch.zeros(1), ((x.cu[1:] - x.cu[:-1]) // POOL).cumsum(0))).long()
    visible = (torch.arange(x.rows) - x.cu.long()[doc] + 1) // POOL
    local = ids[0] - first_pool[doc].unsqueeze(-1)
    assert ((local >= 0) & (local < visible.unsqueeze(-1)))[ids[0] >= 0].all()
    # Rows past the last boundary (static capacity padding) select nothing.
    extra = 5
    q = torch.cat((x.q, torch.ones_like(x.q[:extra])))
    weights = torch.cat((x.weights_hf, torch.ones_like(x.weights_hf[:extra])))
    padded, padded_scores = dsa_cudnn_kernels.run_kpool_pool_topk(
        q=q,
        query_runs=None,
        return_scores=True,
        packed_cu_seqlens=x.cu,
        packed_max_seqlen=x.rows + extra,
        **dict(kwargs, weights=weights),
    )
    same_scores_per_row(padded_scores[:, : x.rows], scores)
    assert (padded[:, x.rows :] == -1).all()


def test_packed_device_selection_reads_nothing_back_to_the_host(frontend, monkeypatch):
    """Every pack-dependent value stays on the device, so one CUDA graph serves every pack."""
    frontend(fakes.Fe131DSA, (1, 31))
    x = _packed_inputs([30, 7, 64, 3], 1, 0, pad_to=6)
    k_pooled, _ = _pooled(x)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("host synchronization in the packed device KPool selection")

    monkeypatch.setattr(torch.Tensor, "item", forbidden)
    monkeypatch.setattr(torch.Tensor, "tolist", forbidden)
    ids = dsa_cudnn_kernels.run_kpool_pool_topk(
        q=x.q,
        k_pooled=k_pooled,
        weights=x.weights_hf,
        pool_topk=4,
        pool_size=POOL,
        query_runs=None,
        hf_scoring=True,
        return_scores=True,
        packed_cu_seqlens=x.cu,
        packed_max_seqlen=x.rows,
    )
    assert ids is not None


@pytest.mark.parametrize(
    "change, words",
    [
        (dict(batch=2), "batch 1"),
        (dict(query_runs=((0, 104, 0),)), "not both"),
        (dict(packed_max_seqlen=None), "host bound"),
        (dict(no_offsets=True), "no q_causal_offsets"),
    ],
)
def test_packed_device_selection_declines(frontend, monkeypatch, caplog, change, words):
    frontend(fakes.Fe131DSA, (1, 31))
    x = _packed_inputs([30, 7, 64, 3], 1, 0, batch=change.get("batch", 1), pad_to=6)
    k_pooled, _ = _pooled(x)
    if change.get("no_offsets"):
        monkeypatch.setattr(dsa_cudnn_kernels, "_supports_indexer_q_causal_offsets", lambda: False)
    with caplog.at_level(logging.WARNING):
        result = dsa_cudnn_kernels.run_kpool_pool_topk(
            q=x.q,
            k_pooled=k_pooled,
            weights=x.weights_hf,
            pool_topk=4,
            pool_size=POOL,
            query_runs=change.get("query_runs"),
            hf_scoring=True,
            packed_cu_seqlens=x.cu,
            packed_max_seqlen=change.get("packed_max_seqlen", x.rows),
        )
    assert result is None
    assert words in caplog.text


def test_attention_builds_a_device_packed_selector_without_cp(monkeypatch, caplog):
    """At CP1 the selector takes the pack's device boundaries, inside a graph capture too."""
    monkeypatch.setattr(dsa, "_KPOOL_SCORER_FALLBACK_REASONS_WARNED", set())
    seen = []

    def fake_hook(config, q, k_pooled, weights, pool_topk, pool_size, query_runs, hf_scoring, **kw):
        seen.append(dict(query_runs=query_runs, **kw))
        return None

    monkeypatch.setattr(dsa_kernels, "run_kpool_pool_topk", fake_hook)
    cu = torch.tensor([0, 40, 100, 100], dtype=torch.int32)
    facts = dict(
        packed_thd=True,
        has_mask=False,
        positions_from_position_ids=False,
        local_rows=128,
        full_rows=128,
        row_start=0,
        skv=128,
        cp_size=1,
        cp_rank=0,
        cu_seqlens=cu,
    )
    _capturing(monkeypatch)
    selector = dsa.DSAttention._kpool_pool_topk_fn(_attention_stub(), **facts)
    selector(None, None, None, 8, POOL, True)
    assert seen[0]["query_runs"] is None
    assert seen[0]["packed_cu_seqlens"] is cu and seen[0]["packed_max_seqlen"] == 128
    # A TP sequence-parallel slice of the pack keeps the PyTorch scorer.
    facts.update(local_rows=64, row_start=64)
    with caplog.at_level(logging.WARNING):
        assert dsa.DSAttention._kpool_pool_topk_fn(_attention_stub(), **facts) is None
    assert "sequence-parallel slice" in caplog.text


def test_kernel_dispatch_declines_device_boundaries_for_a_hook_without_them(monkeypatch):
    def old_hook(q, k_pooled, weights, pool_topk, pool_size, query_runs, hf_scoring, use_relu):
        raise AssertionError("called without the packed device boundaries")

    monkeypatch.setattr(dsa_kernels, "_resolve_fused_hook", lambda config, name: old_hook)
    monkeypatch.setattr(dsa_kernels, "_log_declined_hook", lambda *args: None)
    config = SimpleNamespace(dsa_kernel_backend="cudnn")
    monkeypatch.setattr(dsa_kernels, "_deterministic_dsa_requested", lambda config: False)
    result = dsa_kernels.run_kpool_pool_topk(
        config,
        None,
        None,
        None,
        8,
        POOL,
        None,
        True,
        packed_cu_seqlens=torch.tensor([0, 8]),
        packed_max_seqlen=8,
    )
    assert result is None


@pytest.mark.parametrize("cp_size", [2, 4])
def test_packed_pool_reorder_puts_gathered_rank_pools_in_document_order(cp_size):
    """Pooling each rank's spans, gathering and reordering in pool units gives the
    per-document pools of the gathered tokens, bit for bit (the trainable KPool path)."""
    divisor = 2 * cp_size * POOL
    lengths = [3 * divisor, divisor, 5 * divisor, 2 * divisor]
    ranks = [_packed_inputs(lengths, cp_size, rank) for rank in range(cp_size)]
    x = ranks[0]  # same seed: every rank sees the same global keys
    local_pools = []
    for rank_inputs in ranks:
        rows = rank_inputs.ends - 1  # the rank's global token positions, in local row order
        local_pools.append(
            dsa._kpool_compress_keys(
                x.k[rows], x.gate[rows], x.ape, POOL, kpool_key_dtype_compression=True
            )
        )
    gathered = torch.cat(local_pools)
    for cp_rank in range(cp_size):
        reorder, inverse = dsa_layout.build_packed_kpool_pool_reorder_from_host(
            x.host_cu, cp_size=cp_size, cp_rank=cp_rank, pool_size=POOL, device=torch.device("cpu")
        )
        assert torch.equal(reorder[inverse], torch.arange(reorder.numel()))
        expected, base = dsa._kpool_compress_keys_per_seg(
            x.k, x.gate, x.ape, POOL, x.cu, kpool_key_dtype_compression=True
        )
        assert torch.equal(gathered[reorder], expected)
        assert torch.equal(base, torch.arange(base.numel()) * POOL)


@pytest.mark.parametrize(
    "host_cu, local_rows, words",
    [
        (None, 32, "not on the host"),
        ([0, 40, 64], 32, "multiple of 2 * cp_size * dsa_indexer_kpool = 16"),
        ([0, 32, 64], 40, "not at the 80 rows of the CP group"),
    ],
)
def test_packed_pool_alignment_errors(host_cu, local_rows, words):
    error = dsa_layout.packed_kpool_pool_alignment_error(
        host_cu, cp_size=2, pool_size=POOL, local_rows=local_rows
    )
    assert error is not None and words in error
    assert (
        dsa_layout.packed_kpool_pool_alignment_error(
            [0, 32, 64], cp_size=2, pool_size=POOL, local_rows=32
        )
        is None
    )


def test_trainable_packed_cp_pool_gather_refuses_a_graph_capture(monkeypatch):
    rows = 4 * POOL

    def pool_keys(k, gate_score, cu_seqlens=None):
        return k[::POOL], torch.arange(rows // POOL) * POOL

    attention = SimpleNamespace(
        indexer=SimpleNamespace(kpool_pool_keys=pool_keys, index_kpool=POOL)
    )
    _capturing(monkeypatch)
    with pytest.raises(RuntimeError, match="CUDA graph capture"):
        dsa.DSAttention._kpool_pool_then_gather(
            attention,
            torch.zeros(rows, 1, 8),
            None,
            cu_seqlens_kv=torch.tensor([0, 2 * rows]),
            cp_size=2,
            cp_group=None,
            local_rows=rows,
            cp_rank=0,
            host_cu_seqlens_kv=[0, 2 * rows],
        )
