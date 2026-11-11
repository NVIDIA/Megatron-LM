# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU tests of ``dsa_indexer_kpool_scorer_backend='cudnn'`` (KPool selection on the cuDNN
indexer kernel at ``ratio=dsa_indexer_kpool``).

Fake cuDNN frontends (``dsa_cudnn_fe_fakes``) model the kernel's causal contract: row ``i``
of a call sees keys ``[0, (offset + i + 1) // ratio)``, with the offset explicit or the
frontend's default (bottom-right before 1.26, zero from 1.26). The tests check that the
host-side query runs reproduce the layout's positions and that the fused selection equals
the PyTorch KPool scorer's up to ties at the selection boundary.
"""

import inspect
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


class _NoOffsetsDSA(fakes.BottomRightDefaultDSA):
    """A released frontend <= 1.25: bottom-right default and no ``q_causal_offsets``."""

    @classmethod
    def indexer_forward_wrapper(
        cls, q, k, w, ratio=4, sm_scale=1.0, cu_seqlens_q=None, cu_seqlens_k=None
    ):
        # Bind the parent's implementation to this class so the calls land in cls.calls.
        return fakes.BottomRightDefaultDSA.indexer_forward_wrapper.__func__(
            cls, q, k, w, ratio=ratio, sm_scale=sm_scale
        )


def _expand(runs):
    return [p for start, end, position in runs for p in range(position, position + end - start)]


@pytest.mark.parametrize(
    "seq, cp_size", [(64, 1), (63, 1), (512, 2), (520, 4), (4096, 8), (1048576, 64)]
)
def test_query_runs_match_layout_positions(seq, cp_size):
    rows = seq // cp_size
    for cp_rank in sorted({0, cp_size // 2, cp_size - 1}):
        runs, reason = dsa_layout.nonpacked_query_position_runs(
            local_rows=rows, full_rows=rows, row_start=0, skv=seq, cp_size=cp_size, cp_rank=cp_rank
        )
        assert reason == ""
        positions, _ = dsa_layout.get_cp_positions_from_layout(
            sq=rows,
            skv=seq,
            cp_size=cp_size,
            cp_rank=cp_rank,
            cp_comm_type="allgather",
            device=torch.device("cpu"),
        )
        assert _expand(runs) == positions.tolist()
        assert runs[0][0] == 0 and runs[-1][1] == rows
        assert all(a[1] == b[0] for a, b in zip(runs, runs[1:]))


@pytest.mark.parametrize("tp_size", [4, 3])  # with 3, the middle window straddles the zigzag join
def test_query_runs_slice_sequence_parallel_rows(tp_size):
    seq, cp_size, cp_rank = 96 * 4, 4, 1
    full_rows = seq // cp_size
    positions, _ = dsa_layout.get_cp_positions_from_layout(
        sq=full_rows,
        skv=seq,
        cp_size=cp_size,
        cp_rank=cp_rank,
        cp_comm_type="allgather",
        device=torch.device("cpu"),
    )
    local_rows = full_rows // tp_size
    for tp_rank in range(tp_size):
        row_start = tp_rank * local_rows
        runs, _ = dsa_layout.nonpacked_query_position_runs(
            local_rows=local_rows,
            full_rows=full_rows,
            row_start=row_start,
            skv=seq,
            cp_size=cp_size,
            cp_rank=cp_rank,
        )
        assert _expand(runs) == positions[row_start : row_start + local_rows].tolist()
        assert runs[0][0] == 0 and runs[-1][1] == local_rows
        assert all(a[1] == b[0] for a, b in zip(runs, runs[1:]))


@pytest.mark.parametrize(
    "kwargs, words",
    [
        (
            dict(local_rows=100, full_rows=100, row_start=0, skv=512, cp_size=4, cp_rank=0),
            "uniform",
        ),
        (dict(local_rows=63, full_rows=63, row_start=0, skv=252, cp_size=4, cp_rank=0), "2 * cp"),
        (dict(local_rows=64, full_rows=64, row_start=8, skv=64, cp_size=1, cp_rank=0), "outside"),
    ],
)
def test_query_runs_decline_layouts_without_host_positions(kwargs, words):
    runs, reason = dsa_layout.nonpacked_query_position_runs(**kwargs)
    assert runs is None and words in reason


_runs = ks.query_runs


@pytest.mark.parametrize(
    "namespace, version", [(fakes.Fe131DSA, (1, 31)), (fakes.BottomRightDefaultDSA, (1, 21))]
)
@pytest.mark.parametrize(
    "seq, cp_size, cp_rank", [(512, 1, 0), (512, 4, 0), (520, 4, 1), (528, 4, 3)]
)
@pytest.mark.parametrize("hf_scoring", [True, False])
def test_fused_selection_matches_torch_scorer(
    frontend, namespace, version, seq, cp_size, cp_rank, hf_scoring
):
    frontend(namespace, version)
    topk = 32
    x = ks.kpool_inputs(
        seq=seq, cp_size=cp_size, cp_rank=cp_rank, batch=2, heads=4, dim=16, device="cpu"
    )
    reference = ks.select(x, topk=topk, hf_scoring=hf_scoring)
    called = []
    selector = ks.cudnn_pool_topk_fn(_runs(seq, cp_size, cp_rank))

    def counting_selector(*args):
        result = selector(*args)
        called.append(result is not None)
        return result

    candidate = ks.select(x, topk=topk, hf_scoring=hf_scoring, pool_topk_fn=counting_selector)
    assert called == [True]
    ks.assert_same_selection_up_to_boundary_ties(x, reference, candidate, topk=topk, rel_tol=1e-4)


def test_off_grid_chunks_pass_explicit_offsets(frontend):
    """A run starting off the pool grid needs ``q_causal_offsets`` = its first position."""
    dsa_ns = frontend(fakes.Fe131DSA, (1, 31))
    x = ks.kpool_inputs(seq=520, cp_size=4, cp_rank=1, batch=1, heads=4, dim=16, device="cpu")
    ks.select(x, topk=32, hf_scoring=True, pool_topk_fn=ks.cudnn_pool_topk_fn(_runs(520, 4, 1)))
    offsets = [c["q_causal_offsets"] for c in dsa_ns.calls]
    # Runs start at token 65 (front) and 390 (back) of the 520-token sequence.
    assert [int(o[0]) for o in offsets] == [65, 390]


def test_unaligned_chunks_decline_on_a_frontend_without_offsets(frontend, caplog):
    frontend(_NoOffsetsDSA, (1, 21))
    assert (
        "q_causal_offsets"
        not in inspect.signature(_NoOffsetsDSA.indexer_forward_wrapper).parameters
    )
    x = ks.kpool_inputs(seq=520, cp_size=4, cp_rank=1, batch=1, heads=4, dim=16, device="cpu")
    reference = ks.select(x, topk=32, hf_scoring=True)
    with caplog.at_level(logging.WARNING):
        candidate = ks.select(
            x, topk=32, hf_scoring=True, pool_topk_fn=ks.cudnn_pool_topk_fn(_runs(520, 4, 1))
        )
    assert torch.equal(reference, candidate)  # the PyTorch scorer ran
    assert "q_causal_offsets" in caplog.text


def test_aligned_chunks_run_on_a_frontend_without_offsets(frontend):
    dsa_ns = frontend(_NoOffsetsDSA, (1, 21))
    x = ks.kpool_inputs(seq=512, cp_size=4, cp_rank=2, batch=1, heads=4, dim=16, device="cpu")
    reference = ks.select(x, topk=32, hf_scoring=True)
    calls = []
    candidate = ks.select(
        x,
        topk=32,
        hf_scoring=True,
        pool_topk_fn=ks.cudnn_pool_topk_fn(_runs(512, 4, 2), calls=calls),
    )
    assert calls == [True]
    assert dsa_ns.calls and all(c["q_causal_offsets"] is None for c in dsa_ns.calls)
    ks.assert_same_selection_up_to_boundary_ties(x, reference, candidate, topk=32, rel_tol=1e-4)


@pytest.mark.parametrize(
    "change, words",
    [
        (dict(use_relu=False), "ReLU"),
        (dict(query_runs=None), "host"),
        (dict(query_runs=((0, 10, 0),)), "tile"),
        (dict(q_dtype=torch.float32), "BF16"),
    ],
)
def test_run_kpool_pool_topk_declines(frontend, caplog, change, words):
    frontend(fakes.Fe131DSA, (1, 31))
    x = ks.kpool_inputs(seq=256, cp_size=1, cp_rank=0, batch=1, heads=4, dim=16, device="cpu")
    k_pooled = dsa._kpool_compress_keys(x.k, x.gate, x.ape, 4, kpool_key_dtype_compression=True)
    q = x.q.to(change.get("q_dtype", torch.bfloat16))
    with caplog.at_level(logging.WARNING):
        result = dsa_cudnn_kernels.run_kpool_pool_topk(
            q=q,
            k_pooled=k_pooled,
            weights=x.weights_hf,
            pool_topk=8,
            pool_size=4,
            query_runs=change.get("query_runs", ((0, 256, 0),)),
            hf_scoring=True,
            use_relu=change.get("use_relu", True),
        )
    assert result is None
    assert words in caplog.text


def test_short_rows_keep_every_visible_pool(frontend):
    frontend(fakes.Fe131DSA, (1, 31))
    x = ks.kpool_inputs(seq=64, cp_size=1, cp_rank=0, batch=1, heads=4, dim=16, device="cpu")
    k_pooled = dsa._kpool_compress_keys(x.k, x.gate, x.ape, 4, kpool_key_dtype_compression=True)
    pools = dsa_cudnn_kernels.run_kpool_pool_topk(
        q=x.q,
        k_pooled=k_pooled,
        weights=x.weights_hf,
        pool_topk=8,
        pool_size=4,
        query_runs=((0, 64, 0),),
        hf_scoring=True,
    )
    for row in range(64):
        visible = (row + 1) // 4
        selected = sorted(p for p in pools[0, row].tolist() if p >= 0)
        assert selected == list(range(visible)) if visible <= 8 else len(selected) == 8
        assert all(p < visible for p in selected)


@pytest.mark.parametrize("heads, dim", [(2, 16), (8, 16), (4, 24), (16, 8)])
def test_hf_weights_select_like_torch_for_any_head_count(frontend, heads, dim):
    """HF weights reach the kernel as the exact BF16 projection, whatever (heads * dim)."""
    frontend(fakes.Fe131DSA, (1, 31))
    x = ks.kpool_inputs(seq=512, cp_size=2, cp_rank=1, batch=2, heads=heads, dim=dim, device="cpu")
    reference = ks.select(x, topk=32, hf_scoring=True)
    calls = []
    candidate = ks.select(
        x,
        topk=32,
        hf_scoring=True,
        pool_topk_fn=ks.cudnn_pool_topk_fn(_runs(512, 2, 1), calls=calls),
    )
    assert calls == [True]
    ks.assert_same_selection_up_to_boundary_ties(x, reference, candidate, topk=32, rel_tol=1e-5)


def test_fp32_head_weights_keep_the_torch_scorer(frontend, caplog):
    """FP32 weights (no HF scoring) are not BF16-exact, so the selector declines."""
    frontend(fakes.Fe131DSA, (1, 31))
    x = ks.kpool_inputs(
        seq=512,
        cp_size=1,
        cp_rank=0,
        batch=1,
        heads=4,
        dim=16,
        device="cpu",
        projection_dtype=torch.float32,
    )
    reference = ks.select(x, topk=32, hf_scoring=False)
    calls = []
    with caplog.at_level(logging.WARNING):
        candidate = ks.select(
            x,
            topk=32,
            hf_scoring=False,
            pool_topk_fn=ks.cudnn_pool_topk_fn(_runs(512, 1, 0), calls=calls),
        )
    assert calls == [False]
    assert torch.equal(reference, candidate)
    assert "BF16 head weights" in caplog.text


@pytest.mark.parametrize(
    "namespace, version", [(fakes.Fe131DSA, (1, 31)), (_NoOffsetsDSA, (1, 21))]
)
@pytest.mark.parametrize("seq, cp_size, cp_rank", [(512, 1, 0), (1024, 4, 0), (1024, 4, 2)])
def test_runs_split_into_row_chunks(
    frontend, monkeypatch, namespace, version, seq, cp_size, cp_rank
):
    """The 1M shape splits each run into row chunks; force that at small sizes."""
    dsa_ns = frontend(namespace, version)
    monkeypatch.setattr(dsa_cudnn_kernels, "_indexer_score_chunk_rows", lambda b, sq, sk: 16)
    x = ks.kpool_inputs(
        seq=seq, cp_size=cp_size, cp_rank=cp_rank, batch=2, heads=4, dim=16, device="cpu"
    )
    reference = ks.select(x, topk=32, hf_scoring=True)
    calls = []
    candidate = ks.select(
        x,
        topk=32,
        hf_scoring=True,
        pool_topk_fn=ks.cudnn_pool_topk_fn(_runs(seq, cp_size, cp_rank), calls=calls),
    )
    assert calls == [True]
    rows = seq // cp_size
    # Chunks whose last row sees no complete pool are skipped; the rest each call the kernel.
    assert len(dsa_ns.calls) >= rows // 16 - 1
    ks.assert_same_selection_up_to_boundary_ties(x, reference, candidate, topk=32, rel_tol=1e-4)


@pytest.mark.parametrize("seq", [17, 18, 61, 64])
@pytest.mark.parametrize(
    "namespace, version", [(fakes.Fe131DSA, (1, 31)), (fakes.BottomRightDefaultDSA, (1, 21))]
)
def test_chunks_with_fewer_pools_than_rows_pad_invisible_pools(frontend, seq, namespace, version):
    """The kernel needs rows <= keys * ratio; a chunk from position 0 can have fewer pools."""
    dsa_ns = frontend(namespace, version)
    x = ks.kpool_inputs(seq=seq, cp_size=1, cp_rank=0, batch=2, heads=4, dim=16, device="cpu")
    reference = ks.select(x, topk=8, hf_scoring=True)
    calls = []
    candidate = ks.select(
        x,
        topk=8,
        hf_scoring=True,
        pool_topk_fn=ks.cudnn_pool_topk_fn(_runs(seq, 1, 0), calls=calls),
    )
    assert calls == [True]
    assert all(c["q"].shape[1] <= c["k"].shape[1] * 4 for c in dsa_ns.calls if "q" in c)
    ks.assert_same_selection_up_to_boundary_ties(x, reference, candidate, topk=8, rel_tol=1e-4)


def test_kernel_calls_stay_within_the_safe_row_count(frontend, monkeypatch):
    dsa_ns = frontend(fakes.Fe131DSA, (1, 31))
    monkeypatch.setattr(dsa_cudnn_kernels, "FUSED_INDEXER_MAX_SAFE_ROWS", 64)
    x = ks.kpool_inputs(seq=512, cp_size=1, cp_rank=0, batch=4, heads=4, dim=16, device="cpu")
    reference = ks.select(x, topk=32, hf_scoring=True)
    candidate = ks.select(
        x, topk=32, hf_scoring=True, pool_topk_fn=ks.cudnn_pool_topk_fn(_runs(512, 1, 0))
    )
    assert dsa_ns.calls
    assert all(c["q"].shape[0] * c["q"].shape[1] <= 64 for c in dsa_ns.calls)
    ks.assert_same_selection_up_to_boundary_ties(x, reference, candidate, topk=32, rel_tol=1e-4)


def _attention_stub(**config):
    defaults = dict(
        dsa_indexer_kpool_scorer_backend="cudnn",
        attention_backend="auto",
        dsa_kernel_backend="cudnn",
        dsa_indexer_scoring_relu=True,
    )
    defaults.update(config)
    return SimpleNamespace(config=SimpleNamespace(**defaults))


def _factory(stub, **layout):
    facts = dict(
        packed_thd=False,
        has_mask=False,
        positions_from_position_ids=False,
        local_rows=128,
        full_rows=128,
        row_start=0,
        skv=512,
        cp_size=4,
        cp_rank=1,
    )
    facts.update(layout)
    return dsa.DSAttention._kpool_pool_topk_fn(stub, **facts)


def test_attention_builds_the_selector_only_for_host_known_layouts(monkeypatch, caplog):
    monkeypatch.setattr(dsa, "_KPOOL_SCORER_FALLBACK_REASONS_WARNED", set())
    assert _factory(_attention_stub(dsa_indexer_kpool_scorer_backend="torch")) is None
    assert callable(_factory(_attention_stub()))
    with caplog.at_level(logging.WARNING):
        assert _factory(_attention_stub(attention_backend="unfused")) is None
        assert _factory(_attention_stub(), packed_thd=True) is None
        assert _factory(_attention_stub(), has_mask=True) is None
        assert _factory(_attention_stub(), positions_from_position_ids=True) is None
        assert _factory(_attention_stub(), skv=500) is None
    for words in ("unfused", "packed THD", "attention mask", "position_ids", "uniform"):
        assert words in caplog.text


def test_selector_passes_host_runs_to_the_backend_hook(monkeypatch):
    seen = {}

    def fake_hook(
        config, q, k_pooled, weights, pool_topk, pool_size, query_runs, hf_scoring, use_relu
    ):
        seen.update(query_runs=query_runs, pool_topk=pool_topk, pool_size=pool_size, hf=hf_scoring)
        return None

    monkeypatch.setattr(dsa_kernels, "run_kpool_pool_topk", fake_hook)
    selector = _factory(_attention_stub())
    selector(None, None, None, 512, 4, True)
    assert seen == dict(
        query_runs=((0, 64, 64), (64, 128, 384)), pool_topk=512, pool_size=4, hf=True
    )
