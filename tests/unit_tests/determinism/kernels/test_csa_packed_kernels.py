# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Replay the packed layout, indexer metadata and CP collective operations."""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant.csa_utils import (
    cp_utils,
    csa_indexer_loss_kernels,
    packed_layout,
    packed_sparse_attention,
)
from tests.unit_tests.determinism.kernels.harness import assert_replays_bit_exact, seeded

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def test_cp_compressor_layout_replays():
    seeded()
    cu = torch.tensor([0, 37, 37, 215, 512], device="cuda", dtype=torch.int32)
    hidden = torch.randn(256, 1, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    boundary = torch.randn(16, 1, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    def run(x, halo):
        return cp_utils.prepare_cp_compressor_input(x, halo, cu, 256, 2, 4)

    assert_replays_bit_exact(run, (hidden, boundary), contention=True, what="CSA packed layout")
    layout = run(hidden, boundary)

    def indices(comp_cu, row_map):
        return packed_layout.build_attention_indices(
            cu,
            256,
            256,
            16,
            16,
            4,
            64,
            cu_seqlens_compressed=comp_cu,
            seq_to_rank_row=row_map,
            compressed_rows=layout[1].shape[0] * 2,
            output_alignment=128,
        )

    assert_replays_bit_exact(indices, (layout[-2], layout[-1]), backward=False, contention=True)


def test_packed_indexer_metadata_and_loss_replay():
    seeded()
    cu_q = torch.tensor([0, 65, 65, 256], device="cuda", dtype=torch.int32)
    cu_k = torch.tensor([0, 32, 32, 96], device="cuda", dtype=torch.int32)
    offsets = torch.tensor([63, 0, 65], device="cuda", dtype=torch.int32)
    scores = torch.randn(256, 64, device="cuda")
    candidates = torch.randint(-1, 70, (256, 32), device="cuda", dtype=torch.int32)

    def select(score, ids):
        lengths = packed_layout.build_seq_lens(cu_q, cu_k, 256, 4, offsets)
        return packed_layout.sanitize_topk(ids, score, lengths, output_width=128)

    assert_replays_bit_exact(select, (scores, candidates), backward=False, contention=True)
    indices, _ = select(scores, candidates)
    predict = torch.softmax(torch.randn_like(indices, dtype=torch.float32), dim=-1)
    target = torch.softmax(torch.randn_like(predict), dim=-1)

    def loss(t, p, ids):
        return csa_indexer_loss_kernels.sparse_kl_loss(t, p, ids, 0.2, True, 256)

    assert_replays_bit_exact(loss, (target, predict, indices), backward=False, contention=True)


def test_async_context_parallel_collectives_replay():
    """Launch/wait preserves autograd and bit-exact NCCL replay under the suite's Ring policy."""
    import torch.distributed as dist

    from megatron.core.tensor_parallel.mappings import (
        async_gather_from_sequence_parallel_region,
        async_reduce_scatter_along_first_dim,
    )
    from tests.unit_tests.test_utilities import Utils

    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    try:
        seeded()
        group = dist.group.WORLD
        value = torch.randn(128, 64, device="cuda", requires_grad=True)

        def run(x):
            gathered = async_gather_from_sequence_parallel_region(x, group=group).wait()
            return gathered.square()

        assert_replays_bit_exact(
            run, (value,), contention=True, what="async all-gather/reduce-scatter"
        )
        global_grad = torch.randn(group.size() * 128, 64, device="cuda")

        def reduce(x):
            return async_reduce_scatter_along_first_dim(x, group=group).wait()

        assert_replays_bit_exact(reduce, (global_grad,), backward=False, contention=True)
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("with_lengths", [False, True])
def test_packed_sparse_attention_padding_is_sink_only(monkeypatch, deterministic, with_lengths):
    """Both packed entry points mask padding without mutating caller-owned metadata."""
    attention = packed_sparse_attention
    query = torch.ones(3, 2, 4, requires_grad=True)
    kv = torch.ones(7, 4, requires_grad=True)
    sink = torch.zeros(2, requires_grad=True)
    indices = torch.tensor([[4, -1, 0, -1], [5, 1, -1, -1], [6, -1, 2, 3]], dtype=torch.int32)
    lengths = torch.tensor([2, 2, 3], dtype=torch.int32) if with_lengths else None
    if with_lengths:
        # Direct autograd callers supply an already compact valid prefix.
        indices = torch.tensor([[4, 0, -1, -1], [5, 1, -1, -1], [6, 2, 3, -1]], dtype=torch.int32)
    original_indices = indices.clone()
    original_lengths = None if lengths is None else lengths.clone()
    padding = torch.tensor([False, True, False])
    seen = {}

    def fake_flash(q, _kv, ids, _scale, *, topk_length, **_kwargs):
        seen["ids"] = ids.clone()
        seen["lengths"] = None if topk_length is None else topk_length.clone()
        return torch.zeros_like(q), torch.ones(q.shape[:2]), None

    def fake_backward(q, k, _out, dout, lse, attn_sink, ids, **kwargs):
        seen["backward"] = (ids.clone(), kwargs["topk_length"], dout.clone(), lse.clone())
        return torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(attn_sink)

    monkeypatch.setattr(attention, "get_flash_mla_topk_alignment", lambda: 4)
    monkeypatch.setattr(attention, "_csa_fwd_flash_mla", fake_flash)
    monkeypatch.setattr(attention, "_csa_sparse_attention_backward", fake_backward)
    monkeypatch.setattr(attention, "_ensure_dsa_namespace", lambda: None)
    if with_lengths:
        output = attention.CSASparseAttnFunc.apply(
            query, kv, sink, indices, lengths, 0.5, 0, None, padding, deterministic
        )[0]
    else:
        output = attention.csa_sparse_attn(
            query, kv, sink, indices, 0.5, q_padding_mask=padding, deterministic=deterministic
        )
    assert torch.equal(
        seen["ids"][1], torch.full_like(indices[1], -1)
    ), "Padded queries must be sink-only before FlashMLA"
    if deterministic or with_lengths:
        expected = torch.tensor(
            [[4, 0, -1, -1], [-1, -1, -1, -1], [6, 2, 3, -1]], dtype=torch.int32
        )
        assert torch.equal(seen["ids"], expected)
        assert torch.equal(seen["lengths"], torch.tensor([2, 0, 3], dtype=torch.int32))
    else:
        assert torch.equal(seen["ids"][[0, 2]], original_indices[[0, 2]])
        assert seen["lengths"] is None
    output.sum().backward()
    backward_ids, backward_lengths, dout, lse = seen["backward"]
    assert torch.count_nonzero(dout[padding]) == 0
    assert torch.count_nonzero(lse[padding]) == 0
    if deterministic or with_lengths:
        assert torch.equal(backward_lengths, torch.tensor([2, 1, 3], dtype=torch.int32))
        assert (backward_ids >= 0).all()
    assert torch.equal(indices, original_indices), "Caller-owned Top-K IDs were mutated"
    if with_lengths:
        assert torch.equal(lengths, original_lengths), "Caller-owned Top-K lengths were mutated"


@pytest.mark.parametrize("deterministic", [False, True])
def test_deterministic_packed_attention_compacts_without_lengths(monkeypatch, deterministic):
    """Compressed-first no-grad input uses the same prefix contract as the training wrapper."""
    attention = packed_sparse_attention
    indices = torch.tensor([[-1, 3, -1, 0]], dtype=torch.int32)
    original = indices.clone()
    seen = {}

    def fake_flash(q, _kv, ids, _scale, *, topk_length, **_kwargs):
        seen.update(ids=ids.clone(), lengths=topk_length)
        return torch.zeros_like(q), torch.zeros(q.shape[:2]), None

    monkeypatch.setattr(attention, "get_flash_mla_topk_alignment", lambda: 4)
    monkeypatch.setattr(attention, "_csa_fwd_flash_mla", fake_flash)
    attention.csa_sparse_attn(
        torch.zeros(1, 2, 4),
        torch.zeros(4, 4),
        torch.zeros(2),
        indices,
        0.5,
        deterministic=deterministic,
    )
    if deterministic:
        assert torch.equal(
            seen["ids"], torch.tensor([[3, 0, -1, -1]], dtype=torch.int32)
        ), "Deterministic packed attention must compact without caller lengths"
        assert torch.equal(seen["lengths"], torch.tensor([2], dtype=torch.int32))
    else:
        assert torch.equal(seen["ids"], original)
        assert seen["lengths"] is None
    assert torch.equal(indices, original)


@pytest.mark.parametrize("deterministic", [False, True])
def test_packed_indexer_attention_padding_is_sink_only(monkeypatch, deterministic):
    """Positive-loss forward enforces padding before FlashMLA and preserves caller tensors."""
    attention = packed_sparse_attention
    query = torch.ones(3, 2, 4)
    kv = torch.ones(7, 4)
    indices = torch.tensor([[4, -1, 0, -1], [5, 1, -1, -1], [6, -1, 2, 3]], dtype=torch.int32)
    original_indices = indices.clone()
    indexer_ids = torch.arange(3, dtype=torch.int32).view(3, 1)
    original_indexer_ids = indexer_ids.clone()
    q_indexer, k_indexer, weights = torch.zeros(3, 2, 2), torch.zeros(3, 2), torch.ones(3, 2)
    padding = torch.tensor([False, True, False])
    seen = {}
    saved = []
    ctx = SimpleNamespace(
        needs_input_grad=(False,) * 28, save_for_backward=lambda *xs: saved.extend(xs)
    )

    def fake_flash(q, _kv, ids, _scale, *, topk_length, **_kwargs):
        seen.update(ids=ids.clone(), lengths=topk_length.clone())
        return torch.zeros_like(q), torch.zeros(q.shape[:2]), None

    fake_dsa = SimpleNamespace(
        sparse_indexer_score_recompute_wrapper=lambda _q, _k, _w, ids, **_kw: {
            "predict": torch.ones_like(ids, dtype=torch.float32)
        },
        indexer_backward_wrapper=lambda q, w, k, *_args, **_kw: {
            "d_index_q": torch.zeros_like(q),
            "d_index_k": torch.zeros_like(k),
            "d_weights": torch.zeros_like(w),
        },
    )
    monkeypatch.setattr(attention, "_ensure_dsa_namespace", lambda: None)
    monkeypatch.setattr(attention, "_DSA", fake_dsa)
    monkeypatch.setattr(attention, "_csa_fwd_flash_mla", fake_flash)
    monkeypatch.setattr(attention, "_compute_attn_target", lambda *_args, **_kw: torch.zeros(3, 1))
    monkeypatch.setattr(
        attention.csa_indexer_loss_kernels, "sparse_kl_loss", lambda *_a, **_kw: torch.tensor(0.0)
    )
    attention.FusedCSAIndexerSparseAttnFromTopkFunc.forward(
        ctx,
        query,
        kv,
        torch.zeros(2),
        indices,
        q_indexer,
        k_indexer,
        weights,
        indexer_ids,
        kv[4:],
        0.5,
        1.0,
        0.3,
        2,
        True,
        4,
        3,
        (None, None, None),
        padding,
        k_indexer,
        kv[4:],
        SimpleNamespace(size=lambda: 1),
        4,
        None,
        None,
        None,
        3,
        None,
        deterministic,
    )
    expected = torch.tensor([[4, 0, -1, -1], [-1, -1, -1, -1], [6, 2, 3, -1]], dtype=torch.int32)
    assert torch.equal(
        seen["ids"], expected
    ), "Padded indexer queries must be sink-only before FlashMLA"
    assert torch.equal(seen["lengths"], torch.tensor([2, 0, 3], dtype=torch.int32))
    assert torch.equal(saved[4], torch.tensor([2, 1, 3], dtype=torch.int32))
    assert (saved[3] >= 0).all()
    assert torch.equal(indices, original_indices), "Caller-owned Top-K IDs were mutated"
    assert torch.equal(indexer_ids, original_indexer_ids), "Caller-owned indexer IDs were mutated"
