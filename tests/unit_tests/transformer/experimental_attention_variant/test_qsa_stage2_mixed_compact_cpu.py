# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU contracts for the opt-in compact mixed-packed QSA Stage-2 path."""

from types import MethodType, SimpleNamespace

import torch

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.experimental_attention_variant import qsa as qsa_module
from megatron.core.transformer.experimental_attention_variant.qsa import (
    QSACoreAttention,
    QSAIndexer,
    QSASelection,
)
from megatron.core.transformer.experimental_attention_variant.qsa_id_mixed_router import (
    pool_complete_blocks,
)
from megatron.core.transformer.experimental_attention_variant.qsa_stage2_loss import (
    qsa_stage2_sparse_kl,
)


def _layout(lengths):
    doc = torch.tensor(
        [index for index, length in enumerate(lengths) for _ in range(length)], dtype=torch.int32
    )
    pos = torch.tensor([index for length in lengths for index in range(length)], dtype=torch.int32)
    return doc, pos


def test_qsa_stage2_compact_pool_preserves_key_gradient_and_dummy_slots():
    lengths, ratio = [8, 20, 1, 3], 4
    doc, pos = _layout(lengths)
    raw = torch.arange(32 * 3, dtype=torch.float32).reshape(32, 3).requires_grad_()
    compact, prefix, counts, block_doc, block_relative, valid = pool_complete_blocks(
        raw, doc, pos, num_docs=len(lengths), ratio=ratio
    )
    rectangular, _, _ = QSAIndexer._pool_keys(
        SimpleNamespace(compress_ratio=ratio), raw, doc, pos, max(lengths)
    )
    assert compact.shape == (8, 3)
    assert prefix.tolist() == [0, 2, 7, 7, 7]
    assert counts.tolist() == [2, 5, 0, 0]
    assert valid.tolist() == [True] * 7 + [False]
    weights = torch.arange(1, 8, dtype=torch.float32).unsqueeze(1)
    compact_loss = (compact[:7] * weights).sum()
    rectangular_loss = sum(
        (rectangular[document, :count] * weights[first : first + count]).sum()
        for document, (first, count) in enumerate(((0, 2), (2, 5)))
    )
    torch.testing.assert_close(compact[:2], rectangular[0, :2])
    torch.testing.assert_close(compact[2:7], rectangular[1, :5])
    torch.testing.assert_close(
        torch.autograd.grad(compact_loss, raw, retain_graph=True)[0],
        torch.autograd.grad(rectangular_loss, raw)[0],
    )
    assert block_doc[-1] == 3 and block_relative[-1] == 0


def test_qsa_stage2_mixed_compact_kl_uses_physical_doc_starts_and_prefix(monkeypatch):
    lengths, ratio, total = [8, 20, 1, 3], 4, 32
    doc, pos = _layout(lengths)
    physical_cu = torch.tensor([0, 8, 28, 29, 32], dtype=torch.int32)
    packed = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=physical_cu,
        cu_seqlens_kv=physical_cu,
        cu_seqlens_q_padded=physical_cu,
        cu_seqlens_kv_padded=physical_cu,
        max_seqlen_q=20,
        max_seqlen_kv=20,
    )
    packed.qsa_stage2_layout_cpu = ((0, 8, 28, 29, 32), (5, 18, 0, 2))
    raw = torch.randn(total, 3, requires_grad=True)
    pooled, prefix, _, block_doc, block_relative, valid = pool_complete_blocks(
        raw, doc, pos, num_docs=len(lengths), ratio=ratio
    )
    block_starts = torch.where(
        valid,
        physical_cu[:-1].long()[block_doc] + block_relative.long() * ratio,
        torch.zeros_like(block_relative, dtype=torch.long),
    )
    assert block_starts.tolist() == [0, 4, 8, 12, 16, 20, 24, 0]
    selected_ids = torch.full((total, 2), -1, dtype=torch.int32)
    selected_ids[15, 0] = 1  # second document, relative block 1 -> compact row 3
    selected_ids[5, 0] = 0  # physical padding after first document's valid length
    selection = QSASelection(
        doc_ids=doc.unsqueeze(0),
        positions=pos.unsqueeze(0),
        selected_bits=None,
        bits_per_row=0,
        bits_per_row_t=torch.tensor(0),
        compress_ratio=ratio,
        all_selected=False,
        selected_ids=selected_ids,
        index_query=torch.randn(total, 2, 3, requires_grad=True),
        compressed_key=pooled,
        compact_block_prefix=prefix,
        compact_block_starts=block_starts,
    )
    captured = {}

    def capture(
        index_query,
        compressed_key,
        block_ids,
        starts,
        teacher_query,
        teacher_key,
        document_starts,
        positions,
        **kwargs,
    ):
        captured.update(
            block_ids=block_ids,
            starts=starts,
            document_starts=document_starts,
            positions=positions,
            kwargs=kwargs,
        )
        return index_query.sum() * 0

    monkeypatch.setattr(qsa_module, "qsa_stage2_sparse_kl", capture)
    core = SimpleNamespace(
        config=SimpleNamespace(
            attention_dropout=0.0,
            tensor_model_parallel_size=1,
            qsa_indexer_loss_coeff=0.7,
            calculate_per_token_loss=False,
        ),
        pg_collection=SimpleNamespace(tp=None),
        softmax_scale=3**-0.5,
    )
    output = torch.randn(total, 6)
    result = QSACoreAttention._attach_indexer_loss(
        core,
        output,
        torch.randn(total, 2, 3),
        torch.randn(total, 1, 3),
        selection,
        packed,
        cp_size=1,
    )
    torch.testing.assert_close(result, output)
    assert captured["block_ids"][15, 0] == 3
    assert captured["block_ids"][5, 0] == -1
    assert captured["starts"][3] == 12
    assert captured["starts"][-1] == 0
    assert captured["document_starts"][15] == 8
    assert captured["positions"][15] == 7
    assert captured["kwargs"]["query_valid_rows"][5] == 0
    assert captured["kwargs"]["query_valid_rows"][28] == 0


def test_qsa_stage2_compact_empty_complete_pool_is_differentiable():
    lengths = [1, 1, 1, 1]
    doc, pos = _layout(lengths)
    raw = torch.randn(4, 2, requires_grad=True)
    pooled, prefix, counts, _, _, valid = pool_complete_blocks(raw, doc, pos, num_docs=4, ratio=4)
    assert pooled.shape == (1, 2)
    assert prefix.tolist() == [0, 0, 0, 0, 0]
    assert not valid.any()
    assert not counts.any()
    pooled.sum().backward()
    assert torch.equal(raw.grad, torch.zeros_like(raw))


def test_qsa_stage2_compact_real_kl_updates_queries_and_pooled_keys():
    torch.manual_seed(137)
    lengths, ratio, topk = [8, 16], 4, 2
    doc, pos = _layout(lengths)
    raw = torch.randn(24, 8, requires_grad=True)
    compressed, prefix, _, block_doc, block_relative, valid = pool_complete_blocks(
        raw, doc, pos, num_docs=2, ratio=ratio
    )
    starts = torch.tensor([0, 8], dtype=torch.long)
    block_starts = starts[block_doc] + block_relative.long() * ratio
    assert valid.all()
    relative_ids = torch.full((24, topk), -1, dtype=torch.long)
    relative_ids[:, 0] = torch.where(pos >= 3, 0, -1)
    relative_ids[:, 1] = torch.where(pos >= 7, 1, -1)
    block_ids = torch.where(relative_ids >= 0, relative_ids + prefix[doc.long(), None], -1)
    index_query = torch.randn(24, 2, 8, requires_grad=True)
    loss = qsa_stage2_sparse_kl(
        index_query,
        compressed,
        block_ids,
        block_starts,
        torch.randn(24, 2, 8),
        torch.randn(24, 1, 8),
        starts[doc.long()],
        pos.long(),
        compress_ratio=ratio,
        loss_coeff=0.7,
        query_chunk_size=4,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert index_query.grad is not None and torch.isfinite(index_query.grad).all()
    assert raw.grad is not None and torch.isfinite(raw.grad).all()
    assert index_query.grad.abs().sum() > 0
    assert raw.grad.abs().sum() > 0


def test_qsa_stage2_mixed_indexer_retains_key_norm_and_rope_gradient(monkeypatch):
    from megatron.core.transformer.experimental_attention_variant import qsa_id_mixed_router

    torch.manual_seed(181)
    lengths, ratio, topk = [8, 16], 4, 2
    doc, pos = _layout(lengths)
    cu = torch.tensor([0, 8, 24], dtype=torch.int32)
    packed = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
        cu_seqlens_q_padded=cu,
        cu_seqlens_kv_padded=cu,
        max_seqlen_q=16,
        max_seqlen_kv=16,
    )

    def cpu_routes(query, pooled, doc_ids, positions, prefix, counts, *, ratio, topk):
        result = torch.full((query.shape[0], topk), -1, dtype=torch.int32)
        for row in range(query.shape[0]):
            visible = min((int(positions[row]) + 1) // ratio, int(counts[doc_ids[row]]))
            result[row, : min(visible, topk)] = torch.arange(min(visible, topk))
        return result

    monkeypatch.setattr(qsa_id_mixed_router, "select_document_local_ids", cpu_routes)
    indexer = SimpleNamespace(
        compress_ratio=ratio,
        block_topk=topk,
        k_layernorm=torch.nn.RMSNorm(8),
        rotary_interleaved=False,
    )
    indexer._rope_at_positions = MethodType(QSAIndexer._rope_at_positions, indexer)
    q = torch.randn(24, 2, 8, requires_grad=True)
    raw = torch.randn(24, 8, requires_grad=True)
    freqs = torch.randn(24, 1, 1, 8) * 0.1
    ids, all_selected, compressed, prefix, block_starts = QSAIndexer._select_mixed_packed_ids(
        indexer, q, raw, doc, pos, freqs, packed, 16, True
    )
    assert not all_selected
    assert compressed.requires_grad
    assert prefix.tolist() == [0, 2, 6]
    assert block_starts.tolist() == [0, 4, 8, 12, 16, 20]
    block_ids = torch.where(ids >= 0, ids + prefix[doc.long(), None], -1)
    loss = qsa_stage2_sparse_kl(
        q,
        compressed,
        block_ids,
        block_starts,
        torch.randn(24, 2, 8),
        torch.randn(24, 1, 8),
        cu[:-1].long()[doc.long()],
        pos.long(),
        compress_ratio=ratio,
        loss_coeff=0.7,
        query_chunk_size=4,
    )
    loss.backward()
    assert raw.grad is not None and raw.grad.abs().sum() > 0
    assert indexer.k_layernorm.weight.grad is not None
    assert indexer.k_layernorm.weight.grad.abs().sum() > 0
    with torch.no_grad():
        _, _, no_loss_pool, _, _ = QSAIndexer._select_mixed_packed_ids(
            indexer, q, raw, doc, pos, freqs, packed, 16, True
        )
    assert not no_loss_pool.requires_grad


def test_qsa_stage2_mixed_rejects_more_than_4096_tokens_before_projection():
    indexer = SimpleNamespace(
        config=SimpleNamespace(
            sequence_parallel=False, mrope_section=None, qsa_indexer_loss_coeff=0.7
        ),
        pg_collection=SimpleNamespace(tp=None, cp=None),
        training=True,
    )
    try:
        QSAIndexer.forward(
            indexer, torch.zeros(4097, 1, 1), torch.zeros(1, 1, 1, 2), output_format="ids"
        )
    except NotImplementedError as exc:
        assert "4096" in str(exc)
    else:
        raise AssertionError("positive Stage-2 exceeded its token gate")
