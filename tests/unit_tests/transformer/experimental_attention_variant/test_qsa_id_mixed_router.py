# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Document-local QSA route and compact-pool contracts."""

import math
from types import SimpleNamespace

import pytest
import torch

from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_qsa_module_spec_for_backend,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.experimental_attention_variant.qsa import (
    QSAIndexer,
    QSASelection,
    build_qsa_dense_mask,
)
from megatron.core.transformer.experimental_attention_variant.qsa_id_mixed_router import (
    pool_complete_blocks,
    select_document_local_ids,
)
from megatron.core.transformer.spec_utils import build_module
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.experimental_attention_variant.test_attention_variant_qsa import (
    _make_config,
)


def _layout(lengths):
    doc = torch.tensor(
        [index for index, length in enumerate(lengths) for _ in range(length)], dtype=torch.int32
    )
    pos = torch.tensor([index for length in lengths for index in range(length)], dtype=torch.int32)
    return doc, pos


@pytest.mark.parametrize("ratio,topk", [(2, 2), (4, 2), (8, 3)])
def test_qsa_mixed_compact_pool_and_token_mask_match_rectangular_reference(ratio, topk):
    lengths = [1, 0, 17, ratio - 1, ratio, 27, 0, 12]
    doc, pos = _layout(lengths)
    torch.manual_seed(101 + ratio)
    # Positive nonzero scores avoid torch.topk's unspecified choice among ties.
    raw = (torch.randn(doc.numel(), 16, dtype=torch.bfloat16).abs() + 0.1).requires_grad_()
    query = torch.randn(doc.numel(), 2, 16, dtype=torch.bfloat16).abs() + 0.1
    compact, prefix, counts, block_doc, block_relative, valid = pool_complete_blocks(
        raw, doc, pos, num_docs=len(lengths), ratio=ratio
    )
    assert compact.requires_grad
    reference = SimpleNamespace(compress_ratio=ratio, block_topk=topk)
    rectangular, _, block_valid = QSAIndexer._pool_keys(
        reference, raw.detach(), doc, pos, max(lengths)
    )
    assert counts.tolist() == [length // ratio for length in lengths]
    for document in range(len(lengths)):
        first, last = prefix[document : document + 2].tolist()
        torch.testing.assert_close(compact[first:last], rectangular[document, : last - first])
        assert block_doc[first:last].tolist() == [document] * (last - first)
        assert block_relative[first:last].tolist() == list(range(last - first))
    assert valid.sum().item() == counts.sum().item()

    bits, _ = QSAIndexer._select_blocks(
        reference, query, rectangular, block_valid, doc, pos, output_format="bits"
    )
    expected = build_qsa_dense_mask(
        QSASelection(
            doc_ids=doc.unsqueeze(0),
            positions=pos.unsqueeze(0),
            selected_bits=bits.flatten(),
            bits_per_row=bits.shape[1],
            bits_per_row_t=torch.tensor(bits.shape[1], dtype=torch.int64),
            compress_ratio=ratio,
            all_selected=False,
        ),
        doc.numel(),
    )[0]
    actual = torch.zeros_like(expected)
    for row in range(doc.numel()):
        document = int(doc[row])
        position = int(pos[row])
        visible = (position + 1) // ratio
        if visible <= topk:
            chosen = list(range(visible))
        else:
            first = int(prefix[document])
            keys = compact[first : first + visible]
            scores = (query[row].float() @ keys.float().T).relu().sum(0) / math.sqrt(16)
            chosen = scores.topk(topk).indices.tolist()
        tail_start = visible * ratio
        for key in range(doc.numel()):
            if doc[key] == document and pos[key] <= position:
                actual[row, key] = pos[key] >= tail_start or (int(pos[key]) // ratio) in chosen
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_qsa_mixed_compact_pool_bounds_adversarial_shape_without_rectangular_allocation():
    lengths = [4096] + [1] * 4096
    doc, pos = _layout(lengths)
    raw = torch.ones(doc.numel(), 128, dtype=torch.bfloat16)
    compact, prefix, counts, _, _, valid = pool_complete_blocks(
        raw, doc, pos, num_docs=len(lengths), ratio=4
    )
    assert compact.shape == (2048, 128)
    assert int(prefix[-1]) == 1024
    assert int(valid.sum()) == 1024
    assert int(counts[1:].sum()) == 0
    rectangular_fp32_bytes = len(lengths) * (max(lengths) // 4) * 128 * 4
    assert rectangular_fp32_bytes > 2 * 1024**3
    assert compact.numel() * compact.element_size() == 512 * 1024


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_qsa_mixed_document_local_gpu_routes_match_reference():
    torch.manual_seed(113)
    ratio, topk, dim, heads = 4, 16, 128, 4
    lengths = [1, 96, 3, 128, 4]
    doc_cpu, pos_cpu = _layout(lengths)
    doc, pos = doc_cpu.cuda(), pos_cpu.cuda()
    query = torch.randn(doc.numel(), heads, dim, dtype=torch.bfloat16, device="cuda")
    raw = torch.randn(doc.numel(), dim, dtype=torch.bfloat16, device="cuda")
    pooled, prefix, counts, _, _, _ = pool_complete_blocks(
        raw, doc, pos, num_docs=len(lengths), ratio=ratio
    )
    got = select_document_local_ids(
        query, pooled, doc, pos, prefix, counts, ratio=ratio, topk=topk
    ).cpu()
    for row in range(doc.numel()):
        document = int(doc_cpu[row])
        visible = (int(pos_cpu[row]) + 1) // ratio
        if visible <= topk:
            expected = list(range(visible))
        else:
            first = int(prefix[document])
            keys = pooled[first : first + visible]
            scores = (query[row].float() @ keys.float().T).relu().sum(0) / math.sqrt(dim)
            expected = scores.topk(topk).indices.cpu().tolist()
        assert sorted(got[row].tolist()) == sorted(expected + [-1] * (topk - len(expected)))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_qsa_mixed_document_local_gpu_topk512_multichunk_and_empty_docs():
    ratio, topk, dim, heads = 4, 512, 128, 4
    lengths = [0, 4096, 0, 1, 5, 0]
    doc_cpu, pos_cpu = _layout(lengths)
    doc, pos = doc_cpu.cuda(), pos_cpu.cuda()
    # Two exactly representable BF16 components give a strict order over 1024 blocks.
    block = pos // ratio
    raw = torch.zeros(doc.numel(), dim, dtype=torch.bfloat16, device="cuda")
    raw[:, 0] = (block // 64).to(torch.bfloat16)
    raw[:, 1] = ((block % 64).float() / 64).to(torch.bfloat16)
    query = torch.zeros(doc.numel(), heads, dim, dtype=torch.bfloat16, device="cuda")
    query[:, :, 0] = 64
    query[:, :, 1] = 1
    pooled, prefix, counts, block_doc, relative, valid = pool_complete_blocks(
        raw, doc, pos, num_docs=len(lengths), ratio=ratio
    )
    assert counts.cpu().tolist() == [0, 1024, 0, 0, 1, 0]
    assert block_doc[valid].cpu().tolist() == [1] * 1024 + [4]
    assert relative[valid].cpu().tolist() == list(range(1024)) + [0]
    got = select_document_local_ids(
        query, pooled, doc, pos, prefix, counts, ratio=ratio, topk=topk
    ).cpu()
    for row in (0, 2047, 2048, 2051, 3000, 4095, 4100, 4101):
        visible = (int(pos_cpu[row]) + 1) // ratio
        if visible <= topk:
            expected = list(range(visible))
        else:
            expected = list(range(visible - topk, visible))
        assert sorted(got[row].tolist()) == sorted(expected + [-1] * (topk - len(expected)))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_qsa_mixed_document_local_gpu_zero_score_ties_use_smaller_block_ids():
    ratio, topk, dim, heads = 4, 16, 128, 4
    lengths = [128, 1]
    doc_cpu, pos_cpu = _layout(lengths)
    doc, pos = doc_cpu.cuda(), pos_cpu.cuda()
    raw = torch.ones(doc.numel(), dim, dtype=torch.bfloat16, device="cuda")
    query = torch.zeros(doc.numel(), heads, dim, dtype=torch.bfloat16, device="cuda")
    pooled, prefix, counts, _, _, _ = pool_complete_blocks(
        raw, doc, pos, num_docs=len(lengths), ratio=ratio
    )
    got = select_document_local_ids(
        query, pooled, doc, pos, prefix, counts, ratio=ratio, topk=topk
    ).cpu()
    replay = select_document_local_ids(
        query, pooled, doc, pos, prefix, counts, ratio=ratio, topk=topk
    ).cpu()
    assert torch.equal(got, replay)
    assert sorted(got[127].tolist()) == list(range(topk))
    assert got[-1].tolist() == [-1] * topk


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_qsa_mixed_model_indexer_selected_sets_match_rectangular_after_norm_and_rope(monkeypatch):
    Utils.initialize_model_parallel(1, 1)
    torch.cuda.set_per_process_memory_fraction(0.25)
    try:
        torch.manual_seed(137)
        config = _make_config(
            hidden_size=2560,
            num_attention_heads=24,
            num_query_groups=2,
            kv_channels=256,
            qsa_indexer_n_heads=4,
            qsa_indexer_kv_heads=1,
            qsa_indexer_head_dim=128,
            qsa_indexer_budget=2048,
            qsa_indexer_compress_ratio=4,
            mrope_section=[11, 11, 10],
        )
        attention = (
            build_module(get_qsa_module_spec_for_backend(config), config=config, layer_number=1)
            .cuda()
            .eval()
        )
        cu = torch.tensor([0, 0, 4096, 4096, 4104], dtype=torch.int32, device="cuda")
        packed = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu,
            cu_seqlens_kv=cu,
            cu_seqlens_q_padded=cu,
            cu_seqlens_kv_padded=cu,
            max_seqlen_q=4096,
            max_seqlen_kv=4096,
        )
        hidden = torch.randn(4104, 1, config.hidden_size, dtype=torch.bfloat16, device="cuda")
        freqs = torch.randn(4104, 1, 1, 64, device="cuda")
        indexer = attention.indexer
        captured = {}
        original = indexer._select_mixed_packed_ids

        def record_inputs(*args):
            captured["inputs"] = args
            return original(*args)

        monkeypatch.setattr(indexer, "_select_mixed_packed_ids", record_inputs)
        got = indexer(hidden, freqs, packed, output_format="ids").selected_ids
        q, raw, doc, pos, rotary, _, max_doc_len, is_absolute_mrope = captured["inputs"]
        assert is_absolute_mrope
        rectangular, block_positions, valid = indexer._pool_keys(raw, doc, pos, max_doc_len)
        n_docs, n_blocks, dim = rectangular.shape
        normalized = indexer.k_layernorm(rectangular.reshape(-1, dim))
        rope_positions = (block_positions + cu[:-1, None]).clamp_max(cu[1:, None] - 1)
        normalized = indexer._rope_at_positions(
            normalized, rotary, rope_positions.reshape(-1)
        ).view(n_docs, n_blocks, dim)
        expected, _ = indexer._select_blocks(q, normalized, valid, doc, pos, output_format="ids")
        assert torch.equal(got.sort(dim=-1).values, expected.sort(dim=-1).values)
        cpu_cu = cu.cpu()
        cpu_packed = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cpu_cu,
            cu_seqlens_kv=cpu_cu,
            cu_seqlens_q_padded=cpu_cu,
            cu_seqlens_kv_padded=cpu_cu,
            max_seqlen_q=4096,
            max_seqlen_kv=4096,
        )
        cpu_cu_ids, _ = indexer._select_mixed_packed_ids(
            q, raw, doc, pos, rotary, cpu_packed, max_doc_len, True
        )
        assert torch.equal(got, cpu_cu_ids)
    finally:
        Utils.destroy_model_parallel()
