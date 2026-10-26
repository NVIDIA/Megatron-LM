# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Dynamic-batching inference for DSA over GQA with the simplified indexer."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from megatron.core.inference.config import InferenceConfig
from megatron.core.inference.contexts import DynamicInferenceContext
from megatron.core.transformer.experimental_attention_variant import (
    dsa_gqa_flashinfer,
    dsa_gqa_fused_select,
)
from megatron.core.transformer.experimental_attention_variant import dsa_gqa_inference as dsa_gqa
from megatron.core.transformer.experimental_attention_variant.dsa_gqa_filter_kernels import (
    PREFILL_ROW_BLOCK,
)
from megatron.core.transformer.experimental_attention_variant.dsa_gqa_inference import (
    _deep_select_decode_supported,
    _sparse_attention_rows,
    dynamic_sparse_attention,
)
from megatron.core.transformer.experimental_attention_variant.dsa_gqa_inference_kernels import (
    HAVE_TRITON,
    SCORE_ROW_BLOCK,
    build_row_metadata,
    dsa_gqa_indexer_scores,
    dsa_gqa_sparse_attention,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.skipif(not HAVE_TRITON, reason="Triton is required")


HIDDEN_SIZE = 1024


NUM_HEADS = 16


HEAD_DIM = 256


INDEX_DIM = 128


TOPK = 512


@pytest.fixture(scope="module", autouse=True)
def initialize_decode_test_distributed():
    Utils.initialize_model_parallel(1, 1)
    yield
    Utils.destroy_model_parallel()


def test_speculative_decode_steps_are_rejected():
    context = SimpleNamespace(num_speculative_tokens=2)
    with pytest.raises(NotImplementedError, match="MTP speculative decoding"):
        dynamic_sparse_attention(
            query=None,
            key_cache=None,
            value_cache=None,
            index_query=None,
            index_key_cache=None,
            block_table=None,
            inference_context=context,
            index_topk=TOPK,
            softmax_scale=HEAD_DIM**-0.5,
        )


def _rows_with_padding(query_lengths, kv_lengths, padding_rows, device):
    """Row metadata for requests plus empty padding rows and blocks, on ``device``."""
    row_requests, row_lengths, block_starts, block_rows, block_requests = build_row_metadata(
        torch.tensor(query_lengths), torch.tensor(kv_lengths)
    )
    pad = torch.zeros(padding_rows, dtype=torch.int32)
    row_requests = torch.cat((row_requests, pad - 1))
    row_lengths = torch.cat((row_lengths, pad))
    block_pad = torch.zeros(2, dtype=torch.int32)
    blocks = [
        torch.cat((tensor, block_pad)) for tensor in (block_starts, block_rows, block_requests)
    ]
    return [tensor.to(device) for tensor in (row_requests, row_lengths, *blocks)]


def _reference_scores(index_q, index_keys, table, row_requests, row_lengths, page_size):
    """Float64 scores of each row over its request's keys [0, length)."""
    scores = []
    for row in range(index_q.size(0)):
        length = int(row_lengths[row])
        positions = torch.arange(length, device=index_q.device)
        pages = table[max(int(row_requests[row]), 0), positions // page_size].long()
        keys = index_keys[pages, positions % page_size].double()
        scores.append(keys @ index_q[row].double())
    return scores


def _reference_attention(query, keys, values, table, row_requests, row_lengths, indices, scale):
    """Per-row attention over its valid selected keys; probabilities rounded to BF16 like PV."""
    page_size, groups = keys.size(1), keys.size(2)
    heads_per_group = query.size(1) // groups
    out = torch.zeros(query.shape, dtype=torch.float32, device=query.device)
    for row in range(query.size(0)):
        selected = indices[row]
        selected = selected[(selected >= 0) & (selected < row_lengths[row])].long()
        if selected.numel() == 0:
            continue
        pages = table[int(row_requests[row]), selected // page_size].long()
        for head in range(query.size(1)):
            group = head // heads_per_group
            k = keys[pages, selected % page_size, group].float()
            v = values[pages, selected % page_size, group].float()
            probs = (query[row, head].float() @ k.T * scale).softmax(-1)
            out[row, head] = probs @ v
    return out


def test_build_row_metadata():
    rows = build_row_metadata(torch.tensor([1, 70, 1]), torch.tensor([9, 135, 4]), row_block=32)
    row_requests, row_lengths, block_starts, block_rows, block_requests = rows
    assert row_requests.tolist() == [0] + [1] * 70 + [2]
    # Request 1 appends 70 tokens after 65 cached ones: lengths 66 .. 135.
    assert row_lengths.tolist() == [9] + list(range(66, 136)) + [4]
    assert block_starts.tolist() == [0, 1, 33, 65, 71]
    assert block_rows.tolist() == [1, 32, 32, 6, 1]
    assert block_requests.tolist() == [0, 1, 1, 1, 2]


@pytest.mark.parametrize("head_dim", [128, 192, 256])
def test_inference_kernels_match_reference_on_mixed_rows(head_dim):
    """Decode rows, a chunked-prefill block with a cached prefix, and padding rows."""
    torch.manual_seed(17)
    device = torch.device("cuda", torch.cuda.current_device())
    page_size, max_pages, heads, topk = 64, 4, 4, 16
    query_lengths, kv_lengths = [1, 70, 1], [3, 135, 137]
    row_requests, row_lengths, block_starts, block_rows, block_requests = _rows_with_padding(
        query_lengths, kv_lengths, padding_rows=3, device=device
    )
    num_rows = row_requests.numel()
    num_pages = len(query_lengths) * max_pages
    table = torch.randperm(num_pages, device=device).view(-1, max_pages).to(torch.int32)
    index_keys = torch.randn(num_pages, page_size, head_dim, device=device, dtype=torch.bfloat16)
    index_q = torch.randn(num_rows, head_dim, device=device, dtype=torch.bfloat16)

    scores = torch.empty(num_rows, 151, device=device, dtype=torch.float32)
    dsa_gqa_indexer_scores(
        index_q,
        index_keys,
        table,
        row_lengths,
        block_starts,
        block_rows,
        block_requests,
        out=scores,
        fill_invalid=True,
    )
    expected = _reference_scores(index_q, index_keys, table, row_requests, row_lengths, page_size)
    # Padding rows belong to no score block, so neither mode writes them.
    real_rows = int((row_requests >= 0).sum())
    for row in range(real_rows):
        length = int(row_lengths[row])
        torch.testing.assert_close(
            scores[row, :length].double(), expected[row], atol=2e-4, rtol=2e-4
        )
        assert bool(torch.isneginf(scores[row, length:]).all())

    # Without filling, positions past each length keep whatever the buffer held.
    unfilled = torch.full_like(scores, 7.0)
    dsa_gqa_indexer_scores(
        index_q,
        index_keys,
        table,
        row_lengths,
        block_starts,
        block_rows,
        block_requests,
        out=unfilled,
        fill_invalid=False,
    )
    tail = torch.arange(scores.size(1), device=device)[None, :] >= row_lengths[:, None]
    assert bool((unfilled[tail] == 7.0).all())
    torch.testing.assert_close(unfilled[~tail], scores[~tail])

    indices = scores.topk(topk, dim=-1).indices.to(torch.int32)
    keys = torch.randn(num_pages, page_size, 1, head_dim, device=device, dtype=torch.bfloat16)
    values = torch.randn_like(keys)
    # Attention must accept a query view with the pitched token stride produced by QKV.
    query = torch.randn(num_rows, heads + 2, head_dim, device=device, dtype=torch.bfloat16)
    query = query[:, :heads]
    output = torch.full((num_rows, heads, head_dim), 9.0, device=device, dtype=torch.bfloat16)
    scale = head_dim**-0.5
    dsa_gqa_sparse_attention(
        query, keys, values, table, row_requests, row_lengths, indices, scale, out=output
    )
    expected = _reference_attention(
        query, keys, values, table, row_requests, row_lengths, indices, scale
    )
    torch.testing.assert_close(output.float(), expected, atol=2e-2, rtol=2e-2)
    assert torch.count_nonzero(output[row_lengths == 0]) == 0


def test_inference_kernels_match_reference_across_batches():
    """Changing batches, prefill chunk sizes, score widths, and query pitches."""
    torch.manual_seed(23)
    device = torch.device("cuda", torch.cuda.current_device())
    page_size, max_pages, topk = 64, 8, 64
    scale = HEAD_DIM**-0.5
    for query_lengths in ([1], [1, 1, 1, 1, 1], [90], [1, 1, 33], [1, 200, 1, 7], [1] * 8):
        num_requests = len(query_lengths)
        kv_lengths = [
            int(torch.randint(q, max_pages * page_size + 1, ()).item()) for q in query_lengths
        ]
        row_requests, row_lengths, block_starts, block_rows, block_requests = _rows_with_padding(
            query_lengths, kv_lengths, padding_rows=1, device=device
        )
        num_rows = row_requests.numel()
        num_pages = num_requests * max_pages
        table = torch.randperm(num_pages, device=device).view(-1, max_pages).to(torch.int32)
        index_keys = torch.randn(
            num_pages, page_size, INDEX_DIM, device=device, dtype=torch.bfloat16
        )
        keys = torch.randn(num_pages, page_size, 1, HEAD_DIM, device=device, dtype=torch.bfloat16)
        values = torch.randn_like(keys)
        index_q = torch.randn(num_rows, INDEX_DIM, device=device, dtype=torch.bfloat16)
        width = max(kv_lengths)
        scores = torch.empty(num_rows, width, device=device, dtype=torch.float32)
        dsa_gqa_indexer_scores(
            index_q,
            index_keys,
            table,
            row_lengths,
            block_starts,
            block_rows,
            block_requests,
            out=scores,
            fill_invalid=True,
        )
        expected = _reference_scores(
            index_q, index_keys, table, row_requests, row_lengths, page_size
        )
        for row in range(num_rows):
            length = int(row_lengths[row])
            torch.testing.assert_close(
                scores[row, :length].double(), expected[row], atol=2e-4, rtol=2e-4
            )
        indices = scores.topk(min(topk, width), dim=-1).indices
        indices = torch.nn.functional.pad(indices, (0, topk - indices.size(1)), value=-1)
        indices = indices.to(torch.int32).contiguous()
        query = torch.randn(num_rows, NUM_HEADS + 3, HEAD_DIM, device=device, dtype=torch.bfloat16)
        query = query[:, :NUM_HEADS]
        output = torch.empty(num_rows, NUM_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
        dsa_gqa_sparse_attention(
            query, keys, values, table, row_requests, row_lengths, indices, scale, out=output
        )
        expected = _reference_attention(
            query, keys, values, table, row_requests, row_lengths, indices, scale
        )
        torch.testing.assert_close(output.float(), expected, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("deep_select_backend", [False, True])
def test_sparse_attention_rows_replay_in_cuda_graph(deep_select_backend, monkeypatch):
    """Graph replays with new mixed batches and padding must match eager runs."""
    device = torch.device("cuda", torch.cuda.current_device())
    if deep_select_backend and not _deep_select_decode_supported(device, 64):
        pytest.skip("DeepSelect is not installed or does not support this GPU")
    if not deep_select_backend:
        monkeypatch.setattr(dsa_gqa, "HAVE_DEEP_SELECT", False)
    torch.manual_seed(31)
    num_pages, page_size, max_pages, topk = 64, 64, 4, 64
    max_requests, max_rows = 4, 48
    num_blocks = max_requests + -(-max_rows // SCORE_ROW_BLOCK)
    graph_width = max_pages * page_size
    keys = torch.randn(num_pages, page_size, 1, HEAD_DIM, device=device, dtype=torch.bfloat16)
    values = torch.randn_like(keys)
    index_keys = torch.randn(num_pages, page_size, INDEX_DIM, device=device, dtype=torch.bfloat16)
    query = torch.zeros(max_rows, NUM_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
    index_query = torch.zeros(max_rows, INDEX_DIM, device=device, dtype=torch.bfloat16)
    block_table = torch.zeros(max_requests, max_pages, device=device, dtype=torch.int32)
    row_requests = torch.full((max_rows,), -1, device=device, dtype=torch.int32)
    row_lengths = torch.zeros(max_rows, device=device, dtype=torch.int32)
    blocks = [torch.zeros(num_blocks, device=device, dtype=torch.int32) for _ in range(3)]

    def attend(width):
        return _sparse_attention_rows(
            query=query,
            key_cache=keys,
            value_cache=values,
            index_query=index_query,
            index_key_cache=index_keys,
            block_table=block_table,
            row_metadata=(row_requests, row_lengths, *blocks, num_blocks),
            max_context_length=width,
            index_topk=topk,
            softmax_scale=HEAD_DIM**-0.5,
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        attend(graph_width)  # Compile outside capture, as engine warmup does.
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = attend(graph_width)

    # (query lengths, kv lengths): decode-only, mixed with a cached-prefix chunk, prefill-only.
    for query_lengths, kv_lengths in (
        ([1, 1, 1], [1, 256, 17]),
        ([1, 40, 1], [200, 140, 64]),
        ([45], [45]),
    ):
        metadata = build_row_metadata(torch.tensor(query_lengths), torch.tensor(kv_lengths))
        real_rows, real_blocks = metadata[0].numel(), metadata[2].numel()
        row_requests.fill_(-1)
        row_lengths.zero_()
        for buffer in blocks:
            buffer.zero_()
        row_requests[:real_rows] = metadata[0].to(device)
        row_lengths[:real_rows] = metadata[1].to(device)
        for buffer, values_cpu in zip(blocks, metadata[2:]):
            buffer[:real_blocks] = values_cpu.to(device)
        query.copy_(torch.randn_like(query))
        index_query.copy_(torch.randn_like(index_query))
        block_table.copy_(
            torch.randperm(num_pages, device=device)[: max_requests * max_pages].view(
                max_requests, max_pages
            )
        )
        graph.replay()
        eager = attend(max(kv_lengths))
        live = row_lengths > 0
        torch.testing.assert_close(graph_output[live], eager[live], atol=2e-2, rtol=2e-2)
        assert torch.count_nonzero(graph_output[~live]) == 0


@pytest.mark.parametrize("deep_select_backend", [False, True])
def test_score_windows_match_one_window(deep_select_backend, monkeypatch):
    """Scoring rows in small windows (not aligned to score blocks) must not change the output."""
    device = torch.device("cuda", torch.cuda.current_device())
    if deep_select_backend and not _deep_select_decode_supported(device, 64):
        pytest.skip("DeepSelect is not installed or does not support this GPU")
    if not deep_select_backend:
        monkeypatch.setattr(dsa_gqa, "HAVE_DEEP_SELECT", False)
    torch.manual_seed(43)
    page_size, max_pages, topk = 64, 8, 64
    query_lengths, kv_lengths = [1, 70, 1, 45], [300, 400, 9, 120]
    metadata = build_row_metadata(torch.tensor(query_lengths), torch.tensor(kv_lengths))
    row_metadata = (*[tensor.to(device) for tensor in metadata], metadata[2].numel())
    num_rows, num_pages = metadata[0].numel(), len(query_lengths) * max_pages
    table = torch.randperm(num_pages, device=device).view(-1, max_pages).to(torch.int32)
    keys = torch.randn(num_pages, page_size, 1, HEAD_DIM, device=device, dtype=torch.bfloat16)
    values = torch.randn_like(keys)
    index_keys = torch.randn(num_pages, page_size, INDEX_DIM, device=device, dtype=torch.bfloat16)
    query = torch.randn(num_rows, NUM_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
    index_query = torch.randn(num_rows, INDEX_DIM, device=device, dtype=torch.bfloat16)

    def attend():
        return _sparse_attention_rows(
            query=query,
            key_cache=keys,
            value_cache=values,
            index_query=index_query,
            index_key_cache=index_keys,
            block_table=table,
            row_metadata=row_metadata,
            max_context_length=max(kv_lengths),
            index_topk=topk,
            softmax_scale=HEAD_DIM**-0.5,
        )

    one_window = attend()
    # A 28 KiB buffer holds 14-17 score rows at this width, so windows straddle the 32-row
    # score blocks and the request boundaries.
    monkeypatch.setattr(dsa_gqa, "SCORE_BUFFER_BYTES", 7 * 1024 * 4)
    torch.testing.assert_close(attend(), one_window, atol=2e-2, rtol=2e-2)


def test_decode_attention_splits_groups_and_skips_invalid_slots():
    """Multi-slice, multi-group rows must match a reference over the valid selected keys.

    Selections hold padding (-1), positions past the length, and an all-padding row.
    """
    torch.manual_seed(37)
    device = torch.device("cuda", torch.cuda.current_device())
    batch, groups, heads, page_size, max_pages, topk = 5, 2, 8, 64, 8, 512
    lengths = torch.tensor([1, 70, 300, 512, 0], dtype=torch.int32, device=device)
    requests = torch.tensor([0, 1, 2, 3, -1], dtype=torch.int32, device=device)
    num_pages = batch * max_pages
    keys = torch.randn(num_pages, page_size, groups, HEAD_DIM, device=device, dtype=torch.bfloat16)
    values = torch.randn_like(keys)
    table = torch.randperm(num_pages, device=device).view(batch, max_pages).to(torch.int32)
    indices = torch.full((batch, topk), -1, dtype=torch.int32, device=device)
    for row, length in enumerate(lengths.tolist()):
        chosen = torch.randperm(max(length, 1), device=device)[: min(length, topk - 8)]
        indices[row, : chosen.numel()] = chosen.to(torch.int32)
        # Positions at or past the length must be ignored.
        indices[row, -4:] = length + torch.arange(4, device=device, dtype=torch.int32)
    query = torch.randn(batch, heads, HEAD_DIM, device=device, dtype=torch.bfloat16)
    output = torch.full((batch, heads, HEAD_DIM), 9.0, device=device, dtype=torch.bfloat16)
    scale = HEAD_DIM**-0.5
    dsa_gqa_sparse_attention(
        query, keys, values, table, requests, lengths, indices, scale, out=output
    )
    expected = _reference_attention(
        query, keys, values, table, requests.clamp(min=0), lengths, indices, scale
    )
    torch.testing.assert_close(output.float(), expected, atol=2e-2, rtol=2e-2)
    assert torch.count_nonzero(output[-1]) == 0


def test_deep_select_rows_match_torch_topk(monkeypatch):
    """DeepSelect over unfilled score tails must select what torch.topk selects."""
    device = torch.device("cuda", torch.cuda.current_device())
    if not _deep_select_decode_supported(device, TOPK):
        pytest.skip("DeepSelect is not installed or does not support this GPU")
    torch.manual_seed(41)
    num_pages, page_size, max_pages = 96, 64, 32
    query_lengths, kv_lengths = [1, 1, 1, 1, 1, 100, 1], [1, 100, 511, 512, 513, 700, 2048]
    metadata = build_row_metadata(torch.tensor(query_lengths), torch.tensor(kv_lengths))
    num_rows, num_blocks = metadata[0].numel(), metadata[2].numel()
    row_metadata = (*[tensor.to(device) for tensor in metadata], num_blocks)
    keys = torch.randn(num_pages, page_size, 1, HEAD_DIM, device=device, dtype=torch.bfloat16)
    values = torch.randn_like(keys)
    index_keys = torch.randn(num_pages, page_size, INDEX_DIM, device=device, dtype=torch.bfloat16)
    query = torch.randn(num_rows, NUM_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
    index_query = torch.randn(num_rows, INDEX_DIM, device=device, dtype=torch.bfloat16)
    table = torch.randint(0, num_pages, (len(query_lengths), max_pages), device=device)
    table = table.to(torch.int32)

    def attend():
        return _sparse_attention_rows(
            query=query,
            key_cache=keys,
            value_cache=values,
            index_query=index_query,
            index_key_cache=index_keys,
            block_table=table,
            row_metadata=row_metadata,
            max_context_length=max(kv_lengths),
            index_topk=TOPK,
            softmax_scale=HEAD_DIM**-0.5,
        )

    with_deep_select = attend()
    monkeypatch.setattr(dsa_gqa, "HAVE_DEEP_SELECT", False)
    torch.testing.assert_close(with_deep_select, attend(), atol=2e-2, rtol=2e-2)


FUSED_PAGE_SIZE = 256


def _fused_rows(query_lengths, kv_lengths, padding_rows, device):
    """Rows, score blocks and filter blocks for requests plus padding rows and blocks."""
    rows = _rows_with_padding(query_lengths, kv_lengths, padding_rows, device)
    filter_blocks = build_row_metadata(
        torch.tensor(query_lengths), torch.tensor(kv_lengths), PREFILL_ROW_BLOCK
    )[2:]
    block_pad = torch.zeros(2, dtype=torch.int32)
    filter_blocks = tuple(torch.cat((tensor, block_pad)).to(device) for tensor in filter_blocks)
    return rows[0], rows[1], tuple(rows[2:]), filter_blocks


def _fused_select_case(query_lengths, kv_lengths, padding_rows, device, seed):
    torch.manual_seed(seed)
    max_pages = -(-max(kv_lengths) // FUSED_PAGE_SIZE)
    num_pages = len(query_lengths) * max_pages
    index_keys = torch.randn(
        num_pages, FUSED_PAGE_SIZE, INDEX_DIM, device=device, dtype=torch.bfloat16
    )
    table = torch.randperm(num_pages, device=device).view(-1, max_pages).to(torch.int32)
    row_requests, row_lengths, score_blocks, filter_blocks = _fused_rows(
        query_lengths, kv_lengths, padding_rows, device
    )
    index_query = torch.randn(row_requests.numel(), INDEX_DIM, device=device, dtype=torch.bfloat16)
    return index_query, index_keys, table, row_requests, row_lengths, score_blocks, filter_blocks


def _assert_exact_topk(indices, index_query, index_keys, table, row_requests, row_lengths, topk):
    """Each row selects keys whose scores are the row's top-k scores (ties may swap keys)."""
    scores = _reference_scores(
        index_query, index_keys, table, row_requests, row_lengths, FUSED_PAGE_SIZE
    )
    for row, row_scores in enumerate(scores):
        length = row_scores.numel()
        selected = indices[row]
        valid = selected[selected >= 0].long()
        assert valid.numel() == min(length, topk), row
        assert valid.unique().numel() == valid.numel(), row
        assert bool((valid < length).all()), row
        if length > topk:
            expected = row_scores.topk(topk).values.sort().values
            torch.testing.assert_close(row_scores[valid].sort().values, expected, atol=1e-3, rtol=0)


def _fused_select_supported(device, topk):
    return dsa_gqa_fused_select.fused_select_supported(device, topk, INDEX_DIM, FUSED_PAGE_SIZE)


@pytest.mark.parametrize(
    "query_lengths,kv_lengths,decode_only",
    [
        # Decode rows from fewer than top-k keys to tens of thousands.
        ([1, 1, 1, 1, 1], [300, 600, 5000, 20000, 33000], True),
        # A chunk over a cached prefix (several filter blocks) with decode rows.
        ([1, 300, 1], [9000, 20000, 513], False),
    ],
)
def test_fused_select_matches_full_topk(query_lengths, kv_lengths, decode_only):
    device = torch.device("cuda", torch.cuda.current_device())
    if not _fused_select_supported(device, TOPK):
        pytest.skip("The fused selection needs DeepSelect, CuTe DSL and SM100/SM103")
    inputs = _fused_select_case(query_lengths, kv_lengths, 3, device, seed=53)
    index_query, index_keys, table, row_requests, row_lengths, score_blocks, filter_blocks = inputs
    indices = dsa_gqa_fused_select.fused_select_topk(
        index_query,
        index_keys,
        table,
        row_requests,
        row_lengths,
        score_blocks,
        filter_blocks,
        width=max(kv_lengths),
        index_topk=TOPK,
        decode_only=decode_only,
    )
    _assert_exact_topk(indices, index_query, index_keys, table, row_requests, row_lengths, TOPK)
    assert bool((indices[row_lengths == 0] == -1).all())


@pytest.mark.parametrize(
    "selector",
    [
        # Candidate buffers (DeepSelect needs rows of 256-multiples) too small for the ~1664
        # keys each long row keeps: every long row overflows.
        (39, 768),
        # The largest sample as threshold: every long row keeps fewer than top-k keys.
        (1, 4608),
    ],
)
@pytest.mark.parametrize("decode_only", [True, False])
def test_fused_select_fallback_is_exact(selector, decode_only, monkeypatch):
    device = torch.device("cuda", torch.cuda.current_device())
    if not _fused_select_supported(device, TOPK):
        pytest.skip("The fused selection needs DeepSelect, CuTe DSL and SM100/SM103")
    monkeypatch.setitem(dsa_gqa_fused_select.SELECTOR, TOPK, selector)
    if decode_only:
        query_lengths, kv_lengths = [1, 1, 1], [400, 3000, 12000]
    else:
        query_lengths, kv_lengths = [140, 1], [6000, 2500]
    inputs = _fused_select_case(query_lengths, kv_lengths, 2, device, seed=59)
    index_query, index_keys, table, row_requests, row_lengths, score_blocks, filter_blocks = inputs
    indices = dsa_gqa_fused_select.fused_select_topk(
        index_query,
        index_keys,
        table,
        row_requests,
        row_lengths,
        score_blocks,
        filter_blocks,
        width=max(kv_lengths),
        index_topk=TOPK,
        decode_only=decode_only,
    )
    _assert_exact_topk(indices, index_query, index_keys, table, row_requests, row_lengths, TOPK)


@pytest.mark.parametrize("decode_only", [True, False])
def test_fused_select_replays_in_cuda_graph(decode_only):
    """Graph replays with new keys, queries, tables and lengths must match eager runs."""
    device = torch.device("cuda", torch.cuda.current_device())
    if not _fused_select_supported(device, TOPK):
        pytest.skip("The fused selection needs DeepSelect, CuTe DSL and SM100/SM103")
    if decode_only:
        batches = [([1, 1, 1], [700, 9000, 4000]), ([1, 1, 1], [9000, 513, 8000])]
    else:
        batches = [([200, 1, 1], [9000, 700, 3000]), ([1, 150, 30], [6000, 9000, 160])]
    width = 9000
    max_requests, max_pages = 3, -(-width // FUSED_PAGE_SIZE)
    num_pages = max_requests * max_pages
    max_rows = max(sum(query_lengths) for query_lengths, _ in batches)
    torch.manual_seed(61)
    index_keys = torch.randn(
        num_pages, FUSED_PAGE_SIZE, INDEX_DIM, device=device, dtype=torch.bfloat16
    )
    table = torch.zeros(max_requests, max_pages, device=device, dtype=torch.int32)
    index_query = torch.zeros(max_rows, INDEX_DIM, device=device, dtype=torch.bfloat16)
    row_requests = torch.full((max_rows,), -1, device=device, dtype=torch.int32)
    row_lengths = torch.zeros(max_rows, device=device, dtype=torch.int32)
    score_blocks = tuple(
        torch.zeros(
            max_requests + -(-max_rows // SCORE_ROW_BLOCK), device=device, dtype=torch.int32
        )
        for _ in range(3)
    )
    filter_blocks = tuple(
        torch.zeros(
            max_requests + -(-max_rows // PREFILL_ROW_BLOCK), device=device, dtype=torch.int32
        )
        for _ in range(3)
    )

    def select():
        return dsa_gqa_fused_select.fused_select_topk(
            index_query,
            index_keys,
            table,
            row_requests,
            row_lengths,
            score_blocks,
            filter_blocks,
            width=width,
            index_topk=TOPK,
            decode_only=decode_only,
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        select()  # Compile outside capture, as engine warmup does.
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_indices = select()

    for query_lengths, kv_lengths in batches:
        score_metadata = build_row_metadata(torch.tensor(query_lengths), torch.tensor(kv_lengths))
        filter_metadata = build_row_metadata(
            torch.tensor(query_lengths), torch.tensor(kv_lengths), PREFILL_ROW_BLOCK
        )
        rows = score_metadata[0].numel()
        row_requests.fill_(-1)
        row_lengths.zero_()
        row_requests[:rows] = score_metadata[0].to(device)
        row_lengths[:rows] = score_metadata[1].to(device)
        for buffers, values in (
            (score_blocks, score_metadata[2:]),
            (filter_blocks, filter_metadata[2:]),
        ):
            for buffer, value in zip(buffers, values):
                buffer.zero_()
                buffer[: value.numel()] = value.to(device)
        index_keys.copy_(torch.randn_like(index_keys))
        index_query.copy_(torch.randn_like(index_query))
        table.copy_(torch.randperm(num_pages, device=device).view(max_requests, max_pages))
        graph.replay()
        eager = select()
        torch.testing.assert_close(graph_indices.sort(dim=1).values, eager.sort(dim=1).values)
        _assert_exact_topk(
            graph_indices[:rows],
            index_query[:rows],
            index_keys,
            table,
            row_requests[:rows],
            row_lengths[:rows],
            TOPK,
        )


def test_sparse_attention_rows_take_the_fused_selection_at_long_context():
    """Past FUSED_MIN_CONTEXT, attention through the fused selection matches the materialized
    selection."""
    device = torch.device("cuda", torch.cuda.current_device())
    if not _fused_select_supported(device, TOPK):
        pytest.skip("The fused selection needs DeepSelect, CuTe DSL and SM100/SM103")
    width = dsa_gqa_fused_select.FUSED_MIN_CONTEXT + 1000
    query_lengths, kv_lengths = [1, 40, 1], [width, width - 3000, 5000]
    inputs = _fused_select_case(query_lengths, kv_lengths, 3, device, seed=67)
    index_query, index_keys, table, row_requests, row_lengths, score_blocks, filter_blocks = inputs
    keys = torch.randn(
        index_keys.size(0), FUSED_PAGE_SIZE, 1, HEAD_DIM, device=device, dtype=torch.bfloat16
    )
    values = torch.randn_like(keys)
    query = torch.randn(
        index_query.size(0), NUM_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16
    )
    row_metadata = (row_requests, row_lengths, *score_blocks, score_blocks[0].numel())

    def attend(filter_blocks):
        return _sparse_attention_rows(
            query=query,
            key_cache=keys,
            value_cache=values,
            index_query=index_query,
            index_key_cache=index_keys,
            block_table=table,
            row_metadata=row_metadata,
            max_context_length=width,
            index_topk=TOPK,
            softmax_scale=HEAD_DIM**-0.5,
            filter_blocks=filter_blocks,
            decode_only=False,
        )

    torch.testing.assert_close(attend(filter_blocks), attend(None), atol=2e-2, rtol=2e-2)


def _flashinfer_supported(topk=TOPK):
    device = torch.device("cuda", torch.cuda.current_device())
    return dsa_gqa_flashinfer.flashinfer_dsa_supported(device, topk, INDEX_DIM, 256, NUM_HEADS)


def test_flashinfer_step_plan():
    """Leading single-token requests take the decode path; the rest fill 128-row tiles whose
    8-row groups are consecutive, gap-free union-core batches; sizes follow the padded batch."""
    # Two decode requests, a 300-token chunk, a single-token request after it (prefill path),
    # padded to 400 rows and 8 requests.
    plan = dsa_gqa_flashinfer.build_step_plan(
        [1, 1, 300, 1],
        [50, 9000, 1300, 77],
        padded_rows=400,
        padded_requests=8,
        decode_only=False,
        heads=NUM_HEADS,
    )
    sizes = dsa_gqa_flashinfer.plan_sizes(400, 8, False)
    assert sizes == {"decode_rows": 8, "tiles": 4 + 8, "prefill_requests": 8}
    assert plan["decode_lengths"].tolist() == [50, 9000] + [0] * 6
    assert plan["decode_requests"][:2].tolist() == [0, 1]
    assert len(plan["query_rows"]) == 12 * 128 and len(plan["batch_starts"]) == 12 * 16 + 1
    rows = plan["query_rows"][plan["query_rows"] >= 0]
    assert rows.tolist() == list(range(2, 303))
    # Request 2's rows end at keys 1001 .. 1300; request 3's single row at key 77.
    ends = plan["row_ends"][plan["query_rows"] >= 0]
    assert ends.tolist() == list(range(1001, 1301)) + [77]
    assert plan["tile_requests"][:4].tolist() == [2, 2, 2, 3]
    starts = plan["batch_starts"]
    assert (np.diff(starts) >= 0).all() and starts[0] == 2 and starts[-1] == 303
    assert plan["sample_ends"][:2].tolist() == [1300 // 128 * 3, 77 // 128 * 3]
    decode_plan = dsa_gqa_flashinfer.build_step_plan(
        [1] * 3, [10, 20, 30], padded_rows=4, padded_requests=4, decode_only=True, heads=NUM_HEADS
    )
    assert decode_plan["decode_lengths"].tolist() == [10, 20, 30, 0]
    assert decode_plan["query_rows"].size == 0


@pytest.mark.parametrize("decode_only", [True, False])
@pytest.mark.parametrize("backend", ["flashinfer", "triton_gluon"])
def test_flashinfer_matches_triton_and_replays_in_cuda_graph(decode_only, backend):
    """A captured FlashInfer layer, replayed for new batches of one padded size (new keys,
    queries, tables and lengths, plan buffers refilled in place), matches the Triton path, and
    padding rows stay zero."""
    if backend == "flashinfer" and not _flashinfer_supported():
        pytest.skip("FlashInfer's DSA kernels are not installed or not supported here")
    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(71)
    page_size, bound = 256, 9216
    if decode_only:
        padded_rows = padded_requests = 4
        batches = [([1, 1, 1], [700, 9000, 2600]), ([1, 1, 1, 1], [9000, 513, 4000, 1])]
    else:
        padded_rows, padded_requests = 640, 4
        batches = [([1, 1, 300, 37], [700, 9000, 3000, 37]), ([1, 520], [8000, 9000])]
    max_pages = bound // page_size
    num_pages = padded_requests * max_pages
    keys = torch.randn(num_pages, page_size, 1, HEAD_DIM, device=device, dtype=torch.bfloat16)
    values = torch.randn_like(keys)
    index_keys = torch.randn(num_pages, page_size, INDEX_DIM, device=device, dtype=torch.bfloat16)
    table = torch.zeros(padded_requests, max_pages, device=device, dtype=torch.int32)
    query = torch.zeros(padded_rows, NUM_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
    index_query = torch.zeros(padded_rows, INDEX_DIM, device=device, dtype=torch.bfloat16)
    sizes = dsa_gqa_flashinfer.plan_sizes(padded_rows, padded_requests, decode_only)
    empty = dsa_gqa_flashinfer.build_step_plan(
        [1],
        [1],
        padded_rows=padded_rows,
        padded_requests=padded_requests,
        decode_only=decode_only,
        heads=NUM_HEADS,
    )
    buffers = {
        name: torch.zeros(len(v), device=device, dtype=torch.int32) for name, v in empty.items()
    }
    plan = dsa_gqa_flashinfer.StepPlan(
        decode_rows=sizes["decode_rows"], num_tiles=sizes["tiles"], max_context=bound, **buffers
    )

    native_context = None
    if backend == "triton_gluon":
        native_context = DynamicInferenceContext(
            model_config=TransformerConfig(
                num_layers=1,
                add_bias_linear=False,
                hidden_size=1024,
                num_attention_heads=NUM_HEADS,
                num_query_groups=1,
                kv_channels=HEAD_DIM,
                params_dtype=torch.bfloat16,
                bf16=True,
                experimental_attention_variant="dsa",
                dsa_indexer_mode="simplified",
                dsa_kernel_backend="reference",
                dsa_inference_backend="triton_gluon",
                dsa_indexer_n_heads=1,
                dsa_indexer_head_dim=INDEX_DIM,
                dsa_indexer_topk=TOPK,
                dsa_indexer_loss_coeff=0.1,
            ),
            inference_config=InferenceConfig(
                max_sequence_length=bound,
                max_requests=padded_requests,
                max_tokens=padded_rows,
                block_size_tokens=page_size,
                buffer_size_gb=0.1,
                unified_memory_level=0,
                num_cuda_graphs=None,
                use_flashinfer_fused_rope=False,
            ),
        )
        native_context.padded_active_token_count = padded_rows
        native_context.padded_active_request_count = padded_requests
        native_context.num_prefill_requests = 0 if decode_only else 1

    def fill(query_lengths, kv_lengths):
        arrays = dsa_gqa_flashinfer.build_step_plan(
            query_lengths,
            kv_lengths,
            padded_rows=padded_rows,
            padded_requests=padded_requests,
            decode_only=decode_only,
            heads=NUM_HEADS,
        )
        for name, values_np in arrays.items():
            buffers[name].copy_(torch.from_numpy(values_np))
        real = sum(query_lengths)
        query.zero_()
        index_query.zero_()
        query[:real] = torch.randn(real, NUM_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
        index_query[:real] = torch.randn(real, INDEX_DIM, device=device, dtype=torch.bfloat16)
        keys.copy_(torch.randn_like(keys))
        index_keys.copy_(torch.randn_like(index_keys))
        table.copy_(torch.randperm(num_pages, device=device).view(padded_requests, max_pages))
        if native_context is not None:
            native_context._fill_dsa_row_metadata(
                len(query_lengths),
                padded_requests,
                torch.tensor(query_lengths),
                torch.tensor(kv_lengths),
            )
            native_context._gpu_dsa_row_buf.copy_(native_context._cpu_dsa_row_buf)
        return real

    def attend():
        if native_context is not None:
            return dsa_gqa._native_sparse_attention(
                query=query,
                key_cache=keys,
                value_cache=values,
                index_query=index_query,
                index_key_cache=index_keys,
                block_table=table,
                inference_context=native_context,
                row_metadata=native_context.dsa_row_metadata(),
                max_context_length=bound,
                index_topk=TOPK,
                softmax_scale=HEAD_DIM**-0.5,
            )
        return dsa_gqa_flashinfer.flashinfer_sparse_attention(
            query=query,
            key_cache=keys,
            value_cache=values,
            index_query=index_query,
            index_key_cache=index_keys,
            block_table=table,
            plan=plan,
            index_topk=TOPK,
            softmax_scale=HEAD_DIM**-0.5,
        )

    fill(*batches[0])
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        attend()  # Compile outside capture, as engine warmup does.
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = attend()

    for query_lengths, kv_lengths in batches:
        real = fill(query_lengths, kv_lengths)
        graph.replay()
        metadata = build_row_metadata(torch.tensor(query_lengths), torch.tensor(kv_lengths))
        expected = _sparse_attention_rows(
            query=query[:real],
            key_cache=keys,
            value_cache=values,
            index_query=index_query[:real],
            index_key_cache=index_keys,
            block_table=table[: len(query_lengths)],
            row_metadata=(*[t.to(device) for t in metadata], metadata[2].numel()),
            max_context_length=bound,
            index_topk=TOPK,
            softmax_scale=HEAD_DIM**-0.5,
        )
        torch.testing.assert_close(graph_output[:real], expected, atol=2e-2, rtol=2e-2)
        assert torch.count_nonzero(graph_output[real:]) == 0
        if native_context is not None and decode_only:
            previous = graph_output.clone()
            graph.replay()
            torch.testing.assert_close(graph_output, previous, atol=0, rtol=0)
