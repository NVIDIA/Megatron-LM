# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Dynamic-batching inference for DSA over GQA with the simplified indexer."""

import gc
import logging
import weakref
from types import SimpleNamespace

import pytest
import torch

from megatron.core.inference.config import InferenceConfig, KVCacheManagementMode
from megatron.core.inference.contexts import DynamicInferenceContext
from megatron.core.inference.contexts.dynamic_context import HAVE_TORCH_MEMORY_SAVER
from megatron.core.transformer.experimental_attention_variant import dsa_gqa_inference as dsa_gqa
from megatron.core.transformer.experimental_attention_variant.dsa_gqa_inference import (
    TritonDSADecodeWorkspace,
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
from megatron.core.transformer.experimental_attention_variant.dsa_min_memory import (
    _sort_topk_support_by_position,
)
from megatron.core.transformer.experimental_attention_variant.dsa_min_memory_triton import (
    triton_simplified_index_scores_block,
    triton_sparse_attention_tile,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

logger = logging.getLogger(__name__)

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


@pytest.fixture
def native_context_factory():
    def make_context(*, max_sequence_length, max_requests, max_tokens, **kwargs):
        return DynamicInferenceContext(
            model_config=TransformerConfig(
                num_layers=1,
                add_bias_linear=False,
                hidden_size=HIDDEN_SIZE,
                num_attention_heads=NUM_HEADS,
                num_query_groups=1,
                kv_channels=HEAD_DIM,
                params_dtype=torch.bfloat16,
                bf16=True,
                experimental_attention_variant="dsa",
                dsa_indexer_mode="simplified",
                dsa_kernel_backend="reference",
                dsa_indexer_n_heads=1,
                dsa_indexer_head_dim=INDEX_DIM,
                dsa_indexer_topk=TOPK,
                dsa_indexer_loss_coeff=0.1,
            ),
            inference_config=InferenceConfig(
                max_sequence_length=max_sequence_length,
                max_requests=max_requests,
                max_tokens=max_tokens,
                block_size_tokens=256,
                buffer_size_gb=0.1,
                unified_memory_level=0,
                num_cuda_graphs=None,
                use_flashinfer_fused_rope=False,
                **kwargs,
            ),
        )

    return make_context


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


def test_sparse_attention_rows_replay_in_cuda_graph():
    """Graph replays with new mixed batches and padding must match eager runs."""
    device = torch.device("cuda", torch.cuda.current_device())
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


def test_score_windows_match_one_window(monkeypatch):
    """Scoring rows in small windows (not aligned to score blocks) must not change the output."""
    device = torch.device("cuda", torch.cuda.current_device())
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


@pytest.mark.parametrize(
    "cache_mode,static_pointers",
    [
        (KVCacheManagementMode.PERSIST, True),
        (KVCacheManagementMode.OFFLOAD, False),
        (KVCacheManagementMode.RECOMPUTE, False),
        (KVCacheManagementMode.OFFLOAD, True),
        (KVCacheManagementMode.RECOMPUTE, True),
    ],
    ids=["persist", "offload", "recompute", "offload-static", "recompute-static"],
)
def test_native_decode_workspace_suspend_resume(
    native_context_factory, cache_mode, static_pointers
):
    """Repeated context suspension releases obsolete storage and preserves decode results."""
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("Native DSA decode requires Blackwell")
    pytest.importorskip("cutlass.cute")
    if static_pointers and cache_mode != KVCacheManagementMode.PERSIST:
        if not HAVE_TORCH_MEMORY_SAVER:
            pytest.skip("Static offload/recompute requires torch_memory_saver")

    torch.manual_seed(73)
    rows, bound, page_size = 4, 4096, 256
    context = native_context_factory(
        max_sequence_length=bound,
        max_requests=rows,
        max_tokens=rows,
        kv_cache_management_mode=cache_mode,
        static_kv_memory_pointers=static_pointers,
    )
    context.memory_buffer.normal_()
    context.dsa_memory_buffer.normal_()
    saved_kv = context.memory_buffer.cpu()
    saved_index_keys = context.dsa_memory_buffer.cpu()
    cache_bytes = sum(t.numel() * t.element_size() for t in (saved_kv, saved_index_keys))
    table = torch.arange(rows * bound // page_size, device="cuda", dtype=torch.int32).view(rows, -1)
    query = torch.randn(rows, NUM_HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    index_query = torch.randn(rows, INDEX_DIM, device="cuda", dtype=torch.bfloat16)

    def fill_metadata():
        context.padded_active_token_count = context.padded_active_request_count = rows
        context.num_prefill_requests = 0
        context._fill_dsa_row_metadata(
            3, rows, torch.tensor([1, 1, 1]), torch.tensor([bound, 2048, 33])
        )
        context._gpu_dsa_row_buf.copy_(context._cpu_dsa_row_buf)

    def attend():
        return dsa_gqa._native_sparse_attention(
            query=query,
            key_cache=context.memory_buffer[0, 0],
            value_cache=context.memory_buffer[1, 0],
            index_query=index_query,
            index_key_cache=context.dsa_memory_buffer[0],
            block_table=table,
            inference_context=context,
            row_metadata=context.dsa_row_metadata(),
            max_context_length=bound,
            index_topk=TOPK,
            softmax_scale=HEAD_DIM**-0.5,
        )

    def capture():
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            attend()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = attend()
        return graph, output

    fill_metadata()
    expected = _sparse_attention_rows(
        query=query,
        key_cache=context.memory_buffer[0, 0],
        value_cache=context.memory_buffer[1, 0],
        index_query=index_query,
        index_key_cache=context.dsa_memory_buffer[0],
        block_table=table,
        row_metadata=context.dsa_row_metadata(),
        max_context_length=bound,
        index_topk=TOPK,
        softmax_scale=HEAD_DIM**-0.5,
    )
    graph, output = capture()
    releases_storage = cache_mode != KVCacheManagementMode.PERSIST and not static_pointers
    suspended_allocations = []
    for _ in range(3):
        # Engine graph warmup resets bookkeeping without invalidating graph-owned buffers.
        context.reset()
        fill_metadata()
        graph.replay()
        torch.testing.assert_close(output, expected, atol=2e-2, rtol=2e-2)
        assert len(context._dsa_decode_workspaces) == 1
        workspace_ref = weakref.ref(next(iter(context._dsa_decode_workspaces.values()))[0])
        descriptor_base_ref = weakref.ref(workspace_ref()._key_descriptor.base)
        torch.cuda.synchronize()
        allocated_before = torch.cuda.memory_allocated()

        # Match engine order: release context storage, then discard graphs with moving pointers.
        context.deallocate_inference_state_buffers()
        if releases_storage:
            del output
            graph.reset()
        gc.collect()
        torch.cuda.synchronize()
        if releases_storage:
            assert workspace_ref() is None
            assert descriptor_base_ref() is None
            allocated_after = torch.cuda.memory_allocated()
            assert allocated_before - allocated_after >= cache_bytes
            suspended_allocations.append(allocated_after)
            assert allocated_after <= suspended_allocations[0] + 1024**2
        else:
            assert workspace_ref() is next(iter(context._dsa_decode_workspaces.values()))[0]

        context.reinitialize_inference_state_buffers()
        if cache_mode == KVCacheManagementMode.RECOMPUTE:
            context.memory_buffer.copy_(saved_kv)
            context.dsa_memory_buffer.copy_(saved_index_keys)
        fill_metadata()
        if releases_storage:
            graph, output = capture()
        graph.replay()
        torch.testing.assert_close(output, expected, atol=2e-2, rtol=2e-2)
        assert len(context._dsa_decode_workspaces) == 1


@pytest.mark.parametrize("decode_only", [True, False])
def test_native_decode_and_mixed_batches_replay_in_cuda_graph(decode_only, native_context_factory):
    """Refreshed requests, caches and padding in a captured native layer match eager rows."""
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("Native DSA decode requires Blackwell")
    pytest.importorskip("cutlass.cute")
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
    native_context = native_context_factory(
        max_sequence_length=bound, max_requests=padded_requests, max_tokens=padded_rows
    )
    native_context.padded_active_token_count = padded_rows
    native_context.padded_active_request_count = padded_requests
    native_context.num_prefill_requests = 0 if decode_only else 1

    def fill(query_lengths, kv_lengths):
        real = sum(query_lengths)
        query.zero_()
        index_query.zero_()
        query[:real] = torch.randn(real, NUM_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
        index_query[:real] = torch.randn(real, INDEX_DIM, device=device, dtype=torch.bfloat16)
        keys.copy_(torch.randn_like(keys))
        index_keys.copy_(torch.randn_like(index_keys))
        table.copy_(torch.randperm(num_pages, device=device).view(padded_requests, max_pages))
        native_context._fill_dsa_row_metadata(
            len(query_lengths),
            padded_requests,
            torch.tensor(query_lengths),
            torch.tensor(kv_lengths),
        )
        native_context._gpu_dsa_row_buf.copy_(native_context._cpu_dsa_row_buf)
        return real

    def attend():
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
        if decode_only:
            previous = graph_output.clone()
            graph.replay()
            torch.testing.assert_close(graph_output, previous, atol=0, rtol=0)


def _training_forward(query, index_query, index_keys, keys, values, selected, topk):
    """Call the scoring and attention kernels used by simplified DSA-GQA training."""
    length = index_keys.shape[0]
    index_scale = index_query.shape[-1] ** -0.5
    # Future keys would win routing without the training kernel's causal mask.
    future = index_query.expand(16, -1) * 64
    index_keys = torch.cat((index_keys, future), dim=0)[:, None, None, :]
    keys = torch.cat((keys, query[0].expand(16, -1) * 64), dim=0)[:, None, None, :]
    future_values = values.new_full((16, values.shape[-1]), 32)
    values = torch.cat((values, future_values), dim=0)[:, None, None, :]
    query = query[None, None, :, :]
    scores = triton_simplified_index_scores_block(
        index_query[None, None, None, :], index_keys, index_scale, length - 1, 0
    )
    assert (
        scores is not None
    ), "Training scorer must execute its Triton kernel, not a torch fallback"
    assert bool(torch.isneginf(scores[..., length:]).all())
    selected_scores, indices = scores.topk(min(topk, length), dim=-1)
    _, indices = _sort_topk_support_by_position(selected_scores, indices)
    training_output = triton_sparse_attention_tile(
        query, keys, values, indices, query.shape[-1] ** -0.5, length - 1
    )
    same_support_output = triton_sparse_attention_tile(
        query, keys, values, selected[None, None, :], query.shape[-1] ** -0.5, length - 1
    )
    assert (
        training_output is not None and same_support_output is not None
    ), "Training attention must execute its Triton kernel, not a torch fallback"
    return scores[0, 0, :length], indices[0, 0], training_output[0, 0], same_support_output[0, 0]


@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("seed", [71, 2026, 17])
@pytest.mark.parametrize("context", [32768, 262144])
@pytest.mark.parametrize("topk", [512, 1024, 2048])
def test_native_decode_matches_training_kernels_and_graph_refresh(head_dim, seed, context, topk):
    """Decode matches the training forward kernels on the final causal query row."""
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("Native DSA decode requires Blackwell")
    pytest.importorskip("cutlass.cute")
    torch.manual_seed(seed)
    device = torch.device("cuda", torch.cuda.current_device())
    batch, page, heads = 8, 256, 16
    pages = batch * context // page
    query = torch.randn(batch, heads, head_dim, device=device, dtype=torch.bfloat16)
    index_query = torch.randn(batch, 128, device=device, dtype=torch.bfloat16)
    index_keys = torch.randn(pages, page, 128, device=device, dtype=torch.bfloat16)
    keys = torch.randn(pages, page, 1, head_dim, device=device, dtype=torch.bfloat16)
    values = torch.randn_like(keys)
    table = torch.randperm(pages, device=device).to(torch.int32).view(batch, -1)
    requests = torch.randperm(batch, device=device).to(torch.int32)
    lengths_cpu = [0, 1, 31, 63, 511, 2048, 32767, 32768]
    if context > 65536:
        lengths_cpu = [0, 1, 2047, 2048, 65535, 65536, 200000, context]
    lengths = torch.tensor(lengths_cpu, device=device, dtype=torch.int32)
    workspace = TritonDSADecodeWorkspace(
        batch_size=batch,
        num_query_heads=heads,
        head_dim=head_dim,
        index_topk=topk,
        max_context=context,
        page_size=page,
        device=device,
    )

    def run():
        return workspace(
            query=query,
            index_query=index_query,
            index_key_cache=index_keys,
            key_cache=keys,
            value_cache=values,
            block_table=table,
            context_lengths=lengths,
            request_indices=requests,
            softmax_scale=head_dim**-0.5,
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    max_score_error = 0.0
    max_output_error = 0.0
    for _ in range(2):
        query.neg_()
        index_query.neg_()
        table.copy_(table.roll(1, dims=0))
        graph.replay()
        actual = workspace.output.clone()
        selected = workspace.indices.clone()
        expected = torch.zeros_like(actual)
        same_support = torch.zeros_like(actual)
        for row, length in enumerate(lengths_cpu):
            if not length:
                assert bool((selected[row] == -1).all())
                continue
            logical = torch.arange(length, device=device)
            physical = table[requests[row].long(), logical // page].long()
            chosen = selected[row][selected[row] >= 0].long()
            assert chosen.numel() == min(topk, length)
            assert chosen.unique().numel() == chosen.numel()
            assert bool((chosen < length).all())
            training_scores, training_indices, training_output, same_support_output = (
                _training_forward(
                    query[row],
                    index_query[row],
                    index_keys[physical, logical % page],
                    keys[physical, logical % page, 0],
                    values[physical, logical % page, 0],
                    chosen,
                    topk,
                )
            )
            expected[row] = training_output
            same_support[row] = same_support_output
            # Decode ranks raw dot products; training applies a positive scalar.
            scaled_scores = workspace.scores[row, :length] * 128**-0.5
            torch.testing.assert_close(scaled_scores, training_scores, atol=2e-5, rtol=2e-5)
            torch.testing.assert_close(
                chosen.sort().values, training_indices.sort().values, atol=0, rtol=0
            )
            max_score_error = max(
                max_score_error, (scaled_scores - training_scores).abs().max().item()
            )
        # Blackwell training retains FP32 probabilities; decode rounds BF16 weights
        # before PV. Keep this budget uniform across shapes.
        torch.testing.assert_close(actual, same_support, atol=3e-3, rtol=2e-2)
        torch.testing.assert_close(actual, expected, atol=3e-3, rtol=2e-2)
        max_output_error = max(
            max_output_error, (actual.float() - expected.float()).abs().max().item()
        )
        graph.replay()
        torch.testing.assert_close(workspace.output, actual, atol=0, rtol=0)
        torch.testing.assert_close(workspace.indices, selected, atol=0, rtol=0)

    logger.info(
        "Training/decode parity: seed=%s head_dim=%s max_score_abs=%s max_output_abs=%s",
        seed,
        head_dim,
        max_score_error,
        max_output_error,
    )


@pytest.mark.parametrize(
    "width,topk", [(516, 512), (4096, 1024), (32772, 2048), (65536, 2048), (131072, 2048)]
)
def test_cute_topk_cutoffs_ties_and_graph_replay(width, topk):
    """Replay switches between CuTe and torch top-k with ragged rows and infinities."""
    pytest.importorskip("cutlass.cute")
    torch.manual_seed(71)
    device = torch.device("cuda", torch.cuda.current_device())
    batch = 6
    from megatron.core.transformer.experimental_attention_variant.dsa_gqa_topk import (
        CuteTopKWorkspace,
    )

    alignment = 4
    stride = -(-width // alignment) * alignment
    scores = torch.randn(batch, stride, device=device)
    lengths = torch.tensor([width - 3, 1, 0, width, width, width], device=device, dtype=torch.int32)
    scores[3].zero_()
    indices = torch.empty(batch, topk, device=device, dtype=torch.int32)

    selector = CuteTopKWorkspace(scores, lengths, indices)

    def run():
        selector()

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()

    for adversarial in (False, True, False):
        scores.normal_()
        scores[3].zero_()
        if adversarial:
            # Valid -inf values must outrank padding, even if padding contains +inf.
            scores[0, : width - 3] = -float("inf")
            scores[0, width - 3 :] = float("inf")
            scores[4, :width].copy_(torch.arange(width, device=device).float() / width)
            scores[4, 0] = -1e20
            scores[4, 1] = 1e20
            scores[5, ::3] = float("inf")
            scores[5, 1::3] = -float("inf")
        graph.replay()
        assert bool(selector.flags.any()) == adversarial
        if selector.chunks > 1:
            assert bool(selector.merge_flags.any()) == adversarial
        for row, length in enumerate(lengths.tolist()):
            count = min(topk, length)
            valid = indices[row][indices[row] >= 0].long()
            assert valid.numel() == count
            assert valid.unique().numel() == count
            assert bool((valid < length).all())
            assert bool((indices[row, count:] == -1).all())
            if count:
                torch.testing.assert_close(
                    scores[row, valid].sort().values,
                    scores[row, :length].topk(count).values.sort().values,
                    atol=0,
                    rtol=0,
                )
        expected = indices.clone()
        graph.replay()
        torch.testing.assert_close(indices, expected, atol=0, rtol=0)
        run()
        torch.testing.assert_close(indices, expected, atol=0, rtol=0)
