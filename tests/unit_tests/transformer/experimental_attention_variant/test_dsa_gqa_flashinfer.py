# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""DSA-over-GQA dynamic-batching inference through FlashInfer's DSA kernels."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from megatron.core.inference.config import InferenceConfig
from megatron.core.inference.contexts import DynamicInferenceContext
from megatron.core.inference.contexts.attention_context.dsa_flashinfer_metadata import (
    FlashInferDSAMetadata,
)
from megatron.core.inference.contexts.attention_context.dsa_metadata import DSAMetadata
from megatron.core.transformer.experimental_attention_variant import dsa_gqa_flashinfer
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

HIDDEN_SIZE = 1024
NUM_HEADS = 16
HEAD_DIM = 256
INDEX_DIM = 128
TOPK = 512


def _config(**overrides) -> TransformerConfig:
    kwargs = dict(
        num_layers=1,
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=NUM_HEADS,
        num_query_groups=1,
        kv_channels=HEAD_DIM,
        params_dtype=torch.bfloat16,
        bf16=True,
        normalization="RMSNorm",
        add_bias_linear=False,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        experimental_attention_variant="dsa",
        dsa_indexer_mode="simplified",
        dsa_kernel_backend="reference",
        dsa_indexer_n_heads=1,
        dsa_indexer_head_dim=INDEX_DIM,
        dsa_indexer_topk=TOPK,
        dsa_indexer_loss_coeff=0.1,
    )
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


def _flashinfer_supported(topk=TOPK):
    device = torch.device("cuda", torch.cuda.current_device())
    return (
        dsa_gqa_flashinfer.HAVE_FLASHINFER_DSA
        and dsa_gqa_flashinfer.flashinfer_dsa_supported(device, topk, INDEX_DIM, 256, NUM_HEADS)
    )


requires_flashinfer_dsa = pytest.mark.skipif(
    not torch.cuda.is_available() or not _flashinfer_supported(),
    reason="FlashInfer's DSA kernels are not installed or not supported on this device",
)


@pytest.fixture(scope="module", autouse=True)
def initialize_distributed():
    Utils.initialize_distributed()


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_fill_page_table():
    """The page table splits each KV block into FlashInfer's 64-key pages."""
    device = torch.device("cuda", torch.cuda.current_device())
    table = torch.tensor([[2, 0], [5, 1]], device=device, dtype=torch.int32)
    pages = torch.full((3, 10), -7, device=device, dtype=torch.int32)
    dsa_gqa_flashinfer.fill_page_table(table, 256, pages)
    assert pages[:2, :8].tolist() == [[8, 9, 10, 11, 0, 1, 2, 3], [20, 21, 22, 23, 4, 5, 6, 7]]
    assert (pages[:, 8:] == -7).all() and (pages[2] == -7).all()


def _rows_of(query_lengths, kv_lengths):
    """Each step row's request and causal key count."""
    requests, lengths = [], []
    for request, (q_len, kv_len) in enumerate(zip(query_lengths, kv_lengths)):
        requests += [request] * q_len
        lengths += list(range(kv_len - q_len + 1, kv_len + 1))
    return requests, lengths


def _reference_rows(query, index_query, keys, values, index_keys, table, requests, lengths):
    """Exact float64 top-k selection of each row's causal keys, then attention over them."""
    page_size = keys.size(1)
    out = torch.zeros(query.shape, dtype=torch.float32, device=query.device)
    for row, (request, length) in enumerate(zip(requests, lengths)):
        positions = torch.arange(length, device=query.device)
        pages = table[request, positions // page_size].long()
        slots = positions % page_size
        scores = index_keys[pages, slots].double() @ index_query[row].double()
        selected = scores.topk(min(TOPK, length)).indices
        k = keys[pages[selected], slots[selected], 0].float()
        v = values[pages[selected], slots[selected], 0].float()
        probs = (query[row].float() @ k.T * HEAD_DIM**-0.5).softmax(-1)
        out[row] = probs @ v
    return out


@requires_flashinfer_dsa
@pytest.mark.parametrize("decode_only", [True, False])
def test_flashinfer_replays_in_cuda_graph(decode_only):
    """A captured layer, replayed for new batches of one padded size (new keys, queries, tables
    and lengths, plan buffers refilled in place), matches an exact PyTorch reference, and
    padding rows stay zero."""
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
    page_table = torch.zeros(
        padded_requests,
        dsa_gqa_flashinfer.page_table_columns(max_pages, page_size, bound, TOPK),
        device=device,
        dtype=torch.int32,
    )
    plan = dsa_gqa_flashinfer.StepPlan(
        decode_rows=sizes["decode_rows"],
        num_tiles=sizes["tiles"],
        max_context=bound,
        page_table=page_table,
        **buffers,
    )

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
        dsa_gqa_flashinfer.fill_page_table(table, page_size, page_table)
        return real

    def attend():
        return dsa_gqa_flashinfer.flashinfer_sparse_attention(
            query=query,
            key_cache=keys,
            value_cache=values,
            index_query=index_query,
            index_key_cache=index_keys,
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
        requests, lengths = _rows_of(query_lengths, kv_lengths)
        expected = _reference_rows(
            query[:real], index_query[:real], keys, values, index_keys, table, requests, lengths
        )
        torch.testing.assert_close(graph_output[:real].float(), expected, atol=2e-2, rtol=2e-2)
        assert torch.count_nonzero(graph_output[real:]) == 0


# Dynamic inference context and attention layer.


def _inference_config(**overrides) -> InferenceConfig:
    kwargs = dict(
        max_sequence_length=2048,
        max_requests=4,
        max_tokens=4096,
        block_size_tokens=64,
        buffer_size_gb=1,
        num_cuda_graphs=None,
        unified_memory_level=0,
    )
    kwargs.update(overrides)
    return InferenceConfig(**kwargs)


def test_speculative_decode_steps_are_rejected():
    context = SimpleNamespace(num_speculative_tokens=2)
    with pytest.raises(NotImplementedError, match="MTP speculative decoding"):
        dsa_gqa_flashinfer.dynamic_sparse_attention(
            query=None,
            key_cache=None,
            value_cache=None,
            index_query=None,
            index_key_cache=None,
            inference_context=context,
            index_topk=TOPK,
            softmax_scale=HEAD_DIM**-0.5,
        )


@requires_flashinfer_dsa
def test_context_builds_the_step_plan_of_its_backend():
    """The flashinfer backend pages indexer keys like the triton one, with its own step plan."""
    Utils.initialize_model_parallel(1, 1)
    try:
        triton = DynamicInferenceContext(
            model_config=_config(), inference_config=_inference_config(block_size_tokens=256)
        )
        context = DynamicInferenceContext(
            model_config=_config(), inference_config=_inference_config(dsa_backend="flashinfer")
        )
        assert isinstance(triton.dsa_metadata, DSAMetadata)
        assert isinstance(context.dsa_metadata, FlashInferDSAMetadata)
        assert context.dsa_memory_buffer.shape == (
            1,
            context.kv_block_allocator.pool_size,
            context.block_size_tokens,
            INDEX_DIM,
        )
    finally:
        Utils.destroy_model_parallel()


@requires_flashinfer_dsa
def test_flashinfer_mixed_step_replays_bit_exactly_when_deterministic():
    """With deterministic=True (batch-invariant mode) a mixed decode + prefill step replays
    bit-exactly: decode sorts each row's selected keys, whose order the top-k kernel varies."""
    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(1234)
    page_size, bound = 256, 6144
    query_lengths, kv_lengths = [1, 1, 300, 37], [700, 6000, 3000, 37]
    padded_rows, padded_requests = 384, 4
    max_pages = bound // page_size
    num_pages = padded_requests * max_pages
    keys = torch.randn(num_pages, page_size, 1, HEAD_DIM, device=device, dtype=torch.bfloat16)
    values = torch.randn_like(keys)
    index_keys = torch.randn(num_pages, page_size, INDEX_DIM, device=device, dtype=torch.bfloat16)
    table = torch.randperm(num_pages, device=device, dtype=torch.int32).view(padded_requests, -1)
    query = torch.randn(padded_rows, NUM_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
    index_query = torch.randn(padded_rows, INDEX_DIM, device=device, dtype=torch.bfloat16)
    arrays = dsa_gqa_flashinfer.build_step_plan(
        query_lengths,
        kv_lengths,
        padded_rows=padded_rows,
        padded_requests=padded_requests,
        decode_only=False,
        heads=NUM_HEADS,
    )
    sizes = dsa_gqa_flashinfer.plan_sizes(padded_rows, padded_requests, False)
    page_table = torch.zeros(
        padded_requests,
        dsa_gqa_flashinfer.page_table_columns(max_pages, page_size, bound, TOPK),
        device=device,
        dtype=torch.int32,
    )
    dsa_gqa_flashinfer.fill_page_table(table, page_size, page_table)
    plan = dsa_gqa_flashinfer.StepPlan(
        decode_rows=sizes["decode_rows"],
        num_tiles=sizes["tiles"],
        max_context=bound,
        page_table=page_table,
        **{name: torch.from_numpy(v).to(device) for name, v in arrays.items()},
    )

    def attend():
        return dsa_gqa_flashinfer.flashinfer_sparse_attention(
            query=query.clone(),
            key_cache=keys.clone(),
            value_cache=values.clone(),
            index_query=index_query.clone(),
            index_key_cache=index_keys.clone(),
            plan=plan,
            index_topk=TOPK,
            softmax_scale=HEAD_DIM**-0.5,
            deterministic=True,
        )

    reference = attend()
    for _ in range(5):
        assert torch.equal(attend().view(torch.int16), reference.view(torch.int16))
