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
    return dsa_gqa_flashinfer.HAVE_FLASHINFER_DSA and dsa_gqa_flashinfer.flashinfer_dsa_supported(
        device, topk, INDEX_DIM, 256, NUM_HEADS
    )


requires_flashinfer_dsa = pytest.mark.skipif(
    not torch.cuda.is_available() or not _flashinfer_supported(),
    reason="FlashInfer's DSA kernels are not installed or not supported on this device",
)


requires_flashinfer_nvfp4 = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not _flashinfer_supported()
    or not dsa_gqa_flashinfer.HAVE_FLASHINFER_DSA_NVFP4,
    reason="FlashInfer's NVFP4 indexer kernels are not installed or not supported on this device",
)
INDEX_KEY_DTYPES = ["bf16", pytest.param("nvfp4", marks=requires_flashinfer_nvfp4)]

# E2M1 values by their 4-bit code (sign in bit 3).
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _dequantize_nvfp4(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """FP32 values of NVFP4 rows: packed uint8 ``[..., Di/2]`` (element 2i in the low nibble of
    byte i) and E4M3 scale bytes ``[..., Di/16]``."""
    table = torch.tensor(E2M1_VALUES, device=packed.device)
    table = torch.cat((table, -table))
    codes = torch.stack((packed & 0xF, packed >> 4), dim=-1).flatten(-2).long()
    block_scales = scales.view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=-1)
    return table[codes] * block_scales


def _quantize_nvfp4_reference(x: torch.Tensor):
    """An independent NVFP4 quantizer of ``[..., Di]`` rows: a scale per 16 values, E4M3 of their
    amax over 6 (no global scale), and each value times the scale's reciprocal rounded to the
    nearest E2M1 value, ties to even (saturating at 6). Returns the dequantized values and the
    scale bytes."""
    blocks = x.float().unflatten(-1, (-1, 16))
    scale = (blocks.abs().amax(-1) / 6.0).to(torch.float8_e4m3fn)
    factor = scale.float()
    inverse = torch.where(factor > 0, 1.0 / factor, torch.zeros_like(factor))
    scaled = (blocks * inverse[..., None]).clamp(-6.0, 6.0)
    grid = torch.tensor(E2M1_VALUES, device=x.device, dtype=torch.float64)
    distance = (scaled.abs().double()[..., None] - grid).abs()
    # Ties go to the even code (0, 1, 2 and 4 have an even mantissa); FP32 distances that are not
    # tied differ by far more than this.
    odd = torch.arange(len(E2M1_VALUES), device=x.device) % 2
    nearest = grid[(distance + odd * 1e-12).argmin(-1)].float()
    values = (nearest.copysign(scaled) * factor[..., None]).flatten(-2)
    return values, scale.view(torch.uint8)


def _nvfp4_cache_values(cache: torch.Tensor, blocks: int, block_size: int) -> torch.Tensor:
    """Dequantized keys ``[blocks, block_size, Di]`` of an NVFP4 index-key cache."""
    keys, scales = dsa_gqa_flashinfer.nvfp4_index_pages(cache)
    return _dequantize_nvfp4(keys, scales).view(blocks, block_size, -1)


def _write_all_nvfp4(index_keys: torch.Tensor, index_query: torch.Tensor, cache: torch.Tensor):
    """Quantize every key of a ``[blocks, block_size, Di]`` BF16 cache into ``cache`` and the
    queries into NVFP4 rows, through the context's write path."""
    blocks, block_size, _ = index_keys.shape
    token = torch.arange(blocks * block_size, device=index_keys.device)
    return dsa_gqa_flashinfer.write_nvfp4_index_keys(
        index_keys.view(blocks * block_size, -1),
        index_query,
        cache,
        token // block_size,
        token % block_size,
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


def _reference_rows(
    query, index_query, keys, values, index_keys, table, requests, lengths, tied=None
):
    """Exact float64 top-k selection of each row's causal keys, then attention over them.

    Rows whose top-k boundary is a tie (the k-th and (k+1)-th largest scores are equal) have no
    unique selection; their indices are appended to ``tied`` when it is given."""
    page_size = keys.size(1)
    out = torch.zeros(query.shape, dtype=torch.float32, device=query.device)
    for row, (request, length) in enumerate(zip(requests, lengths)):
        positions = torch.arange(length, device=query.device)
        pages = table[request, positions // page_size].long()
        slots = positions % page_size
        scores = index_keys[pages, slots].double() @ index_query[row].double()
        if tied is not None and length > TOPK:
            boundary = scores.topk(TOPK + 1).values
            if boundary[-1] == boundary[-2]:
                tied.append(row)
        selected = scores.topk(min(TOPK, length)).indices
        k = keys[pages[selected], slots[selected], 0].float()
        v = values[pages[selected], slots[selected], 0].float()
        probs = (query[row].float() @ k.T * HEAD_DIM**-0.5).softmax(-1)
        out[row] = probs @ v
    return out


@requires_flashinfer_dsa
@pytest.mark.parametrize("index_key_dtype", INDEX_KEY_DTYPES)
@pytest.mark.parametrize("decode_only", [True, False])
def test_flashinfer_replays_in_cuda_graph(decode_only, index_key_dtype):
    """A captured layer, replayed for new batches of one padded size (new keys, queries, tables
    and lengths, plan buffers refilled in place), matches an exact PyTorch reference, and
    padding rows stay zero. With NVFP4 index keys the captured region also quantizes the keys
    and queries, and the reference selects by the dequantized NVFP4 scores."""
    nvfp4 = index_key_dtype == "nvfp4"
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
    nvfp4_cache = torch.zeros(
        dsa_gqa_flashinfer.nvfp4_index_cache_shape(num_pages, page_size),
        device=device,
        dtype=torch.uint8,
    )
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

    nvfp4_query = []

    def attend():
        if not nvfp4:
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
        packed, scales = _write_all_nvfp4(index_keys, index_query, nvfp4_cache)
        nvfp4_query[:] = [packed, scales]
        return dsa_gqa_flashinfer.flashinfer_sparse_attention(
            query=query,
            key_cache=keys,
            value_cache=values,
            index_query=packed,
            index_key_cache=nvfp4_cache,
            plan=plan,
            index_topk=TOPK,
            softmax_scale=HEAD_DIM**-0.5,
            index_query_scales=scales,
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
        selector_query, selector_keys = index_query, index_keys
        if nvfp4:
            selector_query = _dequantize_nvfp4(*nvfp4_query)
            selector_keys = _nvfp4_cache_values(nvfp4_cache, num_pages, page_size)
        tied = []
        expected = _reference_rows(
            query[:real],
            selector_query[:real],
            keys,
            values,
            selector_keys,
            table,
            requests,
            lengths,
            tied,
        )
        # NVFP4 scores take few distinct values, so a row's top-k boundary can be a tie, which
        # the kernels may break either way; compare the rows with a unique selection.
        unique = torch.ones(real, dtype=torch.bool, device=device)
        unique[tied] = False
        assert len(tied) <= 0.05 * real, f"{len(tied)} of {real} rows tie at the top-k boundary"
        torch.testing.assert_close(
            graph_output[:real][unique].float(), expected[unique], atol=2e-2, rtol=2e-2
        )
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


# NVFP4 index keys.


def test_nvfp4_index_keys_need_the_flashinfer_backend():
    """NVFP4 index keys are rejected with the triton backend, and unknown formats at all."""
    with pytest.raises(ValueError, match="needs dsa_backend='flashinfer'"):
        _inference_config(dsa_backend="triton", dsa_indexer_key_dtype="nvfp4")
    with pytest.raises(ValueError, match="Unsupported dsa_indexer_key_dtype"):
        _inference_config(dsa_indexer_key_dtype="fp8")
    with pytest.raises(ValueError, match="needs dsa_backend='flashinfer'"):
        _inference_config(dsa_indexer_key_dtype="nvfp4")  # the default backend is triton
    config = _inference_config(dsa_backend="flashinfer", dsa_indexer_key_dtype="nvfp4")
    assert config.dsa_indexer_key_dtype == "nvfp4"


@requires_flashinfer_nvfp4
def test_nvfp4_index_keys_are_rejected_in_batch_invariant_mode():
    """Tied NVFP4 scores make the selection run-dependent, so batch-invariant (deterministic)
    attention refuses NVFP4 index keys rather than claim bit-exact replay."""
    device = torch.device("cuda", torch.cuda.current_device())
    with pytest.raises(ValueError, match="batch-invariant"):
        dsa_gqa_flashinfer.flashinfer_sparse_attention(
            query=torch.zeros(1, NUM_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16),
            key_cache=None,
            value_cache=None,
            index_query=torch.zeros(1, 64, device=device, dtype=torch.uint8),
            index_key_cache=None,
            plan=None,
            index_topk=TOPK,
            softmax_scale=HEAD_DIM**-0.5,
            deterministic=True,
            index_query_scales=torch.zeros(1, 8, device=device, dtype=torch.uint8),
        )


def _decode_scores(packed, scales, cache, table, lengths):
    """FlashInfer's NVFP4 indexer scores of one query row per request."""
    keys, key_scales = dsa_gqa_flashinfer.nvfp4_index_pages(cache)
    width = -(-int(lengths.max()) // 4) * 4
    scores = torch.full((packed.size(0), width), float("nan"), device=packed.device)
    dsa_gqa_flashinfer.indexer_scores(
        packed,
        keys,
        table,
        lengths,
        requests=torch.arange(packed.size(0), device=packed.device, dtype=torch.int32),
        scores=scores,
        query_scales=scales,
        key_scales=key_scales,
    )
    return scores


def _paged(blocks, block_size, requests, device):
    """A random block table of ``requests`` rows over ``blocks`` blocks, and its 64-key pages."""
    per = blocks // requests
    table = torch.randperm(blocks, device=device, dtype=torch.int32).view(requests, per)
    pages = torch.zeros(requests, per * block_size // 64, device=device, dtype=torch.int32)
    dsa_gqa_flashinfer.fill_page_table(table, block_size, pages)
    return table, pages


@requires_flashinfer_nvfp4
def test_nvfp4_quantize_and_score_match_reference():
    """Keys written through the context's NVFP4 path match an independent quantizer (scales
    bit-exactly, values but for rounding ties), padding tokens write nothing, and FlashInfer's
    scores equal FP32 dot products of the dequantized queries and keys."""
    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(5)
    blocks, block_size, requests = 32, 256, 4
    cache = torch.full(
        dsa_gqa_flashinfer.nvfp4_index_cache_shape(blocks, block_size),
        0xAB,
        device=device,
        dtype=torch.uint8,
    )
    # Keys with outlier channels and a wide range of magnitudes, plus padding tokens.
    tokens = 3000
    keys = (
        torch.randn(tokens, INDEX_DIM, device=device)
        * torch.logspace(-2, 1.5, INDEX_DIM, device=device)[
            torch.randperm(INDEX_DIM, device=device)
        ]
    )
    keys = keys.to(torch.bfloat16)
    query = torch.randn(requests, INDEX_DIM, device=device, dtype=torch.bfloat16)
    slot = torch.randperm(blocks * block_size, device=device)[:tokens]
    block_idx, positions = slot // block_size, slot % block_size
    block_idx[::7] = -1  # padding tokens
    packed, query_scales = dsa_gqa_flashinfer.write_nvfp4_index_keys(
        keys, query, cache, block_idx, positions
    )

    cached = _nvfp4_cache_values(cache, blocks, block_size)
    written = block_idx >= 0
    expected_values, expected_scales = _quantize_nvfp4_reference(keys[written])
    key_pages, key_scale_pages = dsa_gqa_flashinfer.nvfp4_index_pages(cache)
    page = (block_idx[written] * block_size + positions[written]) // 64
    row = positions[written] % 64
    assert torch.equal(key_scale_pages[page, row], expected_scales)
    actual = cached[block_idx[written], positions[written]]
    assert (actual == expected_values).float().mean() > 0.9999
    step = expected_values.abs().unflatten(-1, (-1, 16)).amax(-1).repeat_interleave(16, -1) / 6
    assert ((actual - expected_values).abs() <= 2 * step + 1e-6).all()
    # Padding tokens and untouched slots keep their old bytes.
    untouched = torch.ones(blocks * block_size, dtype=torch.bool, device=device)
    untouched[block_idx[written] * block_size + positions[written]] = False
    all_pages = cache.view(-1, dsa_gqa_flashinfer.NVFP4_PAGE_BYTES)
    key_bytes = all_pages[:, : 64 * 64].reshape(-1, 64)
    assert (key_bytes[untouched] == 0xAB).all()
    query_values, query_scale_bytes = _quantize_nvfp4_reference(query)
    assert torch.equal(query_scales, query_scale_bytes)
    assert (_dequantize_nvfp4(packed, query_scales) == query_values).float().mean() > 0.9999

    # Scores, over a cache with every slot written.
    all_keys = torch.randn(blocks, block_size, INDEX_DIM, device=device, dtype=torch.bfloat16)
    packed, query_scales = _write_all_nvfp4(all_keys * 3, query, cache)
    cached = _nvfp4_cache_values(cache, blocks, block_size)
    table, pages = _paged(blocks, block_size, requests, device)
    lengths = torch.tensor([1, 700, 2048, 8 * block_size], device=device, dtype=torch.int32)
    scores = _decode_scores(packed, query_scales, cache, pages, lengths)
    dequantized_query = _dequantize_nvfp4(packed, query_scales)
    for request, length in enumerate(lengths.tolist()):
        position = torch.arange(length, device=device)
        keys_of = cached[table[request, position // block_size].long(), position % block_size]
        reference = keys_of @ dequantized_query[request]
        torch.testing.assert_close(scores[request, :length], reference, atol=1e-3, rtol=1e-4)


def _bf16_and_nvfp4_top_k(index_keys, query, top_k):
    """Each query row's exact top-k keys by BF16 scores and by FlashInfer's NVFP4 scores, as
    selected among ``index_keys`` (one request per query row, all its keys visible)."""
    device = query.device
    rows, context = query.size(0), index_keys.size(1)
    block_size = 256
    blocks = rows * context // block_size
    cache = torch.empty(
        dsa_gqa_flashinfer.nvfp4_index_cache_shape(blocks, block_size),
        device=device,
        dtype=torch.uint8,
    )
    token = torch.arange(rows * context, device=device)
    packed, scales = dsa_gqa_flashinfer.write_nvfp4_index_keys(
        index_keys.reshape(-1, INDEX_DIM), query, cache, token // block_size, token % block_size
    )
    table = torch.arange(blocks, device=device, dtype=torch.int32).view(rows, -1)
    pages = torch.zeros(rows, context // 64, device=device, dtype=torch.int32)
    dsa_gqa_flashinfer.fill_page_table(table, block_size, pages)
    lengths = torch.full((rows,), context, device=device, dtype=torch.int32)
    nvfp4_scores = _decode_scores(packed, scales, cache, pages, lengths)[:, :context]
    bf16_scores = torch.einsum("rkd,rd->rk", index_keys.double(), query.double())
    return bf16_scores.topk(top_k).indices, nvfp4_scores.topk(top_k).indices


def _overlap(first, second):
    """Mean fraction of each row's selected keys that both selections hold."""
    hits = [
        np.intersect1d(a.cpu().numpy(), b.cpu().numpy()).size / a.numel()
        for a, b in zip(first, second)
    ]
    return float(np.mean(hits))


@requires_flashinfer_nvfp4
@pytest.mark.parametrize("top_k", [512, 2048])
def test_nvfp4_top_k_overlap_with_bf16(top_k):
    """Measured top-k overlap of NVFP4 selection with BF16 selection. On i.i.d. Gaussian keys
    (the hardest case: many near-equal scores at the boundary) most keys still agree; on
    structured keys (outlier channels, a low-rank component, and planted relevant keys) every
    planted key is selected and the overlap is higher."""
    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(17)
    rows, context, dim = 8, 8192, INDEX_DIM
    query = torch.randn(rows, dim, device=device)
    random_keys = torch.randn(rows, context, dim, device=device)
    bf16_top, nvfp4_top = _bf16_and_nvfp4_top_k(random_keys.bfloat16(), query.bfloat16(), top_k)
    random_overlap = _overlap(bf16_top, nvfp4_top)

    channel_scale = torch.ones(dim, device=device)
    channel_scale[torch.randperm(dim, device=device)[:6]] = 8.0
    basis = torch.randn(16, dim, device=device)
    structured = (
        torch.randn(rows, context, dim, device=device) * 0.5
        + torch.randn(rows, context, 16, device=device) @ basis * 0.25
    ) * channel_scale
    planted = torch.stack([torch.randperm(context, device=device)[:64] for _ in range(rows)])
    relevant = query / query.norm(dim=-1, keepdim=True) * 4 * dim**0.5
    structured.scatter_add_(
        1, planted[..., None].expand(-1, -1, dim), relevant[:, None].expand(-1, planted.size(1), -1)
    )
    bf16_top_s, nvfp4_top_s = _bf16_and_nvfp4_top_k(structured.bfloat16(), query.bfloat16(), top_k)
    structured_overlap = _overlap(bf16_top_s, nvfp4_top_s)
    print(
        f"NVFP4 vs BF16 top-{top_k} overlap of {context} keys: random {random_overlap:.4f}, "
        f"structured {structured_overlap:.4f}"
    )
    for row in range(rows):
        assert np.isin(planted[row].cpu().numpy(), nvfp4_top_s[row].cpu().numpy()).all()
    assert random_overlap > 0.80
    assert structured_overlap > 0.85


@requires_flashinfer_nvfp4
def test_context_sizes_the_nvfp4_index_key_cache():
    """An NVFP4 index key and its scales take 72 bytes, not 256, of every block."""
    Utils.initialize_model_parallel(1, 1)
    try:
        config = _config()
        bf16 = DynamicInferenceContext(
            model_config=config, inference_config=_inference_config(dsa_backend="flashinfer")
        )
        nvfp4 = DynamicInferenceContext(
            model_config=config,
            inference_config=_inference_config(
                dsa_backend="flashinfer", dsa_indexer_key_dtype="nvfp4"
            ),
        )
        assert nvfp4.dsa_memory_buffer.dtype == torch.uint8
        assert nvfp4.dsa_memory_buffer.shape == (
            1,
            nvfp4.kv_block_allocator.pool_size,
            nvfp4.block_size_tokens // 64,
            64 * 72,
        )
        index_bytes = config.params_dtype.itemsize * INDEX_DIM
        assert nvfp4.block_size_bytes == bf16.block_size_bytes - bf16.block_size_tokens * (
            index_bytes - 72
        )
        # The smaller blocks fit at least as many of them in the same buffer.
        assert nvfp4.kv_block_allocator.pool_size >= bf16.kv_block_allocator.pool_size
    finally:
        Utils.destroy_model_parallel()
