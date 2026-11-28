# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Bit-exact replay of DSA-over-GQA inference through FlashInfer's DSA kernels.

Covers the NVFP4 index-key and query quantize-and-write, and a full mixed
decode + prefill step of ``flashinfer_sparse_attention`` (indexer scoring, threshold selection,
exact top-k, and sparse attention), with the plan built on the host. NVFP4 index keys are
rejected in batch-invariant mode: their scores often tie at the top-k boundary, and which tied
key the selection keeps can vary between runs. Decode replays
bit-exactly only with ``deterministic=True`` (batch-invariant mode), which fixes the order of each
row's selected keys.
"""

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant import dsa_gqa_flashinfer
from tests.unit_tests.determinism.kernels.harness import assert_replays_bit_exact, seeded

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

HEADS, HEAD_DIM, INDEX_DIM, TOPK, PAGE = 16, 256, 128, 512, 256


def _require_flashinfer_dsa(index_key_dtype="bf16"):
    device = torch.device("cuda", torch.cuda.current_device())
    if not (
        dsa_gqa_flashinfer.HAVE_FLASHINFER_DSA
        and dsa_gqa_flashinfer.flashinfer_dsa_supported(device, TOPK, INDEX_DIM, PAGE, HEADS)
    ):
        pytest.skip("FlashInfer's DSA kernels are not installed or not supported here")
    if index_key_dtype == "nvfp4" and not dsa_gqa_flashinfer.HAVE_FLASHINFER_DSA_NVFP4:
        pytest.skip("FlashInfer's NVFP4 indexer kernels are not installed")
    return device


def test_nvfp4_index_key_write_replays():
    """Quantizing a large step of keys (padding tokens included) into the NVFP4 cache, and its
    queries into NVFP4 rows, is bit-exact under contention."""
    device = _require_flashinfer_dsa("nvfp4")
    seeded()
    blocks, tokens = 512, 65536
    keys = torch.randn(tokens, INDEX_DIM, device=device, dtype=torch.bfloat16) * 4
    query = torch.randn(tokens, INDEX_DIM, device=device, dtype=torch.bfloat16)
    slot = torch.randperm(blocks * PAGE, device=device)[:tokens]
    block_idx, positions = slot // PAGE, slot % PAGE
    block_idx[::5] = -1

    def write(keys, query):
        cache = torch.zeros(
            dsa_gqa_flashinfer.nvfp4_index_cache_shape(blocks, PAGE),
            device=device,
            dtype=torch.uint8,
        )
        packed, scales = dsa_gqa_flashinfer.write_nvfp4_index_keys(
            keys, query, cache, block_idx, positions
        )
        return cache, packed, scales

    assert_replays_bit_exact(
        write,
        dict(keys=keys, query=query),
        backward=False,
        contention=True,
        what="FlashInfer NVFP4 index-key write",
    )


@pytest.mark.parametrize("decode_tokens", [1, 3], ids=["decode", "speculative-decode"])
def test_flashinfer_sparse_attention_replays(decode_tokens):
    """A mixed step replays bit-exactly, including speculative-decoding steps whose decode
    requests add 1 + k rows, each a decode-path row with its own causal key count."""
    device = _require_flashinfer_dsa()
    seeded()
    d = decode_tokens
    query_lengths, kv_lengths = [d, d, 300, 37], [700, 6000, 3000, 37]
    padded_rows, padded_requests, bound = 384, 4, 6144
    max_pages = bound // PAGE
    num_pages = padded_requests * max_pages
    keys = torch.randn(num_pages, PAGE, 1, HEAD_DIM, device=device, dtype=torch.bfloat16)
    values = torch.randn_like(keys)
    index_keys = torch.randn(num_pages, PAGE, INDEX_DIM, device=device, dtype=torch.bfloat16)
    table = torch.randperm(num_pages, device=device, dtype=torch.int32).view(padded_requests, -1)
    query = torch.randn(padded_rows, HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
    index_query = torch.randn(padded_rows, INDEX_DIM, device=device, dtype=torch.bfloat16)
    arrays = dsa_gqa_flashinfer.build_step_plan(
        query_lengths,
        kv_lengths,
        padded_rows=padded_rows,
        padded_requests=padded_requests,
        decode_only=False,
        heads=HEADS,
        decode_tokens=d,
    )
    sizes = dsa_gqa_flashinfer.plan_sizes(padded_rows, padded_requests, False, decode_tokens=d)
    page_table = torch.zeros(
        padded_requests,
        dsa_gqa_flashinfer.page_table_columns(max_pages, PAGE, bound, TOPK),
        device=device,
        dtype=torch.int32,
    )
    dsa_gqa_flashinfer.fill_page_table(table, PAGE, page_table)
    plan = dsa_gqa_flashinfer.StepPlan(
        decode_rows=sizes["decode_rows"],
        num_tiles=sizes["tiles"],
        max_context=bound,
        page_table=page_table,
        **{name: torch.from_numpy(v).to(device) for name, v in arrays.items()},
    )

    def attend(query, index_query, keys, values, index_keys):
        return dsa_gqa_flashinfer.flashinfer_sparse_attention(
            query=query,
            key_cache=keys,
            value_cache=values,
            index_query=index_query,
            index_key_cache=index_keys,
            plan=plan,
            index_topk=TOPK,
            softmax_scale=HEAD_DIM**-0.5,
            deterministic=True,
        )

    assert_replays_bit_exact(
        attend,
        dict(query=query, index_query=index_query, keys=keys, values=values, index_keys=index_keys),
        backward=False,
        what="FlashInfer DSA sparse attention",
    )
