# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Bit-exact replay of DSA-over-GQA inference through FlashInfer's DSA kernels.

Covers the page-table fill and a full mixed decode + prefill step of
``flashinfer_sparse_attention`` (indexer scoring, threshold selection, exact top-k, and sparse
attention), with the plan built on the host. Decode replays
bit-exactly only with ``deterministic=True`` (batch-invariant mode), which fixes the order of each
row's selected keys.
"""

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant import dsa_gqa_flashinfer
from tests.unit_tests.determinism.kernels.harness import assert_replays_bit_exact, seeded

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

HEADS, HEAD_DIM, INDEX_DIM, TOPK, PAGE = 16, 256, 128, 512, 256


def test_fill_page_table_replay():
    seeded()
    table = torch.randint(0, 8, (4, 3), device="cuda", dtype=torch.int32)

    def fill(table):
        pages = torch.zeros(4, 16, device="cuda", dtype=torch.int32)
        dsa_gqa_flashinfer.fill_page_table(table, PAGE, pages)
        return pages

    assert_replays_bit_exact(
        fill, dict(table=table), backward=False, what="DSA FlashInfer page table"
    )


def test_flashinfer_sparse_attention_replays():
    device = torch.device("cuda", torch.cuda.current_device())
    if not (
        dsa_gqa_flashinfer.HAVE_FLASHINFER_DSA
        and dsa_gqa_flashinfer.flashinfer_dsa_supported(device, TOPK, INDEX_DIM, PAGE, HEADS)
    ):
        pytest.skip("FlashInfer's DSA kernels are not installed or not supported here")
    seeded()
    query_lengths, kv_lengths = [1, 1, 300, 37], [700, 6000, 3000, 37]
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
    )
    sizes = dsa_gqa_flashinfer.plan_sizes(padded_rows, padded_requests, False)
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
