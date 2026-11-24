# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Bit-exact replay of the FlashInfer DSA step plan's device op (the per-step page-table fill)."""

import pytest
import torch

from megatron.core.inference.contexts.attention_context import dsa_flashinfer_plan
from tests.unit_tests.determinism.kernels.harness import assert_replays_bit_exact, seeded

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

PAGE = 256


def test_fill_page_table_replay():
    seeded()
    table = torch.randint(0, 8, (4, 3), device="cuda", dtype=torch.int32)

    def fill(table):
        pages = torch.zeros(4, 16, device="cuda", dtype=torch.int32)
        dsa_flashinfer_plan.fill_page_table(table, PAGE, pages)
        return pages

    assert_replays_bit_exact(
        fill, dict(table=table), backward=False, what="DSA FlashInfer page table"
    )


def test_gpu_staged_decode_plan_replay():
    """The all-decode-path plan the MTP forwards stage on the GPU (decode rows, their key counts
    and the page table, with -1 padding requests clamped out) replays bit-exactly."""
    from megatron.core.inference.contexts.attention_context.dsa_flashinfer_metadata import (
        FlashInferDSAMetadata,
    )

    seeded()
    requests, blocks = 4, 3
    metadata = FlashInferDSAMetadata(
        max_requests=requests,
        max_tokens=64,
        max_kv_blocks=blocks,
        block_size_tokens=PAGE,
        max_sequence_length=blocks * PAGE,
        query_heads=16,
        index_topk=512,
        decode_tokens=3,
    )
    table = torch.randint(0, 8, (requests, blocks), device="cuda", dtype=torch.int32)
    table[3] = -1
    row_requests = torch.tensor([0, 0, 1, 1, 2, 2], device="cuda")
    row_lengths = torch.randint(1, blocks * PAGE, (6,), device="cuda", dtype=torch.int32)

    def stage(table, row_lengths):
        metadata.stage_decode_rows(row_requests, row_lengths, 8, requests, table)
        plan = metadata.plan()
        return plan.decode_requests.clone(), plan.decode_lengths.clone(), plan.page_table.clone()

    assert_replays_bit_exact(
        stage,
        dict(table=table, row_lengths=row_lengths),
        backward=False,
        what="DSA FlashInfer GPU-staged decode plan",
    )
