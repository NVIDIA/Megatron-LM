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
