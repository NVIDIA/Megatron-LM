# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Bit-exact BI DSA replay with tied routing under competing CUDA work."""

import pytest
import torch

from megatron.core.inference.contexts.attention_context.dsa_metadata import DSARowPlan
from megatron.core.transformer.experimental_attention_variant.dsa_gqa_inference import (
    _sparse_attention_rows,
)
from tests.unit_tests.determinism.kernels.harness import assert_replays_bit_exact

pytestmark = pytest.mark.launch_on_gb200


def test_tied_routing_forward_replays(monkeypatch, record_property):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("Batch-invariant DSA currently targets SM100")
    pytest.importorskip("deep_select")
    torch.manual_seed(39)
    query = torch.randn(3, 16, 128, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(8, 128, 1, 128, device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    index_query = torch.zeros(3, 128, device="cuda", dtype=torch.bfloat16)
    index_key = torch.ones(8, 128, 128, device="cuda", dtype=torch.bfloat16)
    table = torch.tensor([[7, 2, 4, 0, 5, 1, 3, 6]], device="cuda", dtype=torch.int32)
    lengths = torch.tensor([1024, 769, 0], device="cuda", dtype=torch.int32)
    requests = torch.tensor([0, 0, -1], device="cuda", dtype=torch.int32)
    direct_rows = False

    def forward(query, key, value, index_query, index_key, table, lengths, requests):
        plan = DSARowPlan(
            requests,
            lengths,
            torch.arange(3, device="cuda", dtype=torch.int32),
            (lengths > 0).to(torch.int32),
            torch.zeros_like(requests),
            lengths[:0],
            table,
            direct_rows,
        )
        return _sparse_attention_rows(
            query=query,
            key_cache=key,
            value_cache=value,
            index_query=index_query,
            index_key_cache=index_key,
            plan=plan,
            max_context_length=1024,
            index_topk=512,
            softmax_scale=128**-0.5,
            batch_invariant=True,
        )

    inputs = (query, key, value, index_query, index_key, table, lengths, requests)
    expected = forward(*inputs)
    direct_rows = True
    assert torch.equal(forward(*inputs), expected)
    from megatron.core.transformer.experimental_attention_variant import dsa_gqa_inference

    monkeypatch.setattr(dsa_gqa_inference, "SCORE_BUFFER_BYTES", 1024 * torch.float32.itemsize)
    outputs, _ = assert_replays_bit_exact(
        forward,
        inputs,
        backward=False,
        replays=3,
        contention=True,
        what="batch-invariant simplified DSA",
    )
    assert torch.isfinite(outputs["out"]).all()
    assert torch.count_nonzero(outputs["out"][2]) == 0
    assert torch.equal(outputs["out"], expected)
    record_property("score_window_max_abs_diff", 0.0)
