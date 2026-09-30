# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Nucleus support parity and same-input replay at exact cutoffs and ties."""

import pytest
import torch

from megatron.core.inference.sampling.torch_sampling import TorchSampling
from tests.unit_tests.determinism.kernels.harness import assert_replays_bit_exact


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("top_p", [0.1, 0.5, 0.9])
@pytest.mark.parametrize("tied", [False, True])
def test_top_p_matches_training_tail_cutoff(device, top_p, tied):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    logits = torch.zeros(3, 16, device=device)
    if not tied:
        logits[:] = torch.arange(16, device=device).float() / 4
    # Explicit policy-recomputation reference: retain mass above the removed
    # lower tail, including its exact-threshold convention.
    values, indices = logits.sort(dim=-1, descending=False)
    keep = values.softmax(dim=-1).cumsum(dim=-1) > 1 - top_p
    keep[:, -1] = True
    expected = values.masked_fill(~keep, -torch.inf).scatter(
        -1, indices, values.masked_fill(~keep, -torch.inf)
    )
    actual = TorchSampling.filter_logits(logits, 1.0, 0, top_p)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if tied and top_p == 0.5:
        assert torch.isfinite(actual).sum(dim=-1).tolist() == [8, 8, 8]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_torch_top_p_filter_replays():
    logits = torch.zeros(4, 128, device="cuda")
    assert_replays_bit_exact(
        lambda x: TorchSampling.filter_logits(x, 1.0, 0, 0.5),
        (logits,),
        backward=False,
        what="torch top-p support",
    )
