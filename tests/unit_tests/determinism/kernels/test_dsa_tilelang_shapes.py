# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Forward replay for SparseMLA shape dispatch; backward uses atomic accumulation."""

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant.ops import tilelang_dsa
from tests.unit_tests.determinism.kernels.harness import assert_replays_bit_exact


@pytest.mark.parametrize("dim", [512, 576])
@pytest.mark.parametrize("topk", [64, 65])
@pytest.mark.parametrize("heads", [8, 96])
def test_absorbed_shape_forward_replays(dim, topk, heads):
    """NoPE, top-k and head padding retain bit-exact forward replay under stream contention.

    ``heads=8`` is padded up to the 16-head minimum tile by the wrapper; ``heads=96`` is not a
    power of two and runs natively, with the kernel masking its partial final head block.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    pytest.importorskip("tilelang")
    if tilelang_dsa.SparseMLA is None:
        pytest.skip("TileLang SparseMLA is unavailable")
    torch.manual_seed(23)
    query = torch.randn(256, 1, heads, dim, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(256, 1, 1, dim, device="cuda", dtype=torch.bfloat16)
    indices = torch.arange(256, device="cuda")[:, None] - torch.arange(topk, device="cuda")
    indices = indices.clamp_min(-1).unsqueeze(0).to(torch.int32)
    outputs, _ = assert_replays_bit_exact(
        tilelang_dsa.fused_sparse_mla_absorbed,
        (query, key, indices, 0.037, 512),
        backward=False,
        replays=3,
        contention=True,
        what="TileLang absorbed SparseMLA shape adaptation",
    )
    assert torch.isfinite(outputs["out"]).all()
