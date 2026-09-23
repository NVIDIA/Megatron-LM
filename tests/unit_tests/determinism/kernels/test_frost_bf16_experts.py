# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Replay Frost's staging/activation kernels and reject atomic wgrad in deterministic mode."""

import pytest
import torch
import triton

from megatron.core.fusions.frost_bf16_experts import (
    FrostBf16Experts,
    _frost_maps,
    _frost_offsets,
    _frost_pack_backward,
    _frost_pack_forward,
    _frost_unpack_backward,
    _frost_unpack_forward,
    _weighted_swiglu,
)
from megatron.core.transformer.moe.experts import TEGroupedMLP
from tests.unit_tests.determinism.kernels.harness import (
    bytes_equal,
    deterministic_algorithms,
    seeded,
)
from tests.unit_tests.fusions.test_frost_bf16_experts import _config
from tests.unit_tests.fusions.test_frost_bf16_module import _make_arm
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def test_frost_staging_replays_bit_exact():
    """Every padding/gather/scatter stage replays and round-trips an uneven layout."""
    seeded()
    counts = torch.tensor([257, 0, 129], device="cuda", dtype=torch.int64)
    rows, hidden, capacity = 386, 256, 1024
    x = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
    p = torch.rand(rows, device="cuda")

    def run():
        offsets, padded = (torch.empty(3, device="cuda", dtype=torch.int32) for _ in range(2))
        forward = torch.empty(capacity, device="cuda", dtype=torch.int32)
        inverse = torch.empty(rows, device="cuda", dtype=torch.int32)
        packed = torch.full((capacity, hidden), float("nan"), device="cuda", dtype=x.dtype)
        packed_p = torch.full((capacity,), float("nan"), device="cuda")
        packed_dy = torch.empty_like(packed)
        y, dx, dp = torch.empty_like(x), torch.empty_like(x), torch.empty_like(p)
        _frost_offsets[(1,)](counts, offsets, padded, 3, 4)
        _frost_maps[(triton.cdiv(capacity, 128),)](
            offsets, padded, forward, inverse, capacity, 3, 4
        )
        _frost_pack_forward[(triton.cdiv(capacity * hidden, 2048),)](
            x, p, forward, packed, packed_p, capacity, hidden
        )
        _frost_pack_backward[(triton.cdiv(capacity * hidden, 2048),)](
            x, forward, packed_dy, capacity, hidden
        )
        _frost_unpack_forward[(triton.cdiv(rows * hidden, 2048),)](packed, inverse, y, rows, hidden)
        _frost_unpack_backward[(triton.cdiv(rows * hidden, 2048),)](
            packed_dy, packed_p, inverse, dx, dp, rows, hidden
        )
        assert torch.equal(y, x) and torch.equal(dx, x) and torch.equal(dp, p)
        assert torch.equal(offsets, counts.cumsum(0).int())
        assert torch.equal(padded, torch.tensor([512, 512, 768], device="cuda", dtype=torch.int32))
        assert torch.count_nonzero(packed[forward < 0]) == 0
        assert torch.count_nonzero(packed_p[forward < 0]) == 0
        return (offsets, padded, forward, inverse, packed, packed_p, packed_dy, y, dx, dp)

    with deterministic_algorithms(True):
        reference = run()
        for _ in range(3):
            for actual, expected in zip(run(), reference):
                assert bytes_equal(actual, expected)


def test_frost_weighted_activation_replays_bit_exact():
    """Compiled FP32 clamped SwiGLU is bit-exact across poisoned output reuse."""
    seeded()
    intermediate = torch.randn(768, 512, device="cuda", dtype=torch.bfloat16) * 16
    probability = torch.rand(768, 1, 1, device="cuda")
    output = torch.empty(768, 256, device="cuda", dtype=torch.bfloat16)
    compiled = torch.compile(
        _weighted_swiglu, fullgraph=True, options={"emulate_precision_casts": True}
    )
    with deterministic_algorithms(True):
        compiled(intermediate, probability, output, 10.0)
        expected = output.clone()
        for _ in range(3):
            output.fill_(float("nan"))
            compiled(intermediate, probability, output, 10.0)
            assert bytes_equal(output, expected)


def test_frost_atomic_wgrad_rejects_deterministic_mode():
    """Never silently opt a deterministic Megatron run into unordered FP32 atomics."""
    with pytest.raises(ValueError, match="Frost"):
        _config(deterministic_mode=True)
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("runtime Frost contract requires SM100")
    owner = FrostBf16Experts(2, 256, 256, 10.0)
    with deterministic_algorithms(True), pytest.raises(ValueError, match="atomic accumulation"):
        owner(
            torch.empty(2, 256, device="cuda", dtype=torch.bfloat16),
            torch.ones(2, device="cuda", dtype=torch.int64),
            torch.ones(2, device="cuda"),
        )
    assert owner.compiled_plans is None


def test_frost_module_rejects_deterministic_algorithms():
    """Actual TEGroupedMLP dispatch rejects atomic wgrad before modifying main_grad."""
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("runtime Frost contract requires SM100")
    Utils.initialize_model_parallel()
    try:
        module, ddp = _make_arm("frost")
        assert isinstance(module, TEGroupedMLP)
        ddp.zero_grad_buffer()
        with deterministic_algorithms(True), pytest.raises(ValueError, match="atomic accumulation"):
            ddp(
                torch.randn(2, 256, device="cuda", dtype=torch.bfloat16),
                torch.ones(2, device="cuda", dtype=torch.int64),
                torch.ones(2, device="cuda"),
            )
        assert all(torch.count_nonzero(p.main_grad) == 0 for p in module.parameters())
    finally:
        Utils.destroy_model_parallel()
