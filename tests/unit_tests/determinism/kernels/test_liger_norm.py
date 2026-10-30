# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Bit-exact replay of Liger RMSNorm, including its weight-gradient reduction."""

from types import SimpleNamespace

import pytest
import torch

from tests.unit_tests.determinism.kernels.harness import (
    CONTENTION_TOKENS,
    assert_module_replays_bit_exact,
    deterministic_algorithms,
    seeded,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("zero_centered", [False, True])
@pytest.mark.parametrize("deterministic", [False, True])
def test_liger_rmsnorm_replay(dtype, zero_centered, deterministic):
    pytest.importorskip("liger_kernel.megatron")
    from megatron.core.ops.norm.liger_norm import LigerRMSNorm

    seeded()
    config = SimpleNamespace(
        normalization="RMSNorm", layernorm_zero_centered_gamma=zero_centered, sequence_parallel=True
    )
    norm = LigerRMSNorm(config=config, hidden_size=1024, eps=1e-6).to("cuda", dtype=dtype)
    x = torch.randn(CONTENTION_TOKENS, 1, 1024, device="cuda", dtype=dtype, requires_grad=True)
    with deterministic_algorithms(deterministic):
        assert_module_replays_bit_exact(
            norm, (x,), grad_output=torch.randn_like(x), contention=True, what="Liger RMSNorm"
        )
