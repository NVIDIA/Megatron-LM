# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Bit-exact repeatability of the qualified batched expert Newton-Schulz path."""

import pytest
import torch

from tests.unit_tests.determinism.kernels.harness import assert_replays_bit_exact


def test_batched_expert_ns_replays():
    pytest.importorskip("emerging_optimizers")
    from megatron.core.optimizer.muon_expert_normalization import reference_norm_batched_ns

    class SingletonGroup:
        """Local domain: no collectives participate in batched normalization."""

        def size(self):
            return 1

    previous = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("medium")
        value = torch.randn(2, 1856, 672, device="cuda")

        def run(gradient):
            return reference_norm_batched_ns(gradient, 16, "polar_express", SingletonGroup())

        assert_replays_bit_exact(run, (value,), replays=3, backward=False, what="expert NS16")
    finally:
        torch.set_float32_matmul_precision(previous)
