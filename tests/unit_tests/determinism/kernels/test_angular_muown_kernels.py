# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Bit-exact replay of the fused AngularMuOwn row kernels (one program per row, so each row's
column reductions run in a fixed order)."""

import pytest
import torch

from megatron.core.fusions.fused_angular_muown import (
    HAVE_TRITON,
    fused_angular_muown_apply_update,
    fused_angular_muown_momentum_inputs,
)
from tests.unit_tests.determinism.kernels.harness import assert_replays_bit_exact, seeded

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and HAVE_TRITON), reason="needs a GPU and Triton"
)


def _rows(n_rows=512, n_cols=5000):
    """Several column blocks per row, with a partial last block."""
    seeded()
    weight = torch.randn(n_rows, n_cols, device="cuda")
    return (
        weight,
        torch.randn(n_rows, n_cols, device="cuda", dtype=torch.bfloat16),
        torch.randn(n_rows, n_cols, device="cuda"),
        weight.norm(dim=1, keepdim=True),
        torch.randn(n_rows, 1, device="cuda"),
        torch.rand(n_rows, 1, device="cuda"),
    )


@pytest.mark.parametrize("nesterov", [False, True], ids=["momentum", "nesterov"])
def test_momentum_inputs_replay(nesterov):
    def fn(weight, grad, momentum, row_scale, exp_avg, exp_avg_sq):
        ns_input = torch.empty_like(weight)
        fused_angular_muown_momentum_inputs(
            weight,
            grad,
            momentum,
            ns_input,
            row_scale,
            exp_avg,
            exp_avg_sq,
            beta=0.9,
            betas=(0.9, 0.95),
            norm_eps=1e-12,
            nesterov=nesterov,
        )
        return momentum, ns_input, exp_avg, exp_avg_sq

    assert_replays_bit_exact(
        fn, _rows(), replays=8, backward=False, contention=True, what="angular_muown_inputs"
    )


def test_apply_update_replay():
    def fn(weight, update, _, row_scale, exp_avg, exp_avg_sq):
        fused_angular_muown_apply_update(
            weight,
            update,
            row_scale,
            exp_avg,
            exp_avg_sq,
            direction_lr=0.035,
            lr=0.05,
            decay=0.995,
            bias_corrections=(0.271, 0.142625),
            adam_eps=1e-8,
            norm_eps=1e-12,
        )
        return weight, row_scale

    assert_replays_bit_exact(
        fn, _rows(), replays=8, backward=False, contention=True, what="angular_muown_apply"
    )
