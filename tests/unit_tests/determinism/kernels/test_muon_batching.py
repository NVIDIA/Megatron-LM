# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Bit-exact replay of duplicated Muon's batched expert updates."""

import pytest
import torch

pytest.importorskip("emerging_optimizers")

from megatron.core.optimizer.emerging_optimizers import TensorParallelMuon
from megatron.core.utils import is_emerging_optimizers_min_version
from tests.unit_tests.determinism.kernels.harness import assert_replays_bit_exact, seeded

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_emerging_optimizers_min_version("0.3.0"),
    reason="Requires CUDA and emerging-optimizers >= 0.3.0",
)


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("precision", ["highest", "medium"])
def test_duplicated_muon_step_replays(batch_size, precision):
    """Replay weights and momentum over multiple steps, including a singleton tail."""
    seeded()
    initial = [torch.randn(512, 768, device="cuda") for _ in range(7)]
    grads = tuple(torch.randn_like(value) for value in initial)

    def run_steps(*gradients):
        params = [torch.nn.Parameter(value.clone()) for value in initial]
        for param in params:
            param.expert_tp = True
        optimizer = TensorParallelMuon(
            params,
            ns_batch_size=batch_size,
            fp32_matmul_prec=precision,
            momentum=0.9,
            weight_decay=0.1,
        )
        for _ in range(3):
            for param, grad in zip(params, gradients):
                param.grad = grad.clone()
            optimizer.step()
        return {
            "weights": [param.detach().clone() for param in params],
            "momentum": [optimizer.state[param]["momentum_buffer"].clone() for param in params],
        }

    assert_replays_bit_exact(
        run_steps,
        grads,
        replays=3,
        backward=False,
        contention=True,
        what=f"TensorParallelMuon[ns_batch_size={batch_size}, precision={precision}]",
    )
