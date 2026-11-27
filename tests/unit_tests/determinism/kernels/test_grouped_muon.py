# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Replay grouped Muon's per-expert 2-D Newton-Schulz dispatch and state updates."""

import pytest
import torch

from megatron.core.optimizer.emerging_optimizers import HAVE_EMERGING_OPTIMIZERS, TensorParallelMuon
from tests.unit_tests.determinism.kernels.harness import assert_replays_bit_exact

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not HAVE_EMERGING_OPTIMIZERS,
    reason="requires CUDA and emerging_optimizers",
)


def test_grouped_muon_steps_replay_bit_exact():
    generator = torch.Generator(device="cuda").manual_seed(314)
    weight = torch.randn(3, 128, 256, device="cuda", generator=generator)
    gradients = torch.randn(3, *weight.shape, device="cuda", generator=generator)

    def update(initial, gradients):
        param = torch.nn.Parameter(initial.clone())
        param.is_grouped_expert_weight = True
        optimizer = TensorParallelMuon([param], lr=0.01, pg_collection=None)
        history = []
        for gradient in gradients:
            param.grad = gradient.clone()
            optimizer.step()
            history.append(
                (param.detach().clone(), optimizer.state[param]["momentum_buffer"].clone())
            )
        return history

    assert_replays_bit_exact(
        update,
        (weight, gradients),
        replays=3,
        backward=False,
        contention=True,
        what="grouped Muon weights and momentum",
    )
