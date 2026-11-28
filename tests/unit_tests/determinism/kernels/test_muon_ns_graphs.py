# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Bit-exact update/state replay and output ownership for local Muon NS graphs."""

import pytest
import torch

import megatron.core.optimizer.emerging_optimizers as muon
from megatron.core.optimizer.muon_ns_cuda_graph import MuonNewtonSchulzCudaGraphs
from megatron.core.process_groups_config import ProcessGroupCollection
from tests.unit_tests.test_utilities import Utils


def test_cpu_fallback_preserves_arguments_and_output():
    """Ineligible tensors must not initialize CUDA or alter the callable contract."""
    seen = []
    expected = torch.ones(2, 3)

    def reference(value, **kwargs):
        seen.append((value, kwargs))
        return expected

    value = torch.zeros(2, 3)
    replay = MuonNewtonSchulzCudaGraphs(reference)
    assert replay(value, 5, "quintic", None) is expected
    assert seen[0][0] is value
    assert seen[0][1] == dict(
        steps=5, coefficient_type="quintic", tp_group=None, partition_dim=None, tp_mode="duplicated"
    )
    assert not replay.cache


@pytest.mark.skipif(not muon.HAVE_EMERGING_OPTIMIZERS, reason="requires emerging-optimizers")
@pytest.mark.parametrize("nesterov", [False, True])
def test_graphs_preserve_optimizer_updates_and_state(nesterov):
    Utils.initialize_model_parallel(tensor_model_parallel_size=1)
    previous = torch.get_float32_matmul_precision()
    try:
        torch.manual_seed(2109)
        pg = ProcessGroupCollection.use_mpu_process_groups()
        initial = torch.randn(128, 64, device="cuda")
        actual, expected = [torch.nn.Parameter(initial.clone()) for _ in range(2)]
        options = dict(
            lr=0.001,
            momentum=0.9,
            nesterov=nesterov,
            weight_decay=0.1,
            num_ns_steps=5,
            coefficient_type="quintic",
            pg_collection=pg,
        )
        fast = muon.TensorParallelMuon([actual], ns_cuda_graphs=True, **options)
        eager = muon.TensorParallelMuon([expected], **options)
        for _ in range(3):
            gradient = torch.randn_like(initial)
            actual.grad, expected.grad = gradient.clone(), gradient.clone()
            rng = torch.cuda.get_rng_state().clone()
            fast.step()
            eager.step()
            assert torch.equal(torch.cuda.get_rng_state(), rng)
            assert torch.equal(actual, expected)
            assert torch.equal(actual.grad, gradient)
            for key, value in fast.state[actual].items():
                other = eager.state[expected][key]
                if isinstance(value, torch.Tensor):
                    assert torch.equal(value, other)
                else:
                    assert value == other
        assert fast._ns_cuda_graph_replay.cache

        torch.set_float32_matmul_precision("medium")
        replay = MuonNewtonSchulzCudaGraphs(muon.newton_schulz_tp, max_graphs=1)
        kwargs = dict(steps=5, coefficient_type="quintic", tp_group=pg.tp)
        first = torch.randn(64, 32, device="cuda")
        wanted = muon.newton_schulz_tp(first, **kwargs)
        held = replay(first, **kwargs)
        assert torch.equal(held, wanted)
        second = torch.randn_like(first)
        assert torch.equal(replay(second, **kwargs), muon.newton_schulz_tp(second, **kwargs))
        assert torch.equal(held, wanted)
        overflow = torch.randn(32, 16, device="cuda")
        assert torch.equal(replay(overflow, **kwargs), muon.newton_schulz_tp(overflow, **kwargs))
        assert len(replay.cache) == 1
    finally:
        torch.set_float32_matmul_precision(previous)
        Utils.destroy_model_parallel()
