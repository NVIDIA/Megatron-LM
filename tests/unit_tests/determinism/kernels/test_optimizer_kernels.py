# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Bit-exact replay of the optimizer-side kernels: gradient-norm and clipping multi-tensor
kernels (TE / apex / local fallback in ``megatron/core/optimizer/clip_grads.py``) and the fused
Adam update. These run once per step on every parameter, so any drift here changes the
whole trajectory even when the model kernels are deterministic.

``ChainedOptimizer`` decides whether those statistics come from one combined reduction or from
one per-optimizer reduction, so its grad-stats group selection is part of the same kernel
contract and is covered here too.
"""

import math

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.optimizer import Adam, ChainedOptimizer, OptimizerConfig
from megatron.core.optimizer.clip_grads import clip_grad_by_total_norm_fp32, get_grad_norm_fp32
from tests.unit_tests.determinism.kernels.harness import (
    assert_replays_bit_exact,
    bytes_equal,
    seeded,
)
from tests.unit_tests.determinism.utils import RacingStreams
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _grads(num=256, seed=7):
    gen = torch.Generator().manual_seed(seed)
    sizes = torch.randint(1000, 400_000, (num,), generator=gen).tolist()
    return [torch.randn(n, device="cuda", dtype=torch.float32) for n in sizes]


class TestGradNormAndClip:
    def setup_method(self, method):
        Utils.initialize_model_parallel()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("norm_type", [2, float("inf")])
    def test_get_grad_norm_fp32_replays(self, norm_type):
        seeded()
        grads = _grads()

        def fn(*grads):
            return torch.tensor(
                get_grad_norm_fp32(list(grads), norm_type), device="cuda", dtype=torch.float64
            )

        assert_replays_bit_exact(
            fn, tuple(grads), replays=8, backward=False, contention=True, what="get_grad_norm_fp32"
        )

    def test_clip_grad_by_total_norm_replays(self):
        seeded()
        grads = _grads()
        params = []
        for g in grads:
            p = torch.zeros_like(g)
            p.grad = g.clone()
            params.append(p)
        total_norm = get_grad_norm_fp32([p.grad for p in params], 2)

        ref = None
        for i in range(4):
            for p, g in zip(params, grads):
                p.grad.copy_(g)
            with RacingStreams():
                clip_grad_by_total_norm_fp32(params, 1.0, total_norm)
            torch.cuda.synchronize()
            got = [p.grad.clone() for p in params]
            if ref is None:
                ref = got
                continue
            for j, (a, b) in enumerate(zip(ref, got)):
                assert bytes_equal(a, b), f"clipped grad {j} differs on replay {i}"


def test_fused_adam_step_replays():
    """Same params, grads and optimizer state -> identical updated params and moments."""
    seeded()
    shapes = [(4096, 4096), (16384, 2048), (2048,), (65536,)]
    params0 = [torch.randn(*s, device="cuda", dtype=torch.float32) for s in shapes]
    grads = [torch.randn(*s, device="cuda", dtype=torch.float32) for s in shapes]

    def run_step():
        params = [torch.nn.Parameter(p.clone()) for p in params0]
        for p, g in zip(params, grads):
            p.grad = g.clone()
        opt = Adam(params, lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1, eps=1e-8)
        with RacingStreams():
            opt.step()
            opt.step()
        torch.cuda.synchronize()
        out = [p.detach().clone() for p in params]
        for p in params:
            state = opt.state[p]
            out += [state[k].clone() for k in sorted(state) if torch.is_tensor(state[k])]
        return out

    ref = run_step()
    for i in range(1, 4):
        got = run_step()
        assert len(got) == len(ref)
        for j, (a, b) in enumerate(zip(ref, got)):
            assert bytes_equal(a, b), f"Adam tensor {j} differs on replay {i}"


class _ChainedGradStatsOptimizer:
    """Minimal ``MegatronOptimizer`` surface for ``ChainedOptimizer``'s grad-stat paths.

    ``get_grad_norm`` is the optimizer-local reduction that MFSDP v2's
    ``FullyShardedOptimizer`` implements over its own DTensor meshes; the mock reuses the real
    ``get_grad_norm_fp32`` multi-tensor kernel so the chained path is measured with the same
    accumulator and reduction as production.
    """

    is_stub_optimizer = False

    def __init__(self, grads, config, grad_stats_group, *, individual_grad_stats=False):
        self._grads = list(grads)
        self.config = config
        self._grad_stats_group = grad_stats_group
        self.grad_norm_calls = 0
        if individual_grad_stats:
            self.requires_individual_grad_stats = True

    def get_parameters(self):
        return []

    def get_grads_for_grad_norm(self, grad_norm_group=None):
        return list(self._grads)

    def get_grad_stats_parallel_group(self):
        return self._grad_stats_group

    def get_grad_norm(self):
        self.grad_norm_calls += 1
        return get_grad_norm_fp32(self._grads, grad_stats_parallel_group=self._grad_stats_group)


def _norm_grads(num, seed):
    gen = torch.Generator().manual_seed(seed)
    sizes = torch.randint(1000, 200_000, (num,), generator=gen).tolist()
    return [torch.randn(n, device="cuda", dtype=torch.float32) for n in sizes]


class TestChainedOptimizerGradStatsGroups:
    """Chained grad statistics stay per-optimizer when a chained optimizer requires it.

    The MFSDP v2 Muon + Adam chain wraps each part in a ``FullyShardedOptimizer`` whose
    ``requires_individual_grad_stats`` flag is set because its dense and expert DTensor shards
    can live on different device meshes. Group selection must therefore fall back to the
    per-optimizer path (``sqrt(sum(norm_i**2))``) even when every optimizer reports the same
    final-reduction process group; otherwise the combined path reduces concatenated grads from
    different meshes through one group.
    """

    def setup_method(self, method):
        Utils.initialize_model_parallel()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_shared_group_takes_the_combined_path(self):
        group = parallel_state.get_model_parallel_group()
        config = OptimizerConfig(optimizer="adam", lr=1e-3)
        first = _ChainedGradStatsOptimizer(_norm_grads(4, seed=1), config, group)
        second = _ChainedGradStatsOptimizer(_norm_grads(4, seed=2), config, group)
        chained = ChainedOptimizer([first, second])

        assert chained.grads_states_parallel_group_is_shared() is True
        assert chained.get_grad_stats_parallel_group() == group
        expected = get_grad_norm_fp32(
            first.get_grads_for_grad_norm() + second.get_grads_for_grad_norm(),
            grad_stats_parallel_group=group,
        )
        assert chained.get_grad_norm() == expected
        # The combined path never asks an individual optimizer for its own norm.
        assert first.grad_norm_calls == 0
        assert second.grad_norm_calls == 0

    def test_individual_grad_stats_forces_the_per_optimizer_path(self):
        group = parallel_state.get_model_parallel_group()
        config = OptimizerConfig(optimizer="adam", lr=1e-3)
        muon = _ChainedGradStatsOptimizer(
            _norm_grads(4, seed=3), config, group, individual_grad_stats=True
        )
        adam = _ChainedGradStatsOptimizer(_norm_grads(4, seed=4), config, group)
        chained = ChainedOptimizer([muon, adam])

        # Both report the same process group, yet the groups must not be treated as shared.
        assert chained.grads_states_parallel_group_is_shared() is False
        with pytest.raises(AssertionError, match="grads states parallel group"):
            chained.get_grad_stats_parallel_group()

        norms = [muon.get_grad_norm(), adam.get_grad_norm()]
        expected = math.sqrt(sum([norm**2 for norm in norms]))
        assert chained.get_grad_norm() == expected
        assert muon.grad_norm_calls == 2
        assert adam.grad_norm_calls == 2

    def test_different_groups_take_the_per_optimizer_path(self):
        config = OptimizerConfig(optimizer="adam", lr=1e-3)
        first = _ChainedGradStatsOptimizer(
            _norm_grads(4, seed=5), config, parallel_state.get_model_parallel_group()
        )
        second = _ChainedGradStatsOptimizer(
            _norm_grads(4, seed=6), config, parallel_state.get_data_parallel_group()
        )
        chained = ChainedOptimizer([first, second])

        assert chained.grads_states_parallel_group_is_shared() is False

    def test_individual_grad_stats_norm_replays_bit_exact(self):
        grads_a = _norm_grads(8, seed=7)
        grads_b = _norm_grads(8, seed=8)
        config = OptimizerConfig(optimizer="adam", lr=1e-3)
        group = parallel_state.get_model_parallel_group()
        split = len(grads_a)

        def fn(*grads):
            muon = _ChainedGradStatsOptimizer(
                list(grads[:split]), config, group, individual_grad_stats=True
            )
            adam = _ChainedGradStatsOptimizer(list(grads[split:]), config, group)
            chained = ChainedOptimizer([muon, adam])
            assert chained.grads_states_parallel_group_is_shared() is False
            return torch.tensor(chained.get_grad_norm(), device="cuda", dtype=torch.float64)

        assert_replays_bit_exact(
            fn,
            tuple(grads_a + grads_b),
            replays=6,
            backward=False,
            contention=True,
            what="chained_grad_norm_individual_grad_stats",
        )


def test_fully_sharded_optimizer_sets_the_individual_grad_stats_flag():
    """The branch keys on the flag the MFSDP v2 optimizer wrapper declares."""
    from megatron.core.optimizer.fully_sharded_optimizer import FullyShardedOptimizer

    assert FullyShardedOptimizer.requires_individual_grad_stats is True
