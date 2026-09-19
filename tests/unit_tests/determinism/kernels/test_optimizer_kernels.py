# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Bit-exact replay of the optimizer-side kernels: gradient-norm and clipping multi-tensor
kernels (TE / apex / local fallback in ``megatron/core/optimizer/clip_grads.py``) and the fused
Adam update. These run once per step on every parameter, so any drift here changes the
whole trajectory even when the model kernels are deterministic.
"""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.optimizer import Adam, OptimizerConfig
from megatron.core.optimizer.clip_grads import clip_grad_by_total_norm_fp32, get_grad_norm_fp32
from megatron.core.optimizer.optimizer import ChainedOptimizer, FP32Optimizer
from megatron.training.tensor_metrics.definitions import L2NormMetric, _fused_l2_norm_impl
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


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_tensor_metric_l2_replays(monkeypatch, dtype):
    """Replay TE's per-tensor norm outputs through the actual batched metric path."""
    if _fused_l2_norm_impl() is None:
        pytest.skip("TransformerEngine multi-tensor L2 kernel is unavailable")
    seeded()
    # Span many reduction CTAs and include uneven chunk boundaries and small tensors.
    tensors = tuple(
        torch.randn(n, device="cuda", dtype=dtype) for n in (4_194_321, 131_071, 4097, 1)
    )
    metric = L2NormMetric()

    def unexpected_fallback(tensor):
        pytest.fail("Expected the fused multi-tensor L2 path, not the per-tensor fallback")

    monkeypatch.setattr(metric, "contribution", unexpected_fallback)
    assert_replays_bit_exact(
        lambda *values: metric.contribution_batch(values),
        tensors,
        replays=8,
        backward=False,
        contention=True,
        what=f"L2NormMetric.contribution_batch ({dtype})",
    )


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


def test_chained_optimizer_cuda_graph():
    """Clipping across distinct process groups must capture and replay exactly."""
    Utils.initialize_model_parallel()
    extra_group = torch.distributed.new_group()
    graph = torch.cuda.CUDAGraph()
    try:
        config = OptimizerConfig(optimizer="sgd", lr=0.1, clip_grad=1.0, optimizer_cuda_graph=True)
        params = [torch.nn.Parameter(torch.zeros(1, device="cuda")) for _ in range(2)]
        optimizers = []
        for param, group in zip(params, [torch.distributed.group.WORLD, extra_group]):
            param.grad = torch.zeros_like(param)
            optimizer = FP32Optimizer(
                torch.optim.SGD([param], lr=config.lr), config, lambda _: None
            )
            optimizer.grad_stats_parallel_group = group
            optimizers.append(optimizer)
        optimizer = ChainedOptimizer(optimizers)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                for param, value in zip(params, [3.0, 4.0]):
                    param.grad.fill_(value)
                optimizer.step()
        torch.cuda.current_stream().wait_stream(stream)
        with torch.cuda.graph(graph, stream=stream):
            _, grad_norm, _ = optimizer.step()

        for values, local_norm in [((3.0, 4.0), 5.0), ((0.0, 0.0), 0.0)]:
            expected_norm = local_norm * torch.distributed.get_world_size() ** 0.5
            coefficient = min(1.0, config.clip_grad / (expected_norm + 1e-6))
            reference = None
            for _ in range(2):
                for param, value in zip(params, values):
                    param.data.zero_()
                    param.grad.fill_(value)
                graph.replay()
                torch.testing.assert_close(grad_norm, torch.full_like(grad_norm, expected_norm))
                for param, value in zip(params, values):
                    torch.testing.assert_close(
                        param, torch.full_like(param, -config.lr * value * coefficient)
                    )
                outputs = [*params, grad_norm]
                if reference is None:
                    reference = [output.detach().clone() for output in outputs]
                else:
                    for output, expected in zip(outputs, reference):
                        assert bytes_equal(output, expected), "Chained optimizer replay differs"
    finally:
        torch.cuda.synchronize()
        graph.reset()
        torch.distributed.destroy_process_group(extra_group)
        Utils.destroy_model_parallel()


@pytest.mark.parametrize("norms, expected", [([3.0, 4.0], 5.0), ([0.0, 0.0], 0.0)])
def test_chained_optimizer_float_norms(norms, expected):
    """Backends returning Python floats retain the same combined norm."""
    optimizers = [
        SimpleNamespace(
            get_grad_norm=lambda norm=norm: norm,
            get_grad_stats_parallel_group=lambda group=object(): group,
        )
        for norm in norms
    ]
    assert ChainedOptimizer(optimizers).get_grad_norm() == expected
