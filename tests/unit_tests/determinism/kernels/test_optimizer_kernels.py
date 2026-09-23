# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Bit-exact replay of the optimizer-side kernels: gradient-norm and clipping multi-tensor
kernels (TE / apex / local fallback in ``megatron/core/optimizer/clip_grads.py``) and the fused
Adam update. These run once per step on every parameter, so any drift here changes the
whole trajectory even when the model kernels are deterministic.
"""

import pytest
import torch

from megatron.core.optimizer import Adam
from megatron.core.optimizer.clip_grads import clip_grad_by_total_norm_fp32, get_grad_norm_fp32
from megatron.training.tensor_metrics.definitions import L2NormMetric, _fused_l2_norm_impl
from megatron.training.utils.common_utils import _get_param_data
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


def test_row_owned_main_param_selection_replays():
    """Parameter norms consistently select the owner-local FP32 model shard."""
    model_param = torch.nn.Parameter(torch.zeros(16, device="cuda", dtype=torch.bfloat16))
    model_param.main_param = torch.full((16,), 7.0, device="cuda", dtype=torch.float32)
    model_param.main_param_model_shard = torch.arange(16, device="cuda", dtype=torch.float32)
    probe = torch.zeros(1, device="cuda")

    def select_owner_shard(value):
        selected, is_sharded = _get_param_data(model_param, force_create_fp32_copy=True, bf16=True)
        assert is_sharded and selected is model_param.main_param_model_shard
        return selected + value * 0

    assert_replays_bit_exact(
        select_owner_shard,
        (probe,),
        replays=4,
        backward=False,
        contention=True,
        what="row-owned optimizer model shard",
    )
