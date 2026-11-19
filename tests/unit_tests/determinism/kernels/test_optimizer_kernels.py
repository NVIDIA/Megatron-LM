# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Bit-exact replay of the optimizer-side kernels: gradient-norm and clipping multi-tensor
kernels (TE / apex / local fallback in ``megatron/core/optimizer/clip_grads.py``) and the fused
Adam update. These run once per step on every parameter, so any drift here changes the
whole trajectory even when the model kernels are deterministic.
"""

import pytest
import torch

from megatron.core.ops.optimizers.sinkhorn import balance_row_shards
from megatron.core.optimizer import Adam
from megatron.core.optimizer.clip_grads import clip_grad_by_total_norm_fp32, get_grad_norm_fp32
from megatron.core.optimizer.emerging_optimizers import TensorParallelMuon
from megatron.core.optimizer.sinkhorn import SinkhornOptimizer
from megatron.core.tensor_parallel.layers import copy_tensor_model_parallel_attributes
from megatron.core.transformer.attention import QKVLayout
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


def test_sinkhorn_and_whole_head_updates_replay():
    """Replay the actual momentum/NS/balancing paths and copied master metadata."""
    from emerging_optimizers.orthogonalized_optimizers.sinkhorn_utils import sinkhorn_balance

    seeded()
    grad = torch.randn(2048, 64, device="cuda")
    # Include a locally wide shard so the distributed-statistics adapter executes
    # even without a multi-rank communicator, plus the upstream complete-table path.
    for rows in (32, 2048):
        assert_replays_bit_exact(
            lambda x: balance_row_shards(
                x, global_rows=2048, group=None, upstream_balance=sinkhorn_balance, local=True
            ),
            (grad[:rows],),
            replays=3,
            backward=False,
            contention=True,
            what="Sinkhorn axes",
        )
    source = torch.nn.Parameter(torch.zeros_like(grad))
    source.qkv_layout = QKVLayout.from_mla_head_components(8, (256,))
    source.is_qkv = True
    source.qkv_split_shapes = [256] * 8
    source.qkv_split_heads_are_complete = True
    source.sinkhorn_table_ranges = ((0, 1024), (1024, 2048))
    source.sinkhorn_row_start = 0
    source.sinkhorn_group = None

    def update():
        p = torch.nn.Parameter(torch.zeros_like(grad))
        copy_tensor_model_parallel_attributes(p, source)
        muon = TensorParallelMuon(
            [p],
            split_qkv=True,
            split_qkv_per_head=True,
            is_qkv_fn=lambda p: p.is_qkv,
            update_rms=0.18,
        )
        orth = muon.orthogonalize(p, grad)
        opt = SinkhornOptimizer([p])
        p.grad = grad.clone()
        opt.step()
        return orth, p.detach().clone(), opt.state[p]["momentum_buffer"].clone()

    reference = update()
    for _ in range(3):
        with RacingStreams():
            actual = update()
        torch.cuda.synchronize()
        assert all(bytes_equal(a, b) for a, b in zip(reference, actual))


def test_fused_adam_step_replays():
    """Mixed weight-decay groups produce identical updated params and moments on replay."""
    seeded()
    shapes = [(4096, 4096), (16384, 2048), (2048,), (65536,)]
    params0 = [torch.randn(*s, device="cuda", dtype=torch.float32) for s in shapes]
    grads = [torch.randn(*s, device="cuda", dtype=torch.float32) for s in shapes]

    def run_step():
        params = [torch.nn.Parameter(p.clone()) for p in params0]
        for p, g in zip(params, grads):
            p.grad = g.clone()
        # Retention routing places a 1-D controller in a decayed group while ordinary
        # vector parameters remain in a zero-WD group. Exercise both kernel paths together.
        opt = Adam(
            [
                {"params": params[:3], "weight_decay": 0.1},
                {"params": params[3:], "weight_decay": 0.0},
            ],
            lr=1e-3,
            betas=(0.9, 0.95),
            eps=1e-8,
        )
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
