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


@pytest.mark.parametrize(
    "head_split_shapes", [(64,) * 6, (96, 32)], ids=["uniform-heads", "unequal-components"]
)
def test_muon_per_head_orthogonalize_replays(head_split_shapes):
    """Per-head split-QKV Muon: batched Newton-Schulz per head width, replayed under contention."""
    from megatron.core.optimizer import emerging_optimizers

    if not emerging_optimizers.HAVE_EMERGING_OPTIMIZERS:
        pytest.skip("requires emerging-optimizers")
    seeded()
    muon = emerging_optimizers.TensorParallelMuon(
        [torch.nn.Parameter(torch.zeros(1, device="cuda"))],
        split_qkv=True,
        split_qkv_per_head=True,
        is_qkv_fn=lambda p: True,
    )
    num_groups, cols = 8, 1024
    momentum = torch.randn(num_groups * sum(head_split_shapes), cols, device="cuda")
    param = torch.nn.Parameter(torch.empty_like(momentum))
    param.qkv_split_shapes = list(head_split_shapes)

    assert_replays_bit_exact(
        lambda grad: muon.orthogonalize(param, grad),
        (momentum,),
        replays=4,
        backward=False,
        contention=True,
        what=f"per-head Muon {head_split_shapes}",
    )


@pytest.mark.parametrize("recipe", ["blockwise", "mxfp8"])
def test_whole_param_requantize_replays(recipe):
    """LayerWise whole-param gathers requantize BF16 into FP8 params on every rank."""
    from megatron.core.fp8_utils import requantize_whole_params_from_high_precision
    from megatron.training.utils import get_device_arch_version

    try:
        import transformer_engine_torch as tex
        from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockQuantizer
        from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
    except ImportError:
        pytest.skip("requires Transformer Engine quantized tensors")
    arch = get_device_arch_version()
    if recipe == "blockwise" and arch != 9:
        pytest.skip("blockwise FP8 params are supported on Hopper")
    if recipe == "mxfp8" and arch < 10:
        pytest.skip("MXFP8 params need Blackwell")
    seeded()
    quantizer_cls = Float8BlockQuantizer if recipe == "blockwise" else MXFP8Quantizer
    shapes = [(1024, 512), (2048, 256), (384, 1024)]
    values = [torch.randn(*shape, device="cuda", dtype=torch.bfloat16) for shape in shapes]

    def requantize(*srcs):
        quantizer = quantizer_cls(tex.DType.kFloat8E4M3, rowwise=True, columnwise=True)
        params = [
            torch.nn.Parameter(
                quantizer.make_empty(src.shape, dtype=torch.bfloat16, device="cuda"),
                requires_grad=False,
            )
            for src in srcs
        ]
        requantize_whole_params_from_high_precision(params, list(srcs))
        return [
            getattr(param.data, name)
            for param in params
            for name in (
                "_rowwise_data",
                "_rowwise_scale_inv",
                "_columnwise_data",
                "_columnwise_scale_inv",
            )
            if getattr(param.data, name, None) is not None
        ]

    assert_replays_bit_exact(
        requantize,
        tuple(values),
        replays=4,
        backward=False,
        contention=True,
        what=f"{recipe} whole-param requantize",
    )


@pytest.mark.parametrize("recipe", ["blockwise", "mxfp8"])
def test_bf16_master_copy_replays(recipe):
    """LayerWise children on the compact layout write FP8 params that no gather
    publishes as Q(bf16(master)); other params take the multi-tensor copy."""
    from megatron.core.fp8_utils import requantize_whole_params_from_high_precision
    from megatron.core.optimizer.optimizer import Float16OptimizerWithFloat16Params
    from megatron.training.utils import get_device_arch_version

    try:
        import transformer_engine_torch as tex
        from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockQuantizer
        from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
    except ImportError:
        pytest.skip("requires Transformer Engine quantized tensors")
    arch = get_device_arch_version()
    if recipe == "blockwise" and arch != 9:
        pytest.skip("blockwise FP8 params are supported on Hopper")
    if recipe == "mxfp8" and arch < 10:
        pytest.skip("MXFP8 params need Blackwell")
    seeded()
    quantizer_cls = Float8BlockQuantizer if recipe == "blockwise" else MXFP8Quantizer
    shapes = [(1024, 512), (384, 1024), (2048, 256)]
    masters = tuple(torch.randn(*shape, device="cuda") for shape in shapes)

    def fp8_param(shape):
        quantizer = quantizer_cls(tex.DType.kFloat8E4M3, rowwise=True, columnwise=True)
        empty = quantizer.make_empty(shape, dtype=torch.bfloat16, device="cuda")
        return torch.nn.Parameter(empty, requires_grad=False)

    def fp8_data(param):
        names = ("_rowwise_data", "_rowwise_scale_inv", "_columnwise_data", "_columnwise_scale_inv")
        return [getattr(param.data, n) for n in names if getattr(param.data, n, None) is not None]

    def copy(quantized_master, gathered_master, plain_master):
        quantized, gathered = fp8_param(quantized_master.shape), fp8_param(gathered_master.shape)
        # The whole-param gather publishes this one; the copy leaves it alone.
        gathered.requantized_by_layer_wise_gather = True
        plain = torch.nn.Parameter(torch.empty_like(plain_master, dtype=torch.bfloat16))
        optimizer = Float16OptimizerWithFloat16Params.__new__(Float16OptimizerWithFloat16Params)
        optimizer.float16_groups = [[quantized, gathered, plain]]
        optimizer.fp32_from_float16_groups = [[quantized_master, gathered_master, plain_master]]
        optimizer._dummy_overflow_buf = torch.zeros(1, dtype=torch.int, device="cuda")
        optimizer._copy_main_params_to_model_params_via_bf16()
        return fp8_data(quantized) + [plain.data]

    got = copy(*masters)
    reference = fp8_param(shapes[0])
    requantize_whole_params_from_high_precision([reference], [masters[0].to(torch.bfloat16)])
    for a, b in zip(got, fp8_data(reference) + [masters[2].to(torch.bfloat16)]):
        assert bytes_equal(a, b), "the copy must write Q(bf16(master))"
    assert_replays_bit_exact(
        copy, masters, replays=4, backward=False, contention=True, what=f"{recipe} master copy"
    )
