# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import importlib
import os

import pytest
import torch
import torch.distributed as dist

from megatron.core.extensions.transformer_engine import (
    TEColumnParallelLinear,
    TEDotProductAttention,
    TELayerNormColumnParallelLinear,
    TENorm,
    TERowParallelLinear,
    te_general_gemm,
)
from megatron.core.tensor_parallel.layers import ColumnParallelLinear
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
    assert_te_supports_batch_invariant_attention,
    get_batch_invariant_backend,
    matmul_persistent,
    mean_batch_invariant,
    set_batch_invariant_mode,
    te_supports_batch_invariant_attention,
)
from megatron.core.transformer.enums import AttnBackend, AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import init_method_normal, is_te_min_version
from tests.unit_tests.test_utilities import Utils

try:
    import flash_attn_3

    HAVE_FA3 = True
except ImportError:
    HAVE_FA3 = False

try:
    from flash_attn.cute import flash_attn_varlen_func as _fa4_varlen_func  # noqa: F401

    HAVE_FA4 = True
except ImportError:
    HAVE_FA4 = False

# Batch-invariant mode requires an explicit FlashAttention version.
_BIK_FA_VERSION = 4 if HAVE_FA4 else 3
requires_te_batch_invariant_attention = pytest.mark.skipif(
    not te_supports_batch_invariant_attention(),
    reason="Batch-invariant attention requires TransformerEngine PR #3204 or >= 2.18.",
)


@pytest.mark.parametrize("te_version", ("2.17.0+cb4a45fd", "2.18.0", "2.19.0.dev0"))
def test_batch_invariant_accepts_compatible_te(monkeypatch, te_version):
    import transformer_engine

    monkeypatch.setattr(transformer_engine, "__version__", te_version)
    assert_te_supports_batch_invariant_attention()


def test_batch_invariant_rejects_incompatible_te(monkeypatch):
    import transformer_engine

    monkeypatch.setattr(transformer_engine, "__version__", "2.17.0+deadbeef")
    with pytest.raises(AssertionError, match="TransformerEngine PR #3204"):
        assert_te_supports_batch_invariant_attention()


@pytest.mark.parametrize("dim", ([], (), None))
@pytest.mark.parametrize("keepdim", (False, True))
def test_mean_batch_invariant_full_reduction(dim, keepdim):
    """An empty (or None) dim list reduces over every dimension, as aten::mean does."""
    x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    out = mean_batch_invariant(x, dim, keepdim=keepdim)
    torch.testing.assert_close(out, x.mean(dim=(0, 1, 2), keepdim=keepdim))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_full_mean_under_batch_invariant_mode():
    """`x.mean()` / `torch.mean(x)` dispatch to aten::mean.dim with dim=[]."""
    x = torch.randn(4, 8, device="cuda", requires_grad=True)
    ref = x.detach().sum() / x.numel()
    with set_batch_invariant_mode(True):
        out = x.mean()
        out_fn = torch.mean(x)
        out.backward()
    torch.testing.assert_close(out.detach(), ref)
    torch.testing.assert_close(out_fn.detach(), ref)
    torch.testing.assert_close(x.grad, torch.full_like(x, 1.0 / x.numel()))


# ============================================================================
# Batch-Invariant test helpers
# ============================================================================
def _split_concat_equal(layer, x_full, dim=0, forward_kwargs=None, out_dim_concat=0):
    forward_kwargs = forward_kwargs or {}
    b = x_full.shape[dim]
    b1 = max(1, b // 4)
    b2 = b - b1
    xs = [torch.narrow(x_full, dim, 0, b1), torch.narrow(x_full, dim, b1, b2)]
    with torch.no_grad():
        with set_batch_invariant_mode(True):
            out_full = layer(x_full, **forward_kwargs)
            out1 = layer(xs[0], **forward_kwargs)
            out2 = layer(xs[1], **forward_kwargs)
    # Handle (out, bias) tuples from linear wrappers
    if isinstance(out_full, tuple):
        out_full = out_full[0]
        out1 = out1[0]
        out2 = out2[0]
    out_cat = torch.cat([out1, out2], dim=out_dim_concat)
    assert out_full.shape == out_cat.shape
    assert torch.equal(out_full, out_cat)


def _split_many_concat_equal(layer, x_full, splits, dim=0, forward_kwargs=None, out_dim_concat=0):
    forward_kwargs = forward_kwargs or {}
    assert sum(splits) == x_full.shape[dim], "Splits must sum to batch size"
    # Make contiguous chunks to avoid unexpected view behavior
    starts = [0]
    for s in splits[:-1]:
        starts.append(starts[-1] + s)
    xs = [torch.narrow(x_full, dim, st, ln).contiguous() for st, ln in zip(starts, splits)]
    with torch.no_grad():
        with set_batch_invariant_mode(True):
            out_full = layer(x_full, **forward_kwargs)
            outs = [layer(xi, **forward_kwargs) for xi in xs]
    if isinstance(out_full, tuple):
        out_full = out_full[0]
        outs = [o[0] for o in outs]
    out_cat = torch.cat(outs, dim=out_dim_concat)
    assert out_full.shape == out_cat.shape
    assert torch.equal(out_full, out_cat)


def _random_splits(total, num_parts):
    assert num_parts >= 2 and total >= num_parts
    cuts = torch.randperm(total - 1, device="cpu")[: num_parts - 1].tolist()
    cuts = [0] + sorted(cuts) + [total - 1]
    lens = [cuts[i + 1] - cuts[i] + 1 for i in range(len(cuts) - 1)]
    delta = sum(lens) - total
    i = 0
    while delta > 0:
        if lens[i] > 1:
            lens[i] -= 1
            delta -= 1
        i = (i + 1) % len(lens)
    return lens


_FP32, _BF16 = torch.float32, torch.bfloat16
# name: (A shape, B shape, (out shape, out dtype), calls). test_te_linear_parity covers TE's
# linears, whose weight gradients add into main_grad; these are TE's other general_gemm calls.
_GEMM_CASES = {
    # A weight gradient into fp32 main_grad with both operands [s, b, ...], as the op fuser's
    # BasicLinear passes them, with alpha and beta None.
    "wgrad-3d": (
        (24, 2, 128),
        (24, 2, 96),
        ((96, 128), _FP32),
        [dict(accumulate=True, layout="NT", grad=True, alpha=None, beta=None)],
    ),
    # A tracked first microbatch overwrites main_grad, and the next one adds into it.
    "overwrite-then-accumulate": (
        (64, 128),
        (64, 96),
        ((96, 128), _FP32),
        [dict(layout="NT", grad=True), dict(accumulate=True, layout="NT", grad=True)],
    ),
    # A scaled linear plus bias, added into a scaled bf16 residual.
    "alpha-beta-bias": (
        (96, 128),
        (64, 128),
        ((64, 96), _BF16),
        [dict(accumulate=True, alpha=0.5, beta=2.0)],
    ),
    "alpha": ((96, 128), (64, 128), ((64, 96), _BF16), [dict(alpha=0.25)]),
}


@pytest.mark.parametrize("case", list(_GEMM_CASES))
def test_bik_te_general_gemm_honors_accumulate_alpha_beta(case):
    """The batch-invariant general_gemm computes TE's alpha * op(A) @ op(B) + bias + beta * out,
    adding into out instead of overwriting it. Into fp32 it adds exactly the bf16 product that
    the same call without accumulate returns."""
    import transformer_engine.pytorch.cpp_extensions as te_cpp

    torch.manual_seed(5)
    a_shape, b_shape, (out_shape, out_dtype), calls = _GEMM_CASES[case]
    bf16 = _device(torch.bfloat16)
    operands = [(torch.randn(*a_shape, **bf16), torch.randn(*b_shape, **bf16)) for _ in calls]
    out0 = torch.randn(out_shape, device="cuda").to(out_dtype)
    bias = torch.randn(96, **bf16) if case == "alpha-beta-bias" else None

    def run():
        out = out0.clone()
        for (A, B), kwargs in zip(operands, calls):
            gemm = te_cpp.general_gemm(A, B, out_dtype=out_dtype, out=out, bias=bias, **kwargs)
            assert gemm[0] is out
        return out

    reference = run()
    with set_batch_invariant_mode(True, backend="triton"):
        actual = run()
        products = [
            te_cpp.general_gemm(A, B, out_dtype=_BF16, layout=kwargs.get("layout", "TN"))[0]
            for (A, B), kwargs in zip(operands, calls)
        ]
    # Within a few bf16 ulps of the largest value: the batch-invariant product is rounded to
    # bf16 before it is scaled and added, TE's is not.
    scale = reference.float().abs().max().item()
    torch.testing.assert_close(actual.float(), reference.float(), rtol=0, atol=2**-6 * scale)
    if out_dtype == _FP32:
        expected = out0
        for product, kwargs in zip(products, calls):
            kept = expected if kwargs.get("accumulate") else 0.0
            expected = kept + product.float().reshape(out_shape)
        assert torch.equal(actual, expected)


@pytest.mark.parametrize("transformer_impl", ["transformer_engine", "inference_optimized"])
def test_triton_backend_rejects_fp8_te_gemms(transformer_impl):
    """The triton backend's dense GEMM takes no quantized operands, so TransformerConfig rejects
    FP8 for TE layers. Inference-optimized layers run their own GEMMs in inference, so their FP8
    is accepted. (TE layers reject a per-module quant_recipe precision when they resolve it.)"""

    def build():
        TransformerConfig(
            num_layers=1,
            hidden_size=64,
            num_attention_heads=4,
            bf16=True,
            params_dtype=torch.bfloat16,
            attention_backend=AttnBackend.flash,
            flash_attention_version=_BIK_FA_VERSION,
            attention_dropout=0.0,
            batch_invariant_mode=True,
            batch_invariant_backend="triton",
            transformer_impl=transformer_impl,
            add_bias_linear=transformer_impl != "inference_optimized",
            normalization="RMSNorm",
            fp8="hybrid",
            fp8_recipe="mxfp8",
        )

    if transformer_impl == "transformer_engine":
        with pytest.raises(AssertionError, match="batch_invariant_backend='triton'"):
            build()
    else:
        build()


# ============================================================================
# Randomized Batch Invariant Tests
# ============================================================================


def test_te_column_parallel_linear_batch_invariant_randomized():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    Utils.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(123)

    cfg = TransformerConfig(
        num_layers=1,
        hidden_size=256,
        num_attention_heads=8,
        use_cpu_initialization=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        batch_invariant_mode=True,
        flash_attention_version=_BIK_FA_VERSION,
        params_dtype=torch.bfloat16,
        normalization="RMSNorm",
        layernorm_epsilon=1e-5,
        attention_backend=AttnBackend.flash,
    )

    layer = (
        TEColumnParallelLinear(
            input_size=cfg.hidden_size,
            output_size=512,
            config=cfg,
            init_method=init_method_normal(cfg.init_method_std),
            gather_output=False,
            bias=True,
            skip_bias_add=False,
            is_expert=False,
        )
        .cuda()
        .eval()
    )

    torch.manual_seed(123)
    for _ in range(3):
        B = int(torch.randint(48, 129, (1,)).item())
        parts = int(torch.randint(4, 9, (1,)).item())
        splits = _random_splits(B, parts)
        x = torch.randn(B, cfg.hidden_size, device="cuda", dtype=torch.bfloat16)
        _split_many_concat_equal(layer, x, splits=splits, dim=0, out_dim_concat=0)

    Utils.destroy_model_parallel()


def test_te_row_parallel_linear_batch_invariant_randomized():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    Utils.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(123)

    cfg = TransformerConfig(
        num_layers=1,
        hidden_size=256,
        num_attention_heads=8,
        use_cpu_initialization=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        batch_invariant_mode=True,
        flash_attention_version=_BIK_FA_VERSION,
        params_dtype=torch.bfloat16,
        normalization="RMSNorm",
        layernorm_epsilon=1e-5,
        attention_backend=AttnBackend.flash,
    )

    layer = (
        TERowParallelLinear(
            input_size=cfg.hidden_size,
            output_size=384,
            config=cfg,
            init_method=init_method_normal(cfg.init_method_std),
            bias=True,
            input_is_parallel=True,
            skip_bias_add=False,
            is_expert=False,
        )
        .cuda()
        .eval()
    )

    torch.manual_seed(321)
    for _ in range(3):
        B = int(torch.randint(48, 129, (1,)).item())
        parts = int(torch.randint(4, 9, (1,)).item())
        splits = _random_splits(B, parts)
        x = torch.randn(B, cfg.hidden_size, device="cuda", dtype=torch.bfloat16)
        _split_many_concat_equal(layer, x, splits=splits, dim=0, out_dim_concat=0)

    Utils.destroy_model_parallel()


def test_te_layernorm_column_parallel_linear_batch_invariant_randomized():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    Utils.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(123)

    cfg = TransformerConfig(
        num_layers=1,
        hidden_size=256,
        num_attention_heads=8,
        use_cpu_initialization=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        batch_invariant_mode=True,
        flash_attention_version=_BIK_FA_VERSION,
        params_dtype=torch.bfloat16,
        normalization="RMSNorm",
        layernorm_epsilon=1e-5,
        attention_backend=AttnBackend.flash,
    )

    layer = (
        TELayerNormColumnParallelLinear(
            input_size=cfg.hidden_size,
            output_size=512,
            config=cfg,
            init_method=init_method_normal(cfg.init_method_std),
            gather_output=False,
            bias=True,
            skip_bias_add=False,
            is_expert=False,
        )
        .cuda()
        .eval()
    )

    torch.manual_seed(456)
    for _ in range(3):
        B = int(torch.randint(48, 129, (1,)).item())
        parts = int(torch.randint(4, 9, (1,)).item())
        splits = _random_splits(B, parts)
        x = torch.randn(B, cfg.hidden_size, device="cuda", dtype=torch.bfloat16)
        _split_many_concat_equal(layer, x, splits=splits, dim=0, out_dim_concat=0)

    Utils.destroy_model_parallel()


def test_te_norm_batch_invariant_randomized():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    Utils.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(123)

    cfg = TransformerConfig(
        num_layers=1,
        hidden_size=256,
        num_attention_heads=8,
        use_cpu_initialization=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        batch_invariant_mode=True,
        flash_attention_version=_BIK_FA_VERSION,
        params_dtype=torch.bfloat16,
        normalization="RMSNorm",
        layernorm_epsilon=1e-5,
        attention_backend=AttnBackend.flash,
    )

    layer = TENorm(config=cfg, hidden_size=cfg.hidden_size, eps=cfg.layernorm_epsilon).cuda().eval()

    torch.manual_seed(789)
    for _ in range(3):
        B = int(torch.randint(48, 129, (1,)).item())
        parts = int(torch.randint(4, 9, (1,)).item())
        splits = _random_splits(B, parts)
        x = torch.randn(B, cfg.hidden_size, device="cuda", dtype=torch.bfloat16)
        _split_many_concat_equal(layer, x, splits=splits, dim=0, out_dim_concat=0)

    Utils.destroy_model_parallel()


def test_column_parallel_linear_batch_invariant_randomized():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    Utils.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(123)

    cfg = TransformerConfig(
        num_layers=1,
        hidden_size=256,
        num_attention_heads=8,
        use_cpu_initialization=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        batch_invariant_mode=True,
        flash_attention_version=_BIK_FA_VERSION,
        params_dtype=torch.bfloat16,
        normalization="RMSNorm",
        layernorm_epsilon=1e-5,
        attention_backend=AttnBackend.flash,
    )

    layer = (
        ColumnParallelLinear(
            input_size=cfg.hidden_size,
            output_size=320,
            config=cfg,
            init_method=init_method_normal(cfg.init_method_std),
            gather_output=False,
            bias=True,
            skip_bias_add=False,
            is_expert=False,
        )
        .cuda()
        .eval()
    )

    torch.manual_seed(246)
    for _ in range(3):
        B = int(torch.randint(48, 129, (1,)).item())
        parts = int(torch.randint(4, 9, (1,)).item())
        splits = _random_splits(B, parts)
        x = torch.randn(B, cfg.hidden_size, device="cuda", dtype=torch.bfloat16)
        _split_many_concat_equal(layer, x, splits=splits, dim=0, out_dim_concat=0)

    Utils.destroy_model_parallel()


@pytest.mark.skipif(
    not (is_te_min_version("2.10.0") and HAVE_FA3),
    reason="TE attention BIK tests require TE >= 2.10.0 and FlashAttention-3",
)
@requires_te_batch_invariant_attention
def test_te_attention_layer_batch_invariant_randomized():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    Utils.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(123)
    os.environ["NVTE_FUSED_ATTN"] = "0"
    os.environ["NVTE_FLASH_ATTN"] = "1"
    os.environ["NVTE_UNFUSED_ATTN"] = "0"

    cfg = TransformerConfig(
        num_layers=1,
        hidden_size=256,
        num_attention_heads=8,
        use_cpu_initialization=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        batch_invariant_mode=True,
        flash_attention_version=_BIK_FA_VERSION,
        params_dtype=torch.bfloat16,
        normalization="RMSNorm",
        layernorm_epsilon=1e-5,
        attention_backend=AttnBackend.flash,
    )

    attn = TEDotProductAttention(
        config=cfg, layer_number=1, attn_mask_type=AttnMaskType.causal, attention_type="self"
    )
    assert getattr(attn, "num_splits", None) == 1

    torch.manual_seed(135)
    for _ in range(3):
        B = int(torch.randint(32, 97, (1,)).item())
        parts = int(torch.randint(4, 9, (1,)).item())
        splits = _random_splits(B, parts)
        S = int(torch.randint(256, 513, (1,)).item())
        H = cfg.num_attention_heads
        D = cfg.hidden_size // H

        q = torch.randn(S, B, H, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(S, B, H, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(S, B, H, D, device="cuda", dtype=torch.bfloat16)

        # Random permutation of the batch dimension.
        perm = torch.randperm(B, device="cuda")

        # Also build contiguous chunks to continue testing split invariance.
        starts = [0]
        for s in splits[:-1]:
            starts.append(starts[-1] + s)
        q_chunks = [q[:, st : st + ln] for st, ln in zip(starts, splits)]
        k_chunks = [k[:, st : st + ln] for st, ln in zip(starts, splits)]
        v_chunks = [v[:, st : st + ln] for st, ln in zip(starts, splits)]

        with torch.no_grad():
            with set_batch_invariant_mode(True):
                # Full batch
                out_full = attn(q, k, v, attention_mask=None, attn_mask_type=AttnMaskType.causal)
                # Chunked batches (batch-split invariance)
                outs = [
                    attn(qc, kc, vc, attention_mask=None, attn_mask_type=AttnMaskType.causal)
                    for qc, kc, vc in zip(q_chunks, k_chunks, v_chunks)
                ]
                out_cat = torch.cat(outs, dim=1)

                # Permuted batch (ordering invariance): permute B, run attention,
                # then undo the permutation on the output batch dimension.
                out_perm = attn(
                    q[:, perm],
                    k[:, perm],
                    v[:, perm],
                    attention_mask=None,
                    attn_mask_type=AttnMaskType.causal,
                )

        assert out_full.shape == out_cat.shape == out_perm.shape

        # Batch-split invariance: processing different contiguous chunks should
        # produce exactly the same result as processing the full batch.
        assert torch.equal(out_full, out_cat)

        # Batch-order invariance: reordering the batch and then undoing the
        # permutation on the output should give back the same tensor.
        out_perm_unpermed = out_perm[:, perm.argsort()]
        assert torch.equal(out_full, out_perm_unpermed)

    Utils.destroy_model_parallel()


# ============================================================================
# Parity Tests: Batch-Invariant vs Regular TE Layers
# ============================================================================


@pytest.mark.parametrize("fused_wgrad", [False, True], ids=["grad", "main-grad"])
@pytest.mark.parametrize("input_shape", [(64,), (32, 1), (16, 4)], ids=["2d", "sbh-b1", "sbh-b4"])
@pytest.mark.parametrize(
    "layer_cls",
    [TEColumnParallelLinear, TELayerNormColumnParallelLinear],
    ids=["linear", "layernorm-linear"],
)
def test_te_linear_parity(layer_cls, input_shape, fused_wgrad):
    """Batch-invariant and regular TE linears give the same outputs and gradients over two
    microbatches, for 2D and [s, b, h] inputs. With gradient_accumulation_fusion, main_grad
    gains each microbatch's bf16 weight gradient instead of keeping only the last one."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    Utils.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(123)

    def build(batch_invariant):
        cfg = TransformerConfig(
            num_layers=1,
            hidden_size=128,
            num_attention_heads=4,
            use_cpu_initialization=True,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            batch_invariant_mode=batch_invariant,
            params_dtype=torch.bfloat16,
            normalization="RMSNorm",
            layernorm_epsilon=1e-5,
            attention_backend=AttnBackend.flash,
            # The regular layer's per-microbatch weight gradients are the reference.
            gradient_accumulation_fusion=batch_invariant and fused_wgrad,
            **({"flash_attention_version": _BIK_FA_VERSION} if batch_invariant else {}),
        )
        torch.manual_seed(456)  # Same seed for same initialization
        layer = layer_cls(
            input_size=cfg.hidden_size,
            output_size=256,
            config=cfg,
            init_method=init_method_normal(cfg.init_method_std),
            gather_output=False,
            bias=True,
            skip_bias_add=False,
            is_expert=False,
        ).cuda()
        if cfg.gradient_accumulation_fusion:
            layer.weight.main_grad = torch.zeros(
                layer.weight.shape, device="cuda", dtype=torch.float32
            )
        return layer

    try:
        layer_bik, layer_regular = build(True), build(False)
        main_grad = torch.zeros(layer_regular.weight.shape, device="cuda", dtype=torch.float32)
        torch.manual_seed(789)
        for _ in range(2):
            x = torch.randn(*input_shape, 128, device="cuda", dtype=torch.bfloat16)
            grad_output = torch.randn(*input_shape, 256, device="cuda", dtype=torch.bfloat16)
            results = []
            for layer, batch_invariant in ((layer_bik, True), (layer_regular, False)):
                layer.weight.grad = None
                x_in = x.clone().requires_grad_(True)
                with set_batch_invariant_mode(batch_invariant):
                    out, _ = layer(x_in)
                    out.backward(grad_output)
                results.append((out.detach(), x_in.grad, layer.weight.grad))
            (out_bik, dx_bik, dw_bik), (out_regular, dx_regular, dw_regular) = results
            assert out_bik.shape == out_regular.shape
            pairs = [
                ("Forward output", out_bik, out_regular),
                ("Input gradient", dx_bik, dx_regular),
            ]
            if fused_wgrad:
                main_grad += dw_regular.float()
            else:
                pairs.append(("Weight gradient", dw_bik, dw_regular))
            # bf16 tolerances: the batch-invariant RMSNorm can round an ln_out element one
            # ulp away from TE's, which moves a layernorm-linear output by an ulp.
            for label, actual, expected in pairs:
                torch.testing.assert_close(
                    actual,
                    expected,
                    **_tols(torch.bfloat16),
                    msg=lambda m, label=label: f"{label}: {m}",
                )
        if fused_wgrad:
            torch.testing.assert_close(
                layer_bik.weight.main_grad,
                main_grad,
                **_tols(torch.bfloat16),
                msg=lambda m: f"main_grad: {m}",
            )
    finally:
        Utils.destroy_model_parallel()


def test_te_rmsnorm_parity():
    """Test that batch-invariant and regular TE RMSNorm produce same forward/backward results."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    Utils.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(123)

    cfg_bik = TransformerConfig(
        num_layers=1,
        hidden_size=128,
        num_attention_heads=4,
        use_cpu_initialization=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        batch_invariant_mode=True,
        flash_attention_version=_BIK_FA_VERSION,
        params_dtype=torch.bfloat16,
        normalization="RMSNorm",
        layernorm_epsilon=1e-5,
        attention_backend=AttnBackend.flash,
    )

    cfg_regular = TransformerConfig(
        num_layers=1,
        hidden_size=128,
        num_attention_heads=4,
        use_cpu_initialization=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        batch_invariant_mode=False,
        params_dtype=torch.bfloat16,
        normalization="RMSNorm",
        layernorm_epsilon=1e-5,
        attention_backend=AttnBackend.flash,
    )

    # Create layers with same weights
    torch.manual_seed(789)
    layer_bik = TENorm(
        config=cfg_bik, hidden_size=cfg_bik.hidden_size, eps=cfg_bik.layernorm_epsilon
    ).cuda()

    torch.manual_seed(789)
    layer_regular = TENorm(
        config=cfg_regular, hidden_size=cfg_regular.hidden_size, eps=cfg_regular.layernorm_epsilon
    ).cuda()

    # Test forward pass
    x = torch.randn(
        48, cfg_bik.hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    x_clone = x.clone().detach().requires_grad_(True)
    with set_batch_invariant_mode(False):
        out_regular = layer_regular(x_clone)

    with set_batch_invariant_mode(True):
        out_bik = layer_bik(x)

    # Check forward outputs are close
    assert out_bik.shape == out_regular.shape
    assert out_bik.dtype == out_regular.dtype
    max_diff = (out_bik - out_regular).abs().max().item()
    assert max_diff < 1e-3, f"Forward output difference too large: {max_diff}"

    # Test backward pass
    grad_output = torch.randn_like(out_bik)

    out_bik.backward(grad_output)
    out_regular.backward(grad_output.clone())

    # Check gradients are close
    grad_diff = (x.grad - x_clone.grad).abs().max().item()
    assert grad_diff < 1e-3, f"Input gradient difference too large: {grad_diff}"

    weight_grad_diff = (layer_bik.weight.grad - layer_regular.weight.grad).abs().max().item()
    assert weight_grad_diff < 1e-3, f"Weight gradient difference too large: {weight_grad_diff}"

    Utils.destroy_model_parallel()


# Some tolerance for numerical differences between cuBLASLt and Triton
def _tols(dtype: torch.dtype):
    if dtype == torch.float16:
        return dict(rtol=1e-2, atol=1e-2)
    if dtype == torch.bfloat16:
        return dict(rtol=2e-2, atol=2e-2)
    return dict(rtol=1e-2, atol=5e-2)


def _device(dtype=torch.float16):
    return dict(device="cuda", dtype=dtype)


# Helper to call TE general_gemm via Megatron's wrapper that manages workspace, etc.
def _te_general_gemm(*args, **kwargs):
    if te_general_gemm is None:
        pytest.skip("TransformerEngine general_gemm is not available in this environment.")
    return te_general_gemm(*args, **kwargs)


@pytest.mark.parametrize("supported", [False, True])
def test_te_native_device_metadata_workspace(monkeypatch, supported):
    """Restrict the full-workspace exception and keep graph layouts isolated."""
    import transformer_engine.pytorch.cpp_extensions.gemm as te_gemm

    if not hasattr(te_gemm, "_get_grouped_cublas_workspace"):
        # Exercise the compatibility hook even with TE releases predating the
        # device-metadata API used in production.
        def _mock_grouped_workspace(device, layout):
            del layout
            return torch.empty(
                te_gemm.get_cublas_workspace_size_bytes(), dtype=torch.uint8, device=device
            )

        monkeypatch.setattr(
            te_gemm, "_get_grouped_cublas_workspace", _mock_grouped_workspace, raising=False
        )

    monkeypatch.setattr(
        "megatron.core.transformer.custom_layers.batch_invariant_kernels."
        "te_supports_batch_invariant_grouped_gemm",
        lambda device: supported,
    )
    unrestricted_bytes = te_gemm.get_cublas_workspace_size_bytes()
    original_grouped_workspace = te_gemm._get_grouped_cublas_workspace
    with set_batch_invariant_mode(True, backend="te_native"):
        assert te_gemm.get_cublas_workspace_size_bytes() == 1024
        device = torch.cuda.current_device()
        if supported:
            workspaces = [
                te_gemm._get_grouped_cublas_workspace(device, layout)
                for layout in ("TN", "NN", "NT")
            ]
            assert all(ws.numel() == unrestricted_bytes for ws in workspaces)
            assert len({ws.data_ptr() for ws in workspaces}) == 3
            assert te_gemm._get_grouped_cublas_workspace(device, "TN") is workspaces[0]
        else:
            with pytest.raises(RuntimeError, match="moe_use_grouped_tensor=False"):
                te_gemm._get_grouped_cublas_workspace(device, "TN")

    assert te_gemm._get_grouped_cublas_workspace is original_grouped_workspace
    assert te_gemm.get_cublas_workspace_size_bytes() == unrestricted_bytes


# ============================================================================
# Numerical Tests for General GEMM
# ============================================================================


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_bik_te_general_gemm_chunking_deterministic(dtype):
    torch.manual_seed(123)
    M1, M2, K, N = 37, 23, 128, 128
    A1 = torch.randn(M1, K, **_device(dtype))
    A2 = torch.randn(M2, K, **_device(dtype))
    A = torch.cat([A1, A2], dim=0)
    B = torch.randn(K, N, **_device(dtype))

    layout = "TN"
    with set_batch_invariant_mode(True):
        # Full batch
        C_full = _te_general_gemm(A, B, out_dtype=dtype, layout=layout)[0]
        # Chunked batches
        C_part1 = _te_general_gemm(A1, B, out_dtype=dtype, layout=layout)[0]
        C_part2 = _te_general_gemm(A2, B, out_dtype=dtype, layout=layout)[0]
        # For TN, output is [N, M]; concatenation should be along dim=1
        cat_dim = 1 if layout == "TN" else 0
        C_cat = torch.cat([C_part1, C_part2], dim=cat_dim)

    # For TN, shapes are [N, M]
    assert C_full.shape == (N, M1 + M2)
    assert C_cat.shape == (N, M1 + M2)
    # Exact equality expected due to deterministic Triton kernel
    assert torch.equal(C_full, C_cat)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_bik_te_general_gemm_numerical_parity(dtype):
    torch.manual_seed(111)
    M, K, N = 64, 96, 96
    A = torch.randn(M, K, **_device(dtype))
    B = torch.randn(K, N, **_device(dtype))

    C_ref = _te_general_gemm(A, B, out_dtype=dtype, layout="TN")[0]

    # Batch-invariant inside context
    with set_batch_invariant_mode(True):
        C_bik = _te_general_gemm(A, B, out_dtype=dtype, layout="TN")[0]

    torch.testing.assert_close(C_bik, C_ref, **_tols(dtype))


@pytest.mark.parametrize("rows", [1, 4, 8, 32, 64, 96, 127, 128, 129, 160, 192, 224, 255, 256, 257])
def test_bik_te_general_gemm_triton_strided_weight_is_bit_exact(rows):
    """A partial M tile and strided weight preserve bits across batch sizes."""
    torch.manual_seed(193)
    weight = torch.randn(256, 128, **_device(torch.bfloat16))
    full_inputs = torch.randn(384, 128, **_device(torch.bfloat16))
    inputs = full_inputs[:rows].contiguous()
    pointer = weight.untyped_storage().data_ptr()
    with set_batch_invariant_mode(True, backend="triton"):
        expected = matmul_persistent(inputs, weight.transpose(0, 1).contiguous())
        actual = _te_general_gemm(weight, inputs, out_dtype=torch.bfloat16, layout="TN")[0]
        full_actual = _te_general_gemm(weight, full_inputs, out_dtype=torch.bfloat16, layout="TN")[
            0
        ]
    assert torch.equal(actual, expected)
    assert torch.equal(actual, full_actual[:rows])
    assert weight.untyped_storage().data_ptr() == pointer
    assert weight.stride(1) == 1


def test_deepgemm_backend_is_deprecated_to_te_native():
    """The removed DeepGEMM dense backend still enables, as te_native, with a warning."""
    with pytest.warns(DeprecationWarning, match="deepgemm"):
        with set_batch_invariant_mode(True, backend="deepgemm"):
            assert get_batch_invariant_backend() == "te_native"
