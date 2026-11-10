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
    HAVE_DEEPGEMM_BF16,
    assert_te_supports_batch_invariant_attention,
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dim", ([], (), None))
@pytest.mark.parametrize("keepdim", (False, True))
def test_mean_batch_invariant_full_reduction(dim, keepdim):
    """An empty (or None) dim list reduces over every dimension, as aten::mean does."""
    x = torch.arange(24, dtype=torch.float32, device="cuda").reshape(2, 3, 4)
    out = mean_batch_invariant(x, dim, keepdim=keepdim)
    torch.testing.assert_close(out, x.mean(dim=(0, 1, 2), keepdim=keepdim))


def _exact_values(shape, dtype, seed=0):
    """Multiples of 1/8 below 32 in magnitude: exact in bf16 and fp16, and any sum of up to
    2**16 of them is exact in fp32."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randint(-255, 256, shape, device="cuda", generator=g) / 8).to(dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16), ids=("bf16", "fp16"))
@pytest.mark.parametrize(
    "shape,dim",
    (((5, 8192), [-1]), ((2, 2048, 3), [1]), ((3, 32, 64), [1, 2]), ((16, 512), None)),
    ids=("last", "middle", "two-dims", "all-dims"),
)
def test_mean_batch_invariant_sums_half_inputs_in_fp32(dtype, shape, dim):
    """bf16/fp16 means are summed in fp32 and rounded once to the input dtype, as aten's are.

    The sums are exact in fp32 and the counts are powers of two, so each mean must be the
    exact mean rounded once. Partial sums kept in bf16/fp16 would drop bits.
    """
    x = _exact_values(shape, dtype)
    with set_batch_invariant_mode(True):
        out = torch.mean(x, dim=dim)
    assert out.dtype == dtype
    assert torch.equal(out, x.double().mean(dim=dim).to(dtype))


# name: (input dtype, dim, dtype argument)
_MEAN_DTYPE_CASES = {
    "bf16-dtype-fp32": (torch.bfloat16, [-1], torch.float32),
    "fp32-dtype-bf16": (torch.float32, [1, 2], torch.bfloat16),
    "int64-dtype-fp32": (torch.int64, [1], torch.float32),
    "bf16-two-dims": (torch.bfloat16, [1, 2], None),
    "fp16-all-dims": (torch.float16, None, None),
    "fp64-last": (torch.float64, [-1], None),
    "fp64-split-dims": (torch.float64, [0, 2], None),
    "complex64-two-dims": (torch.complex64, [1, 2], None),
}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("name", tuple(_MEAN_DTYPE_CASES))
def test_mean_batch_invariant_output_dtype_matches_aten(name):
    """The output dtype is aten's (``dtype`` if given, else the input's) for the input dtypes
    and ``dtype`` arguments torch.mean accepts. The inputs are small integers, so every sum is
    exact and the values must equal aten's as well."""
    in_dtype, dim, dtype = _MEAN_DTYPE_CASES[name]
    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randint(-8, 9, (4, 16, 32), device="cuda", generator=g)
    if in_dtype.is_complex:
        x = torch.complex(x.float(), x.flip(-1).float())
    x = x.to(in_dtype)
    kwargs = {} if dtype is None else {"dtype": dtype}
    expected = torch.mean(x, dim=dim, **kwargs)
    with set_batch_invariant_mode(True):
        out = torch.mean(x, dim=dim, **kwargs)
    assert out.dtype == expected.dtype
    assert torch.equal(out, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_mean_batch_invariant_rejects_integer_outputs():
    """aten has no integer mean: without a floating ``dtype`` it raises, and so does the
    batch-invariant mean."""
    x = torch.arange(6, device="cuda").view(2, 3)
    with pytest.raises(RuntimeError, match="could not infer output dtype"):
        x.mean(1)
    with set_batch_invariant_mode(True):
        with pytest.raises(RuntimeError, match="could not infer output dtype"):
            x.mean(1)


# name: (shape, None marking the batch dim; batch dim; reduced dims; keepdim)
_MEAN_BATCH_FORMS = {
    "last": ((None, 4096), 0, [-1], False),
    "two-dims": ((None, 33, 160), 0, [1, 2], False),
    "two-dims-keepdim": ((None, 33, 160), 0, [1, 2], True),
    "split-dims": ((33, None, 160), 1, [0, 2], False),
    "leading-dims": ((33, 160, None), 2, [0, 1], False),
}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("name", tuple(_MEAN_BATCH_FORMS))
def test_mean_batch_invariant_does_not_depend_on_the_batch(name):
    """Five probe slices keep the bits of their means among 6, 129 and 1029 slices: for one
    dim, several adjacent dims, dims on both sides of the batch dim and leading dims."""
    shape, batch_dim, dim, keepdim = _MEAN_BATCH_FORMS[name]
    g = torch.Generator(device="cuda").manual_seed(0)
    probe = torch.randn([5 if s is None else s for s in shape], device="cuda", generator=g)
    filler = torch.randn([1029 if s is None else s for s in shape], device="cuda", generator=g)
    out_dim = batch_dim if keepdim else 0
    with set_batch_invariant_mode(True):
        base = torch.mean(probe, dim=dim, keepdim=keepdim)
        for rows in (6, 129, 1029):
            x = filler.narrow(batch_dim, 0, rows).clone()
            x.narrow(batch_dim, rows - 5, 5).copy_(probe)
            out = torch.mean(x, dim=dim, keepdim=keepdim)
            assert torch.equal(out.narrow(out_dim, rows - 5, 5), base), f"{name}: rows={rows}"


def _require_free_memory(nbytes):
    """Skip, with the numbers, when the device has less than ``nbytes`` free."""
    torch.cuda.empty_cache()
    free, _ = torch.cuda.mem_get_info()
    if free < nbytes:
        pytest.skip(f"needs {nbytes / 2**30:.1f} GiB of device memory, {free / 2**30:.1f} free")


@pytest.mark.large
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("form", ("rows", "column-slice", "first-dim"))
def test_mean_batch_invariant_offsets_past_int32(form):
    """Element offsets of 2**31 and more: means of the rows of a 4 GiB bf16 input that start
    past 2**31 elements, of a 16-column slice of it (fewer elements, the same span) and over
    its first dim read their own elements.

    Opt in with ``-m large``; run one case per process, since a wrapped offset can leave a
    sticky illegal-address error.
    """
    rows, cols = 2**20, 2064  # rows 1040448 and on start past element 2**31
    _require_free_memory(rows * cols * 2 + 2**30)
    x = torch.zeros(rows, cols, dtype=torch.bfloat16, device="cuda")
    x[:8] = -0.25
    x[-8:] = 1.5
    with set_batch_invariant_mode(True):
        if form == "first-dim":
            out = torch.mean(x, dim=0)
        else:
            out = torch.mean(x if form == "rows" else x[:, :16], dim=-1)
    if form == "first-dim":
        # 8 * 1.5 - 8 * 0.25 = 10 in every column, over 2**20 rows: exact in bf16.
        assert torch.equal(out, torch.full_like(out, 10 / rows))
    else:
        assert torch.equal(out[:8], torch.full_like(out[:8], -0.25))
        assert torch.equal(out[-8:], torch.full_like(out[-8:], 1.5))
        assert not out[8:-8].any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    "n",
    (
        3 * 1024 + 1,
        pytest.param(2**31 - 1024, marks=pytest.mark.large),
        pytest.param(2**31 - 1023, marks=pytest.mark.large),
        pytest.param(2**31 - 1, marks=pytest.mark.large),
    ),
    ids=("small", "counter-fits", "counter-wraps-first", "counter-wraps-last"),
)
def test_mean_batch_invariant_loop_counter_past_int32(n):
    """The mean of n bf16 ones is 1. Its loop steps the counter a block past the last element,
    which passes 2**31 for n from 2**31 - 1023 to 2**31 - 1 although every offset fits in int32.

    The three cases near 2**31 need 4 GiB each: opt in with ``-m large`` and run one case per
    process, since a wrapped counter reads before the input.
    """
    _require_free_memory(n * 2 + 2**30)
    x = torch.ones(n, dtype=torch.bfloat16, device="cuda")
    with set_batch_invariant_mode(True):
        out = torch.mean(x, dim=0)
    assert torch.equal(out, torch.ones_like(out))


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


def test_te_column_parallel_linear_parity():
    """Test that batch-invariant and regular TE linear produce same forward/backward results."""
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
    torch.manual_seed(456)
    layer_bik = TEColumnParallelLinear(
        input_size=cfg_bik.hidden_size,
        output_size=256,
        config=cfg_bik,
        init_method=init_method_normal(cfg_bik.init_method_std),
        gather_output=False,
        bias=True,
        skip_bias_add=False,
        is_expert=False,
    ).cuda()

    torch.manual_seed(456)  # Same seed for same initialization
    layer_regular = TEColumnParallelLinear(
        input_size=cfg_regular.hidden_size,
        output_size=256,
        config=cfg_regular,
        init_method=init_method_normal(cfg_regular.init_method_std),
        gather_output=False,
        bias=True,
        skip_bias_add=False,
        is_expert=False,
    ).cuda()

    # Test forward pass
    x = torch.randn(
        64, cfg_bik.hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    x_clone = x.clone().detach().requires_grad_(True)

    with set_batch_invariant_mode(True):
        out_bik, _ = layer_bik(x)

    with set_batch_invariant_mode(False):
        out_regular, _ = layer_regular(x_clone)

    # Check forward outputs are close
    assert (
        out_bik.shape == out_regular.shape
    ), f"Shape mismatch: {out_bik.shape} vs {out_regular.shape}"
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


def test_te_layernorm_linear_parity():
    """Test that batch-invariant and regular fused LayerNorm+Linear produce same results."""
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

    torch.manual_seed(321)
    layer_bik = TELayerNormColumnParallelLinear(
        input_size=cfg_bik.hidden_size,
        output_size=256,
        config=cfg_bik,
        init_method=init_method_normal(cfg_bik.init_method_std),
        gather_output=False,
        bias=True,
        skip_bias_add=False,
        is_expert=False,
    ).cuda()

    torch.manual_seed(321)
    layer_regular = TELayerNormColumnParallelLinear(
        input_size=cfg_regular.hidden_size,
        output_size=256,
        config=cfg_regular,
        init_method=init_method_normal(cfg_regular.init_method_std),
        gather_output=False,
        bias=True,
        skip_bias_add=False,
        is_expert=False,
    ).cuda()

    x = torch.randn(
        48, cfg_bik.hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    x_clone = x.clone().detach().requires_grad_(True)

    with set_batch_invariant_mode(True):
        out_bik, _ = layer_bik(x)

    with set_batch_invariant_mode(False):
        out_regular, _ = layer_regular(x_clone)

    assert out_bik.shape == out_regular.shape
    max_diff = (out_bik - out_regular).abs().max().item()
    assert max_diff < 1e-3, f"Forward output difference too large: {max_diff}"

    grad_output = torch.randn_like(out_bik)

    out_bik.backward(grad_output)
    out_regular.backward(grad_output.clone())

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


@pytest.mark.skipif(not HAVE_DEEPGEMM_BF16, reason="DeepGEMM bf16 bindings are unavailable")
def test_bik_te_general_gemm_deepgemm_backend_supports_fp32_router():
    torch.manual_seed(123)
    M1, M2, K, N = 37, 23, 128, 128
    A1 = torch.randn(M1, K, **_device(torch.float32))
    A2 = torch.randn(M2, K, **_device(torch.float32))
    A = torch.cat([A1, A2], dim=0)
    B = torch.randn(K, N, **_device(torch.float32))

    with set_batch_invariant_mode(True, backend="deepgemm"):
        full = _te_general_gemm(A, B, out_dtype=torch.float32, layout="TN")[0]
        chunks = torch.cat(
            [
                _te_general_gemm(A1, B, out_dtype=torch.float32, layout="TN")[0],
                _te_general_gemm(A2, B, out_dtype=torch.float32, layout="TN")[0],
            ],
            dim=1,
        )

    assert torch.equal(full, chunks)
