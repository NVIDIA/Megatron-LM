# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import importlib
import os

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

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
from tests.unit_tests.test_utilities import Utils, is_lts_ci_image

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


# ============================================================================
# TE operation fuser (use_transformer_engine_op_fuser) under batch invariance
# ============================================================================

# DeepGEMM ships only in the dev CI image; there the deepgemm cases fail without it.
_OP_FUSER_BACKENDS = [
    "triton",
    pytest.param(
        "deepgemm",
        marks=pytest.mark.skipif(is_lts_ci_image(), reason="DeepGEMM is not in the LTS image."),
    ),
]


def _forbid_te_dense_gemm(monkeypatch):
    """Make TE's cuBLASLt dense GEMM raise, so a bypass of the batch-invariant GEMM fails."""
    import transformer_engine_torch as tex

    def raise_(*args, **kwargs):
        raise AssertionError("tex.generic_gemm ran under batch-invariant mode")

    monkeypatch.setattr(tex, "generic_gemm", raise_)


@pytest.mark.parametrize("backend", _OP_FUSER_BACKENDS)
def test_te_op_fuser_mlp_batch_invariant(monkeypatch, backend):
    """TEFusedMLP (the op fuser's RMSNorm, BasicLinear, bias, SwiGLU, BasicLinear) gives a
    token the same bits whatever its co-batch, and never runs TE's cuBLASLt GEMM."""
    import transformer_engine.pytorch.ops.basic.rmsnorm as te_ops_rmsnorm

    from megatron.core.extensions.transformer_engine import TEFusedMLP
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.transformer.custom_layers import batch_invariant_kernels as bik
    from megatron.core.transformer.mlp import MLPSubmodules

    assert backend != "deepgemm" or HAVE_DEEPGEMM_BF16
    Utils.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(123)
    try:
        cfg = TransformerConfig(
            num_layers=1,
            hidden_size=256,
            ffn_hidden_size=512,
            num_attention_heads=8,
            use_cpu_initialization=True,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            batch_invariant_mode=True,
            batch_invariant_backend=backend,
            flash_attention_version=_BIK_FA_VERSION,
            params_dtype=torch.bfloat16,
            bf16=True,
            normalization="RMSNorm",
            layernorm_epsilon=1e-5,
            gated_linear_unit=True,
            activation_func=F.silu,
            attention_backend=AttnBackend.flash,
            use_transformer_engine_op_fuser=True,
        )
        mlp = (
            TEFusedMLP.as_mlp_submodule(
                submodules=MLPSubmodules(
                    linear_fc1=TELayerNormColumnParallelLinear, linear_fc2=TERowParallelLinear
                ),
                config=cfg,
                pg_collection=ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp"]),
                is_mtp_layer=False,
            )
            .cuda()
            .eval()
        )
        torch.manual_seed(7)
        target = torch.randn(8, 1, cfg.hidden_size, device="cuda", dtype=torch.bfloat16)
        _forbid_te_dense_gemm(monkeypatch)
        with torch.no_grad(), set_batch_invariant_mode(True, backend=backend):
            assert te_ops_rmsnorm.rmsnorm_fwd is bik._te_rmsnorm_fwd_patched
            alone, _ = mlp(target)
            assert alone.abs().max() > 0
            for left, right in ((0, 1), (5, 23), (37, 64), (120, 3)):
                filler = torch.randn(
                    left + right, 1, cfg.hidden_size, device="cuda", dtype=torch.bfloat16
                )
                batch = torch.cat([filler[:left], target, filler[left:]])
                out, _ = mlp(batch)
                assert torch.equal(out[left : left + 8], alone), (left, right)
        assert te_ops_rmsnorm.rmsnorm_fwd is not bik._te_rmsnorm_fwd_patched
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize("backend", _OP_FUSER_BACKENDS)
@pytest.mark.parametrize("case", ["accumulate-fp32", "alpha-beta-bias", "alpha"])
def test_bik_te_general_gemm_honors_accumulate_alpha_beta(backend, case):
    """The batch-invariant general_gemm computes what TE's computes with accumulate, alpha
    and beta: it adds into out (a weight gradient into fp32 main_grad, a scaled linear
    into a residual) instead of overwriting it."""
    import transformer_engine.pytorch.cpp_extensions as te_cpp

    assert backend != "deepgemm" or HAVE_DEEPGEMM_BF16
    torch.manual_seed(5)
    bf16 = _device(torch.bfloat16)
    if case == "accumulate-fp32":
        # Weight gradient (NT) of x [64, 128] and dy [64, 96], added into fp32 main_grad.
        A, B = torch.randn(64, 128, **bf16), torch.randn(64, 96, **bf16)
        out0 = torch.randn(96, 128, device="cuda", dtype=torch.float32)
        kwargs = dict(out_dtype=torch.float32, accumulate=True, layout="NT", grad=True)
    elif case == "alpha-beta-bias":
        # Scaled linear plus bias, added into a scaled bf16 residual.
        A, B = torch.randn(96, 128, **bf16), torch.randn(64, 128, **bf16)
        out0 = torch.randn(64, 96, **bf16)
        bias = torch.randn(96, **bf16)
        kwargs = dict(out_dtype=torch.bfloat16, accumulate=True, alpha=0.5, beta=2.0, bias=bias)
    else:
        A, B = torch.randn(96, 128, **bf16), torch.randn(64, 128, **bf16)
        out0 = torch.randn(64, 96, **bf16)
        kwargs = dict(out_dtype=torch.bfloat16, alpha=0.25)
    reference = out0.clone()
    te_cpp.general_gemm(A, B, out=reference, **kwargs)
    actual = out0.clone()
    with set_batch_invariant_mode(True, backend=backend):
        result = te_cpp.general_gemm(A, B, out=actual, **kwargs)[0]
    assert result is actual
    torch.testing.assert_close(actual.float(), reference.float(), rtol=2e-2, atol=1e-1)


@pytest.mark.parametrize("zero_centered_gamma", [False, True])
def test_te_op_fuser_rmsnorm_batch_invariant(zero_centered_gamma):
    """The op fuser's te.ops.RMSNorm, which imports TE's rmsnorm_fwd by name, runs the
    batch-invariant RMSNorm in forward and still back-propagates through TE's rmsnorm_bwd."""
    import transformer_engine.pytorch as te
    import transformer_engine.pytorch.ops.basic.rmsnorm as te_ops_rmsnorm

    from megatron.core.transformer.custom_layers import batch_invariant_kernels as bik

    hidden, eps = 256, 1e-5
    norm = te.ops.RMSNorm(
        hidden,
        eps=eps,
        zero_centered_gamma=zero_centered_gamma,
        device="cuda",
        dtype=torch.bfloat16,
    )
    torch.manual_seed(11)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(hidden) * 0.1 + (0.0 if zero_centered_gamma else 1.0))
    target = torch.randn(8, hidden, device="cuda", dtype=torch.bfloat16)
    with set_batch_invariant_mode(True, backend="triton"):
        assert te_ops_rmsnorm.rmsnorm_fwd is bik._te_rmsnorm_fwd_patched
        with torch.no_grad():
            alone = norm(target)
            expected, _ = bik._rmsnorm_batch_invariant_fp32(
                target, norm.weight, eps, zero_centered_gamma
            )
            assert torch.equal(alone, expected.to(torch.bfloat16))
            for left, right in ((0, 1), (5, 23), (37, 64), (120, 3)):
                filler = torch.randn(left + right, hidden, device="cuda", dtype=torch.bfloat16)
                out = norm(torch.cat([filler[:left], target, filler[left:]]))
                assert torch.equal(out[left : left + 8], alone), (left, right)
        x = target.clone().requires_grad_(True)
        dy = torch.randn_like(target)
        norm(x).backward(dy)
    assert te_ops_rmsnorm.rmsnorm_fwd is not bik._te_rmsnorm_fwd_patched
    x64 = target.double().requires_grad_(True)
    w64 = norm.weight.detach().double() + (1.0 if zero_centered_gamma else 0.0)
    (x64 * torch.rsqrt(x64.pow(2).mean(-1, keepdim=True) + eps) * w64).backward(dy.double())
    torch.testing.assert_close(x.grad.double(), x64.grad, rtol=2e-2, atol=2e-2)
