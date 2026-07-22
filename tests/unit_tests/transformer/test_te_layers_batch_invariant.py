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
from megatron.core.transformer.custom_layers.batch_invariant_kernels import set_batch_invariant_mode
from megatron.core.transformer.enums import AttnBackend, AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import init_method_normal, is_te_min_version
from tests.unit_tests.test_utilities import Utils

try:
    import flash_attn_3

    HAVE_FA3 = True
except ImportError:
    HAVE_FA3 = False


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
