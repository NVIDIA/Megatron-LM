# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from megatron.core.transformer.enums import CudaGraphModule
from megatron.core.transformer.moe.megakernel.mok import weights
from megatron.core.transformer.transformer_config import TransformerConfig


def _mok_transformer_config(**overrides):
    values = {
        "num_layers": 2,
        "bf16": True,
        "moe_grouped_gemm": True,
        "hidden_size": 128,
        "attention_dropout": 0.0,
        "hidden_dropout": 0.0,
        "num_attention_heads": 4,
        "num_moe_experts": 8,
        "moe_ffn_hidden_size": 256,
        "moe_shared_expert_intermediate_size": 256,
        "expert_model_parallel_size": 4,
        "gated_linear_unit": True,
        "activation_func": F.silu,
        "gradient_accumulation_fusion": True,
        "moe_megakernel_backend": "mok",
    }
    values.update(overrides)
    return TransformerConfig(**values)


def test_mxfp8_shared_expert_config_expresses_bf16_module(monkeypatch, recwarn):
    original_recipe = object()
    config = SimpleNamespace(
        fp8="hybrid", fp8_param=True, quant_recipe=original_recipe, sentinel=object()
    )
    monkeypatch.setattr(weights, "_SHARED_EXPERT_BF16_WARNING_EMITTED", False)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)

    shared_config = weights.prepare_shared_expert_config(config)

    assert shared_config is not config
    assert config.fp8 == "hybrid" and config.fp8_param
    assert shared_config.fp8 is None and not shared_config.fp8_param
    assert shared_config.quant_recipe is original_recipe
    assert shared_config.sentinel is config.sentinel
    assert len(recwarn) == 1


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"moe_shared_expert_gate": True},
        {"fp8": "hybrid", "fp8_recipe": "mxfp8", "fp8_param": True, "moe_shared_expert_gate": True},
        {
            "fp8": "hybrid",
            "fp8_recipe": "mxfp8",
            "fp8_param": True,
            "cuda_graph_impl": "full_iteration",
            "cuda_graph_modules": [],
            "moe_layer_recompute": True,
        },
        # fp8_recipe=custom + fp8_param=False: on-the-fly BF16→MXFP8 quantization in MOK.
        {
            "fp8": "hybrid",
            "fp8_recipe": "custom",
            "fp8_param": False,
            "moe_single_grouped_weight": True,
            "moe_use_grouped_tensor": True,
            "fp8_quantizer_factory": "low_precision_recipes.routed_expert_mxfp8_fwd_factory",
        },
    ],
)
def test_mok_accepts_key_supported_configurations(overrides):
    assert _mok_transformer_config(**overrides).moe_megakernel_backend == "mok"


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"bf16": False, "fp16": True}, "FP32, FP16, and FP4 are not supported"),
        ({"fp4": "e2m1"}, "FP32, FP16, and FP4 are not supported"),
        ({"moe_grouped_gemm": False}, "moe_grouped_gemm=True"),
        (
            {"overlap_moe_expert_parallel_comm": True},
            "does not support overlap_moe_expert_parallel_comm",
        ),
        ({"gradient_accumulation_fusion": False}, "gradient_accumulation_fusion=True"),
        # fp8_recipe=mxfp8 without fp8_param=True is still rejected (no on-the-fly path for mxfp8 recipe).
        ({"fp8": "hybrid", "fp8_recipe": "mxfp8", "fp8_param": False}, "fp8_param=True"),
    ],
)
def test_mok_rejects_key_incompatible_configurations(overrides, error):
    with pytest.raises(ValueError, match=error):
        _mok_transformer_config(**overrides)


@pytest.mark.parametrize("cuda_graph_impl", ["local", "transformer_engine"])
def test_mok_rejects_per_layer_whole_layer_cuda_graph(cuda_graph_impl):
    with pytest.raises(ValueError, match="whole-layer CUDA Graph capture"):
        _mok_transformer_config(
            cuda_graph_impl=cuda_graph_impl,
            cuda_graph_modules=[],
        )


@pytest.mark.parametrize("cuda_graph_impl", ["local", "transformer_engine"])
@pytest.mark.parametrize(
    "cuda_graph_modules",
    [
        [CudaGraphModule.moe],
        [CudaGraphModule.moe_router],
        [CudaGraphModule.moe_preprocess],
        [CudaGraphModule.moe_router, CudaGraphModule.moe_preprocess],
    ],
)
def test_mok_rejects_per_layer_cuda_graph_covering_moe(
    cuda_graph_impl, cuda_graph_modules
):
    with pytest.raises(ValueError, match="moe/moe_router/moe_preprocess"):
        _mok_transformer_config(
            cuda_graph_impl=cuda_graph_impl,
            cuda_graph_modules=cuda_graph_modules,
        )


@pytest.mark.parametrize("cuda_graph_impl", ["local", "transformer_engine"])
def test_mok_accepts_per_layer_cuda_graph_outside_moe(cuda_graph_impl):
    cuda_graph_modules = [CudaGraphModule.attn]
    config = _mok_transformer_config(
        cuda_graph_impl=cuda_graph_impl,
        cuda_graph_modules=cuda_graph_modules,
    )

    assert config.cuda_graph_modules == cuda_graph_modules


# ---------------------------------------------------------------------------
# On-the-fly BF16→MXFP8 quantization tests (fp8_param=False path)
# ---------------------------------------------------------------------------

# Dimensions chosen so E*R is a multiple of 128 and C is a multiple of 128.
_OTF_CASES = [
    (4, 128, 128),   # minimal: E=4, R=128, C=128; E*R=512 (%128=0)
    (4, 256, 128),   # rectangular: E*R=1024
    (2, 256, 256),   # larger C
]


def _have_mxfp8_quantizer():
    try:
        import transformer_engine.pytorch.cpp_extensions  # noqa: F401
        from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer  # noqa: F401
        return True
    except ImportError:
        return False


_skip_no_mxfp8 = pytest.mark.skipif(
    not torch.cuda.is_available() or not _have_mxfp8_quantizer(),
    reason="requires CUDA and transformer_engine MXFP8Quantizer",
)


@_skip_no_mxfp8
@pytest.mark.parametrize("num_experts,rows,columns", _OTF_CASES)
def test_bf16_mxfp8_on_the_fly_output_shapes(num_experts, rows, columns):
    """_native_single_grouped_weight_view returns correctly shaped tensors for BF16 weights."""
    shape = (num_experts, rows, columns)
    weight = torch.nn.Parameter(torch.randn(shape, dtype=torch.bfloat16, device="cuda"))

    result = weights._native_single_grouped_weight_view(
        weight,
        num_experts=num_experts,
        rows=rows,
        columns=columns,
        use_mxfp8=True,
        cached_view=None,
    )

    row_data, swizzled_row_scale, col_data, swizzled_col_scale, flag = result

    assert flag is True
    assert row_data.shape == shape
    assert row_data.dtype == torch.float8_e4m3fn
    assert col_data.shape == shape
    assert col_data.dtype == torch.float8_e4m3fn

    # Swizzled scale: (E*R//128, C//128, 32, 16) and (E*C//128, R//128, 32, 16)
    assert swizzled_row_scale.shape == (num_experts * rows // 128, columns // 128, 32, 16)
    assert swizzled_col_scale.shape == (num_experts * columns // 128, rows // 128, 32, 16)


@_skip_no_mxfp8
@pytest.mark.parametrize("num_experts,rows,columns", _OTF_CASES)
def test_bf16_mxfp8_on_the_fly_numerical_equivalence(num_experts, rows, columns):
    """On-the-fly BF16→MXFP8 quantization must produce bit-identical FP8 data to a direct
    MXFP8Quantizer call, and the dequantized result must be close to the original BF16 weight."""
    import transformer_engine.pytorch.cpp_extensions as _tex
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
    TeDType = _tex.DType  # noqa: N806

    shape = (num_experts, rows, columns)
    torch.manual_seed(42)
    weight = torch.nn.Parameter(torch.randn(shape, dtype=torch.bfloat16, device="cuda"))

    # --- our path ---
    result = weights._native_single_grouped_weight_view(
        weight,
        num_experts=num_experts,
        rows=rows,
        columns=columns,
        use_mxfp8=True,
        cached_view=None,
    )
    row_data, _, col_data, _, _ = result

    # --- reference: direct MXFP8Quantizer ---
    quantizer = MXFP8Quantizer(fp8_dtype=TeDType.kFloat8E4M3, rowwise=True, columnwise=True)
    mxfp8_ref = quantizer.quantize_impl(weight.data)

    ref_row = mxfp8_ref._rowwise_data.view(torch.float8_e4m3fn).view(shape)
    ref_col = mxfp8_ref._columnwise_data.view(torch.float8_e4m3fn).view(shape)

    # FP8 payload must be bit-for-bit identical: our code reads from the same
    # MXFP8Tensor that quantize_impl produces, so any discrepancy is a bug.
    assert torch.equal(row_data, ref_row), "Rowwise FP8 data mismatch vs. direct quantizer"
    assert torch.equal(col_data, ref_col), "Columnwise FP8 data mismatch vs. direct quantizer"

    # Sanity: dequantized values should be close to the original BF16 weight.
    dequant = mxfp8_ref.dequantize(dtype=torch.bfloat16)
    max_err = (weight.data - dequant).abs().max().item()
    # MXFP8 E4M3 (3 mantissa bits) allows ~12.5% relative error per block;
    # for random N(0,1) weights the absolute error is well under 0.5.
    assert max_err < 0.5, f"Dequantization error {max_err:.4f} unexpectedly large"


@_skip_no_mxfp8
def test_bf16_mxfp8_on_the_fly_cache_reuse():
    """Second call with cached_view reuses the swizzled scale buffers (same data_ptr)."""

    num_experts, rows, columns = 4, 128, 128
    shape = (num_experts, rows, columns)
    weight = torch.nn.Parameter(torch.randn(shape, dtype=torch.bfloat16, device="cuda"))

    first = weights._native_single_grouped_weight_view(
        weight, num_experts=num_experts, rows=rows, columns=columns,
        use_mxfp8=True, cached_view=None,
    )
    second = weights._native_single_grouped_weight_view(
        weight, num_experts=num_experts, rows=rows, columns=columns,
        use_mxfp8=True, cached_view=first,
    )

    # Scale buffers must be updated in-place: same address, potentially new values.
    assert first[1].data_ptr() == second[1].data_ptr(), "Row scale buffer address changed"
    assert first[3].data_ptr() == second[3].data_ptr(), "Col scale buffer address changed"
