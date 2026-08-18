# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import argparse
import os
from contextlib import nullcontext

import pytest
import torch
import torch.nn.functional as F

from megatron.core.extensions.transformer_engine import TEActivationOp
from megatron.core.fusions.cutedsl_situ_glu import (
    CuTeDSLSiTUGLU,
    ScaledSiTUGLU,
    install_grouped_situ_glu_kernels,
    make_scaled_situ_glu,
    make_situ_glu,
    situ_glu_reference,
)
from megatron.core.transformer.transformer_config import TransformerConfig


def test_situ_glu_reference_formula():
    input_ = torch.tensor([[0.5, -1.0, 2.0, -3.0]], dtype=torch.float64)
    gate, up = input_.chunk(2, dim=-1)
    expected = 4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)
    expected = expected * 25.0 * torch.tanh(up / 25.0)
    torch.testing.assert_close(situ_glu_reference(input_), expected)


@pytest.mark.parametrize("flag", ["--situ-glu", "--moe-use-situ-glu"])
def test_situ_glu_cli_aliases_enable_the_global_activation(flag):
    from megatron.training.arguments import _add_network_size_args

    parser = argparse.ArgumentParser()
    _add_network_size_args(parser)

    args = parser.parse_args([flag])

    assert args.use_situ_glu is True


@pytest.mark.parametrize(
    "precision_kwargs",
    [
        {"bf16": True},
        {"fp8": "hybrid", "fp8_recipe": "mxfp8"},
        {"fp4": "e2m1", "fp4_recipe": "nvfp4"},
    ],
    ids=["bf16", "mxfp8", "nvfp4"],
)
def test_situ_glu_selects_common_te_activation_builder(monkeypatch, precision_kwargs):
    import transformer_engine.pytorch as te

    monkeypatch.delattr(te.ops, "SiTUGLU", raising=False)
    config = TransformerConfig(
        num_layers=1,
        hidden_size=32,
        num_attention_heads=4,
        activation_func=F.silu,
        gated_linear_unit=True,
        use_te_activation_func=True,
        use_situ_glu=True,
        **precision_kwargs,
    )

    activation = TEActivationOp(config)

    assert isinstance(activation, CuTeDSLSiTUGLU)


def test_situ_glu_prefers_native_te_op(monkeypatch):
    import transformer_engine.pytorch as te

    class NativeSiTUGLU(torch.nn.Module):
        def __init__(self, beta1, beta2):
            super().__init__()
            self.beta1 = beta1
            self.beta2 = beta2

    monkeypatch.setattr(te.ops, "SiTUGLU", NativeSiTUGLU, raising=False)

    activation = make_situ_glu(beta1=4.0, beta2=25.0)

    assert isinstance(activation, NativeSiTUGLU)
    assert activation.beta1 == 4.0
    assert activation.beta2 == 25.0


def test_scaled_situ_glu_bf16_fallback_does_not_require_cudnn(monkeypatch):
    import transformer_engine.pytorch as te

    monkeypatch.delattr(te.ops, "ScaledSiTUGLU", raising=False)
    activation = make_scaled_situ_glu(
        beta1=4.0, beta2=25.0, install_grouped_fallback=False, glu_interleave_size=None
    )

    assert isinstance(activation, ScaledSiTUGLU)


@pytest.mark.parametrize(
    ("precision_kwargs", "error"),
    [
        ({"fp8": "hybrid", "fp8_recipe": "delayed"}, "supports FP8 only"),
        (
            {"fp4": "e2m1", "fp4_recipe": "custom", "fp4_quantizer_factory": "test.fake.factory"},
            "supports FP4 only",
        ),
    ],
)
def test_situ_glu_rejects_unsupported_quantization_recipe(precision_kwargs, error):
    with pytest.raises(ValueError, match=error):
        TransformerConfig(
            num_layers=1,
            hidden_size=32,
            num_attention_heads=4,
            activation_func=F.silu,
            gated_linear_unit=True,
            use_te_activation_func=True,
            use_situ_glu=True,
            **precision_kwargs,
        )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_cutedsl_situ_glu_forward_backward(dtype):
    torch.manual_seed(1234)
    input_ref = torch.randn(5, 64, device="cuda", dtype=dtype, requires_grad=True)
    input_cute = input_ref.detach().clone().requires_grad_(True)
    grad = torch.randn(5, 32, device="cuda", dtype=dtype)

    output_ref = situ_glu_reference(input_ref)
    output_ref.backward(grad)

    output_cute = CuTeDSLSiTUGLU()(input_cute)
    output_cute.backward(grad)

    torch.testing.assert_close(output_cute, output_ref, rtol=2.0e-2, atol=2.0e-2)
    torch.testing.assert_close(input_cute.grad, input_ref.grad, rtol=3.0e-2, atol=3.0e-2)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_cutedsl_situ_glu_interleaved_forward_backward(dtype):
    torch.manual_seed(4321)
    rows = 5
    width = 64
    interleave_size = 8
    input_ref = torch.randn(rows, width, device="cuda", dtype=dtype, requires_grad=True)
    input_interleaved = (
        input_ref.detach()
        .reshape(rows, 2, width // (2 * interleave_size), interleave_size)
        .transpose(1, 2)
        .contiguous()
        .view(rows, width)
        .requires_grad_(True)
    )
    grad = torch.randn(rows, width // 2, device="cuda", dtype=dtype)

    output_ref = situ_glu_reference(input_ref)
    output_ref.backward(grad)

    output_cute = CuTeDSLSiTUGLU(interleave_size=interleave_size)(input_interleaved)
    output_cute.backward(grad)
    grad_interleaved_ref = (
        input_ref.grad.reshape(rows, 2, width // (2 * interleave_size), interleave_size)
        .transpose(1, 2)
        .contiguous()
        .view(rows, width)
    )

    torch.testing.assert_close(output_cute, output_ref, rtol=2.0e-2, atol=2.0e-2)
    torch.testing.assert_close(
        input_interleaved.grad, grad_interleaved_ref, rtol=3.0e-2, atol=3.0e-2
    )


def test_grouped_situ_glu_installs_with_cutedsl_baseline(monkeypatch):
    import transformer_engine.pytorch as te
    from cudnn.grouped_gemm.grouped_gemm_dglu import api as dglu_api
    from cudnn.grouped_gemm.grouped_gemm_glu import api as glu_api

    try:
        from cudnn.grouped_gemm.grouped_gemm_glu_hadamard import api as glu_hadamard_api
    except ImportError:
        try:
            from cudnn.gemm.cutedsl.grouped.glu_hadamard import api as glu_hadamard_api
        except ImportError:
            glu_hadamard_api = None

    monkeypatch.setenv("NVTE_CUTEDSL_FUSED_GROUPED_MLP", "1")
    te.ops.fused.GroupedMLP_CuTeGEMMGLU.is_supported.cache_clear()
    install_grouped_situ_glu_kernels(4.0, 25.0)

    assert glu_api.BlockScaledMoEGroupedGemmGluBiasKernel.__name__ == "_SiTUForwardKernel"
    assert dglu_api.BlockScaledMoEGroupedGemmDgluDbiasKernel.__name__ == "_SiTUBackwardKernel"
    if glu_hadamard_api is not None:
        assert (
            glu_hadamard_api.BlockScaledMoEGroupedGemmGluHadamardKernel.__name__
            == "_SiTUForwardHadamardKernel"
        )
    assert te.ops.fused.GroupedMLP_CuTeGEMMGLU.is_supported()


@pytest.mark.parametrize("precision", ["bf16", "mxfp8", "nvfp4"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.internal
def test_mcore_moe_situ_glu_forward_backward_uses_expected_backend(precision):
    """Run a real MCore MoE and prove block-scaled cases select TE's fused op."""
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("Fused block-scaled grouped MLP requires Blackwell")
    if int(os.environ.get("NVTE_CUTEDSL_FUSED_GROUPED_MLP", "0")) <= 0:
        pytest.skip("NVTE_CUTEDSL_FUSED_GROUPED_MLP is not enabled")

    import transformer_engine.pytorch as te

    from megatron.core.fp4_utils import get_fp4_context
    from megatron.core.fp8_utils import get_fp8_context
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_with_transformer_engine_submodules,
    )
    from megatron.core.transformer.module import Float16Module
    from megatron.core.transformer.moe.experts import TEGroupedMLP
    from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
    from megatron.core.transformer.spec_utils import get_submodules
    from megatron.training.initialize import _set_random_seed
    from tests.unit_tests.test_utilities import Utils

    Utils.destroy_model_parallel()
    Utils.initialize_model_parallel(1, 1)
    precision_kwargs = {}
    if precision == "mxfp8":
        precision_kwargs = {"fp8": "hybrid", "fp8_recipe": "mxfp8"}
    elif precision == "nvfp4":
        precision_kwargs = {"fp4": "e2m1", "fp4_recipe": "nvfp4"}

    config = TransformerConfig(
        num_layers=1,
        hidden_size=256,
        ffn_hidden_size=512,
        num_attention_heads=8,
        num_moe_experts=4,
        moe_router_topk=2,
        moe_router_load_balancing_type="none",
        moe_token_dispatcher_type="alltoall",
        moe_grouped_gemm=True,
        use_transformer_engine_op_fuser=True,
        moe_mlp_glu_interleave_size=32,
        use_cpu_initialization=False,
        add_bias_linear=False,
        gated_linear_unit=True,
        activation_func=F.silu,
        use_te_activation_func=True,
        use_situ_glu=True,
        bias_activation_fusion=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        **precision_kwargs,
    )
    _set_random_seed(seed_=123, data_parallel_random_init=False)
    submodules = get_submodules(
        get_gpt_layer_with_transformer_engine_submodules(
            config.num_moe_experts, moe_grouped_gemm=True
        ).mlp
    )
    assert isinstance(submodules, MoESubmodules)
    layer = MoELayer(config, submodules)
    layer = Float16Module(layer.config, layer).module.cuda()
    assert isinstance(layer.experts, TEGroupedMLP)
    assert layer.experts._with_fused_impl

    hidden_states = torch.randn(
        (4096, 1, config.hidden_size), dtype=torch.bfloat16, device="cuda", requires_grad=True
    )
    context = nullcontext()
    if precision == "mxfp8":
        context = get_fp8_context(config)
    elif precision == "nvfp4":
        context = get_fp4_context(config)
    with context:
        output, _ = layer(hidden_states)
    output.float().square().mean().backward()

    assert hidden_states.grad is not None
    assert torch.isfinite(output).all()
    assert all(parameter.grad is not None for parameter in layer.experts.parameters())
    (ops,) = layer.experts._fused_ops
    assert isinstance(ops[1], ScaledSiTUGLU)
    assert ops._module_groups is not None
    fuser = ops._module_groups[0]
    fused_types = tuple(type(op) for op, _ in fuser._forward_ops)
    if precision in ("mxfp8", "nvfp4"):
        assert te.ops.fused.GroupedMLP_CuTeGEMMGLU in fused_types
    else:
        assert te.ops.fused.GroupedMLP_CuTeGEMMGLU not in fused_types
    if precision == "nvfp4":
        try:
            from cudnn.grouped_gemm.grouped_gemm_glu_hadamard import api as glu_hadamard_api
        except ImportError:
            from cudnn.gemm.cutedsl.grouped.glu_hadamard import api as glu_hadamard_api

        assert (
            glu_hadamard_api.BlockScaledMoEGroupedGemmGluHadamardKernel.__name__
            == "_SiTUForwardHadamardKernel"
        )

    Utils.destroy_model_parallel()
