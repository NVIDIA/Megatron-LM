# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Exact custom/native recipe parity through MCore's real transformer and MoE paths."""

import copy

import pytest
import torch
import torch.nn.functional as F

from megatron.core.extensions import transformer_engine as te_extension
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.quantization.te_recipe import get_te_quantization_recipe
from megatron.core.tensor_parallel.random import (
    _get_all_rng_states,
    _set_all_rng_states,
    model_parallel_cuda_manual_seed,
)
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

pytestmark = [
    pytest.mark.internal,
    pytest.mark.launch_on_gb200,
    pytest.mark.skipif(not te_extension.HAVE_TE, reason="Transformer Engine is required"),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
]

_FACTORY_ROLES = set()


@pytest.fixture(scope="module")
def model_parallel():
    # All cases use TP=PP=EP=1. Reuse these groups rather than retaining a new
    # set of NCCL communicators for each parameter combination until session teardown.
    Utils.initialize_model_parallel(1, 1)
    try:
        yield
    finally:
        Utils.destroy_model_parallel()


def mxfp8_test_factory(role):
    """Create E4M3 block-32 quantizers matching native MXFP8 defaults."""
    from transformer_engine.pytorch.constants import DType
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer

    if role is not None:
        _FACTORY_ROLES.add((role.name, role.tensor_type))
    return MXFP8Quantizer(DType.kFloat8E4M3)


def blockwise_test_factory(role):
    """Match native FP8 block scaling: 2D weights and 1D activations/gradients."""
    from transformer_engine.pytorch.constants import DType
    from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockQuantizer

    if role is not None:
        _FACTORY_ROLES.add((role.name, role.tensor_type))
    return Float8BlockQuantizer(
        DType.kFloat8E4M3,
        rowwise=True,
        columnwise=True,
        block_scaling_dim=2 if role is not None and role.tensor_type == "weight" else 1,
        force_pow_2_scales=True,
    )


def nvfp4_test_factory(role):
    """Match native NVFP4's weight, activation, and stochastic gradient policies."""
    from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer

    if role is not None:
        _FACTORY_ROLES.add((role.name, role.tensor_type))
    weight = role is not None and role.tensor_type == "weight"
    return NVFP4Quantizer(
        with_rht=not weight,
        with_post_rht_amax=not weight,
        with_2d_quantization=weight,
        stochastic_rounding=role is not None and role.tensor_type == "grad_output",
    )


def _make_block(precision, path, checkpoint, custom, routing):
    moe = path != "dense"
    quant_kwargs = (
        {"custom_recipe": f"{__name__}.{precision}_test_factory"}
        if custom
        else (
            {"fp8": "e4m3", "fp8_recipe": precision}
            if precision in ("mxfp8", "blockwise")
            else {"fp4": "e2m1", "fp4_recipe": "nvfp4"}
        )
    )
    config = TransformerConfig(
        num_layers=1,
        hidden_size=128,
        ffn_hidden_size=256,
        num_attention_heads=4,
        params_dtype=torch.bfloat16,
        bf16=True,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        add_bias_linear=False,
        gated_linear_unit=True,
        activation_func=F.silu,
        bias_activation_fusion=False,
        bias_dropout_fusion=False,
        num_moe_experts=2 if moe else None,
        moe_ffn_hidden_size=256 if moe else None,
        moe_grouped_gemm=moe,
        moe_use_grouped_tensor=path in ("grouped", "fuser", "cutedsl"),
        use_transformer_engine_op_fuser=path in ("fuser", "cutedsl"),
        moe_mlp_glu_interleave_size=32 if path == "cutedsl" else None,
        # Top-1 produces uneven expert segments; top-2 sends every token to both experts.
        moe_router_topk=2 if routing == "aligned" else 1,
        moe_router_pre_softmax=True,
        moe_router_load_balancing_type="none",
        moe_aux_loss_coeff=0.0,
        moe_token_dispatcher_type="alltoall",
        moe_router_padding_for_quantization=False,
        recompute_granularity="full" if checkpoint else None,
        recompute_method="uniform" if checkpoint else None,
        recompute_num_layers=1 if checkpoint else None,
        **quant_kwargs,
    )
    if custom and path == "cutedsl":
        get_te_quantization_recipe(config).enable_cutedsl_fused_grouped_mlp = True
    block = TransformerBlock(
        config,
        get_gpt_layer_with_transformer_engine_spec(
            num_experts=config.num_moe_experts, moe_grouped_gemm=moe
        ),
        name="decoder",
    ).cuda()
    if moe:
        experts = block.layers[0].mlp.experts
        assert experts._use_grouped_tensor == (path in ("grouped", "fuser", "cutedsl"))
        assert experts._with_fused_impl == (path in ("fuser", "cutedsl"))
    return block


def _run_step(block, inputs, grad_output):
    block.zero_grad(set_to_none=True)
    inputs = inputs.detach().clone().requires_grad_(True)
    output = block(hidden_states=inputs, attention_mask=None)
    output.backward(grad_output)
    gradients = {}
    for name, parameter in block.named_parameters():
        assert parameter.grad is not None, f"missing gradient for {name}"
        gradients[name] = parameter.grad.detach().clone()
    return output.detach().clone(), inputs.grad.detach().clone(), gradients


def _assert_exact(actual, expected, label):
    assert torch.isfinite(actual).all(), f"nonfinite {label}"
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, msg=label)
    assert torch.equal(
        actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
    ), f"{label}: storage bits differ"


@pytest.mark.parametrize("precision", ["mxfp8", "nvfp4", "blockwise"])
@pytest.mark.parametrize(
    ("path", "routing"),
    [("dense", "aligned")]
    + [
        (path, routing)
        for path in ("discrete", "grouped", "fuser", "cutedsl")
        for routing in ("aligned", "uneven", "empty")
    ],
)
@pytest.mark.parametrize("checkpoint", [False, True], ids=["eager", "recompute"])
def test_custom_recipe_matches_native(
    precision, path, routing, checkpoint, monkeypatch, request, model_parallel
):
    """Compare forward, dgrad, and every wgrad, including a second quantizer-state reuse step."""
    from transformer_engine.pytorch.quantization import FP8GlobalStateManager

    availability_check = (
        "is_fp8_block_scaling_available"
        if precision == "blockwise"
        else f"is_{precision}_available"
    )
    available, reason = getattr(te_extension.te.pytorch, availability_check)(return_reason=True)
    if not available:
        pytest.skip(reason)
    if precision == "blockwise":
        if path == "cutedsl":
            pytest.skip("TE CuTeDSL grouped MLP fusion supports only MXFP8 and NVFP4 recipes")
        if path == "grouped" and torch.cuda.get_device_capability()[0] >= 10:
            pytest.skip("Native TE GroupedTensor FP8 block scaling is unsupported on Blackwell")
    # Keep algorithm selection fixed across the pair; the fuser path still exercises
    # the TE operation fuser with its supported grouped-GEMM implementation.
    monkeypatch.setenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "0")
    monkeypatch.setenv("NVTE_CUTEDSL_FUSED_GROUPED_MLP", "0")
    if path == "cutedsl":
        fused = request.getfixturevalue("grouped_mlp_fusions")
        if not fused.GroupedMLP_CuTeGEMMGLU.is_supported():
            pytest.skip("CuTeDSL grouped MLP kernels are not available")
    try:
        model_parallel_cuda_manual_seed(123)
        FP8GlobalStateManager.reset()
        reference = _make_block(precision, path, checkpoint, custom=False, routing=routing)
        candidate = _make_block(precision, path, checkpoint, custom=True, routing=routing)
        # Extra state contains recipe objects; only copy trainable parameters.
        reference_parameters = dict(reference.named_parameters())
        candidate_parameters = dict(candidate.named_parameters())
        assert candidate_parameters.keys() == reference_parameters.keys()
        with torch.no_grad():
            for name, parameter in reference_parameters.items():
                candidate_parameters[name].copy_(parameter)
        token_counts = {"native": [], "custom": []}
        if path != "dense":
            for label, block in (("native", reference), ("custom", candidate)):
                if routing == "empty":
                    with torch.no_grad():
                        block.layers[0].mlp.router.weight.zero_()

                def record_counts(module, args, label=label):
                    token_counts[label].append(torch.as_tensor(args[1]).cpu().clone())

                block.layers[0].mlp.experts.register_forward_pre_hook(record_counts)
        _FACTORY_ROLES.clear()
        for step in range(2):
            # GroupedTensor kernels require 256-token expert alignment; the other
            # routing cases deliberately exercise padding from smaller segments.
            tokens = 256 if path != "dense" and routing == "aligned" else 128
            inputs = torch.randn(tokens, 1, 128, device="cuda", dtype=torch.bfloat16)
            grad_output = torch.randn_like(inputs)
            rng = copy.deepcopy(_get_all_rng_states())
            expected = _run_step(reference, inputs, grad_output)
            _set_all_rng_states(*copy.deepcopy(rng))
            actual = _run_step(candidate, inputs, grad_output)
            if path == "cutedsl":
                for block in (reference, candidate):
                    (ops,) = block.layers[0].mlp.experts._fused_ops
                    (fuser,) = ops._module_groups
                    assert len(fuser._forward_ops) == 1
                    assert isinstance(fuser._forward_ops[0][0], fused.GroupedMLP_CuTeGEMMGLU)
            _assert_exact(actual[0], expected[0], f"step {step} output")
            _assert_exact(actual[1], expected[1], f"step {step} input gradient")
            assert actual[2].keys() == expected[2].keys()
            for name in expected[2]:
                _assert_exact(actual[2][name], expected[2][name], f"step {step} {name}")
            if path != "dense":
                assert len(token_counts["native"]) == len(token_counts["custom"])
                for native_counts, custom_counts in zip(*token_counts.values()):
                    torch.testing.assert_close(custom_counts, native_counts, rtol=0, atol=0)
                    if routing == "aligned":
                        assert (native_counts % 256 == 0).all()
                    elif routing == "empty":
                        assert (native_counts == 0).any()
                    elif routing == "uneven":
                        assert (native_counts % 128 != 0).any()
            if step == 0:
                _set_all_rng_states(*copy.deepcopy(rng))
                replay = _run_step(candidate, inputs, grad_output)
                _assert_exact(replay[0], actual[0], "replayed output")
                _assert_exact(replay[1], actual[1], "replayed input gradient")
                for name in actual[2]:
                    _assert_exact(replay[2][name], actual[2][name], f"replayed {name}")
            for counts in token_counts.values():
                counts.clear()
            # A weight update makes step two exercise quantization of changed parameters.
            with torch.no_grad():
                for name, parameter in reference_parameters.items():
                    if routing == "empty" and name.endswith("mlp.router.weight"):
                        continue
                    parameter.add_(expected[2][name], alpha=-0.01)
                    candidate_parameters[name].add_(actual[2][name], alpha=-0.01)
        # Ensure the custom path actually called the factory at the MCore locations.
        mlp = "mlp" if path == "dense" else "mlp.experts"
        assert (f"decoder.layers.0.{mlp}.linear_fc1", "weight") in _FACTORY_ROLES
        assert ("decoder.layers.0.self_attention.linear_qkv", "input") in _FACTORY_ROLES
    finally:
        FP8GlobalStateManager.reset()
