# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Precision tests for the DSA indexer weight projection.

The independent reference follows DeepSeek-V3.2 ``Indexer`` at
``deepseek-ai/DeepSeek-V3.2@a7e62ac04ecb2c0a54d736dc46601c5606cf10a6``:
the model-dtype projection parameter is consumed by an FP32 linear operation.
"""

import dataclasses
from argparse import ArgumentParser
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import megatron.core.transformer.experimental_attention_variant.dsa as dsa_module
from megatron.core.extensions.transformer_engine import HAVE_TE, TELinear, TENorm
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.quantization.quant_config import MatchContext, RecipeConfig
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.experimental_attention_variant.dsa import (
    DSAIndexer,
    DSAIndexerSubmodules,
    fused_qk_topk_naive,
)
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.argument_utils import ArgumentGroupFactory
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.experimental_attention_variant.dsa_native_parity_utils import (
    _make_config,
    run_absorbed_mla_dsa_parity,
)

pytestmark = pytest.mark.launch_on_gb200


def _precision_config(
    precision: str,
    *,
    use_quantization: bool = False,
    weights_proj_output_dtype: str = "fp32",
    dsa_kernel_backend: str = "none",
    delay_wgrad_compute: bool = False,
    gradient_accumulation_fusion: bool = False,
):
    config = _make_config(
        use_sparse_loss=True, calculate_per_token_loss=False, dsa_kernel_backend=dsa_kernel_backend
    )
    if precision == "bf16":
        config = dataclasses.replace(config, fp8=None, fp8_param=False)
    elif precision == "mxfp8":
        config = dataclasses.replace(config, fp8="hybrid", fp8_recipe="mxfp8", fp8_param=False)
    elif precision == "fp8_param":
        config = dataclasses.replace(config, fp8="hybrid", fp8_recipe="delayed", fp8_param=True)
    else:
        raise ValueError(f"Unsupported test precision: {precision}")

    config = dataclasses.replace(
        config,
        dsa_indexer_weights_proj_use_quantization=use_quantization,
        dsa_indexer_weights_proj_output_dtype=weights_proj_output_dtype,
    )
    # Production validation ties delayed WGrad to the MoE overlap schedule. These
    # module-level tests exercise only the indexer's split-backward contract.
    object.__setattr__(config, "delay_wgrad_compute", delay_wgrad_compute)
    object.__setattr__(config, "gradient_accumulation_fusion", gradient_accumulation_fusion)
    return config


def _skip_if_precision_unsupported(precision: str) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for DSA indexer precision tests")
    if not HAVE_TE:
        pytest.skip("Transformer Engine is required for DSA indexer precision tests")
    if precision == "mxfp8" and torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("MXFP8 requires Blackwell or newer")
    if precision in ("mxfp8", "fp8_param"):
        from transformer_engine.pytorch.fp8 import check_fp8_support

        supported, reason = check_fp8_support()
        if not supported:
            pytest.skip(reason)


def _build_indexer(config) -> DSAIndexer:
    submodules = DSAIndexerSubmodules(
        linear_wq_b=ModuleSpec(module=TELinear),
        linear_wk=ModuleSpec(module=TELinear),
        k_norm=ModuleSpec(module=TENorm),
        linear_weights_proj=ModuleSpec(module=TELinear),
    )
    init_context = get_fp8_context(config, layer_no=0, is_init=True)
    with init_context:
        return DSAIndexer(config=config, submodules=submodules).cuda()


def _torch_weights_proj_output_dtype(weights_proj_output_dtype: str) -> torch.dtype:
    return torch.bfloat16 if weights_proj_output_dtype == "bf16" else torch.float32


def test_fp32_weights_projection_ensures_weight_ready_before_direct_gemm(monkeypatch):
    events = []
    weight = SimpleNamespace(_ensure_param_ready_callback=lambda: events.append("ready"))
    indexer = SimpleNamespace(linear_weights_proj=SimpleNamespace(weight=weight))
    expected = object()

    def fake_apply(x, actual_weight, linear, actual_indexer):
        del x, linear
        assert actual_weight is weight
        assert actual_indexer is indexer
        events.append("gemm")
        return expected

    monkeypatch.setattr(dsa_module._DSAWeightsProjection, "apply", fake_apply)

    output = dsa_module._dsa_weights_projection_fp32(object(), indexer)

    assert output is expected
    assert events == ["ready", "gemm"]


@pytest.fixture(autouse=True)
def _model_parallel():
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=1)
    yield
    Utils.destroy_model_parallel()


@pytest.mark.internal
@pytest.mark.parametrize(
    ("precision", "use_quantization", "weights_proj_output_dtype"),
    [
        pytest.param("bf16", True, "bf16", id="bf16-quantized-contract-bf16-output"),
        pytest.param("bf16", False, "bf16", id="bf16-unquantized-bf16-output"),
        pytest.param("bf16", False, "fp32", id="bf16-unquantized-fp32-output"),
        pytest.param("mxfp8", True, "bf16", id="mxfp8-quantized-bf16-output"),
        pytest.param("mxfp8", False, "bf16", id="mxfp8-unquantized-bf16-output"),
        pytest.param("mxfp8", False, "fp32", id="mxfp8-unquantized-fp32-output"),
        pytest.param("fp8_param", True, "bf16", id="fp8-param-quantized-bf16-output"),
        pytest.param("fp8_param", False, "bf16", id="fp8-param-unquantized-bf16-output"),
        pytest.param("fp8_param", False, "fp32", id="fp8-param-unquantized-fp32-output"),
    ],
)
def test_weights_projection_precision_contract(
    precision: str, use_quantization: bool, weights_proj_output_dtype: str
):
    _skip_if_precision_unsupported(precision)
    config = _precision_config(
        precision,
        use_quantization=use_quantization,
        weights_proj_output_dtype=weights_proj_output_dtype,
    )
    indexer = _build_indexer(config)

    x = torch.randn(
        32, 1, config.hidden_size, dtype=torch.bfloat16, device="cuda", requires_grad=True
    )
    qr = torch.randn(
        32, 1, config.q_lora_rank, dtype=torch.bfloat16, device="cuda", requires_grad=True
    )
    runtime_context = get_fp8_context(config, layer_no=0)
    with runtime_context:
        _q, _k, weights = indexer.forward_before_topk(x, qr)

    projection = indexer.linear_weights_proj
    assert weights.dtype == _torch_weights_proj_output_dtype(weights_proj_output_dtype)

    from transformer_engine.pytorch.float8_tensor import Float8Tensor

    if use_quantization and precision == "fp8_param":
        assert isinstance(projection.weight, Float8Tensor)
    else:
        assert projection.weight.dtype == torch.bfloat16
        assert indexer.state_dict()["linear_weights_proj.weight"].dtype == torch.bfloat16

    if not use_quantization:
        assert not getattr(projection, "fp8", False)
        assert not getattr(projection, "_fp8_workspaces", {})
    elif precision != "bf16":
        assert projection.fp8

    if precision != "bf16":
        assert indexer.linear_wq_b.fp8
        assert indexer.linear_wk.fp8
    if precision == "fp8_param":
        assert isinstance(indexer.linear_wq_b.weight, Float8Tensor)
        assert isinstance(indexer.linear_wk.weight, Float8Tensor)
        assert isinstance(projection.weight, Float8Tensor) is use_quantization


@pytest.mark.internal
def test_precision_config_defaults_and_invalid_combinations():
    default_config = _make_config(
        use_sparse_loss=True, calculate_per_token_loss=False, dsa_kernel_backend="none"
    )
    assert default_config.dsa_indexer_weights_proj_use_quantization is True
    assert default_config.dsa_indexer_weights_proj_output_dtype == "bf16"

    with pytest.raises(ValueError, match="dsa_indexer_weights_proj_output_dtype must be"):
        dataclasses.replace(default_config, dsa_indexer_weights_proj_output_dtype="fp8")

    with pytest.raises(ValueError, match="requires.*use_quantization=False"):
        dataclasses.replace(
            default_config,
            dsa_indexer_weights_proj_use_quantization=True,
            dsa_indexer_weights_proj_output_dtype="fp32",
        )

    with pytest.raises(ValueError, match="not supported.*cudnn"):
        dataclasses.replace(
            default_config,
            dsa_kernel_backend="cudnn",
            dsa_indexer_weights_proj_use_quantization=False,
            dsa_indexer_weights_proj_output_dtype="fp32",
        )


@pytest.mark.internal
def test_precision_config_rejects_fp32_output_with_matching_quant_recipe():
    module_path = "decoder.layers.0.self_attention.core_attention.indexer.linear_weights_proj"
    quant_recipe = RecipeConfig.from_config_dict(
        {
            "matchers": {
                "dsa_weights_proj": {
                    "type": "glob",
                    "enabled": True,
                    "pattern": "*indexer.linear_weights_proj",
                    "config": "fp8",
                }
            },
            "configs": {
                "fp8": {
                    "transformer_engine_config_type": "TEQuantizationParams",
                    "training_recipe": {"fp8_quantization_recipe": "tensorwise"},
                }
            },
        }
    )
    assert quant_recipe.match(MatchContext(module_path=module_path, layer_number=0)) is not None

    with pytest.raises(ValueError, match="requires.*use_quantization=False"):
        dataclasses.replace(
            _make_config(
                use_sparse_loss=True, calculate_per_token_loss=False, dsa_kernel_backend="none"
            ),
            quant_recipe=quant_recipe,
            dsa_indexer_weights_proj_use_quantization=True,
            dsa_indexer_weights_proj_output_dtype="fp32",
        )


@pytest.mark.internal
def test_precision_config_contract_is_validated_when_dsa_indexer_is_unused():
    """Explicit precision settings have one contract regardless of the active ModuleSpec."""
    with pytest.raises(ValueError, match="dsa_indexer_weights_proj_output_dtype must be"):
        dataclasses.replace(
            _make_config(
                use_sparse_loss=True, calculate_per_token_loss=False, dsa_kernel_backend="none"
            ),
            experimental_attention_variant=None,
            dsa_indexer_weights_proj_output_dtype="fp8",
        )


@pytest.mark.internal
def test_dsv4_config_validates_dsa_indexer_precision_contract():
    """DSA indexer precision fields remain self-consistent when DSv4 is selected."""
    with pytest.raises(ValueError, match="requires.*use_quantization=False"):
        dataclasses.replace(
            _make_config(
                use_sparse_loss=True, calculate_per_token_loss=False, dsa_kernel_backend="none"
            ),
            experimental_attention_variant="dsv4_hybrid",
            csa_compress_ratios=[4],
            fp8="hybrid",
            fp8_recipe="mxfp8",
            dsa_indexer_weights_proj_use_quantization=True,
            dsa_indexer_weights_proj_output_dtype="fp32",
        )


@pytest.mark.internal
def test_precision_cli_contract():
    fields = {"dsa_indexer_weights_proj_use_quantization", "dsa_indexer_weights_proj_output_dtype"}
    exclude = [
        field.name for field in dataclasses.fields(TransformerConfig) if field.name not in fields
    ]
    parser = ArgumentParser()
    ArgumentGroupFactory(TransformerConfig, exclude=exclude).build_group(parser)

    defaults = parser.parse_args([])
    assert defaults.dsa_indexer_weights_proj_use_quantization is True
    assert defaults.dsa_indexer_weights_proj_output_dtype == "bf16"

    explicit = parser.parse_args(
        [
            "--no-dsa-indexer-weights-proj-use-quantization",
            "--dsa-indexer-weights-proj-output-dtype",
            "fp32",
        ]
    )
    assert explicit.dsa_indexer_weights_proj_use_quantization is False
    assert explicit.dsa_indexer_weights_proj_output_dtype == "fp32"

    with pytest.raises(SystemExit):
        parser.parse_args(["--dsa-indexer-weights-proj-output-dtype", "fp8"])


@pytest.mark.internal
def test_weights_projection_forward_backward_and_near_tie_topk_match_fp32_reference():
    _skip_if_precision_unsupported("bf16")
    config = _precision_config("bf16")
    indexer = _build_indexer(config)
    projection = indexer.linear_weights_proj

    x = torch.randn(
        8, 1, config.hidden_size, dtype=torch.bfloat16, device="cuda", requires_grad=True
    )
    ref_x = x.detach().clone().requires_grad_(True)
    ref_weight = projection.weight.detach().clone().requires_grad_(True)
    grad_output = torch.randn(8, 1, config.dsa_indexer_n_heads, dtype=torch.float32, device="cuda")

    output = indexer._project_indexer_weights(x)
    reference = F.linear(ref_x.float(), ref_weight.float())
    torch.testing.assert_close(output, reference, rtol=2.0e-2, atol=1.0e-2)

    output.backward(grad_output)
    reference.backward(grad_output)
    torch.testing.assert_close(x.grad, ref_x.grad, rtol=2.0e-2, atol=1.0e-2)
    torch.testing.assert_close(projection.weight.grad, ref_weight.grad, rtol=2.0e-2, atol=1.0e-2)

    # A BF16 output would round these two accumulated head weights to a tie. The
    # FP32 projection retains the one-BF16-ULP contribution and therefore the same
    # top-1 key as the independent FP32 reference.
    with torch.no_grad():
        projection.weight.zero_()
        projection.weight[0].fill_(1.0)
        projection.weight[1].fill_(1.0)
        projection.weight[1, -1] = torch.tensor(1.0078125, dtype=torch.bfloat16, device="cuda")
    near_tie_x = torch.ones(1, 1, config.hidden_size, dtype=torch.bfloat16, device="cuda")
    weights = indexer._project_indexer_weights(near_tie_x)
    ref_weights = F.linear(near_tie_x.float(), projection.weight.float())
    assert weights[0, 0, 1] > weights[0, 0, 0]
    assert weights.to(torch.bfloat16)[0, 0, 1] == weights.to(torch.bfloat16)[0, 0, 0]

    q = torch.zeros(
        1,
        1,
        config.dsa_indexer_n_heads,
        config.dsa_indexer_head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    k = torch.zeros(2, 1, config.dsa_indexer_head_dim, dtype=torch.bfloat16, device="cuda")
    q[0, 0, 0, 0] = 1
    q[0, 0, 1, 1] = 1
    k[0, 0, 0] = 1
    k[1, 0, 1] = 1
    scores, topk = fused_qk_topk_naive(q, k, weights, index_topk=1, mask=None)
    ref_scores, ref_topk = fused_qk_topk_naive(q, k, ref_weights, index_topk=1, mask=None)
    torch.testing.assert_close(scores, ref_scores, rtol=2.0e-2, atol=1.0e-2)
    assert torch.equal(topk, ref_topk)
    assert topk.item() == 1


@pytest.mark.internal
def test_fp32_weights_projection_prefers_te_gemm(monkeypatch):
    """Use TE GEMM with an explicit FP32 output when that combination is supported."""
    _skip_if_precision_unsupported("bf16")
    calls = []

    def fake_te_general_gemm(weight, x, *, out_dtype, layout):
        calls.append((out_dtype, layout))
        return (torch.mm(x.float(), weight.float().t()),)

    monkeypatch.setattr(dsa_module, "te_general_gemm", fake_te_general_gemm)
    x = torch.randn(8, 16, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(4, 16, dtype=torch.bfloat16, device="cuda")

    output, supported = dsa_module._dsa_weights_proj_forward_gemm(x, weight, None)

    assert supported is True
    assert calls == [(torch.float32, "TN")]
    torch.testing.assert_close(output, torch.mm(x.float(), weight.float().t()))


@pytest.mark.internal
@pytest.mark.parametrize(
    "message",
    [
        "CUBLAS_STATUS_NOT_SUPPORTED",
        "Unable to find suitable cuBLAS GEMM algorithm",
        "Unable to find any suitable algorithms",
    ],
)
def test_fp32_weights_projection_recognizes_te_gemm_capability_errors(message):
    assert dsa_module._dsa_weights_proj_te_gemm_is_unsupported(RuntimeError(message))


@pytest.mark.internal
@pytest.mark.parametrize(
    "message",
    ["not implemented", "unsupported dtype", "operation not supported", "CUDA out of memory"],
)
def test_fp32_weights_projection_rejects_unrelated_te_gemm_errors(message):
    assert not dsa_module._dsa_weights_proj_te_gemm_is_unsupported(RuntimeError(message))


@pytest.mark.internal
def test_fp32_weights_projection_caches_te_gemm_fallback(monkeypatch):
    """Cache an unsupported TE combination and use a genuine FP32 fallback thereafter."""
    _skip_if_precision_unsupported("bf16")
    calls = 0
    warning_calls = []

    def unsupported_te_general_gemm(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("CUBLAS_STATUS_NOT_SUPPORTED")

    monkeypatch.setattr(dsa_module, "te_general_gemm", unsupported_te_general_gemm)
    monkeypatch.setattr(
        dsa_module, "log_single_rank", lambda *args, **kwargs: warning_calls.append((args, kwargs))
    )
    monkeypatch.setattr(dsa_module, "_DSA_WEIGHTS_PROJ_TE_GEMM_FALLBACK_WARNED", False)
    x = torch.randn(8, 16, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(4, 16, dtype=torch.bfloat16, device="cuda")
    reference = torch.mm(x.float(), weight.float().t())

    output, supported = dsa_module._dsa_weights_proj_forward_gemm(x, weight, None)
    cached_output, cached_supported = dsa_module._dsa_weights_proj_forward_gemm(
        x, weight, supported
    )

    assert supported is False
    assert cached_supported is False
    assert calls == 1
    assert len(warning_calls) == 1
    torch.testing.assert_close(output, reference)
    torch.testing.assert_close(cached_output, reference)


@pytest.mark.internal
def test_fp32_weights_projection_warns_once_across_independent_fallbacks(monkeypatch):
    """Multiple indexers may probe TE independently, but fallback logging is process-global."""
    _skip_if_precision_unsupported("bf16")
    calls = 0
    warning_calls = []

    def unsupported_te_general_gemm(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("Unable to find suitable cuBLAS GEMM algorithm")

    monkeypatch.setattr(dsa_module, "te_general_gemm", unsupported_te_general_gemm)
    monkeypatch.setattr(
        dsa_module, "log_single_rank", lambda *args, **kwargs: warning_calls.append((args, kwargs))
    )
    monkeypatch.setattr(dsa_module, "_DSA_WEIGHTS_PROJ_TE_GEMM_FALLBACK_WARNED", False)
    x = torch.randn(8, 16, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(4, 16, dtype=torch.bfloat16, device="cuda")
    reference = torch.mm(x.float(), weight.float().t())

    first_output, first_supported = dsa_module._dsa_weights_proj_forward_gemm(x, weight, None)
    second_output, second_supported = dsa_module._dsa_weights_proj_forward_gemm(x, weight, None)

    assert first_supported is False
    assert second_supported is False
    assert calls == 2
    assert len(warning_calls) == 1
    torch.testing.assert_close(first_output, reference)
    torch.testing.assert_close(second_output, reference)


@pytest.mark.internal
def test_fp32_weights_projection_propagates_unrelated_te_gemm_errors(monkeypatch):
    """Do not hide TE failures that are unrelated to an unsupported GEMM capability."""
    _skip_if_precision_unsupported("bf16")
    warning_calls = []

    def failing_te_general_gemm(*args, **kwargs):
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    monkeypatch.setattr(dsa_module, "te_general_gemm", failing_te_general_gemm)
    monkeypatch.setattr(
        dsa_module, "log_single_rank", lambda *args, **kwargs: warning_calls.append((args, kwargs))
    )
    monkeypatch.setattr(dsa_module, "_DSA_WEIGHTS_PROJ_TE_GEMM_FALLBACK_WARNED", False)
    x = torch.randn(8, 16, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(4, 16, dtype=torch.bfloat16, device="cuda")

    with pytest.raises(RuntimeError, match="illegal memory access"):
        dsa_module._dsa_weights_proj_forward_gemm(x, weight, None)
    assert warning_calls == []


@pytest.mark.internal
def test_unquantized_bf16_projection_forward_backward_matches_reference():
    """The independent precision knobs must also preserve a genuine BF16 path."""
    _skip_if_precision_unsupported("bf16")
    config = _precision_config("bf16", use_quantization=False, weights_proj_output_dtype="bf16")
    indexer = _build_indexer(config)
    projection = indexer.linear_weights_proj

    x = torch.randn(
        8, 1, config.hidden_size, dtype=torch.bfloat16, device="cuda", requires_grad=True
    )
    ref_x = x.detach().clone().requires_grad_(True)
    ref_weight = projection.weight.detach().clone().requires_grad_(True)
    grad_output = torch.randn(8, 1, config.dsa_indexer_n_heads, dtype=torch.bfloat16, device="cuda")

    output = indexer._project_indexer_weights(x)
    reference = F.linear(ref_x, ref_weight)
    assert output.dtype == torch.bfloat16
    torch.testing.assert_close(output, reference, rtol=2.0e-2, atol=1.0e-2)

    output.backward(grad_output)
    reference.backward(grad_output)
    torch.testing.assert_close(x.grad, ref_x.grad, rtol=2.0e-2, atol=1.0e-2)
    torch.testing.assert_close(projection.weight.grad, ref_weight.grad, rtol=2.0e-2, atol=1.0e-2)


@pytest.mark.internal
def test_weights_projection_gradient_accumulation_and_deferred_wgrad():
    _skip_if_precision_unsupported("bf16")

    accumulation_config = _precision_config("bf16", gradient_accumulation_fusion=True)
    accumulation_config = dataclasses.replace(
        accumulation_config, disable_parameter_transpose_cache=True
    )
    accumulation_indexer = _build_indexer(accumulation_config)
    accumulation_linear = accumulation_indexer.linear_weights_proj
    accumulation_linear.weight.main_grad = torch.full_like(
        accumulation_linear.weight, 7.0, dtype=torch.float32
    )
    accumulation_linear.weight.grad_added_to_main_grad = False
    x = torch.randn(
        4,
        1,
        accumulation_config.hidden_size,
        dtype=torch.bfloat16,
        device="cuda",
        requires_grad=True,
    )
    grad_output = torch.randn(
        4, 1, accumulation_config.dsa_indexer_n_heads, dtype=torch.float32, device="cuda"
    )
    accumulation_indexer._project_indexer_weights(x).backward(grad_output)
    expected_wgrad = torch.mm(
        grad_output.reshape(-1, grad_output.size(-1)).t(), x.float().squeeze(1)
    )
    torch.testing.assert_close(accumulation_linear.weight.main_grad, expected_wgrad)
    assert accumulation_linear.weight.grad_added_to_main_grad
    accumulation_indexer._project_indexer_weights(x).backward(grad_output)
    torch.testing.assert_close(accumulation_linear.weight.main_grad, 2 * expected_wgrad)

    delayed_config = _precision_config("bf16", delay_wgrad_compute=True)
    delayed_indexer = _build_indexer(delayed_config)
    delayed_linear = delayed_indexer.linear_weights_proj
    delayed_x = x.detach().clone().requires_grad_(True)
    delayed_indexer._project_indexer_weights(delayed_x).backward(grad_output)
    assert delayed_linear.weight.grad is None
    delayed_linear.backward_dw()
    assert delayed_linear.weight.grad is not None
    torch.testing.assert_close(
        delayed_linear.weight.grad.float(), expected_wgrad, rtol=2.0e-2, atol=1.0e-2
    )


@pytest.mark.internal
def test_weights_projection_deferred_fused_wgrad_preserves_dummy_for_hook():
    _skip_if_precision_unsupported("bf16")
    config = _precision_config("bf16", delay_wgrad_compute=True, gradient_accumulation_fusion=True)
    object.__setattr__(config, "disable_parameter_transpose_cache", True)
    indexer = _build_indexer(config)
    projection = indexer.linear_weights_proj
    projection.weight.main_grad = torch.zeros_like(projection.weight, dtype=torch.float32)
    projection.weight.grad_added_to_main_grad = False
    projection.weight.zero_out_wgrad = True

    hook_calls = []

    def overlap_grad_reduce_hook():
        assert projection.weight.grad is not None
        hook_calls.append(projection.weight.grad.clone())
        projection.weight.grad = None

    projection.register_wgrad_accumulation_and_reduce_hooks(overlap_grad_reduce_hook)
    x = torch.randn(
        4, 1, config.hidden_size, dtype=torch.bfloat16, device="cuda", requires_grad=True
    )
    grad_output = torch.randn(4, 1, config.dsa_indexer_n_heads, dtype=torch.float32, device="cuda")
    indexer._project_indexer_weights(x).backward(grad_output)

    torch.testing.assert_close(
        projection.weight.main_grad, torch.zeros_like(projection.weight.main_grad)
    )
    assert not projection.weight.grad_added_to_main_grad
    torch.testing.assert_close(projection.weight.grad, torch.zeros_like(projection.weight))
    assert hook_calls == []

    projection.backward_dw()

    expected_wgrad = torch.mm(
        grad_output.reshape(-1, grad_output.size(-1)).t(), x.float().squeeze(1)
    )
    torch.testing.assert_close(projection.weight.main_grad, expected_wgrad)
    assert projection.weight.grad_added_to_main_grad
    assert len(hook_calls) == 1
    torch.testing.assert_close(hook_calls[0], torch.zeros_like(projection.weight))
    assert projection.weight.grad is None


@pytest.mark.internal
def test_weights_projection_zeroes_dummy_wgrad_when_requested():
    _skip_if_precision_unsupported("bf16")
    config = _precision_config("bf16", gradient_accumulation_fusion=True)
    config = dataclasses.replace(config, disable_parameter_transpose_cache=True)
    indexer = _build_indexer(config)
    projection = indexer.linear_weights_proj
    projection.weight.main_grad = torch.zeros_like(projection.weight, dtype=torch.float32)
    projection.weight.grad_added_to_main_grad = False
    projection.weight.zero_out_wgrad = True

    x = torch.randn(
        4, 1, config.hidden_size, dtype=torch.bfloat16, device="cuda", requires_grad=True
    )
    grad_output = torch.randn(4, 1, config.dsa_indexer_n_heads, dtype=torch.float32, device="cuda")
    indexer._project_indexer_weights(x).backward(grad_output)

    expected_wgrad = torch.mm(
        grad_output.reshape(-1, grad_output.size(-1)).t(), x.float().squeeze(1)
    )
    torch.testing.assert_close(projection.weight.main_grad, expected_wgrad)
    assert projection.weight.grad_added_to_main_grad
    torch.testing.assert_close(projection.weight.grad, torch.zeros_like(projection.weight))


@pytest.mark.internal
def test_weights_projection_uses_fsdp_main_grad_buffer():
    _skip_if_precision_unsupported("bf16")
    config = _precision_config("bf16", gradient_accumulation_fusion=True)
    indexer = _build_indexer(config)
    projection = indexer.linear_weights_proj
    fsdp_main_grad = torch.zeros_like(projection.weight, dtype=torch.float32)
    projection.weight.__fsdp_param__ = True
    projection.weight.get_main_grad = lambda: fsdp_main_grad
    projection.weight.grad_added_to_main_grad = False

    x = torch.randn(4, 1, config.hidden_size, dtype=torch.bfloat16, device="cuda")
    grad_output = torch.randn(4, 1, config.dsa_indexer_n_heads, dtype=torch.float32, device="cuda")
    indexer._project_indexer_weights(x).backward(grad_output)

    expected = torch.mm(grad_output.reshape(-1, grad_output.size(-1)).t(), x.float().squeeze(1))
    torch.testing.assert_close(fsdp_main_grad, expected)
    assert projection.weight.main_grad is fsdp_main_grad
    assert projection.weight.grad_added_to_main_grad


@pytest.mark.internal
@pytest.mark.parametrize("weights_proj_output_dtype", ["bf16", "fp32"])
def test_weights_projection_cuda_graph_smoke(weights_proj_output_dtype: str):
    _skip_if_precision_unsupported("bf16")
    config = _precision_config(
        "bf16", use_quantization=False, weights_proj_output_dtype=weights_proj_output_dtype
    )
    indexer = _build_indexer(config)
    projection = indexer.linear_weights_proj
    static_x = torch.randn(8, 1, config.hidden_size, dtype=torch.bfloat16, device="cuda")

    # Resolve the TE-GEMM capability before capture so capture itself is static.
    indexer._project_indexer_weights(static_x)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = indexer._project_indexer_weights(static_x)
    graph.replay()
    torch.cuda.synchronize()

    if weights_proj_output_dtype == "fp32":
        reference = F.linear(static_x.float(), projection.weight.float())
    else:
        reference = F.linear(static_x, projection.weight)
    assert captured.dtype == _torch_weights_proj_output_dtype(weights_proj_output_dtype)
    torch.testing.assert_close(captured, reference, rtol=2.0e-2, atol=1.0e-2)


@pytest.mark.internal
@pytest.mark.parametrize("sparse_loss", [False, True], ids=["dense-loss", "sparse-loss"])
def test_unquantized_fp32_dsa_matches_native(sparse_loss: bool):
    """Match the published FP32 weights-projection path against an independent reference."""
    run_absorbed_mla_dsa_parity(
        kernel_backend="none",
        seqlen=1024,
        attention_backend=AttnBackend.unfused,
        calculate_per_token_loss=False,
        use_sparse_loss=sparse_loss,
        num_iterations=1,
        dsa_indexer_weights_proj_use_quantization=False,
        dsa_indexer_weights_proj_output_dtype="fp32",
    )


@pytest.mark.internal
def test_unquantized_fp32_tilelang_dsa_matches_native():
    """TileLang must consume the projection's FP32 output without narrowing it."""
    run_absorbed_mla_dsa_parity(
        kernel_backend="tilelang",
        seqlen=1024,
        attention_backend=AttnBackend.auto,
        calculate_per_token_loss=False,
        use_sparse_loss=True,
        num_iterations=1,
        dsa_indexer_weights_proj_use_quantization=False,
        dsa_indexer_weights_proj_output_dtype="fp32",
    )


@pytest.mark.internal
def test_unquantized_bf16_cudnn_dsa_matches_native():
    """cuDNN must retain its BF16 weights-tensor contract when quantization is bypassed."""
    run_absorbed_mla_dsa_parity(
        kernel_backend="cudnn",
        seqlen=1024,
        attention_backend=AttnBackend.auto,
        calculate_per_token_loss=False,
        use_sparse_loss=True,
        num_iterations=1,
        dsa_indexer_weights_proj_use_quantization=False,
        dsa_indexer_weights_proj_output_dtype="bf16",
    )
