# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Numerical contract tests for the released GLM5Next standard RMSNorm."""

import os

import pytest
import torch

from megatron.core.extensions import transformer_engine
from megatron.core.transformer.glm5_next_norm import GLM5NextRMSNorm


@pytest.fixture(autouse=True)
def _bind_local_cuda_device():
    """Bind each CI worker without making CPU-only norm checks require CUDA."""
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))


def _official_glm5_next_rms_norm(
    hidden_states: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    """Transcribe the released operation order from an immutable HF revision.

    Source: transformers@93c8b7b485963a10800c91f55304db6be211c2bd,
    ``src/transformers/models/glm5_next/modeling_glm5_next.py``.
    """

    input_dtype = hidden_states.dtype
    normalized = hidden_states.float()
    variance = normalized.pow(2).mean(dim=-1, keepdim=True)
    normalized = normalized * torch.rsqrt(variance + eps)
    return weight * normalized.to(input_dtype)


def _generic_fp32_affine(
    hidden_states: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    normalized = hidden_states.float()
    variance = normalized.square().mean(dim=-1, keepdim=True)
    normalized = normalized * torch.rsqrt(variance + eps)
    return (normalized * weight.float()).to(hidden_states.dtype)


def test_forward_and_backward_match_released_bf16_rounding_contract():
    torch.manual_seed(1953)
    eps = 1e-5
    hidden_states = (torch.randn(32, 64, dtype=torch.float32) * 1.7).to(torch.bfloat16)
    weight = (1.0 + 0.2 * torch.randn(64, dtype=torch.float32)).to(torch.bfloat16)
    output_grad = torch.randn_like(hidden_states)

    actual_input = hidden_states.detach().clone().requires_grad_(True)
    module = GLM5NextRMSNorm(64, eps=eps, params_dtype=torch.bfloat16)
    with torch.no_grad():
        module.weight.copy_(weight)
    actual = module(actual_input)
    actual.backward(output_grad)

    reference_input = hidden_states.detach().clone().requires_grad_(True)
    reference_weight = weight.detach().clone().requires_grad_(True)
    expected = _official_glm5_next_rms_norm(reference_input, reference_weight, eps)
    expected.backward(output_grad)

    assert torch.equal(actual, expected)
    assert torch.equal(actual_input.grad, reference_input.grad)
    assert torch.equal(module.weight.grad, reference_weight.grad)

    # This guards the otherwise tempting Transformer Engine operation order.
    generic = _generic_fp32_affine(hidden_states, weight, eps)
    assert not torch.equal(actual.detach(), generic)


def test_sequence_parallel_metadata_and_unit_gamma_initialization():
    module = GLM5NextRMSNorm(8, eps=1e-6, sequence_parallel=True, params_dtype=torch.bfloat16)

    assert module.normalized_shape == (8,)
    assert module.eps == 1e-6
    assert module.sequence_parallel is True
    assert module.weight.sequence_parallel is True
    assert torch.equal(module.weight, torch.ones_like(module.weight))


def test_plain_state_dict_contains_only_the_checkpoint_gamma():
    """The custom VJP must not add dummy checkpoint state."""

    source = GLM5NextRMSNorm(8, params_dtype=torch.bfloat16)
    with torch.no_grad():
        source.weight.copy_(torch.arange(8, dtype=torch.bfloat16))

    state = source.state_dict()
    assert list(state) == ["weight"]

    target = GLM5NextRMSNorm(8, params_dtype=torch.bfloat16)
    incompatible = target.load_state_dict(state, strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    assert torch.equal(target.weight, source.weight)


def test_fp32_parameter_promotion_matches_released_module():
    hidden_states = torch.tensor([[0.25, -1.0, 2.0, 0.5]], dtype=torch.bfloat16)
    output_grad = torch.tensor([[0.7, -0.3, 0.2, 1.1]], dtype=torch.float32)
    module = GLM5NextRMSNorm(4, params_dtype=torch.float32)
    with torch.no_grad():
        module.weight.copy_(torch.tensor([0.9, 1.1, 1.2, 0.8]))

    actual_input = hidden_states.detach().clone().requires_grad_(True)
    actual = module(actual_input)
    actual.backward(output_grad)

    reference_input = hidden_states.detach().clone().requires_grad_(True)
    reference_weight = module.weight.detach().clone().requires_grad_(True)
    expected = _official_glm5_next_rms_norm(reference_input, reference_weight, module.eps)
    expected.backward(output_grad)

    assert actual.dtype == torch.float32
    assert torch.equal(actual, expected)
    assert torch.equal(actual_input.grad, reference_input.grad)
    assert torch.equal(module.weight.grad, reference_weight.grad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("width", [128, 512, 1536, 4096])
def test_exact_vjp_matches_released_widths_on_cuda(width):
    """Lock forward, input-grad, and gamma-grad at production hidden widths."""

    torch.manual_seed(5353 + width)
    device = torch.device("cuda")
    eps = 1e-5
    hidden_states = torch.randn(3, 17, width, device=device, dtype=torch.bfloat16)
    weight = (1.0 + 0.2 * torch.randn(width, device=device, dtype=torch.float32)).to(torch.bfloat16)
    output_grad = torch.randn_like(hidden_states)

    actual_input = hidden_states.detach().clone().requires_grad_(True)
    module = GLM5NextRMSNorm(width, eps=eps, params_dtype=torch.bfloat16, device=device)
    with torch.no_grad():
        module.weight.copy_(weight)
    actual = module(actual_input)
    actual.backward(output_grad)

    reference_input = hidden_states.detach().clone().requires_grad_(True)
    reference_weight = weight.detach().clone().requires_grad_(True)
    expected = _official_glm5_next_rms_norm(reference_input, reference_weight, eps)
    expected.backward(output_grad)

    assert torch.equal(actual, expected)
    assert torch.equal(actual_input.grad, reference_input.grad)
    assert torch.equal(module.weight.grad, reference_weight.grad)


def test_meta_materialization_reset_restores_parameter_contract():
    module = GLM5NextRMSNorm(8, device="meta", params_dtype=torch.bfloat16, sequence_parallel=True)
    assert module.weight.is_meta

    module.to_empty(device="cpu")
    module.reset_parameters()

    assert not module.weight.is_meta
    assert module.weight.dtype == torch.bfloat16
    assert module.weight.sequence_parallel is True
    assert torch.equal(module.weight, torch.ones_like(module.weight))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA graph capture")
@pytest.mark.parametrize("width", [128, 512, 1536, 4096])
def test_cuda_graph_training_replay_preserves_exact_rounding(width):
    torch.manual_seed(1954 + width)
    device = torch.device("cuda")
    module = GLM5NextRMSNorm(width, params_dtype=torch.bfloat16, device=device)
    with torch.no_grad():
        module.weight.copy_((1.0 + 0.2 * torch.randn(width, device=device)).to(torch.bfloat16))
    static_input = torch.randn(
        3, 17, width, device=device, dtype=torch.bfloat16, requires_grad=True
    )
    static_output_grad = torch.randn_like(static_input)

    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            warmup_output = module(static_input)
            torch.autograd.grad(warmup_output, (static_input, module.weight), static_output_grad)
    torch.cuda.current_stream().wait_stream(warmup_stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = module(static_input)
        captured_input_grad, captured_weight_grad = torch.autograd.grad(
            captured, (static_input, module.weight), static_output_grad
        )

    replay_input = torch.randn_like(static_input)
    replay_output_grad = torch.randn_like(static_output_grad)
    static_input.data.copy_(replay_input)
    static_output_grad.copy_(replay_output_grad)
    graph.replay()
    torch.cuda.synchronize()

    reference_input = replay_input.detach().clone().requires_grad_(True)
    reference_weight = module.weight.detach().clone().requires_grad_(True)
    expected = _official_glm5_next_rms_norm(reference_input, reference_weight, module.eps)
    expected_input_grad, expected_weight_grad = torch.autograd.grad(
        expected, (reference_input, reference_weight), replay_output_grad
    )

    assert torch.equal(captured, expected)
    assert torch.equal(captured_input_grad, expected_input_grad)
    assert torch.equal(captured_weight_grad, expected_weight_grad)


def _te_config(**overrides):
    values = {
        "experimental_attention_variant": "dsa",
        "rmsnorm_cast_before_weight": True,
        "fused_residual_rmsnorm": False,
        "init_model_with_meta_device": False,
        "layernorm_zero_centered_gamma": False,
        "normalization": "RMSNorm",
        "params_dtype": torch.bfloat16,
        "sequence_parallel": True,
        "use_cpu_initialization": True,
    }
    values.update(overrides)
    return type("Config", (), values)()


def test_te_norm_factory_selects_exact_glm5_next_implementation(monkeypatch):
    # TENorm normally requires TE because the surrounding GLM stack uses TE
    # linears. Patching only its availability/version isolates this selection
    # contract from the local test runner's optional compiled TE installation.
    monkeypatch.setattr(transformer_engine, "HAVE_TE", True)
    monkeypatch.setattr(transformer_engine, "is_te_min_version", lambda _: True)

    module = transformer_engine.TENorm(_te_config(), 64, eps=1e-5)

    assert isinstance(module, GLM5NextRMSNorm)
    assert module.weight.device.type == "cpu"
    assert module.weight.dtype == torch.bfloat16
    assert module.weight.sequence_parallel is True


def test_te_norm_factory_rejects_fused_residual_for_glm5_next(monkeypatch):
    monkeypatch.setattr(transformer_engine, "HAVE_TE", True)
    with pytest.raises(ValueError, match="does not support fused residual RMSNorm"):
        transformer_engine.TENorm(_te_config(fused_residual_rmsnorm=True), 64, has_residual=True)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"normalized_shape": 0}, "positive dimensions"),
        ({"normalized_shape": 8, "eps": 0.0}, "finite and positive"),
        ({"normalized_shape": 8, "zero_centered_gamma": True}, "zero-centered gamma"),
    ],
)
def test_invalid_contracts_fail_closed(kwargs, match):
    with pytest.raises(ValueError, match=match):
        GLM5NextRMSNorm(**kwargs)


def test_input_shape_must_end_in_normalized_shape():
    module = GLM5NextRMSNorm(8)
    with pytest.raises(ValueError, match="does not end in normalized_shape"):
        module(torch.ones(2, 7))
