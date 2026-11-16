# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Hidden-state fusion behavior with native Transformer Engine modules."""

from dataclasses import replace
from functools import partial

import pytest
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from megatron.core.models.hybrid.hidden_state_feedback.specs import (
    additive_hidden_state_fusion_spec,
    gated_hidden_state_fusion_spec,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


@pytest.fixture
def fusion_pg():
    """Use real process groups and the CUDA RNG tracker required by TE projections."""
    Utils.initialize_model_parallel(1, 1)
    torch.manual_seed(123)
    model_parallel_cuda_manual_seed(123)
    allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with torch.backends.cudnn.flags(allow_tf32=False):
            yield ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "dp_cp"])
    finally:
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        Utils.destroy_model_parallel()


def _config(hidden_size=64, **overrides):
    kwargs = dict(
        num_layers=1,
        hidden_size=hidden_size,
        num_attention_heads=1,
        normalization="RMSNorm",
        layernorm_epsilon=1e-6,
        use_cpu_initialization=True,
        gradient_accumulation_fusion=False,
        delay_wgrad_compute=False,
    )
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


def _fusion(spec, pg_collection, hidden_size=64, **config_overrides):
    config = _config(hidden_size, **config_overrides)
    fusion = build_module(spec, config=config, pg_collection=pg_collection)
    return fusion


def _with_params(spec, **params):
    return replace(spec, params={**spec.params, **params})


def _assert_matches_reference(actual, expected, name):
    """Compare a TE result with the FP32 reference at the reference tensor's scale.

    TE requests TF32 for FP32 GEMMs regardless of PyTorch's TF32 flags. That rounding
    error scales with the GEMM operands, not with each result element, so near-zero
    elements of a cancelling sum, such as a weight gradient over a few tokens, need an
    absolute tolerance tied to the tensor's magnitude. The relative L2 bound also
    limits aggregate error, which a large reference element could otherwise hide.
    """
    torch.testing.assert_close(
        actual,
        expected,
        rtol=3e-3,
        atol=3e-3 * expected.abs().max().item(),
        msg=lambda message: f"{name}: {message}",
    )
    error_norm = (actual.detach() - expected.detach()).norm().item()
    allowed_norm = 3e-3 * expected.detach().norm().item()
    assert (
        error_norm <= allowed_norm
    ), f"{name}: L2 error {error_norm:.6g} exceeds allowed {allowed_norm:.6g}"


@pytest.mark.parametrize(
    "spec",
    [additive_hidden_state_fusion_spec, gated_hidden_state_fusion_spec],
    ids=["additive", "gated"],
)
def test_fusion_rejects_broadcast_inputs(spec, fusion_pg):
    fusion = _fusion(spec, fusion_pg).cuda()
    embedding = torch.randn(3, 2, 64, device="cuda")
    hidden = torch.randn_like(embedding)
    for invalid_embedding, invalid_hidden in (
        (embedding, hidden[:1]),
        (embedding[0], hidden[0]),
        (embedding[..., :-1], hidden[..., :-1]),
    ):
        with pytest.raises(ValueError, match="matching"):
            fusion(invalid_embedding, invalid_hidden)


@pytest.mark.parametrize("bias", [False, True])
def test_additive_zero_init_learns_feedback_after_optimizer_step(bias, fusion_pg):
    fusion = _fusion(_with_params(additive_hidden_state_fusion_spec, bias=bias), fusion_pg).cuda()
    optimizer = torch.optim.SGD(fusion.parameters(), lr=0.01)
    embedding = torch.randn(3, 2, 64, device="cuda", requires_grad=True)
    hidden = torch.randn_like(embedding, requires_grad=True)
    probe = torch.randn_like(embedding)
    actual = fusion(embedding, hidden)
    torch.testing.assert_close(actual, embedding, rtol=0, atol=0)
    (actual * probe).sum().backward()
    assert fusion.hidden_state_projection.weight.grad.abs().sum() > 0
    torch.testing.assert_close(hidden.grad, torch.zeros_like(hidden), rtol=0, atol=0)
    torch.testing.assert_close(
        fusion.hidden_state_norm.weight.grad,
        torch.zeros_like(fusion.hidden_state_norm.weight),
        rtol=0,
        atol=0,
    )
    assert fusion.hidden_state_projection.weight.use_muon
    assert fusion.hidden_state_norm.weight.is_rmsnorm_weight
    assert fusion.hidden_state_norm.weight.is_hidden_state_feedback_parameter
    assert not fusion.hidden_state_norm.weight.use_muon

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    embedding.grad = hidden.grad = None
    updated = fusion(embedding, hidden)
    assert not torch.equal(updated, embedding)
    (updated * probe).sum().backward()
    for gradient in (hidden.grad, fusion.hidden_state_norm.weight.grad):
        assert gradient is not None and torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0


def test_additive_uses_configured_rmsnorm_epsilon(fusion_pg):
    fusion = _fusion(
        additive_hidden_state_fusion_spec,
        fusion_pg,
        layernorm_epsilon=1 / 64,
        layernorm_zero_centered_gamma=False,
    ).cuda()
    with torch.no_grad():
        fusion.hidden_state_norm.weight.fill_(1.0)
        fusion.hidden_state_projection.weight.copy_(torch.eye(64, device="cuda"))
    embedding = torch.zeros(3, 2, 64, device="cuda")
    hidden = torch.full_like(embedding, 1 / 8)
    # mean(hidden**2) and epsilon are both 1/64, so the normalized value is 1/sqrt(2).
    # Ignoring the configured epsilon instead gives approximately 1.
    expected = torch.full_like(hidden, 2**-0.5)
    _assert_matches_reference(fusion(embedding, hidden), expected, "epsilon-sensitive output")


@pytest.mark.parametrize(
    "gated,normalization,zero_centered_gamma,bias",
    [
        (False, "RMSNorm", False, False),
        (False, "LayerNorm", True, True),
        (True, "RMSNorm", True, False),
        (True, "LayerNorm", False, False),
    ],
    ids=["additive-rms", "additive-layer-bias", "gated-rms", "gated-layer"],
)
def test_fusion_matches_functional_reference(
    gated, normalization, zero_centered_gamma, bias, fusion_pg
):
    spec = (
        gated_hidden_state_fusion_spec
        if gated
        else _with_params(additive_hidden_state_fusion_spec, bias=bias)
    )
    fusion = _fusion(
        spec,
        fusion_pg,
        normalization=normalization,
        layernorm_epsilon=0.01,
        layernorm_zero_centered_gamma=zero_centered_gamma,
    ).cuda()
    # Exercise learned values, including nonzero additive feedback and LayerNorm biases.
    with torch.no_grad():
        for name, parameter in fusion.named_parameters():
            if parameter.ndim == 2:
                parameter.normal_(std=0.1)
            elif name.endswith("weight"):
                parameter.copy_(
                    torch.linspace(0.5, 1.5, 64, device="cuda") - int(zero_centered_gamma)
                )
            else:
                parameter.copy_(torch.linspace(-0.2, 0.2, 64, device="cuda"))
    embedding = (torch.randn(3, 2, 64, device="cuda") + 0.7).requires_grad_()
    hidden = (0.7 * torch.randn_like(embedding) - 0.4).requires_grad_()
    reference_embedding = embedding.detach().clone().requires_grad_()
    reference_hidden = hidden.detach().clone().requires_grad_()
    parameters = dict(fusion.named_parameters())
    reference_parameters = {
        name: parameter.detach().clone().requires_grad_() for name, parameter in parameters.items()
    }

    def norm(value, prefix):
        weight = reference_parameters[f"{prefix}.weight"] + int(zero_centered_gamma)
        if normalization == "LayerNorm":
            return F.layer_norm(
                value, (64,), weight, reference_parameters[f"{prefix}.bias"], eps=0.01
            )
        return value * (value.square().mean(-1, keepdim=True) + 0.01).rsqrt() * weight

    def linear(value, prefix):
        return F.linear(
            value,
            reference_parameters[f"{prefix}.weight"],
            reference_parameters.get(f"{prefix}.bias"),
        )

    if gated:
        gate = torch.sigmoid(
            linear(norm(reference_embedding, "token_embedding_norm"), "gate_projection")
        )
        expected = norm(linear(reference_hidden, "hidden_state_projection") * gate, "output_norm")
    else:
        expected = reference_embedding + linear(
            norm(reference_hidden, "hidden_state_norm"), "hidden_state_projection"
        )
    actual = fusion(embedding, hidden)
    probe = torch.randn_like(actual)
    expected_gradients = torch.autograd.grad(
        (expected * probe).sum(),
        (reference_embedding, reference_hidden, *reference_parameters.values()),
    )
    actual_gradients = torch.autograd.grad(
        (actual * probe).sum(), (embedding, hidden, *parameters.values())
    )
    _assert_matches_reference(actual, expected, "output")
    for name, actual_gradient, expected_gradient in zip(
        ("embedding", "hidden", *parameters), actual_gradients, expected_gradients
    ):
        assert torch.isfinite(actual_gradient).all(), name
        assert actual_gradient.abs().sum() > 0, name
        _assert_matches_reference(actual_gradient, expected_gradient, name)


@pytest.mark.parametrize(
    "spec",
    [additive_hidden_state_fusion_spec, gated_hidden_state_fusion_spec],
    ids=["additive", "gated"],
)
def test_fusion_checkpoint_round_trip(spec, tmp_path, fusion_pg):
    original = _fusion(spec, fusion_pg, normalization="LayerNorm").cuda()
    # Include learned norm weights/biases and a nonzero additive projection.
    with torch.no_grad():
        for parameter in original.parameters():
            parameter.add_(0.01 * torch.randn_like(parameter))
    embedding = torch.randn(3, 2, 64, device="cuda")
    hidden = torch.randn_like(embedding)
    expected = original(embedding, hidden)
    path = tmp_path / "feedback.pt"
    torch.save(original.state_dict_for_save_checkpoint(), path)
    restored = _fusion(spec, fusion_pg, normalization="LayerNorm").cuda()
    restored.load_state_dict(torch.load(path, weights_only=True))
    restored_parameters = dict(restored.named_parameters())
    for name, parameter in original.named_parameters():
        torch.testing.assert_close(restored_parameters[name], parameter, rtol=0, atol=0)
    torch.testing.assert_close(restored(embedding, hidden), expected, rtol=0, atol=0)

    sharded = restored.sharded_state_dict(
        prefix="feedback.", metadata={"dp_cp_group": fusion_pg.dp_cp}
    )
    for name, parameter in restored.named_parameters():
        entry = sharded[f"feedback.{name}"]
        assert entry.data is parameter
        assert entry.global_shape == parameter.shape


def test_te_projections_use_configured_initializer_dtype_and_checkpoint_group(fusion_pg):
    fusion = _fusion(
        gated_hidden_state_fusion_spec,
        fusion_pg,
        bf16=True,
        params_dtype=torch.bfloat16,
        init_method=partial(torch.nn.init.constant_, val=0.125),
    )
    for projection in (fusion.hidden_state_projection, fusion.gate_projection):
        assert projection.weight.device.type == "cpu"
        assert projection.weight.dtype == torch.bfloat16
        torch.testing.assert_close(
            projection.weight, torch.full_like(projection.weight, 0.125), rtol=0, atol=0
        )
        assert projection._tp_group is fusion_pg.tp
        assert projection.weight.tensor_model_parallel is False
        assert projection.weight.allreduce


def test_additive_projection_uses_config_initializer_without_zero_init(fusion_pg):
    spec = _with_params(additive_hidden_state_fusion_spec, zero_init=False)
    fusion = _fusion(spec, fusion_pg, init_method=partial(torch.nn.init.constant_, val=0.125))
    weight = fusion.hidden_state_projection.weight
    torch.testing.assert_close(weight, torch.full_like(weight, 0.125), rtol=0, atol=0)


@pytest.mark.parametrize("recompute", [False, True])
def test_reused_te_projections_accumulate_gradients(recompute, fusion_pg):
    # Enable first-microbatch tracking, but run outside FP8 autocast: this tests
    # gradient accumulation with native FP32 kernels, not FP8 numerical behavior.
    options = dict(fp8="hybrid", disable_parameter_transpose_cache=False)
    reference = _fusion(gated_hidden_state_fusion_spec, fusion_pg, **options).cuda()
    actual = _fusion(
        gated_hidden_state_fusion_spec, fusion_pg, gradient_accumulation_fusion=True, **options
    ).cuda()
    actual.load_state_dict(reference.state_dict())
    for projection in (actual.hidden_state_projection, actual.gate_projection):
        projection.weight.main_grad = torch.zeros_like(projection.weight, dtype=torch.float32)
        projection.weight.grad_added_to_main_grad = False
        projection.weight.zero_out_wgrad = True
    embeddings = torch.randn(4, 2, 64, device="cuda")
    hidden_states = torch.randn_like(embeddings)
    probe = torch.randn_like(embeddings)

    def run(fusion, checkpoint_second_call):
        embedding = embeddings.detach().clone().requires_grad_(True)
        hidden = hidden_states.detach().clone().requires_grad_(True)
        first = fusion(embedding, hidden)
        second_inputs = (0.5 * embedding, 1.5 * hidden + 0.1 * first)
        second = (
            checkpoint(fusion, *second_inputs, use_reentrant=True)
            if checkpoint_second_call
            else fusion(*second_inputs)
        )
        loss = ((first + 0.3 * second) * probe).sum()
        loss.backward()
        return loss.detach(), embedding.grad, hidden.grad

    expected = run(reference, False)
    result = run(actual, recompute)
    for actual_value, expected_value in zip(result, expected):
        torch.testing.assert_close(actual_value, expected_value, rtol=1e-4, atol=1e-5)
    reference_parameters = dict(reference.named_parameters())
    for name, parameter in actual.named_parameters():
        gradient = parameter.main_grad if parameter.ndim == 2 else parameter.grad
        assert gradient is not None and torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0
        torch.testing.assert_close(gradient, reference_parameters[name].grad, rtol=1e-4, atol=1e-5)
