# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import argparse
import math
from collections.abc import Generator

import pytest
import torch

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_observation import capture_tensor_observations
from megatron.core.transformer.moe.moe_utils import (
    compute_normalized_router_scores,
    fused_topk_with_score_function,
)
from megatron.core.transformer.moe.router import InferenceTopKRouter, TopKRouter
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.arguments import _add_network_size_args
from tests.unit_tests.test_utilities import Utils


@pytest.fixture
def pg_collection() -> Generator[ProcessGroupCollection, None, None]:
    """Use real process groups for router and auxiliary-loss reductions."""
    Utils.initialize_model_parallel(1, 1)
    yield ProcessGroupCollection.use_mpu_process_groups()
    Utils.destroy_model_parallel()


def _config(**overrides) -> TransformerConfig:
    kwargs = dict(
        num_layers=1,
        hidden_size=32,
        num_attention_heads=4,
        num_moe_experts=8,
        moe_router_topk=2,
        moe_router_score_function="sigmoid",
        moe_aux_loss_coeff=0,
        bf16=True,
        params_dtype=torch.bfloat16,
        moe_router_dtype="fp32",
        add_bias_linear=True,
    )
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


@pytest.mark.parametrize(
    "overrides",
    [
        dict(moe_router_score_function="softmax"),
        dict(moe_router_score_function="sqrtsoftplus"),
        dict(num_moe_experts=None),
        dict(moe_router_load_balancing_type="sinkhorn"),
    ],
)
def test_unsupported_router(overrides: dict) -> None:
    """Input scaling requires sigmoid MoE routing without Sinkhorn."""
    with pytest.raises(ValueError, match="moe_router_sigmoid_input_scaling"):
        _config(moe_router_sigmoid_input_scaling=True, **overrides)


def test_cli_argument() -> None:
    """The generated training CLI exposes the new opt-in flag."""
    parser = _add_network_size_args(argparse.ArgumentParser())
    assert not parser.parse_args([]).moe_router_sigmoid_input_scaling
    assert parser.parse_args(
        ["--moe-router-sigmoid-input-scaling"]
    ).moe_router_sigmoid_input_scaling


@pytest.mark.parametrize("hidden_size,num_experts", [(32, 8), (8, 16), (8192, 256)])
@pytest.mark.parametrize("router_dtype", [None, "fp32", "fp64"])
def test_gate_scaling(
    pg_collection: ProcessGroupCollection,
    hidden_size: int,
    num_experts: int,
    router_dtype: str | None,
) -> None:
    """Derive the scale from both dimensions and scale gate bias with the logits."""
    config = _config(
        hidden_size=hidden_size, num_moe_experts=num_experts, moe_router_dtype=router_dtype
    )
    router = TopKRouter(config, pg_collection).cuda()
    inputs = torch.randn(2, 1, hidden_size, device="cuda", dtype=torch.bfloat16)
    original = router.gating(inputs)
    scaled = TopKRouter(
        _config(
            hidden_size=hidden_size,
            num_moe_experts=num_experts,
            moe_router_dtype=router_dtype,
            moe_router_sigmoid_input_scaling=True,
        ),
        pg_collection,
    ).cuda()
    scaled.load_state_dict(router.state_dict())
    expected = original.float() if original.dtype == torch.bfloat16 else original
    torch.testing.assert_close(
        scaled.gating(inputs), expected * math.sqrt(num_experts / hidden_size)
    )
    assert set(scaled.state_dict()) == set(router.state_dict()) == {"weight", "bias"}


@pytest.mark.parametrize("aux_loss_coeff", [0.0, 0.01])
@pytest.mark.parametrize("fused", [False, True])
def test_forward_backward_and_observations(
    pg_collection: ProcessGroupCollection, aux_loss_coeff: float, fused: bool
) -> None:
    """Match explicit logit scaling, including expert bias, aux gradients and fused top-k."""
    if fused and fused_topk_with_score_function is None:
        pytest.skip("TE fused router ops not available")
    router = TopKRouter(
        _config(
            moe_router_sigmoid_input_scaling=True,
            moe_aux_loss_coeff=aux_loss_coeff,
            moe_router_fusion=fused,
            moe_router_enable_expert_bias=True,
        ),
        pg_collection,
    ).cuda()
    reference = TopKRouter(
        _config(moe_aux_loss_coeff=aux_loss_coeff, moe_router_enable_expert_bias=True),
        pg_collection,
    ).cuda()
    with torch.no_grad():
        router.bias.fill_(0.25)
        router.expert_bias.copy_(torch.linspace(-0.03, 0.03, 8, device="cuda"))
    reference.load_state_dict(router.state_dict())
    router.set_layer_number(1)
    reference.set_layer_number(1)
    inputs = torch.randn(16, 2, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    reference_inputs = inputs.detach().clone().requires_grad_(True)
    reference_logits = reference.gating(reference_inputs) * math.sqrt(8 / 32)
    expected_probs, expected_map = reference.routing(reference_logits)
    observed = {}

    def record(owner, name, kind, value, tp_shard_dim, sequence_dim, batch_dim):
        observed[name] = value.detach().clone()

    with capture_tensor_observations(record, frozenset({"router_logits", "router_scores"})):
        probs, routing_map = router(inputs)

    torch.testing.assert_close(probs, expected_probs, rtol=1e-5, atol=1e-6)
    assert torch.equal(routing_map, expected_map)
    torch.testing.assert_close(observed["router_logits"], reference_logits)
    torch.testing.assert_close(
        observed["router_scores"], compute_normalized_router_scores(reference_logits, "sigmoid")
    )
    upstream = torch.randn_like(probs)
    (probs * upstream).sum().backward()
    (expected_probs * upstream).sum().backward()
    torch.testing.assert_close(inputs.grad, reference_inputs.grad, rtol=0.02, atol=5e-4)
    torch.testing.assert_close(router.weight.grad, reference.weight.grad, rtol=0.02, atol=5e-4)
    torch.testing.assert_close(router.bias.grad, reference.bias.grad, rtol=0.02, atol=5e-4)


def test_inference_matches_training(pg_collection: ProcessGroupCollection) -> None:
    """Dense inference and sparse training routing use the same fixed scale."""
    router = InferenceTopKRouter(
        _config(moe_router_sigmoid_input_scaling=True), pg_collection
    ).cuda()
    inputs = torch.randn(16, 1, 32, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        probs, routing_map = router(inputs)
        dense_probs, indices = router._forward(inputs)
    torch.testing.assert_close(dense_probs, probs.gather(1, indices))
    assert torch.equal(torch.zeros_like(routing_map).scatter_(1, indices, True), routing_map)
