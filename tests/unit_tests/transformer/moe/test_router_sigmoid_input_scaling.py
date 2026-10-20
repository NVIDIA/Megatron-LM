# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import argparse
from collections.abc import Generator
from types import SimpleNamespace
from unittest import mock

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
from megatron.training.checkpointing import CheckpointType, load_args_from_checkpoint
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
    with pytest.raises(ValueError, match="moe_router_sigmoid_input_scaling_factor"):
        _config(moe_router_sigmoid_input_scaling_factor=0.612, **overrides)


def test_cli_argument() -> None:
    """The generated training CLI accepts an explicit numeric multiplier."""
    parser = _add_network_size_args(argparse.ArgumentParser())
    assert parser.parse_args([]).moe_router_sigmoid_input_scaling_factor == 1.0
    assert (
        parser.parse_args(
            ["--moe-router-sigmoid-input-scaling-factor", "0.612"]
        ).moe_router_sigmoid_input_scaling_factor
        == 0.612
    )


@pytest.mark.parametrize("factor", [0.0, -0.5, float("nan"), float("inf"), -float("inf")])
def test_invalid_factor(factor: float) -> None:
    """Reject gains that cannot define a positive finite sigmoid input scale."""
    with pytest.raises(ValueError, match="must be positive and finite"):
        _config(moe_router_sigmoid_input_scaling_factor=factor)


@pytest.mark.parametrize("factor", [0.612, 1.0, 1.5])
@pytest.mark.parametrize("hidden_size,num_experts", [(32, 8), (8, 16), (8192, 256)])
@pytest.mark.parametrize("router_dtype", [None, "fp32", "fp64"])
def test_gate_scaling(
    pg_collection: ProcessGroupCollection,
    hidden_size: int,
    num_experts: int,
    router_dtype: str | None,
    factor: float,
) -> None:
    """Use the requested factor at any dimensions, including identity dtype preservation."""
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
            moe_router_sigmoid_input_scaling_factor=factor,
        ),
        pg_collection,
    ).cuda()
    scaled.load_state_dict(router.state_dict())
    actual = scaled.gating(inputs)
    expected = original
    if factor != 1.0:
        if expected.dtype == torch.bfloat16:
            expected = expected.float()
        expected = expected * factor
    assert actual.dtype == expected.dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
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
            moe_router_sigmoid_input_scaling_factor=0.612,
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
    reference_logits = reference.gating(reference_inputs) * 0.612
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
        _config(moe_router_sigmoid_input_scaling_factor=0.612), pg_collection
    ).cuda()
    inputs = torch.randn(16, 1, 32, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        probs, routing_map = router(inputs)
        dense_probs, indices = router._forward(inputs)
    torch.testing.assert_close(dense_probs, probs.gather(1, indices))
    assert torch.equal(torch.zeros_like(routing_map).scatter_(1, indices, True), routing_map)


@pytest.mark.parametrize(
    "saved_factor,runtime_factor,expected",
    [(0.612, 1.0, 0.612), (1.0, 0.612, 1.0), (None, 1.0, 1.0), (None, 1.5, 1.5)],
)
def test_checkpoint_argument(
    saved_factor: float | None, runtime_factor: float, expected: float
) -> None:
    """Restore the saved factor; checkpoints without the argument keep the runtime value."""
    saved = SimpleNamespace()
    if saved_factor is not None:
        saved.moe_router_sigmoid_input_scaling_factor = saved_factor
    args = SimpleNamespace(
        load="checkpoint",
        iteration=0,
        moe_router_sigmoid_input_scaling_factor=runtime_factor,
        use_tokenizer_model_from_checkpoint_args=False,
        use_mp_args_from_checkpoint_args=False,
    )
    with mock.patch(
        "megatron.training.checkpointing._load_base_checkpoint",
        return_value=({"args": saved, "iteration": 12}, "checkpoint", False, CheckpointType.LEGACY),
    ):
        restored_args, _ = load_args_from_checkpoint(args)
    assert restored_args.moe_router_sigmoid_input_scaling_factor == expected
