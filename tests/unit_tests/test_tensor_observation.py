# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

import megatron.core.transformer.moe.router as router_mod
from megatron.core import parallel_state, tensor_parallel
from megatron.core.extensions.transformer_engine import HAVE_TE, te_checkpoint
from megatron.core.tensor_observation import (
    capture_tensor_observations,
    observe_layer_residuals,
    observe_tensor,
)
from megatron.core.transformer.moe.router import TopKRouter
from tests.unit_tests.test_utilities import Utils


def test_tensor_observation_scope_filters_source_kinds_and_restores_noop():
    owner = object()
    observed = []

    def observer(*args):
        observed.append(args)

    with capture_tensor_observations(observer, frozenset({"output_logits"})):
        observe_tensor(owner, "output_logits", "output_logits", torch.tensor([1.0]))
        observe_tensor(owner, "router_logits", "router_logits", torch.tensor([2.0]))
    observe_tensor(owner, "output_logits", "output_logits", torch.tensor([3.0]))

    assert len(observed) == 1
    assert observed[0][:3] == (owner, "output_logits", "output_logits")
    torch.testing.assert_close(observed[0][3], torch.tensor([1.0]))
    assert observed[0][4] is None


def test_layer_residual_observation_separates_accumulator_and_net_contribution():
    layer = SimpleNamespace(config=SimpleNamespace(sequence_parallel=True))
    accumulator = torch.tensor([1.0, 2.0], requires_grad=True)
    output = accumulator + torch.tensor([3.0, 5.0])
    observed = []

    with capture_tensor_observations(
        lambda *args: observed.append(args),
        frozenset({"residual_accumulator", "residual_contribution"}),
    ):
        observe_layer_residuals(layer, accumulator, output)

    assert [observation[2] for observation in observed] == [
        "residual_accumulator",
        "residual_contribution",
    ]
    torch.testing.assert_close(observed[0][3], torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(observed[1][3], torch.tensor([3.0, 5.0]))
    assert observed[0][4] == 0
    assert observed[1][4] == 0
    assert not observed[1][3].requires_grad


def test_layer_residual_observation_includes_no_grad_forward():
    layer = SimpleNamespace(config=SimpleNamespace(sequence_parallel=False))
    observed = []

    with capture_tensor_observations(
        lambda *args: observed.append(args), frozenset({"residual_accumulator"})
    ):
        with torch.no_grad():
            observe_layer_residuals(layer, torch.tensor([1.0]), torch.tensor([2.0]))

    assert len(observed) == 1
    torch.testing.assert_close(observed[0][3], torch.tensor([1.0]))


def test_tensor_observation_observes_checkpoint_forward_not_recomputation():
    owner = object()
    executions = []
    observed = []

    def checkpointed(value):
        executions.append(torch.is_grad_enabled())
        doubled = value * 2.0
        observe_tensor(owner, "activation", "activation", doubled)
        return doubled.square()

    value = torch.tensor([1.0, 2.0], device="cuda", requires_grad=True)
    with capture_tensor_observations(
        lambda *args: observed.append((torch.is_grad_enabled(), args)), frozenset({"activation"})
    ):
        output = tensor_parallel.checkpoint(checkpointed, False, value)
        output.sum().backward()

    assert executions == [False, True]
    assert len(observed) == 1
    observed_with_grad, observation = observed[0]
    assert not observed_with_grad
    assert observation[:3] == (owner, "activation", "activation")
    torch.testing.assert_close(observation[3], value.detach() * 2.0)
    torch.testing.assert_close(value.grad, torch.tensor([8.0, 16.0], device="cuda"))


@pytest.mark.skipif(not HAVE_TE, reason="Transformer Engine not available")
def test_tensor_observation_observes_te_checkpoint_forward_not_recomputation():
    owner = object()
    executions = []
    observed = []

    def checkpointed(value):
        executions.append(torch.is_grad_enabled())
        doubled = value * 2.0
        observe_tensor(owner, "activation", "activation", doubled)
        return doubled.square()

    Utils.initialize_model_parallel()
    try:
        value = torch.tensor([1.0, 2.0], device="cuda", requires_grad=True)
        with capture_tensor_observations(
            lambda *args: observed.append((torch.is_grad_enabled(), args)),
            frozenset({"activation"}),
        ):
            output = te_checkpoint(
                checkpointed,
                False,
                tensor_parallel.random.get_cuda_rng_tracker,
                parallel_state.get_tensor_model_parallel_group(),
                value,
            )
            output.sum().backward()

        assert executions == [False, True]
        assert len(observed) == 1
        observed_with_grad, observation = observed[0]
        assert not observed_with_grad
        assert observation[:3] == (owner, "activation", "activation")
        torch.testing.assert_close(observation[3], value.detach() * 2.0)
        torch.testing.assert_close(value.grad, torch.tensor([8.0, 16.0], device="cuda"))
    finally:
        Utils.destroy_model_parallel()


def test_tensor_observation_observes_checkpoint_without_output_forward_not_recomputation():
    owner = object()
    executions = []
    observed = []
    checkpoint = tensor_parallel.CheckpointWithoutOutput(fp8=None)

    def checkpointed(value):
        executions.append(torch.is_grad_enabled())
        doubled = value * 2.0
        observe_tensor(owner, "activation", "activation", doubled)
        return doubled

    value = torch.tensor([1.0, 2.0], device="cuda", requires_grad=True)
    with capture_tensor_observations(
        lambda *args: observed.append((torch.is_grad_enabled(), args[3].detach().clone())),
        frozenset({"activation"}),
    ):
        checkpointed_value = checkpoint.checkpoint(checkpointed, value)
        output = checkpointed_value * value
        checkpoint.discard_output_and_register_recompute(output)
        output.sum().backward()

    assert executions == [False, True]
    assert len(observed) == 1
    observed_with_grad, observed_value = observed[0]
    assert not observed_with_grad
    torch.testing.assert_close(observed_value, value.detach() * 2.0)
    torch.testing.assert_close(value.grad, torch.tensor([4.0, 8.0], device="cuda"))


def test_router_observes_raw_logits_before_forced_benchmark_routing(monkeypatch):
    router = TopKRouter.__new__(TopKRouter)
    torch.nn.Module.__init__(router)
    router.config = SimpleNamespace(
        sequence_parallel=True, moe_router_force_load_balancing=True, moe_router_force_biased=None
    )
    router.layer_number = 1
    router._maintain_float32_expert_bias = lambda: None
    router.apply_input_jitter = lambda tensor: tensor
    raw_logits = torch.tensor([[1.0, 2.0]])
    router.gating = lambda tensor: raw_logits
    routed_logits = []

    def routing(logits, padding_mask=None):
        routed_logits.append(logits)
        return logits, torch.ones_like(logits, dtype=torch.bool)

    router.routing = routing
    monkeypatch.setattr(router_mod, "apply_random_logits", lambda logits: logits + 100.0)
    observed = []

    with capture_tensor_observations(
        lambda *args: observed.append(args), frozenset({"router_logits"})
    ):
        router(torch.ones(1, 1))

    assert len(observed) == 1
    assert observed[0][:3] == (router, "router_logits", "router_logits")
    torch.testing.assert_close(observed[0][3], raw_logits)
    assert observed[0][4] == 0
    torch.testing.assert_close(routed_logits[0], raw_logits + 100.0)


def test_router_observes_normalized_configured_decision_scores():
    raw_logits = torch.tensor([[0.0, torch.log(torch.tensor(3.0))]])
    expected_scores = {
        "softmax": torch.tensor([[0.25, 0.75]]),
        "sigmoid": torch.tensor([[0.4, 0.6]]),
    }
    sqrtsoftplus = torch.nn.functional.softplus(raw_logits).sqrt()
    expected_scores["sqrtsoftplus"] = sqrtsoftplus / sqrtsoftplus.sum(dim=-1, keepdim=True)

    for score_function, expected in expected_scores.items():
        router = TopKRouter.__new__(TopKRouter)
        torch.nn.Module.__init__(router)
        router.config = SimpleNamespace(
            sequence_parallel=False,
            moe_router_force_load_balancing=False,
            moe_router_force_biased=None,
        )
        router.score_function = score_function
        router.layer_number = 1
        router._maintain_float32_expert_bias = lambda: None
        router.apply_input_jitter = lambda tensor: tensor
        router.gating = lambda tensor: raw_logits
        router.routing = lambda logits, padding_mask=None: (
            logits,
            torch.ones_like(logits, dtype=torch.bool),
        )
        observed = []

        with capture_tensor_observations(
            lambda *args: observed.append(args), frozenset({"router_scores"})
        ):
            router(torch.ones(1, 1))

        assert len(observed) == 1
        assert observed[0][:3] == (router, "router_scores", "router_scores")
        torch.testing.assert_close(observed[0][3], expected)
        assert observed[0][4] is None
        assert not observed[0][3].requires_grad
