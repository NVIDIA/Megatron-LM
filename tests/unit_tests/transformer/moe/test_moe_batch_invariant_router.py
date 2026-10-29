# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Batch-invariant sigmoid top-k routing in training and inference.

In batch-invariant mode, sigmoid routing with an expert bias runs the Triton sigmoid
top-k kernel in both the trainer's router and the inference router, so routing
probabilities agree bit for bit between the two sides. The trainer differentiates
the kernel's output through the reference formula.
"""

import pytest
import torch

from megatron.core.inference.moe.sigmoid_topk import sigmoid_topk_routing
from megatron.core.inference.utils import InferenceMode
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.custom_layers.batch_invariant_kernels import set_batch_invariant_mode
from megatron.core.transformer.moe import moe_utils
from megatron.core.transformer.moe.moe_utils import topk_routing_with_score_function
from megatron.core.transformer.moe.router import InferenceTopKRouter, TopKRouter
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

ROUTING_CONFIGS = [
    pytest.param(128, 6, 2.5, id="nano"),
    pytest.param(512, 22, 5.0, id="super-ultra"),
]


def _inputs(num_tokens, num_experts, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.randn(num_tokens, num_experts, generator=gen, device="cuda") * 3
    bias = torch.randn(num_experts, generator=gen, device="cuda") * 0.1
    return logits, bias


def _bits_equal(a, b):
    return torch.equal(a.view(torch.int32), b.view(torch.int32))


def _route(logits, bias, topk, scaling_factor, dense_output):
    return topk_routing_with_score_function(
        logits,
        topk,
        scaling_factor=scaling_factor,
        score_function="sigmoid",
        expert_bias=bias,
        dense_output=dense_output,
    )


@pytest.mark.parametrize("num_experts,topk,scaling_factor", ROUTING_CONFIGS)
def test_kernel_rows_do_not_depend_on_batch(num_experts, topk, scaling_factor):
    """Each row's routing is the same alone and inside batches of up to 4096 tokens."""
    logits, bias = _inputs(4096, num_experts)
    probs, indices = sigmoid_topk_routing(logits, bias, topk, scaling_factor, batch_invariant=True)
    for num_tokens in (1, 7, 64, 1000, 2048):
        for start in (0, 1001, 4096 - num_tokens):
            rows = slice(start, start + num_tokens)
            p, i = sigmoid_topk_routing(
                logits[rows].contiguous(), bias, topk, scaling_factor, batch_invariant=True
            )
            assert torch.equal(i, indices[rows]), (num_tokens, start)
            assert _bits_equal(p, probs[rows]), (num_tokens, start)


@pytest.mark.parametrize("num_experts,topk,scaling_factor", ROUTING_CONFIGS)
def test_batch_invariant_routing_uses_kernel(num_experts, topk, scaling_factor):
    logits, bias = _inputs(300, num_experts)
    ref_probs, ref_indices = sigmoid_topk_routing(
        logits, bias, topk, scaling_factor, batch_invariant=True
    )
    with set_batch_invariant_mode(True):
        probs, indices = _route(logits, bias, topk, scaling_factor, dense_output=True)
        routing_probs, routing_map = _route(logits, bias, topk, scaling_factor, dense_output=False)
    assert torch.equal(indices, ref_indices)
    assert _bits_equal(probs, ref_probs)
    expected_map = torch.zeros_like(routing_map).scatter(1, ref_indices, True)
    assert torch.equal(routing_map, expected_map)
    assert _bits_equal(routing_probs, torch.zeros_like(logits).scatter(1, ref_indices, ref_probs))


@pytest.mark.parametrize("num_experts,topk,scaling_factor", ROUTING_CONFIGS)
def test_training_values_match_inference(num_experts, topk, scaling_factor):
    """The differentiable training path returns the inference kernel's values."""
    logits, bias = _inputs(257, num_experts)
    with set_batch_invariant_mode(True):
        with torch.inference_mode():
            inf_probs, inf_indices = _route(logits, bias, topk, scaling_factor, dense_output=True)
        train_logits = logits.clone().requires_grad_()
        probs, indices = _route(train_logits, bias, topk, scaling_factor, dense_output=True)
    assert probs.requires_grad
    assert torch.equal(indices, inf_indices)
    assert _bits_equal(probs.detach(), inf_probs)


@pytest.mark.parametrize("num_experts,topk,scaling_factor", ROUTING_CONFIGS)
@pytest.mark.parametrize("dense_output", [False, True], ids=["map", "dense"])
def test_gradient_matches_reference(num_experts, topk, scaling_factor, dense_output):
    """Gradients match autograd through the reference routing formula."""
    logits, bias = _inputs(1024, num_experts, seed=1)

    x_bi = logits.clone().requires_grad_()
    with set_batch_invariant_mode(True):
        probs_bi, indices_bi = _route(x_bi, bias, topk, scaling_factor, dense_output)
    x_ref = logits.clone().requires_grad_()
    probs_ref, indices_ref = _route(x_ref, bias, topk, scaling_factor, dense_output)

    if dense_output:
        # The reference sorts its top-k by score; compare per expert.
        dense_bi = torch.zeros_like(logits).scatter(1, indices_bi, probs_bi)
        dense_ref = torch.zeros_like(logits).scatter(1, indices_ref, probs_ref)
    else:
        assert torch.equal(indices_bi, indices_ref)
        dense_bi, dense_ref = probs_bi, probs_ref
    torch.testing.assert_close(dense_bi, dense_ref, rtol=1e-6, atol=1e-6)

    grad = torch.randn_like(logits)
    dense_bi.backward(grad)
    dense_ref.backward(grad)
    torch.testing.assert_close(x_bi.grad, x_ref.grad, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param(dict(dtype=torch.bfloat16), id="bf16_logits"),
        pytest.param(dict(expert_bias=None), id="no_expert_bias"),
        pytest.param(dict(num_groups=8, group_topk=4), id="group_limited"),
        pytest.param(dict(topk=1), id="top1"),
    ],
)
def test_unsupported_configurations_keep_reference_routing(monkeypatch, overrides):
    def unexpected(*args, **kwargs):
        pytest.fail("unsupported configuration must keep the reference routing")

    monkeypatch.setattr(moe_utils, "sigmoid_topk_routing", unexpected)
    logits, bias = _inputs(64, 128)
    logits = logits.to(overrides.get("dtype", torch.float32))
    with set_batch_invariant_mode(True):
        topk_routing_with_score_function(
            logits,
            overrides.get("topk", 6),
            scaling_factor=2.5,
            score_function="sigmoid",
            expert_bias=overrides.get("expert_bias", bias),
            num_groups=overrides.get("num_groups"),
            group_topk=overrides.get("group_topk"),
        )


def test_default_mode_keeps_reference_routing(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("routing outside batch-invariant mode must not change")

    monkeypatch.setattr(moe_utils, "sigmoid_topk_routing", unexpected)
    logits, bias = _inputs(64, 128)
    _route(logits, bias, 6, 2.5, dense_output=False)


class TestTrainingAndInferenceRouters:
    """The training router and the inference router route identically under BI."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("num_experts,topk,scaling_factor", ROUTING_CONFIGS)
    def test_routers_match(self, num_experts, topk, scaling_factor):
        config = TransformerConfig(
            num_layers=1,
            hidden_size=256,
            num_attention_heads=4,
            num_moe_experts=num_experts,
            moe_router_topk=topk,
            moe_router_load_balancing_type="none",
            moe_aux_loss_coeff=0.0,
            moe_router_score_function="sigmoid",
            moe_router_enable_expert_bias=True,
            moe_router_topk_scaling_factor=scaling_factor,
            moe_router_dtype="fp32",
            add_bias_linear=False,
            bf16=True,
            params_dtype=torch.bfloat16,
        )
        pgs = ProcessGroupCollection.use_mpu_process_groups()
        train_router = TopKRouter(config, pg_collection=pgs).cuda()
        infer_router = InferenceTopKRouter(config, pg_collection=pgs).cuda()
        for router in (train_router, infer_router):
            router.set_layer_number(1)
        with torch.no_grad():
            train_router.expert_bias.uniform_(-0.05, 0.05)
        infer_router.load_state_dict(train_router.state_dict())

        hidden = torch.randn(96, 2, 256, device="cuda", dtype=torch.bfloat16)
        with set_batch_invariant_mode(True):
            train_router.train()
            probs, routing_map = train_router(hidden)
            with InferenceMode.active(), torch.inference_mode():
                inf_probs, inf_indices = infer_router(hidden[:, :1])
                inf_probs_b, inf_indices_b = infer_router(hidden[:, 1:])

        # The training router sees tokens in [seq, batch] order; the inference router
        # gets each sequence separately.
        dense = torch.zeros(96, 2, num_experts, device="cuda")
        dense[:, 0].scatter_(1, inf_indices, inf_probs)
        dense[:, 1].scatter_(1, inf_indices_b, inf_probs_b)
        dense = dense.view(-1, num_experts)
        assert torch.equal(routing_map, dense != 0)
        assert _bits_equal(probs.detach(), dense)

        probs.sum().backward()
        assert train_router.weight.grad is not None
        assert torch.isfinite(train_router.weight.grad).all()
