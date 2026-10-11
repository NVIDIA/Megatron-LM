# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for megatron.core.inference.moe.sigmoid_topk.

The kernel must reproduce, bit for bit, the inference router's compiled
sigmoid routing evaluated with gradients disabled (top-k with sorted=False).
"""

import pytest
import torch

from megatron.core.inference.moe.sigmoid_topk import sigmoid_topk_routing
from megatron.core.inference.utils import InferenceMode
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.moe import router as router_module
from megatron.core.transformer.moe.router import InferenceTopKRouter
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


def _compiled_reference(logits, expert_bias, topk, scaling_factor):
    return InferenceTopKRouter._compiled_topk_routing(
        logits,
        topk,
        False,
        None,
        None,
        scaling_factor,
        "sigmoid",
        expert_bias,
        False,
        None,
        True,
        None,
    )


def _inputs(kind, num_tokens, num_experts, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.randn(num_tokens, num_experts, generator=gen, device="cuda")
    bias = torch.randn(num_experts, generator=gen, device="cuda") * 0.05
    if kind == "wide":
        logits, bias = logits * 8, bias * 10
    elif kind == "ties":
        # Few distinct scores: many rows tie at the k-th value.
        logits = torch.round(logits * 2) / 2
        bias = torch.round(bias * 64) / 64
    elif kind == "zero_scores":
        # sigmoid underflows to +0.0 everywhere but every 7th expert; signed-zero bias.
        logits = torch.full_like(logits, -200.0)
        logits[:, ::7] = torch.randn(
            num_tokens, logits[:, ::7].shape[1], generator=gen, device="cuda"
        )
        bias = torch.zeros_like(bias)
        bias[::2] = -0.0
    return logits.contiguous(), bias


@pytest.mark.internal
class TestSigmoidTopkRouting:

    # One static (experts, top-k, scale) configuration per test, as in a server whose
    # router configuration is fixed: dynamo specializes top-k and inductor unrolls the
    # k-term sum in slot order. Compiling the routing for several top-k values in one
    # process makes top-k symbolic, and the compiled sum then reduces in another order.
    @pytest.mark.parametrize(
        "num_experts,topk,scaling_factor", [(128, 6, 2.5), (64, 7, None), (96, 2, 1.5)]
    )
    def test_bitwise_matches_compiled_router(self, num_experts, topk, scaling_factor):
        torch._dynamo.reset()
        try:
            for kind in ("normal", "wide", "ties", "zero_scores"):
                for num_tokens in (1, 3, 8, 33, 256, 2048):
                    logits, bias = _inputs(kind, num_tokens, num_experts, seed=num_tokens)
                    with torch.inference_mode():
                        ref_probs, ref_indices = _compiled_reference(
                            logits, bias, topk, scaling_factor
                        )
                        probs, indices = sigmoid_topk_routing(logits, bias, topk, scaling_factor)
                    case = (kind, num_tokens)
                    assert indices.dtype == ref_indices.dtype, case
                    assert probs.dtype == ref_probs.dtype, case
                    assert torch.equal(indices, ref_indices), case
                    assert torch.equal(probs.view(torch.int32), ref_probs.view(torch.int32)), case
        finally:
            torch._dynamo.reset()

    def test_empty_batch(self):
        logits = torch.empty(0, 128, device="cuda")
        probs, indices = sigmoid_topk_routing(logits, torch.zeros(128, device="cuda"), 6, 2.5)
        assert probs.shape == (0, 6) and indices.shape == (0, 6)

    def test_rejects_single_expert_routing(self):
        with pytest.raises(AssertionError):
            sigmoid_topk_routing(torch.randn(4, 8, device="cuda"), torch.zeros(8, device="cuda"), 1)


@pytest.mark.internal
class TestInferenceRouterUsesKernel:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        torch._dynamo.reset()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def _router(self, **overrides):
        kwargs = dict(
            num_layers=1,
            hidden_size=256,
            num_attention_heads=4,
            num_moe_experts=128,
            moe_router_topk=6,
            moe_router_load_balancing_type="none",
            moe_router_score_function="sigmoid",
            moe_router_enable_expert_bias=True,
            moe_router_topk_scaling_factor=2.5,
            moe_router_dtype="fp32",
            add_bias_linear=False,
            bf16=True,
            params_dtype=torch.bfloat16,
        )
        kwargs.update(overrides)
        config = TransformerConfig(**kwargs)
        router = InferenceTopKRouter(
            config, pg_collection=ProcessGroupCollection.use_mpu_process_groups()
        ).cuda()
        router.set_layer_number(1)
        if router.expert_bias is not None:
            with torch.no_grad():
                router.expert_bias.uniform_(-0.05, 0.05)
        return router

    def test_router_output_matches_compiled_path(self, monkeypatch):
        router = self._router()
        hidden = torch.randn(40, 1, 256, device="cuda", dtype=torch.bfloat16)
        logits = router.gating(hidden).squeeze(1)
        with torch.inference_mode():
            assert router._uses_sigmoid_topk_kernel(logits, None)
        outputs = []
        for enabled in (True, False):
            monkeypatch.setattr(router_module, "HAVE_SIGMOID_TOPK_KERNEL", enabled)
            with InferenceMode.active(), torch.inference_mode():
                outputs.append(router(hidden))
        (probs, indices), (ref_probs, ref_indices) = outputs
        assert torch.equal(indices, ref_indices)
        assert torch.equal(probs.view(torch.int32), ref_probs.view(torch.int32))

    def test_grad_enabled_preserves_compiled_routing_and_autograd(self):
        """InferenceMode alone does not disable gradients or sorted top-k selection."""
        router = self._router()
        hidden = torch.randn(40, 1, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        with InferenceMode.active(), torch.enable_grad():
            probs, indices = router(hidden)
            ref_probs, ref_indices = _compiled_reference(
                router.gating(hidden).squeeze(1), router.expert_bias, 6, 2.5
            )
            assert probs.requires_grad
            assert torch.equal(indices, ref_indices)
            assert torch.equal(probs, ref_probs)
            weights = torch.arange(6, device="cuda", dtype=torch.float32)
            grad = torch.autograd.grad((probs * weights).sum(), hidden)[0]
            ref_grad = torch.autograd.grad((ref_probs * weights).sum(), hidden)[0]
        assert torch.equal(grad, ref_grad)

    @pytest.mark.parametrize("num_tokens", [1, 17, 65, 1024, 8192])
    def test_graph_replay_uses_updated_logits_and_bias(self, num_tokens):
        """Captured routing must read new inputs on every replay, including tied scores."""
        logits, bias = _inputs("normal", num_tokens, 128, seed=17)
        with torch.inference_mode():
            # Compile both paths before capture. Reference arithmetic is fixed at top-6.
            _compiled_reference(logits, bias, 6, 2.5)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                sigmoid_topk_routing(logits, bias, 6, 2.5)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                probs, indices = sigmoid_topk_routing(logits, bias, 6, 2.5)
            for seed, kind in enumerate(("wide", "ties", "zero_scores", "normal")):
                new_logits, new_bias = _inputs(kind, num_tokens, 128, seed=seed)
                logits.copy_(new_logits)
                bias.copy_(new_bias)
                graph.replay()
                ref_probs, ref_indices = _compiled_reference(logits, bias, 6, 2.5)
                assert torch.equal(indices, ref_indices), kind
                assert torch.equal(probs.view(torch.int32), ref_probs.view(torch.int32)), kind

    @torch.no_grad()
    def test_unsupported_configurations_keep_compiled_path(self):
        logits = torch.randn(4, 128, device="cuda")
        # At top-k 8 inductor no longer unrolls the probability sum in slot order.
        assert not self._router(moe_router_topk=8)._uses_sigmoid_topk_kernel(logits, None)
        assert not self._router(moe_router_enable_expert_bias=False)._uses_sigmoid_topk_kernel(
            logits, None
        )
        assert not self._router()._uses_sigmoid_topk_kernel(logits.double(), None)
        assert not self._router()._uses_sigmoid_topk_kernel(
            logits, torch.zeros(4, 6, dtype=torch.int64, device="cuda")
        )
