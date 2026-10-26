# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for megatron.core.inference.moe.sigmoid_topk.

Expert indices must match the compiled inference router bit for bit. Probabilities
match bitwise for top-k below eight; for larger top-k the compiled path sums the k
scores in another order, so they may differ in the last bit. Batch-invariant mode
retains its existing behavior.
"""

import pytest
import torch

from megatron.core.inference.moe.sigmoid_topk import sigmoid_topk_routing
from megatron.core.inference.utils import InferenceMode
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.custom_layers.batch_invariant_kernels import set_batch_invariant_mode
from megatron.core.transformer.moe import router as router_module
from megatron.core.transformer.moe.router import InferenceTopKRouter
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

MODEL_ROUTING_CONFIGS = [
    pytest.param(128, 6, 2.5, id="nano"),
    pytest.param(512, 22, 5.0, id="super-ultra"),
]


def _assert_matches_compiled(probs, indices, ref_probs, ref_indices, case=None):
    assert indices.dtype == ref_indices.dtype, case
    assert probs.dtype == ref_probs.dtype, case
    assert torch.equal(indices, ref_indices), case
    if probs.shape[-1] < 8:
        assert torch.equal(probs.view(torch.int32), ref_probs.view(torch.int32)), case
    else:
        # The compiled sum of eight or more scores uses another reduction order.
        torch.testing.assert_close(probs, ref_probs, rtol=1e-6, atol=0, msg=str(case))


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
    elif kind == "all_equal":
        logits = torch.zeros_like(logits)
        bias = torch.zeros_like(bias)
    elif kind == "zero_scores":
        # sigmoid underflows to +0.0 everywhere but every 7th expert; signed-zero bias.
        logits = torch.full_like(logits, -200.0)
        logits[:, ::7] = torch.randn(
            num_tokens, logits[:, ::7].shape[1], generator=gen, device="cuda"
        )
        bias = torch.zeros_like(bias)
        bias[::2] = -0.0
    return logits.contiguous(), bias


def _strided_expert_bias(bias):
    # Unused entries must differ from the logical bias to expose contiguous loads.
    storage = torch.full((2 * bias.numel() + 1,), 10.0, device=bias.device, dtype=bias.dtype)
    view = storage[1::2]
    view.copy_(bias)
    assert view.storage_offset() == 1 and view.stride() == (2,)
    return view


@pytest.mark.internal
class TestSigmoidTopkRouting:

    # One static (experts, top-k, scale) configuration per test, as in a server whose
    # router configuration is fixed: dynamo specializes top-k, which determines the
    # compiled probability reduction. Compiling several top-k values without resetting
    # dynamo can make top-k symbolic and change the reduction order.
    @pytest.mark.parametrize(
        "num_experts,topk,scaling_factor",
        [*MODEL_ROUTING_CONFIGS, (64, 7, None), (96, 2, 1.5), (128, 8, 2.5), (512, 32, None)],
    )
    def test_matches_compiled_router_across_batch_sizes(self, num_experts, topk, scaling_factor):
        torch._dynamo.reset()
        try:
            for kind in ("normal", "wide", "ties", "all_equal", "zero_scores"):
                for num_tokens in (1, 2, 3, 8, 33, 256, 2048):
                    logits, bias = _inputs(kind, num_tokens, num_experts, seed=num_tokens)
                    with torch.inference_mode():
                        ref_probs, ref_indices = _compiled_reference(
                            logits, bias, topk, scaling_factor
                        )
                        probs, indices = sigmoid_topk_routing(logits, bias, topk, scaling_factor)
                    _assert_matches_compiled(
                        probs, indices, ref_probs, ref_indices, case=(kind, num_tokens)
                    )
        finally:
            torch._dynamo.reset()

    @pytest.mark.parametrize("num_experts,topk,scaling_factor", MODEL_ROUTING_CONFIGS)
    @pytest.mark.parametrize("layout", ["strided", "broadcast"])
    def test_expert_bias_views_match_contiguous(self, num_experts, topk, scaling_factor, layout):
        torch._dynamo.reset()
        try:
            logits, bias = _inputs("normal", 3, num_experts, seed=37)
            if layout == "strided":
                bias = _strided_expert_bias(bias)
            else:
                # Keep enough backing storage for an incorrect contiguous load to
                # read different values without accessing an invalid address.
                storage = torch.full_like(bias, 10.0)
                storage[0] = bias[0]
                bias = storage[:1].expand(num_experts)
                # Expert zero must be selected with the broadcast bias, but an
                # incorrect contiguous load gives every other expert a +10 bias.
                logits[:, 0] = 10.0
                assert bias.stride() == (0,)
            with torch.inference_mode():
                probs, indices = sigmoid_topk_routing(logits, bias, topk, scaling_factor)
                contiguous_probs, contiguous_indices = sigmoid_topk_routing(
                    logits, bias.contiguous(), topk, scaling_factor
                )
                ref_probs, ref_indices = _compiled_reference(logits, bias, topk, scaling_factor)
            assert torch.equal(indices, contiguous_indices)
            assert torch.equal(probs.view(torch.int32), contiguous_probs.view(torch.int32))
            _assert_matches_compiled(probs, indices, ref_probs, ref_indices, case=layout)
        finally:
            torch._dynamo.reset()

    @pytest.mark.parametrize("num_experts,topk,scaling_factor", MODEL_ROUTING_CONFIGS)
    def test_empty_batch(self, num_experts, topk, scaling_factor):
        logits = torch.empty(0, num_experts, device="cuda")
        probs, indices = sigmoid_topk_routing(
            logits, torch.zeros(num_experts, device="cuda"), topk, scaling_factor
        )
        assert probs.shape == (0, topk) and indices.shape == (0, topk)

    def test_rejects_single_expert_routing(self):
        with pytest.raises(AssertionError):
            sigmoid_topk_routing(torch.randn(4, 8, device="cuda"), torch.zeros(8, device="cuda"), 1)

    @pytest.mark.parametrize("num_experts,topk", [(512, 33), (1024, 6), (16, 22)])
    def test_rejects_unsupported_shapes(self, num_experts, topk):
        with pytest.raises(AssertionError):
            sigmoid_topk_routing(
                torch.randn(2, num_experts, device="cuda"),
                torch.zeros(num_experts, device="cuda"),
                topk,
                5.0,
            )

    @pytest.mark.parametrize("invalid_bias", ["rank", "device"])
    def test_rejects_invalid_expert_bias(self, invalid_bias):
        logits = torch.randn(2, 128, device="cuda")
        bias = (
            torch.zeros(128, 1, device="cuda")
            if invalid_bias == "rank"
            else torch.zeros(128, device="cpu")
        )
        with pytest.raises(AssertionError):
            sigmoid_topk_routing(logits, bias, 6, 2.5)


@pytest.mark.internal
class TestInferenceRouterUsesKernel:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        torch._dynamo.reset()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()
        torch._dynamo.reset()

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
            inference_moe_triton_router=True,
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

    @pytest.mark.parametrize("num_experts,topk,scaling_factor", MODEL_ROUTING_CONFIGS)
    @pytest.mark.parametrize("num_tokens", [2, 40])
    def test_router_output_matches_compiled_path(
        self, monkeypatch, num_experts, topk, scaling_factor, num_tokens
    ):
        router = self._router(
            num_moe_experts=num_experts,
            moe_router_topk=topk,
            moe_router_topk_scaling_factor=scaling_factor,
        )
        hidden = torch.randn(num_tokens, 1, 256, device="cuda", dtype=torch.bfloat16)
        logits = router.gating(hidden).squeeze(1)
        with torch.inference_mode():
            assert router._uses_sigmoid_topk_kernel(logits, None)
        outputs = []
        for enabled in (True, False):
            monkeypatch.setattr(router_module, "HAVE_SIGMOID_TOPK_KERNEL", enabled)
            with InferenceMode.active(), torch.inference_mode():
                outputs.append(router(hidden))
        (probs, indices), (ref_probs, ref_indices) = outputs
        _assert_matches_compiled(probs, indices, ref_probs, ref_indices)

    @pytest.mark.parametrize("num_experts,topk,scaling_factor", MODEL_ROUTING_CONFIGS)
    def test_router_preserves_strided_expert_bias(
        self, monkeypatch, num_experts, topk, scaling_factor
    ):
        router = self._router(
            num_moe_experts=num_experts,
            moe_router_topk=topk,
            moe_router_topk_scaling_factor=scaling_factor,
        )
        bias = _strided_expert_bias(router.expert_bias)
        router.expert_bias = bias
        # The router's fp32 conversion must preserve this view, not hide the stride.
        assert router.expert_bias.float().data_ptr() == bias.data_ptr()
        assert router.expert_bias.float().stride() == (2,)
        hidden = torch.randn(2, 1, 256, device="cuda", dtype=torch.bfloat16)
        with InferenceMode.active(), torch.inference_mode():
            assert router._uses_sigmoid_topk_kernel(router.gating(hidden).squeeze(1), None)
            probs, indices = router(hidden)
            router.expert_bias = bias.contiguous()
            contiguous_probs, contiguous_indices = router(hidden)
            router.expert_bias = bias
            monkeypatch.setattr(router_module, "HAVE_SIGMOID_TOPK_KERNEL", False)
            ref_probs, ref_indices = router(hidden)
        assert torch.equal(indices, contiguous_indices)
        assert torch.equal(probs.view(torch.int32), contiguous_probs.view(torch.int32))
        _assert_matches_compiled(probs, indices, ref_probs, ref_indices)

    @pytest.mark.parametrize("num_experts,topk,scaling_factor", MODEL_ROUTING_CONFIGS)
    def test_grad_enabled_inference_uses_kernel(self, num_experts, topk, scaling_factor):
        """Inference with gradients enabled (e.g. RL evaluation) routes like no-grad inference."""
        router = self._router(
            num_moe_experts=num_experts,
            moe_router_topk=topk,
            moe_router_topk_scaling_factor=scaling_factor,
        )
        hidden = torch.randn(40, 1, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        with InferenceMode.active():
            with torch.enable_grad():
                assert router._uses_sigmoid_topk_kernel(router.gating(hidden).squeeze(1), None)
                probs, indices = router(hidden)
            with torch.no_grad():
                ref_probs, ref_indices = router(hidden)
        assert not probs.requires_grad
        assert torch.equal(indices, ref_indices)
        assert torch.equal(probs.view(torch.int32), ref_probs.view(torch.int32))

    @pytest.mark.parametrize("num_experts,topk,scaling_factor", MODEL_ROUTING_CONFIGS)
    @pytest.mark.parametrize("num_tokens", [1, 2, 3, 4, 17, 65, 1024, 8192])
    def test_graph_replay_uses_updated_logits_and_bias(
        self, monkeypatch, num_experts, topk, scaling_factor, num_tokens
    ):
        """Captured routing must read new inputs on every replay, including tied scores."""
        logits, bias = _inputs("normal", num_tokens, num_experts, seed=17)
        router = self._router(
            num_moe_experts=num_experts,
            moe_router_topk=topk,
            moe_router_topk_scaling_factor=scaling_factor,
        )
        monkeypatch.setattr(router, "gating", lambda logits: logits)
        router.expert_bias = bias
        routing_input = logits.unsqueeze(1)
        with InferenceMode.active(), torch.inference_mode():
            assert router._uses_sigmoid_topk_kernel(logits, None)
            # Compile both paths before capture with a fixed top-k specialization.
            _compiled_reference(logits, bias, topk, scaling_factor)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                router(routing_input)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                probs, indices = router(routing_input)
            for seed, kind in enumerate(("wide", "ties", "all_equal", "zero_scores", "normal")):
                new_logits, new_bias = _inputs(kind, num_tokens, num_experts, seed=seed)
                logits.copy_(new_logits)
                bias.copy_(new_bias)
                graph.replay()
                ref_probs, ref_indices = _compiled_reference(logits, bias, topk, scaling_factor)
                _assert_matches_compiled(probs, indices, ref_probs, ref_indices, case=kind)

    @pytest.mark.parametrize("num_experts,topk,scaling_factor", MODEL_ROUTING_CONFIGS)
    def test_batch_invariant_mode_preserves_uncompiled_routing(
        self, monkeypatch, num_experts, topk, scaling_factor
    ):
        router = self._router(
            num_moe_experts=num_experts,
            moe_router_topk=topk,
            moe_router_topk_scaling_factor=scaling_factor,
        )
        hidden = torch.randn(2, 1, 256, device="cuda", dtype=torch.bfloat16)

        def unexpected_routing(*args, **kwargs):
            pytest.fail("Batch-invariant routing must keep the uncompiled reference path")

        monkeypatch.setattr(router_module, "sigmoid_topk_routing", unexpected_routing)
        monkeypatch.setattr(router, "_compiled_topk_routing", unexpected_routing)
        with set_batch_invariant_mode(True), InferenceMode.active(), torch.inference_mode():
            logits = router.gating(hidden).squeeze(1)
            assert not router._uses_sigmoid_topk_kernel(logits, None)
            ref_probs, ref_indices = router_module.topk_routing_with_score_function(
                logits,
                topk,
                scaling_factor=scaling_factor,
                score_function="sigmoid",
                expert_bias=router.expert_bias,
                dense_output=True,
            )
            probs, indices = router(hidden)
        assert torch.equal(indices, ref_indices)
        assert torch.equal(probs.view(torch.int32), ref_probs.view(torch.int32))

    @torch.no_grad()
    def test_unsupported_configurations_keep_compiled_path(self):
        logits = torch.randn(4, 128, device="cuda")
        # Unvalidated reduction widths and expert counts retain the compiled path.
        assert not self._router(moe_router_topk=33)._uses_sigmoid_topk_kernel(logits, None)
        assert not self._router(num_moe_experts=1024)._uses_sigmoid_topk_kernel(
            torch.randn(4, 1024, device="cuda"), None
        )
        assert not self._router(moe_router_enable_expert_bias=False)._uses_sigmoid_topk_kernel(
            logits, None
        )
        assert not self._router()._uses_sigmoid_topk_kernel(logits.double(), None)
        assert not self._router()._uses_sigmoid_topk_kernel(
            logits, torch.zeros(4, 6, dtype=torch.int64, device="cuda")
        )
