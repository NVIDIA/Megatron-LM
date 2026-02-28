# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for inference-optimized MoE components.

Config is modeled after nanov3 (Nemotron-6 3B Hybrid MoE) with smaller
dimensions for fast unit test execution:
- squared_relu activation (not swiglu/gated)
- sigmoid router score function with expert bias
- topk=6, topk_scaling_factor=2.5
- shared experts
"""

import pytest
import torch

from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_te_min_version, is_torch_min_version
from megatron.training.initialize import _set_random_seed
from megatron.core.activations import squared_relu
from tests.unit_tests.test_utilities import Utils

# Reusable skip decorators
requires_te = pytest.mark.skipif(
    not is_te_min_version("1.7.0.dev0"),
    reason="Requires transformer-engine >= 1.7.0",
)
requires_torch_grouped_mm = pytest.mark.skipif(
    not is_torch_min_version("2.10") or not hasattr(torch, '_grouped_mm'),
    reason="Requires PyTorch >= 2.10 with torch._grouped_mm",
)

# ──────────────────────────────────────────────────────────────────────
# NanoV3-like config (scaled down from 2688→128 hidden, 128→8 experts)
# ──────────────────────────────────────────────────────────────────────

NANOV3_BASE = dict(
    num_layers=1,
    hidden_size=128,
    ffn_hidden_size=128,
    num_attention_heads=4,
    num_query_groups=2,
    num_moe_experts=8,
    moe_ffn_hidden_size=128,
    moe_router_topk=6,
    moe_router_score_function="sigmoid",
    moe_router_enable_expert_bias=True,
    moe_router_topk_scaling_factor=2.5,
    moe_shared_expert_intermediate_size=256,
    moe_router_dtype='fp32',
    moe_shared_expert_overlap=False,
    moe_grouped_gemm=True,
    moe_token_dispatcher_type="alltoall",
    moe_aux_loss_coeff=0.01,
    activation_func=squared_relu,
    normalization="RMSNorm",
    add_bias_linear=False,
    bf16=True,
    params_dtype=torch.bfloat16
)


def _make_base_config(**overrides):
    """Create a TransformerConfig with nanov3-like defaults."""
    params = {**NANOV3_BASE, **overrides}
    return TransformerConfig(**params)


# ──────────────────────────────────────────────────────────────────────
# InferenceTopKRouter
# ──────────────────────────────────────────────────────────────────────


class TestInferenceTopKRouter:

    @classmethod
    def setup_class(cls):
        Utils.initialize_model_parallel(1, 1)
        _set_random_seed(seed_=123, data_parallel_random_init=False)

    @classmethod
    def teardown_class(cls):
        Utils.destroy_model_parallel()

    def _make_router(self, **config_overrides):
        from megatron.core.transformer.moe.moe_utils import get_default_pg_collection
        from megatron.core.transformer.moe.router import InferenceTopKRouter

        config = _make_base_config(**config_overrides)
        return InferenceTopKRouter(config=config, pg_collection=get_default_pg_collection()).cuda().to(torch.bfloat16)

    def test_init_rejects_num_groups(self):
        """InferenceTopKRouter requires moe_router_num_groups=None."""
        with pytest.raises(AssertionError, match="moe_router_num_groups"):
            self._make_router(moe_router_num_groups=2)

    @pytest.mark.parametrize("score_fn", ["none", "invalid"])
    def test_init_rejects_unsupported_score_function(self, score_fn):
        """InferenceTopKRouter requires sigmoid or softmax score function."""
        with pytest.raises(AssertionError, match="moe_router_score_function"):
            self._make_router(moe_router_score_function=score_fn,
                              moe_router_enable_expert_bias=False)

    @pytest.mark.parametrize("score_fn", ["sigmoid", "softmax"])
    def test_init_accepts_valid_score_function(self, score_fn):
        """InferenceTopKRouter accepts sigmoid and softmax."""
        # Expert bias only valid with sigmoid; disable it for softmax
        enable_bias = score_fn == "sigmoid"
        router = self._make_router(
            moe_router_score_function=score_fn,
            moe_router_enable_expert_bias=enable_bias,
        )
        assert router is not None

    def test_set_unset_inference_mode(self):
        """Toggle is_inference_cuda_graphed_iteration flag."""
        router = self._make_router()
        assert not router.is_inference_cuda_graphed_iteration

        router.set_inference_cuda_graphed_iteration()
        assert router.is_inference_cuda_graphed_iteration

        router.unset_inference_cuda_graphed_iteration()
        assert not router.is_inference_cuda_graphed_iteration

    def test_training_mode_forward_returns_sparse(self):
        """In training mode, forward delegates to parent and returns sparse tensors."""
        router = self._make_router()
        router.train()
        num_tokens = 16
        num_experts = NANOV3_BASE["num_moe_experts"]

        input_tensor = torch.randn(
            num_tokens, NANOV3_BASE["hidden_size"], device="cuda", dtype=torch.bfloat16
        )
        probs, routing_map = router(input_tensor)

        # Parent TopKRouter returns [num_tokens, num_experts] sparse routing_map
        assert routing_map.shape == (num_tokens, num_experts)

    def test_inference_vs_training_selects_same_experts(self):
        """Inference and training modes should select the same top-k experts."""
        router = self._make_router()
        num_tokens = 16
        topk = NANOV3_BASE["moe_router_topk"]

        input_tensor = torch.randn(
            num_tokens, NANOV3_BASE["hidden_size"], device="cuda", dtype=torch.bfloat16
        )

        # Training mode: get routing_map (sparse) and extract top expert indices
        router.train()
        _, routing_map = router(input_tensor.clone())
        # routing_map is [num_tokens, num_experts] bool
        training_experts = set()
        for i in range(num_tokens):
            experts_for_token = routing_map[i].nonzero(as_tuple=True)[0]
            for e in experts_for_token:
                training_experts.add((i, e.item()))

        # Inference mode: get top_indices (dense)
        router.eval()
        router.set_inference_cuda_graphed_iteration()
        _, top_indices = router(input_tensor.clone())

        inference_experts = set()
        for i in range(num_tokens):
            for k in range(topk):
                inference_experts.add((i, top_indices[i, k].item()))

        # Same expert selections
        assert training_experts == inference_experts

    def test_cuda_graph_capture_and_replay(self):
        """Router forward can be captured in a CUDA graph and replayed."""
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

        router = self._make_router()
        router.eval()
        router.set_inference_cuda_graphed_iteration()

        num_tokens = 16
        hidden_size = NANOV3_BASE["hidden_size"]

        # Static input buffer for CUDA graph (seeded for reproducibility)
        static_input = torch.randn(
            num_tokens, hidden_size, device="cuda", dtype=torch.bfloat16
        )

        # Warmup (required before CUDA graph capture)
        with torch.no_grad():
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                router(static_input)
            torch.cuda.current_stream().wait_stream(s)

        # Capture
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_probs, static_indices = router(static_input)

        router.unset_inference_cuda_graphed_iteration()

        # Fill indices with -1, replay, check that the graph overwrote them
        static_indices.fill_(-1)
        static_input.copy_(torch.randn_like(static_input))
        graph.replay()
        assert (static_indices != -1).all(), "Graph replay should overwrite all expert indices"

        expected_indices = [
            [2, 6, 4, 5, 3, 7], [4, 1, 3, 2, 6, 0], [4, 1, 3, 7, 5, 2],
            [6, 0, 7, 5, 2, 4], [0, 7, 5, 1, 4, 2], [5, 6, 0, 7, 1, 4],
            [6, 2, 0, 7, 4, 1], [0, 2, 1, 7, 4, 5], [0, 7, 5, 3, 1, 6],
            [1, 4, 7, 3, 0, 6], [6, 7, 0, 2, 3, 1], [3, 0, 7, 6, 4, 2],
            [6, 7, 0, 4, 1, 3], [1, 3, 6, 5, 0, 2], [6, 1, 0, 7, 3, 2],
            [1, 5, 0, 4, 3, 7],
        ]
        assert static_indices.tolist() == expected_indices, (
            f"Expert indices mismatch:\n{static_indices.tolist()}\n!=\n{expected_indices}"
        )

