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

from megatron.core.activations import squared_relu
from megatron.core.inference.communication.torch_symm_triton import are_tensors_nvls_eligible
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_te_min_version, is_torch_min_version
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.test_utilities import Utils

# Reusable skip decorators
requires_te = pytest.mark.skipif(
    not is_te_min_version("1.7.0.dev0"), reason="Requires transformer-engine >= 1.7.0"
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
    params_dtype=torch.bfloat16,
    transformer_impl="inference_optimized",
)


def _make_base_config(**overrides):
    """Create a TransformerConfig with nanov3-like defaults."""
    params = {**NANOV3_BASE, **overrides}
    return TransformerConfig(**params)


# ──────────────────────────────────────────────────────────────────────
# InferenceTopKRouter
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.internal
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
        return (
            InferenceTopKRouter(config=config, pg_collection=get_default_pg_collection())
            .cuda()
            .to(torch.bfloat16)
        )

    def test_init_rejects_num_groups(self):
        """InferenceTopKRouter requires moe_router_num_groups=None."""
        with pytest.raises(AssertionError, match="moe_router_num_groups"):
            self._make_router(moe_router_num_groups=2)

    def test_config_rejects_non_fp32_router_dtype(self):
        """inference_optimized config requires moe_router_dtype='fp32'."""
        with pytest.raises(ValueError, match="moe-router-dtype"):
            _make_base_config(
                transformer_impl="inference_optimized", add_qkv_bias=False, moe_router_dtype=None
            )

    @pytest.mark.parametrize("score_fn", ["none", "invalid"])
    def test_init_rejects_unsupported_score_function(self, score_fn):
        """InferenceTopKRouter requires sigmoid or softmax score function."""
        with pytest.raises(AssertionError, match="moe_router_score_function"):
            self._make_router(
                moe_router_score_function=score_fn, moe_router_enable_expert_bias=False
            )

    @pytest.mark.parametrize("score_fn", ["sigmoid", "softmax"])
    def test_init_accepts_valid_score_function(self, score_fn):
        """InferenceTopKRouter accepts sigmoid and softmax."""
        # Expert bias only valid with sigmoid; disable it for softmax
        enable_bias = score_fn == "sigmoid"
        router = self._make_router(
            moe_router_score_function=score_fn, moe_router_enable_expert_bias=enable_bias
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


# ──────────────────────────────────────────────────────────────────────
# InferenceCUDAGraphTokenDispatcher
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.internal
class TestInferenceCUDAGraphTokenDispatcher:

    @classmethod
    def setup_class(cls):
        from megatron.core.parallel_state import _set_global_symmetric_memory_buffer

        Utils.initialize_model_parallel(1, 1, expert_model_parallel_size=Utils.world_size)
        _set_random_seed(seed_=123, data_parallel_random_init=False)
        _set_global_symmetric_memory_buffer()

    @classmethod
    def teardown_class(cls):
        from megatron.core.parallel_state import destroy_global_symmetric_memory_buffer

        destroy_global_symmetric_memory_buffer()
        Utils.destroy_model_parallel()

    def _make_dispatcher(self, **config_overrides):
        from megatron.core.transformer.moe.moe_utils import get_default_pg_collection
        from megatron.core.transformer.moe.token_dispatcher_inference import (
            InferenceCUDAGraphTokenDispatcher,
        )

        config_overrides.setdefault("expert_model_parallel_size", Utils.world_size)
        config = _make_base_config(**config_overrides)
        num_local_experts = config.num_moe_experts // Utils.world_size
        ep_rank = torch.distributed.get_rank() if Utils.world_size > 1 else 0
        local_expert_indices = [ep_rank * num_local_experts + i for i in range(num_local_experts)]

        return InferenceCUDAGraphTokenDispatcher(
            num_local_experts=num_local_experts,
            local_expert_indices=local_expert_indices,
            config=config,
            pg_collection=get_default_pg_collection(),
        )

    def test_init(self):
        """Dispatcher can be constructed with nanov3-like config and EP=world_size."""
        dispatcher = self._make_dispatcher()
        assert dispatcher.topk == NANOV3_BASE["moe_router_topk"]
        assert dispatcher.ep_size == Utils.world_size

    def test_symmetric_memory_buffer_initialized(self):
        """EP symmetric memory buffer is accessible after _set_global_symmetric_memory_buffer."""
        from megatron.core.parallel_state import get_global_symmetric_memory_buffer_ep

        buf = get_global_symmetric_memory_buffer_ep()
        assert buf is not None

    @pytest.mark.parametrize("seed", [42, 123, 7])
    @pytest.mark.parametrize(
        "num_local_tokens",
        [
            1,
            2,
            4,
            8,
            16,
            24,
            32,
            40,
            48,
            56,
            64,
            72,
            80,
            88,
            96,
            104,
            112,
            120,
            128,
            136,
            144,
            152,
            160,
            168,
            176,
            184,
            192,
            200,
            208,
            216,
            224,
            232,
            240,
            248,
            256,
            272,
            288,
            304,
            320,
            336,
            352,
            368,
            384,
            400,
            416,
            432,
            448,
            464,
            480,
            496,
            512,
        ],
    )
    def test_cuda_graph_dispatch_combine(self, num_local_tokens, seed):
        """Dispatch+combine can be captured in a CUDA graph and replayed.
        Creates global buffers, shards per rank, and verifies:
        - NVLS AllGather output matches the full globalwol buffer
        - NVLS ReduceScatter output matches fp32-accumulated reference
        All tensor byte sizes are 128-bit aligned for NVLS eligibility.
        """

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        dispatcher = self._make_dispatcher()
        ep_size = dispatcher.ep_size
        hidden_size = NANOV3_BASE["hidden_size"]
        topk = NANOV3_BASE["moe_router_topk"]
        num_experts = NANOV3_BASE["num_moe_experts"]
        rank = torch.distributed.get_rank() if ep_size > 1 else 0
        num_global_tokens = num_local_tokens * ep_size

        # Create global buffers on rank 0 and broadcast to all ranks
        global_hidden = torch.randn(
            num_global_tokens, hidden_size, device="cuda", dtype=torch.bfloat16
        )
        global_probs = torch.randn(num_global_tokens, topk, device="cuda", dtype=torch.float32)
        global_routing_map = torch.randint(0, num_experts, (num_global_tokens, topk), device="cuda")
        torch.distributed.broadcast(global_hidden, src=0)
        torch.distributed.broadcast(global_probs, src=0)
        torch.distributed.broadcast(global_routing_map, src=0)

        # Each rank grabs their own shard
        start = rank * num_local_tokens
        end = start + num_local_tokens
        static_hidden = global_hidden[start:end].contiguous()
        static_probs = global_probs[start:end].contiguous()
        static_routing_map = global_routing_map[start:end].contiguous()

        if not are_tensors_nvls_eligible(static_hidden, static_probs, static_routing_map):
            pytest.skip(
                "Tensors are not NVLS-eligible (need SM>=9 and each tensor's memory to be a multiple of 16 bytes)"
            )

        # 3 warmup iterations on a side stream
        with torch.no_grad():
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    dispatcher.routing_map = static_routing_map
                    d_hidden, d_probs = dispatcher.token_dispatch(static_hidden, static_probs)
                    d_hidden = d_hidden.clone()
                    d_probs = d_probs.clone()
                    dispatcher.routing_map = dispatcher.routing_map.clone()
                    dispatcher.token_combine(d_hidden.clone())
            torch.cuda.current_stream().wait_stream(s)

        # Capture
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            dispatcher.routing_map = static_routing_map
            d_hidden, d_probs = dispatcher.token_dispatch(static_hidden, static_probs)
            graph_hidden = d_hidden.clone()
            graph_probs = d_probs.clone()
            graph_routing_map = dispatcher.routing_map.clone()
            graph_combined = dispatcher.token_combine(d_hidden.clone())

        # Verify shapes: dispatch expands by ep_size, combine shrinks back
        assert graph_hidden.shape == (num_global_tokens, hidden_size)
        assert graph_probs.shape == (num_global_tokens, topk)
        assert graph_combined.shape == (num_local_tokens, hidden_size)

        # Replay
        graph.replay()

        # Verify AllGather: all gathered tensors should match global buffers
        torch.testing.assert_close(graph_hidden, global_hidden, atol=0, rtol=0)
        torch.testing.assert_close(graph_probs, global_probs, atol=0, rtol=0)
        torch.testing.assert_close(graph_routing_map, global_routing_map, atol=0, rtol=0)

        # Verify ReduceScatter: all ranks have the same all-gathered data, so
        # rank r gets ep_size * chunk_r. Compute reference in fp32 then downcast.
        # Exact match (atol=0, rtol=0) is possible because the NVLS triton kernels
        # accumulate in fp32 before writing bf16 output.
        expected_combined = (global_hidden[start:end].float() * ep_size).bfloat16()
        torch.testing.assert_close(graph_combined, expected_combined, atol=0, rtol=0)
