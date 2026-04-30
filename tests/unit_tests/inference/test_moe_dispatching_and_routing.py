# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for inference-optimized MoE components.

Config is modeled after nanov3 (Nemotron-6 3B Hybrid MoE) with smaller
dimensions for fast unit test execution:
- squared_relu activation (not swiglu/gated)
- sigmoid router score function with expert bias
- topk=6, topk_scaling_factor=2.5
- shared experts
"""

import gc

import pytest
import torch

from megatron.core.activations import squared_relu
from megatron.core.inference.communication.torch_symm_triton import are_tensors_nvls_eligible
from megatron.core.transformer.enums import AttnBackend
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
    num_layers=4,
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
    expert_tensor_parallel_size=1,
    use_cpu_initialization=True,
    attention_backend=AttnBackend.local,
    cuda_graph_impl="local",
    cuda_graph_scope="full_iteration_inference",
    moe_pad_experts_for_cuda_graph_inference=False,
    mamba_state_dim=128,
    mamba_head_dim=64,
    mamba_num_groups=8,
    mamba_num_heads=64,
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
        _, top_indices = router(input_tensor.clone())

        inference_experts = set()
        for i in range(num_tokens):
            for k in range(topk):
                inference_experts.add((i, top_indices[i, k].item()))

        # Same expert selections
        assert training_experts == inference_experts


# ──────────────────────────────────────────────────────────────────────
# NCCLAllGatherDispatcher
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.internal
class TestNCCLAllGatherDispatcher:

    @classmethod
    def setup_class(cls):
        Utils.initialize_model_parallel(1, 1, expert_model_parallel_size=Utils.world_size)
        _set_random_seed(seed_=123, data_parallel_random_init=False)

    @classmethod
    def teardown_class(cls):
        Utils.destroy_model_parallel()

    def teardown_method(self, method):
        gc.collect()
        torch.cuda.empty_cache()

    def _make_dispatcher(self, **config_overrides):
        from megatron.core.transformer.moe.moe_utils import get_default_pg_collection
        from megatron.core.transformer.moe.token_dispatcher_inference import NCCLAllGatherDispatcher

        NCCLAllGatherDispatcher.allocate_buffers()
        config_overrides.setdefault("expert_model_parallel_size", Utils.world_size)
        config = _make_base_config(**config_overrides)
        num_local_experts = config.num_moe_experts // Utils.world_size
        ep_rank = torch.distributed.get_rank() if Utils.world_size > 1 else 0
        local_expert_indices = [ep_rank * num_local_experts + i for i in range(num_local_experts)]
        return NCCLAllGatherDispatcher(
            num_local_experts=num_local_experts,
            local_expert_indices=local_expert_indices,
            config=config,
            pg_collection=get_default_pg_collection(),
            runs_metadata_sync=True,
        )

    def test_init(self):
        """Dispatcher can be constructed with nanov3-like config and EP=world_size."""
        dispatcher = self._make_dispatcher()
        assert dispatcher.topk == NANOV3_BASE["moe_router_topk"]
        assert dispatcher.ep_size == Utils.world_size

    @pytest.mark.parametrize("use_allgather_v", [False, True])
    def test_dispatch_combine(self, use_allgather_v):
        """Dispatch+combine correctness for both CG (equal-count) and prefill (variable-count) paths.

        All ranks share the same global reference tensors (broadcast from rank 0).
        Each rank contributes its own slice, then we verify:
        - dispatch gathers all slices back to the full global tensor
        - combine reduce-scatters the gathered data, giving each rank ep_size * its_slice
        """
        from megatron.core.transformer.moe.token_dispatcher_inference import NCCLAllGatherDispatcher

        if use_allgather_v and Utils.world_size == 1:
            pytest.skip("Variable-token prefill path requires EP > 1")

        dispatcher = self._make_dispatcher()
        ep_size = dispatcher.ep_size
        rank = torch.distributed.get_rank() if ep_size > 1 else 0
        hidden_size = NANOV3_BASE["hidden_size"]
        topk = NANOV3_BASE["moe_router_topk"]
        num_experts = NANOV3_BASE["num_moe_experts"]

        if use_allgather_v:
            # Variable token counts: rank r contributes (r+1)*8 tokens
            tokens_per_rank = [(r + 1) * 8 for r in range(ep_size)]
        else:
            tokens_per_rank = [16] * ep_size

        local_tokens = tokens_per_rank[rank]
        total_tokens = sum(tokens_per_rank)

        NCCLAllGatherDispatcher._use_allgather_v = use_allgather_v

        global_hidden = torch.randn(total_tokens, hidden_size, device="cuda", dtype=torch.bfloat16)
        global_probs = torch.randn(total_tokens, topk, device="cuda", dtype=torch.float32)
        global_routing_map = torch.randint(0, num_experts, (total_tokens, topk), device="cuda")
        if ep_size > 1:
            torch.distributed.broadcast(global_hidden, src=0)
            torch.distributed.broadcast(global_probs, src=0)
            torch.distributed.broadcast(global_routing_map, src=0)

        offset = sum(tokens_per_rank[:rank])
        hidden = global_hidden[offset : offset + local_tokens].contiguous()
        probs = global_probs[offset : offset + local_tokens].contiguous()
        dispatcher.routing_map = global_routing_map[offset : offset + local_tokens].contiguous()

        if ep_size == 1:
            d_hidden, d_probs = dispatcher.token_dispatch(hidden, probs)
            assert d_hidden is hidden
            assert d_probs is probs
            return

        d_hidden, d_probs = dispatcher.token_dispatch(hidden, probs)

        assert d_hidden.shape == (total_tokens, hidden_size)
        assert d_probs.shape == (total_tokens, topk)
        torch.testing.assert_close(d_hidden, global_hidden, atol=0, rtol=0)
        torch.testing.assert_close(d_probs, global_probs, atol=0, rtol=0)
        torch.testing.assert_close(dispatcher.routing_map, global_routing_map, atol=0, rtol=0)

        # All ranks have identical gathered data, so rank r's reduce-scatter output
        # is ep_size * its slice of global_hidden.
        combined = dispatcher.token_combine(d_hidden)
        assert combined.shape == (local_tokens, hidden_size)
        expected = (global_hidden[offset : offset + local_tokens].float() * ep_size).bfloat16()
        torch.testing.assert_close(combined, expected)


# ──────────────────────────────────────────────────────────────────────
# NVLSAllGatherVDispatcher
# ──────────────────────────────────────────────────────────────────────

_NVLS_ENGINE_MAX_TOKENS = 512


@pytest.mark.internal
class TestNVLSAllGatherVDispatcher:

    @classmethod
    def setup_class(cls):
        Utils.initialize_model_parallel(1, 1, expert_model_parallel_size=Utils.world_size)
        _set_random_seed(seed_=123, data_parallel_random_init=False)

    @classmethod
    def teardown_class(cls):
        from megatron.core.inference.symmetric_memory import SymmetricMemoryManager

        SymmetricMemoryManager.destroy()
        Utils.destroy_model_parallel()

    def _make_dispatcher(self):
        from megatron.core.parallel_state import get_expert_model_parallel_group
        from megatron.core.transformer.moe.moe_utils import get_default_pg_collection
        from megatron.core.transformer.moe.token_dispatcher_inference import (
            NVLSAllGatherVDispatcher,
        )

        config = _make_base_config(expert_model_parallel_size=Utils.world_size)
        num_local_experts = config.num_moe_experts // Utils.world_size
        ep_rank = torch.distributed.get_rank() if Utils.world_size > 1 else 0
        local_expert_indices = [ep_rank * num_local_experts + i for i in range(num_local_experts)]
        ep_group = get_expert_model_parallel_group()

        NVLSAllGatherVDispatcher.allocate_buffers(
            per_rank_worst_case_token_count=_NVLS_ENGINE_MAX_TOKENS,
            topk=NANOV3_BASE["moe_router_topk"],
            hidden_size=NANOV3_BASE["hidden_size"],
            ep_group=ep_group,
        )

        return NVLSAllGatherVDispatcher(
            num_local_experts=num_local_experts,
            local_expert_indices=local_expert_indices,
            config=config,
            pg_collection=get_default_pg_collection(),
            runs_metadata_sync=True,
        )

    def test_init(self):
        """Dispatcher can be constructed with nanov3-like config and EP=world_size."""
        dispatcher = self._make_dispatcher()
        assert dispatcher.topk == NANOV3_BASE["moe_router_topk"]
        assert dispatcher.ep_size == Utils.world_size

    @pytest.mark.parametrize("seed", [42, 123, 7])
    @pytest.mark.parametrize(
        "max_rank_tokens",
        # Covers: small, unaligned, power-of-2, and large up to engine_max
        [1, 7, 16, 24, 64, 128, 256, 512],
    )
    def test_cuda_graph_dispatch_combine(self, max_rank_tokens, seed):
        """Dispatch+combine can be captured in a CUDA graph and replayed.

        Uses uneven token counts across EP ranks (rank r gets
        max(1, max_rank_tokens + r - (ep_size - 1)) tokens) to exercise the
        AllGatherV variable-length path. Verifies:
        - AllGatherV output matches the global reference (valid prefix only)
        - ReduceScatterV output matches fp32-accumulated reference
        Exact match (atol=0) is possible because the NVLS triton kernels
        accumulate in fp32 before writing bf16 output.
        """

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        dispatcher = self._make_dispatcher()
        ep_size = dispatcher.ep_size
        hidden_size = NANOV3_BASE["hidden_size"]
        topk = NANOV3_BASE["moe_router_topk"]
        num_experts = NANOV3_BASE["num_moe_experts"]
        rank = torch.distributed.get_rank() if ep_size > 1 else 0

        # Uneven token counts: rank r gets max(1, max_rank_tokens + r - (ep_size-1))
        # so the last rank always has max_rank_tokens (≤ engine_max) and earlier
        # ranks have fewer, exercising the variable-length AllGatherV path.
        tokens_per_rank = [max(1, max_rank_tokens + r - (ep_size - 1)) for r in range(ep_size)]
        local_tokens = tokens_per_rank[rank]
        total_tokens = sum(tokens_per_rank)
        global_max = _NVLS_ENGINE_MAX_TOKENS * ep_size

        global_hidden = torch.randn(total_tokens, hidden_size, device="cuda", dtype=torch.bfloat16)
        global_probs = torch.randn(total_tokens, topk, device="cuda", dtype=torch.float32)
        global_routing_map = torch.randint(0, num_experts, (total_tokens, topk), device="cuda")
        if ep_size > 1:
            torch.distributed.broadcast(global_hidden, src=0)
            torch.distributed.broadcast(global_probs, src=0)
            torch.distributed.broadcast(global_routing_map, src=0)

        start = sum(tokens_per_rank[:rank])
        end = start + local_tokens
        static_hidden = global_hidden[start:end].contiguous()
        static_probs = global_probs[start:end].contiguous()
        static_routing_map = global_routing_map[start:end].contiguous()

        # Warmup on a side stream
        with torch.no_grad():
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    dispatcher.routing_map = static_routing_map
                    dispatcher._local_tokens = local_tokens
                    d_hidden, d_probs = dispatcher.token_dispatch(static_hidden, static_probs)
                    dispatcher.token_combine(d_hidden.clone())
            torch.cuda.current_stream().wait_stream(s)

        # Capture
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            dispatcher.routing_map = static_routing_map
            dispatcher._local_tokens = local_tokens
            d_hidden, d_probs = dispatcher.token_dispatch(static_hidden, static_probs)
            graph_hidden = d_hidden[:total_tokens].clone()
            graph_probs = d_probs[:total_tokens].clone()
            graph_routing_map = dispatcher.routing_map[:total_tokens].clone()
            graph_combined = dispatcher.token_combine(d_hidden.clone())

        # dispatch output is (global_max, *); only first total_tokens are valid
        assert d_hidden.shape == (global_max, hidden_size)
        assert d_probs.shape == (global_max, topk)
        assert graph_combined.shape == (local_tokens, hidden_size)

        graph.replay()

        torch.testing.assert_close(graph_hidden, global_hidden, atol=0, rtol=0)
        torch.testing.assert_close(graph_probs, global_probs, atol=0, rtol=0)
        torch.testing.assert_close(graph_routing_map, global_routing_map, atol=0, rtol=0)

        expected_combined = (global_hidden[start:end].float() * ep_size).bfloat16()
        torch.testing.assert_close(graph_combined, expected_combined, atol=0, rtol=0)
