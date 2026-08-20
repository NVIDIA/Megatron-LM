# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.


from types import SimpleNamespace
from typing import cast

import pytest
import torch

import megatron.core.transformer.moe.moe_utils as moe_utils
import megatron.core.transformer.moe.router as router_module
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_submodules
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.moe.moe_logging import MoEMetricsTracker
from megatron.core.transformer.moe.moe_utils import (
    get_default_pg_collection,
    get_updated_expert_bias,
    router_gating_linear,
    topk_routing_with_score_function,
)
from megatron.core.transformer.moe.router import Router, TopKRouter
from megatron.core.transformer.spec_utils import get_submodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.test_utilities import Utils

try:
    # Check availability of TE fused router ops
    from megatron.core.extensions.transformer_engine import (
        fused_topk_with_score_function as _fused_topk_with_score_function,
    )

    HAVE_ROUTER_FUSION = _fused_topk_with_score_function is not None
except Exception:  # pragma: no cover - defensive
    HAVE_ROUTER_FUSION = False

HAVE_DENSE_ROUTER_FUSION = (
    HAVE_ROUTER_FUSION and moe_utils.fused_topk_with_score_function_supports_topk_indices
)


@pytest.mark.parametrize(
    "backend,routing_map_mode,num_experts,capacity_factor,expected_dtype",
    [
        ("deepep", "bool", 8, None, torch.int64),
        ("deepepv2", "bool", 8, None, torch.int64),
        ("ncclep", "bool", 8, None, torch.int64),
        ("hybridep", "bool", 8, None, None),
        ("hybridep", "indices", 1 << 15, None, torch.int16),
        ("hybridep", "indices", (1 << 15) + 1, None, None),
        ("hybridep", "indices", 8, 1.0, None),
    ],
)
def test_dense_route_indices_dtype(
    monkeypatch, backend, routing_map_mode, num_experts, capacity_factor, expected_dtype
):
    monkeypatch.setattr(router_module, "fused_topk_with_score_function_supports_topk_indices", True)
    monkeypatch.setattr(router_module, "HAVE_HYBRIDEP_DENSE_ROUTING", True)
    router = SimpleNamespace(
        config=SimpleNamespace(
            moe_router_fusion=True,
            moe_token_dispatcher_type="flex",
            moe_expert_capacity_factor=capacity_factor,
            moe_flex_dispatcher_backend=backend,
            moe_hybridep_routing_map_mode=routing_map_mode,
            num_moe_experts=num_experts,
        ),
        expt_tp_group=SimpleNamespace(size=lambda: 1),
    )

    assert TopKRouter._dense_route_indices_dtype(router) == expected_dtype


@pytest.mark.parametrize("supports_topk_indices", [False, True])
def test_fused_router_only_forwards_supported_topk_indices(monkeypatch, supports_topk_indices):
    received_kwargs = {}

    def fake_fused_router(**kwargs):
        received_kwargs.update(kwargs)
        return torch.zeros_like(kwargs["logits"]), kwargs.get(
            "topk_indices", torch.zeros_like(kwargs["logits"], dtype=torch.bool)
        )

    monkeypatch.setattr(moe_utils, "HAVE_TE", True)
    monkeypatch.setattr(moe_utils, "fused_topk_with_score_function", fake_fused_router)
    monkeypatch.setattr(
        moe_utils, "fused_topk_with_score_function_supports_topk_indices", supports_topk_indices
    )
    logits = torch.randn(4, 8)
    topk_indices = torch.empty(4, 2, dtype=torch.int64)

    topk_routing_with_score_function(logits, 2, fused=True, topk_indices=topk_indices)

    assert ("topk_indices" in received_kwargs) is supports_topk_indices
    if supports_topk_indices:
        assert received_kwargs["topk_indices"] is topk_indices

    received_kwargs.clear()
    topk_routing_with_score_function(logits, 2, fused=True)
    assert "topk_indices" not in received_kwargs


class TestTop2Router:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        _set_random_seed(seed_=123, data_parallel_random_init=False)
        print("done intializing")
        num_moe_experts = 4
        self.transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            num_moe_experts=num_moe_experts,
            use_cpu_initialization=True,
            moe_router_load_balancing_type="aux_loss",
            moe_router_topk=2,
            moe_aux_loss_coeff=0,
            bf16=True,
            params_dtype=torch.bfloat16,
            add_bias_linear=False,
        )
        submodules = get_submodules(
            get_gpt_layer_local_submodules(num_experts=num_moe_experts, moe_grouped_gemm=False).mlp
        )
        assert isinstance(submodules, MoESubmodules)
        self.sequential_mlp = MoELayer(self.transformer_config, submodules)
        self.router = cast(Router, self.sequential_mlp.router)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    def test_constructor(self):
        assert isinstance(self.router, Router)

        num_weights = sum([p.numel() for p in self.router.parameters()])
        assert num_weights == 12 * 4, num_weights

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("moe_router_pre_softmax", [(True), (False)])
    @pytest.mark.parametrize("score_function", ["sigmoid", "softmax"])
    def test_router_forward(self, moe_router_pre_softmax, score_function):
        with torch.no_grad():
            self.router = self.router.cuda()
            self.router.config.moe_router_pre_softmax = moe_router_pre_softmax
            self.router.config.moe_router_score_function = score_function
            # [num tokens, hidden size]
            hidden_states = torch.randn((32, 2, self.router.config.hidden_size))
            hidden_states = hidden_states.cuda().bfloat16()
            scores, indices = self.router(hidden_states)

    @pytest.mark.internal
    @pytest.mark.skipif(
        not torch.cuda.is_available() or not HAVE_ROUTER_FUSION,
        reason="TE fused router ops not available",
    )
    @pytest.mark.parametrize("score_function", ["sigmoid", "softmax"])
    def test_router_forward_fusion_equivalence(self, score_function):
        with torch.no_grad():
            self.router = self.router.cuda()
            self.router.config.moe_router_score_function = score_function
            hidden_states = torch.randn((32, 2, self.router.config.hidden_size))
            hidden_states = hidden_states.cuda().bfloat16()

            # Unfused
            self.router.config.moe_router_fusion = False
            scores_ref, routing_ref = self.router(hidden_states)

            # Fused
            self.router.config.moe_router_fusion = True
            scores_fused, routing_fused = self.router(hidden_states)

            assert torch.equal(routing_ref, routing_fused), "Routing map mismatch"
            torch.testing.assert_close(scores_ref, scores_fused)
            # restore the config
            self.router.config.moe_router_fusion = False

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_aux_loss(self):
        self.sequential_mlp = self.sequential_mlp.cuda()

        # Without aux loss
        hidden_states = torch.randn((32, 2, self.router.config.hidden_size))
        hidden_states = hidden_states.cuda().bfloat16()
        out = self.sequential_mlp(hidden_states)[0]
        out.sum().mul_(0).backward()
        assert self.sequential_mlp.router.weight.grad.abs().sum() == 0

        # With aux loss
        self.transformer_config.moe_aux_loss_coeff = 1
        out = self.sequential_mlp(hidden_states)[0]
        out.sum().mul_(0).backward()
        assert self.sequential_mlp.router.weight.grad.abs().sum() > 0

        # With Z loss
        self.transformer_config.moe_aux_loss_coeff = 0
        self.transformer_config.moe_z_loss_coeff = 1
        self.sequential_mlp.router.weight.grad.fill_(0)
        out = self.sequential_mlp(hidden_states)[0]
        out.sum().mul_(0).backward()
        assert self.sequential_mlp.router.weight.grad.abs().sum() > 0

    @pytest.mark.internal
    def test_nested_hybrid_mtp_z_loss_uses_mtp_depth_for_metric_index(self, monkeypatch):
        self.router.config.moe_z_loss_coeff = 1.0
        self.router.config.mtp_num_layers = 1
        self.router.is_mtp_layer = True
        self.router.layer_number = 2
        self.router.mtp_layer_number = 1

        tracker = MoEMetricsTracker()
        monkeypatch.setattr(
            "megatron.core.transformer.moe.router.get_moe_metrics_tracker", lambda: tracker
        )
        logits = torch.randn(8, self.router.config.num_moe_experts, requires_grad=True)

        self.router.apply_z_loss(logits)

        values = tracker.metrics["z_loss"].values
        assert values.shape == (self.router.config.num_layers + 1,)
        assert torch.count_nonzero(values[:-1]) == 0
        assert values[-1] > 0

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("router_fusion", [False, True])
    def test_router_with_padding_mask(self, router_fusion):
        """Test that HybridEP excludes padding tokens from routing."""
        if router_fusion and not HAVE_ROUTER_FUSION:
            pytest.skip("TE fused router ops not available")
        self.router = self.router.cuda()
        self.router.config.moe_router_fusion = router_fusion
        self.router.config.moe_token_dispatcher_type = "flex"
        self.router.config.moe_flex_dispatcher_backend = "hybridep"
        self.router.config.moe_hybridep_routing_map_mode = "bool"
        seq_len = 32
        batch_size = 2
        hidden_size = self.router.config.hidden_size

        # Create input with shape [seq_len, batch_size, hidden_size]
        hidden_states = torch.randn((seq_len, batch_size, hidden_size)).cuda().bfloat16()

        # Create padding mask: first half valid, second half padding
        # padding_mask shape: [seq_len, batch_size]
        # Convention: True = padding (exclude), False = valid (include)
        padding_mask = torch.zeros((seq_len, batch_size), dtype=torch.bool, device='cuda')
        padding_mask[seq_len // 2 :, :] = True  # Second half is padding

        # Test forward pass with padding mask
        with torch.no_grad():
            probs_with_mask, routing_map_with_mask = self.router(
                hidden_states, padding_mask=padding_mask
            )

            # Test forward pass without padding mask (only valid tokens)
            hidden_states_valid = hidden_states[: seq_len // 2, :, :]
            probs_without_mask, routing_map_without_mask = self.router(hidden_states_valid)

            # The valid part of routing with mask should match routing without mask
            probs_valid_part = probs_with_mask.reshape(seq_len, batch_size, -1)[
                : seq_len // 2, :, :
            ]
            probs_valid_part = probs_valid_part.reshape(-1, probs_valid_part.shape[-1])

            # Check that shapes are as expected
            assert probs_with_mask.shape == (
                seq_len * batch_size,
                self.router.config.num_moe_experts,
            )
            assert routing_map_with_mask.shape == (
                seq_len * batch_size,
                self.router.config.num_moe_experts,
            )

            padding_rows = padding_mask.reshape(-1)
            assert torch.count_nonzero(probs_with_mask[padding_rows]) == 0
            assert not routing_map_with_mask[padding_rows].any()

            # Verify that probs for valid tokens are similar
            assert torch.equal(probs_valid_part, probs_without_mask)

    @pytest.mark.internal
    @pytest.mark.skipif(
        not torch.cuda.is_available() or not HAVE_DENSE_ROUTER_FUSION,
        reason="TE dense fused router output is not available",
    )
    def test_hybridep_dense_routing_masks_padding(self, monkeypatch):
        monkeypatch.setattr(router_module, "HAVE_HYBRIDEP_DENSE_ROUTING", True)
        self.router = self.router.cuda()
        self.router.config.moe_router_fusion = True
        self.router.config.moe_token_dispatcher_type = "flex"
        self.router.config.moe_flex_dispatcher_backend = "hybridep"
        self.router.config.moe_hybridep_routing_map_mode = "indices"
        hidden_states = torch.randn(
            (8, 2, self.router.config.hidden_size), device="cuda", dtype=torch.bfloat16
        )
        padding_mask = torch.zeros((8, 2), dtype=torch.bool, device="cuda")
        padding_mask[4:, :] = True

        with torch.no_grad():
            probs, routing_map = self.router(hidden_states, padding_mask=padding_mask)

        padding_rows = padding_mask.reshape(-1)
        assert routing_map.dtype == torch.int16
        assert routing_map.shape == (16, self.router.config.moe_router_topk)
        assert torch.all(routing_map[padding_rows] == -1)
        assert torch.count_nonzero(probs[padding_rows]) == 0

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize(
        "dispatcher,backend,capacity_factor,rank_capacity_factor",
        [
            ("allgather", "deepep", None, None),
            ("alltoall", "deepep", None, None),
            ("flex", "deepep", None, None),
            ("flex", "deepepv2", None, None),
            ("flex", "hybridep", 1.0, None),
            ("flex", "hybridep", None, 1.0),
        ],
    )
    def test_padding_mask_preserves_routes_outside_dropless_hybridep(
        self, dispatcher, backend, capacity_factor, rank_capacity_factor
    ):
        """Only dropless HybridEP may consume a sparse route map."""
        self.router = self.router.cuda()
        self.router.config.moe_token_dispatcher_type = dispatcher
        self.router.config.moe_flex_dispatcher_backend = backend
        self.router.config.moe_expert_capacity_factor = capacity_factor
        self.router.config.moe_expert_rank_capacity_factor = rank_capacity_factor
        hidden_states = torch.randn(
            (16, 2, self.router.config.hidden_size), device="cuda", dtype=torch.bfloat16
        )
        padding_mask = torch.zeros((16, 2), dtype=torch.bool, device="cuda")
        padding_mask[8:, :] = True

        with torch.no_grad():
            probs_with_mask, routing_map_with_mask = self.router(
                hidden_states, padding_mask=padding_mask
            )
            probs_without_mask, routing_map_without_mask = self.router(hidden_states)

        torch.testing.assert_close(probs_with_mask, probs_without_mask)
        assert torch.equal(routing_map_with_mask, routing_map_without_mask)

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_router_dtype(self):
        self.router = self.router.cuda()
        self.sequential_mlp = self.sequential_mlp.cuda()
        hidden_states = torch.randn((32, 2, self.router.config.hidden_size), dtype=torch.bfloat16)
        hidden_states = hidden_states.cuda()

        # Test with default setting (bf16)
        self.router.config.moe_router_dtype = None
        with torch.no_grad():
            scores, routing_map = self.router(hidden_states)
            out = self.sequential_mlp(hidden_states)
            assert scores.dtype == torch.bfloat16, "Router output should be bf16 by default"
            assert out[0].dtype == torch.bfloat16

        # Test with fp32 enabled
        self.router.config.moe_router_dtype = 'fp32'
        with torch.no_grad():
            scores, routing_map = self.router(hidden_states)
            out = self.sequential_mlp(hidden_states)
            assert scores.dtype == torch.float32, "Router output should be fp32 when enabled"
            assert out[0].dtype == torch.bfloat16
            self.sequential_mlp.config.moe_token_dispatcher_type = "alltoall"
            out = self.sequential_mlp(hidden_states)
            assert out[0].dtype == torch.bfloat16
            self.sequential_mlp.config.moe_token_dispatcher_type = "allgather"

        # Test with fp64 enabled
        self.router.config.moe_router_dtype = 'fp64'
        with torch.no_grad():
            scores, routing_map = self.router(hidden_states)
            out = self.sequential_mlp(hidden_states)
            assert scores.dtype == torch.float64, "Router output should be fp64 when enabled"
            assert out[0].dtype == torch.bfloat16

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_force_load_balancing(self):
        hidden_states = torch.randn(
            (32, 2, self.router.config.hidden_size), device="cuda", dtype=torch.bfloat16
        )
        hidden_states.requires_grad = True

        # First forward pass with normal routing
        normal_scores, normal_routing_map = self.router(hidden_states)

        # Second forward pass with force load balancing
        self.router.config.moe_router_force_load_balancing = True
        force_scores, force_routing_map = self.router(hidden_states)

        assert normal_scores.shape == force_scores.shape
        assert normal_routing_map.shape == force_routing_map.shape
        assert torch.equal(normal_scores, force_scores) == False

        # Backward pass for force load balancing
        self.router.zero_grad()
        force_scores.sum().backward()
        assert hidden_states.grad is not None
        assert self.router.weight.grad.norm() > 0

        self.router.config.moe_router_force_load_balancing = False

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("capacity_factor", [None, 1.0, 2.0])
    @pytest.mark.parametrize("drop_policy", ["probs", "position"])
    @pytest.mark.parametrize("pad_to_capacity", [True, False])
    def test_token_dropping(self, capacity_factor, drop_policy, pad_to_capacity):
        if capacity_factor is None and pad_to_capacity:
            pytest.skip("Capacity factor is None, so no token dropping should be applied")

        num_tokens = 32
        self.router = self.router.cuda()
        self.router.config.moe_expert_capacity_factor = capacity_factor
        self.router.config.moe_token_drop_policy = drop_policy
        self.router.config.moe_pad_expert_input_to_capacity = pad_to_capacity

        hidden_states = torch.randn(
            (num_tokens, self.router.config.hidden_size), dtype=torch.bfloat16, device="cuda"
        )
        hidden_states.requires_grad = True
        probs, routing_map = self.router(hidden_states)

        if capacity_factor is not None:
            if pad_to_capacity:
                assert (
                    routing_map.sum().item()
                    == num_tokens * self.router.config.moe_router_topk * capacity_factor
                )
            else:
                assert (
                    routing_map.sum().item()
                    <= num_tokens * self.router.config.moe_router_topk * capacity_factor
                )
        else:
            assert routing_map.sum().item() == num_tokens * self.router.config.moe_router_topk

        # restore the config
        self.router.config.moe_expert_capacity_factor = None
        self.router.config.moe_token_drop_policy = "probs"
        self.router.config.moe_pad_expert_input_to_capacity = False


class TestGroupLimitedRouter:
    def setup_method(self, method):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=8,
            context_parallel_size=1,
        )
        _set_random_seed(seed_=123, data_parallel_random_init=False)
        print("done intializing")

        num_moe_experts = 16
        self.transformer_config = TransformerConfig(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=8,
            context_parallel_size=1,
            num_moe_experts=num_moe_experts,
            moe_router_topk=4,
            moe_router_group_topk=2,
            moe_router_num_groups=8,
            moe_router_pre_softmax=True,
            moe_router_load_balancing_type="aux_loss",
            moe_aux_loss_coeff=0,
            moe_router_dtype='fp32',
            moe_token_dispatcher_type="alltoall",
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            use_cpu_initialization=True,
            bf16=True,
            params_dtype=torch.bfloat16,
            add_bias_linear=False,
        )

        # init MoE layer
        submodules = get_submodules(
            get_gpt_layer_local_submodules(num_experts=num_moe_experts, moe_grouped_gemm=False).mlp
        )
        assert isinstance(submodules, MoESubmodules)
        self.moe_layer = MoELayer(self.transformer_config, submodules).cuda()
        self.router = cast(Router, self.moe_layer.router)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    def test_constructor(self):
        assert isinstance(self.router, Router)

        num_weights = sum([p.numel() for p in self.router.parameters()])
        assert (
            num_weights
            == self.transformer_config.hidden_size * self.transformer_config.num_moe_experts
        ), num_weights

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("moe_router_group_topk,moe_router_num_groups", [(3, 8), (2, 4)])
    @pytest.mark.parametrize("moe_router_pre_softmax", [(True), (False)])
    @pytest.mark.parametrize("score_function", ["sigmoid", "softmax"])
    def test_router_forward(
        self, moe_router_group_topk, moe_router_num_groups, moe_router_pre_softmax, score_function
    ):
        with torch.no_grad():
            self.router.config.moe_router_group_topk = moe_router_group_topk
            self.router.config.moe_router_num_groups = moe_router_num_groups
            self.router.config.moe_router_pre_softmax = moe_router_pre_softmax
            self.router.config.moe_router_score_function = score_function
            if moe_router_pre_softmax:
                self.router.config.moe_router_topk_scaling_factor = 16.0

            seq_len = 128
            batch_size = 4
            num_tokens = seq_len * batch_size
            # hidden_states shape: [seq_len, batch_size, hidden_size]
            hidden_states = (
                torch.randn((seq_len, batch_size, self.router.config.hidden_size)).cuda().bfloat16()
            )
            scores, routing_map = self.router(hidden_states)
            assert scores.shape == (num_tokens, self.router.config.num_moe_experts), scores.shape
            assert routing_map.shape == (
                num_tokens,
                self.router.config.num_moe_experts,
            ), routing_map.shape

            group_routing_map = (
                routing_map.reshape(num_tokens, moe_router_num_groups, -1).max(dim=-1).values
            )
            assert torch.all(group_routing_map.sum(dim=-1) <= moe_router_group_topk)

    @pytest.mark.internal
    @pytest.mark.skipif(
        not torch.cuda.is_available() or not HAVE_ROUTER_FUSION,
        reason="TE fused router ops not available",
    )
    @pytest.mark.parametrize("score_function", ["sigmoid", "softmax"])
    def test_router_forward_fusion_equivalence(self, score_function):
        with torch.no_grad():
            self.router = self.router.cuda()
            self.router.score_function = score_function
            seq_len = 32
            batch_size = 4
            hidden_states = torch.randn((seq_len, batch_size, self.router.config.hidden_size))
            hidden_states = hidden_states.cuda().bfloat16()

            # Unfused
            self.router.config.moe_router_fusion = False
            scores_ref, routing_ref = self.router(hidden_states)

            # Fused
            self.router.config.moe_router_fusion = True
            scores_fused, routing_fused = self.router(hidden_states)

            assert torch.equal(routing_ref, routing_fused), "Routing map mismatch"
            torch.testing.assert_close(scores_ref, scores_fused)
            # restore the config
            self.router.config.moe_router_fusion = False


class TestAuxLossFreeTop2Router:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1, expert_model_parallel_size=8)
        _set_random_seed(seed_=123, data_parallel_random_init=False)
        print("done intializing")
        num_moe_experts = 8
        self.transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            num_moe_experts=num_moe_experts,
            use_cpu_initialization=True,
            expert_model_parallel_size=8,
            moe_router_load_balancing_type="none",  # No aux loss
            moe_router_score_function="sigmoid",  # Using sigmoid scoring
            moe_router_enable_expert_bias=True,  # Enable expert bias
            moe_router_bias_update_rate=0.1,  # Set bias update rate
            moe_router_topk=2,
            bf16=True,
            params_dtype=torch.bfloat16,
            add_bias_linear=False,
        )
        submodules = get_submodules(
            get_gpt_layer_local_submodules(num_experts=num_moe_experts, moe_grouped_gemm=False).mlp
        )
        assert isinstance(submodules, MoESubmodules)
        self.moe_layer = MoELayer(self.transformer_config, submodules)
        self.router = cast(Router, self.moe_layer.router)
        assert self.router.expert_bias is not None
        assert self.router.local_tokens_per_expert is not None

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_router_forward_aux_free(self):
        hidden_states = torch.randn((32, 2, self.router.config.hidden_size))
        hidden_states = hidden_states.cuda().bfloat16()
        self.router = self.router.cuda()

        # First forward pass
        initial_bias = self.router.expert_bias.clone()
        scores1, indices1 = self.router(hidden_states)
        initial_tokens = self.router.local_tokens_per_expert.clone()
        updated_bias = get_updated_expert_bias(
            self.router.local_tokens_per_expert,
            self.router.expert_bias,
            self.router.config.moe_router_bias_update_rate,
        )

        # Verify expert bias was updated
        assert not torch.equal(initial_bias, updated_bias), "Expert bias should be updated"

        # Basic output checks
        assert scores1.shape == (64, 8), "Router scores shape mismatch"
        assert indices1.shape == (64, 8), "Router indices shape mismatch"

        # Print some debug info
        print("Updated bias after first forward pass:", updated_bias)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("deterministic", [False, True])
    def test_dense_expert_bias_token_counts(self, deterministic):
        self.router = self.router.cuda()
        self.router.local_tokens_per_expert.zero_()
        topk_indices = torch.tensor(
            [[0, 3], [1, 4], [0, 7], [2, 5]], device="cuda", dtype=torch.int16
        )
        padding_mask = torch.tensor([False, True, False, False], device="cuda")

        previous_deterministic = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(deterministic)
        try:
            self.router._apply_expert_bias(topk_indices, padding_mask=padding_mask)
        finally:
            torch.use_deterministic_algorithms(previous_deterministic)

        expected = torch.tensor([2, 0, 1, 1, 0, 1, 0, 1], device="cuda", dtype=torch.float32)
        torch.testing.assert_close(self.router.local_tokens_per_expert, expected)

    @pytest.mark.internal
    @pytest.mark.skipif(
        not torch.cuda.is_available() or not HAVE_DENSE_ROUTER_FUSION,
        reason="TE dense fused router output is not available",
    )
    def test_fused_dense_routing_with_expert_bias(self):
        self.router = self.router.cuda()
        self.router.config.moe_router_fusion = True
        self.router.config.moe_token_dispatcher_type = "flex"
        self.router.config.moe_flex_dispatcher_backend = "deepep"
        self.router.local_tokens_per_expert.zero_()
        self.router.expert_bias.copy_(
            torch.arange(self.router.config.num_moe_experts, device="cuda", dtype=torch.float32)
        )
        hidden_states = torch.randn(
            (4, 2, self.router.config.hidden_size), device="cuda"
        ).bfloat16()
        padding_mask = torch.tensor(
            [[False, True], [False, False], [True, False], [False, False]], device="cuda"
        )

        _, topk_indices = self.router(hidden_states, padding_mask=padding_mask)

        assert topk_indices.dtype == torch.int64
        assert topk_indices.shape == (8, self.router.config.moe_router_topk)
        expected = torch.bincount(
            topk_indices[~padding_mask.reshape(-1)].reshape(-1),
            minlength=self.router.config.num_moe_experts,
        ).to(torch.float32)
        torch.testing.assert_close(self.router.local_tokens_per_expert, expected)

    @pytest.mark.internal
    @pytest.mark.skipif(
        not torch.cuda.is_available() or not HAVE_ROUTER_FUSION,
        reason="TE fused router ops not available",
    )
    @pytest.mark.parametrize("score_function", ["sigmoid", "softmax"])
    def test_router_forward_fusion_equivalence(self, score_function):
        with torch.no_grad():
            # Build two fresh routers to avoid bias update interference
            submodules = get_submodules(
                get_gpt_layer_local_submodules(
                    num_experts=self.transformer_config.num_moe_experts, moe_grouped_gemm=False
                ).mlp
            )
            assert isinstance(submodules, MoESubmodules)
            moe_layer_ref = MoELayer(self.transformer_config, submodules)
            moe_layer_fused = MoELayer(self.transformer_config, submodules)
            router_ref = moe_layer_ref.router.cuda()
            router_fused = moe_layer_fused.router.cuda()

            # Ensure identical initial parameters/state
            router_fused.weight.copy_(router_ref.weight)
            expert_bias_sample = torch.randn_like(router_ref.expert_bias)
            router_ref.expert_bias.copy_(expert_bias_sample)
            router_fused.expert_bias.copy_(expert_bias_sample)

            router_ref.config.moe_router_score_function = score_function
            router_fused.config.moe_router_score_function = score_function

            hidden_states = torch.randn((32, 2, router_ref.config.hidden_size))
            hidden_states = hidden_states.cuda().bfloat16()

            # Unfused
            router_ref.config.moe_router_fusion = False
            scores_ref, routing_ref = router_ref(hidden_states)

            # Fused
            router_fused.config.moe_router_fusion = True
            scores_fused, routing_fused = router_fused(hidden_states)

            assert torch.equal(routing_ref, routing_fused)
            torch.testing.assert_close(scores_ref, scores_fused)


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("router_dtype", [torch.bfloat16, torch.float32, torch.float64])
def test_router_gating_linear(router_dtype):
    tols = dict(rtol=2.0e-2, atol=1.0e-3)

    ref_inp = torch.randn((4096, 7168), dtype=torch.bfloat16, device="cuda")
    ref_weight = torch.randn((256, 7168), dtype=torch.bfloat16, device="cuda")
    ref_inp.requires_grad = True
    ref_weight.requires_grad = True
    bwd_input = torch.randn((4096, 256), dtype=router_dtype, device="cuda")

    ref_output = torch.nn.functional.linear(ref_inp.to(router_dtype), ref_weight.to(router_dtype))
    ref_output.backward(bwd_input)

    inp = ref_inp.detach()
    weight = ref_weight.detach()
    inp.requires_grad = True
    weight.requires_grad = True
    bias = None
    output = router_gating_linear(inp, weight, bias, router_dtype)
    output.backward(bwd_input)

    assert output.dtype == router_dtype
    assert ref_inp.grad.dtype == ref_inp.dtype
    assert ref_weight.grad.dtype == ref_weight.dtype
    # Relax atol for float32: TE general_gemm produces results ~6.5e-3 away from
    # torch.nn.functional.linear, which exceeds the default 1e-3 atol.
    if router_dtype == torch.float32:
        tols = dict(rtol=2.0e-2, atol=1.0e-2)
    assert torch.allclose(output, ref_output, **tols)
    assert torch.allclose(inp.grad, ref_inp.grad, **tols)
    assert torch.allclose(weight.grad, ref_weight.grad, **tols)


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("router_dtype", [torch.bfloat16, torch.float32, torch.float64])
def test_router_gating_linear_bias(router_dtype):
    tols = dict(rtol=2.0e-2, atol=1.0e-3)

    ref_inp = torch.randn((4096, 7168), dtype=router_dtype, device="cuda")
    ref_weight = torch.randn((256, 7168), dtype=router_dtype, device="cuda")
    ref_bias = torch.randn((256,), dtype=router_dtype, device="cuda")
    ref_inp.requires_grad = True
    ref_weight.requires_grad = True
    ref_bias.requires_grad = True
    bwd_input = torch.randn((4096, 256), dtype=router_dtype, device="cuda")

    ref_output = torch.nn.functional.linear(
        ref_inp.to(router_dtype), ref_weight.to(router_dtype), ref_bias.to(router_dtype)
    )
    ref_output.backward(bwd_input)

    inp = ref_inp.detach()
    weight = ref_weight.detach()
    bias = ref_bias.detach()
    inp.requires_grad = True
    weight.requires_grad = True
    bias.requires_grad = True
    output = router_gating_linear(inp, weight, bias, router_dtype)
    output.backward(bwd_input)

    assert output.dtype == router_dtype
    assert ref_inp.grad.dtype == ref_inp.dtype
    assert ref_weight.grad.dtype == ref_weight.dtype
    assert ref_bias.grad.dtype == ref_bias.dtype
    assert torch.allclose(output, ref_output, **tols)
    assert torch.allclose(inp.grad, ref_inp.grad, **tols)
    assert torch.allclose(weight.grad, ref_weight.grad, **tols)
    assert torch.allclose(bias.grad, ref_bias.grad, **tols)


# ============================================================
# Hash-based MoE routing tests
# ============================================================


def _hash_routing_config(**overrides):
    """Create a base TransformerConfig suitable for hash routing tests."""
    defaults = dict(
        num_layers=2,
        hidden_size=16,
        num_attention_heads=8,
        num_moe_experts=4,
        moe_router_topk=2,
        moe_router_load_balancing_type="aux_loss",
        moe_aux_loss_coeff=0.0,
        moe_router_dtype="fp32",
        add_bias_linear=False,
        use_cpu_initialization=True,
        moe_n_hash_layers=1,
        actual_vocab_size=128,
    )
    defaults.update(overrides)
    return TransformerConfig(**defaults)


class TestHashRouting:
    """Test hash-based MoE routing (_hash_routing, is_hash_layer, config validation)."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=1,
        )
        _set_random_seed(seed_=42, data_parallel_random_init=False)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("score_function", ["softmax", "sigmoid", "sqrtsoftplus"])
    def test_hash_routing_correctness(self, score_function):
        """Verify expert selection matches tid2eid and scores are computed correctly."""
        config = _hash_routing_config(moe_router_score_function=score_function)
        pg_collection = get_default_pg_collection()
        router = TopKRouter(config=config, pg_collection=pg_collection, layer_number=1)

        num_tokens, num_experts = 16, 4
        logits = torch.randn(num_tokens, num_experts, device="cuda")
        input_ids = torch.randint(0, 128, (4, 4), device="cuda")

        routing_probs, routing_map = router._hash_routing(logits, input_ids)

        # Compute expected
        if score_function == "softmax":
            scores = torch.softmax(logits, dim=-1, dtype=torch.float32).type_as(logits)
        elif score_function == "sigmoid":
            scores = torch.sigmoid(logits.float()).type_as(logits)
        else:
            scores = torch.nn.functional.softplus(logits.float()).sqrt().type_as(logits)

        flat_ids = input_ids.T.reshape(-1)
        top_indices = router.tid2eid[flat_ids].long()
        probs = scores.gather(1, top_indices)
        if score_function != "softmax":
            probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-20)

        # Each token routed to exactly topk experts matching tid2eid
        assert (routing_map.sum(dim=1) == router.topk).all()
        for i in range(num_tokens):
            actual = routing_map[i].nonzero(as_tuple=True)[0].sort().values
            expected = top_indices[i].sort().values
            assert torch.equal(actual, expected)
            for k in range(router.topk):
                expert_idx = top_indices[i, k].item()
                assert torch.isclose(routing_probs[i, expert_idx], probs[i, k], atol=1e-5)

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_is_hash_layer_logic(self):
        """Test layer boundary, MTP guard, and expert bias interaction."""
        pg_collection = get_default_pg_collection()

        # Boundary: layers within/beyond moe_n_hash_layers
        config = _hash_routing_config(moe_n_hash_layers=2)
        r1 = TopKRouter(config=config, pg_collection=pg_collection, layer_number=1)
        r2 = TopKRouter(config=config, pg_collection=pg_collection, layer_number=2)
        r3 = TopKRouter(config=config, pg_collection=pg_collection, layer_number=3)
        assert r1.is_hash_layer is True and r1.tid2eid is not None
        assert r2.is_hash_layer is True
        assert r3.is_hash_layer is False and r3.tid2eid is None

        # MTP layers bypass hash routing
        mtp_router = TopKRouter(
            config=config, pg_collection=pg_collection, layer_number=1, is_mtp_layer=True
        )
        assert mtp_router.is_hash_layer is False and mtp_router.tid2eid is None

        # Expert bias disabled on hash layers
        bias_config = _hash_routing_config(
            moe_n_hash_layers=1,
            moe_router_enable_expert_bias=True,
            moe_router_score_function="sigmoid",
        )
        hash_r = TopKRouter(config=bias_config, pg_collection=pg_collection, layer_number=1)
        normal_r = TopKRouter(config=bias_config, pg_collection=pg_collection, layer_number=2)
        assert hash_r.enable_expert_bias is False
        assert normal_r.enable_expert_bias is True

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_moe_layer_hash_routing_integration(self):
        """End-to-end MoELayer forward/backward with hash routing; raises without input_ids."""
        config = _hash_routing_config(moe_n_hash_layers=1)
        submodules = get_submodules(
            get_gpt_layer_local_submodules(
                num_experts=config.num_moe_experts, moe_grouped_gemm=False
            ).mlp
        )
        moe_layer = MoELayer(config, submodules, layer_number=1).cuda()

        hidden_states = torch.randn(8, 2, 16, device="cuda", requires_grad=True)
        input_ids = torch.randint(0, 128, (2, 8), device="cuda")

        # Forward succeeds with input_ids
        output, _ = moe_layer(hidden_states, input_ids=input_ids)
        assert output.shape == hidden_states.shape
        assert not torch.isnan(output).any()

        # Backward succeeds
        output.sum().backward()
        assert hidden_states.grad is not None
        assert not torch.isnan(hidden_states.grad).any()
