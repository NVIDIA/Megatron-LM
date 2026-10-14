# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Behavior of the shared-prefix MoE capability.

``hybrid_star_moe_expert_bias_v1``: a shared prompt row stands for G dense rows, so a real
``TopKRouter`` must count it G times in ``local_tokens_per_expert`` (and every branch token
once), including when the MoE layer recomputes its forward in backward.
"""

import pytest
import torch

from tests.unit_tests.ssm.shared_prefix_test_utils import SharedPrefixProblem
from tests.unit_tests.test_utilities import Utils


def _local_moe_layer(**overrides):
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_submodules
    from megatron.core.transformer.moe.moe_layer import MoELayer
    from megatron.core.transformer.spec_utils import get_submodules
    from megatron.core.transformer.transformer_config import TransformerConfig

    config = dict(
        num_layers=1,
        hidden_size=64,
        ffn_hidden_size=128,
        num_attention_heads=4,
        num_moe_experts=16,
        moe_router_topk=4,
        moe_router_load_balancing_type="none",
        moe_router_score_function="sigmoid",
        moe_router_enable_expert_bias=True,
        # FP64 routing makes the dense and shared router logits of a token identical.
        moe_router_dtype="fp64",
        moe_token_dispatcher_type="alltoall",
        bf16=True,
        params_dtype=torch.bfloat16,
        add_bias_linear=False,
        use_cpu_initialization=True,
    )
    config.update(overrides)
    config = TransformerConfig(**config)
    submodules = get_submodules(
        get_gpt_layer_local_submodules(
            num_experts=config.num_moe_experts, moe_grouped_gemm=False
        ).mlp
    )
    return MoELayer(config, submodules).cuda().train()


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
class TestSharedPrefixExpertBiasCounts:
    """Logical token multiplicities through a real (compiled) TopKRouter."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

        model_parallel_cuda_manual_seed(7)
        torch.manual_seed(7)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("exclude_sequence_padding", [False, True])
    def test_router_counts_match_dense_expansion(self, exclude_sequence_padding):
        layer = _local_moe_layer()
        router = layer.router
        problem = SharedPrefixProblem(((7, (4, 3, 2)), (9, (6, 1))), padding_multiple=4)
        layout = problem.layout(forest=True)
        star = torch.randn(problem.physical_len, 1, 64, device="cuda").bfloat16()
        dense_index = torch.cat([row.star_indices() for row in problem.rows]).cuda()
        dense = star.index_select(0, dense_index)
        # Ordinary per-branch padding of the dense rows.
        padding = torch.cat(
            [
                torch.arange(row.dense_len) >= row.prefix_len + row.logical_len
                for row in problem.rows
            ]
        ).cuda()
        multiplicities = layout.padded_token_multiplicities(
            problem.physical_len, "cuda", exclude_sequence_padding=exclude_sequence_padding
        )

        def counts(hidden, **kwargs):
            router.local_tokens_per_expert.zero_()
            with torch.enable_grad():
                _, routing_map = router(hidden, **kwargs)
            return router.local_tokens_per_expert.clone(), routing_map

        shared_counts, shared_map = counts(star, token_multiplicities=multiplicities)
        dense_kwargs = {"padding_mask": padding.unsqueeze(1)} if exclude_sequence_padding else {}
        dense_counts, dense_map = counts(dense, **dense_kwargs)
        assert torch.equal(shared_map.index_select(0, dense_index), dense_map)
        torch.testing.assert_close(shared_counts, dense_counts, rtol=0, atol=0)
        expected_total = (
            sum(row.dense_len for row in problem.rows)
            - (int(padding.sum()) if exclude_sequence_padding else 0)
        ) * router.topk
        assert int(shared_counts.sum()) == expected_total
        # Without multiplicities the router keeps counting physical tokens.
        plain_counts, plain_map = counts(star)
        assert torch.equal(plain_counts, plain_map.sum(0).to(plain_counts.dtype))

    def test_dense_index_counts_use_multiplicities(self):
        """Dense ``[tokens, topk]`` indices (flex backends), including invalid ``-1`` routes."""
        layer = _local_moe_layer()
        router = layer.router
        indices = torch.tensor([[0, 3], [1, -1], [2, 15], [0, 0]], device="cuda", dtype=torch.int16)
        multiplicities = torch.tensor([3.0, 1.0, 0.0, 2.0], device="cuda")
        router.local_tokens_per_expert.zero_()
        with torch.enable_grad():
            router._apply_expert_bias(
                indices, padding_mask=None, token_multiplicities=multiplicities
            )
        expected = torch.zeros_like(router.local_tokens_per_expert)
        expected[0] = 3 + 2 + 2
        expected[3] = 3
        expected[1] = 1
        torch.testing.assert_close(router.local_tokens_per_expert, expected, rtol=0, atol=0)

    @pytest.mark.parametrize("recompute", [False, True], ids=["plain", "moe-recompute"])
    def test_moe_layer_counts_logical_tokens(self, recompute):
        """With MoE selective recompute, counting happens in backward from the closure capture."""
        overrides = {"num_moe_experts": 8, "moe_router_topk": 2, "ffn_hidden_size": 128}
        if recompute:
            overrides.update(recompute_granularity="selective", recompute_modules=["moe"])
        layer = _local_moe_layer(**overrides)
        router = layer.router
        assert layer.moe_layer_recompute == recompute
        hidden = torch.randn(40, 1, 64, device="cuda").bfloat16().requires_grad_()
        multiplicities = torch.randint(0, 4, (40,), device="cuda").float()
        router.local_tokens_per_expert.zero_()
        layer._shared_prefix_token_multiplicities = multiplicities
        try:
            output, _ = layer(hidden)
        finally:
            # The shared-prefix stack removes the scope before backward runs.
            del layer._shared_prefix_token_multiplicities
        output.float().pow(2).sum().backward()
        counted = router.local_tokens_per_expert.clone()
        router.local_tokens_per_expert.zero_()
        with torch.enable_grad():
            router(hidden.detach(), token_multiplicities=multiplicities)
        torch.testing.assert_close(counted, router.local_tokens_per_expert, rtol=0, atol=0)
