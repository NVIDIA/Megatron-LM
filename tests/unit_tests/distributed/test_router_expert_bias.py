# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from megatron.core.distributed.finalize_model_grads import (
    _update_router_expert_bias,
    reset_model_temporary_tensors,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestUpdateRouterExpertBias:
    def setup_method(self):
        Utils.initialize_model_parallel(1, 1)
        self.config = SimpleNamespace(
            moe_router_enable_expert_bias=True,
            moe_router_load_balancing_type="none",
            moe_router_bias_update_rate=0.25,
        )

    def teardown_method(self):
        Utils.destroy_model_parallel()

    def _router(self, num_experts, bias_dtype=torch.float32, count_dtype=torch.int64):
        router = torch.nn.Module()
        device = torch.device("cuda", torch.cuda.current_device())
        counts = torch.arange(num_experts, device=device, dtype=count_dtype) + dist.get_rank()
        router.register_buffer("local_tokens_per_expert", counts)
        router.register_buffer(
            "expert_bias", torch.arange(num_experts, device=device, dtype=bias_dtype) / 8
        )
        return router

    def _expected_bias(self, router):
        counts = router.local_tokens_per_expert.clone()
        dist.all_reduce(counts, group=dist.group.WORLD)
        direction = torch.sign(counts.sum() - counts * counts.numel())
        return router.expert_bias + direction.to(router.expert_bias.dtype) * 0.25

    @pytest.mark.parametrize("expert_counts", [(4, 8, 4), (8, 4, 8), (4, 4), (4,)])
    def test_updates_each_layer_and_preserves_local_counts(self, expert_counts):
        routers = [self._router(n) for n in expert_counts]
        for index, router in enumerate(routers):
            router.local_tokens_per_expert.copy_(router.local_tokens_per_expert.roll(shifts=index))
            router.expert_bias.add_(index / 2)
        expected = [self._expected_bias(router) for router in routers]
        local_counts = [router.local_tokens_per_expert.clone() for router in routers]
        bias_ptrs = [router.expert_bias.data_ptr() for router in routers]

        # Different model chunks can contain different expert counts.
        _update_router_expert_bias(routers, self.config, tp_dp_cp_group=dist.group.WORLD)

        for router, bias, counts, ptr in zip(routers, expected, local_counts, bias_ptrs):
            torch.testing.assert_close(router.expert_bias, bias, rtol=0, atol=0)
            torch.testing.assert_close(router.local_tokens_per_expert, counts, rtol=0, atol=0)
            assert router.expert_bias.data_ptr() == ptr
            assert router.local_tokens_per_expert.dtype == torch.int64

    def test_bucket_order_is_independent_of_first_expert_count(self):
        routers = {n: self._router(n) for n in (4, 8)}
        expected = {n: self._expected_bias(router) for n, router in routers.items()}
        order = (8, 4) if dist.get_rank() % 2 == 0 else (4, 8)

        _update_router_expert_bias(
            [routers[n] for n in order], self.config, tp_dp_cp_group=dist.group.WORLD
        )

        for n, router in routers.items():
            torch.testing.assert_close(router.expert_bias, expected[n], rtol=0, atol=0)

    def test_large_integer_counts_keep_update_direction(self):
        routers = [self._router(n) for n in (4, 8)]
        for router in routers:
            router.local_tokens_per_expert.fill_(2**24)
            router.local_tokens_per_expert[0] += 1
            router.local_tokens_per_expert[1] -= 1
        expected = [router.expert_bias.clone() for router in routers]
        for bias in expected:
            bias[0] -= 0.25
            bias[1] += 0.25

        _update_router_expert_bias(routers, self.config, tp_dp_cp_group=dist.group.WORLD)

        for router, bias in zip(routers, expected):
            torch.testing.assert_close(router.expert_bias, bias, rtol=0, atol=0)
            assert router.local_tokens_per_expert.dtype == torch.int64

    def test_mixed_dtypes_do_not_promote_integer_counts(self, monkeypatch):
        routers = [
            self._router(4, torch.float32, torch.int64),
            self._router(4, torch.float64, torch.int64),
            self._router(4, torch.float64, torch.float64),
            self._router(8, torch.float32, torch.int64),
        ]
        routers[0].local_tokens_per_expert.fill_(2**53)
        routers[0].local_tokens_per_expert[0] += 1
        routers[0].local_tokens_per_expert[1] -= 1
        expected = [self._expected_bias(router) for router in routers]
        all_reduce = dist.all_reduce
        reduced_dtypes = []

        def record_all_reduce(tensor, *args, **kwargs):
            reduced_dtypes.append(tensor.dtype)
            return all_reduce(tensor, *args, **kwargs)

        monkeypatch.setattr(dist, "all_reduce", record_all_reduce)
        _update_router_expert_bias(routers, self.config, tp_dp_cp_group=dist.group.WORLD)

        assert reduced_dtypes.count(torch.int64) == 3
        assert reduced_dtypes.count(torch.float64) == 1
        for router, bias in zip(routers, expected):
            torch.testing.assert_close(router.expert_bias, bias, rtol=0, atol=0)

    def test_homogeneous_layers_share_one_collective(self, monkeypatch):
        routers = [self._router(4) for _ in range(3)]
        expected = [self._expected_bias(router) for router in routers]
        all_reduce = dist.all_reduce
        reduced_shapes = []

        def record_all_reduce(tensor, *args, **kwargs):
            reduced_shapes.append(tuple(tensor.shape))
            return all_reduce(tensor, *args, **kwargs)

        monkeypatch.setattr(dist, "all_reduce", record_all_reduce)
        _update_router_expert_bias(routers, self.config, tp_dp_cp_group=dist.group.WORLD)

        assert reduced_shapes == [(3, 4)]
        for router, bias in zip(routers, expected):
            torch.testing.assert_close(router.expert_bias, bias, rtol=0, atol=0)

    def test_skips_eval_frozen_and_missing_bias(self):
        active = self._router(4)
        evaluation = self._router(8).eval()
        frozen = self._router(16)
        frozen.frozen_expert_bias = True
        no_bias = torch.nn.Module()
        no_bias.expert_bias = None
        routers = [evaluation, active, frozen, no_bias, torch.nn.Linear(2, 2)]
        expected = self._expected_bias(active)
        skipped_biases = [evaluation.expert_bias.clone(), frozen.expert_bias.clone()]

        _update_router_expert_bias(routers, self.config, tp_dp_cp_group=dist.group.WORLD)

        torch.testing.assert_close(active.expert_bias, expected, rtol=0, atol=0)
        for router, bias in zip((evaluation, frozen), skipped_biases):
            torch.testing.assert_close(router.expert_bias, bias, rtol=0, atol=0)

    def test_no_active_routers_do_not_reduce(self, monkeypatch):
        evaluation = self._router(8).eval()
        frozen = self._router(4)
        frozen.frozen_expert_bias = True

        def unexpected_all_reduce(*args, **kwargs):
            pytest.fail("No collective is needed when there are no active routers")

        monkeypatch.setattr(dist, "all_reduce", unexpected_all_reduce)
        for routers in ([], [torch.nn.Linear(2, 2)], [evaluation, frozen]):
            _update_router_expert_bias(routers, self.config, tp_dp_cp_group=dist.group.WORLD)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_real_heterogeneous_routers(self, dtype):
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        routers = []
        for num_experts in (4, 8):
            config = TransformerConfig(
                num_layers=2,
                hidden_size=512,
                num_attention_heads=8,
                ffn_hidden_size=1024,
                num_moe_experts=num_experts,
                moe_router_topk=2,
                moe_router_score_function="sigmoid",
                moe_router_load_balancing_type="none",
                moe_router_enable_expert_bias=True,
                moe_router_bias_update_rate=0.25,
                use_cpu_initialization=True,
                params_dtype=dtype,
                bf16=dtype == torch.bfloat16,
                add_bias_linear=False,
            )
            routers.append(TopKRouter(config, pg_collection=pg_collection).cuda())
        hidden = torch.randn(256, 1, 512, device="cuda", dtype=dtype, requires_grad=True)
        for router in routers:
            probabilities, _ = router(hidden)
            probabilities.square().sum().backward()
            assert router.local_tokens_per_expert.sum().item() == 256 * 2
        expected = [self._expected_bias(router) for router in routers]

        _update_router_expert_bias(routers, self.config, tp_dp_cp_group=dist.group.WORLD)

        for router, bias in zip(routers, expected):
            torch.testing.assert_close(router.expert_bias, bias, rtol=0, atol=0)
            assert router.local_tokens_per_expert.dtype == torch.int64
        reset_model_temporary_tensors(self.config, routers)
        for router in routers:
            assert torch.count_nonzero(router.local_tokens_per_expert).item() == 0
