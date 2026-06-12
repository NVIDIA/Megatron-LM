# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
from unittest.mock import MagicMock

import pytest

from megatron.rl.agent.api import (
    GroupedRolloutGenerator,
    GroupedRolloutRequest,
    Rollout,
    RolloutGenerator,
    RolloutGroup,
)
from megatron.rl.agent.weighted_multi_task import AgentConfig, WeightedMultiTask
from megatron.rl.inference import ReturnsRaw


class MockGenerator(RolloutGenerator, GroupedRolloutGenerator):
    """Mock generator with configurable per-call delays."""

    def __init__(self, env_id="test", num_slow_calls=0, **kwargs):
        super().__init__(**kwargs)
        self.env_id = env_id
        self.num_slow_calls = num_slow_calls
        self._call_count = 0

    async def rollout(self, request):
        raise NotImplementedError

    async def group_rollout(self, request):
        idx = self._call_count
        self._call_count += 1
        if idx < self.num_slow_calls:
            await asyncio.sleep(0.03)
        return [
            Rollout(
                trajectory=[f"t{idx}"],
                reward=float(idx),
                env_id=self.env_id,
                policy_epoch=[[(0, 0)]],
                kv_cache_epoch=[[(0, 0)]],
                num_evictions=[0],
            )
            for _ in range(request.rollouts_per_group)
        ]


class TestGroupedRollouts:

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "case",
        [
            pytest.param(
                {"num_slow_calls": 0, "streaming": False, "num_groups": 8, "expected_count": 8},
                id="non_batched",
            ),
            pytest.param(
                {"num_slow_calls": 0, "streaming": False, "num_groups": 4, "expected_count": 4},
                id="non_streaming_fewer_than_parallel",
            ),
            pytest.param(
                {
                    "num_slow_calls": 4,
                    "streaming": True,
                    "num_groups": 2,
                    "expected_count": 8,
                    "expected_batch_ids": [0, 0, 1, 1, 2, 2, 3, 3],
                    "expected_index_in_batch": [0, 1, 0, 1, 0, 1, 0, 1],
                },
                id="batched_submission_order",
            ),
            pytest.param(
                {"num_slow_calls": 0, "streaming": True, "num_groups": 1, "expected_count": 10},
                id="streaming",
            ),
            pytest.param(
                {
                    "num_slow_calls": 4,
                    "streaming": True,
                    "num_groups": 1,
                    "expected_count": 8,
                    "enforce_order": False,
                    "expected_trajectory_prefix": [f"t{i}" for i in range(4, 8)],
                },
                id="group_submit_group_consume_completion_order",
            ),
            pytest.param(
                {
                    "num_slow_calls": 4,
                    "streaming": True,
                    "num_groups": 1,
                    "expected_count": 8,
                    "enforce_order": True,
                    "expected_batch_ids": list(range(8)),
                    "expected_index_in_batch": [0] * 8,
                    "expected_trajectories": [f"t{i}" for i in range(8)],
                },
                id="group_submit_batch_consume_submission_order",
            ),
        ],
    )
    async def test_get_grouped_rollouts(self, case):
        gen = MockGenerator(
            parallel_generation_tasks=case.get("parallel_generation_tasks", 8),
            num_slow_calls=case.get("num_slow_calls", 0),
        )

        request = GroupedRolloutRequest(
            num_groups=case["num_groups"],
            rollouts_per_group=case.get("rollouts_per_group", 1),
            inference_interface=MagicMock(spec=ReturnsRaw),
            streaming=case["streaming"],
            enforce_order=case.get("enforce_order", case["num_groups"] > 1),
            filter_groups_with_same_reward=case.get("filter_groups_with_same_reward", False),
        )

        async def collect_groups():
            groups = []
            async for group in gen.get_grouped_rollouts(request):
                groups.append(group)
                if request.streaming and len(groups) >= case["expected_count"]:
                    break
            return groups

        if "timeout" in case:
            groups = await asyncio.wait_for(collect_groups(), timeout=case["timeout"])
        else:
            groups = await collect_groups()

        has_order_expectations = any(
            key in case
            for key in (
                "expected_batch_ids",
                "expected_index_in_batch",
                "expected_trajectory_prefix",
                "expected_trajectories",
            )
        )
        if not has_order_expectations:
            assert len(groups) == case["expected_count"]
        if "expected_batch_ids" in case:
            assert [g.batch_id for g in groups] == case["expected_batch_ids"]
        if "expected_index_in_batch" in case:
            assert [g.index_in_batch for g in groups] == case["expected_index_in_batch"]
        if "expected_trajectory_prefix" in case:
            expected = case["expected_trajectory_prefix"]
            assert [group[0].trajectory[0] for group in groups[: len(expected)]] == expected
        if "expected_trajectories" in case:
            assert [group[0].trajectory[0] for group in groups] == case["expected_trajectories"]

    @pytest.mark.asyncio
    async def test_weighted_multi_task(self):
        configs = [
            AgentConfig(agent_type=MockGenerator, agent_args={"env_id": "a"}, weight=3.0),
            AgentConfig(agent_type=MockGenerator, agent_args={"env_id": "b"}, weight=1.0),
        ]
        mt = WeightedMultiTask(configs)
        mt.parallel_generation_tasks = 4

        captured = []
        for agent in mt.agents:
            original = agent.get_grouped_rollouts

            async def spy(req, orig=original):
                captured.append(req)
                async for group in orig(req):
                    yield group

            agent.get_grouped_rollouts = spy

        request = GroupedRolloutRequest(
            num_groups=4,
            rollouts_per_group=1,
            inference_interface=MagicMock(spec=ReturnsRaw),
            streaming=False,
            enforce_order=False,
        )
        groups = []
        async for group in mt.get_grouped_rollouts(request):
            groups.append(group)

        assert len(groups) == 4
        # Weights 3:1 → agent "a" produces 3 groups, agent "b" produces 1.
        env_ids = [g[0].env_id for g in groups]
        assert sorted(env_ids) == ["a", "a", "a", "b"]
        for sub_req in captured:
            assert sub_req.num_groups in (1, 3)  # distributed proportionally by weight
            assert sub_req.enforce_order == request.enforce_order
            assert sub_req.streaming == request.streaming
