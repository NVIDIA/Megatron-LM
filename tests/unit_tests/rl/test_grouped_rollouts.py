# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
from unittest.mock import MagicMock

import pytest

from megatron.rl.agent.api import GroupedRolloutGenerator, GroupedRolloutRequest, Rollout
from megatron.rl.agent.weighted_multi_task import AgentConfig, WeightedMultiTask
from megatron.rl.inference import ReturnsRaw


class MockGenerator(GroupedRolloutGenerator):
    """Mock generator with configurable per-call delays."""

    def __init__(self, env_id="test", slow_first=0, **kwargs):
        super().__init__(**kwargs)
        self.env_id = env_id
        self.slow_first = slow_first
        self._call_count = 0

    async def group_rollout(self, request):
        idx = self._call_count
        self._call_count += 1
        if idx < self.slow_first:
            await asyncio.sleep(0.03)
        return [
            Rollout(
                trajectory=[f"t{idx}"],
                reward=float(idx),
                env_id=self.env_id,
                policy_staleness=[[0]],
                kv_cache_staleness=[[0]],
                completed_at_step=[0],
                num_evictions=[0],
            )
            for _ in range(request.rollouts_per_group)
        ]


class TestGroupedRollouts:

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "slow_first, streaming, generation_batch_size, expected_count, expected_batch_ids",
        [
            pytest.param(0, False, 1, 4, None, id="non_batched"),
            pytest.param(4, False, 2, 4, [0, 0, 1, 1], id="batched_submission_order"),
            pytest.param(0, True, 1, 10, None, id="streaming"),
        ],
    )
    async def test_get_grouped_rollouts(
        self, slow_first, streaming, generation_batch_size, expected_count, expected_batch_ids
    ):
        gen = MockGenerator(parallel_generation_tasks=8, slow_first=slow_first)
        request = GroupedRolloutRequest(
            num_groups=4,
            rollouts_per_group=1,
            inference_interface=MagicMock(spec=ReturnsRaw),
            streaming=streaming,
            generation_batch_size=generation_batch_size,
            enforce_order=generation_batch_size > 1,
        )
        groups = []
        async for group in gen.get_grouped_rollouts(request):
            groups.append(group)
            if request.streaming and len(groups) >= expected_count:
                break

        assert len(groups) == expected_count
        if expected_batch_ids is not None:
            assert [g[0].batch_id for g in groups] == expected_batch_ids

    @pytest.mark.asyncio
    async def test_weighted_multi_task(self):
        configs = [
            AgentConfig(agent_type=MockGenerator, agent_args={"env_id": "a"}, weight=3.0),
            AgentConfig(agent_type=MockGenerator, agent_args={"env_id": "b"}, weight=1.0),
        ]
        mt = WeightedMultiTask(configs)
        mt.parallel_generation_tasks = 8

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
            generation_batch_size=1,
            enforce_order=False,
        )
        groups = []
        async for group in mt.get_grouped_rollouts(request):
            groups.append(group)

        assert len(groups) == 4
        assert [g[0].env_id for g in groups] == ["a", "a", "a", "b"]
        for sub_req in captured:
            assert sub_req.generation_batch_size == request.generation_batch_size
            assert sub_req.enforce_order == request.enforce_order
            assert sub_req.streaming == request.streaming
