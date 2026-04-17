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
        "num_slow_calls, streaming, num_groups, expected_count, expected_batch_ids",
        [
            pytest.param(0, False, 8, 8, None, id="non_batched"),
            pytest.param(0, False, 4, 4, None, id="non_streaming_fewer_than_parallel"),
            pytest.param(4, True, 2, 8, [0, 0, 1, 1, 2, 2, 3, 3], id="batched_submission_order"),
            pytest.param(0, True, 1, 10, None, id="streaming"),
        ],
    )
    async def test_get_grouped_rollouts(
        self, num_slow_calls, streaming, num_groups, expected_count, expected_batch_ids
    ):
        gen = MockGenerator(parallel_generation_tasks=8, num_slow_calls=num_slow_calls)
        request = GroupedRolloutRequest(
            num_groups=num_groups,
            rollouts_per_group=1,
            inference_interface=MagicMock(spec=ReturnsRaw),
            streaming=streaming,
            enforce_order=num_groups > 1,
        )
        groups = []
        async for group in gen.get_grouped_rollouts(request):
            groups.append(group)
            if request.streaming and len(groups) >= expected_count:
                break

        assert len(groups) == expected_count
        if expected_batch_ids is not None:
            assert [g.batch_id for g in groups] == expected_batch_ids

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "agents, num_groups, pgt, streaming, expected_count, expected_env_ids",
        [
            pytest.param(
                [("a", 3.0), ("b", 1.0)], 4, 4, False, 4, ["a", "a", "a", "b"], id="unequal w"
            ),
            pytest.param(
                [("a", 3), ("b", 3), ("c", 4)], 4, 100, False, 4, ["a", "b", "c", "c"], id="no_hang"
            ),
            pytest.param(
                [("a", 1.0), ("b", 1.0)], 2, 3, False, 2, ["a", "b"], id="small_pgt"
            ),
            pytest.param(
                [("a", 1.0), ("b", 1.0)], 5, 100, True, 10, ["a", "b"] * 5, id="stream_interleave"
            ),
            pytest.param(
                [("a", 2), ("b", 3), ("c", 5)], 3, 3, True, 6, ["a", "b", "c"] * 2, id="jagged"
            ),
        ],
    )
    async def test_weighted_multi_task(
        self, agents, num_groups, pgt, streaming, expected_count, expected_env_ids
    ):
        configs = [
            AgentConfig(agent_type=MockGenerator, agent_args={"env_id": env_id}, weight=weight)
            for env_id, weight in agents
        ]
        mt = WeightedMultiTask(configs)
        mt.parallel_generation_tasks = pgt

        request = GroupedRolloutRequest(
            num_groups=num_groups,
            rollouts_per_group=1,
            inference_interface=MagicMock(spec=ReturnsRaw),
            streaming=streaming,
            enforce_order=False,
        )
        groups = []
        async for group in mt.get_grouped_rollouts(request):
            groups.append(group)
            if streaming and len(groups) >= expected_count:
                break

        assert len(groups) == expected_count
        env_ids = [g[0].env_id for g in groups]
        assert env_ids == expected_env_ids

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "agents, num_groups, pgt",
        [pytest.param([("a", 0.01), ("b", 0.01), ("c", 0.98)], 3, 100, id="extreme_weights")],
    )
    async def test_weighted_multi_task_error(self, agents, num_groups, pgt):
        configs = [
            AgentConfig(agent_type=MockGenerator, agent_args={"env_id": env_id}, weight=weight)
            for env_id, weight in agents
        ]
        mt = WeightedMultiTask(configs)
        mt.parallel_generation_tasks = pgt

        request = GroupedRolloutRequest(
            num_groups=num_groups,
            rollouts_per_group=1,
            inference_interface=MagicMock(spec=ReturnsRaw),
            streaming=False,
            enforce_order=False,
        )
        with pytest.raises(ValueError, match="too small for the configured weights"):
            async for _ in mt.get_grouped_rollouts(request):
                pass
