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
            await asyncio.Event().wait()  # Block forever; cancelled when test completes
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
            pytest.param(4, True, 2, 8, None, id="streaming_batched"),
            pytest.param(0, True, 2, 16, None, id="streaming_steady_state_order"),
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
        if num_slow_calls > 0 and streaming:
            # Warmup should not block on slow batches.
            batch_ids = [g.batch_id for g in groups]
            num_slow_batches = num_slow_calls // num_groups
            slow_batches = set(range(num_slow_batches))
            assert (
                batch_ids[0] not in slow_batches
            ), f"Expected first group from a fast batch, got batch_id={batch_ids[0]}"
        if streaming and num_groups > 1:
            # Verify steady-state batches arrive in sequential order.
            num_workers = gen.parallel_generation_tasks // num_groups
            steady = [g for g in groups if g.batch_id >= num_workers]
            if steady:
                batch_order = [steady[0].batch_id]
                for g in steady[1:]:
                    if g.batch_id != batch_order[-1]:
                        batch_order.append(g.batch_id)
                expected = list(range(num_workers, num_workers + len(batch_order)))
                assert batch_order == expected, f"Steady-state batches out of order: {batch_order}"

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
