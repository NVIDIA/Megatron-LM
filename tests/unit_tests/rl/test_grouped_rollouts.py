# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from megatron.rl.agent.api import (
    GroupedRolloutGenerator,
    GroupedRolloutRequest,
    Rollout,
    RolloutGenerator,
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
        self.submission_gate_seen = False

    async def rollout(self, request):
        raise NotImplementedError

    async def group_rollout(self, request, submission_gate=None):
        if submission_gate is not None:
            self.submission_gate_seen = True
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
    @pytest.mark.parametrize("field", ["submission_granularity", "consumption_granularity"])
    def test_grouped_rollout_request_rejects_unknown_granularity(self, field):
        request_kwargs = {
            "num_groups": 1,
            "rollouts_per_group": 1,
            "inference_interface": MagicMock(spec=ReturnsRaw),
            field: "X",
        }
        with pytest.raises(ValidationError) as exc_info:
            GroupedRolloutRequest(**request_kwargs)
        assert any(error["loc"] == (field,) for error in exc_info.value.errors())

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        (
            "num_slow_calls, streaming, num_groups, submission_granularity, "
            "consumption_granularity, expected_count, expected_batch_ids, "
            "expected_trajectories"
        ),
        [
            pytest.param(0, False, 8, "B", "B", 8, None, None, id="non_batched"),
            pytest.param(
                0, False, 4, "B", "B", 4, None, None, id="non_streaming_fewer_than_parallel"
            ),
            pytest.param(
                4,
                True,
                2,
                "B",
                "B",
                8,
                [0, 0, 1, 1, 2, 2, 3, 3],
                None,
                id="batched_submission_order",
            ),
            pytest.param(0, True, 1, "G", "B", 10, None, None, id="streaming"),
            pytest.param(
                4,
                True,
                1,
                "G",
                "G",
                8,
                None,
                [f"t{i}" for i in range(4, 8)],
                id="group_consume_completion_order",
            ),
            pytest.param(
                4,
                True,
                1,
                "G",
                "B",
                8,
                list(range(8)),
                [f"t{i}" for i in range(8)],
                id="batch_consume_submission_order",
            ),
        ],
    )
    async def test_get_grouped_rollouts(
        self,
        num_slow_calls,
        streaming,
        num_groups,
        submission_granularity,
        consumption_granularity,
        expected_count,
        expected_batch_ids,
        expected_trajectories,
    ):
        gen = MockGenerator(parallel_generation_tasks=8, num_slow_calls=num_slow_calls)
        request = GroupedRolloutRequest(
            num_groups=num_groups,
            rollouts_per_group=1,
            inference_interface=MagicMock(spec=ReturnsRaw),
            streaming=streaming,
            submission_granularity=submission_granularity,
            consumption_granularity=consumption_granularity,
        )

        groups = []
        async for group in gen.get_grouped_rollouts(request):
            groups.append(group)
            if request.streaming and len(groups) >= expected_count:
                break

        assert len(groups) == expected_count
        if expected_batch_ids is not None:
            assert [g.batch_id for g in groups] == expected_batch_ids
        if expected_trajectories is not None:
            trajectories = [group[0].trajectory[0] for group in groups]
            assert trajectories[: len(expected_trajectories)] == expected_trajectories

    @pytest.mark.asyncio
    async def test_rollout_submission_granularity_passes_submission_gate(self):
        gen = MockGenerator(parallel_generation_tasks=2)
        request = GroupedRolloutRequest(
            num_groups=1,
            rollouts_per_group=2,
            inference_interface=MagicMock(spec=ReturnsRaw),
            streaming=True,
            submission_granularity="R",
            consumption_granularity="B",
        )

        groups = []
        async for group in gen.get_grouped_rollouts(request):
            groups.append(group)
            break

        assert len(groups) == 1
        assert len(groups[0]) == 2
        assert gen.submission_gate_seen

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "submission_granularity, consumption_granularity, expected_parallel_generation_tasks",
        [
            pytest.param("B", "B", [4, 4], id="batch_submission"),
            pytest.param("G", "G", [3, 1], id="group_submission"),
        ],
    )
    async def test_weighted_multi_task(
        self, submission_granularity, consumption_granularity, expected_parallel_generation_tasks
    ):
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
            submission_granularity=submission_granularity,
            consumption_granularity=consumption_granularity,
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
            assert sub_req.streaming == request.streaming
            assert sub_req.submission_granularity == request.submission_granularity
            assert sub_req.consumption_granularity == request.consumption_granularity
        assert [agent.parallel_generation_tasks for agent in mt.agents] == (
            expected_parallel_generation_tasks
        )
