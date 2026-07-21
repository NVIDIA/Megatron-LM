# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
from unittest.mock import MagicMock

import numpy as np
import pytest
from pydantic import ValidationError

from megatron.rl.agent.api import (
    GroupedRolloutGenerator,
    GroupedRolloutRequest,
    GroupRolloutParams,
    Rollout,
    RolloutGenerator,
    RolloutRequest,
    _SubmissionGate,
)
from megatron.rl.agent.reward_only_agent import RewardOnlyAgent
from megatron.rl.agent.weighted_multi_task import (
    AgentConfig,
    PgtRebalanceConfig,
    WeightedMultiTask,
    _PgtRebalancer,
)
from megatron.rl.inference import InferenceResponse, LLMChatMessage, ReturnsRaw


class MockInferenceInterface(ReturnsRaw):
    """Mock raw-text inference interface with configurable per-prompt delays."""

    num_slow_calls: int = 0
    active_requests: int = 0
    max_active_requests: int = 0

    async def base_generate(self, request):
        prompt = request.prompt[0].content
        idx = int(prompt.removeprefix("t"))
        self.active_requests += 1
        self.max_active_requests = max(self.max_active_requests, self.active_requests)
        try:
            if idx < self.num_slow_calls:
                await asyncio.sleep(0.03)
            else:
                await asyncio.sleep(0)
            return InferenceResponse(
                response=LLMChatMessage(role="assistant", content=prompt),
                raw_text=prompt,
                finish_reason="stop",
                policy_epoch=[(0, 0)],
                kv_cache_epoch=[(0, 0)],
                num_evictions=0,
            )
        finally:
            self.active_requests -= 1


class MockGenerator(RolloutGenerator, GroupedRolloutGenerator):
    """Mock generator with configurable per-call delays."""

    def __init__(self, env_id="test", **kwargs):
        super().__init__(**kwargs)
        self.env_id = env_id
        self._call_count = 0
        self.prepare_group_rollout_calls = 0

    async def get_reward_rollouts(self, request):
        raise NotImplementedError

    async def get_rollout_response(self, request, inference_request):
        return await request.inference_interface.agenerate(inference_request)

    async def prepare_group_rollout(self, request):
        idx = self._call_count
        self._call_count += 1
        self.prepare_group_rollout_calls += 1
        inference_request = request.inference_interface.prepare_request(
            f"t{idx}", request.generation_args
        )

        async def build_rollout(response):
            response_idx = int(response.response.content.removeprefix("t"))
            return Rollout(
                trajectory=[response.raw_text],
                reward=float(response_idx),
                env_id=self.env_id,
                policy_epoch=[response.policy_epoch],
                kv_cache_epoch=[response.kv_cache_epoch],
                num_evictions=[response.num_evictions],
            )

        return GroupRolloutParams(inference_request=inference_request, build_rollout=build_rollout)


class CountingRewardAgent(RewardOnlyAgent):
    """Minimal RewardOnlyAgent: prompts t0, t1, ... and reward = echoed index."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.env_id = "reward-test"
        self._prompt_count = 0

    async def get_prompt(self, validation):
        idx = self._prompt_count
        self._prompt_count += 1
        return f"t{idx}", {"idx": idx}

    async def get_reward(self, response, golden, finish_reason):
        return float(int(response.removeprefix("t")) == golden["idx"])


async def _flush(rounds: int = 50):
    """Let pipeline stage tasks settle (mock inference is zero-delay)."""
    for _ in range(rounds):
        await asyncio.sleep(0)


class TestSubmissionGate:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("submission", ["R", "G", "B"])
    async def test_release_requires_matching_granularity(self, submission):
        gate = _SubmissionGate(capacity=1, submission=submission)
        await gate.acquire_for(submission)
        assert gate.held == 1
        for granularity in ("R", "G", "B"):
            if granularity == submission:
                continue
            gate.release_for(granularity)
        assert gate.held == 1
        assert gate.release_calls == 0
        gate.release_for(submission)
        assert gate.held == 0
        assert gate.release_calls == 1

    @pytest.mark.asyncio
    async def test_grow_wakes_blocked_acquirer(self):
        gate = _SubmissionGate(capacity=1, submission="G")
        await gate.acquire_for("G")
        blocked = asyncio.create_task(gate.acquire_for("G"))
        await _flush()
        assert not blocked.done()
        gate.set_capacity(2)
        await asyncio.wait_for(blocked, timeout=5)
        assert gate.held == 2

    @pytest.mark.asyncio
    async def test_shrink_below_held_never_revokes(self):
        gate = _SubmissionGate(capacity=4, submission="G")
        for _ in range(3):
            await gate.acquire_for("G")
        gate.set_capacity(2)
        # In-flight slots are untouched; only new acquires block.
        assert gate.held == 3
        blocked = asyncio.create_task(gate.acquire_for("G"))
        await _flush()
        assert not blocked.done()
        gate.release_for("G")  # held 2, still at the new capacity
        await _flush()
        assert not blocked.done()
        gate.release_for("G")  # held 1 < 2: the waiter proceeds
        await asyncio.wait_for(blocked, timeout=5)
        assert gate.held == 2

    def test_set_capacity_clamps_to_one(self):
        gate = _SubmissionGate(capacity=4, submission="G")
        gate.set_capacity(0)
        assert gate.capacity == 1


class TestConsumptionRelease:
    """G-submission gate slots must recycle on trainer consumption, not assembly."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "consumption_granularity, num_groups",
        [
            pytest.param("G", 1, id="group_consumption"),
            pytest.param("B", 2, id="batch_consumption"),
        ],
    )
    async def test_group_submission_stalls_until_consumption(
        self, consumption_granularity, num_groups
    ):
        capacity = 4
        gen = MockGenerator(parallel_generation_tasks=capacity)
        request = GroupedRolloutRequest(
            num_groups=num_groups,
            rollouts_per_group=1,
            inference_interface=MockInferenceInterface(),
            streaming=True,
            submission_granularity="G",
            consumption_granularity=consumption_granularity,
        )
        it = gen.get_grouped_rollouts(request)
        try:
            for pulled in range(1, capacity + 3):
                # wait_for turns the deadlock failure mode (a slot never freed)
                # into a test failure instead of a hang.
                await asyncio.wait_for(anext(it), timeout=10)
                await _flush()
                # Each yield frees exactly one group slot on the consumer's next
                # resume, so submission tracks consumption with a one-slot skew
                # (the release for the latest pull hasn't fired yet). On
                # assembly-release semantics this runs away unbounded; if no
                # consume-site release existed, the loop would deadlock at
                # `pulled == capacity + 1`.
                assert gen.prepare_group_rollout_calls == capacity + pulled - 1
        finally:
            await it.aclose()

    @pytest.mark.asyncio
    async def test_batch_submission_releases_once_per_batch(self):
        gen = MockGenerator(parallel_generation_tasks=1)
        request = GroupedRolloutRequest(
            num_groups=2,
            rollouts_per_group=1,
            inference_interface=MockInferenceInterface(),
            streaming=True,
            submission_granularity="B",
            consumption_granularity="B",
        )
        it = gen.get_grouped_rollouts(request)
        try:
            await asyncio.wait_for(anext(it), timeout=10)
            await asyncio.wait_for(anext(it), timeout=10)
            await _flush()
            gate = gen._active_pipeline.gate
            # Batch 0 fully yielded but the consumer hasn't come back yet: its
            # single batch slot is still held (a per-group release here would
            # show release_calls == 2 and prepared == 4).
            assert gate.release_calls == 0
            assert gen.prepare_group_rollout_calls == 2
            await asyncio.wait_for(anext(it), timeout=10)
            await _flush()
            assert gate.release_calls == 1
            assert gen.prepare_group_rollout_calls == 4
        finally:
            await it.aclose()


class TestRewardRollouts:
    @pytest.mark.asyncio
    async def test_get_reward_rollouts_matches_per_rollout_composition(self):
        agent = CountingRewardAgent()
        request = RolloutRequest(num_rollouts=4, inference_interface=MockInferenceInterface())
        rollouts = await agent.get_reward_rollouts(request)
        assert len(rollouts) == 4
        assert sorted(r.trajectory[0] for r in rollouts) == ["t0", "t1", "t2", "t3"]
        assert all(r.reward == 1.0 for r in rollouts)
        assert all(r.env_id == "reward-test" for r in rollouts)


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
        "num_groups, submission_granularity, consumption_granularity",
        [
            pytest.param(1, "B", "B", id="num_groups_1_batch"),
            pytest.param(4, "G", "G", id="num_groups_gt_1_group"),
            pytest.param(4, "R", "B", id="num_groups_gt_1_rollout"),
        ],
    )
    async def test_filter_groups_with_same_reward_rejected(
        self, num_groups, submission_granularity, consumption_granularity
    ):
        gen = MockGenerator(parallel_generation_tasks=8)
        request = GroupedRolloutRequest(
            num_groups=num_groups,
            rollouts_per_group=2,
            inference_interface=MockInferenceInterface(),
            filter_groups_with_same_reward=True,
            submission_granularity=submission_granularity,
            consumption_granularity=consumption_granularity,
        )
        with pytest.raises(AssertionError, match="filter_groups_with_same_reward"):
            async for _ in gen.get_grouped_rollouts(request):
                pass

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
        gen = MockGenerator(parallel_generation_tasks=8)
        request = GroupedRolloutRequest(
            num_groups=num_groups,
            rollouts_per_group=1,
            inference_interface=MockInferenceInterface(num_slow_calls=num_slow_calls),
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
    async def test_rollout_submission_granularity_limits_inference_concurrency(self):
        gen = MockGenerator(parallel_generation_tasks=2)
        inference_interface = MockInferenceInterface(num_slow_calls=100)
        request = GroupedRolloutRequest(
            num_groups=1,
            rollouts_per_group=4,
            inference_interface=inference_interface,
            streaming=True,
            submission_granularity="R",
            consumption_granularity="B",
        )

        groups = []
        async for group in gen.get_grouped_rollouts(request):
            groups.append(group)
            break

        assert len(groups) == 1
        assert len(groups[0]) == 4
        assert inference_interface.max_active_requests <= gen.parallel_generation_tasks

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
            inference_interface=MockInferenceInterface(),
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

    @pytest.mark.parametrize(
        "num_groups, all_envs_active",
        [
            pytest.param(1, False, id="num_groups_1_starves_an_env"),
            pytest.param(8, True, id="trainer_batch_size_keeps_all_envs_active"),
        ],
    )
    def test_multi_env_distribution_requires_num_groups_above_one(
        self, num_groups, all_envs_active
    ):
        """Regression for the removed ``num_groups=1`` streaming override.

        With multiple weighted environments, ``num_groups=1`` hands the single
        group to one environment and leaves the other with zero groups. It also
        collapses ``agent_slots`` (computed without remainder distribution) to all
        zeros, so ``np.gcd.reduce`` is 0 and the per-agent slot counts become
        ``nan`` -- which stalls ``get_grouped_rollouts``. Keeping ``num_groups`` at
        the trainer batch size (> 1) keeps every environment active with a valid,
        non-zero slot count.
        """
        configs = [
            AgentConfig(agent_type=MockGenerator, agent_args={"env_id": "a"}, weight=3.0),
            AgentConfig(agent_type=MockGenerator, agent_args={"env_id": "b"}, weight=1.0),
        ]
        mt = WeightedMultiTask(configs)

        agent_groups = mt._distribute_counts(num_groups)
        agent_slots = mt._distribute_counts(num_groups, distribute_remainder=False)

        assert all(groups > 0 for groups in agent_groups) is all_envs_active
        if all_envs_active:
            assert min(agent_slots) > 0
            assert np.gcd.reduce(agent_slots) > 0
        else:
            assert min(agent_groups) == 0
            assert all(slots == 0 for slots in agent_slots)
            assert np.gcd.reduce(agent_slots) == 0


class TestPgtAllocation:
    def test_slower_env_gets_proportionally_more(self):
        # Equal weights, 1.2 min vs 1.0 min: the slower env gets 1.2x slots.
        new = WeightedMultiTask._compute_pgt_allocation(
            [0.5, 0.5], [1.2, 1.0], 22, [11, 11], min_pgts=[1, 1], max_step_fraction=0.25
        )
        assert new == [12, 10]

    def test_weight_and_duration_combine(self):
        # weight 3:1, equal durations: allocation follows the weights.
        new = WeightedMultiTask._compute_pgt_allocation(
            [0.75, 0.25], [1.0, 1.0], 8, [6, 2], min_pgts=[1, 1], max_step_fraction=1.0
        )
        assert new == [6, 2]

    @pytest.mark.parametrize("total", [7, 8, 23])
    def test_sum_invariant_under_rounding(self, total):
        current = [total // 2, total - total // 2]
        new = WeightedMultiTask._compute_pgt_allocation(
            [0.5, 0.5], [1.3, 0.7], total, current, min_pgts=[1, 1], max_step_fraction=1.0
        )
        assert sum(new) == total

    def test_all_unknown_durations_keep_current(self):
        assert WeightedMultiTask._compute_pgt_allocation(
            [0.5, 0.5], [None, None], 8, [5, 3], min_pgts=[1, 1], max_step_fraction=1.0
        ) == [5, 3]

    def test_unknown_duration_falls_back_to_weighted_mean(self):
        # An env with no estimate is treated as average-speed. Envs 0 and 1
        # are known (3.0 vs 1.0 → mean 2.0); env 2 is unknown, so it lands
        # between them instead of defaulting to fastest or slowest.
        new = WeightedMultiTask._compute_pgt_allocation(
            [1 / 3, 1 / 3, 1 / 3],
            [3.0, 1.0, None],
            12,
            [4, 4, 4],
            min_pgts=[1, 1, 1],
            max_step_fraction=1.0,
        )
        assert new[0] > new[2] > new[1]
        assert sum(new) == 12

    def test_min_pgts_respected(self):
        new = WeightedMultiTask._compute_pgt_allocation(
            [0.5, 0.5], [100.0, 0.01], 8, [4, 4], min_pgts=[1, 3], max_step_fraction=1.0
        )
        assert new[1] >= 3
        assert sum(new) == 8

    def test_max_step_limits_movement(self):
        # Extreme imbalance, but each update moves at most 25% of current.
        new = WeightedMultiTask._compute_pgt_allocation(
            [0.5, 0.5], [10.0, 0.1], 20, [10, 10], min_pgts=[1, 1], max_step_fraction=0.25
        )
        assert new == [12, 8]


class _FakeGate:
    def __init__(self):
        self.capacity_history = []

    def set_capacity(self, capacity):
        self.capacity_history.append(capacity)


class _FakePipeline:
    def __init__(self, ema, samples):
        self.engine_dwell_ema = ema
        self.engine_dwell_sample_count = samples
        self.gate = _FakeGate()


class _FakeAgent:
    def __init__(self, pipeline, pgt):
        self._active_pipeline = pipeline
        self.parallel_generation_tasks = pgt


class TestPgtRebalancer:
    def _make(self, emas, samples, pgts, interval=0.0, min_samples=1):
        agents = [
            _FakeAgent(_FakePipeline(ema, n), pgt) for ema, n, pgt in zip(emas, samples, pgts)
        ]
        rebalancer = _PgtRebalancer(
            PgtRebalanceConfig(
                min_interval_s=interval, min_samples_per_env=min_samples, max_step_fraction=1.0
            ),
            agent_indices=list(range(len(agents))),
            weights=[1.0 / len(agents)] * len(agents),
            current_pgts=list(pgts),
            min_pgts=[1] * len(agents),
        )
        return agents, rebalancer

    def test_shifts_capacity_toward_slow_env(self):
        agents, rebalancer = self._make([3.0, 1.0], [10, 10], [8, 8])
        event = rebalancer.maybe_rebalance(agents)
        assert event is not None
        assert event["new"] == [12, 4]
        assert sum(event["new"]) == 16
        assert agents[0]._active_pipeline.gate.capacity_history == [12]
        assert agents[1]._active_pipeline.gate.capacity_history == [4]
        assert [a.parallel_generation_tasks for a in agents] == [12, 4]

    def test_interval_throttles_checks(self):
        agents, rebalancer = self._make([3.0, 1.0], [10, 10], [8, 8], interval=1000.0)
        assert rebalancer.maybe_rebalance(agents) is not None
        # EMAs still differ, but the next check is inside the interval.
        assert rebalancer.maybe_rebalance(agents) is None

    def test_insufficient_samples_is_noop(self):
        agents, rebalancer = self._make([3.0, 1.0], [0, 0], [8, 8], min_samples=5)
        assert rebalancer.maybe_rebalance(agents) is None
        assert [a.parallel_generation_tasks for a in agents] == [8, 8]


class TestEngineDwellEma:
    @pytest.mark.asyncio
    async def test_ema_tracks_inferred_rollouts(self):
        gen = MockGenerator(parallel_generation_tasks=2)
        gen.max_parallel_generation_tasks = 5
        request = GroupedRolloutRequest(
            num_groups=1,
            rollouts_per_group=1,
            inference_interface=MockInferenceInterface(),
            streaming=True,
            submission_granularity="R",
            consumption_granularity="B",
        )
        it = gen.get_grouped_rollouts(request)
        try:
            for _ in range(3):
                await asyncio.wait_for(anext(it), timeout=10)
            pipeline = gen._active_pipeline
            # Workers are sized for the rebalancing ceiling, not the split.
            assert pipeline.num_infer_workers == 5
            assert pipeline.engine_dwell_ema is not None
            assert pipeline.engine_dwell_sample_count >= 3
        finally:
            await it.aclose()
