# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from megatron.rl.agent.api import (
    GroupedRolloutGenerator,
    GroupedRolloutRequest,
    GroupRolloutParams,
    Rollout,
    RolloutGenerator,
    RolloutRequest,
)
from megatron.rl.agent.reward_only_agent import RewardOnlyAgent
from megatron.rl.agent.rollout_pipeline import RolloutPipeline
from megatron.rl.agent.weighted_multi_task import AgentConfig, WeightedMultiTask
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
        self.get_rollout_response_calls = 0

    async def get_reward_rollouts(self, request):
        raise NotImplementedError

    async def get_rollout_response(self, request, inference_request):
        self.get_rollout_response_calls += 1
        return await request.inference_interface.agenerate(inference_request)

    async def prepare_group_rollout(self, request, env_index: int = 0):
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

        return GroupRolloutParams(
            inference_request=inference_request, build_rollout=build_rollout, agent=self
        )


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
        gen = MockGenerator()
        request = GroupedRolloutRequest(
            num_groups=num_groups,
            rollouts_per_group=2,
            inference_interface=MockInferenceInterface(),
            filter_groups_with_same_reward=True,
            submission_granularity=submission_granularity,
            consumption_granularity=consumption_granularity,
        )
        with pytest.raises(AssertionError, match="filter_groups_with_same_reward"):
            RolloutPipeline(gen, request, parallel_generation_tasks=8)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        (
            "num_slow_calls, num_groups, submission_granularity, "
            "consumption_granularity, expected_count, expected_batch_ids, "
            "expected_trajectories"
        ),
        [
            pytest.param(0, 8, "B", "B", 8, None, None, id="single_batch"),
            pytest.param(0, 4, "B", "B", 4, None, None, id="fewer_groups_than_parallel"),
            pytest.param(
                4, 2, "B", "B", 8, [0, 0, 1, 1, 2, 2, 3, 3], None, id="batched_submission_order"
            ),
            pytest.param(0, 1, "G", "B", 10, None, None, id="streaming"),
            pytest.param(
                4,
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
    async def test_grouped_rollout_generation(
        self,
        num_slow_calls,
        num_groups,
        submission_granularity,
        consumption_granularity,
        expected_count,
        expected_batch_ids,
        expected_trajectories,
    ):
        gen = MockGenerator()
        request = GroupedRolloutRequest(
            num_groups=num_groups,
            rollouts_per_group=1,
            inference_interface=MockInferenceInterface(num_slow_calls=num_slow_calls),
            submission_granularity=submission_granularity,
            consumption_granularity=consumption_granularity,
        )

        groups = []
        async for group in RolloutPipeline(gen, request, parallel_generation_tasks=8).run():
            groups.append(group)
            if len(groups) >= expected_count:
                break

        assert len(groups) == expected_count
        if expected_batch_ids is not None:
            assert [g.batch_id for g in groups] == expected_batch_ids
        if expected_trajectories is not None:
            trajectories = [group[0].trajectory[0] for group in groups]
            assert trajectories[: len(expected_trajectories)] == expected_trajectories

    @pytest.mark.asyncio
    async def test_rollout_submission_granularity_limits_inference_concurrency(self):
        # parallel_generation_tasks is a depth in batches; the R gate admits at
        # most depth x (num_groups x rollouts_per_group) rollouts at once.
        parallel_generation_tasks = 1
        gen = MockGenerator()
        inference_interface = MockInferenceInterface(num_slow_calls=100)
        request = GroupedRolloutRequest(
            num_groups=2,
            rollouts_per_group=2,
            inference_interface=inference_interface,
            submission_granularity="R",
            consumption_granularity="B",
        )

        groups = []
        pipeline = RolloutPipeline(
            gen, request, parallel_generation_tasks=parallel_generation_tasks
        )
        async for group in pipeline.run():
            groups.append(group)
            if len(groups) >= 4:
                break

        assert all(len(group) == 2 for group in groups)
        assert inference_interface.max_active_requests <= (
            parallel_generation_tasks * request.num_groups * request.rollouts_per_group
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "submission_granularity, consumption_granularity",
        [
            pytest.param("B", "B", id="batch_batch"),
            pytest.param("G", "G", id="group_group"),
            pytest.param("E", "E", id="env_env"),
            pytest.param("G", "E", id="group_env"),
        ],
    )
    async def test_weighted_multi_task(self, submission_granularity, consumption_granularity):
        configs = [
            AgentConfig(agent_type=MockGenerator, agent_args={"env_id": "a"}, weight=3.0),
            AgentConfig(agent_type=MockGenerator, agent_args={"env_id": "b"}, weight=1.0),
        ]
        mt = WeightedMultiTask(configs)

        request = GroupedRolloutRequest(
            num_groups=4,
            rollouts_per_group=1,
            inference_interface=MockInferenceInterface(),
            submission_granularity=submission_granularity,
            consumption_granularity=consumption_granularity,
        )
        pipeline = RolloutPipeline(mt, request, parallel_generation_tasks=1)
        gen = pipeline.run()
        groups = [await anext(gen) for _ in range(8)]

        # Weights 3:1 → env "a" owns 3 of every 4 batch slots; the single
        # pipeline routes preparation and generation to the owning sub-agent.
        env_ids = [g[0].env_id for g in groups]
        assert sorted(env_ids) == ["a"] * 6 + ["b"] * 2
        assert mt.latest_distribution["agent_groups"] == [3, 1]
        if consumption_granularity in ("B", "E"):
            # Batch-order and balanced-env consumption keep every 4-window at the exact env mix.
            for start in (0, 4):
                assert sorted(env_ids[start : start + 4]) == ["a", "a", "a", "b"]
        if consumption_granularity == "B":
            # With depth-1 gating and consumed-release, nothing is buffered or in flight.
            assert pipeline.prepared_count == (pipeline.yielded_count * request.rollouts_per_group)
            assert pipeline.infer_queue.qsize() == 0
            assert pipeline.assemble_queue.qsize() == 0
            assert pipeline.output_queue.qsize() == 0
            assert not pipeline._assemble_pending
            assert not pipeline._consume_pending

    @pytest.mark.asyncio
    async def test_env_consumption_balances_each_batch(self):
        """Balanced-E: every trainer-batch window holds each env's exact share."""
        configs = [
            AgentConfig(agent_type=MockGenerator, agent_args={"env_id": "a"}, weight=3.0),
            AgentConfig(agent_type=MockGenerator, agent_args={"env_id": "b"}, weight=1.0),
        ]
        mt = WeightedMultiTask(configs)

        request = GroupedRolloutRequest(
            num_groups=4,
            rollouts_per_group=1,
            inference_interface=MockInferenceInterface(num_slow_calls=2),
            submission_granularity="E",
            consumption_granularity="E",
        )
        groups = []
        async for group in RolloutPipeline(mt, request, parallel_generation_tasks=2).run():
            groups.append(group)
            if len(groups) >= 12:
                break

        for start in range(0, 12, 4):
            env_ids = [g[0].env_id for g in groups[start : start + 4]]
            assert sorted(env_ids) == ["a", "a", "a", "b"]

    @pytest.mark.asyncio
    async def test_lag0_streaming_matches_non_streaming_boundaries(self):
        """lag=0 (B/B, depth-1 gate): each iteration of the persistent stream is exactly
        one batch, generated entirely after the previous boundary — the old
        non-streaming per-iteration contract, enforced by assert_no_inflight_rollouts."""
        from megatron.rl.rl_utils import assert_no_inflight_rollouts

        configs = [
            AgentConfig(agent_type=MockGenerator, agent_args={"env_id": "a"}, weight=1.0),
            AgentConfig(agent_type=MockGenerator, agent_args={"env_id": "b"}, weight=1.0),
        ]
        mt = WeightedMultiTask(configs)
        request = GroupedRolloutRequest(
            num_groups=4,
            rollouts_per_group=2,
            inference_interface=MockInferenceInterface(),
            submission_granularity="B",
            consumption_granularity="B",
        )
        pipeline = RolloutPipeline(mt, request, parallel_generation_tasks=1)
        gen = pipeline.run()
        for iteration in range(3):
            groups = [await anext(gen) for _ in range(4)]
            # Exactly this iteration's batch, whole and in order.
            assert [g.batch_id for g in groups] == [iteration] * 4
            # Nothing of the next batch has even been prepared: everything the
            # next iteration consumes is generated after this boundary.
            assert sum(a.prepare_group_rollout_calls for a in mt.agents) == (iteration + 1) * 4
            assert_no_inflight_rollouts(pipeline)

    @pytest.mark.asyncio
    async def test_assert_no_inflight_rollouts_detects_run_ahead(self):
        """With lag>0 the gate legitimately runs ahead; the boundary checker must fire."""
        from megatron.rl.rl_utils import assert_no_inflight_rollouts

        request = GroupedRolloutRequest(
            num_groups=4,
            rollouts_per_group=1,
            inference_interface=MockInferenceInterface(),
            submission_granularity="B",
            consumption_granularity="B",
        )
        pipeline = RolloutPipeline(MockGenerator(), request, parallel_generation_tasks=2)
        gen = pipeline.run()
        [await anext(gen) for _ in range(4)]
        with pytest.raises(AssertionError, match="Non-streaming RL"):
            assert_no_inflight_rollouts(pipeline)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "submission_granularity, consumption_granularity",
        [
            pytest.param("B", "G", id="batch_group"),
            pytest.param("B", "E", id="batch_env"),
            pytest.param("E", "G", id="env_group"),
        ],
    )
    async def test_consumption_finer_than_submission_rejected(
        self, submission_granularity, consumption_granularity
    ):
        gen = MockGenerator()
        request = GroupedRolloutRequest(
            num_groups=2,
            rollouts_per_group=1,
            inference_interface=MockInferenceInterface(),
            submission_granularity=submission_granularity,
            consumption_granularity=consumption_granularity,
        )
        with pytest.raises(AssertionError, match="no finer"):
            RolloutPipeline(gen, request, parallel_generation_tasks=1)

    @pytest.mark.parametrize(
        "weights, num_groups, expected_layout, warns",
        [
            # 8 groups cannot realize 1:2 exactly; quantized with a warning.
            pytest.param([1.0, 2.0], 8, [3, 5], True, id="quantized"),
            # A weight below 1/num_groups keeps one group per batch.
            pytest.param([0.01, 0.99], 8, [1, 7], True, id="zero_share_rounded_up"),
            pytest.param([3.0, 1.0], 8, [6, 2], False, id="exact"),
            pytest.param([1.0, 1.0, 1.0], 3, [1, 1, 1], False, id="one_group_each"),
            # Only an env count exceeding the batch size is infeasible.
            pytest.param([1.0, 1.0, 1.0], 2, None, False, id="too_many_envs"),
        ],
    )
    def test_multi_env_layout(self, caplog, weights, num_groups, expected_layout, warns):
        """Weights quantize to a constant split (warned); eval-only envs take no slot."""
        configs = [
            AgentConfig(agent_type=MockGenerator, agent_args={"env_id": f"e{i}"}, weight=w)
            for i, w in enumerate(weights)
        ] + [
            AgentConfig(
                agent_type=MockGenerator,
                agent_args={"env_id": "eval"},
                weight=1.0,
                evaluation_only=True,
            )
        ]
        mt = WeightedMultiTask(configs)
        if expected_layout is None:
            with pytest.raises(ValueError, match="cannot fit"):
                mt.rollout_group_layout(num_groups)
            return
        # The split is identical on every call.
        assert [mt.rollout_group_layout(num_groups) for _ in range(3)] == [expected_layout] * 3
        assert warns == any("weights changed" in message for message in caplog.messages)
