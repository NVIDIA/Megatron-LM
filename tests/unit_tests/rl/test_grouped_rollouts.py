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
    """Mock raw-text inference interface with configurable per-prompt delays.

    Prompts at index >= stall_after_calls park forever, modeling a suspended
    engine whose set of completable rollouts is exact and scheduling-independent.
    """

    num_slow_calls: int = 0
    stall_after_calls: int | None = None
    active_requests: int = 0
    max_active_requests: int = 0

    async def base_generate(self, request):
        prompt = request.prompt[0].content
        idx = int(prompt.removeprefix("t"))
        if self.stall_after_calls is not None and idx >= self.stall_after_calls:
            await asyncio.Event().wait()
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
            "num_slow_calls, stall_after_calls, streaming, num_groups, "
            "submission_granularity, consumption_granularity, expected_count, "
            "expected_batch_ids, expected_trajectories, expected_ready_batches"
        ),
        [
            pytest.param(0, None, False, 8, "B", "B", 8, None, None, 0, id="non_batched"),
            pytest.param(
                0,
                None,
                False,
                4,
                "B",
                "B",
                4,
                None,
                None,
                0,
                id="non_streaming_fewer_than_parallel",
            ),
            pytest.param(
                4,
                None,
                True,
                2,
                "B",
                "B",
                8,
                [0, 0, 1, 1, 2, 2, 3, 3],
                None,
                None,
                id="batched_submission_order",
            ),
            pytest.param(0, None, True, 1, "G", "B", 10, None, None, None, id="streaming"),
            pytest.param(
                4,
                None,
                True,
                1,
                "G",
                "G",
                8,
                None,
                [f"t{i}" for i in range(4, 8)],
                None,
                id="group_consume_completion_order",
            ),
            pytest.param(
                4,
                None,
                True,
                1,
                "G",
                "B",
                8,
                list(range(8)),
                [f"t{i}" for i in range(8)],
                None,
                id="batch_consume_submission_order",
            ),
            # 6 completable rollouts, then the engine stalls (as when
            # suspended): 3 batches of 2 groups can bank without further
            # generation; consuming one leaves 2 ready.
            pytest.param(
                0,
                6,
                True,
                2,
                "B",
                "B",
                2,
                [0, 0],
                ["t0", "t1"],
                2,
                id="stalled_engine_banks_ready_batches",
            ),
        ],
    )
    async def test_grouped_rollout_generation(
        self,
        num_slow_calls,
        stall_after_calls,
        streaming,
        num_groups,
        submission_granularity,
        consumption_granularity,
        expected_count,
        expected_batch_ids,
        expected_trajectories,
        expected_ready_batches,
    ):
        gen = MockGenerator()
        request = GroupedRolloutRequest(
            num_groups=num_groups,
            rollouts_per_group=1,
            inference_interface=MockInferenceInterface(
                num_slow_calls=num_slow_calls, stall_after_calls=stall_after_calls
            ),
            streaming=streaming,
            submission_granularity=submission_granularity,
            consumption_granularity=consumption_granularity,
        )

        groups = []
        pipeline = RolloutPipeline(gen, request, parallel_generation_tasks=8)
        async for group in pipeline.run():
            groups.append(group)
            if request.streaming and len(groups) >= expected_count:
                break

        assert len(groups) == expected_count
        if expected_batch_ids is not None:
            assert [g.batch_id for g in groups] == expected_batch_ids
        if expected_trajectories is not None:
            trajectories = [group[0].trajectory[0] for group in groups]
            assert trajectories[: len(expected_trajectories)] == expected_trajectories
        if expected_ready_batches is not None:
            for _ in range(2 * request.num_groups):
                await asyncio.sleep(0)
            assert pipeline.ready_batches == expected_ready_batches
        assert pipeline.yielded_count == len(groups)
        assert len(pipeline.output_queue_dwell) == len(groups)

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
            streaming=True,
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
            streaming=False,
            submission_granularity=submission_granularity,
            consumption_granularity=consumption_granularity,
        )
        groups = []
        pipeline = RolloutPipeline(mt, request, parallel_generation_tasks=1)
        async for group in pipeline.run():
            groups.append(group)

        assert len(groups) == 4
        # Weights 3:1 → env "a" owns 3 batch slots, env "b" owns 1; the single
        # pipeline routes preparation and generation to the owning sub-agent.
        env_ids = [g[0].env_id for g in groups]
        assert sorted(env_ids) == ["a", "a", "a", "b"]
        assert [agent.prepare_group_rollout_calls for agent in mt.agents] == [3, 1]
        assert [agent.get_rollout_response_calls for agent in mt.agents] == [3, 1]
        assert mt.latest_distribution["agent_groups"] == [3, 1]
        # The pipeline drains fully: every gate slot is released at exhaustion.
        assert pipeline.gate.held == 0

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "num_slow_calls, stall_after_calls, collect_count, expected_ready_batches",
        [
            pytest.param(2, None, 12, None, id="balanced_windows"),
            # Both envs complete prompts t0-t5: env "a" (3 groups/batch) has
            # two complete units, env "b" six, so exactly two balanced rounds
            # can bank; consuming one leaves one ready.
            pytest.param(0, 6, 4, 1, id="stalled_engine_banks_complete_rounds"),
        ],
    )
    async def test_env_consumption_balances_each_batch(
        self, num_slow_calls, stall_after_calls, collect_count, expected_ready_batches
    ):
        """Balanced-E: every trainer-batch window holds each env's exact share,
        and a banked batch is one complete round."""
        configs = [
            AgentConfig(agent_type=MockGenerator, agent_args={"env_id": "a"}, weight=3.0),
            AgentConfig(agent_type=MockGenerator, agent_args={"env_id": "b"}, weight=1.0),
        ]
        mt = WeightedMultiTask(configs)

        request = GroupedRolloutRequest(
            num_groups=4,
            rollouts_per_group=1,
            inference_interface=MockInferenceInterface(
                num_slow_calls=num_slow_calls, stall_after_calls=stall_after_calls
            ),
            streaming=True,
            submission_granularity="E",
            consumption_granularity="E",
        )
        groups = []
        pipeline = RolloutPipeline(mt, request, parallel_generation_tasks=2)
        async for group in pipeline.run():
            groups.append(group)
            if len(groups) >= collect_count:
                break

        for start in range(0, collect_count, 4):
            env_ids = [g[0].env_id for g in groups[start : start + 4]]
            assert sorted(env_ids) == ["a", "a", "a", "b"]
        if expected_ready_batches is not None:
            for _ in range(2 * request.num_groups):
                await asyncio.sleep(0)
            assert pipeline.ready_batches == expected_ready_batches

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

    def test_multi_env_layout_rejects_starving_batch_size(self):
        """The layout raises rather than silently starving a weighted env."""
        configs = [
            AgentConfig(agent_type=MockGenerator, agent_args={"env_id": "a"}, weight=3.0),
            AgentConfig(agent_type=MockGenerator, agent_args={"env_id": "b"}, weight=1.0),
        ]
        mt = WeightedMultiTask(configs)
        with pytest.raises(ValueError, match="starved"):
            mt.rollout_group_layout(1)
        assert mt.rollout_group_layout(8) == [6, 2]

        # Evaluation-only envs take no groups and never count as starved.
        mt = WeightedMultiTask(
            configs
            + [
                AgentConfig(
                    agent_type=MockGenerator,
                    agent_args={"env_id": "c"},
                    weight=1.0,
                    evaluation_only=True,
                )
            ]
        )
        assert mt.rollout_group_layout(8) == [6, 2]
