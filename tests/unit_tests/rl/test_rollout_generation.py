# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
import itertools
from contextlib import aclosing
from unittest.mock import MagicMock

import numpy as np
import pytest
from pydantic import Field, ValidationError

from megatron.rl.agent.api import (
    EpisodeResult,
    GroupedRolloutGenerator,
    GroupedRolloutRequest,
    GroupRolloutParams,
    Rollout,
    RolloutGenerator,
    RolloutRequest,
    TokenRollout,
)
from megatron.rl.agent.reward_only_agent import RewardOnlyAgent
from megatron.rl.agent.rollout_pipeline import RolloutPipeline, _SubmissionGate
from megatron.rl.agent.weighted_multi_task import AgentConfig, WeightedMultiTask
from megatron.rl.inference import InferenceResponse, LLMChatMessage, ReturnsRaw, ReturnsTokens


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

    async def prepare_group_rollout(self, request):
        idx = self._call_count
        self._call_count += 1
        self.prepare_group_rollout_calls += 1

        async def run_episode():
            # Single-turn agent: the episode is one inference on the group's prompt.
            turn_request = request.inference_interface.prepare_request(
                f"t{idx}", request.generation_args
            )
            response = await self.get_rollout_response(request, turn_request)
            return EpisodeResult(
                responses=[response], conversation=[*turn_request.prompt, response.response]
            )

        async def build_rollout(episode):
            responses = episode.responses
            reward = float(responses[-1].response.content.removeprefix("t"))
            return Rollout(
                trajectory=[r.raw_text for r in responses], reward=reward, env_id=self.env_id
            )

        return GroupRolloutParams(run_episode=run_episode, build_rollout=build_rollout)


class FilteringMockGenerator(MockGenerator):
    """Mock generator whose first `num_degenerate` prepared groups have zero reward variance."""

    def __init__(self, num_degenerate=0, **kwargs):
        super().__init__(**kwargs)
        self.num_degenerate = num_degenerate

    async def prepare_group_rollout(self, request):
        idx = self._call_count
        params = await super().prepare_group_rollout(request)
        degenerate = idx < self.num_degenerate
        rollout_counter = itertools.count()
        base_build = params.build_rollout

        async def build_rollout(episode):
            rollout = await base_build(episode)
            rollout.reward = 0.0 if degenerate else float(next(rollout_counter))
            return rollout

        return GroupRolloutParams(run_episode=params.run_episode, build_rollout=build_rollout)


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
        # Gate capacity in G-submission slots is parallel_generation_tasks
        # (a depth in batches) x num_groups (groups per batch).
        capacity = 4
        gen = MockGenerator()
        request = GroupedRolloutRequest(
            num_groups=num_groups,
            rollouts_per_group=1,
            inference_interface=MockInferenceInterface(),
            streaming=True,
            submission_granularity="G",
            consumption_granularity=consumption_granularity,
        )
        pipeline = RolloutPipeline(gen, request, parallel_generation_tasks=capacity // num_groups)
        it = pipeline.run()
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
        gen = MockGenerator()
        request = GroupedRolloutRequest(
            num_groups=2,
            rollouts_per_group=1,
            inference_interface=MockInferenceInterface(),
            streaming=True,
            submission_granularity="B",
            consumption_granularity="B",
        )
        pipeline = RolloutPipeline(gen, request, parallel_generation_tasks=1)
        it = pipeline.run()
        try:
            await asyncio.wait_for(anext(it), timeout=10)
            await asyncio.wait_for(anext(it), timeout=10)
            await _flush()
            gate = pipeline.gate
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


class TestStageFailurePropagation:
    """A dead stage must fail run() loudly, never read as a clean end-of-stream.

    A stage that dies runs the queue-shutdown cascade, which reaches
    stage_consume exactly like a clean end-of-stream; before run() reaped the
    stage tasks, the caller saw StopAsyncIteration and waited forever for
    rollouts nobody would ever generate (observed live 2026-07-30: a TypeError
    in the first prepare_group_rollout idled a training job to its time limit).
    """

    @pytest.mark.asyncio
    async def test_prepare_failure_raises_out_of_run(self):
        class BrokenPrepareGenerator(MockGenerator):
            async def prepare_group_rollout(self, request):
                raise TypeError("agent/pipeline interface mismatch")

        request = GroupedRolloutRequest(
            num_groups=2,
            rollouts_per_group=2,
            inference_interface=MockInferenceInterface(),
            submission_granularity="R",
            consumption_granularity="G",
        )
        pipeline = RolloutPipeline(BrokenPrepareGenerator(), request, parallel_generation_tasks=1)
        async with aclosing(pipeline.run()) as it:
            with pytest.raises(RuntimeError, match="stage died") as excinfo:
                # wait_for turns the pre-fix failure mode (an eternal hang once
                # the cascade is mistaken for end-of-stream) into a test failure.
                await asyncio.wait_for(anext(it), timeout=10)
        assert isinstance(excinfo.value.__cause__, TypeError)

    @pytest.mark.asyncio
    async def test_midstream_stage_failure_raises_after_delivered_groups(self):
        class BrokenBuildGenerator(MockGenerator):
            """First group builds normally; every later group's build_rollout raises."""

            async def prepare_group_rollout(self, request):
                idx = self._call_count
                params = await super().prepare_group_rollout(request)
                if idx < 1:
                    return params

                async def broken_build(episode):
                    raise ValueError("reward model exploded")

                return GroupRolloutParams(
                    run_episode=params.run_episode, build_rollout=broken_build
                )

        request = GroupedRolloutRequest(
            num_groups=2,
            rollouts_per_group=2,
            inference_interface=MockInferenceInterface(),
            submission_granularity="G",
            consumption_granularity="G",
        )
        pipeline = RolloutPipeline(BrokenBuildGenerator(), request, parallel_generation_tasks=1)
        async with aclosing(pipeline.run()) as it:
            # Group 0 is healthy and must still be delivered.
            group = await asyncio.wait_for(anext(it), timeout=10)
            assert len(group.rollouts) == 2
            # Group 1's build_rollout kills stage_assemble; the next pull must
            # surface that failure instead of hanging on the drained stream.
            with pytest.raises(RuntimeError, match="stage died") as excinfo:
                await asyncio.wait_for(anext(it), timeout=10)
        assert isinstance(excinfo.value.__cause__, ValueError)


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
        "streaming, submission_granularity, consumption_granularity, num_degenerate",
        [
            pytest.param(False, "B", "B", 3, id="batch_submission"),
            pytest.param(False, "G", "B", 3, id="group_submission"),
            pytest.param(False, "R", "B", 3, id="rollout_submission"),
            pytest.param(False, "G", "G", 3, id="group_consumption"),
            pytest.param(False, "R", "G", 3, id="rollout_submission_group_consume"),
            pytest.param(False, "B", "B", 9, id="cascading_regeneration"),
            pytest.param(True, "B", "B", 3, id="streaming_batch_consume"),
            pytest.param(True, "G", "G", 3, id="streaming_group_consume"),
        ],
    )
    async def test_filter_groups_and_regenerate(
        self, streaming, submission_granularity, consumption_granularity, num_degenerate
    ):
        num_groups = 4
        gen = FilteringMockGenerator(num_degenerate=num_degenerate)
        request = GroupedRolloutRequest(
            num_groups=num_groups,
            rollouts_per_group=2,
            inference_interface=MockInferenceInterface(),
            filter_groups_with_same_reward=True,
            streaming=streaming,
            submission_granularity=submission_granularity,
            consumption_granularity=consumption_granularity,
        )
        pipeline = RolloutPipeline(gen, request, parallel_generation_tasks=8)

        expected_count = 2 * num_groups if streaming else num_groups
        groups, filtered_counts = [], []
        async with aclosing(pipeline.run()) as it:
            async for group in it:
                groups.append(group)
                filtered_counts.append(pipeline.filtered_count)
                if streaming and len(groups) >= expected_count:
                    break

        assert len(groups) == expected_count
        # Every delivered group carries reward signal.
        for group in groups:
            assert np.std([rollout.reward for rollout in group]) > 1e-6
        if consumption_granularity == "B":
            # Batches complete and arrive in submission order despite drops.
            assert [g.batch_id for g in groups] == sorted(g.batch_id for g in groups)
            for batch_start in range(0, expected_count, num_groups):
                batch = groups[batch_start : batch_start + num_groups]
                assert sorted(g.index_in_batch for g in batch) == list(range(num_groups))
        if not streaming:
            # One extra prepare per dropped group, and no under-delivery.
            assert gen.prepare_group_rollout_calls == num_groups + num_degenerate
            assert filtered_counts[-1] == num_degenerate

    @pytest.mark.asyncio
    async def test_pending_regeneration_does_not_block_delivery(self):
        """A regeneration pinned on R-slot acquires must not stall delivery of ready groups.

        Rollouts run for minutes in production: with gate capacity 2, group 2's
        gated rollouts hog both R slots, so group 0's regeneration sits inside
        its slot acquires while group 1 is already assembled and deliverable.
        """
        events = {"t2": asyncio.Event()}

        class GatedInference(MockInferenceInterface):
            async def base_generate(self, request):
                event = events.get(request.prompt[0].content)
                if event is not None:
                    await event.wait()
                return await super().base_generate(request)

        gen = FilteringMockGenerator(num_degenerate=1)
        request = GroupedRolloutRequest(
            num_groups=3,
            rollouts_per_group=2,
            inference_interface=GatedInference(),
            filter_groups_with_same_reward=True,
            submission_granularity="R",
            consumption_granularity="G",
        )
        pipeline = RolloutPipeline(gen, request, parallel_generation_tasks=1 / 3)
        assert pipeline.gate.capacity == 2
        async with aclosing(pipeline.run()) as it:
            # wait_for turns the pre-fix failure mode (the regeneration awaited
            # inline on the delivery path, delivering nothing) into a test failure.
            group = await asyncio.wait_for(anext(it), timeout=10)
            await _flush()
            assert [rollout.trajectory[0] for rollout in group.rollouts] == ["t1", "t1"]
            assert pipeline.filtered_count == 1
            assert len(pipeline._regen_tasks) == 1
            assert gen.prepare_group_rollout_calls == 4
            events["t2"].set()
            rest = [await asyncio.wait_for(anext(it), timeout=10) for _ in range(2)]
        assert sorted(g.rollouts[0].trajectory[0] for g in rest) == ["t2", "t3"]
        assert not pipeline._regen_tasks

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
    async def test_grouped_rollout_generation(
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
        gen = MockGenerator()
        request = GroupedRolloutRequest(
            num_groups=num_groups,
            rollouts_per_group=1,
            inference_interface=MockInferenceInterface(num_slow_calls=num_slow_calls),
            streaming=streaming,
            submission_granularity=submission_granularity,
            consumption_granularity=consumption_granularity,
        )

        groups = []
        async for group in RolloutPipeline(gen, request, parallel_generation_tasks=8).run():
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
        [pytest.param("B", "B", id="batch_batch"), pytest.param("G", "G", id="group_group")],
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
    async def test_group_consumption_balances_each_batch(self):
        """Balanced G: every trainer-batch window holds each env's exact share."""
        configs = [
            AgentConfig(agent_type=MockGenerator, agent_args={"env_id": "a"}, weight=3.0),
            AgentConfig(agent_type=MockGenerator, agent_args={"env_id": "b"}, weight=1.0),
        ]
        mt = WeightedMultiTask(configs)

        request = GroupedRolloutRequest(
            num_groups=4,
            rollouts_per_group=1,
            inference_interface=MockInferenceInterface(num_slow_calls=2),
            streaming=True,
            submission_granularity="G",
            consumption_granularity="G",
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
    @pytest.mark.parametrize(
        "submission_granularity, consumption_granularity",
        [pytest.param("B", "G", id="batch_group")],
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
            mt.rollout_allocations(1)
        assert [a.num_groups for a in mt.rollout_allocations(8)] == [6, 2]

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
        assert [a.num_groups for a in mt.rollout_allocations(8)] == [6, 2]


def make_response(
    prompt_length, total_len, content="resp", finish_reason="stop", completion_id=None
):
    return InferenceResponse(
        response=LLMChatMessage(role="assistant", content=content),
        raw_text=content,
        token_ids=list(range(total_len)),
        prompt_length=prompt_length,
        logprobs=[0.0] * (total_len - prompt_length),
        finish_reason=finish_reason,
        completion_id=completion_id,
    )


# Conversation length -> response spec: length 1 is the first turn (the bare prompt), length 3
# the second (assistant reply + observation appended).
TWO_TURN_SCRIPT = {
    1: dict(prompt_length=3, total_len=7, content="a0", completion_id="cc-1"),
    3: dict(prompt_length=6, total_len=11, content="a1", completion_id="cc-3"),
}

# Both two-turn termination modes (env-signaled done, max_turns exhausted) must produce this
# identical episode; only the env-consultation trace (observation_turns) differs per case.
TWO_TURN_EXPECTED = dict(
    seen_roles=[["user"], ["user", "assistant", "user"]],
    reward_conv=[("user", "hello"), ("assistant", "a0"), ("user", "obs0"), ("assistant", "a1")],
    rewarded=[("a1", "stop")],
    genmask_sums=[4, 5],
    completion_ids=["cc-1", "cc-3"],
)


class ScriptedInterface(ReturnsTokens, ReturnsRaw):
    """Inference stub whose reply is a pure function of the request: the conversation length
    maps to a response spec, so it stays deterministic under pipeline concurrency."""

    by_prompt_length: dict = Field(default_factory=dict)
    seen_conversations: list = Field(default_factory=list)

    async def agenerate(self, request):
        self.seen_conversations.append(list(request.prompt))
        return make_response(**self.by_prompt_length[len(request.prompt)])


class EpisodeAgent(RewardOnlyAgent):
    """Configurable multi-turn agent.

    `done_at_turn` controls when get_observation signals done: at every turn >= done_at_turn
    it returns (None, True); None means it never signals done, so the episode ends only by
    exhausting max_turns. Records get_reward calls and the conversation get_trajectory_reward saw.
    """

    env_id: str = "test"
    max_turns: int = 1
    done_at_turn: int | None = None
    rewarded: list = Field(default_factory=list)
    reward_conversation: list = Field(default_factory=list)
    observation_turns: list = Field(default_factory=list)

    async def get_prompt(self, validation):
        return "hello", {"problem_id": "p0"}

    async def get_observation(self, turn_idx, response, conversation, golden):
        self.observation_turns.append(turn_idx)
        if self.done_at_turn is not None and turn_idx >= self.done_at_turn:
            return None, True
        return f"obs{turn_idx}", False

    async def get_reward(self, response, golden, finish_reason):
        self.rewarded.append((response, finish_reason))
        return 1.5

    async def get_trajectory_reward(self, responses, conversation, golden):
        self.reward_conversation.extend(conversation)
        return await super().get_trajectory_reward(responses, conversation, golden)


class TestMultiTurnEpisode:

    @pytest.mark.parametrize("driver", ["reward_rollouts", "pipeline"])
    @pytest.mark.parametrize(
        "max_turns, done_at_turn, scripted, expected",
        [
            # Single turn: get_observation is never consulted (no continuation is possible).
            pytest.param(
                1,
                None,
                {1: dict(prompt_length=2, total_len=6, content="only", completion_id="cc-1")},
                dict(
                    seen_roles=[["user"]],
                    reward_conv=[("user", "hello"), ("assistant", "only")],
                    rewarded=[("only", "stop")],
                    genmask_sums=[4],
                    completion_ids=["cc-1"],
                    observation_turns=[],
                ),
                id="single_turn",
            ),
            # Multi-turn ended by the environment: turn 0 yields an observation, turn 1 is done.
            pytest.param(
                3,
                1,
                TWO_TURN_SCRIPT,
                dict(TWO_TURN_EXPECTED, observation_turns=[0, 1]),
                id="multi_turn_env_done",
            ),
            # Ended by exhausting max_turns instead (env never signals done): the same episode,
            # except get_observation must not run for the final allowed turn.
            pytest.param(
                2,
                None,
                TWO_TURN_SCRIPT,
                dict(TWO_TURN_EXPECTED, observation_turns=[0]),
                id="multi_turn_max_turns_exhausted",
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_run_episode(self, driver, max_turns, done_at_turn, scripted, expected):
        """Episodes grow the conversation each turn and collapse into one per-turn rollout,
        identically through get_reward_rollouts and through the real _RolloutPipeline
        (get_grouped_rollouts) -- the latter proving run_episode runs in the infer stage."""
        iface = ScriptedInterface(by_prompt_length=scripted)
        agent = EpisodeAgent(max_turns=max_turns, done_at_turn=done_at_turn)

        if driver == "reward_rollouts":
            rollouts = await agent.get_reward_rollouts(
                RolloutRequest(num_rollouts=1, inference_interface=iface)
            )
        else:
            groups = []

            async def _drain():
                request = GroupedRolloutRequest(
                    num_groups=1, rollouts_per_group=1, inference_interface=iface
                )
                async with aclosing(
                    RolloutPipeline(agent, request, parallel_generation_tasks=1).run()
                ) as iterator:
                    async for group in iterator:
                        groups.append(group)
                        break

            # Bounded so a wedged pipeline fails fast instead of hanging.
            await asyncio.wait_for(_drain(), timeout=5.0)
            (group,) = groups
            rollouts = group.rollouts
        (rollout,) = rollouts

        assert isinstance(rollout, TokenRollout)
        assert rollout.reward == 1.5
        assert rollout.problem_id == "p0"
        # Per-turn backend response ids ride onto the rollout for the ledger join.
        assert rollout.completion_ids == expected["completion_ids"]
        # One trajectory entry per generated turn.
        assert len(rollout.trajectory) == len(expected["genmask_sums"])
        # Each turn's inference request = prior conversation (reply + observation appended).
        assert [[m.role for m in conv] for conv in iface.seen_conversations] == expected[
            "seen_roles"
        ]
        # Default trajectory reward scores only the final response.
        assert agent.rewarded == expected["rewarded"]
        # Per-turn generation masks cover exactly each turn's generated tokens.
        assert [sum(mask) for mask in rollout.generation_mask] == expected["genmask_sums"]
        # get_observation is consulted only when another generation is still possible -- never on
        # the final allowed turn.
        assert agent.observation_turns == expected["observation_turns"]
        # get_trajectory_reward sees the full dialogue, ending on the final reply exactly once.
        assert [(m.role, m.content) for m in agent.reward_conversation] == expected["reward_conv"]
