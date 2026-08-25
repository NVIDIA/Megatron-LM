# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
import itertools
from contextlib import aclosing
from unittest.mock import AsyncMock, MagicMock

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
from megatron.rl.inference import (
    InferenceRequest,
    InferenceResponse,
    LLMChatMessage,
    ReturnsRaw,
    ReturnsTokens,
)
from megatron.rl.inference.megatron import MegatronLocal


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "temperature, expected_temperature", [(None, 1.0), (0.0, 0.0)], ids=["default", "greedy"]
)
async def test_megatron_local_preserves_explicit_greedy_temperature(
    monkeypatch, temperature, expected_temperature
):
    monkeypatch.setattr("megatron.rl.inference.megatron.get_args", lambda: MagicMock())
    monkeypatch.setattr("megatron.rl.inference.megatron.get_tokenizer", lambda: MagicMock(bos=None))

    choice = MagicMock(finish_reason="stop")
    choice.message.model_dump.return_value = {"role": "assistant", "content": "response"}
    choice.message.raw_text = "response"
    choice.message.prompt_token_ids = [1]
    choice.message.generation_token_ids = [2]
    choice.message.generation_log_probs = [0.0]
    choice.message.policy_epoch = [(0, 0)]
    choice.message.kv_cache_epoch = [(0, 0)]
    choice.message.num_evictions = 0

    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=MagicMock(choices=[choice]))
    server = MegatronLocal(host="localhost", port=0)
    server._openai_client = client
    request = InferenceRequest(
        prompt=[LLMChatMessage(role="user", content="prompt")],
        generation_args={"temperature": temperature},
    )

    await server.base_generate(request)

    client.chat.completions.create.assert_awaited_once()
    assert client.chat.completions.create.await_args.kwargs["temperature"] == expected_temperature


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


class PlaceholderMockGenerator(MockGenerator):
    """First `num_placeholder` prepared groups contain empty-trajectory members.

    placeholder_members bounds how many members of each affected group come
    back empty (None => every member). Failed episodes keep the group
    rectangular with empty trajectories, which is how placeholders arrive
    from the environment side; placeholder_status/placeholder_reason stamp
    the adapter-side status and failure-cause labels on them.
    """

    def __init__(
        self,
        num_placeholder=0,
        placeholder_members=None,
        placeholder_status='ok',
        placeholder_reason=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_placeholder = num_placeholder
        self.placeholder_members = placeholder_members
        self.placeholder_status = placeholder_status
        self.placeholder_reason = placeholder_reason

    async def prepare_group_rollout(self, request):
        idx = self._call_count
        params = await super().prepare_group_rollout(request)
        make_placeholder = idx < self.num_placeholder
        member_counter = itertools.count()
        base_build = params.build_rollout

        async def build_rollout(episode):
            rollout = await base_build(episode)
            member = next(member_counter)
            if make_placeholder and (
                self.placeholder_members is None or member < self.placeholder_members
            ):
                rollout.trajectory = []
                rollout.rollout_status = self.placeholder_status
                rollout.failure_reason = self.placeholder_reason
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


def _assert_batches_arrive_in_submission_order(groups, num_groups):
    """Batches complete and arrive in submission order despite drops/refills."""
    assert [g.batch_id for g in groups] == sorted(g.batch_id for g in groups)
    for batch_start in range(0, len(groups), num_groups):
        batch = groups[batch_start : batch_start + num_groups]
        assert sorted(g.index_in_batch for g in batch) == list(range(num_groups))


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
        (
            "submission_granularity, consumption_granularity, num_degenerate, "
            "parallel_generation_tasks"
        ),
        [
            pytest.param("B", "B", 3, 1, id="batch_submission"),
            pytest.param("G", "B", 3, 8, id="group_submission"),
            pytest.param("R", "B", 3, 8, id="rollout_submission"),
            pytest.param("G", "G", 3, 8, id="group_consumption"),
            pytest.param("R", "G", 3, 8, id="rollout_submission_group_consume"),
            pytest.param("B", "B", 9, 1, id="cascading_regeneration"),
        ],
    )
    async def test_filter_groups_and_regenerate(
        self,
        submission_granularity,
        consumption_granularity,
        num_degenerate,
        parallel_generation_tasks,
    ):
        num_groups = 4
        gen = FilteringMockGenerator(num_degenerate=num_degenerate)
        request = GroupedRolloutRequest(
            num_groups=num_groups,
            rollouts_per_group=2,
            inference_interface=MockInferenceInterface(),
            filter_groups_with_same_reward=True,
            submission_granularity=submission_granularity,
            consumption_granularity=consumption_granularity,
        )
        pipeline = RolloutPipeline(
            gen, request, parallel_generation_tasks=parallel_generation_tasks
        )

        expected_count = 2 * num_groups
        groups = []
        async with aclosing(pipeline.run()) as it:
            async for group in it:
                groups.append(group)
                if len(groups) >= expected_count:
                    break
            if submission_granularity == "B" and parallel_generation_tasks == 1:
                # lag=0: the boundary is quiescent, checked before close() drops
                # shutdown sentinels into the queues on Python < 3.13.
                pipeline.assert_no_inflight_rollouts()

        assert len(groups) == expected_count
        # Every delivered group carries reward signal.
        for group in groups:
            assert np.std([rollout.reward for rollout in group]) > 1e-6
        if consumption_granularity == "B":
            _assert_batches_arrive_in_submission_order(groups, num_groups)
        if submission_granularity == "B" and parallel_generation_tasks == 1:
            # Every dropped group cost exactly one extra prepare (no under-delivery).
            assert gen.prepare_group_rollout_calls == expected_count + num_degenerate
            assert pipeline.filtered_count == num_degenerate

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
            # t0-t2 plus the pinned regeneration, plus the streaming prepare of
            # the next batch's first group (pinned on the same full gate).
            assert gen.prepare_group_rollout_calls == 5
            events["t2"].set()
            # Completion order may interleave the perpetual stream's next-batch
            # groups; keep pulling (bounded) until both stragglers land.
            rest = []
            for _ in range(5):
                rest.append(await asyncio.wait_for(anext(it), timeout=10))
                if {(0, 0), (0, 2)} <= {(g.batch_id, g.index_in_batch) for g in rest}:
                    break
        # The gated group (slot 2) and the regenerated replacement (slot 0) both deliver.
        assert {(0, 0), (0, 2)} <= {(g.batch_id, g.index_in_batch) for g in rest}
        assert not pipeline._regen_tasks

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        (
            "pull_two_windows, submission_granularity, consumption_granularity, "
            "num_placeholder, placeholder_members, parallel_generation_tasks"
        ),
        [
            pytest.param(False, "B", "B", 3, None, 8, id="batch_all_placeholder"),
            # Depth-1 gate so the refilled groups deliver inside the observed
            # window and their contents are actually inspected.
            pytest.param(False, "G", "G", 3, None, 1, id="group_all_placeholder"),
            pytest.param(False, "B", "B", 3, 1, 8, id="partial_group_delivers"),
            # Depth-1 gate so refills draw placeholder indices — i.e. refilled
            # groups are themselves refilled.
            pytest.param(False, "B", "B", 9, None, 1, id="cascading_refill"),
            pytest.param(True, "B", "B", 3, None, 8, id="two_window_batch_consume"),
            pytest.param(True, "G", "G", 3, 1, 8, id="two_window_partial_delivers"),
        ],
    )
    async def test_all_placeholder_groups_are_refilled(
        self,
        pull_two_windows,
        submission_granularity,
        consumption_granularity,
        num_placeholder,
        placeholder_members,
        parallel_generation_tasks,
    ):
        """Groups whose members are ALL empty-trajectory placeholders are
        dropped and regenerated in place; groups with any real member deliver
        as-is (their placeholders contribute zero training rows). Unconditional:
        no filter flag is set."""
        num_groups = 4
        gen = PlaceholderMockGenerator(
            num_placeholder=num_placeholder, placeholder_members=placeholder_members
        )
        request = GroupedRolloutRequest(
            num_groups=num_groups,
            rollouts_per_group=2,
            inference_interface=MockInferenceInterface(),
            submission_granularity=submission_granularity,
            consumption_granularity=consumption_granularity,
        )

        pipeline = RolloutPipeline(
            gen, request, parallel_generation_tasks=parallel_generation_tasks
        )

        all_placeholder = placeholder_members is None
        expected_count = 2 * num_groups if pull_two_windows else num_groups
        groups, refilled_at_delivery = [], 0
        async with aclosing(pipeline.run()) as it:
            # stage_prepare no longer stops itself for non-streaming requests
            # (persistent-stream redesign); every caller bounds its own pull.
            while len(groups) < expected_count:
                # wait_for turns the deadlock failure mode (a refill never
                # resubmitted) into a test failure instead of a hang.
                groups.append(await asyncio.wait_for(anext(it), timeout=10))
                refilled_at_delivery = pipeline.refilled_placeholder_groups
            if submission_granularity == "B" and parallel_generation_tasks == 1:
                # lag=0: the boundary is quiescent; this also locks the
                # prepared == (yielded + filtered + refilled) * R identity
                # under a refill storm.
                pipeline.assert_no_inflight_rollouts()

        assert len(groups) == expected_count
        if not pull_two_windows:
            # Refills land in the dropped groups' own batch slots: the observed
            # window is exactly batch 0, whole.
            assert {(g.batch_id, g.index_in_batch) for g in groups} == {
                (0, index) for index in range(num_groups)
            }
        if all_placeholder:
            # Refilled: no placeholder member ever reaches the consumer.
            for group in groups:
                for rollout in group:
                    assert rollout.trajectory
        else:
            # Partial groups deliver their real members alongside the padding,
            # and nothing is refilled.
            placeholder_groups = [g for g in groups if any(not r.trajectory for r in g)]
            assert placeholder_groups
            if not pull_two_windows:
                assert len(placeholder_groups) == num_placeholder
            for group in placeholder_groups:
                assert any(r.trajectory for r in group)
        if consumption_granularity == "B":
            _assert_batches_arrive_in_submission_order(groups, num_groups)
        # No under-delivery: exactly the expected refills by the time the
        # observed window is fully delivered (placeholders only ever affect
        # batch 0's prepare indices, so this holds for two-window pulls too).
        expected_refills = num_placeholder if all_placeholder else 0
        assert refilled_at_delivery == expected_refills
        # The accounting total and the attribution counters stay in lockstep.
        assert pipeline.dropped_count == (
            pipeline.filtered_count + pipeline.refilled_placeholder_groups
        )
        if submission_granularity == "B" and parallel_generation_tasks == 1:
            # Every refill cost exactly one extra prepare (no under-delivery).
            assert gen.prepare_group_rollout_calls == num_groups + expected_refills

    @pytest.mark.asyncio
    async def test_refill_diagnostics_survive_metric_resets(self):
        """The refill warning's lifetime count and the dropped members' failure
        reasons are kept independently of the public per-window counter that
        every metrics collection zeroes."""
        num_placeholder = 3
        num_groups = 4
        rollouts_per_group = 2
        gen = PlaceholderMockGenerator(
            num_placeholder=num_placeholder, placeholder_reason="episode timeout"
        )
        request = GroupedRolloutRequest(
            num_groups=num_groups,
            rollouts_per_group=rollouts_per_group,
            inference_interface=MockInferenceInterface(),
            submission_granularity="B",
            consumption_granularity="B",
        )
        pipeline = RolloutPipeline(gen, request, parallel_generation_tasks=1)

        groups = []
        async with aclosing(pipeline.run()) as it:
            while len(groups) < num_groups:
                groups.append(await asyncio.wait_for(anext(it), timeout=10))
            pipeline.assert_no_inflight_rollouts()

        # The dropped originals' failure reasons are kept for the metrics
        # export; their real replacements contribute none.
        assert pipeline.refill_failure_reasons == {
            "episode timeout": num_placeholder * rollouts_per_group
        }
        # The warning throttle counts for the pipeline's lifetime: zeroing the
        # public counter (as every metrics collection does) must not reset it.
        pipeline.refilled_placeholder_groups = 0
        assert pipeline._lifetime_refilled_groups == num_placeholder

    @pytest.mark.asyncio
    async def test_deliberately_voided_empty_groups_deliver(self):
        """Empty-trajectory rollouts stamped with a deliberate non-placeholder
        status ('masked'/'graded') are adapter-voided on purpose: their groups
        must deliver for status accounting, not be refilled."""
        num_masked = 2
        num_groups = 4
        gen = PlaceholderMockGenerator(num_placeholder=num_masked, placeholder_status='masked')
        request = GroupedRolloutRequest(
            num_groups=num_groups,
            rollouts_per_group=2,
            inference_interface=MockInferenceInterface(),
            submission_granularity="B",
            consumption_granularity="B",
        )
        pipeline = RolloutPipeline(gen, request, parallel_generation_tasks=1)

        groups = []
        async with aclosing(pipeline.run()) as it:
            while len(groups) < num_groups:
                groups.append(await asyncio.wait_for(anext(it), timeout=10))
            pipeline.assert_no_inflight_rollouts()

        voided = [g for g in groups if all(r.is_placeholder for r in g)]
        assert len(voided) == num_masked
        for group in voided:
            assert all(r.rollout_status == 'masked' for r in group)
        # The refill path never touched them.
        assert pipeline.refilled_placeholder_groups == 0
        assert gen.prepare_group_rollout_calls == num_groups

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "placeholder_reward",
        [
            # An unrewarded failed episode used to crash np.std with a
            # TypeError; a numeric sentinel used to manufacture fake variance
            # that let the zero-signal group through.
            pytest.param(None, id="unrewarded_placeholder"),
            pytest.param(0.0, id="sentinel_rewarded_placeholder"),
        ],
    )
    async def test_same_reward_filter_ignores_placeholder_members(self, placeholder_reward):
        """The same-reward filter judges variance over real members only —
        the same population the trainer's masked group statistics use."""

        class PartialDegenerateMockGenerator(MockGenerator):
            """Group 0: member 0 fails, real members share one reward; later
            groups carry distinct member rewards."""

            async def prepare_group_rollout(self, request):
                idx = self._call_count
                params = await super().prepare_group_rollout(request)
                affected = idx < 1
                member_counter = itertools.count()
                base_build = params.build_rollout

                async def build_rollout(episode):
                    rollout = await base_build(episode)
                    member = next(member_counter)
                    if affected and member == 0:
                        rollout.trajectory = []
                        rollout.reward = placeholder_reward
                    elif affected:
                        rollout.reward = 1.0
                    else:
                        rollout.reward = float(member)
                    return rollout

                return GroupRolloutParams(
                    run_episode=params.run_episode, build_rollout=build_rollout
                )

        num_groups = 4
        gen = PartialDegenerateMockGenerator()
        request = GroupedRolloutRequest(
            num_groups=num_groups,
            rollouts_per_group=2,
            inference_interface=MockInferenceInterface(),
            filter_groups_with_same_reward=True,
            submission_granularity="B",
            consumption_granularity="B",
        )
        pipeline = RolloutPipeline(gen, request, parallel_generation_tasks=1)

        groups = []
        async with aclosing(pipeline.run()) as it:
            while len(groups) < num_groups:
                # Pre-fix, the None-reward case killed stage_assemble with a
                # TypeError inside np.std and this pull raised RuntimeError.
                groups.append(await asyncio.wait_for(anext(it), timeout=10))
            pipeline.assert_no_inflight_rollouts()

        # Group 0's lone real member carries no signal (the trainer would give
        # it advantage exactly 0), so the group is filtered and regenerated —
        # not delivered on placeholder-manufactured variance, and not a crash.
        assert pipeline.dropped_count == 1
        assert pipeline.filtered_count == 1
        assert pipeline.refilled_placeholder_groups == 0
        assert gen.prepare_group_rollout_calls == num_groups + 1
        for group in groups:
            real_rewards = [r.reward for r in group if r.trajectory]
            assert np.std(real_rewards) > 1e-6

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
    async def test_batch_order_starts_at_initial_batch_id(self):
        request = GroupedRolloutRequest(
            num_groups=2,
            rollouts_per_group=1,
            inference_interface=MockInferenceInterface(),
            streaming=True,
            submission_granularity="B",
            consumption_granularity="B",
        )
        pipeline = RolloutPipeline(
            MockGenerator(), request, parallel_generation_tasks=1, initial_batch_id=10
        )

        async with aclosing(pipeline.run()) as groups:
            first_two_batches = [
                await asyncio.wait_for(anext(groups), timeout=10) for _ in range(4)
            ]

        assert [group.batch_id for group in first_two_batches] == [10, 10, 11, 11]
        assert [group.index_in_batch for group in first_two_batches] == [0, 1, 0, 1]

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
        (
            "submission_granularity, consumption_granularity, num_slow_calls, "
            "parallel_generation_tasks, windows"
        ),
        [
            pytest.param("B", "B", 0, 1, 2, id="batch_batch"),
            pytest.param("G", "G", 0, 1, 2, id="group_group"),
            # Balanced G under out-of-order completion and a depth-2 gate.
            pytest.param("G", "G", 2, 2, 3, id="group_group_out_of_order"),
        ],
    )
    async def test_weighted_multi_task(
        self,
        submission_granularity,
        consumption_granularity,
        num_slow_calls,
        parallel_generation_tasks,
        windows,
    ):
        """Routing and consumption keep every trainer-batch window at the exact env mix."""
        mt = WeightedMultiTask(
            [
                AgentConfig(agent_type=MockGenerator, agent_args={"env_id": "a"}, weight=3.0),
                AgentConfig(agent_type=MockGenerator, agent_args={"env_id": "b"}, weight=1.0),
            ]
        )
        request = GroupedRolloutRequest(
            num_groups=4,
            rollouts_per_group=1,
            inference_interface=MockInferenceInterface(num_slow_calls=num_slow_calls),
            submission_granularity=submission_granularity,
            consumption_granularity=consumption_granularity,
        )
        pipeline = RolloutPipeline(mt, request, parallel_generation_tasks=parallel_generation_tasks)
        gen = pipeline.run()
        groups = [await anext(gen) for _ in range(windows * 4)]

        assert [a.num_groups for a in pipeline.allocations] == [3, 1]
        # Weights 3:1 → env "a" owns 3 of every 4 batch slots; the pipeline routes
        # each slot to the owning sub-agent regardless of completion order.
        env_ids = [g[0].env_id for g in groups]
        for start in range(0, windows * 4, 4):
            assert sorted(env_ids[start : start + 4]) == ["a", "a", "a", "b"]
        if consumption_granularity == "B":
            # With depth-1 gating and consumed-release, nothing is buffered or in flight.
            pipeline.assert_no_inflight_rollouts()

    @pytest.mark.asyncio
    async def test_lag0_streaming_matches_non_streaming_boundaries(self):
        """lag=0 (B/B, depth-1 gate): each iteration of the persistent stream is exactly
        one batch, generated entirely after the previous boundary — the old
        non-streaming per-iteration contract, enforced by assert_no_inflight_rollouts."""
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
            pipeline.assert_no_inflight_rollouts()

    @pytest.mark.asyncio
    async def test_assert_no_inflight_rollouts_detects_run_ahead(self):
        """With lag>0 the gate legitimately runs ahead; the boundary checker must fire."""
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
        with pytest.raises(AssertionError, match="The rollout pipeline"):
            pipeline.assert_no_inflight_rollouts()

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
        """Weights quantize to a constant split (warned); weight-0 and eval-only envs take
        no slot — the min-one-group bump must not revive boot-only entries."""
        configs = [
            AgentConfig(agent_type=MockGenerator, agent_args={"env_id": f"e{i}"}, weight=w)
            for i, w in enumerate(weights)
        ]
        configs.insert(
            1, AgentConfig(agent_type=MockGenerator, agent_args={"env_id": "boot"}, weight=0.0)
        )
        configs.append(
            AgentConfig(
                agent_type=MockGenerator,
                agent_args={"env_id": "eval"},
                weight=1.0,
                evaluation_only=True,
            )
        )
        mt = WeightedMultiTask(configs)
        assert mt._rollout_env_ids == [f"e{i}" for i in range(len(weights))]
        if expected_layout is None:
            with pytest.raises(ValueError, match="cannot fit"):
                mt.rollout_allocations(num_groups)
            with pytest.raises(ValueError, match="cannot fit"):
                mt._distribute_counts(num_groups)
            return
        # The split is identical on every call.
        assert [[a.num_groups for a in mt.rollout_allocations(num_groups)] for _ in range(3)] == [
            expected_layout
        ] * 3
        assert warns == any("weights changed" in message for message in caplog.messages)
        # Config-order distribution: the boot (index 1) and eval (last) slots pinned to 0.
        assert mt._distribute_counts(num_groups) == (
            expected_layout[:1] + [0] + expected_layout[1:] + [0]
        )


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
