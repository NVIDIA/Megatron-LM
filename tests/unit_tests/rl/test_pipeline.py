# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
import itertools
from collections import Counter
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
from megatron.rl.rollout_bank import RolloutBank
from megatron.rl.types import RolloutGroup


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

    async def prepare_group_rollout(self, request, *, problem_state=None):
        idx = problem_state["idx"] if problem_state else self._call_count
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

    async def prepare_group_rollout(self, request, *, problem_state=None):
        idx = problem_state["idx"] if problem_state else self._call_count
        params = await super().prepare_group_rollout(request)
        degenerate = idx < self.num_degenerate
        rollout_counter = itertools.count()
        base_build = params.build_rollout

        async def build_rollout(episode):
            rollout = await base_build(episode)
            rollout.reward = 0.0 if degenerate else float(next(rollout_counter))
            return rollout

        return GroupRolloutParams(
            run_episode=params.run_episode,
            build_rollout=build_rollout,
            problem_state=params.problem_state,
        )


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

    async def prepare_group_rollout(self, request, *, problem_state=None):
        idx = self._call_count
        params = await super().prepare_group_rollout(request, problem_state=problem_state)
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
            async def prepare_group_rollout(self, request, *, problem_state=None):
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

            async def prepare_group_rollout(self, request, *, problem_state=None):
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

            async def prepare_group_rollout(self, request, *, problem_state=None):
                idx = self._call_count
                params = await super().prepare_group_rollout(request, problem_state=problem_state)
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


class _RestoringGenerator(MockGenerator):
    """MockGenerator that hands back preloaded groups before generating fresh ones."""

    def __init__(self, restored=None, **kwargs):
        super().__init__(**kwargs)
        self._restored = list(restored or [])
        self.resumed_states = []

    def take_restored_group(self, env_id):
        return self._restored.pop(0) if self._restored else None

    async def prepare_group_rollout(self, request, *, problem_state=None):
        if problem_state is not None:
            self.resumed_states.append(problem_state)
        params = await super().prepare_group_rollout(request, problem_state=problem_state)
        # Incomplete groups are only banked for envs that can regenerate them.
        return params._replace(problem_state=problem_state or {"idx": self._call_count})


class _StallingGenerator(_RestoringGenerator):
    """Generator whose episodes past `stall_after` never return, so a group never fills."""

    def __init__(self, stall_after=1, **kwargs):
        super().__init__(**kwargs)
        self.stall_after = stall_after
        self.started = 0
        self._never = asyncio.Event()

    async def prepare_group_rollout(self, request, *, problem_state=None):
        params = await super().prepare_group_rollout(request, problem_state=problem_state)
        base_run = params.run_episode

        async def run_episode():
            self.started += 1
            if self.started > self.stall_after:
                await self._never.wait()
            return await base_run()

        return params._replace(run_episode=run_episode)


class TestProblemStateContract:
    """problem_state is handed out so the bank can store it, and taken back to replay."""

    @staticmethod
    def _request():
        return GroupedRolloutRequest(
            num_groups=1,
            rollouts_per_group=2,
            inference_interface=MockInferenceInterface(),
            generation_args={},
        )

    def test_params_default_to_no_problem_state(self):
        """Agents that opt out keep working; the field is optional."""
        params = GroupRolloutParams(run_episode=lambda: None, build_rollout=lambda e: None)
        assert params.problem_state is None

    @pytest.mark.asyncio
    async def test_agent_exposes_the_problem_it_drew(self):
        agent = CountingRewardAgent()
        params = await agent.prepare_group_rollout(self._request())
        assert params.problem_state == {"prompt": "t0", "golden": {"idx": 0}}

    @pytest.mark.asyncio
    async def test_replaying_a_state_does_not_advance_the_dataset(self):
        """Completing a restored group must reuse its prompt, not draw a new one."""
        agent = CountingRewardAgent()
        first = await agent.prepare_group_rollout(self._request())
        await agent.prepare_group_rollout(self._request())  # dataset moves on

        resumed = await agent.prepare_group_rollout(
            self._request(), problem_state=first.problem_state
        )
        assert agent._prompt_count == 2, "no third draw"
        assert resumed.problem_state == first.problem_state
        episode = await resumed.run_episode()
        assert episode.conversation[0].content == "t0"


class TestDurableBank:
    """The pipeline's half of incomplete-group durability."""

    ROLLOUTS_PER_GROUP = 4

    @classmethod
    def _request(cls, **kwargs):
        kwargs.setdefault("num_groups", 1)
        kwargs.setdefault("rollouts_per_group", cls.ROLLOUTS_PER_GROUP)
        kwargs.setdefault("inference_interface", MockInferenceInterface())
        kwargs.setdefault("submission_granularity", "R")
        kwargs.setdefault("consumption_granularity", "G")
        return GroupedRolloutRequest(**kwargs)

    @classmethod
    def _bank(cls, tmp_path, **kwargs):
        bank = RolloutBank(str(tmp_path), rollouts_per_group=cls.ROLLOUTS_PER_GROUP, **kwargs)
        bank.set_collection(1)
        return bank

    @staticmethod
    async def _take(pipeline, count):
        async with aclosing(pipeline.run()) as stream:
            return [await asyncio.wait_for(anext(stream), timeout=10) for _ in range(count)]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "stall, expected_members",
        [
            pytest.param(False, ROLLOUTS_PER_GROUP, id="complete-group-fully-persisted"),
            pytest.param(True, ROLLOUTS_PER_GROUP - 1, id="stalled-group-persists-what-it-has"),
        ],
    )
    async def test_members_persist_as_they_are_graded(self, tmp_path, stall, expected_members):
        """Members are durable without waiting for the group.

        Under append-at-assembly the stalled case would leave nothing on disk,
        because that group is one member short forever.
        """
        bank = self._bank(tmp_path)
        agent = (
            _StallingGenerator(stall_after=self.ROLLOUTS_PER_GROUP - 1)
            if stall
            else _RestoringGenerator()
        )
        try:
            pipeline = RolloutPipeline(agent, self._request(), parallel_generation_tasks=1, bank=bank)
            if stall:
                async with aclosing(pipeline.run()) as stream:
                    with pytest.raises(asyncio.TimeoutError):
                        await asyncio.wait_for(anext(stream), timeout=1.0)
            else:
                await self._take(pipeline, 1)
                await pipeline.drain_bank()
            restored = bank.restore(0)
        finally:
            bank.close()

        assert len(restored) == 1
        assert len(restored[0].rollouts) == expected_members

    @pytest.mark.asyncio
    async def test_restored_group_regenerates_only_its_missing_members(self, tmp_path):
        """Persisted members are reused; only the gaps are generated, same prompt."""
        incomplete = RolloutGroup(
            rollouts=[
                Rollout(trajectory=["kept-0"], reward=1.0, env_id="test"),
                Rollout(trajectory=["kept-2"], reward=0.0, env_id="test"),
            ],
            uid="nonce/7",
            member_indices=[0, 2],
            problem_state={"idx": 0},
        )
        agent = _RestoringGenerator(restored=[incomplete])
        bank = self._bank(tmp_path)
        try:
            pipeline = RolloutPipeline(agent, self._request(), parallel_generation_tasks=1, bank=bank)
            group = (await self._take(pipeline, 1))[0]
        finally:
            bank.close()

        assert len(group.rollouts) == self.ROLLOUTS_PER_GROUP
        assert group.uid == "nonce/7", "completion must not mint a new identity"
        assert group.member_indices == [0, 1, 2, 3]
        assert agent.resumed_states == [{"idx": 0}]
        kept = [r.trajectory[0] for r in group.rollouts if r.trajectory[0].startswith("kept")]
        assert sorted(kept) == ["kept-0", "kept-2"], "persisted members are not regenerated"

    @pytest.mark.asyncio
    async def test_writes_are_coalesced_and_run_off_the_event_loop(self, tmp_path):
        """fsync is blocking, so it must batch and must not run on the loop thread."""
        import threading

        loop_thread = threading.get_ident()
        writes, threads = [], set()

        class RecordingBank(RolloutBank):
            def write_records(self, pending):
                writes.append(len(pending))
                threads.add(threading.get_ident())
                super().write_records(pending)

        bank = RecordingBank(str(tmp_path), rollouts_per_group=self.ROLLOUTS_PER_GROUP)
        bank.set_collection(1)
        try:
            pipeline = RolloutPipeline(
                _RestoringGenerator(),
                self._request(num_groups=2),
                parallel_generation_tasks=2,
                bank=bank,
            )
            await self._take(pipeline, 2)
            await pipeline.drain_bank()
        finally:
            bank.close()

        assert threads and loop_thread not in threads, "a blocking fsync ran on the event loop"
        assert len(writes) < sum(writes), "records must coalesce, not fsync one at a time"

    @pytest.mark.asyncio
    async def test_write_failure_surfaces_at_the_durability_barrier(self, tmp_path):
        """Persistence is best-effort mid-stream, but a failure cannot be swallowed."""

        class BrokenBank(RolloutBank):
            def write_records(self, pending):
                raise RuntimeError("disk exploded")

        bank = BrokenBank(str(tmp_path), rollouts_per_group=self.ROLLOUTS_PER_GROUP)
        bank.set_collection(1)
        try:
            pipeline = RolloutPipeline(
                _RestoringGenerator(), self._request(), parallel_generation_tasks=1, bank=bank
            )
            groups = await self._take(pipeline, 1)
            assert len(groups) == 1, "a bank failure must not stop rollout generation"
            with pytest.raises(RuntimeError, match="disk exploded"):
                await pipeline.drain_bank()
            await pipeline.drain_bank()  # latched errors report once, then clear
        finally:
            bank.close()

    @pytest.mark.asyncio
    async def test_placeholders_are_not_persisted(self, tmp_path):
        """A failed episode has no decode work to save, and must not be restored.

        main drops all-placeholder groups so they can be refilled. Persisting a
        placeholder would let a restored group carry one past that policy; leaving
        the slot empty means restore regenerates it, which is the same outcome.
        """
        # One member of the first group comes back empty; the rest are real.
        agent = PlaceholderMockGenerator(num_placeholder=1, placeholder_members=1)
        bank = self._bank(tmp_path)
        try:
            pipeline = RolloutPipeline(
                agent, self._request(), parallel_generation_tasks=1, bank=bank
            )
            await self._take(pipeline, 1)
            await pipeline.drain_bank()
            restored = bank.restore(0)
        finally:
            bank.close()

        for group in restored:
            assert all(not r.is_placeholder for r in group.rollouts), "placeholder persisted"

    @pytest.mark.asyncio
    async def test_pipeline_without_a_bank_is_unaffected(self, tmp_path):
        """Persistence is opt-in."""
        pipeline = RolloutPipeline(
            _RestoringGenerator(), self._request(), parallel_generation_tasks=1
        )
        group = (await self._take(pipeline, 1))[0]
        assert len(group.rollouts) == self.ROLLOUTS_PER_GROUP
        assert group.uid is None


class TestRestoredGroupRouting:
    """Recovered groups keep their per-env weighting and drain before fresh work."""

    @staticmethod
    def _env_group(env_id, problem_id):
        return RolloutGroup(
            rollouts=[
                Rollout(trajectory=["cached"], reward=1.0, env_id=env_id, problem_id=problem_id)
            ]
        )

    @staticmethod
    def _weighted_agent(env_weights):
        return WeightedMultiTask(
            [
                AgentConfig(agent_type=MockGenerator, agent_args={"env_id": env_id}, weight=weight)
                for env_id, weight in env_weights
            ]
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "env_weights, restored_env, restored_count, request_groups, take_count, "
        "expected_envs, expected_fresh_calls, write_through",
        [
            pytest.param(
                [("a", 1.0), ("b", 1.0)], "a", 2, 4, 4, {"a": 2, "b": 2}, [0, 2], False,
                id="restored-groups-replace-fresh",
            ),
            pytest.param(
                [("a", 1.0), ("b", 1.0)], "a", 4, 4, 12, {"a": 6, "b": 6}, [2, 6], False,
                id="streaming-backlog-drains-before-fresh",
            ),
            pytest.param(
                [("", 1.0)], "", 1, 2, 2, {"": 2}, [1], False, id="single-env-without-env-id"
            ),
            pytest.param(
                [("a", 1.0)], "a", 0, 4, 4, {"a": 4}, [4], True, id="fresh-groups-write-through"
            ),
        ],
    )
    async def test_restored_groups_are_consumed_before_fresh_generation(
        self, tmp_path, env_weights, restored_env, restored_count, request_groups, take_count,
        expected_envs, expected_fresh_calls, write_through,
    ):
        agent = self._weighted_agent(env_weights)
        restored = [
            self._env_group(restored_env, f"cached-{i}") for i in range(restored_count)
        ]
        assert agent.set_restored_groups(restored) == restored_count

        if len(env_weights) > 1:
            with pytest.raises(ValueError, match="not in the current"):
                agent.set_restored_groups([self._env_group("unknown", "drift")])
            assert agent.set_restored_groups(restored) == restored_count

        bank = RolloutBank(str(tmp_path), rollouts_per_group=1) if write_through else None
        if bank is not None:
            bank.set_collection(0)
        request = GroupedRolloutRequest(
            num_groups=request_groups,
            rollouts_per_group=1,
            inference_interface=MockInferenceInterface(),
        )
        pipeline = RolloutPipeline(
            agent, request, parallel_generation_tasks=1, initial_batch_id=20, bank=bank
        )

        async with aclosing(pipeline.run()) as groups:
            produced = [
                await asyncio.wait_for(anext(groups), timeout=10) for _ in range(take_count)
            ]
            if bank is not None:
                await pipeline.drain_bank()

        assert Counter(group[0].env_id for group in produced) == expected_envs
        assert [g.prepare_group_rollout_calls for g in agent.agents] == expected_fresh_calls
        assert {f"cached-{i}" for i in range(restored_count)} <= {
            group[0].problem_id for group in produced
        }
        assert not agent._restored_groups.get(restored_env)
        assert [group.batch_id for group in produced] == [
            20 + i // request_groups for i in range(take_count)
        ]
        assert [group.index_in_batch for group in produced] == [
            i % request_groups for i in range(take_count)
        ]

        if bank is not None:
            bank.close()
            persisted = RolloutBank(str(tmp_path), rollouts_per_group=1).restore(trained_through=0)
            assert {group.uid for group in persisted} == {group.uid for group in produced}

    def test_multiple_envs_require_env_ids_for_restore_routing(self):
        agent = self._weighted_agent([("", 1.0), ("b", 1.0)])
        with pytest.raises(ValueError, match="configuring multiple active agents"):
            agent.set_restored_groups([])
