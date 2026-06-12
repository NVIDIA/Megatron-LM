# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
from unittest.mock import MagicMock

import pytest
from pydantic import Field, PrivateAttr

from megatron.rl.agent.api import (
    GroupedRolloutGenerator,
    GroupedRolloutRequest,
    Rollout,
    RolloutGenerator,
    RolloutGroup,
)
from megatron.rl.agent.reward_only_agent import RewardOnlyAgent
from megatron.rl.agent.weighted_multi_task import AgentConfig, WeightedMultiTask
from megatron.rl.inference import InferenceResponse, LLMChatMessage, ReturnsRaw
from megatron.rl.rollout_granularity import RLRolloutGranularity


class MockGenerator(RolloutGenerator, GroupedRolloutGenerator):
    """Mock generator with configurable per-call delays."""

    def __init__(self, env_id="test", num_slow_calls=0, **kwargs):
        super().__init__(**kwargs)
        self.env_id = env_id
        self.num_slow_calls = num_slow_calls
        self._call_count = 0

    async def rollout(self, request):
        raise NotImplementedError

    async def group_rollout(self, request, submission_gate=None):
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


class RecordingInference(ReturnsRaw):
    """Inference interface that records rollout start/finish order."""

    slow_calls: set[int] = Field(default_factory=set)
    same_reward_prompt_ids: set[int] = Field(default_factory=set)
    _call_count: int = PrivateAttr(default=0)
    _prompt_call_counts: dict[str, int] = PrivateAttr(default_factory=dict)
    _events: list[tuple[str, int, str, int]] = PrivateAttr(default_factory=list)

    @property
    def events(self):
        return self._events

    async def base_generate(self, request):
        call_id = self._call_count
        self._call_count += 1
        prompt = request.prompt[0].content
        prompt_rollout_idx = self._prompt_call_counts.get(prompt, 0)
        self._prompt_call_counts[prompt] = prompt_rollout_idx + 1
        self._events.append(("start", call_id, prompt, prompt_rollout_idx))
        if call_id in self.slow_calls:
            await asyncio.sleep(0.03)
        else:
            await asyncio.sleep(0)
        self._events.append(("finish", call_id, prompt, prompt_rollout_idx))

        prompt_id = int(prompt.rsplit("-", 1)[1])
        reward = 0 if prompt_id in self.same_reward_prompt_ids else prompt_rollout_idx
        return InferenceResponse(
            response=LLMChatMessage(role="assistant", content=str(reward)),
            raw_text=f"{prompt}/rollout-{prompt_rollout_idx}/call-{call_id}",
            finish_reason="stop",
            policy_epoch=[(0, call_id)],
            kv_cache_epoch=[(0, call_id)],
            num_evictions=0,
        )


class RecordingRewardOnlyAgent(RewardOnlyAgent):
    """RewardOnlyAgent test double that issues deterministic prompt ids."""

    env_id: str = "recording"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prompt_count = 0

    async def get_prompt(self, validation: bool):
        prompt_id = self.prompt_count
        self.prompt_count += 1
        return f"prompt-{prompt_id}", {
            "problem_id": f"problem-{prompt_id}",
            "prompt_id": prompt_id,
        }

    async def get_reward(self, response: str, golden, finish_reason: str) -> float:
        return float(response)

    async def evaluation_prompts(self, num_prompts: int, validation: bool = False):
        return []


async def collect_groups(agent, request, count):
    groups = []
    async for group in agent.get_grouped_rollouts(request):
        groups.append(group)
        if len(groups) >= count:
            break
    return groups


def group_prompt(group):
    return group[0].trajectory[0].split("/", maxsplit=1)[0]


def event_index(events, event_name, *, call_id=None, prompt=None):
    for idx, event in enumerate(events):
        name, event_call_id, event_prompt, _ = event
        if name != event_name:
            continue
        if call_id is not None and event_call_id != call_id:
            continue
        if prompt is not None and event_prompt != prompt:
            continue
        return idx
    raise AssertionError(f"Event not found: {event_name=} {call_id=} {prompt=}")


class TestGroupedRollouts:

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "case",
        [
            pytest.param(
                {"num_slow_calls": 0, "streaming": False, "num_groups": 8, "expected_count": 8},
                id="non_batched",
            ),
            pytest.param(
                {"num_slow_calls": 0, "streaming": False, "num_groups": 4, "expected_count": 4},
                id="non_streaming_fewer_than_parallel",
            ),
            pytest.param(
                {
                    "num_slow_calls": 4,
                    "streaming": True,
                    "num_groups": 2,
                    "expected_count": 8,
                    "expected_batch_ids": [0, 0, 1, 1, 2, 2, 3, 3],
                    "expected_index_in_batch": [0, 1, 0, 1, 0, 1, 0, 1],
                },
                id="batched_submission_order",
            ),
            pytest.param(
                {"num_slow_calls": 0, "streaming": True, "num_groups": 1, "expected_count": 10},
                id="streaming",
            ),
            pytest.param(
                {
                    "num_slow_calls": 4,
                    "streaming": True,
                    "num_groups": 1,
                    "expected_count": 8,
                    "enforce_order": False,
                    "expected_trajectory_prefix": [f"t{i}" for i in range(4, 8)],
                },
                id="group_submit_group_consume_completion_order",
            ),
            pytest.param(
                {
                    "num_slow_calls": 4,
                    "streaming": True,
                    "num_groups": 1,
                    "expected_count": 8,
                    "enforce_order": True,
                    "expected_batch_ids": list(range(8)),
                    "expected_index_in_batch": [0] * 8,
                    "expected_trajectories": [f"t{i}" for i in range(8)],
                },
                id="group_submit_batch_consume_submission_order",
            ),
        ],
    )
    async def test_get_grouped_rollouts(self, case):
        gen = MockGenerator(
            parallel_generation_tasks=case.get("parallel_generation_tasks", 8),
            num_slow_calls=case.get("num_slow_calls", 0),
        )

        request = GroupedRolloutRequest(
            num_groups=case["num_groups"],
            rollouts_per_group=case.get("rollouts_per_group", 1),
            inference_interface=MagicMock(spec=ReturnsRaw),
            streaming=case["streaming"],
            enforce_order=case.get("enforce_order", case["num_groups"] > 1),
            filter_groups_with_same_reward=case.get("filter_groups_with_same_reward", False),
        )

        async def collect_groups():
            groups = []
            async for group in gen.get_grouped_rollouts(request):
                groups.append(group)
                if request.streaming and len(groups) >= case["expected_count"]:
                    break
            return groups

        if "timeout" in case:
            groups = await asyncio.wait_for(collect_groups(), timeout=case["timeout"])
        else:
            groups = await collect_groups()

        has_order_expectations = any(
            key in case
            for key in (
                "expected_batch_ids",
                "expected_index_in_batch",
                "expected_trajectory_prefix",
                "expected_trajectories",
            )
        )
        if not has_order_expectations:
            assert len(groups) == case["expected_count"]
        if "expected_batch_ids" in case:
            assert [g.batch_id for g in groups] == case["expected_batch_ids"]
        if "expected_index_in_batch" in case:
            assert [g.index_in_batch for g in groups] == case["expected_index_in_batch"]
        if "expected_trajectory_prefix" in case:
            expected = case["expected_trajectory_prefix"]
            assert [group[0].trajectory[0] for group in groups[: len(expected)]] == expected
        if "expected_trajectories" in case:
            assert [group[0].trajectory[0] for group in groups] == case["expected_trajectories"]

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

    @pytest.mark.asyncio
    async def test_rollout_submit_reuses_prompt_within_group(self):
        gen = RecordingRewardOnlyAgent(parallel_generation_tasks=2)
        request = GroupedRolloutRequest(
            num_groups=1,
            rollouts_per_group=2,
            inference_interface=RecordingInference(),
            streaming=True,
            enforce_order=True,
            submission_granularity=RLRolloutGranularity.ROLLOUT,
        )

        groups = await collect_groups(gen, request, 1)

        assert [rollout.trajectory[0].split("/", maxsplit=1)[0] for rollout in groups[0]] == [
            "prompt-0",
            "prompt-0",
        ]
        assert [rollout.problem_id for rollout in groups[0]] == ["problem-0", "problem-0"]

    @pytest.mark.asyncio
    async def test_rollout_submit_starts_later_group_before_slow_peer_finishes(self):
        inference = RecordingInference(slow_calls={0})
        gen = RecordingRewardOnlyAgent(parallel_generation_tasks=2)
        request = GroupedRolloutRequest(
            num_groups=1,
            rollouts_per_group=2,
            inference_interface=inference,
            streaming=True,
            enforce_order=True,
            submission_granularity=RLRolloutGranularity.ROLLOUT,
        )

        await collect_groups(gen, request, 1)

        assert event_index(inference.events, "start", prompt="prompt-1") < event_index(
            inference.events, "finish", call_id=0
        )

    @pytest.mark.asyncio
    async def test_rollout_submit_enforced_order_yields_submission_order(self):
        inference = RecordingInference(slow_calls={0})
        gen = RecordingRewardOnlyAgent(parallel_generation_tasks=2)
        request = GroupedRolloutRequest(
            num_groups=1,
            rollouts_per_group=2,
            inference_interface=inference,
            streaming=True,
            enforce_order=True,
            submission_granularity=RLRolloutGranularity.ROLLOUT,
        )

        groups = await collect_groups(gen, request, 2)

        assert [group_prompt(group) for group in groups] == ["prompt-0", "prompt-1"]
        assert event_index(inference.events, "finish", prompt="prompt-1") < event_index(
            inference.events, "finish", call_id=0
        )

    @pytest.mark.asyncio
    async def test_rollout_submit_unordered_yields_completion_order(self):
        gen = RecordingRewardOnlyAgent(parallel_generation_tasks=2)
        request = GroupedRolloutRequest(
            num_groups=1,
            rollouts_per_group=2,
            inference_interface=RecordingInference(slow_calls={0}),
            streaming=True,
            enforce_order=False,
            submission_granularity=RLRolloutGranularity.ROLLOUT,
        )

        groups = await collect_groups(gen, request, 2)

        assert [group_prompt(group) for group in groups] == ["prompt-1", "prompt-0"]

    @pytest.mark.asyncio
    async def test_rollout_submit_restores_rollout_order_inside_group(self):
        gen = RecordingRewardOnlyAgent(parallel_generation_tasks=2)
        request = GroupedRolloutRequest(
            num_groups=1,
            rollouts_per_group=2,
            inference_interface=RecordingInference(slow_calls={0}),
            streaming=True,
            enforce_order=True,
            submission_granularity=RLRolloutGranularity.ROLLOUT,
        )

        groups = await collect_groups(gen, request, 1)

        assert [rollout.trajectory[0] for rollout in groups[0]] == [
            "prompt-0/rollout-0/call-0",
            "prompt-0/rollout-1/call-1",
        ]

    @pytest.mark.asyncio
    async def test_rollout_submit_filter_resubmits_same_ordered_slot(self):
        gen = RecordingRewardOnlyAgent(parallel_generation_tasks=2)
        request = GroupedRolloutRequest(
            num_groups=1,
            rollouts_per_group=2,
            inference_interface=RecordingInference(same_reward_prompt_ids={0}),
            streaming=True,
            enforce_order=True,
            filter_groups_with_same_reward=True,
            submission_granularity=RLRolloutGranularity.ROLLOUT,
        )

        groups = await collect_groups(gen, request, 1)

        assert group_prompt(groups[0]) == "prompt-1"
        assert groups[0].batch_id == 0
        assert groups[0].index_in_batch == 0
        assert [rollout.reward for rollout in groups[0]] == [0, 1]
