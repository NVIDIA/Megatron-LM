# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
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
                trajectory=[r.raw_text for r in responses],
                reward=reward,
                env_id=self.env_id,
                policy_epoch=[r.policy_epoch for r in responses],
                kv_cache_epoch=[r.kv_cache_epoch for r in responses],
                num_evictions=[r.num_evictions for r in responses],
            )

        return GroupRolloutParams(run_episode=run_episode, build_rollout=build_rollout)


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


def make_response(epochs, prompt_length, total_len, content="resp", finish_reason="stop"):
    return InferenceResponse(
        response=LLMChatMessage(role="assistant", content=content),
        raw_text=content,
        token_ids=list(range(total_len)),
        prompt_length=prompt_length,
        logprobs=[0.0] * (total_len - prompt_length),
        finish_reason=finish_reason,
        policy_epoch=epochs,
        kv_cache_epoch=epochs,
        num_evictions=0,
    )


# Conversation length -> response spec: length 1 is the first turn (the bare prompt), length 3
# the second (assistant reply + observation appended).
TWO_TURN_SCRIPT = {
    1: dict(epochs=[(0, 5)], prompt_length=3, total_len=7, content="a0"),
    3: dict(epochs=[(0, 5)], prompt_length=6, total_len=11, content="a1"),
}

# Both two-turn termination modes (env-signaled done, max_turns exhausted) must produce this
# identical episode; only the env-consultation trace (observation_turns) differs per case.
TWO_TURN_EXPECTED = dict(
    seen_roles=[["user"], ["user", "assistant", "user"]],
    reward_conv=[("user", "hello"), ("assistant", "a0"), ("user", "obs0"), ("assistant", "a1")],
    rewarded=[("a1", "stop")],
    genmask_sums=[4, 5],
    policy_epoch=[[(0, 5)], [(0, 5)]],
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
                {1: dict(epochs=[(0, 7)], prompt_length=2, total_len=6, content="only")},
                dict(
                    seen_roles=[["user"]],
                    reward_conv=[("user", "hello"), ("assistant", "only")],
                    rewarded=[("only", "stop")],
                    genmask_sums=[4],
                    policy_epoch=[[(0, 7)]],
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
                async for group in agent.get_grouped_rollouts(
                    GroupedRolloutRequest(
                        num_groups=1, rollouts_per_group=1, inference_interface=iface
                    )
                ):
                    groups.append(group)

            # Bounded so a wedged pipeline fails fast instead of hanging.
            await asyncio.wait_for(_drain(), timeout=5.0)
            (group,) = groups
            rollouts = group.rollouts
        (rollout,) = rollouts

        assert isinstance(rollout, TokenRollout)
        assert rollout.reward == 1.5
        assert rollout.problem_id == "p0"
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
        # Per-turn (engine-frame) staleness nesting is preserved.
        assert rollout.policy_epoch == expected["policy_epoch"]
        assert rollout.kv_cache_epoch == expected["policy_epoch"]
        # get_observation is consulted only when another generation is still possible -- never on
        # the final allowed turn.
        assert agent.observation_turns == expected["observation_turns"]
        # get_trajectory_reward sees the full dialogue, ending on the final reply exactly once.
        assert [(m.role, m.content) for m in agent.reward_conversation] == expected["reward_conv"]
