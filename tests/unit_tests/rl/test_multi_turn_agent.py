# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
from pydantic import Field

from megatron.rl.agent.api import RolloutRequest, TokenRollout
from megatron.rl.agent.reward_only_agent import RewardOnlyAgent
from megatron.rl.inference import (
    InferenceResponse,
    LLMChatMessage,
    ReturnsRaw,
    ReturnsTokens,
)


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


class ScriptedInterface(ReturnsTokens, ReturnsRaw):
    """Token-returning inference stub that replays scripted responses."""

    scripted: list = Field(default_factory=list)
    seen_conversations: list = Field(default_factory=list)

    async def agenerate(self, request):
        self.seen_conversations.append(list(request.prompt))
        return self.scripted.pop(0)


class TwoTurnAgent(RewardOnlyAgent):
    """Ends the episode after one observation; records reward calls."""

    env_id: str = "test"
    max_turns: int = 3
    rewarded: list = Field(default_factory=list)

    async def get_prompt(self, validation):
        return "hello", {"problem_id": "p0"}

    async def get_observation(self, turn_idx, response, conversation, golden):
        if turn_idx >= 1:
            return None, True
        return f"obs{turn_idx}", False

    async def get_reward(self, response, golden, finish_reason):
        self.rewarded.append((response, finish_reason))
        return 1.5


class TestMultiTurnEpisode:

    @pytest.mark.asyncio
    async def test_multi_turn_episode(self):
        r0 = make_response([(0, 5)], prompt_length=3, total_len=7, content="a0")
        r1 = make_response([(0, 5)], prompt_length=6, total_len=11, content="a1")
        iface = ScriptedInterface(scripted=[r0, r1])
        agent = TwoTurnAgent()

        rollout = await agent.rollout(
            RolloutRequest(num_rollouts=1, inference_interface=iface)
        )

        assert isinstance(rollout, TokenRollout)
        # Turn 0 produced an observation; turn 1 reported done.
        assert len(rollout.trajectory) == 2
        assert rollout.reward == 1.5
        assert rollout.problem_id == "p0"

        # Conversation growth: turn-1 request = prompt + assistant reply + observation.
        first, second = iface.seen_conversations
        assert [m.role for m in first] == ["user"]
        assert [m.role for m in second] == ["user", "assistant", "user"]
        assert second[1].content == "a0"
        assert second[2].content == "obs0"

        # Default trajectory reward scores only the final response.
        assert agent.rewarded == [("a1", "stop")]

        # Per-turn generation masks cover exactly the generated regions.
        assert sum(rollout.generation_mask[0]) == 4
        assert sum(rollout.generation_mask[1]) == 5

        # Staleness fields keep the per-turn (engine-frame) nesting.
        assert rollout.policy_epoch == [[(0, 5)], [(0, 5)]]
        assert rollout.kv_cache_epoch == [[(0, 5)], [(0, 5)]]
        assert rollout.num_evictions == [0, 0]

    @pytest.mark.asyncio
    async def test_single_turn_default_unchanged(self):
        r0 = make_response([(0, 7)], prompt_length=2, total_len=6, content="only")
        iface = ScriptedInterface(scripted=[r0])

        class SingleTurnAgent(TwoTurnAgent):
            max_turns: int = 1

        agent = SingleTurnAgent()
        rollout = await agent.rollout(
            RolloutRequest(num_rollouts=1, inference_interface=iface)
        )

        # Exactly one generation; default hooks preserve single-turn behaviour.
        assert len(iface.seen_conversations) == 1
        assert len(rollout.trajectory) == 1
        assert rollout.policy_epoch == [[(0, 7)]]
        assert agent.rewarded == [("only", "stop")]
