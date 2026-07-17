# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
import functools
from typing import Any

import numpy as np
from tqdm.asyncio import tqdm

from ..inference import (
    InferenceRequest,
    InferenceResponse,
    LLMChatMessage,
    ReturnsRaw,
    ReturnsTokens,
)
from .api import (
    EpisodeResult,
    EvaluationAgent,
    EvaluationRequest,
    EvaluationResponse,
    GroupedRolloutGenerator,
    GroupedRolloutRequest,
    GroupRolloutParams,
    RewardEvaluationResult,
    Rollout,
    RolloutGenerator,
    RolloutRequest,
    TokenRollout,
)
from .pass_at_evaluation_agent import PassAtEvaluationAgent


class RewardOnlyEvaluationResponse(EvaluationResponse[RewardEvaluationResult]):
    type_name: str = 'RewardOnlyEvaluationResponse'

    def metrics(self):
        return {'reward': [el.reward for el in self.results]}


class RewardOnlyAgent(RolloutGenerator, GroupedRolloutGenerator, PassAtEvaluationAgent):
    """Agent that returns rollouts generated via default inference with a fixed reward function."""

    env_id: str | None = None
    max_turns: int = 1

    def get_dataset(self, validation: bool = False):
        """Return validation or train dataset."""
        raise NotImplementedError("Derived class must implement get_dataset.")

    async def get_reward(
        self, response: str, golden: Any, finish_reason: str
    ) -> float:
        """Given the LLM response and the golden data, provide a reward."""
        raise NotImplementedError("Derived class must implement get_reward")

    async def get_prompt(self, validation: bool) -> tuple[str, Any]:
        """Return a tuple with the prompt string and the golden data."""
        raise NotImplementedError("Derived class must implement get_prompt")

    async def evaluation_prompts(
        self, num_prompts: int, validation: bool = False
    ) -> list[tuple[str, Any]]:
        """Get evaluation prompts for the agent. This method should be overridden by subclasses."""
        raise NotImplementedError

    def _get_rank_subset(
        self, prompts: list[tuple[str, Any]], num_prompts: int, rank: int, world_size: int
    ) -> list[tuple[str, Any]]:
        """Helper method to get the subset of prompts for a given rank.

        Args:
            prompts: List of all prompts
            num_prompts: Total number of prompts to use
            rank: Current process rank
            world_size: Total number of processes

        Returns:
            Subset of prompts for the current rank
        """
        # Take first num_prompts from all prompts
        prompts = prompts[:num_prompts]

        # Split prompts into chunks for each rank
        chunk_size = (len(prompts) + world_size - 1) // world_size
        start_idx = rank * chunk_size
        end_idx = min(start_idx + chunk_size, len(prompts))

        return prompts[start_idx:end_idx]

    async def get_observation(
        self,
        turn_idx: int,
        response: InferenceResponse,
        conversation: list[LLMChatMessage],
        golden: Any,
    ) -> tuple[str | None, bool]:
        """Return (observation, done) after a generation turn. Skipped on the last turn.

        Override to implement multi-turn interactions. Must not mutate `conversation` or `golden`!

        Args:
            turn_idx: 0-based index of the turn that just completed.
            response: The inference response for this turn.
            conversation: Message history before this turn's response was appended.
            golden: Ground-truth / task data for reward computation.

        Returns:
            (observation, done): If done is True the episode ends; observation is ignored.
            If done is False, observation is a non-empty string that becomes the next user message.
        """
        return None, True

    async def get_trajectory_reward(
        self,
        responses: list[InferenceResponse],
        conversation: list[LLMChatMessage],
        golden: Any,
    ) -> float:
        """Compute a scalar reward for the full trajectory.

        Override for trajectory-level or per-turn accumulated rewards.
        """
        return await self.get_reward(
            responses[-1].response.content, golden, responses[-1].finish_reason
        )

    async def _run_episode(
        self,
        request: RolloutRequest | GroupedRolloutRequest,
        *,
        prompt: str | list[LLMChatMessage],
        golden: Any,
    ) -> EpisodeResult:
        """Run one (possibly multi-turn) episode over the group's prompt.

        Every turn takes the same path: prepare_request() on the conversation so far, then
        get_rollout_response().
        get_observation() is consulted only while another generation is still possible;
        on continue, the reply and observation are appended.

        Runs inside the infer stage, holding one submission slot for the whole episode.
        """
        conversation = prompt
        responses: list[InferenceResponse] = []

        for turn_idx in range(self.max_turns):
            turn_request = request.inference_interface.prepare_request(
                conversation, request.generation_args
            )
            # Adopt the request's prompt as the conversation: turn 0 may start from a bare
            # string, which prepare_request normalizes into a single user message.
            conversation = list(turn_request.prompt)
            responses.append(await self.get_rollout_response(request, turn_request))
            if turn_idx == self.max_turns - 1:
                break

            observation, done = await self.get_observation(
                turn_idx, responses[-1], conversation, golden
            )
            if done or observation is None:
                break
            assert observation, (
                "get_observation must return a non-empty observation when continuing (done=False)."
            )
            conversation = conversation + [
                responses[-1].response,
                LLMChatMessage(role="user", content=observation),
            ]

        # The loop appends a reply only when continuing, so the final turn's reply is not in
        # `conversation` yet; append it once so get_trajectory_reward sees the full dialogue.
        return EpisodeResult(
            responses=responses, conversation=conversation + [responses[-1].response]
        )

    async def _rollout_from_episode(
        self,
        request: RolloutRequest | GroupedRolloutRequest,
        episode: EpisodeResult,
        golden: Any,
    ) -> Rollout | TokenRollout:
        """Package a completed episode into a single rollout, one trajectory entry per turn.

        Calls `get_trajectory_reward()` once over all of the episode's responses.
        """
        responses = episode.responses
        reward = await self.get_trajectory_reward(responses, episode.conversation, golden)
        problem_id = golden['problem_id'] if 'problem_id' in golden else None

        if isinstance(request.inference_interface, ReturnsTokens):
            return TokenRollout(
                trajectory=[r.token_ids for r in responses],
                reward=reward,
                logprobs=[r.logprobs for r in responses],
                generation_mask=[
                    [x >= r.prompt_length for x in range(len(r.token_ids))]
                    for r in responses
                ],
                env_id=self.env_id,
                problem_id=problem_id,
                policy_epoch=[r.policy_epoch for r in responses],
                kv_cache_epoch=[r.kv_cache_epoch for r in responses],
                num_evictions=[r.num_evictions for r in responses],
            )
        else:
            return Rollout(
                trajectory=[r.raw_text for r in responses],
                reward=reward,
                env_id=self.env_id,
                problem_id=problem_id,
                policy_epoch=[r.policy_epoch for r in responses],
                kv_cache_epoch=[r.kv_cache_epoch for r in responses],
                num_evictions=[r.num_evictions for r in responses],
            )

    async def get_rollout_response(
        self,
        request: RolloutRequest | GroupedRolloutRequest | EvaluationRequest,
        inference_request: InferenceRequest,
    ) -> InferenceResponse:
        return await request.inference_interface.agenerate(inference_request)

    async def get_reward_rollouts(self, request: RolloutRequest) -> list[Rollout]:
        assert isinstance(
            request.inference_interface, ReturnsRaw
        ), "InferenceInterface must support raw_text return to provide rollouts."

        async def _single_rollout() -> Rollout:
            params = await self.prepare_group_rollout(request)
            return await params.build_rollout(await params.run_episode())

        return list(
            await asyncio.gather(*[_single_rollout() for _ in range(request.num_rollouts)])
        )

    async def prepare_group_rollout(
        self,
        request: GroupedRolloutRequest,
    ) -> GroupRolloutParams:

        prompt, golden = await self.get_prompt(validation=request.validation)

        # Every rollout runs as a (possibly multi-turn) episode over the group's shared prompt.
        return GroupRolloutParams(
            run_episode=functools.partial(self._run_episode, request, prompt=prompt, golden=golden),
            build_rollout=functools.partial(self._rollout_from_episode, request, golden=golden),
        )

    async def _evaluation(
        self, prompt: str, golden: Any, request: EvaluationRequest
    ) -> RewardOnlyEvaluationResponse:

        inference_request = request.inference_interface.prepare_request(
            prompt, request.generation_args
        )

        response = await self.get_rollout_response(request, inference_request)
        response_text = response.response.content

        result = RewardEvaluationResult(
            env_id=self.env_id,
            prompt=[prompt] if isinstance(prompt, LLMChatMessage) else prompt,
            response=response.response,
            reward=await self.get_reward(response_text, golden, response.finish_reason),
            problem_id=golden['problem_id'] if 'problem_id' in golden else None,
        )

        return RewardOnlyEvaluationResponse(results=[result], env_id=self.env_id)

    async def run_evaluation(self, request: EvaluationRequest):

        # Get all prompts first
        all_prompts = list(
            await self.evaluation_prompts(
                num_prompts=request.num_prompts, validation=request.validation
            )
        )

        # Then get this rank's subset if needed
        if request.rank_info is not None:
            prompts_to_evaluate = self._get_rank_subset(
                all_prompts, request.num_prompts, request.rank_info[0], request.rank_info[1]
            )
        else:
            prompts_to_evaluate = all_prompts

        results = await tqdm.gather(
            *[self.evaluation(p, g, request) for p, g in prompts_to_evaluate],
            desc="Evaluating prompts..",
        )
        return type(results[0])(
            results=sum([result.results for result in results], []), env_id=self.env_id
        )
