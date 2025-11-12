# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
from typing import Any

import numpy as np
from tqdm.asyncio import tqdm

from ..inference import (
    ChatInferenceInterface,
    ChatInferenceResponse,
    InferenceResponse,
    LLMChatMessage,
    ReturnsRaw,
    ReturnsTokens,
)
from .api import (
    EvaluationAgent,
    EvaluationRequest,
    EvaluationResponse,
    GroupedRolloutGenerator,
    GroupedRolloutRequest,
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

    def get_dataset(self, validation: bool = False):
        """Return validation or train dataset."""
        raise NotImplementedError("Derived class must implement get_dataset.")

    async def get_reward(self, response: str, golden: Any) -> float:
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

    async def rollout_from_response(
        self, request: RolloutRequest, response: InferenceResponse, golden: Any
    ) -> Rollout:
        assert isinstance(
            request.inference_interface, ReturnsRaw
        ), "InferenceInterface must support raw_text return to provide rollouts."
        raw_text = response.raw_text

        response_text = (
            response.response.content
            if isinstance(response, ChatInferenceResponse)
            else response.response
        )

        if isinstance(request.inference_interface, ReturnsTokens):
            logprobs = response.logprobs
            generation_mask = [
                True if (x >= response.prompt_length) else False
                for x in range(len(response.token_ids))
            ]
            rollout = TokenRollout(
                trajectory=response.token_ids,
                reward=await self.get_reward(response_text, golden),
                logprobs=logprobs,
                generation_mask=generation_mask,
                env_id=self.env_id,
                problem_id=golden['problem_id'] if 'problem_id' in golden else None,
            )
        else:
            rollout = Rollout(
                trajectory=raw_text,
                reward=await self.get_reward(response_text, golden),
                env_id=self.env_id,
                problem_id=golden['problem_id'] if 'problem_id' in golden else None,
            )

        return rollout

    async def rollout(self, request: RolloutRequest) -> Rollout:

        prompt, golden = await self.get_prompt(validation=request.validation)

        inference_request = request.inference_interface.prepare_request(
            [prompt], request.generation_args
        )

        responses = await request.inference_interface.agenerate(inference_request)
        assert (
            len(responses) == 1
        ), "get_reward_rollouts only requested a single response but got multiple responses"
        response = responses[0]

        return await self.rollout_from_response(request, response, golden)

    async def group_rollout(self, request: GroupedRolloutRequest) -> list[Rollout]:

        prompt, golden = await self.get_prompt(validation=request.validation)

        inference_request = request.inference_interface.prepare_request(
            [prompt], request.generation_args
        )
        inference_request.n = request.rollouts_per_group

        groups = await request.inference_interface.agenerate(inference_request)
        assert (
            len(groups) == 1
        ), "get_grouped_rollouts only requested a single group but got multiple groups"
        responses = groups[0].responses

        rollouts = await asyncio.gather(
            *[self.rollout_from_response(request, response, golden) for response in responses]
        )

        return rollouts

    async def _evaluation(
        self, prompt: str, golden: Any, request: EvaluationRequest
    ) -> RewardOnlyEvaluationResponse:

        inference_request = request.inference_interface.prepare_request(
            [prompt], request.generation_args
        )

        responses = await request.inference_interface.agenerate(inference_request)
        assert (
            len(responses) == 1
        ), "evaluation only requested a single response but got multiple responses"
        response = responses[0]

        response_text = (
            response.response.content
            if isinstance(response, ChatInferenceResponse)
            else response.response
        )

        result = RewardEvaluationResult(
            env_id=self.env_id,
            prompt=[prompt] if isinstance(prompt, LLMChatMessage) else prompt,
            response=response.response,
            reward=await self.get_reward(response_text, golden),
            problem_id=golden['problem_id'] if 'problem_id' in golden else None,
        )

        return RewardOnlyEvaluationResponse(results=[result], env_id=self.env_id)

    async def run_evaluation(self, request: EvaluationRequest):

        if isinstance(request.inference_interface, ChatInferenceInterface):
            self.chat_mode = True
        else:
            self.chat_mode = False

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
