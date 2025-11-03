# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from ..__init__ import GenericGenerationArgs
from ..inference import ChatInferenceResponse, LLMChatMessage
from .api import EvaluationAgent, EvaluationRequest, EvaluationResponse, RewardEvaluationResult


def pass_at_k(n_samples: int, n_correct: int, k: int) -> float:
    """Lower variance estimator of pass@k."""
    assert n_samples >= 0, "n_samples should be non-negative"
    assert n_correct >= 0, "n_correct should be non-negative"
    assert k <= n_samples, "k should be less than or equal to n_samples"

    if n_samples - n_correct < k:
        return 1.0

    return 1.0 - np.prod(1.0 - k / np.arange(n_samples - n_correct + 1, n_samples + 1))


class PassAtEvaluationResult(RewardEvaluationResult):
    pass_at: dict[int, float]
    response: list[str] | list[LLMChatMessage]
    reward: list[float]
    greedy_response: str | LLMChatMessage
    greedy_reward: float


class PassAtEvaluationResponse(EvaluationResponse[PassAtEvaluationResult]):
    type_name: str = 'PassAtEvaluationResponse'

    def metrics(self):
        metrics = {}
        if self.results:
            pass_at_k_keys = self.results[0].pass_at.keys()
            for k in pass_at_k_keys:
                metrics[f'pass_at_{k}'] = [el.pass_at[k] for el in self.results]
            metrics['greedy_reward'] = [el.greedy_reward for el in self.results]
        return metrics


class PassAtEvaluationAgent(EvaluationAgent, ABC):

    def __init__(self, max_k=32, **kwargs):
        super().__init__(**kwargs)
        self.max_k = max_k

    @abstractmethod
    async def _evaluation(
        self, prompt: Any, golden: dict | None, request: EvaluationRequest
    ) -> EvaluationResponse[RewardEvaluationResult]: ...

    async def evaluation(
        self, prompt: Any, golden: dict | None, request: EvaluationRequest
    ) -> PassAtEvaluationResponse:

        evaluations = [self._evaluation(prompt, golden, request) for _ in range(self.max_k)]
        responses = await asyncio.gather(*evaluations)

        rewards = [
            result.reward for result in sum([response.results for response in responses], [])
        ]
        response_texts = [
            result.response for result in sum([response.results for response in responses], [])
        ]

        # Count number of passing solutions (reward == 1.0)
        pass_count = sum(1 for reward in rewards if reward == 1.0)
        total_count = len(rewards)

        # Calculate pass@N for different N values
        pass_at = {
            k: pass_at_k(total_count, pass_count, k)
            for k in [1, self.max_k]  # You can adjust these values as needed
        }

        greedy_generation_args = request.generation_args.add(
            GenericGenerationArgs(top_k=1, temperature=0.0, top_p=0.0)
        )
        greedy_request = request.model_copy(update={'generation_args': greedy_generation_args})
        greedy_responses = await self._evaluation(prompt, golden, greedy_request)
        assert (
            len(greedy_responses.results) == 1
        ), "Evaluation only requested a single response but got multiple responses"
        greedy_response = greedy_responses.results[0]
        result = PassAtEvaluationResult(
            prompt=greedy_response.prompt,
            problem_id=golden['problem_id'] if golden and 'problem_id' in golden else None,
            pass_at=pass_at,
            response=response_texts,
            reward=rewards,
            greedy_response=greedy_response.response,
            greedy_reward=greedy_response.reward,
        )
        return PassAtEvaluationResponse(results=[result], env_id=self.env_id)
