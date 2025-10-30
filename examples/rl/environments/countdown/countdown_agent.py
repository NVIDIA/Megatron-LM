# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import random
from typing import Any, Iterable

from megatron.rl.agent.pass_at_evaluation_agent import PassAtEvaluationAgent
from megatron.rl.agent.reward_only_agent import RewardOnlyAgent

from .countdown import compute_score, test_dataset, train_dataset


class CountdownAgent(RewardOnlyAgent):
    env_id: str = "countdown"

    def make_prefix(self, target, nums) -> str:
        if self.chat_mode:
            prefix = f"""Using the numbers {nums}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. 
        Return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>. Do not include an = sign."""
        else:
            prefix = f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
        User: Using the numbers {nums}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. 
        And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>. Do not include an = sign.
        Assistant: Let me solve this step by step.
        <think>"""
        return prefix

    def get_dataset(self, validation: bool = False):
        return train_dataset if not validation else test_dataset

    async def evaluation_prompts(
        self, num_prompts: int, validation: bool = False
    ) -> Iterable[tuple[str, Any]]:
        dataset = self.get_dataset(validation)
        return [
            (self.make_prefix(**golden), golden)
            for golden in [dataset[i] for i in range(num_prompts)]
        ]

    async def get_prompt(self, validation=False) -> tuple[str, dict]:
        dataset = self.get_dataset(validation)
        golden = dataset[random.randrange(len(dataset))]
        return self.make_prefix(**golden), golden

    async def get_reward(self, response, golden: dict) -> float:
        return compute_score(response, golden)
