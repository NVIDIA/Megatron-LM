# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import random
from typing import Any, Iterable

from megatron.rl.agent.huggingface_dataset_agent import HFDatasetAgent
from megatron.rl.agent.reward_only_agent import RewardOnlyAgent

from .countdown import compute_score


class CountdownAgent(RewardOnlyAgent, HFDatasetAgent):
    env_id: str = "countdown"

    def make_prefix(self, target, nums) -> str:
        prefix = f"""Using the numbers {nums}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. 
        Return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>. Do not include an = sign."""
        return prefix

    def get_dataset(self, validation: bool = False):
        TRAIN_SIZE = 327680
        TEST_SIZE = 1024

        assert len(self.dataset) > TRAIN_SIZE + TEST_SIZE
        train_dataset = self.dataset.select(range(TRAIN_SIZE))
        test_dataset = self.dataset.select(range(TRAIN_SIZE, TRAIN_SIZE + TEST_SIZE))
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
