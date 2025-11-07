# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import random

import datasets

from .math_agent import MathAgent

raw_dataset = datasets.load_dataset("openai/gsm8k", "main")

TRAIN_SIZE = 7473
TEST_SIZE = 1319

train_dataset = raw_dataset["train"]
test_dataset = raw_dataset["test"]

assert (
    len(train_dataset) == TRAIN_SIZE
), f"GSM8K loading error: expected {TRAIN_SIZE} samples but got {len(train_dataset)}"
assert (
    len(test_dataset) == TEST_SIZE
), f"GSM8K loading error: expected {TEST_SIZE} samples but got {len(test_dataset)}"


class GSM8KAgent(MathAgent):
    def __init__(self, answer_format: str = "boxed", format_reward: float = 0.0, **kwargs):
        super().__init__(format_reward=format_reward, answer_format=answer_format, **kwargs)
        self.env_id: str = "gsm8k"

    def reformat_datum(self, datum: dict) -> dict:
        return {
            "problem": datum["question"],
            "answer": datum["answer"],
            "numeric_answer": datum["answer"].split("#### ")[-1],
        }

    def get_dataset(self, validation: bool = False):
        return train_dataset if not validation else test_dataset

    async def evaluation_prompts(
        self, num_prompts: int, validation: bool = False
    ) -> list[tuple[str, dict]]:
        dataset = self.get_dataset(validation)
        return [
            (self.make_prefix(**golden), golden)
            for golden in [self.reformat_datum(dataset[i]) for i in range(num_prompts)]
        ]

    async def get_prompt(self, validation=False) -> tuple[str, dict]:
        dataset = self.get_dataset(validation)
        golden = dataset[random.randrange(len(dataset))]
        golden = self.reformat_datum(golden)
        prompt = self.make_prefix(**golden)
        return prompt, golden

    async def get_reward(self, response, golden: dict) -> float:
        return self.compute_score(response, golden, golden_key="numeric_answer")


# pytest
