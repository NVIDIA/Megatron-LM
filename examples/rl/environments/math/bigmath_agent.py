import random

import datasets

from .math_agent import MathAgent

raw_dataset = datasets.load_dataset("SynthLabsAI/Big-Math-RL-Verified", split="train")
TRAIN_SIZE = len(raw_dataset) - 1024
TEST_SIZE = 1024

assert len(raw_dataset) >= TRAIN_SIZE + TEST_SIZE
train_dataset = raw_dataset.select(range(TRAIN_SIZE))
test_dataset = raw_dataset.select(range(TRAIN_SIZE, TRAIN_SIZE + TEST_SIZE))


class BigMathAgent(MathAgent):
    env_id: str = "bigmath"

    def get_dataset(self, validation: bool = False):
        return train_dataset if not validation else test_dataset

    async def evaluation_prompts(
        self, num_prompts: int, validation: bool = False
    ) -> list[tuple[str, dict]]:
        dataset = self.get_dataset(validation)
        return [
            (self.make_prefix(**golden), golden)
            for golden in [dataset[i] for i in range(num_prompts)]
        ]

    async def get_prompt(self, validation=False) -> tuple[str, dict]:
        dataset = self.get_dataset(validation)
        golden = dataset[random.randrange(len(dataset))]
        prompt = self.make_prefix(**golden)
        return prompt, golden

    async def get_reward(self, response, golden: dict) -> float:
        return self.compute_score(response, golden, golden_key="answer")
