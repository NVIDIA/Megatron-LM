import random

import datasets

from .math_agent import MathAgent

raw_dataset = datasets.load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", split="train")
TRAIN_SIZE = 17917 - 1024
TEST_SIZE = 1024

train_dataset = raw_dataset.select(range(TRAIN_SIZE))
test_dataset = raw_dataset.select(range(TRAIN_SIZE, TRAIN_SIZE + TEST_SIZE))


class DAPOAgent(MathAgent):
    env_id: str = "dapo"

    def reformat_datum(self, datum: dict) -> dict:
        return {
            "problem": datum['prompt'][0]['content']
            .replace(
                'The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n',
                '',
            )
            .replace('\nRemember to put your answer on its own line after "Answer:".', ''),
            "answer": datum["reward_model"]["ground_truth"],
            "problem_id": datum["extra_info"]["index"],
        }

    def get_dataset(self, validation: bool = False):
        return train_dataset if not validation else test_dataset

    async def evaluation_prompts(
        self, num_prompts: int, validation: bool = False
    ) -> list[tuple[str, dict]]:
        dataset = self.get_dataset(validation)
        prompts = []
        for i, golden in [(i, dataset[i]) for i in range(num_prompts)]:
            golden = self.reformat_datum(golden)
            prompts.append((self.make_prefix(**golden), golden))
        return prompts

    async def get_prompt(self, validation=False) -> tuple[str, dict]:
        dataset = self.get_dataset(validation)
        golden = dataset[random.randrange(len(dataset))]
        golden = self.reformat_datum(golden)
        prompt = self.make_prefix(**golden)
        return prompt, golden

    async def get_reward(self, response, golden: dict) -> float:
        return self.compute_score(response, golden, golden_key="answer")
