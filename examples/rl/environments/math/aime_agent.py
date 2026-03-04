import datasets

from .math_agent import MathAgent

raw_dataset = datasets.load_dataset("Maxwell-Jia/AIME_2024", split="train")

TRAIN_SIZE = 0
TEST_SIZE = len(raw_dataset) - TRAIN_SIZE

assert len(raw_dataset) >= TRAIN_SIZE + TEST_SIZE
train_dataset = raw_dataset.select(range(TRAIN_SIZE))
test_dataset = raw_dataset.select(range(TRAIN_SIZE, TRAIN_SIZE + TEST_SIZE))


class AIMEAgent(MathAgent):
    env_id: str = "aime"

    def get_dataset(self, validation: bool = False):
        assert validation, "AIME prompts are not available for training."
        return test_dataset

    async def evaluation_prompts(
        self, num_prompts: int, validation: bool = False
    ) -> list[tuple[str, dict]]:
        dataset = self.get_dataset(validation)
        return [
            (
                self.make_prefix(**golden, problem_key="Problem"),
                {**golden, "problem_id": golden["ID"]},
            )
            for _, golden in enumerate([dataset[i] for i in range(num_prompts) if i < len(dataset)])
        ]

    async def get_prompt(self, validation=False) -> tuple[str, dict]:
        print("WARNING: AIME prompts should not be used for training.")
        validation = True
        dataset = train_dataset if not validation else test_dataset
        problem_id = 0
        golden = dataset[problem_id]
        golden = {**golden, "problem_id": golden["ID"]}
        prompt = self.make_prefix(**golden, problem_key="Problem")
        return prompt, golden

    async def get_reward(self, response, golden: dict) -> float:
        return self.compute_score(response, golden, golden_key="Answer")
