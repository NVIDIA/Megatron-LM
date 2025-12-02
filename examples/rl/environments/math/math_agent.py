# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import re
import traceback

from megatron.rl.agent.pass_at_evaluation_agent import PassAtEvaluationAgent
from megatron.rl.agent.reward_only_agent import RewardOnlyAgent

try:
    from math_verify import parse, verify
except ImportError:
    print(
        "math_verify is not installed. Install it using `pip install math-verify`. Continuing using exact match verification."
    )
    MATHVERIFY_AVAILABLE = False
else:
    print("math_verify is installed. Using math_verify to verify answers.")
    MATHVERIFY_AVAILABLE = True

assert (
    MATHVERIFY_AVAILABLE
), "math_verify is not installed but now required. Install it using `pip install math-verify` to continue."

NEGATIVE_REWARD = 0.0


class MathAgent(RewardOnlyAgent):
    def __init__(self, format_reward: float = 0.0, answer_format: str = "tagged", **kwargs):
        super().__init__(**kwargs)
        assert answer_format in ["tagged", "boxed"], "Invalid answer format"
        self.format_reward = format_reward
        self.answer_format = answer_format

    def compute_score(self, response: str, golden: dict, golden_key: str = "answer") -> float:
        """Take a response and a golden answer and return a score. Supports tagged or boxed answers.

        Uses the final answer in the response string to compute the score.
        """
        # Allow <answer> tags or \boxed{} tags (this is a bit of cheating in favor of deepseek distilled models I think)
        for pattern in [
            r'<answer>(.*?)</answer>',
            r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}",
        ]:
            match = re.finditer(pattern, response, re.DOTALL)
            matches = list(match)
            if matches:
                final_answer = matches[-1].group(1).strip()
                break
        else:
            # Did not format the answer correctly
            return NEGATIVE_REWARD

        try:
            parsed_answer = parse(final_answer)
        except ValueError as e:
            print("Failed to parse the answer.")
            traceback.print_stack()
            return NEGATIVE_REWARD

        correct_answer = verify(str(golden[golden_key]), parsed_answer)
        if correct_answer:
            return 1.0
        else:
            # Formatting is correct but the answer is incorrect
            return self.format_reward

    def make_prefix(self, problem_key: str = "problem", **kwargs) -> str:
        """Take a string math problem and return the prompt. Supports requesting tagged or boxed answers. Supports chat mode prompts."""
        if self.answer_format == "boxed":
            answer_format = "Please reason step by step and provide your answer between \\boxed{} tags, for example \\boxed{20\\sqrt{3}}."
        elif self.answer_format == "tagged":
            answer_format = "Please reason step by step and provide your answer between <answer> </answer> tags, for example <answer> 20\\sqrt{3} </answer>. Do not include an = sign."
        else:
            raise ValueError(f"Invalid answer format: {self.answer_format}")

        if self.chat_mode:
            prefix = f"""{kwargs[problem_key]}\n{answer_format}"""
        else:
            prefix = f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
    The question will be a word math problem. Show your work in <think> </think> tags. 
    {answer_format}
    User: {kwargs[problem_key]}
    Assistant: Let me solve this step by step.
    <think>"""
        return prefix
