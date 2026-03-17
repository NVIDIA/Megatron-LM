# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import re
import traceback

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

class MathAgent(RewardOnlyAgent):
    def __init__(self,
        format_reward: float = 0.0,
        answer_format: str = "tagged",
        negative_reward: float = 0.0,
        partial_end_reward: float = 0.0,
        **kwargs):
        """
        Args:
            format_reward (float): Reward given when the answer is in the expected format,
                even if the answer is incorrect or is missing the end-of-text token.
            answer_format (str): Which answer format is expected: "tagged" for <answer> tags,
                or "boxed" for \boxed{} LaTeX formatting.
            negative_reward (float): Reward assigned for a clearly incorrect or unparseable answer.
            partial_end_reward (float): Reward when the answer is correct but an expected end token is not matched exactly.
            **kwargs: Additional arguments for the base RewardOnlyAgent.
        """
        super().__init__(**kwargs)

        assert answer_format in ["tagged", "boxed"], "Invalid answer format"

        self.format_reward = format_reward
        self.answer_format = answer_format
        self.negative_reward = negative_reward
        self.partial_end_reward = partial_end_reward

    def compute_score(self, response: str, golden: dict, golden_key: str = "answer") -> float:
        """Take a response and a golden answer and return a score. Supports tagged or boxed answers.

        Uses the final answer in the response string to compute the score.
        """
        # Allow <answer> tags or \boxed{} tags (this is a bit of cheating in favor of deepseek distilled models I think)
        matched_format = None
        end_tokens = ["<|end_of_text|>", "<|endoftext|>", "</s>", "<|eot_id|>", "<|im_end|>"]

        # Only an answer immediately followed by a known end token yields 1.0 reward.
        answer_tag_pattern = r'<answer>(.*?)</answer>'
        answer_tag_match = list(re.finditer(answer_tag_pattern, response, re.DOTALL))
        if answer_tag_match:
            # Only consider the last occurrence
            last_match = answer_tag_match[-1]
            final_answer = last_match.group(1).strip()
            after = response[last_match.end():].lstrip()  # strip whitespace between </answer> and token

            try:
                parsed_answer = parse(final_answer)
            except ValueError as e:
                print("Failed to parse the answer.")
                traceback.print_stack()
                return self.negative_reward

            correct_answer = verify(str(golden[golden_key]), parsed_answer)
            if correct_answer:
                # Accept either <|end_of_text|> or <|endoftext|> as valid terminators, for flexibility.
                for token in end_tokens:
                    if after.startswith(token):
                        return 1.0
                # If the end token is present later (extra text before it), give partial credit.
                for token in end_tokens:
                    if token in after:
                        return self.partial_end_reward
                # If a correct answer but missing immediate end, give format reward (not NEGATIVE_REWARD).
                return self.format_reward
            else:
                # Incorrect answer, regardless of format/end-of-text
                return self.format_reward
        else:
            # Fallback: check boxed answer format for diagnostic/format reward as before
            boxed_pattern = r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}"
            boxed_match = list(re.finditer(boxed_pattern, response, re.DOTALL))
            if boxed_match:
                last_match = boxed_match[-1]
                final_answer = last_match.group(1).strip()
                after = response[last_match.end():].lstrip()
                try:
                    parsed_answer = parse(final_answer)
                except ValueError as e:
                    print("Failed to parse the answer.")
                    traceback.print_stack()
                    return self.negative_reward

                correct_answer = verify(str(golden[golden_key]), parsed_answer)
                if correct_answer:
                    for token in end_tokens:
                        if after.startswith(token):
                            return 1.0
                    for token in end_tokens:
                        if token in after:
                            return self.partial_end_reward
                    return self.format_reward
                else:
                    # Formatting is correct but the answer is incorrect
                    return self.format_reward
            else:
                # Did not format the answer correctly
                return self.negative_reward

    def make_prefix(self, problem_key: str = "problem", **kwargs) -> str:
        """Take a string math problem and return the prompt. Supports requesting tagged or boxed answers. Supports chat mode prompts."""
        if self.answer_format == "boxed":
            answer_format = "Please reason step by step and provide your answer between \\boxed{} tags, for example \\boxed{20\\sqrt{3}}."
        elif self.answer_format == "tagged":
            answer_format = "Please reason step by step and provide your answer between <answer> </answer> tags, for example <answer> 20\\sqrt{3} </answer>. Do not include an = sign."
        else:
            raise ValueError(f"Invalid answer format: {self.answer_format}")

        prefix = f"""{kwargs[problem_key]}\n{answer_format}"""

        return prefix
