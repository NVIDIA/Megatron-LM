# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import sys
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError


def _mock_math_verify():
    """Patch math_verify so MathAgent can be imported without the real package."""
    mock_module = MagicMock()
    mock_module.parse = lambda x: x
    mock_module.verify = lambda golden, parsed: str(golden).strip() == str(parsed).strip()
    return mock_module


# Ensure math_verify mock is in place before any MathAgent import at module level.
sys.modules.setdefault("math_verify", _mock_math_verify())

from examples.rl.environments.countdown.countdown import compute_score as countdown_compute_score
from examples.rl.environments.math.math_agent import MathAgent
from megatron.rl.agent.pass_at_evaluation_agent import PassAtEvaluationResult
from megatron.rl.inference import LLMChatMessage


def _make_math_agent(
    answer_format="tagged", format_reward=0.05, negative_reward=-0.5, partial_end_reward=0.1
):
    return MathAgent(
        answer_format=answer_format,
        format_reward=format_reward,
        negative_reward=negative_reward,
        partial_end_reward=partial_end_reward,
    )


class TestRLAgentRewards:
    """Test reward computation for all RL environment agents.

    Covers MathAgent.compute_score (used by AIME, BigMath, DAPO, GSM8K, OpenMath)
    and CountdownAgent / countdown.compute_score.
    """

    @pytest.mark.parametrize(
        "answer_format, correct, wrong",
        [
            pytest.param("tagged", "<answer>42</answer>", "<answer>99</answer>", id="tagged"),
            pytest.param("boxed", r"\boxed{42}", r"\boxed{99}", id="boxed"),
        ],
    )
    @pytest.mark.parametrize(
        "response_fn, finish_reason, expected_reward",
        [
            # Correct answer, stopped cleanly, nothing trailing.
            pytest.param(lambda c, w: c, "stop", 1.0, id="correct-stop-clean"),
            # Correct answer, hit token limit, nothing trailing.
            pytest.param(lambda c, w: c, "length", 0.05, id="correct-length-clean"),
            # Correct answer, stopped cleanly, but trailing text.
            pytest.param(lambda c, w: c + " extra", "stop", 0.1, id="correct-stop-trailing"),
            # Wrong answer, correct format.
            pytest.param(lambda c, w: w, "stop", 0.05, id="wrong-answer"),
            # No answer tag at all.
            pytest.param(lambda c, w: "I think 42", "stop", -0.5, id="no-tag"),
        ],
    )
    def test_math_agent_compute_score(
        self, answer_format, correct, wrong, response_fn, finish_reason, expected_reward
    ):
        agent = _make_math_agent(answer_format=answer_format)
        response = response_fn(correct, wrong)
        score = agent.compute_score(response, {"answer": "42"}, finish_reason=finish_reason)
        assert score == pytest.approx(expected_reward), (
            f"compute_score({response!r}, finish_reason={finish_reason!r}) "
            f"= {score}, expected {expected_reward}"
        )

    @pytest.mark.parametrize(
        "response, ground_truth, expected_reward",
        [
            # Correct equation
            pytest.param(
                "<answer>(10 + 5) * 2</answer>",
                {"target": 30, "nums": [10, 5, 2]},
                1.0,
                id="countdown-correct",
            ),
            # Wrong result
            pytest.param(
                "<answer>10 + 5 + 2</answer>",
                {"target": 30, "nums": [10, 5, 2]},
                0.1,
                id="countdown-wrong-result",
            ),
            # No answer tag
            pytest.param(
                "I have no idea", {"target": 30, "nums": [10, 5, 2]}, 0, id="countdown-no-tag"
            ),
            # Uses a number not in the set
            pytest.param(
                "<answer>30 * 1</answer>",
                {"target": 30, "nums": [10, 5, 2]},
                0.1,
                id="countdown-invalid-numbers",
            ),
        ],
    )
    def test_countdown_compute_score(self, response, ground_truth, expected_reward):
        score = countdown_compute_score(response, ground_truth)
        assert score == pytest.approx(expected_reward)


def test_pass_at_result_accepts_mixed_responses():
    """Aggregated pass@k responses mix assistant messages with '' placeholders
    for dropped/over-context rollouts; the result type validates element-wise."""
    msg = LLMChatMessage(role='assistant', content='the answer is 4')
    result = PassAtEvaluationResult(
        prompt='2+2?',
        response=[msg, '', 'plain-string response'],
        reward=[1.0, 0.0, 0.5],
        pass_at={1: 0.5},
        greedy_response=msg,
        greedy_reward=1.0,
    )
    # Assert that the plain-string responses are not type-cast.
    assert isinstance(result.response[0], LLMChatMessage)
    assert result.response[0].content == 'the answer is 4'
    assert result.response[1] == ''
    assert result.response[2] == 'plain-string response'

    # Element-wise validation: invalid elements still fail.
    with pytest.raises(ValidationError):
        PassAtEvaluationResult(
            prompt='x',
            response=[{'not': 'valid'}],
            reward=[0.0],
            pass_at={1: 0.0},
            greedy_response='y',
            greedy_reward=0.0,
        )
