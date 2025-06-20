import json
import logging
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_inference_pipeline(golden_values_path: str, test_values_path: str) -> None:

    with open(golden_values_path, 'r') as f1, open(test_values_path, 'r') as f2:
        golden_values_content = f1.read()
        tensorboard_content = f2.read()

    output_groundtruth = json.loads(golden_values_content)

    if isinstance(output_groundtruth, str):
        output_groundtruth = json.loads(output_groundtruth)

    output_current = json.loads(tensorboard_content)
    if isinstance(output_current, str):
        output_current = json.loads(output_current)

    # Extract values directly from the loaded dictionaries
    tokens_groundtruth = output_groundtruth["generated_tokens"]
    tokens_current = output_current["generated_tokens"]
    logprobs_groundtruth = output_groundtruth["logprobs"]
    logprobs_current = output_current["logprobs"]

    # Check token equality
    assert (
        tokens_groundtruth == tokens_current
    ), f"Token mismatch:\nGround truth: {tokens_groundtruth}\nCurrent: {tokens_current}"

    # Check logprobs length and tolerance
    assert len(logprobs_groundtruth) == len(
        logprobs_current
    ), f"Logprobs length mismatch: {len(logprobs_groundtruth)} vs {len(logprobs_current)}"

    for i, (lp1, lp2) in enumerate(zip(logprobs_groundtruth, logprobs_current)):
        assert math.isclose(
            lp1, lp2, abs_tol=0.001
        ), f"Logprobs differ at index {i}: {lp1:.5f} vs {lp2:.5f}"
