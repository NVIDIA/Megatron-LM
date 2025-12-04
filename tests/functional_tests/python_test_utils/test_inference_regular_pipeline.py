# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import json
import logging
import math
from statistics import median

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_inference_pipeline(golden_values_path: str, test_values_path: str) -> None:

    with open(golden_values_path, 'r') as f1, open(test_values_path, 'r') as f2:
        golden_values_content = f1.read()
        tensorboard_content = f2.read()

    output_groundtruth = json.loads(golden_values_content)

    if isinstance(output_groundtruth, str):
        # Handle JSONL output, assume only one line in this case.
        output_groundtruth = json.loads(output_groundtruth)

    output_current = json.loads(tensorboard_content)
    if isinstance(output_current, str):
        # Handle JSONL output, assume only one line in this case.
        output_current = json.loads(output_current)

    assert set(output_groundtruth.keys()).issuperset(
        set(output_current.keys())
    ), f"Some IDs from groundtruth are missing in current: {output_groundtruth.keys()} vs {output_current.keys()}"
    if set(output_groundtruth.keys()) != set(output_current.keys()):
        logger.warning(
            f"Some IDs from groundtruth are missing in output, only the subset of ids in groundtruth will be tested: {output_groundtruth.keys()} vs {output_current.keys()}"
        )
    assert len(output_groundtruth) > 0, "No test performed for output"

    # Throughput assertions.
    if "throughput" in output_groundtruth.keys():

        # First warmup iteration is excluded from throughput statistics.
        throughput_sampled = median(output_current["throughput"][1:])
        throughput_golden = median(output_groundtruth["throughput"][1:])

        # 10% is empirically observed to be within hardware variance.
        assert (
            throughput_sampled >= 0.9 * throughput_golden
        ), f"Throughput is slower than expected! Expected to be within 10% of ~{throughput_golden} tok/s but benchmarked {output_current['throughput']} tok/s"

        # If throughput is significantly improved (> 20%), update golden values accordingly.
        assert (
            throughput_sampled < throughput_golden * 1.2
        ), f"Throughput has been improved from expected ~{throughput_golden} tok/s to {output_current['throughput']} tok/s. Please update golden values in the functional tests."

        output_groundtruth.pop('throughput')

    for request_id, groundtruth_results in output_groundtruth.items():
        current_results = output_current[request_id]

        at_least_one_test_loop = False
        if "generated_tokens" in groundtruth_results:
            at_least_one_test_loop = True
            tokens_groundtruth = groundtruth_results["generated_tokens"]
            tokens_current = current_results["generated_tokens"]
            # Check token equality
            assert (
                tokens_groundtruth == tokens_current
            ), f"Token mismatch:\nGround truth: {tokens_groundtruth}\nCurrent: {tokens_current}"

        if "logprobs" in groundtruth_results:
            at_least_one_test_loop = True
            logprobs_groundtruth = groundtruth_results["logprobs"]
            logprobs_current = current_results["logprobs"]
            # Check logprobs length and tolerance
            assert len(logprobs_groundtruth) == len(
                logprobs_current
            ), f"Logprobs length mismatch: {len(logprobs_groundtruth)} vs {len(logprobs_current)}"

            for i, (lp1, lp2) in enumerate(zip(logprobs_groundtruth, logprobs_current)):
                assert math.isclose(
                    lp1, lp2, abs_tol=0.001
                ), f"Logprobs differ at index {i}: {lp1:.5f} vs {lp2:.5f}"

        if "generated_text" in groundtruth_results:
            at_least_one_test_loop = True
            generated_text_groundtruth = groundtruth_results["generated_text"]
            generated_text_current = current_results["generated_text"]
            min_len = min(len(generated_text_groundtruth), len(generated_text_current))
            assert min_len > 0, (
                "Generated text mismatch:"
                f"\nGround truth: {generated_text_groundtruth}\nCurrent: {generated_text_current}"
            )
            assert generated_text_groundtruth[:min_len] == generated_text_current[:min_len], (
                "Generated text mismatch:"
                f"\nGround truth (truncated to {min_len} chars): {generated_text_groundtruth[:min_len]}"
                f"\nCurrent (truncated to {min_len} chars): {generated_text_current[:min_len]}"
            )

        if not at_least_one_test_loop:
            raise AssertionError(f"No test performed for output {groundtruth_results}")
