# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import json
import logging
import math
from statistics import median

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_NON_REQUEST_TOP_LEVEL_KEYS = {
    # System-level metrics
    "throughput",
    # Peak memory metrics (added by inference scripts; optionally checked if present in golden values)
    "mem-max-allocated-bytes",
}


def _median_as_float(value):
    """Convert scalar or list metric to a single float (median).

    For list metrics (e.g., per-request throughput), treat the first element as
    warmup if length > 1, matching existing throughput behavior.
    """
    if isinstance(value, list):
        assert len(value) > 0, "Metric list is empty."
        values = [float(v) for v in value]
        if len(values) > 1:
            values = values[1:]
        return float(median(values))
    return float(value)


def _bytes_to_gib(num_bytes: float) -> float:
    return float(num_bytes) / (1024.0**3)


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

    groundtruth_request_ids = set(output_groundtruth.keys()) - _NON_REQUEST_TOP_LEVEL_KEYS
    current_request_ids = set(output_current.keys()) - _NON_REQUEST_TOP_LEVEL_KEYS

    assert groundtruth_request_ids.issuperset(current_request_ids), (
        "Some request IDs from groundtruth are missing in current or current has unexpected IDs: "
        f"{sorted(groundtruth_request_ids)} vs {sorted(current_request_ids)}"
    )
    if groundtruth_request_ids != current_request_ids:
        logger.warning(
            "Some request IDs from groundtruth are missing in output; only the subset of ids in groundtruth will be tested: "
            f"{sorted(groundtruth_request_ids)} vs {sorted(current_request_ids)}"
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

    # Peak memory regression checks (optional: only if present in golden values).
    if "mem-max-allocated-bytes" in output_groundtruth:
        assert "mem-max-allocated-bytes" in output_current, (
            f"Golden values include mem-max-allocated-bytes but current output does not. "
            "Ensure the inference script records memory metrics to the output JSON."
        )
        sampled = _median_as_float(output_current["mem-max-allocated-bytes"])
        golden = _median_as_float(output_groundtruth["mem-max-allocated-bytes"])
        assert golden > 0, f"Golden mem_max_allocated_bytes must be > 0, got {golden}."

        low = 0.95 * golden
        high = 1.05 * golden

        if sampled < low:
            raise AssertionError(
                f"Memory is too low for mem-max-allocated-bytes: "
                f"expected within 5% of {golden:.0f} bytes ({_bytes_to_gib(golden):.3f} GiB) "
                f"but got {sampled:.0f} bytes ({_bytes_to_gib(sampled):.3f} GiB). "
                "This is >5% lower than expected; please update golden values in the functional tests."
            )
        if sampled > high:
            raise AssertionError(
                f"Memory is too high for mem-max-allocated-bytes: "
                f"expected within ±5% of {golden:.0f} bytes ({_bytes_to_gib(golden):.3f} GiB) "
                f"but got {sampled:.0f} bytes ({_bytes_to_gib(sampled):.3f} GiB). "
                "This is >5% higher than expected; this is likely a regression."
            )
        output_groundtruth.pop("mem-max-allocated-bytes")

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
