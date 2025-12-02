# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import argparse
import json
import os
import shutil
import subprocess
import time

OUTPUT_DIR = "/tmp/dyn-inf-cuda-graph-test"
NUM_CUDA_GRAPHS_LIST = [0, 1, 2, 4, 8, 16]
INCOMING_REQUESTS_PER_STEP = 8
NUM_TRIALS = 1  # use >1 if re-adding latency validation


def clear_output_dir() -> None:
    """Clear output directory."""
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    os.mkdir(OUTPUT_DIR)


def get_output_path(num_cuda_graphs: int, trial_idx: int) -> str:
    """Get output path for a given test.

    Args:
        num_cuda_graphs (int): Number of cuda graphs used.
        trial_idx (int): Trial index, for when running multiple trials per test.

    Return:
        (str) Test output path.
    """
    return os.path.join(OUTPUT_DIR, f"n{num_cuda_graphs}_t{trial_idx}.json")


def run_tests(args: argparse.Namespace) -> None:
    """Run all tests, iterating over `num_cuda_graphs` and `NUM_TRIALS`."""

    # Iterate `num_cuda_graphs` and `NUM_TRIALS`.
    for num_cuda_graphs_idx, num_cuda_graphs in enumerate(NUM_CUDA_GRAPHS_LIST):
        for trial_idx in range(NUM_TRIALS):

            # Print progress.
            print()
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print(
                "[n %d/%d, t %d/%d] num_cuda_graphs: %d."
                % (
                    num_cuda_graphs_idx,
                    len(NUM_CUDA_GRAPHS_LIST),
                    trial_idx,
                    NUM_TRIALS,
                    num_cuda_graphs,
                )
            )
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print()
            time.sleep(2)

            # Environment variables.
            os.environ["CHECKPOINT_DIR"] = args.checkpoint_dir
            os.environ["TOKENIZER_MODEL"] = args.tokenizer_model
            os.environ["NUM_CUDA_GRAPHS"] = str(num_cuda_graphs)
            os.environ["INCOMING_REQUESTS_PER_STEP"] = str(INCOMING_REQUESTS_PER_STEP)
            os.environ["OUTPUT_PATH"] = get_output_path(num_cuda_graphs, trial_idx)

            # Run test.
            result = subprocess.run(
                [
                    "bash",
                    "tests/functional_tests/test_cases/gpt/gpt_dynamic_inference_tp1_pp1_583m_cuda_graphs_validation/cuda_graphs.sh",
                ],
                # capture_output=True, # uncomment to read stderr below
                # text=True,           # uncomment to read stderr below
            )
            assert result.returncode == 0, result.stderr


def load_results(num_cuda_graphs: int) -> dict:
    """Load all trial outputs for a given `num_cuda_graphs`.

    Args:
        num_cuda_graphs (int): Number of cuda graphs used.

    Return:
        (dict) Dictionary containing test results.
    """

    # Load trial outputs.
    results = [
        list(json.load(open(get_output_path(num_cuda_graphs, t))).values())[-1]
        for t in range(NUM_TRIALS)
    ]

    # Use minimum latency across trials.
    min_latency = min(r["latency"] for r in results)

    # Initialize merged result. (All values should be identical except latency.)
    result = {**results[0], "latency": min_latency}

    # Latency proxy is the inner product between `cuda_graph_request_count` and
    # its corresponding usage (i.e., number of times that cuda graph was used).
    if num_cuda_graphs == 0:
        result["latency_proxy"] = None
    else:
        result["latency_proxy"] = sum(
            int(k) * v for k, v in results[0]["cuda_graph_request_count_map"].items()
        )
        result["cuda_graph_request_count_map"] = {
            int(k): v for k, v in result["cuda_graph_request_count_map"].items()
        }

    return result


def validate_cuda_graph_request_counts(result_map: dict) -> None:
    """Validate `cuda_graph_request_count` usage across tests.

    For each test (i.e., each `num_cuda_graphs`), we validate how many times each
    cuda graph was used within the test.

    Args:
        result_map (dict): Map of `num_cuda_graphs` to test results.
    """

    # Expected counts maps 'num cuda graphs' to 'cuda graph request count usage'.
    expected_cuda_graph_request_count_maps = {
        0: None,
        1: {2000: 277},
        2: {2000: 129, 1000: 148},
        4: {2000: 65, 1512: 63, 1008: 63, 504: 86},
        8: {2000: 32, 1792: 30, 1536: 32, 1280: 32, 1024: 32, 768: 32, 512: 32, 256: 55},
        16: {
            2000: 13,
            1920: 19,
            1792: 14,
            1664: 16,
            1536: 16,
            1408: 16,
            1280: 16,
            1152: 16,
            1024: 16,
            896: 16,
            768: 16,
            640: 16,
            512: 16,
            384: 16,
            256: 16,
            128: 39,
        },
    }

    # Validate each test.
    for n in NUM_CUDA_GRAPHS_LIST:
        expected_cuda_graph_request_count_map = expected_cuda_graph_request_count_maps[n]
        actual_cuda_graph_request_count_map = result_map[n]["cuda_graph_request_count_map"]
        assert expected_cuda_graph_request_count_map == actual_cuda_graph_request_count_map


def validate_step_counts(result_map: dict) -> None:
    """Validate engine step counts.

    This value should be identical across all tests, regardless of whether cuda
    graphs are enabled or not.

    Args:
        result_map (dict): Map of `num_cuda_graphs` to test results.
    """
    gold_step_count = result_map[0]["step_count"]
    for n in NUM_CUDA_GRAPHS_LIST:
        assert result_map[n]["step_count"] == gold_step_count


def validate_latencies(result_map: dict) -> None:
    """Validate that latency decreases as we increase the number of cuda graphs.

    *Note*: This test is disabled for now, since the latency difference between
    these small tests is small, rendering this check unstable.

    Args:
        result_map (dict): Map of `num_cuda_graphs` to test results.
    """
    raise Exception("This check is currently disabled, for stability reasons.")
    latency_n0 = result_map[0]["total_time"]
    latency_n1 = result_map[1]["total_time"]
    latency_n8 = result_map[8]["total_time"]
    assert latency_n0 > latency_n1 > latency_n8


def validate_latency_proxies(result_map: dict) -> None:
    """Validate that the latency 'proxy' decreases as we increase the number of cuda graphs.

    Latency proxy is computed as the sum of `cuda_graph_request_count` *
    `cuda_graph_usage` for a given test.

    Args:
        result_map (dict): Map of `num_cuda_graphs` to test results.
    """
    prev_latency_proxy = None
    for n in NUM_CUDA_GRAPHS_LIST:
        crnt_latency_proxy = result_map[n]["latency_proxy"]
        if prev_latency_proxy is not None:
            assert crnt_latency_proxy < prev_latency_proxy
        prev_latency_proxy = crnt_latency_proxy


def validate_logprobs(result_map: dict) -> None:
    """Validate the logprob tensors.

    The logprobs should remain bitwise equal whether cuda graphs are enabled or not.

    Args:
        result_map (dict): Map of `num_cuda_graphs` to test results.
    """
    gold_logprobs = result_map[0]["logprobs"]
    for n in NUM_CUDA_GRAPHS_LIST:
        assert result_map[n]["logprobs"] == gold_logprobs


def validate_results() -> None:
    """Validate test results, by running various checks."""

    # Load results.
    result_map = {n: load_results(n) for n in NUM_CUDA_GRAPHS_LIST}

    # Validate.
    validate_cuda_graph_request_counts(result_map)
    validate_step_counts(result_map)
    # validate_latencies(result_map) # disabled for now due to noisy timing.
    validate_latency_proxies(result_map)
    validate_logprobs(result_map)

    # Success.
    print("~~~")
    print("success ... dynamic inference cuda graph tests.")
    print("~~~")


def main() -> None:
    """Run and validate inference tests."""

    # Arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True, help="Path to checkpoint directory.")
    parser.add_argument("--tokenizer-model", required=True, help="Path to tokenizer model.")
    args = parser.parse_args()

    # Clear output directory with json results.
    clear_output_dir()

    # Run inference with varying number of cuda graphs.
    run_tests(args)

    # Validate results between tests.
    validate_results()


if __name__ == "__main__":
    main()
