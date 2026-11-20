# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Select the launcher for a unit-test bucket."""

import argparse
import json
import sys

STANDARD_LAUNCHER = "tests/unit_tests/run_ci_test.sh"


def launcher_for(platform: str, bucket: str) -> str:
    """Return the launcher path relative to the test checkout."""
    if (
        platform in {"dgx_h100", "dgx_gb200"}
        and bucket == "tests/unit_tests/inference/test_batch_invariant_fp8.py"
    ):
        return "tests/test_utils/run_batch_invariant_fp8.sh"
    if platform == "dgx_h100" and bucket in {
        "tests/unit_tests/inference/engines/test_disagg_coordinator_pairwise.py",
        "tests/unit_tests/inference/test_disagg_transfer_pairwise.py",
    }:
        return "tests/test_utils/run_disagg_pairwise.sh"
    return STANDARD_LAUNCHER


def supports_testmon(launcher: str) -> bool:
    """Return whether the launcher records and consumes Testmon data."""
    return launcher == STANDARD_LAUNCHER


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--testmon-buckets", action="store_true")
    parser.add_argument("platform")
    parser.add_argument("bucket", nargs="?")
    args = parser.parse_args()
    if args.testmon_buckets:
        if args.bucket is not None:
            parser.error("--testmon-buckets reads the bucket list from standard input")
        buckets = json.load(sys.stdin)
        if not isinstance(buckets, list) or not all(isinstance(bucket, str) for bucket in buckets):
            parser.error("expected a JSON array of bucket names")
        supported = [
            bucket for bucket in buckets if supports_testmon(launcher_for(args.platform, bucket))
        ]
        print(json.dumps(supported))
    elif args.bucket is None:
        parser.error("specify a bucket or use --testmon-buckets")
    else:
        print(launcher_for(args.platform, args.bucket))


if __name__ == "__main__":
    main()
