# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Launch captured collective replay with the standard early determinism policy."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    """Run under the capture's torchrun topology; keep raw blobs on that host."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--max-bytes", type=int, default=256 * 1024 * 1024)
    args = parser.parse_args(argv)
    if args.max_bytes < 1 or not args.capture.is_dir():
        parser.error("An existing capture directory and positive byte limit are required")
    from megatron.determinism import bootstrap_training_determinism

    bootstrap_training_determinism(["--deterministic-mode"])
    import pytest

    os.environ["MCORE_DETERMINISM_COLLECTIVE_CAPTURE"] = str(args.capture.resolve())
    os.environ["MCORE_DETERMINISM_COLLECTIVE_MAX_BYTES"] = str(args.max_bytes)
    # Generic unit-test conftest sets NCCL defaults that may differ from the
    # original recipe. This dedicated fixture owns its groups and needs no data.
    directory = Path(__file__).resolve().parents[2] / "tests/unit_tests/determinism/kernels"
    return pytest.main(
        [
            "-p",
            "tools.determinism.pytest_plugin",
            "--determinism-evidence-dir",
            str(args.evidence),
            "--confcutdir=" + str(directory),
            str(directory / "test_captured_collectives.py"),
            "-q",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
