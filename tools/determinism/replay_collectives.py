# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Launch captured collective replay with the standard early determinism policy."""

from __future__ import annotations

import argparse
from pathlib import Path


class CaptureOptions:
    """Register the replay module's capture options for this pytest session."""

    def pytest_addoption(self, parser):
        parser.addoption("--collective-capture", type=Path, default=None)
        parser.addoption("--collective-max-bytes", type=int, default=256 * 1024 * 1024)


def main(argv: list[str] | None = None) -> int:
    """Run under the capture's torchrun topology; keep raw blobs on that host."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--max-bytes", type=int, default=256 * 1024 * 1024)
    args = parser.parse_args(argv)
    if args.max_bytes < 1 or not args.capture.is_dir():
        parser.error("An existing capture directory and positive byte limit are required")
    try:
        from megatron.determinism import bootstrap_training_determinism
        from tools.determinism import pytest_plugin
    except ImportError as error:
        if error.name not in (
            "megatron.determinism",
            "tools.determinism.pytest_plugin",
            "tools.determinism",
        ):
            raise
        parser.exit(
            2,
            "Collective replay requires megatron.determinism and "
            "tools.determinism.pytest_plugin.\n",
        )

    bootstrap_training_determinism(["--deterministic-mode"])
    import pytest

    # Generic unit-test conftest sets NCCL defaults that may differ from the
    # original recipe. This dedicated fixture owns its groups and needs no data.
    directory = Path(__file__).resolve().parents[2] / "tests/unit_tests/determinism/kernels"

    class PassedCases:
        passed = 0

        def pytest_runtest_logreport(self, report):
            if report.when == "call" and report.passed:
                self.passed += 1

    results = PassedCases()
    status = pytest.main(
        [
            "--determinism-evidence-dir",
            str(args.evidence),
            "--confcutdir=" + str(directory),
            "--collective-capture",
            str(args.capture.resolve()),
            "--collective-max-bytes",
            str(args.max_bytes),
            str(directory / "test_captured_collectives.py"),
            "-q",
        ],
        plugins=[pytest_plugin, CaptureOptions(), results],
    )
    if status == 0 and results.passed == 0:
        parser.exit(2, "Collective replay produced no passed cases; coverage is not verified.\n")
    return int(status)


if __name__ == "__main__":
    raise SystemExit(main())
