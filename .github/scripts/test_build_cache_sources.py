#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/_build_ci_container.yml"
REGISTRY = "registry.example.test/megatron-lm"
RUN_CACHE = f"{REGISTRY}:main-dev-7455-buildcache-aws-h100"
BASE_PR_CACHE = f"{REGISTRY}:main-dev-7092-buildcache-aws-h100"
BASELINE_CACHE = f"{REGISTRY}:main-dev-baseline-buildcache-aws-h100"
LEGACY_CACHE = f"{REGISTRY}:0-buildcache-aws-h100"


def _cache_selection_script() -> str:
    """Extract this literal YAML block without adding a lint dependency."""
    step = WORKFLOW.read_text().split("        id: cache_from\n", 1)[1]
    step = step.split("\n      - ", 1)[0]
    return textwrap.dedent(step.split("        run: |\n", 1)[1])


def _read_sources(path: Path) -> list[str]:
    """Read the multiline sources value written to GITHUB_OUTPUT."""
    lines = path.read_text().splitlines()
    header = next(line for line in lines if line.startswith("sources<<"))
    delimiter = header.split("<<", 1)[1]
    start = lines.index(header) + 1
    return lines[start : lines.index(delimiter, start)]


class TestBuildCacheSources(unittest.TestCase):
    """Exercise workflow cache discovery against a simulated registry."""

    def _assert_sources(self, available, expected, expected_probes, **overrides):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            output = directory / "github-output"
            probes = directory / "probes"
            docker = directory / "docker"
            docker.write_text(f"#!{sys.executable}\n" + textwrap.dedent("""\
                    import json
                    import os
                    import sys

                    assert sys.argv[1:4] == ["buildx", "imagetools", "inspect"]
                    assert len(sys.argv) == 5
                    candidate = sys.argv[4]
                    with open(os.environ["CACHE_PROBES"], "a") as stream:
                        stream.write(candidate + "\\n")
                    sys.exit(0 if candidate in json.loads(os.environ["AVAILABLE_CACHES"]) else 1)
                    """))
            docker.chmod(0o755)
            result = subprocess.run(
                ["bash", "-e", "-o", "pipefail"],
                input=_cache_selection_script(),
                cwd=directory,
                env={
                    **os.environ,
                    "PATH": str(directory) + os.pathsep + os.environ["PATH"],
                    "AVAILABLE_CACHES": json.dumps(available),
                    "CACHE_PROBES": str(probes),
                    "GITHUB_OUTPUT": str(output),
                    "BASE_PR_KEY": "main-dev-7092",
                    "BASE_PR_CACHE": BASE_PR_CACHE,
                    "BASELINE_CACHE": BASELINE_CACHE,
                    "EVENT_NAME": "push",
                    "LEGACY_CACHE": LEGACY_CACHE,
                    "RUN_CACHE": RUN_CACHE,
                    **overrides,
                },
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(
                _read_sources(output), [f"type=registry,ref={reference}" for reference in expected]
            )
            self.assertEqual(probes.read_text().splitlines(), expected_probes)

    def test_imports_every_available_pr_cache_source(self):
        """BuildKit receives readable namespaced caches and ignores the legacy cache."""
        references = [RUN_CACHE, BASE_PR_CACHE, BASELINE_CACHE]
        self._assert_sources([*references, LEGACY_CACHE], references, references)

    def test_available_base_pr_does_not_hide_the_main_baseline(self):
        """A readable older donor must not prevent importing newer main layers."""
        available = [BASE_PR_CACHE, BASELINE_CACHE, LEGACY_CACHE]
        self._assert_sources(
            available, [BASE_PR_CACHE, BASELINE_CACHE], [RUN_CACHE, BASE_PR_CACHE, BASELINE_CACHE]
        )

    def test_missing_base_pr_key_does_not_probe_its_reference(self):
        """Builds without an associated base PR ignore that candidate."""
        self._assert_sources(
            [RUN_CACHE, BASE_PR_CACHE, BASELINE_CACHE, LEGACY_CACHE],
            [RUN_CACHE, BASELINE_CACHE],
            [RUN_CACHE, BASELINE_CACHE],
            BASE_PR_KEY="",
        )

    def test_merge_group_keeps_its_baseline_first_sources(self):
        """Merge groups import the baseline first and ignore the legacy cache."""
        self._assert_sources(
            [RUN_CACHE, BASE_PR_CACHE, BASELINE_CACHE, LEGACY_CACHE],
            [BASELINE_CACHE, RUN_CACHE],
            [BASELINE_CACHE, RUN_CACHE],
            EVENT_NAME="merge_group",
        )

    def test_duplicate_main_and_run_reference_is_inspected_and_imported_once(self):
        """Main baseline builds reuse one probe when the run and baseline keys match."""
        self._assert_sources(
            [BASELINE_CACHE, LEGACY_CACHE],
            [BASELINE_CACHE],
            [BASELINE_CACHE],
            BASE_PR_KEY="",
            RUN_CACHE=BASELINE_CACHE,
        )

    def test_only_legacy_cache_emits_empty_sources(self):
        """A registry with only the legacy cache produces an empty cache input."""
        self._assert_sources([LEGACY_CACHE], [], [RUN_CACHE, BASE_PR_CACHE, BASELINE_CACHE])


if __name__ == "__main__":
    unittest.main()
