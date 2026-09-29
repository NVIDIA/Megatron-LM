#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path

WORKFLOW = Path(__file__).resolve().parents[1] / "workflows" / "_build_ci_container.yml"


class TestBuildCacheDonor(unittest.TestCase):
    """Execute the workflow selector against a registry with controlled cache availability."""

    @classmethod
    def setUpClass(cls) -> None:
        selector_step = next(
            step
            for step in WORKFLOW.read_text().split("\n      - name: ")
            if step.startswith("Select cache donor\n")
        )
        _, run = selector_step.split("\n        run: |\n", 1)
        cls.selector = textwrap.dedent(run)

    def setUp(self) -> None:
        self.caches = {
            "PR_BUILD_HASH_CACHE": "registry.test/megatron-lm:main-dev-pr-7696-build-new-gb-gpu",
            "TARGET_BUILD_HASH_CACHE": "registry.test/megatron-lm:main-dev-build-new-gb-gpu",
            "PR_FALLBACK_CACHE": "registry.test/megatron-lm:main-dev-7696-buildcache-gb-gpu",
            "BASELINE_CACHE": "registry.test/megatron-lm:main-dev-baseline-buildcache-gb-gpu",
            "LEGACY_CACHE": "registry.test/megatron-lm:0-buildcache-gb-gpu",
        }

    def select_cache(
        self, available: set[str], *, pr_key: str = "new", fallback: str | None = None
    ) -> tuple[str, list[str]]:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docker = root / "docker"
            docker.write_text("""#!/usr/bin/env bash
set -euo pipefail
[[ "$#" -eq 4 && "$1" = buildx && "$2" = imagetools && "$3" = inspect ]]
printf '%s\n' "$4" >> "$INSPECTION_LOG"
while IFS= read -r candidate; do
    if [ "$candidate" = "$4" ]; then
        exit 0
    fi
done < "$AVAILABLE_CACHES"
exit 1
""")
            docker.chmod(0o755)
            manifests = root / "manifests"
            manifests.write_text("".join(f"{cache}\n" for cache in sorted(available)))
            inspection_log = root / "inspections"
            output = root / "output"
            environment = {
                **os.environ,
                **self.caches,
                "PATH": f"{root}{os.pathsep}{os.defpath}",
                "PR_BUILD_HASH_KEY": pr_key,
                "PR_FALLBACK_CACHE": (
                    self.caches["PR_FALLBACK_CACHE"] if fallback is None else fallback
                ),
                "AVAILABLE_CACHES": str(manifests),
                "INSPECTION_LOG": str(inspection_log),
                "GITHUB_OUTPUT": str(output),
            }
            result = subprocess.run(
                ["bash", "--noprofile", "--norc", "-e", "-o", "pipefail", "-c", self.selector],
                env=environment,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            outputs = dict(line.split("=", 1) for line in output.read_text().splitlines())
            return outputs["donor"], inspection_log.read_text().splitlines()

    def test_exact_pr_hash_has_priority(self) -> None:
        donor, inspected = self.select_cache(set(self.caches.values()))
        self.assertEqual(donor, self.caches["PR_BUILD_HASH_CACHE"])
        self.assertEqual(inspected, [donor])

    def test_exact_target_hash_precedes_pr_fallback(self) -> None:
        available = set(self.caches.values()) - {self.caches["PR_BUILD_HASH_CACHE"]}
        donor, inspected = self.select_cache(available)
        self.assertEqual(donor, self.caches["TARGET_BUILD_HASH_CACHE"])
        self.assertEqual(inspected, [self.caches["PR_BUILD_HASH_CACHE"], donor])

    def test_pr_fallback_survives_a_dependency_hash_change(self) -> None:
        available = {
            "registry.test/megatron-lm:main-dev-pr-7696-build-old-gb-gpu",
            self.caches["PR_FALLBACK_CACHE"],
            self.caches["BASELINE_CACHE"],
            self.caches["LEGACY_CACHE"],
        }
        donor, inspected = self.select_cache(available)
        self.assertEqual(donor, self.caches["PR_FALLBACK_CACHE"])
        self.assertEqual(inspected, list(self.caches.values())[:3])

    def test_missing_pr_caches_fall_back_to_baseline(self) -> None:
        donor, inspected = self.select_cache(
            {self.caches["BASELINE_CACHE"], self.caches["LEGACY_CACHE"]}
        )
        self.assertEqual(donor, self.caches["BASELINE_CACHE"])
        self.assertEqual(inspected, list(self.caches.values())[:4])

    def test_legacy_is_used_when_newer_caches_are_absent(self) -> None:
        for available in ({self.caches["LEGACY_CACHE"]}, set()):
            with self.subTest(available=available):
                donor, inspected = self.select_cache(available)
                self.assertEqual(donor, self.caches["LEGACY_CACHE"])
                self.assertEqual(inspected, list(self.caches.values()))

    def test_empty_pr_fallback_is_not_inspected(self) -> None:
        # LTS callers receive an empty PR_FALLBACK_CACHE from the workflow expression.
        donor, inspected = self.select_cache({self.caches["BASELINE_CACHE"]}, fallback="")
        self.assertEqual(donor, self.caches["BASELINE_CACHE"])
        self.assertEqual(
            inspected,
            [
                self.caches[key]
                for key in ("PR_BUILD_HASH_CACHE", "TARGET_BUILD_HASH_CACHE", "BASELINE_CACHE")
            ],
        )

    def test_non_pr_build_ignores_pr_caches(self) -> None:
        available = {
            self.caches["PR_BUILD_HASH_CACHE"],
            self.caches["PR_FALLBACK_CACHE"],
            self.caches["BASELINE_CACHE"],
        }
        donor, inspected = self.select_cache(available, pr_key="")
        self.assertEqual(donor, self.caches["BASELINE_CACHE"])
        self.assertEqual(
            inspected, [self.caches["TARGET_BUILD_HASH_CACHE"], self.caches["BASELINE_CACHE"]]
        )


if __name__ == "__main__":
    unittest.main()
