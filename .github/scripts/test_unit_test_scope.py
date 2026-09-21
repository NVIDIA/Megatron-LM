#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Exercise unit-test scope decisions against complete temporary Git histories."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("unit_test_scope.py").resolve()
GOLDEN = "tests/functional_tests/test_cases/gpt/small/golden_values_dev_dgx_h100.json"
CONFIG = "tests/functional_tests/test_cases/gpt/small/model_config.yaml"
LAUNCH = "examples/gpt/train.sh"


class TestUnitTestScope(unittest.TestCase):
    """Check allowed paths, mixed changes, renames, and conservative fallbacks."""

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.repo = Path(self.directory.name)
        self.environment = {
            **os.environ,
            "GIT_AUTHOR_NAME": "Scope Test",
            "GIT_AUTHOR_EMAIL": "scope@example.test",
            "GIT_COMMITTER_NAME": "Scope Test",
            "GIT_COMMITTER_EMAIL": "scope@example.test",
        }
        self.git("init", "--quiet")
        self.write("megatron/core/model.py", "runtime\n")
        self.write(GOLDEN)
        self.write(LAUNCH, "launch\n")
        self.base = self.commit()

    def git(self, *args: str) -> str:
        """Run Git in the fixture repository without using the developer's identity."""
        return subprocess.run(
            ["git", *args],
            cwd=self.repo,
            env=self.environment,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    def write(self, path: str, text: str = "initial\n") -> None:
        """Create or replace a fixture file."""
        target = self.repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)

    def commit(self, parent: str | None = None) -> str:
        """Create fixture commit objects without invoking local hooks or signing."""
        self.git("add", "--all")
        tree = self.git("write-tree")
        parents = ["-p", parent] if parent else []
        commit = self.git("commit-tree", tree, *parents, "-m", "Fixture")
        self.git("update-ref", "HEAD", commit)
        return commit

    def assert_decision(self, expected: bool, *args: str, cwd: Path | None = None) -> None:
        """Check the public CLI output and successful conservative fallback."""
        result = subprocess.run(
            [sys.executable, str(SCRIPT), *args],
            cwd=cwd or self.repo,
            env=self.environment,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.stdout, f"skip_unit_tests={str(expected).lower()}\n", result.stderr)
        self.assertIn("Unit tests", result.stderr)

    def assert_head_decision(self, expected: bool) -> None:
        """Commit the current changes and compare the full fixture PR."""
        head = self.commit(self.base)
        self.assert_decision(expected, "--base", self.base, "--head", head)

    def test_each_allowed_category(self) -> None:
        for path in (
            GOLDEN,
            "tests/functional_tests/test_cases/gpt/small/golden_values.json",
            CONFIG,
            "tests/test_utils/recipes/h100/gpt.yaml",
            "tests/test_utils/recipes/gb200/gpt.yaml",
            LAUNCH,
            "examples/model/slurm/config.sh",
        ):
            with self.subTest(path=path):
                self.git("reset", "--hard", self.base)
                self.write(path, "changed\n")
                self.assert_head_decision(True)

    def test_combining_exempt_categories(self) -> None:
        for path in (GOLDEN, CONFIG, LAUNCH, "tests/test_utils/recipes/h100/gpt.yaml"):
            self.write(path, "changed\n")
        self.assert_head_decision(True)

    def test_other_paths_keep_unit_tests(self) -> None:
        for path in (
            "megatron/core/model.py",
            "tests/unit_tests/test_model.py",
            "tests/test_utils/recipes/h100/unit-tests.yaml",
            "tests/test_utils/recipes/gb200/unit-tests.yaml",
            "tests/test_utils/recipes/_build-mcore-dev.yaml",
            "tests/test_utils/recipes/_cleanup.yaml",
            "tests/test_utils/recipes/new_platform/gpt.yaml",
            "tests/test_utils/recipes/h100/nested/gpt.yaml",
            "tests/functional_tests/shell_test_utils/run_ci_test.sh",
            "tests/functional_tests/test_cases/gpt/small/helper.py",
            "tests/functional_tests/test_cases/gpt/small/golden_values.txt",
            "tests/functional_tests/test_cases/gpt/small/nested/model_config.yaml",
            "examples/gpt/train.py",
            "tools/launch.sh",
            "docs/guide.md",
            ".github/workflows/cicd-main.yml",
            "uv.lock",
        ):
            with self.subTest(path=path):
                self.git("reset", "--hard", self.base)
                self.write(path, "changed\n")
                self.assert_head_decision(False)

    def test_mixed_exempt_and_nonexempt_paths(self) -> None:
        self.write(GOLDEN, "changed\n")
        self.write("megatron/core/model.py", "changed\n")
        self.assert_head_decision(False)

    def test_full_pr_diff_keeps_earlier_runtime_change(self) -> None:
        self.write("megatron/core/model.py", "changed\n")
        earlier_head = self.commit(self.base)
        self.write(GOLDEN, "changed\n")
        head = self.commit(earlier_head)
        self.assert_decision(False, "--base", self.base, "--head", head)

    def test_rename_from_nonexempt_to_exempt_path(self) -> None:
        (self.repo / "megatron/core/model.py").rename(self.repo / LAUNCH)
        self.assert_head_decision(False)

    def test_rename_from_exempt_to_nonexempt_path(self) -> None:
        (self.repo / LAUNCH).rename(self.repo / "megatron/core/model.py")
        self.assert_head_decision(False)

    def test_rename_within_exempt_paths(self) -> None:
        (self.repo / LAUNCH).rename(self.repo / "examples/gpt/renamed.sh")
        self.assert_head_decision(True)

    def test_exempt_deletion(self) -> None:
        (self.repo / GOLDEN).unlink()
        self.assert_head_decision(True)

    def test_nonexempt_deletion(self) -> None:
        (self.repo / "megatron/core/model.py").unlink()
        self.assert_head_decision(False)

    def test_whitespace_and_newline_paths(self) -> None:
        self.write("examples/gpt/space and\nnewline.sh")
        self.assert_head_decision(True)
        self.write("megatron/core/unexpected\nexamples/gpt/train.sh")
        self.assert_head_decision(False)

    def test_empty_diff_keeps_unit_tests(self) -> None:
        self.assert_decision(False, "--base", self.base, "--head", self.base)

    def test_missing_and_invalid_refs_keep_unit_tests(self) -> None:
        self.assert_decision(False)
        for ref in ("", "invalid", "--help", "a" * 40):
            with self.subTest(ref=ref):
                self.assert_decision(False, f"--base={ref}", "--head", self.base)
                self.assert_decision(False, "--base", self.base, f"--head={ref}")

    def test_unrelated_commits_keep_unit_tests(self) -> None:
        self.write(GOLDEN, "changed\n")
        unrelated_head = self.commit()
        self.assert_decision(False, "--base", self.base, "--head", unrelated_head)

    def test_git_failure_keeps_unit_tests(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.assert_decision(
                False, "--base", self.base, "--head", self.base, cwd=Path(directory)
            )


if __name__ == "__main__":
    unittest.main()
