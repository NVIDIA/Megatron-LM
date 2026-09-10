#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import base64
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Sequence
from unittest import mock

from select_unit_tests import (
    Change,
    CommandResult,
    _run_command,
    build_bucket_ownership,
    find_changes,
    full_run_reason,
    read_always_run_tests,
    read_buckets,
    select_unit_tests,
)


def _decode_test_files(payload: str) -> list[str]:
    return json.loads(base64.urlsafe_b64decode(payload).decode())


class TestSelectUnitTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        repo_root = Path(temporary_directory.name) / "repo"
        repo_root.mkdir()
        self.repo_root = repo_root.resolve()

        self._write("megatron/core/__init__.py")
        self._write("megatron/core/layers.py")
        self._write("megatron/core/helper_test.py")
        self._write("tests/unit_tests/test_root.py")
        self._write("tests/unit_tests/data/test_data.py")
        self._write("tests/unit_tests/models/test_model.py")
        self._write("tests/unit_tests/models/special/test_special.py")
        self._write("tests/unit_tests/always_run_tests.json", '["tests/unit_tests/test_root.py"]')

        self.buckets = [
            "tests/unit_tests/**",
            "tests/unit_tests/models/**",
            "tests/unit_tests/models/special/test_special.py",
        ]
        self.runner_calls: list[tuple[list[str], Path]] = []

    def _write(self, relative_path: str, contents: str = "") -> Path:
        path = self.repo_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)
        return path

    def _runner(self, result: CommandResult):
        def run(command: Sequence[str], cwd: Path) -> CommandResult:
            self.runner_calls.append((list(command), cwd))
            return result

        return run

    def _initialize_git_repository(self) -> str:
        subprocess.run(["git", "init", "--quiet"], cwd=self.repo_root, check=True)
        subprocess.run(
            ["git", "config", "user.email", "selective-tests@example.com"],
            cwd=self.repo_root,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Selective Tests"], cwd=self.repo_root, check=True
        )
        subprocess.run(["git", "config", "commit.gpgsign", "false"], cwd=self.repo_root, check=True)
        subprocess.run(["git", "add", "."], cwd=self.repo_root, check=True)
        subprocess.run(
            ["git", "commit", "--quiet", "-m", "initial"], cwd=self.repo_root, check=True
        )
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=self.repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    def _select(
        self,
        result: CommandResult,
        changes: Sequence[Change] | None = None,
        force_full: str | None = None,
        record_impact: bool = False,
        execute_full: str | None = None,
    ) -> dict[str, object]:
        return select_unit_tests(
            repo_root=self.repo_root,
            buckets=self.buckets,
            changes=(
                [Change(status="M", path="megatron/core/layers.py")] if changes is None else changes
            ),
            git_mode="branch",
            base_ref="base-sha",
            force_full=force_full,
            runner=self._runner(result),
            record_impact=record_impact,
            execute_full=execute_full,
        )

    def test_policy_allows_discoverable_source_tests_and_documentation(self) -> None:
        changes = [
            Change(status="M", path="megatron/core/layers.py"),
            Change(status="A", path="tests/unit_tests/test_root.py"),
            Change(status="M", path="docs/selective_testing.md"),
        ]

        self.assertIsNone(full_run_reason(self.repo_root, changes))

    def test_policy_fails_closed_for_unsafe_changes(self) -> None:
        cases = {
            "no changes": ([], "no changed files"),
            "deleted file": (
                [Change(status="D", path="megatron/core/layers.py")],
                "git status 'D'",
            ),
            "renamed file": (
                [Change(status="R", old_path="megatron/core/old.py", path="megatron/core/new.py")],
                "git status 'R'",
            ),
            "high-impact file": (
                [Change(status="M", path=".github/workflows/cicd-main.yml")],
                "high-impact file changed",
            ),
            "baseline configuration": (
                [Change(status="M", path="tests/unit_tests/always_run_tests.json")],
                "high-impact file changed",
            ),
            "dynamically loaded tokenizer backend": (
                [
                    Change(
                        status="M", path="megatron/core/tokenizers/text/libraries/null_tokenizer.py"
                    )
                ],
                "high-impact file changed",
            ),
            "package root": (
                [Change(status="A", path="megatron/__init__.py")],
                "high-impact file changed",
            ),
            "nested source package initializer": (
                [Change(status="M", path="megatron/core/models/gpt/__init__.py")],
                "high-impact file changed",
            ),
            "unit-test package initializer in a mixed change": (
                [
                    Change(status="M", path="megatron/core/layers.py"),
                    Change(status="M", path="tests/unit_tests/models/__init__.py"),
                ],
                "high-impact file changed",
            ),
            "non-Python file": (
                [Change(status="M", path="megatron/core/kernel.cu")],
                "unsupported non-Python file changed",
            ),
            "outside package": (
                [Change(status="M", path="tools/generate.py")],
                "Python file outside the analyzed package changed",
            ),
            "undiscoverable namespace": (
                [Change(status="M", path="megatron/experimental/kernel.py")],
                "pytest-impacted cannot discover namespace path",
            ),
        }

        for label, (changes, expected_reason) in cases.items():
            with self.subTest(label=label):
                reason = full_run_reason(self.repo_root, changes)
                self.assertIsNotNone(reason)
                self.assertIn(expected_reason, reason)

    def test_branch_changes_use_an_exact_base_commit(self) -> None:
        base_sha = self._initialize_git_repository()
        self._write("megatron/core/layers.py", "changed = True\n")
        subprocess.run(["git", "add", "."], cwd=self.repo_root, check=True)
        subprocess.run(["git", "commit", "--quiet", "-m", "change"], cwd=self.repo_root, check=True)

        self.assertEqual(
            find_changes(self.repo_root, "branch", base_sha),
            [Change(status="M", path="megatron/core/layers.py")],
        )
        with self.assertRaisesRegex(ValueError, "exact commit SHA"):
            find_changes(self.repo_root, "branch", "HEAD~1")

    def test_branch_changes_reject_a_non_ancestor_base_commit(self) -> None:
        base_sha = self._initialize_git_repository()
        self._write("megatron/core/layers.py", "feature = True\n")
        subprocess.run(["git", "add", "."], cwd=self.repo_root, check=True)
        subprocess.run(
            ["git", "commit", "--quiet", "-m", "feature"], cwd=self.repo_root, check=True
        )
        feature_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=self.repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

        subprocess.run(
            ["git", "checkout", "--quiet", "--detach", base_sha], cwd=self.repo_root, check=True
        )
        self._write("docs/diverged.md", "diverged\n")
        subprocess.run(["git", "add", "."], cwd=self.repo_root, check=True)
        subprocess.run(
            ["git", "commit", "--quiet", "-m", "diverged"], cwd=self.repo_root, check=True
        )
        divergent_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=self.repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        subprocess.run(
            ["git", "checkout", "--quiet", "--detach", feature_sha], cwd=self.repo_root, check=True
        )

        with self.assertRaisesRegex(ValueError, "is not an ancestor of HEAD"):
            find_changes(self.repo_root, "branch", divergent_sha)

    def test_unstaged_mode_fails_closed_for_staged_changes(self) -> None:
        self._initialize_git_repository()
        self._write("megatron/core/layers.py", "staged = True\n")
        subprocess.run(["git", "add", "megatron/core/layers.py"], cwd=self.repo_root, check=True)

        self.assertEqual(
            find_changes(self.repo_root, "unstaged", None),
            [Change(status="S", path="<staged changes>")],
        )

    def test_bucket_ownership_uses_the_most_specific_bucket(self) -> None:
        ownership = build_bucket_ownership(self.repo_root, self.buckets)

        self.assertEqual(
            ownership["tests/unit_tests/**"],
            {"tests/unit_tests/test_root.py", "tests/unit_tests/data/test_data.py"},
        )
        self.assertEqual(
            ownership["tests/unit_tests/models/**"], {"tests/unit_tests/models/test_model.py"}
        )
        self.assertEqual(
            ownership["tests/unit_tests/models/special/test_special.py"],
            {"tests/unit_tests/models/special/test_special.py"},
        )

    def test_bucket_ownership_rejects_unowned_tests(self) -> None:
        with self.assertRaisesRegex(ValueError, "not owned by a recipe bucket"):
            build_bucket_ownership(self.repo_root, ["tests/unit_tests/models/**"])

    def test_bucket_ownership_rejects_ambiguous_tests(self) -> None:
        with self.assertRaisesRegex(ValueError, "ambiguous bucket ownership"):
            build_bucket_ownership(
                self.repo_root,
                [
                    "tests/unit_tests/**",
                    "tests/unit_tests/models/test_*.py",
                    "tests/unit_tests/models/*.py",
                ],
            )

    def test_bucket_ownership_rejects_empty_recipe_bucket(self) -> None:
        with self.assertRaisesRegex(ValueError, "do not own any tests"):
            build_bucket_ownership(
                self.repo_root, ["tests/unit_tests/**", "tests/unit_tests/missing/**"]
            )

    def test_bucket_reader_rejects_path_traversal(self) -> None:
        bucket_file = self._write(
            "unsafe-buckets.json", '[{"bucket": "tests/unit_tests/../../test_escape.py"}]'
        )

        with self.assertRaisesRegex(ValueError, "invalid unit-test bucket"):
            read_buckets(bucket_file)

    def test_success_selects_exact_files_and_encodes_each_bucket_payload(self) -> None:
        model_test = self.repo_root / "tests/unit_tests/models/test_model.py"
        result = CommandResult(
            returncode=0,
            stdout=(
                f"{model_test}\n"
                "tests/unit_tests/models/special/test_special.py\n"
                "tests/unit_tests/models/test_model.py\n"
                "megatron/core/helper_test.py\n"
            ),
            stderr="harmless diagnostic\n",
        )

        report = self._select(result)

        self.assertEqual(report["mode"], "selective")
        self.assertEqual(report["selected_count"], 3)
        self.assertEqual(report["total_count"], 4)
        self.assertEqual(report["impacted_count"], 2)
        self.assertEqual(report["always_run_count"], 1)
        self.assertEqual(
            report["selected_files"],
            [
                "tests/unit_tests/models/special/test_special.py",
                "tests/unit_tests/models/test_model.py",
                "tests/unit_tests/test_root.py",
            ],
        )
        self.assertEqual([entry["bucket"] for entry in report["matrix"]], self.buckets)
        self.assertEqual(
            _decode_test_files(report["matrix"][0]["unit_test_files"]),
            ["tests/unit_tests/test_root.py"],
        )
        self.assertEqual(
            _decode_test_files(report["matrix"][1]["unit_test_files"]),
            ["tests/unit_tests/models/test_model.py"],
        )
        self.assertEqual(
            _decode_test_files(report["matrix"][2]["unit_test_files"]),
            ["tests/unit_tests/models/special/test_special.py"],
        )
        self.assertEqual(len(self.runner_calls), 1)
        command, cwd = self.runner_calls[0]
        self.assertEqual(cwd, self.repo_root)
        self.assertEqual(
            command,
            [
                "impacted-tests",
                "--module",
                "megatron",
                "--tests-dir",
                "tests",
                "--root-dir",
                str(self.repo_root),
                "--git-mode",
                "branch",
                "--base-branch",
                "base-sha",
            ],
        )

    def test_directly_changed_test_is_selected_when_tool_output_is_empty(self) -> None:
        report = self._select(
            CommandResult(returncode=0, stdout="", stderr=""),
            changes=[Change(status="M", path="tests/unit_tests/test_root.py")],
        )

        self.assertEqual(report["mode"], "selective")
        self.assertEqual(report["selected_files"], ["tests/unit_tests/test_root.py"])
        self.assertEqual(len(report["matrix"]), 1)
        self.assertEqual(report["matrix"][0]["bucket"], "tests/unit_tests/**")
        self.assertEqual(
            _decode_test_files(report["matrix"][0]["unit_test_files"]),
            ["tests/unit_tests/test_root.py"],
        )

    def test_empty_tool_output_falls_back_to_all_buckets(self) -> None:
        report = self._select(CommandResult(returncode=0, stdout="\n", stderr=""))

        self._assert_full_report(
            report, "pytest-impacted returned no unit tests for an analyzable change"
        )

    def test_documentation_only_runs_baseline_without_the_analysis_tool(self) -> None:
        report = self._select(
            CommandResult(returncode=7, stdout="", stderr="tool unavailable"),
            changes=[Change(status="M", path="docs/guide.md")],
        )

        self.assertEqual(report["mode"], "selective")
        self.assertEqual(report["selected_files"], ["tests/unit_tests/test_root.py"])
        self.assertEqual(report["always_run_count"], 1)
        self.assertEqual(report["impacted_count"], 0)
        self.assertEqual(self.runner_calls, [])

    def test_baseline_is_deduplicated_from_impacted_tests(self) -> None:
        report = self._select(
            CommandResult(returncode=0, stdout="tests/unit_tests/test_root.py\n", stderr="")
        )

        self.assertEqual(report["selected_count"], 1)
        self.assertEqual(report["impacted_count"], 1)
        self.assertEqual(report["always_run_count"], 1)

    def test_changed_test_and_baseline_do_not_mask_empty_source_analysis(self) -> None:
        for stdout in ("", "tests/unit_tests/test_root.py\n"):
            with self.subTest(stdout=stdout):
                report = self._select(
                    CommandResult(returncode=0, stdout=stdout, stderr=""),
                    changes=[
                        Change(status="M", path="megatron/core/layers.py"),
                        Change(status="M", path="tests/unit_tests/test_root.py"),
                    ],
                )

                self._assert_full_report(
                    report, "pytest-impacted returned no unit tests for an analyzable change"
                )

    def test_missing_baseline_falls_back_to_full(self) -> None:
        (self.repo_root / "tests/unit_tests/always_run_tests.json").unlink()

        report = self._select(CommandResult(returncode=0, stdout="", stderr=""))

        self._assert_full_report(report, "always-run test configuration is invalid")
        self.assertEqual(self.runner_calls, [])

    def test_invalid_baselines_fall_back_to_full(self) -> None:
        cases = [
            "not JSON",
            "[]",
            "{}",
            '["tests/unit_tests/test_missing.py"]',
            '["tests/unit_tests/../../test_escape.py"]',
            '["tests/unit_tests/test_*.py"]',
            '["tests/unit_tests/test_root.py", "tests/unit_tests/test_root.py"]',
            '["megatron/core/layers.py"]',
            '["tests/unit_tests/models/../test_root.py"]',
            '[42]',
        ]
        for contents in cases:
            with self.subTest(contents=contents):
                self._write("tests/unit_tests/always_run_tests.json", contents)
                report = self._select(CommandResult(returncode=0, stdout="", stderr=""))
                self._assert_full_report(report, "always-run test configuration is invalid")

    def test_repository_baseline_contains_ten_existing_owned_test_files(self) -> None:
        root = Path(__file__).resolve().parents[2]
        ownership = build_bucket_ownership(root, ["tests/unit_tests/**"])

        baseline = read_always_run_tests(
            root, root / "tests/unit_tests/always_run_tests.json", ownership
        )

        self.assertEqual(len(baseline), 10)

    def test_runner_setup_failure_falls_back_to_full(self) -> None:
        with mock.patch("select_unit_tests.run_pytest_impacted", side_effect=FileNotFoundError):
            report = self._select(CommandResult(returncode=0, stdout="", stderr=""))

        self._assert_full_report(report, "pytest-impacted could not run")

    def test_invalid_unchanged_consumer_cannot_silently_remove_dependency_edges(self) -> None:
        self._write(
            "tests/unit_tests/models/helpers.py",
            "from megatron.core.layers import Layer\ndef invalid(:\n",
        )

        report = self._select(
            CommandResult(returncode=0, stdout="tests/unit_tests/test_root.py\n", stderr="")
        )

        self._assert_full_report(report, "cannot analyze Python dependency input")
        self.assertEqual(self.runner_calls, [])

    def test_non_utf8_input_cannot_silently_remove_dependency_edges(self) -> None:
        (self.repo_root / "megatron/core/layers.py").write_bytes(b"# \xff\n")

        report = self._select(
            CommandResult(returncode=0, stdout="tests/unit_tests/test_root.py\n", stderr="")
        )

        self._assert_full_report(report, "cannot analyze Python dependency input")
        self.assertEqual(self.runner_calls, [])

    @unittest.skipUnless(
        shutil.which("impacted-tests"), "requires the selective-testing environment"
    )
    def test_real_analysis_follows_relative_imports_and_transitive_test_helpers(self) -> None:
        from pytest_impacted._rust import RUST_AVAILABLE

        if not RUST_AVAILABLE:
            self.skipTest("requires pytest-impacted[fast]")
        self._write("megatron/core/__init__.py", "raise RuntimeError('must not import source')\n")
        self._write("megatron/core/leaf.py", "VALUE = 1\n")
        self._write("megatron/core/layers.py", "from .leaf import VALUE\n")
        self._write(
            "tests/unit_tests/models/helpers.py", "from megatron.core.layers import VALUE\n"
        )
        self._write(
            "tests/unit_tests/models/test_model.py",
            "from tests.unit_tests.models.helpers import VALUE\n",
        )
        base_sha = self._initialize_git_repository()
        real_calls: list[list[str]] = []

        def real_runner(command: Sequence[str], cwd: Path) -> CommandResult:
            real_calls.append(list(command))
            return _run_command(command, cwd)

        for changed_path, contents in (
            ("megatron/core/leaf.py", "VALUE = 2\n"),
            (
                "tests/unit_tests/models/helpers.py",
                "from megatron.core.layers import VALUE\nRESULT = VALUE + 1\n",
            ),
        ):
            with self.subTest(changed_path=changed_path):
                calls_before = len(real_calls)
                self._write(changed_path, contents)
                subprocess.run(["git", "add", "."], cwd=self.repo_root, check=True)
                subprocess.run(
                    ["git", "commit", "--quiet", "-m", "change"], cwd=self.repo_root, check=True
                )
                report = select_unit_tests(
                    self.repo_root,
                    self.buckets,
                    find_changes(self.repo_root, "branch", base_sha),
                    "branch",
                    base_sha,
                    runner=real_runner,
                    record_impact=True,
                )
                self.assertEqual(report["mode"], "selective", report["reason"])
                self.assertEqual(len(real_calls), calls_before + 1)
                self.assertEqual(report["impact_analysis"]["status"], "succeeded")
                self.assertEqual(
                    report["impacted_files"], ["tests/unit_tests/models/test_model.py"]
                )
                self.assertEqual(
                    report["selected_files"],
                    ["tests/unit_tests/models/test_model.py", "tests/unit_tests/test_root.py"],
                )
                base_sha = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=self.repo_root,
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()

    def test_source_test_output_is_ignored_then_empty_result_falls_back(self) -> None:
        report = self._select(
            CommandResult(returncode=0, stdout="megatron/core/helper_test.py\n", stderr="")
        )

        self._assert_full_report(
            report, "pytest-impacted returned no unit tests for an analyzable change"
        )

    def test_impacted_conftest_falls_back_to_all_buckets(self) -> None:
        report = self._select(
            CommandResult(returncode=0, stdout="tests/unit_tests/conftest.py\n", stderr="")
        )

        self._assert_full_report(report, "pytest-impacted reported an impacted shared fixture")

    def test_tool_failure_falls_back_to_all_buckets(self) -> None:
        report = self._select(CommandResult(returncode=7, stdout="", stderr="analysis failed"))

        self._assert_full_report(report, "pytest-impacted failed with exit code 7")

    def test_unsafe_tool_diagnostics_fall_back_to_all_buckets(self) -> None:
        diagnostics = [
            "ERROR: failed to import module",
            "Traceback (most recent call last):",
            "pretrain_gpt.py could not be resolved to a\n         known module",
            "Syntax error while parsing megatron/core/layers.py",
            "module not found in discovered submodules",
        ]

        for stderr in diagnostics:
            with self.subTest(stderr=stderr):
                report = self._select(
                    CommandResult(
                        returncode=0,
                        stdout="tests/unit_tests/models/test_model.py\n",
                        stderr=stderr,
                    )
                )
                self._assert_full_report(report, "pytest-impacted reported an analysis error")

    def test_selector_process_timeout_falls_back_to_all_buckets(self) -> None:
        with mock.patch(
            "select_unit_tests.subprocess.run",
            side_effect=subprocess.TimeoutExpired(["impacted-tests"], timeout=300),
        ):
            result = _run_command(["impacted-tests"], self.repo_root)

        self.assertEqual(result.returncode, 124)
        self.assertIn("timed out", result.stderr)
        self._assert_full_report(self._select(result), "pytest-impacted failed with exit code 124")

    def test_missing_tool_output_path_falls_back_to_all_buckets(self) -> None:
        report = self._select(
            CommandResult(
                returncode=0, stdout="tests/unit_tests/models/test_missing.py\n", stderr=""
            )
        )

        self._assert_full_report(report, "selector returned a missing test file")

    def test_out_of_repository_tool_output_falls_back_to_all_buckets(self) -> None:
        outside_test = self.repo_root.parent / "outside" / "test_escape.py"
        outside_test.parent.mkdir()
        outside_test.write_text("")

        report = self._select(
            CommandResult(returncode=0, stdout="../outside/test_escape.py\n", stderr="")
        )

        self._assert_full_report(report, "selector returned a path outside the repository")

    def test_explicit_full_override_does_not_run_pytest_impacted(self) -> None:
        report = self._select(
            CommandResult(returncode=0, stdout="tests/unit_tests/test_root.py\n", stderr=""),
            force_full="Run tests label requested the complete suite",
        )

        self._assert_full_report(report, "Run tests label requested the complete suite")
        self.assertEqual(self.runner_calls, [])

    def test_recorded_candidate_is_preserved_when_execution_remains_full(self) -> None:
        path = "tests/unit_tests/models/test_model.py"
        report = self._select(
            CommandResult(returncode=0, stdout=f"{path}\n", stderr=""),
            record_impact=True,
            execute_full="no selective-testing label",
        )

        self._assert_full_report(report, "no selective-testing label")
        candidate = report["candidate_selection"]
        self.assertEqual(candidate["mode"], "selective")
        self.assertEqual(candidate["selected_files"], [path, "tests/unit_tests/test_root.py"])
        self.assertEqual(candidate["selected_count"], 2)
        self.assertEqual(len(candidate["matrix"]), 2)
        analysis = report["impact_analysis"]
        self.assertEqual(analysis["status"], "succeeded")
        self.assertEqual(analysis["selected_files"], [path])
        self.assertEqual(analysis["raw_selected_files"], [path])
        self.assertEqual(analysis["selected_count"], 1)
        self.assertEqual(analysis["returncode"], 0)
        self.assertGreaterEqual(analysis["duration_seconds"], 0)
        self.assertEqual(len(self.runner_calls), 1)

    def test_recording_runs_analyzer_even_for_high_impact_changes(self) -> None:
        path = "tests/unit_tests/models/test_model.py"
        report = self._select(
            CommandResult(returncode=0, stdout=f"{path}\n", stderr=""),
            changes=[Change(status="M", path=".github/workflows/cicd-main.yml")],
            record_impact=True,
        )

        self._assert_full_report(report, "high-impact file changed")
        self.assertEqual(report["candidate_selection"]["mode"], "full")
        self.assertEqual(report["impact_analysis"]["selected_files"], [path])
        self.assertEqual(len(self.runner_calls), 1)

    def test_recording_documentation_analysis_keeps_baseline_candidate(self) -> None:
        report = self._select(
            CommandResult(returncode=0, stdout="", stderr=""),
            changes=[Change(status="M", path="docs/guide.md")],
            record_impact=True,
            execute_full="no selective-testing label",
        )

        self._assert_full_report(report, "no selective-testing label")
        self.assertEqual(report["candidate_selection"]["selected_count"], 1)
        self.assertEqual(report["candidate_selection"]["mode"], "selective")
        self.assertEqual(report["impact_analysis"]["status"], "succeeded")
        self.assertEqual(report["impact_analysis"]["selected_count"], 0)
        self.assertEqual(len(self.runner_calls), 1)

    def test_recording_failures_preserve_raw_output_without_claiming_empty_success(self) -> None:
        path = "tests/unit_tests/models/test_model.py"
        for result in (
            CommandResult(returncode=7, stdout=f"{path}\n", stderr="analysis failed"),
            CommandResult(returncode=0, stdout=f"{path}\n", stderr="ERROR: incomplete analysis"),
            CommandResult(returncode=124, stdout="", stderr="timed out"),
            CommandResult(
                returncode=0, stdout=f"{path}\ntests/unit_tests/conftest.py\n", stderr=""
            ),
        ):
            with self.subTest(result=result):
                before = len(self.runner_calls)
                report = self._select(result, record_impact=True)
                self.assertEqual(report["candidate_selection"]["mode"], "full")
                analysis = report["impact_analysis"]
                self.assertEqual(analysis["status"], "failed")
                self.assertIsNone(analysis["selected_files"])
                self.assertIsNone(analysis["selected_count"])
                self.assertEqual(analysis["raw_selected_files"], result.stdout.splitlines())
                self.assertEqual(len(self.runner_calls), before + 1)

    def test_failed_documentation_analysis_produces_full_candidate(self) -> None:
        report = self._select(
            CommandResult(returncode=7, stdout="", stderr="failed"),
            changes=[Change(status="M", path="docs/guide.md")],
            record_impact=True,
        )

        self._assert_full_report(report, "pytest-impacted failed with exit code 7")
        self.assertEqual(report["impact_analysis"]["status"], "failed")
        self.assertEqual(report["candidate_selection"]["mode"], "full")

    def test_recording_attempts_analysis_even_for_invalid_python_inputs(self) -> None:
        self._write("tests/unit_tests/models/helpers.py", "def broken(:\n")
        report = self._select(
            CommandResult(
                returncode=0, stdout="tests/unit_tests/models/test_model.py\n", stderr=""
            ),
            record_impact=True,
        )

        self._assert_full_report(report, "cannot analyze Python dependency input")
        self.assertEqual(report["impact_analysis"]["status"], "failed")
        self.assertIsNone(report["impact_analysis"]["selected_count"])
        self.assertEqual(len(self.runner_calls), 1)

    def test_recording_launch_failure_is_not_retried(self) -> None:
        runner = mock.Mock(side_effect=FileNotFoundError("impacted-tests unavailable"))
        report = select_unit_tests(
            self.repo_root,
            self.buckets,
            [Change(status="M", path="megatron/core/layers.py")],
            "branch",
            "base-sha",
            runner=runner,
            record_impact=True,
        )

        self._assert_full_report(report, "pytest-impacted could not run")
        self.assertEqual(report["impact_analysis"]["status"], "failed")
        self.assertIsNone(report["impact_analysis"]["selected_count"])
        runner.assert_called_once()

    def test_hard_full_fallback_skips_analysis_and_has_no_candidate(self) -> None:
        report = self._select(
            CommandResult(returncode=0, stdout="", stderr=""),
            force_full="no valid merge comparison",
            record_impact=True,
        )

        self._assert_full_report(report, "no valid merge comparison")
        self.assertIsNone(report["candidate_selection"])
        self.assertEqual(report["impact_analysis"]["status"], "not_run")
        self.assertIsNone(report["impact_analysis"]["selected_count"])
        self.assertEqual(self.runner_calls, [])

    def _assert_full_report(self, report: dict[str, object], reason: str) -> None:
        self.assertEqual(report["mode"], "full")
        self.assertIn(reason, report["reason"])
        self.assertEqual(report["selected_files"], [])
        self.assertEqual(report["selected_count"], 4)
        self.assertEqual(report["total_count"], 4)
        self.assertEqual([entry["bucket"] for entry in report["matrix"]], self.buckets)
        self.assertTrue(all(entry["unit_test_files"] == "" for entry in report["matrix"]))


if __name__ == "__main__":
    unittest.main()
