#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU-only checks for selection, CI policy, and selected-file transport."""

import ast
import base64
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import types
import unittest
from pathlib import Path
from unittest import mock

import yaml
from click.testing import CliRunner
from select_unit_tests import (
    Change,
    CommandResult,
    _only_always_run_changes,
    _run_command,
    build_bucket_ownership,
    find_changes,
    full_run_reason,
    read_always_run_tests,
    read_buckets,
    select_unit_tests,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
UNIT_ROOT = "tests/unit_tests"
BASELINE = f"{UNIT_ROOT}/test_root.py"
MODEL = f"{UNIT_ROOT}/models/test_model.py"
SOURCE = "megatron/core/layers.py"


def _encode(files: object) -> str:
    return base64.urlsafe_b64encode(json.dumps(files).encode()).decode()


class TemporaryRepository(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        self.bin = self.root / "bin"
        self.bin.mkdir()

    def _write(self, path: str, contents: str = "") -> Path:
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(textwrap.dedent(contents))
        return target

    def _executable(self, name: str, contents: str) -> None:
        target = self._write(f"bin/{name}", f"#!{sys.executable}\n" + textwrap.dedent(contents))
        target.chmod(0o755)

    def _git(self, *args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=self.root, check=True, capture_output=True, text=True
        ).stdout.strip()

    def _commit(self, message: str = "change") -> str:
        self._git("add", ".")
        self._git("commit", "--quiet", "-m", message)
        return self._git("rev-parse", "HEAD")

    def _init_git(self) -> str:
        self._git("init", "--quiet")
        self._git("config", "user.name", "Selection test")
        self._git("config", "user.email", "selection@example.com")
        self._git("config", "commit.gpgsign", "false")
        return self._commit("base")


class TestSelection(TemporaryRepository):
    def setUp(self) -> None:
        super().setUp()
        self.special = f"{UNIT_ROOT}/models/special/test_special.py"
        self.buckets = [f"{UNIT_ROOT}/**", f"{UNIT_ROOT}/models/**", self.special]
        for path in (
            "megatron/core/__init__.py",
            SOURCE,
            "megatron/core/helper_test.py",
            BASELINE,
            MODEL,
            self.special,
            f"{UNIT_ROOT}/data/test_data.py",
        ):
            self._write(path)
        self._write(f"{UNIT_ROOT}/always_run_tests.json", json.dumps([BASELINE]))
        self.runner = mock.Mock(return_value=CommandResult(0, MODEL + "\n", ""))

    def _select(self, changes=None, **options) -> dict:
        return select_unit_tests(
            self.root,
            self.buckets,
            [Change("M", SOURCE)] if changes is None else changes,
            "branch",
            "1234567",
            runner=self.runner,
            **options,
        )

    def _assert_full(self, report: dict, reason: str) -> None:
        self.assertEqual(report["mode"], "full")
        self.assertIn(reason, report["reason"])
        self.assertEqual(report["selected_files"], [])
        self.assertEqual(report["selected_count"], 4)
        self.assertEqual(report["total_count"], 4)
        self.assertEqual(
            report["matrix"], [{"bucket": bucket, "unit_test_files": ""} for bucket in self.buckets]
        )

    def test_exact_impacted_baseline_and_directly_changed_files(self) -> None:
        fixture = f"{UNIT_ROOT}/conftest.py"
        self._write(fixture)
        self.runner.return_value = CommandResult(
            0,
            f"{self.root / MODEL}\n{MODEL}\n{BASELINE}\n{fixture}\nmegatron/core/helper_test.py\n",
            "",
        )
        report = self._select([Change("M", SOURCE), Change("A", self.special)])
        self.assertEqual(report["mode"], "selective", report["reason"])
        self.assertEqual(report["selected_files"], sorted([BASELINE, MODEL, self.special]))
        self.assertEqual(report["selected_count"], 3)
        self.assertEqual(report["always_run_count"], 1)
        self.assertEqual(report["impacted_count"], 3)
        self.assertEqual([entry["bucket"] for entry in report["matrix"]], self.buckets)
        for entry, expected in zip(report["matrix"], ([BASELINE], [MODEL], [self.special])):
            self.assertEqual(
                json.loads(base64.urlsafe_b64decode(entry["unit_test_files"])), expected
            )
        self.runner.assert_called_once_with(
            [
                "impacted-tests",
                "--module",
                "megatron",
                "--tests-dir",
                "tests",
                "--root-dir",
                str(self.root),
                "--git-mode",
                "branch",
                "--base-branch",
                "1234567",
            ],
            self.root,
        )

    def test_empty_analysis_cannot_be_rescued_by_baseline_or_changed_test(self) -> None:
        self._write(f"{UNIT_ROOT}/conftest.py")
        for output in (
            "",
            "megatron/core/helper_test.py\n",
            f"{UNIT_ROOT}/conftest.py\n",
            BASELINE,
        ):
            with self.subTest(output=output):
                self.runner.return_value = CommandResult(0, output, "")
                self._assert_full(
                    self._select([Change("M", SOURCE), Change("M", BASELINE)]),
                    "returned no unit tests for an analyzable change",
                )
        self.runner.return_value = CommandResult(0, "", "")
        report = self._select([Change("M", MODEL)])
        self.assertEqual(report["selected_files"], sorted([BASELINE, MODEL]))

    def test_all_matching_patterns_and_documentation_use_baseline_only(self) -> None:
        changes = [
            Change("A", "tests/functional_tests/nested/golden.json"),
            Change("D", "skills/example/scripts/removed.py"),
            Change("M", ".github/workflows/claude_review.yml"),
        ]
        cases = [[change] for change in changes] + [changes, [Change("M", "docs/guide.md")]]
        cases += [
            [Change(status, "tests/functional_tests/new.py", "skills/old.py")]
            for status in ("R", "C")
        ]
        for changes in cases:
            with self.subTest(changes=changes):
                report = self._select(changes)
                self.assertEqual(report["selected_files"], [BASELINE])
                self.assertEqual(report["mode"], "selective")
                self.assertEqual(report["impacted_count"], 0)
                self.assertEqual(report["impact_analysis"]["status"], "not_run")
                self.assertEqual(
                    report["changed_files"], sorted({p for c in changes for p in c.paths})
                )
                self.assertEqual(report["matrix"][0]["unit_test_files"], _encode([BASELINE]))
        self.runner.assert_not_called()

    def test_pattern_matching_rejects_incomplete_or_unsafe_paths(self) -> None:
        cases = [
            [],
            [Change("S", "skills/a.py")],
            [Change("T", "skills/a.py")],
            [Change("R", "skills/new.py")],
            [Change("C", "skills/new.py")],
            [Change("M", "skills/new.py", "skills/old.py")],
        ]
        cases += [
            [Change("M", path)]
            for path in (
                "",
                "/skills/a.py",
                "skills/../megatron/core/layers.py",
                "skills/./a.py",
                "skills//a.py",
                "skills/a.py/",
                "skills/..\\outside.py",
                "skills/a.py\n",
                "tests/functional_tests_extra/a.py",
                ".github/workflows/claude_review.yml.bak",
            )
        ]
        for changes in cases:
            with self.subTest(changes=changes):
                self.assertFalse(_only_always_run_changes(changes))

    def test_mixed_changes_and_boundary_renames_keep_normal_policy(self) -> None:
        for status in ("R", "C"):
            for old, new in (("skills/old.py", SOURCE), (SOURCE, "skills/new.py")):
                with self.subTest(status=status, old=old):
                    self._assert_full(
                        self._select([Change(status, new, old)]), f"git status '{status}'"
                    )
        for path, reason in (
            ("tests/functional_tests/golden.json", "unsupported non-Python file"),
            ("skills/example.py", "Python file outside the analyzed package"),
            (".github/workflows/claude_review.yml", "high-impact file"),
        ):
            with self.subTest(path=path):
                self._assert_full(self._select([Change("M", SOURCE), Change("M", path)]), reason)
        self.runner.assert_not_called()
        report = self._select([Change("M", SOURCE), Change("M", "skills/guide.md")])
        self.assertEqual(report["selected_files"], sorted([BASELINE, MODEL]))
        self.runner.assert_called_once()

    def test_full_overrides_win_without_analyzing_matching_patterns(self) -> None:
        for override in ("force_full", "execute_full"):
            with self.subTest(override=override):
                self._assert_full(
                    self._select(
                        [Change("M", "skills/guide.md")],
                        record_impact=True,
                        **{override: "explicit full suite"},
                    ),
                    "explicit full suite",
                )
        self._assert_full(self._select(force_full="non-PR full suite"), "non-PR full suite")
        self.runner.assert_not_called()

    def test_baseline_configuration_is_validated_before_selection(self) -> None:
        for contents in (
            None,
            "not JSON",
            "[]",
            "{}",
            "[42]",
            json.dumps([BASELINE, BASELINE]),
            '["tests/unit_tests/test_missing.py"]',
            '["tests/unit_tests/models/../test_root.py"]',
            '["megatron/core/layers.py"]',
        ):
            with self.subTest(contents=contents):
                path = self.root / UNIT_ROOT / "always_run_tests.json"
                if contents is None:
                    path.unlink()
                else:
                    self._write(str(path), contents)
                for changes in ([Change("M", SOURCE)], [Change("M", "skills/guide.md")]):
                    self._assert_full(
                        self._select(changes), "always-run test configuration is invalid"
                    )
        self.runner.assert_not_called()
        ownership = build_bucket_ownership(REPO_ROOT, [f"{UNIT_ROOT}/**"])
        self.assertEqual(
            len(
                read_always_run_tests(
                    REPO_ROOT, REPO_ROOT / UNIT_ROOT / "always_run_tests.json", ownership
                )
            ),
            10,
        )

    def test_high_impact_and_unsupported_changes_fail_closed(self) -> None:
        cases = [([], "no changed files")]
        cases += [
            ([Change(status, SOURCE)], f"git status '{status}'") for status in ("D", "S", "T")
        ]
        cases += [
            ([Change("M", path)], "high-impact file")
            for path in (
                ".github/workflows/cicd-main.yml",
                "docker/Dockerfile",
                "uv.lock",
                "megatron/__init__.py",
                "megatron/core/models/gpt/__init__.py",
                "megatron/core/tokenizers/text/libraries/null_tokenizer.py",
                f"{UNIT_ROOT}/conftest.py",
                f"{UNIT_ROOT}/models/conftest.py",
                f"{UNIT_ROOT}/always_run_tests.json",
                f"{UNIT_ROOT}/models/__init__.py",
            )
        ]
        cases += [
            ([Change("M", path)], reason)
            for path, reason in (
                ("megatron/core/kernel.cu", "unsupported non-Python"),
                ("tools/generate.py", "outside the analyzed package"),
                ("megatron/experimental/kernel.py", "cannot discover namespace path"),
            )
        ]
        for changes, reason in cases:
            with self.subTest(changes=changes):
                self._assert_full(self._select(changes), reason)
        self.runner.assert_not_called()
        self.assertIsNone(full_run_reason(self.root, [Change("M", SOURCE), Change("A", MODEL)]))

    def test_analyzer_failures_and_invalid_output_fail_closed(self) -> None:
        cases = [(CommandResult(7, "", "failed"), "failed with exit code 7")]
        cases += [
            (CommandResult(0, MODEL, diagnostic), "reported an analysis error")
            for diagnostic in (
                "ERROR: import failed",
                "Traceback (most recent call last):",
                "module could not be resolved to a\n known module",
                "Syntax error while parsing source",
                "module not found in discovered submodules",
            )
        ]
        cases += [
            (CommandResult(0, path, ""), reason)
            for path, reason in (
                (f"{UNIT_ROOT}/test_missing.py", "missing test file"),
                (f"{UNIT_ROOT}/conftest.py\n{MODEL}", "missing test file"),
                ("../outside/test_escape.py", "outside the repository"),
            )
        ]
        for result, reason in cases:
            with self.subTest(result=result):
                self.runner.return_value = result
                self._assert_full(self._select(), reason)
        self.runner.side_effect = FileNotFoundError("impacted-tests unavailable")
        self._assert_full(self._select(), "could not run")
        self.runner.side_effect = None
        with mock.patch(
            "select_unit_tests.subprocess.run",
            side_effect=subprocess.TimeoutExpired("impacted-tests", 300),
        ):
            self.runner.return_value = _run_command(["impacted-tests"], self.root)
        self._assert_full(self._select(), "failed with exit code 124")

    def test_invalid_python_inputs_cannot_silently_remove_dependency_edges(self) -> None:
        helper = self._write(f"{UNIT_ROOT}/models/helpers.py")
        for contents in (b"from megatron.core.layers import Layer\ndef invalid(:\n", b"# \xff\n"):
            with self.subTest(contents=contents):
                helper.write_bytes(contents)
                self._assert_full(self._select(), "cannot analyze Python dependency input")
        self.runner.assert_not_called()

    def test_bucket_partition_and_invalid_ownership(self) -> None:
        ownership = build_bucket_ownership(self.root, self.buckets)
        self.assertEqual(
            ownership,
            {
                self.buckets[0]: {BASELINE, f"{UNIT_ROOT}/data/test_data.py"},
                self.buckets[1]: {MODEL},
                self.buckets[2]: {self.special},
            },
        )
        for buckets, reason in (
            ([self.buckets[1]], "not owned by a recipe bucket"),
            (
                [self.buckets[0], f"{UNIT_ROOT}/models/test_*.py", f"{UNIT_ROOT}/models/*.py"],
                "ambiguous",
            ),
            ([self.buckets[0], f"{UNIT_ROOT}/missing/**"], "do not own any tests"),
        ):
            with self.subTest(buckets=buckets), self.assertRaisesRegex(ValueError, reason):
                build_bucket_ownership(self.root, buckets)
        for buckets in ([], ["tests/unit_tests/../../test_escape.py"], [self.buckets[0]] * 2):
            with self.subTest(buckets=buckets), self.assertRaises(ValueError):
                read_buckets(self._write("buckets.json", json.dumps(buckets)))

    def test_git_comparison_requires_exact_ancestor_and_rejects_staged_changes(self) -> None:
        base = self._init_git()
        self._write(SOURCE, "changed = True\n")
        self._git("add", SOURCE)
        self.assertEqual(
            find_changes(self.root, "unstaged", None), [Change("S", "<staged changes>")]
        )
        self._commit()
        self.assertEqual(find_changes(self.root, "branch", base), [Change("M", SOURCE)])
        with self.assertRaisesRegex(ValueError, "exact commit SHA"):
            find_changes(self.root, "branch", "HEAD~1")
        unrelated = self._git(
            "commit-tree", self._git("rev-parse", "HEAD^{tree}"), "-m", "unrelated"
        )
        with self.assertRaisesRegex(ValueError, "not an ancestor"):
            find_changes(self.root, "branch", unrelated)

    @unittest.skipUnless(
        shutil.which("impacted-tests"), "requires the selective-testing environment"
    )
    def test_real_analyzer_follows_transitive_imports_without_expanding_unchanged_fixtures(
        self,
    ) -> None:
        from pytest_impacted._rust import RUST_AVAILABLE

        if not RUST_AVAILABLE:
            self.skipTest("requires pytest-impacted[fast]")
        self._write("megatron/core/__init__.py", "raise RuntimeError('must not import source')\n")
        self._write("megatron/core/leaf.py", "VALUE = 1\n")
        self._write(SOURCE, "from .leaf import VALUE\n")
        helper = f"{UNIT_ROOT}/models/helpers.py"
        self._write(helper, "from megatron.core.layers import VALUE\n")
        self._write(MODEL, "from tests.unit_tests.models.helpers import VALUE\n")
        fixtures = [f"{UNIT_ROOT}/conftest.py", f"{UNIT_ROOT}/models/conftest.py"]
        for path in fixtures:
            self._write(path, "from tests.unit_tests.models.helpers import VALUE\n")
        base = self._init_git()
        for path, contents in (
            ("megatron/core/leaf.py", "VALUE = 2\n"),
            (helper, "from megatron.core.layers import VALUE\nRESULT = VALUE + 1\n"),
        ):
            with self.subTest(path=path):
                self._write(path, contents)
                head = self._commit()
                changes = find_changes(self.root, "branch", base)
                self.assertEqual(changes, [Change("M", path)])
                runner = mock.Mock(wraps=_run_command)
                report = select_unit_tests(
                    self.root, self.buckets, changes, "branch", base, runner=runner
                )
                self.assertEqual(report["mode"], "selective", report["reason"])
                self.assertEqual(report["impacted_files"], [MODEL])
                self.assertEqual(report["selected_files"], sorted([BASELINE, MODEL]))
                raw = {
                    (self.root / p).resolve().relative_to(self.root).as_posix()
                    for p in report["impact_analysis"]["raw_selected_files"]
                }
                self.assertTrue(set(fixtures + [MODEL]).issubset(raw), raw)
                runner.assert_called_once()
                base = head


@unittest.skipUnless(shutil.which("jq"), "workflow policy requires jq")
class TestWorkflow(TemporaryRepository):
    def setUp(self) -> None:
        super().setUp()
        self.workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/cicd-main.yml").read_text())
        self.environment = {
            **os.environ,
            "PATH": f"{self.bin}{os.pathsep}{os.environ['PATH']}",
            "GITHUB_OUTPUT": str(self.root / "output"),
            "GITHUB_STEP_SUMMARY": str(self.root / "summary"),
            "IS_PR": "true",
            "IS_MERGE_GROUP": "false",
            "IS_CI_WORKLOAD": "false",
            "FORCE_RUN_ALL": "false",
            "EVENT_NAME": "push",
            "PR_NUMBER": "123",
            "REPOSITORY": "NVIDIA/Megatron-LM",
            "TEST_LABELS": "[]",
            "SELECTIVE_TESTS": "true",
            "UNIT_TEST_REASON": "PR-wide impact analysis plus baseline",
        }
        self.environment.pop("BASE_SHA", None)
        self._executable(
            "gh",
            """
            import os, sys
            if os.environ.get('TEST_LABEL_FAILURE'): sys.exit(1)
            print(os.environ['TEST_LABELS'] if sys.argv[1] == 'pr' else '0')
        """,
        )
        self._executable("yq", 'print(\'[{"bucket": "tests/unit_tests/**/*.py"}]\')')
        self._executable("timeout", "import os, sys\nos.execvp(sys.argv[4], sys.argv[4:])")
        self._executable(
            "uv",
            """
            import os, sys
            from pathlib import Path
            Path('uv-called').touch()
            if os.environ.get('TEST_SELECTOR_FAILURE'): sys.exit(1)
            args = sys.argv[sys.argv.index('python'):]
            os.execvp(args[0], args)
        """,
        )
        self._executable(
            "python",
            """
            import json, os, sys
            from pathlib import Path
            args = sys.argv[2:]
            Path('selector-args.json').write_text(json.dumps(args))
            full = '--force-full' in args
            reason = args[args.index('--force-full') + 1] if full else 'impacted plus baseline'
            matrix = [{'bucket': 'tests/unit_tests/**/*.py', 'unit_test_files': '' if full else 'selected'}]
            if os.environ.get('TEST_INVALID_MATRIX') and not full: matrix = []
            Path(args[args.index('--output') + 1]).write_text(json.dumps({
                'mode': 'full' if full else 'selective', 'matrix': matrix, 'reason': reason,
            }))
            Path(args[args.index('--summary') + 1]).write_text(reason)
        """,
        )

    def _step(self, job: str, name: str) -> dict:
        return next(
            step
            for step in self.workflow["jobs"][job]["steps"]
            if step.get("name", step.get("id")) == name
        )

    def _run_step(
        self, job="cicd-parse-unit-tests", name="Parse unit tests", **environment
    ) -> dict:
        output = self._write("output")
        (self.root / "uv-called").unlink(missing_ok=True)
        result = subprocess.run(
            ["bash", "-euo", "pipefail", "-c", self._step(job, name)["run"]],
            cwd=self.root,
            env={**self.environment, **environment},
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return dict(line.split("=", 1) for line in output.read_text().splitlines())

    def _merge_commit(self) -> str:
        self._write("README.md", "base\n")
        self.branch_point = self._init_git()
        self._git("checkout", "--quiet", "-b", "feature")
        for name in ("first", "second"):
            self._write(f"megatron/core/pr_{name}.py", f"{name} = True\n")
            self._commit(name)
        self._git("checkout", "--quiet", "--detach", self.branch_point)
        self._write("megatron/core/main_only.py", "main = True\n")
        base = self._commit("main advances")
        self._git("merge", "--quiet", "--no-ff", "feature", "-m", "synthetic PR merge")
        return base

    def test_label_precedence_and_non_pr_events_control_actual_selector_launch(self) -> None:
        self._merge_commit()
        cases = [
            ([], {}, False),
            (["Run tests"], {}, False),
            (["Run functional tests"], {}, False),
            (["Run selective unit tests on latest commit"], {}, False),
        ]
        cases += [
            (["Run selective unit tests", *extra], {}, True)
            for extra in ([], ["Run tests"], ["Run functional tests"])
        ]
        cases += [(["Run selective unit tests", "Run full unit tests"], {}, False)]
        cases += [
            (["Run selective unit tests"], env, False)
            for env in (
                {"FORCE_RUN_ALL": "true"},
                {"TEST_LABEL_FAILURE": "true"},
                {"IS_PR": "false", "IS_MERGE_GROUP": "true", "EVENT_NAME": "merge_group"},
                {"IS_PR": "false", "IS_CI_WORKLOAD": "true", "EVENT_NAME": "schedule"},
                {"IS_PR": "false", "EVENT_NAME": "workflow_dispatch"},
                {"IS_PR": "false", "EVENT_NAME": "push"},
            )
        ]
        for labels, environment, selective in cases:
            with self.subTest(labels=labels, environment=environment):
                config = self._run_step(
                    "configure", "Configure", TEST_LABELS=json.dumps(labels), **environment
                )
                self.assertEqual(config["selective_tests"], str(selective).lower())
                outputs = self._run_step(
                    SELECTIVE_TESTS=config["selective_tests"],
                    UNIT_TEST_REASON=config["unit_test_reason"],
                    **environment,
                )
                self.assertEqual(
                    json.loads(outputs["unit-tests"])[0]["unit_test_files"],
                    "selected" if selective else "",
                )
                self.assertEqual((self.root / "uv-called").exists(), selective)
                args = json.loads((self.root / "selector-args.json").read_text())
                self.assertEqual("--force-full" in args, not selective)
                self.assertNotIn("--record-impact", args)
                self.assertNotIn("--execute-full", args)

    def test_comparison_uses_tested_merge_parent_and_all_pr_commits(self) -> None:
        base = self._merge_commit()
        head = self._git("rev-parse", "HEAD")
        self._run_step(BASE_SHA=self.branch_point)
        args = json.loads((self.root / "selector-args.json").read_text())
        self.assertEqual(args[args.index("--base-ref") + 1], base)
        self.assertEqual(
            self._git("diff", "--name-only", base, "HEAD").splitlines(),
            ["megatron/core/pr_first.py", "megatron/core/pr_second.py"],
        )
        self.assertEqual(self._git("rev-parse", "HEAD"), head)

    def test_selector_failure_or_invalid_matrix_falls_back_to_full(self) -> None:
        self._merge_commit()
        for environment, reason in (
            ({"TEST_SELECTOR_FAILURE": "true"}, "could not be installed, completed, or started"),
            ({"TEST_INVALID_MATRIX": "true"}, "invalid or empty matrix"),
        ):
            with self.subTest(environment=environment):
                outputs = self._run_step(**environment)
                self.assertEqual(json.loads(outputs["unit-tests"])[0]["unit_test_files"], "")
                report = json.loads((self.root / "unit-test-selection.json").read_text())
                self.assertEqual(report["mode"], "full")
                self.assertIn(reason, report["reason"])

    def test_missing_comparison_history_falls_back_without_analyzer(self) -> None:
        base = self._merge_commit()
        head = self._git("rev-parse", "HEAD")
        self._git("checkout", "--quiet", "--detach", base)
        for missing_parent in (False, True):
            with self.subTest(missing_parent=missing_parent):
                if missing_parent:
                    self._git("checkout", "--quiet", "--detach", head)
                    (self.root / ".git/objects" / base[:2] / base[2:]).unlink()
                outputs = self._run_step()
                self.assertEqual(json.loads(outputs["unit-tests"])[0]["unit_test_files"], "")
                args = json.loads((self.root / "selector-args.json").read_text())
                self.assertIn("--force-full", args)
                self.assertNotIn("--base-ref", args)
                self.assertFalse((self.root / "uv-called").exists())


class TestRuntime(TemporaryRepository):
    def setUp(self) -> None:
        super().setUp()
        self.selected = f"{UNIT_ROOT}/test_selected.py"
        self.command_log = self.root / "commands.jsonl"
        self._write("tests/__init__.py")
        self._write(f"{UNIT_ROOT}/__init__.py")
        # Extract only the existing empty-phase hook; never import GPU conftest.
        source = ast.parse((REPO_ROOT / UNIT_ROOT / "conftest.py").read_text())
        hook = next(
            node
            for node in source.body
            if isinstance(node, ast.FunctionDef) and node.name == "pytest_sessionfinish"
        )
        self._write(
            f"{UNIT_ROOT}/conftest.py",
            "def pytest_addoption(parser):\n    parser.addoption('--experimental', action='store_true')\n"
            + ast.unparse(hook)
            + "\n",
        )
        self._write(
            "pytest.ini",
            """
            [pytest]
            markers =
                experimental: experimental suite
                internal: unavailable in legacy
                flaky: unavailable in lts
                flaky_in_dev: unavailable in dev
                launch_on_gb200: enabled on gb200
        """,
        )
        self._write(self.selected, "def test_selected(): pass\n")
        self._write(f"{UNIT_ROOT}/test_unselected.py", "def test_unselected(): assert False\n")
        shutil.copyfile(
            REPO_ROOT / UNIT_ROOT / "selective_test_guard.py",
            self.root / UNIT_ROOT / "selective_test_guard.py",
        )
        source = (REPO_ROOT / UNIT_ROOT / "run_ci_test.sh").read_text()
        self.runner = self._write(
            f"{UNIT_ROOT}/run_ci_test.sh",
            source.replace("/opt/megatron-lm-legacy/", str(self.root)).replace(
                "/opt/megatron-lm", str(self.root)
            ),
        )
        self._write(
            f"{UNIT_ROOT}/find_test_cases.py",
            """
            import os
            if os.environ.get('FAIL_BUCKET_LOOKUP'): raise SystemExit(3)
            print('--ignore=tests/unit_tests/test_unselected.py')
        """,
        )
        # Run real pytest with the actual shell arguments, but no torch/GPU.
        # torchrun maps a failing worker (including collection errors) to exit 1.
        self._executable(
            "uv",
            """
            import json, os, subprocess, sys
            with open(os.environ['COMMAND_LOG'], 'a') as stream:
                stream.write(json.dumps(sys.argv[1:]) + '\\n')
            result = subprocess.run([sys.executable, '-m', 'pytest', *sys.argv[sys.argv.index('pytest') + 1:]])
            raise SystemExit(0 if result.returncode == 0 else 1)
        """,
        )
        self._executable(
            "coverage",
            """
            import json, os, sys
            with open(os.environ['COMMAND_LOG'], 'a') as stream:
                stream.write(json.dumps(['coverage', *sys.argv[1:]]) + '\\n')
        """,
        )
        (self.bin / "python").symlink_to(sys.executable)

    def _run(
        self,
        files=None,
        *,
        payload=None,
        tag="latest",
        environment="dev",
        platform="h100",
        repeat=1,
        **extra_env,
    ):
        self.command_log.unlink(missing_ok=True)
        bash = (
            "/opt/homebrew/bin/bash"
            if sys.platform == "darwin" and Path("/opt/homebrew/bin/bash").exists()
            else "bash"
        )
        return subprocess.run(
            [
                bash,
                str(self.runner),
                "--tag",
                tag,
                "--environment",
                environment,
                "--platform",
                platform,
                "--bucket",
                f"{UNIT_ROOT}/**/*.py",
                "--unit-test-repeat",
                str(repeat),
                "--log-dir",
                str(self.root / "logs"),
            ],
            cwd=self.root,
            env={
                **os.environ,
                "PATH": f"{self.bin}{os.pathsep}{os.environ['PATH']}",
                "PYTHONPATH": str(self.root),
                "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
                "COMMAND_LOG": str(self.command_log),
                "UNIT_TEST_FILES_B64": payload if payload is not None else _encode(files),
                **extra_env,
            },
            text=True,
            capture_output=True,
            timeout=30,
        )

    def _commands(self) -> list[list[str]]:
        return (
            [json.loads(line) for line in self.command_log.read_text().splitlines()]
            if self.command_log.exists()
            else []
        )

    def test_exact_files_reach_both_marker_phases_and_every_repeat(self) -> None:
        spaced = f"{UNIT_ROOT}/test_path with spaces.py"
        self._write(
            spaced, "import pytest\n@pytest.mark.experimental\ndef test_experimental(): pass\n"
        )
        files = [self.selected, spaced]
        result = self._run(files, repeat=2)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        commands = self._commands()
        self.assertEqual(len(commands), 5)
        for index, command in enumerate(commands[:-1]):
            self.assertEqual(command[-2:], files)
            self.assertIn("tests.unit_tests.selective_test_guard", command)
            self.assertEqual("--experimental" in command, index % 2 == 1)
            self.assertEqual("coverage" in command, index % 2 == 0)
            self.assertFalse(any(arg.startswith("--ignore=") for arg in command))
        self.assertEqual(commands[-1], ["coverage", "combine", "-q"])

    def test_marker_filters_allow_one_empty_phase_but_reject_zero_survivors(self) -> None:
        self._write(
            self.selected, "import pytest\n@pytest.mark.experimental\ndef test_selected(): pass\n"
        )
        result = self._run([self.selected])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(len(self._commands()), 3)
        for platform in ("h100", "gb200"):
            with self.subTest(platform=platform):
                self._write(
                    self.selected,
                    "import pytest\n@pytest.mark.flaky_in_dev\ndef test_selected(): pass\n",
                )
                result = self._run([self.selected], platform=platform)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("No selectively chosen tests survived", result.stdout)
        self._write(self.selected, "def test_selected(): pass\n")
        result = self._run([self.selected], tag="legacy", environment="lts")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("not experimental and not internal and not flaky", self._commands()[0])
        self.assertEqual(len(self._commands()), 2)

    def test_invalid_or_stale_payload_is_rejected_before_pytest(self) -> None:
        outside = self._write("test_outside.py", "def test_outside(): pass\n")
        (self.root / UNIT_ROOT / "test_symlink.py").symlink_to(outside)
        cases = [
            [],
            None,
            {},
            [42],
            [self.selected] * 2,
            [f"{UNIT_ROOT}/test_missing.py"],
            ["tests/unit_tests/../test_outside.py"],
            ["/" + self.selected],
            [self.selected + "\n"],
            [f"{UNIT_ROOT}/conftest.py"],
            [f"{UNIT_ROOT}/test_symlink.py"],
        ]
        for files in cases:
            with self.subTest(files=files):
                result = self._run(files)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("invalid or stale", result.stdout)
                self.assertEqual(self._commands(), [])
        self.assertNotEqual(self._run(payload="not-base64!").returncode, 0)
        self.assertEqual(self._commands(), [])

    def test_pytest_failures_and_collection_errors_propagate(self) -> None:
        for contents in ("def test_selected(): assert False\n", "this is invalid python!\n"):
            with self.subTest(contents=contents):
                self._write(self.selected, contents)
                self.assertNotEqual(self._run([self.selected]).returncode, 0)
                self.assertEqual(len(self._commands()), 1)

    def test_full_suite_keeps_bucket_exclusions_and_stops_on_discovery_failure(self) -> None:
        result = self._run(payload="")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        commands = self._commands()
        self.assertEqual(len(commands), 3)
        for command in commands[:-1]:
            self.assertEqual(command[-1], UNIT_ROOT)
            self.assertIn("--ignore=tests/unit_tests/test_unselected.py", command)
            self.assertNotIn("tests.unit_tests.selective_test_guard", command)
        self.assertIn("coverage", commands[0])
        self.assertNotIn("coverage", commands[1])
        self.assertEqual(commands[-1], ["coverage", "combine", "-q"])
        self.assertNotEqual(self._run(payload="", FAIL_BUCKET_LOOKUP="1").returncode, 0)
        self.assertEqual(self._commands(), [])


class TestWorkloadTransport(TemporaryRepository):
    def test_action_payload_bindings_and_setup_failure(self) -> None:
        workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/cicd-main.yml").read_text())
        jobs = workflow["jobs"]
        unit_job = jobs["cicd-unit-tests-latest"]
        for dependency in ("cicd-parse-unit-tests", "cicd-container-build"):
            self.assertIn(dependency, unit_job["needs"])
            self.assertIn(f"needs.{dependency}.result == 'success'", unit_job["if"])
        source_sha = jobs["cicd-container-build"]["with"]["source-sha"]
        for job in (jobs["cicd-parse-unit-tests"], unit_job):
            checkout = next(step for step in job["steps"] if step.get("name") == "Checkout")
            self.assertEqual(checkout["with"]["ref"], source_sha)
        self.assertEqual(
            unit_job["strategy"]["matrix"]["include"],
            "${{ fromJson(needs.cicd-parse-unit-tests.outputs.unit-tests) }}",
        )
        action_call = next(
            step for step in unit_job["steps"] if "unit_test_files" in step.get("with", {})
        )
        self.assertEqual(action_call["with"]["unit_test_files"], "${{ matrix.unit_test_files }}")
        action = yaml.safe_load((REPO_ROOT / ".github/actions/action.yml").read_text())
        step = next(
            step
            for step in action["runs"]["steps"]
            if step["name"] == "Create run-script (unit test)"
        )
        self.assertEqual(step["env"]["UNIT_TEST_FILES_B64"], "${{ inputs.unit_test_files }}")
        payload = _encode([f"{UNIT_ROOT}/test_path with spaces.py"])
        script = step["run"]
        for key, value in {
            "test_case": f"{UNIT_ROOT}/**/*.py",
            "platform": "dgx_h100",
            "tag": "latest",
            "container-image": "test-image",
        }.items():
            script = script.replace("${{ inputs." + key + " }}", value)
        self._executable(
            "uv",
            """
            import json, os, sys
            with open('commands.jsonl', 'a') as stream:
                stream.write(json.dumps(sys.argv[1:]) + '\\n')
            if os.environ.get('FAIL_SYNC') and sys.argv[1] == 'sync': raise SystemExit(17)
        """,
        )
        env = {
            **os.environ,
            "PATH": f"{self.bin}{os.pathsep}{os.environ['PATH']}",
            "UNIT_TEST_FILES_B64": payload,
        }
        subprocess.run(
            ["bash", "-euo", "pipefail", "-c", script],
            cwd=self.root,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        for fail in (False, True):
            with self.subTest(fail=fail):
                (self.root / "commands.jsonl").unlink(missing_ok=True)
                result = subprocess.run(
                    ["bash", "job.sh"],
                    cwd=self.root,
                    env={**env, "FAIL_SYNC": "1" if fail else ""},
                    capture_output=True,
                    text=True,
                )
                commands = [
                    json.loads(line)
                    for line in (self.root / "commands.jsonl").read_text().splitlines()
                ]
                self.assertEqual(result.returncode, 17 if fail else 0, result.stderr)
                if fail:
                    self.assertEqual(commands[-1][0], "sync")
                else:
                    self.assertEqual(
                        commands[-1][commands[-1].index("--unit-test-files") + 1], payload
                    )
                    self.assertEqual(
                        commands[-1][commands[-1].index("--test-case") + 1], f"{UNIT_ROOT}/**/*.py"
                    )

    def test_cli_payload_reaches_docker_environment_unchanged(self) -> None:
        parser = types.ModuleType("recipe_parser")
        parser.load_workloads = mock.Mock(
            return_value=[
                types.SimpleNamespace(
                    type="basic",
                    spec={
                        "name": "unit-tests",
                        "test_case": f"{UNIT_ROOT}/**/*.py",
                        "script": "echo selected",
                    },
                )
            ]
        )
        scripts = types.ModuleType("tests.test_utils.python_scripts")
        scripts.recipe_parser = parser
        nemo_run = mock.MagicMock()
        nemo_run.Experiment.return_value.__enter__.return_value.status.return_value = {
            "task": {"status": "SUCCEEDED"}
        }
        modules = {
            "nemo_run": nemo_run,
            "tests": types.ModuleType("tests"),
            "tests.test_utils": types.ModuleType("tests.test_utils"),
            "tests.test_utils.python_scripts": scripts,
        }
        spec = importlib.util.spec_from_file_location(
            "selective_runtime_launcher",
            REPO_ROOT / "tests/test_utils/python_scripts/launch_nemo_run_workload.py",
        )
        module = importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules, modules):
            spec.loader.exec_module(module)
        for payload in ("", _encode([f"{UNIT_ROOT}/test_path with spaces.py"])):
            with self.subTest(payload=payload):
                args = [
                    "--scope",
                    "unit-tests",
                    "--model",
                    "unit-tests",
                    "--test-case",
                    f"{UNIT_ROOT}/**/*.py",
                    "--environment",
                    "dev",
                    "--platform",
                    "dgx_h100",
                    "--container-image",
                    "test-image",
                ]
                if payload:
                    args.extend(["--unit-test-files", payload])
                result = CliRunner().invoke(module.main, args)
                self.assertEqual(result.exit_code, 0, result.output or repr(result.exception))
                self.assertEqual(
                    nemo_run.DockerExecutor.call_args.kwargs["env_vars"]["UNIT_TEST_FILES_B64"],
                    payload,
                )


if __name__ == "__main__":
    unittest.main()
