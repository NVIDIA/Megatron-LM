#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Exercise the workflow's actual shell policy without GitHub or GPU jobs."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

WORKFLOW = Path(__file__).resolve().parents[1] / "workflows/cicd-main.yml"


@unittest.skipUnless(shutil.which("jq"), "workflow policy requires jq")
class TestUnitTestWorkflow(unittest.TestCase):
    def setUp(self):
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.root = Path(temporary_directory.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.workflow = yaml.safe_load(WORKFLOW.read_text())
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
            "UNIT_TEST_REASON": "PR-wide impact analysis plus the always-run baseline",
            "BASE_SHA": "",
        }
        self._command(
            "gh",
            """
import os, sys
if os.environ.get('TEST_LABEL_FAILURE'):
    sys.exit(1)
print(os.environ['TEST_LABELS'] if sys.argv[1] == 'pr' else '0')
""",
        )
        self._command("yq", 'print(\'[{"bucket": "tests/unit_tests/**/*.py"}]\')')
        self._command("timeout", "import os, sys\nos.execvp(sys.argv[4], sys.argv[4:])")
        self._command(
            "uv",
            """
import os, sys
if os.environ.get('TEST_SELECTOR_FAILURE'):
    sys.exit(1)
args = sys.argv[sys.argv.index('python'):]
os.execvp(args[0], args)
""",
        )
        self._command(
            "python",
            """
import json, os, sys
from pathlib import Path
args = sys.argv[2:]
Path('selector-args.json').write_text(json.dumps(args))
full = '--force-full' in args
matrix = [{'bucket': 'tests/unit_tests/**/*.py', 'unit_test_files': '' if full else 'selected'}]
if os.environ.get('TEST_INVALID_MATRIX') and not full:
    matrix = []
Path(args[args.index('--output') + 1]).write_text(json.dumps({'matrix': matrix}))
Path(args[args.index('--summary') + 1]).write_text('Full suite' if full else 'Selected tests')
""",
        )

    def _command(self, name, script):
        path = self.bin / name
        path.write_text(f"#!{sys.executable}\n{script}\n")
        path.chmod(0o755)

    def _step(self, job, name):
        return next(
            step
            for step in self.workflow["jobs"][job]["steps"]
            if step.get("name", step.get("id")) == name
        )

    def _run(self, job, name, **environment):
        output_path = self.root / "output"
        output_path.write_text("")
        result = subprocess.run(
            ["bash", "-e", "-u", "-o", "pipefail", "-c", self._step(job, name)["run"]],
            cwd=self.root,
            env={**self.environment, **environment},
            capture_output=True,
            text=True,
        )
        outputs = dict(line.split("=", 1) for line in output_path.read_text().splitlines())
        return result, outputs

    def _configure(self, labels=(), **environment):
        result, outputs = self._run(
            "configure", "Configure", TEST_LABELS=json.dumps(labels), **environment
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return outputs

    def _git(self, *arguments):
        return subprocess.run(
            ["git", *arguments], cwd=self.root, check=True, capture_output=True, text=True
        ).stdout.strip()

    def _merge_commit(self):
        self._git("init", "--quiet")
        self._git("config", "user.name", "Workflow test")
        self._git("config", "user.email", "workflow@example.com")
        self._git("config", "commit.gpgsign", "false")
        self._git("commit", "--quiet", "--allow-empty", "-m", "base")
        base = self._git("rev-parse", "HEAD")
        self._git("checkout", "--quiet", "-b", "feature")
        self._git("commit", "--quiet", "--allow-empty", "-m", "first change")
        self._git("commit", "--quiet", "--allow-empty", "-m", "second change")
        self._git("checkout", "--quiet", "--detach", base)
        self._git("merge", "--quiet", "--no-ff", "feature", "-m", "synthetic PR merge")
        self.environment["BASE_SHA"] = base
        return base

    def test_pr_without_opt_in_runs_full_units_and_preserves_functional_scope(self):
        cases = [
            ([], "L0", "2", "false"),
            (["Run tests"], "L1", "1", "true"),
            (["Run functional tests"], "L1", "5", "false"),
            (["Run selective unit tests on latest commit"], "L0", "2", "false"),
        ]
        for labels, scope, repeats, lightweight in cases:
            with self.subTest(labels=labels):
                outputs = self._configure(labels)
                self.assertEqual(outputs["selective_tests"], "false")
                self.assertEqual(outputs["selective_label_present"], "false")
                self.assertEqual(outputs["scope"], scope)
                self.assertEqual(outputs["n_repeat"], repeats)
                self.assertEqual(outputs["lightweight"], lightweight)

    def test_opt_in_label_enables_selection_with_functional_labels(self):
        for extra_labels in ([], ["Run tests"], ["Run functional tests"]):
            with self.subTest(extra_labels=extra_labels):
                outputs = self._configure(["Run selective unit tests", *extra_labels])
                self.assertEqual(outputs["selective_tests"], "true")
                self.assertEqual(outputs["selective_label_present"], "true")

    def test_full_unit_override_takes_precedence_over_opt_in(self):
        for labels in (
            ["Run full unit tests"],
            ["Run full unit tests", "Run selective unit tests"],
        ):
            with self.subTest(labels=labels):
                outputs = self._configure(labels)
                self.assertEqual(outputs["selective_tests"], "false")
                self.assertEqual(outputs["scope"], "L0")
                self.assertEqual(outputs["cadence_bypass"], "false")
                self.assertEqual(outputs["selective_label_present"], str(len(labels) == 2).lower())

    def test_non_pr_triggers_and_preflight_override_use_full_units(self):
        cases = [
            {"IS_PR": "false", "IS_MERGE_GROUP": "true", "EVENT_NAME": "merge_group"},
            {"IS_PR": "false", "IS_CI_WORKLOAD": "true", "EVENT_NAME": "schedule"},
            {"IS_PR": "false", "EVENT_NAME": "workflow_dispatch"},
            {"IS_PR": "false", "EVENT_NAME": "push"},
            {"FORCE_RUN_ALL": "true"},
            {"TEST_LABEL_FAILURE": "true"},
        ]
        for environment in cases:
            with self.subTest(environment=environment):
                outputs = self._configure(["Run selective unit tests"], **environment)
                self.assertEqual(outputs["selective_tests"], "false")
                if environment.get("TEST_LABEL_FAILURE"):
                    self.assertEqual(outputs["selective_label_present"], "unknown")

    def test_selection_and_image_build_are_independent_prerequisites_of_execution(self):
        jobs = self.workflow["jobs"]

        def ancestors(job):
            pending = list(jobs[job].get("needs", []))
            seen = set()
            while pending:
                dependency = pending.pop()
                if dependency not in seen:
                    seen.add(dependency)
                    pending.extend(jobs[dependency].get("needs", []))
            return seen

        selection = "cicd-parse-unit-tests"
        build = "cicd-container-build"
        execution = "cicd-unit-tests-latest"
        self.assertNotIn(build, ancestors(selection))
        self.assertNotIn(selection, ancestors(build))
        for job in (selection, build):
            self.assertIn("cicd-wait-in-queue", ancestors(job))
            self.assertIn(job, ancestors(execution))
            self.assertIn(f"needs.{job}.result == 'success'", jobs[execution]["if"])
        source_sha = jobs[build]["with"]["source-sha"]
        self.assertEqual(self._step(selection, "Checkout")["with"]["ref"], source_sha)
        self.assertEqual(self._step(execution, "Checkout")["with"]["ref"], source_sha)

    def test_selection_compares_entire_pr_and_keeps_the_tested_merge_checkout(self):
        base = self._merge_commit()
        tested_merge = self._git("rev-parse", "HEAD")
        result, outputs = self._run("cicd-parse-unit-tests", "Parse unit tests")
        self.assertEqual(result.returncode, 0, result.stderr)
        args = json.loads((self.root / "selector-args.json").read_text())
        self.assertEqual(args[args.index("--base-ref") + 1], base)
        self.assertEqual(self._git("rev-parse", "HEAD"), tested_merge)
        self.assertEqual(json.loads(outputs["unit-tests"])[0]["unit_test_files"], "selected")

    def test_ambiguous_base_or_selector_failure_falls_back_to_full_matrix(self):
        self._merge_commit()
        cases = [
            {"BASE_SHA": ""},
            {"BASE_SHA": "a" * 40},
            {"TEST_SELECTOR_FAILURE": "true"},
            {"TEST_INVALID_MATRIX": "true"},
            {"SELECTIVE_TESTS": "false", "UNIT_TEST_REASON": "full suite requested"},
        ]
        for environment in cases:
            with self.subTest(environment=environment):
                result, outputs = self._run(
                    "cicd-parse-unit-tests", "Parse unit tests", **environment
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads(outputs["unit-tests"])[0]["unit_test_files"], "")
                args = json.loads((self.root / "selector-args.json").read_text())
                self.assertIn("--force-full", args)

    def test_docs_only_still_requires_unit_success_but_skips_training(self):
        result, outputs = self._run("cicd-integration-gate", "gate", DOCS_ONLY="true")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(outputs["should_run"], "false")
        for unit_result, expected_status in [("success", 0), ("skipped", 1), ("failure", 1)]:
            with self.subTest(unit_result=unit_result):
                result, _ = self._run(
                    "Nemo_CICD_Test",
                    "Get workflow result",
                    DOCS_ONLY="true",
                    IS_DEPLOYMENT="false",
                    IS_MAINTAINER="true",
                    ENABLE_GB200_TESTING="true",
                    UNIT_RESULT=unit_result,
                    UNIT_GB200_RESULT="success",
                    H100_RESULT="skipped",
                    GB200_RESULT="skipped",
                    GITHUB_RUN_ID="123",
                )
                self.assertEqual(result.returncode, expected_status, result.stderr)
        jobs = self.workflow["jobs"]
        self.assertNotIn("docs_only", jobs["linting"]["if"])
        self.assertNotIn("docs_only", jobs["cicd-wait-in-queue"]["if"])
        self.assertEqual(jobs["cicd-wait-in-queue"]["environment"], "test")

    def test_collection_step_records_data_with_and_without_label(self):
        reporter = WORKFLOW.parents[1] / "scripts/report_unit_test_selection.py"
        destination = self.root / ".github/scripts/report_unit_test_selection.py"
        destination.parent.mkdir(parents=True)
        shutil.copyfile(reporter, destination)
        plan_dir = self.root / "unit-test-plan"
        plan_dir.mkdir()
        bucket = "tests/unit_tests/**/*.py"
        jobs = {
            "jobs": [
                {
                    "databaseId": 123,
                    "name": f"{bucket} - latest",
                    "status": "completed",
                    "conclusion": "success",
                    "startedAt": "2026-09-09T10:00:00Z",
                    "completedAt": "2026-09-09T10:00:20Z",
                }
            ]
        }
        self._command("gh", f"print({json.dumps(jobs)!r})")
        for label, mode, selected in [("true", "selective", 11), ("false", "full", 460)]:
            with self.subTest(label=label):
                plan = {
                    "mode": mode,
                    "reason": "workflow regression fixture",
                    "selected_count": selected,
                    "total_count": 460,
                    "total_bucket_count": 36,
                    "matrix": [{"bucket": bucket, "unit_test_files": ""}],
                    "selector_duration_seconds": 0.1,
                    "planning_duration_seconds": 2,
                }
                (plan_dir / "unit-test-selection.json").write_text(json.dumps(plan))
                result, _ = self._run(
                    "cicd-unit-test-report",
                    "Collect unit-test comparison data",
                    SELECTIVE_LABEL_PRESENT=label,
                    RUN_ID="1234",
                    RUN_ATTEMPT="2",
                    TESTED_SHA="a" * 40,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                metrics = self.root / "unit-test-metrics"
                report = json.loads((metrics / "unit-test-metrics.json").read_text())
                self.assertEqual(report["selective_label_present"], label == "true")
                self.assertEqual(report["mode"], mode)
                self.assertEqual(report["selected_file_count"], selected)
                self.assertEqual(report["sum_job_execution_seconds"], 20)
                self.assertTrue(report["timing_complete"])
                self.assertTrue((metrics / "unit-test-metrics.csv").is_file())
                self.assertTrue((self.root / "summary").read_text())

    def test_full_and_selective_plans_are_both_uploaded_for_metrics(self):
        jobs = self.workflow["jobs"]
        upload = self._step("cicd-parse-unit-tests", "Upload unit-test selection plan")
        self.assertEqual(upload["if"], "always()")
        self.assertIn("github.run_attempt", upload["with"]["name"])
        self.assertEqual(upload["with"]["retention-days"], 90)
        reporter = jobs["cicd-unit-test-report"]
        self.assertNotIn("selective_tests", reporter["if"])
        self.assertIn("cicd-unit-tests-latest", reporter["needs"])
        collect = self._step("cicd-unit-test-report", "Collect unit-test comparison data")
        self.assertIn('--attempt "$RUN_ATTEMPT"', collect["run"])
        self.assertIn("selective_label_present", collect["env"]["SELECTIVE_LABEL_PRESENT"])


if __name__ == "__main__":
    unittest.main()
