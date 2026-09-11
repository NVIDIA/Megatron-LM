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
        }
        self.environment.pop("BASE_SHA", None)
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
import json, os, sys
from pathlib import Path
Path('uv-args.json').write_text(json.dumps(sys.argv[1:]))
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
reason = args[args.index('--force-full') + 1] if full else 'impacted plus baseline'
matrix = [{'bucket': 'tests/unit_tests/**/*.py', 'unit_test_files': '' if full else 'selected'}]
if os.environ.get('TEST_INVALID_MATRIX') and not full:
    matrix = []
report = {
    'mode': 'full' if full else 'selective', 'matrix': matrix,
    'reason': reason,
}
Path(args[args.index('--output') + 1]).write_text(json.dumps(report))
Path(args[args.index('--summary') + 1]).write_text(reason)
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
        (self.root / "uv-args.json").unlink(missing_ok=True)
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
        self._commit_file("README.md", "Common branch point\n", "base")
        self.branch_point_sha = self._git("rev-parse", "HEAD")
        self._git("checkout", "--quiet", "-b", "feature")
        self._commit_file("megatron/core/pr_first.py", "first = True\n", "first PR change")
        self._commit_file("megatron/core/pr_second.py", "second = True\n", "second PR change")
        self._git("checkout", "--quiet", "--detach", self.branch_point_sha)
        self._commit_file("megatron/core/main_only.py", "main = True\n", "main advances")
        base = self._git("rev-parse", "HEAD")
        self._git("merge", "--quiet", "--no-ff", "feature", "-m", "synthetic PR merge")
        return base

    def _commit_file(self, relative_path, contents, message):
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)
        self._git("add", relative_path)
        self._git("commit", "--quiet", "-m", message)

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
                self.assertEqual(outputs["scope"], scope)
                self.assertEqual(outputs["n_repeat"], repeats)
                self.assertEqual(outputs["lightweight"], lightweight)

    def test_opt_in_label_enables_selection_with_functional_labels(self):
        for extra_labels in ([], ["Run tests"], ["Run functional tests"]):
            with self.subTest(extra_labels=extra_labels):
                outputs = self._configure(["Run selective unit tests", *extra_labels])
                self.assertEqual(outputs["selective_tests"], "true")

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

    def test_selection_uses_tested_merge_parent_and_ignores_external_base_metadata(self):
        base = self._merge_commit()
        tested_merge = self._git("rev-parse", "HEAD")
        unrelated = self._git(
            "commit-tree", self._git("rev-parse", "HEAD^{tree}"), "-m", "unrelated"
        )
        cases = [{}, {"BASE_SHA": ""}, {"BASE_SHA": self.branch_point_sha}, {"BASE_SHA": unrelated}]
        for environment in cases:
            with self.subTest(environment=environment):
                result, outputs = self._run(
                    "cicd-parse-unit-tests", "Parse unit tests", **environment
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(
                    json.loads(outputs["unit-tests"])[0]["unit_test_files"], "selected"
                )
                args = json.loads((self.root / "selector-args.json").read_text())
                selected_base = args[args.index("--base-ref") + 1]
                self.assertEqual(selected_base, base)
                self.assertEqual(
                    self._git("diff", "--name-only", selected_base, "HEAD").splitlines(),
                    ["megatron/core/pr_first.py", "megatron/core/pr_second.py"],
                )
                self.assertEqual(self._git("rev-parse", "HEAD"), tested_merge)
                summary = (self.root / "summary").read_text()
                self.assertIn(tested_merge, summary)
                self.assertIn(base, summary)

    def test_selector_failure_falls_back_to_full_with_reason(self):
        self._merge_commit()
        cases = [
            ({"TEST_SELECTOR_FAILURE": "true"}, "could not be installed, completed, or started"),
            ({"TEST_INVALID_MATRIX": "true"}, "invalid or empty matrix"),
        ]
        for environment, reason in cases:
            with self.subTest(environment=environment):
                result, outputs = self._run(
                    "cicd-parse-unit-tests", "Parse unit tests", **environment
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads(outputs["unit-tests"])[0]["unit_test_files"], "")
                args = json.loads((self.root / "selector-args.json").read_text())
                self.assertIn("--force-full", args)
                manifest = json.loads((self.root / "unit-test-selection.json").read_text())
                self.assertEqual(manifest["mode"], "full")
                self.assertIn(reason, manifest["reason"])
                self.assertIn(reason, (self.root / "summary").read_text())

    def test_only_opted_in_prs_run_impact_analysis(self):
        base = self._merge_commit()
        setup_uv = self._step("cicd-parse-unit-tests", "Setup uv")
        self.assertEqual(setup_uv["if"], "needs.configure.outputs.selective_tests == 'true'")
        cases = [
            ([], {}, "full"),
            (["Run selective unit tests"], {}, "selective"),
            (["Run selective unit tests", "Run full unit tests"], {}, "full"),
            (["Run selective unit tests"], {"FORCE_RUN_ALL": "true"}, "full"),
            (["Run selective unit tests"], {"TEST_LABEL_FAILURE": "true"}, "full"),
        ]
        for labels, environment, expected_mode in cases:
            with self.subTest(labels=labels, environment=environment):
                config = self._configure(labels, **environment)
                result, outputs = self._run(
                    "cicd-parse-unit-tests",
                    "Parse unit tests",
                    SELECTIVE_TESTS=config["selective_tests"],
                    UNIT_TEST_REASON=config["unit_test_reason"],
                    **environment,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                args = json.loads((self.root / "selector-args.json").read_text())
                self.assertNotIn("--record-impact", args)
                self.assertNotIn("--execute-full", args)
                manifest = json.loads((self.root / "unit-test-selection.json").read_text())
                self.assertEqual(manifest["mode"], expected_mode)
                actual_matrix = json.loads(outputs["unit-tests"])
                if expected_mode == "selective":
                    uv_args = json.loads((self.root / "uv-args.json").read_text())
                    self.assertEqual(uv_args[uv_args.index("--base-ref") + 1], base)
                    self.assertNotIn("--force-full", args)
                    self.assertEqual(actual_matrix[0]["unit_test_files"], "selected")
                else:
                    self.assertFalse((self.root / "uv-args.json").exists())
                    self.assertIn("--force-full", args)
                    self.assertNotIn("--base-ref", args)
                    self.assertEqual(actual_matrix[0]["unit_test_files"], "")

    def test_non_pr_triggers_do_not_run_pr_impact_analysis(self):
        self._merge_commit()
        for event in ("merge_group", "workflow_dispatch", "schedule"):
            with self.subTest(event=event):
                result, outputs = self._run(
                    "cicd-parse-unit-tests", "Parse unit tests", IS_PR="false", EVENT_NAME=event
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads(outputs["unit-tests"])[0]["unit_test_files"], "")
                self.assertFalse((self.root / "uv-args.json").exists())
                args = json.loads((self.root / "selector-args.json").read_text())
                self.assertIn("--force-full", args)

    def test_non_merge_commit_falls_back_to_full_matrix_without_comparison_base(self):
        base = self._merge_commit()
        for ref in (base, self.branch_point_sha):
            with self.subTest(ref=ref):
                self._git("checkout", "--quiet", "--detach", ref)
                result, outputs = self._run("cicd-parse-unit-tests", "Parse unit tests")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads(outputs["unit-tests"])[0]["unit_test_files"], "")
                args = json.loads((self.root / "selector-args.json").read_text())
                self.assertIn("--force-full", args)
                self.assertNotIn("--base-ref", args)
                summary = (self.root / "unit-test-selection-summary.md").read_text()
                self.assertIn(ref, summary)
                self.assertNotIn("PR comparison:", summary)
                self.assertFalse((self.root / "uv-args.json").exists())
                self.assertEqual(self._git("rev-parse", "HEAD"), ref)

    def test_missing_merge_base_object_falls_back_to_full_matrix(self):
        base = self._merge_commit()
        tested_merge = self._git("rev-parse", "HEAD")
        # Model incomplete checkout history without replacing the real git CLI.
        (self.root / ".git" / "objects" / base[:2] / base[2:]).unlink()
        result, outputs = self._run("cicd-parse-unit-tests", "Parse unit tests")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(outputs["unit-tests"])[0]["unit_test_files"], "")
        args = json.loads((self.root / "selector-args.json").read_text())
        self.assertIn("--force-full", args)
        self.assertNotIn("--base-ref", args)
        summary = (self.root / "unit-test-selection-summary.md").read_text()
        self.assertIn(tested_merge, summary)
        self.assertNotIn("PR comparison:", summary)

    def test_docs_only_skips_tests_and_accepts_skipped_results(self):
        result, outputs = self._run(
            "cicd-integration-gate", "gate", WAIT_RESULT="skipped", UNIT_RESULT="skipped"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(outputs["should_run"], "false")
        for unit_result in ("success", "skipped"):
            with self.subTest(unit_result=unit_result):
                result, _ = self._run(
                    "Nemo_CICD_Test",
                    "Get workflow result",
                    DOCS_ONLY="true",
                    IS_DEPLOYMENT="false",
                    IS_MAINTAINER="true",
                    ENABLE_GB200_TESTING="true",
                    UNIT_RESULT=unit_result,
                    UNIT_GB200_RESULT="skipped",
                    H100_RESULT="skipped",
                    GB200_RESULT="skipped",
                    GITHUB_RUN_ID="123",
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("test checks skipped", result.stdout)
        jobs = self.workflow["jobs"]
        self.assertIn("needs.pre-flight.outputs.docs_only == 'false'", jobs["linting"]["if"])
        self.assertIn(
            "needs.pre-flight.outputs.docs_only == 'true'", jobs["cicd-wait-in-queue"]["if"]
        )
        self.assertEqual(jobs["cicd-wait-in-queue"]["environment"], "test")
        for job in ("cicd-container-build", "cicd-parse-unit-tests"):
            self.assertIn("cicd-wait-in-queue", jobs[job]["needs"])
            self.assertIn("success()", jobs[job]["if"])
        self.assertIn("needs.pre-flight.outputs.docs_only == 'true'", jobs["Coverage_Fake"]["if"])
        self.assertEqual(jobs["Coverage"]["needs"], ["Nemo_CICD_Test"])

    def test_selection_plan_is_uploaded_for_debugging(self):
        upload = self._step("cicd-parse-unit-tests", "Upload unit-test selection plan")
        self.assertEqual(upload["if"], "always()")
        self.assertIn("github.run_attempt", upload["with"]["name"])
        self.assertIn("unit-test-selection.json", upload["with"]["path"])
        self.assertIn("unit-test-selection-summary.md", upload["with"]["path"])


if __name__ == "__main__":
    unittest.main()
