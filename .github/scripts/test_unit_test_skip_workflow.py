#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Exercise the workflow's real gates when GPU unit tests are intentionally skipped."""

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github/workflows/cicd-main.yml"
SKIP_OUTPUT = "${{ needs.configure.outputs.skip_unit_tests }}"


class TestUnitTestSkipWorkflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.jobs = yaml.safe_load(WORKFLOW.read_text())["jobs"]

    def run_step(self, job_name, step_id, **overrides):
        step = next(step for step in self.jobs[job_name]["steps"] if step.get("id") == step_id)
        environment = {
            "WAIT_RESULT": "success",
            "BUILD_RESULT": "success",
            "UNIT_RESULT": "success",
            "UNIT_GB200_RESULT": "success",
            "H100_RESULT": "success",
            "GB200_RESULT": "success",
            "SKIP_UNIT_TESTS": "false",
            "IS_MERGE_GROUP": "false",
            "IS_CI_WORKLOAD": "false",
            "FORCE_RUN_ALL": "false",
            "DOCS_ONLY": "false",
            "IS_DEPLOYMENT": "false",
            "IS_MAINTAINER": "true",
            "ENABLE_GB200_TESTING": "false",
            "GITHUB_RUN_ID": "123",
            "MOCK_BAD_JOBS": "0",
            "MOCK_LABELS": "[]",
            "MOCK_LABELS_EXIT": "0",
            "EVENT_NAME": "push",
            "REF": "refs/heads/pull-request/123",
        }
        environment.update(overrides)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            output = directory / "output"
            output.touch()
            # The summary queries GitHub only to count/list failed jobs. Keep this
            # regression test offline while executing the workflow's own shell.
            gh = directory / "gh"
            gh.write_text(
                '#!/bin/bash\nif [[ "$1 $2" == "pr view" ]]; then\n'
                '  echo "$MOCK_LABELS"\n  exit "$MOCK_LABELS_EXIT"\n'
                'elif [[ "$*" == *length* ]]; then\n'
                '  echo "$MOCK_BAD_JOBS"\nelse\n  echo "mock job → failure"\nfi\n'
            )
            gh.chmod(0o755)
            script = step["run"].replace(
                "${{ fromJSON(steps.get-pr-info.outputs.pr-info || '{}').number }}", "123"
            )
            script = script.replace("${{ github.repository }}", "NVIDIA/Megatron-LM")
            result = subprocess.run(
                ["bash", "-e", "-u", "-o", "pipefail", "-c", script],
                cwd=REPO_ROOT,
                env={
                    **os.environ,
                    **environment,
                    "PATH": f"{directory}{os.pathsep}{os.environ['PATH']}",
                    "GITHUB_OUTPUT": str(output),
                    "GITHUB_STEP_SUMMARY": str(directory / "summary"),
                },
                capture_output=True,
                text=True,
                timeout=10,
            )
            outputs = dict(line.split("=", 1) for line in output.read_text().splitlines())
        return result, outputs

    @unittest.skipUnless(shutil.which("jq"), "Workflow configure shell requires jq")
    def test_skip_eligibility_respects_full_validation_overrides(self):
        cases = [
            ({}, True),
            ({"MOCK_LABELS": json.dumps(["Run functional tests"])}, True),
            ({"MOCK_LABELS": json.dumps(["Run tests"])}, False),
            ({"MOCK_LABELS": json.dumps(["container::lts"])}, False),
            ({"MOCK_LABELS": json.dumps(["force-run-all"])}, False),
            ({"FORCE_RUN_ALL": "true"}, False),
            ({"MOCK_LABELS_EXIT": "1"}, False),
            ({"EVENT_NAME": "workflow_dispatch"}, False),
            ({"EVENT_NAME": "schedule"}, False),
            ({"REF": "refs/heads/main"}, False),
            ({"REF": "refs/heads/deploy-release/test"}, False),
            ({"REF": "refs/heads/pull-request/not-a-number"}, False),
            (
                {"EVENT_NAME": "merge_group", "IS_MERGE_GROUP": "true", "MOCK_LABELS_EXIT": "1"},
                True,
            ),
            (
                {"EVENT_NAME": "merge_group", "IS_MERGE_GROUP": "true", "FORCE_RUN_ALL": "true"},
                False,
            ),
        ]
        for environment, eligible in cases:
            with self.subTest(**environment):
                result, outputs = self.run_step("configure", "configure", **environment)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(outputs["unit_test_skip_eligible"], str(eligible).lower())

    def test_integration_gate_requires_a_success_or_an_authorized_skip(self):
        cases = [
            ({"UNIT_RESULT": "success"}, True),
            ({"UNIT_RESULT": "skipped", "SKIP_UNIT_TESTS": "true"}, True),
            ({"UNIT_RESULT": "skipped", "SKIP_UNIT_TESTS": "false"}, False),
            ({"UNIT_RESULT": "skipped", "SKIP_UNIT_TESTS": ""}, False),
            ({"UNIT_RESULT": "failure", "SKIP_UNIT_TESTS": "true"}, False),
            ({"UNIT_RESULT": "cancelled", "SKIP_UNIT_TESTS": "true"}, False),
            ({"UNIT_RESULT": "", "SKIP_UNIT_TESTS": "true"}, False),
            (
                {"UNIT_RESULT": "skipped", "SKIP_UNIT_TESTS": "true", "WAIT_RESULT": "skipped"},
                False,
            ),
            (
                {"UNIT_RESULT": "skipped", "SKIP_UNIT_TESTS": "true", "WAIT_RESULT": "failure"},
                False,
            ),
            (
                {
                    "UNIT_RESULT": "skipped",
                    "SKIP_UNIT_TESTS": "true",
                    "WAIT_RESULT": "skipped",
                    "IS_MERGE_GROUP": "true",
                },
                True,
            ),
            # Preserve nightly/forced functional coverage after a unit failure.
            ({"UNIT_RESULT": "failure", "IS_CI_WORKLOAD": "true"}, True),
            ({"UNIT_RESULT": "failure", "FORCE_RUN_ALL": "true"}, True),
        ]
        for environment, should_run in cases:
            with self.subTest(**environment):
                result, outputs = self.run_step("cicd-integration-gate", "gate", **environment)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(outputs["should_run"], str(should_run).lower())

    def test_summary_accepts_only_authorized_unit_skips_and_keeps_functional_checks(self):
        cases = [
            ({}, True),
            ({"UNIT_RESULT": "skipped", "SKIP_UNIT_TESTS": "true"}, True),
            ({"UNIT_RESULT": "skipped", "SKIP_UNIT_TESTS": "false"}, False),
            ({"UNIT_RESULT": "skipped", "SKIP_UNIT_TESTS": ""}, False),
            ({"UNIT_RESULT": "failure", "SKIP_UNIT_TESTS": "true"}, False),
            ({"UNIT_RESULT": "cancelled", "SKIP_UNIT_TESTS": "true"}, False),
            ({"UNIT_RESULT": "", "SKIP_UNIT_TESTS": "true"}, False),
            (
                {"UNIT_RESULT": "skipped", "SKIP_UNIT_TESTS": "true", "H100_RESULT": "failure"},
                False,
            ),
            (
                {"UNIT_RESULT": "skipped", "SKIP_UNIT_TESTS": "true", "H100_RESULT": "skipped"},
                False,
            ),
            (
                {
                    "UNIT_RESULT": "skipped",
                    "UNIT_GB200_RESULT": "skipped",
                    "SKIP_UNIT_TESTS": "true",
                    "ENABLE_GB200_TESTING": "true",
                },
                True,
            ),
            ({"UNIT_GB200_RESULT": "skipped", "ENABLE_GB200_TESTING": "true"}, False),
            (
                {
                    "UNIT_RESULT": "skipped",
                    "UNIT_GB200_RESULT": "skipped",
                    "SKIP_UNIT_TESTS": "true",
                    "GB200_RESULT": "skipped",
                    "ENABLE_GB200_TESTING": "true",
                },
                False,
            ),
            (
                {
                    "UNIT_RESULT": "skipped",
                    "UNIT_GB200_RESULT": "skipped",
                    "SKIP_UNIT_TESTS": "true",
                    "GB200_RESULT": "skipped",
                    "ENABLE_GB200_TESTING": "true",
                    "IS_MAINTAINER": "false",
                },
                True,
            ),
            (
                {
                    "UNIT_RESULT": "skipped",
                    "SKIP_UNIT_TESTS": "true",
                    "GB200_RESULT": "failure",
                    "ENABLE_GB200_TESTING": "true",
                    "MOCK_BAD_JOBS": "1",
                },
                False,
            ),
            (
                {
                    "UNIT_RESULT": "skipped",
                    "SKIP_UNIT_TESTS": "true",
                    "UNIT_GB200_RESULT": "failure",
                    "ENABLE_GB200_TESTING": "true",
                    "MOCK_BAD_JOBS": "1",
                },
                False,
            ),
        ]
        for environment, succeeds in cases:
            with self.subTest(**environment):
                result, _ = self.run_step("Nemo_CICD_Test", "result", **environment)
                self.assertEqual(result.returncode == 0, succeeds, result.stdout + result.stderr)

    def test_intentional_unit_skip_cannot_bypass_a_missing_successful_build(self):
        for build_result in ("failure", "skipped", "cancelled", ""):
            for merge_group in ("false", "true"):
                with self.subTest(build_result=build_result, merge_group=merge_group):
                    result, outputs = self.run_step(
                        "cicd-integration-gate",
                        "gate",
                        UNIT_RESULT="skipped",
                        SKIP_UNIT_TESTS="true",
                        BUILD_RESULT=build_result,
                        IS_MERGE_GROUP=merge_group,
                        WAIT_RESULT="skipped" if merge_group == "true" else "success",
                    )
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertEqual(outputs["should_run"], "false")

    def test_skip_output_reaches_all_consumers(self):
        self.assertEqual(
            self.jobs["configure"]["outputs"]["skip_unit_tests"],
            "${{ steps.unit-test-scope.outputs.skip_unit_tests || 'false' }}",
        )
        for name in (
            "cicd-parse-unit-tests",
            "cicd-unit-tests-latest",
            "cicd-parse-unit-tests-gb200",
            "cicd-unit-tests-latest-gb200",
        ):
            with self.subTest(job=name):
                self.assertIn("configure", self.jobs[name]["needs"])
                self.assertIn(
                    "needs.configure.outputs.skip_unit_tests != 'true'", self.jobs[name]["if"]
                )
        for job_name, step_id in (("cicd-integration-gate", "gate"), ("Nemo_CICD_Test", "result")):
            with self.subTest(job=job_name):
                self.assertIn("configure", self.jobs[job_name]["needs"])
                step = next(
                    step for step in self.jobs[job_name]["steps"] if step.get("id") == step_id
                )
                self.assertEqual(step["env"]["SKIP_UNIT_TESTS"], SKIP_OUTPUT)
        integration = next(
            step for step in self.jobs["cicd-integration-gate"]["steps"] if step.get("id") == "gate"
        )
        self.assertEqual(
            integration["env"]["BUILD_RESULT"], "${{ needs.cicd-container-build.result }}"
        )

    def test_skipped_unit_tests_do_not_try_to_aggregate_missing_coverage(self):
        coverage = self.jobs["Coverage"]
        self.assertIn("configure", coverage["needs"])
        self.assertIn("pre-flight", coverage["needs"])
        self.assertIn("needs.configure.outputs.skip_unit_tests != 'true'", coverage["if"])
        fake = self.jobs["Coverage_Fake"]
        self.assertIn("configure", fake["needs"])
        self.assertIn("needs.configure.outputs.skip_unit_tests == 'true'", fake["if"])


if __name__ == "__main__":
    unittest.main()
