#!/usr/bin/env python3
"""Exercise the workflow's actual approver program with an offline GitHub API."""

import contextlib
import io
import json
import os
import sys
import tempfile
import textwrap
import types
import unittest
from pathlib import Path
from unittest.mock import patch

WORKFLOW = Path(sys.argv.pop(1))
PROGRAM = textwrap.dedent(
    WORKFLOW.read_text()
    .split("        shell: python\n        run: |\n", 1)[1]
    .split("\n  notify:", 1)[0]
).replace("${{ matrix.branch }}", "main")
PROGRAM = compile(PROGRAM, str(WORKFLOW), "exec")


def run(run_id, status="in_progress", attempt=1):
    return {
        "id": run_id,
        "name": "CICD Megatron-LM",
        "head_branch": f"pull-request/{run_id}",
        "display_title": f"PR {run_id}",
        "created_at": f"2026-09-28T00:00:{run_id:02d}Z",
        "status": status,
        "run_attempt": attempt,
    }


def gate(status="completed", conclusion="success"):
    return {"name": "cicd-wait-in-queue", "status": status, "conclusion": conclusion}


def page(key, items, total=None):
    return {key: items, "total_count": len(items) if total is None else total}


class ApproverTests(unittest.TestCase):
    def setUp(self):
        self.active = [run(1)]
        self.waiting = [run(2, "waiting")]
        self.labels = {}
        self.branches = {}
        self.overrides = {}
        self.contributor = "internal"

    def execute(self):
        payloads = {
            f"actions/runs?status={status}&per_page=100&page=1": page(
                "workflow_runs", [item for item in self.active if item["status"] == status]
            )
            for status in ("queued", "in_progress")
        }
        payloads["actions/runs?status=waiting&per_page=100&page=1"] = page(
            "workflow_runs", self.waiting
        )
        for item in self.active:
            payloads[
                f"actions/runs/{item['id']}/attempts/{item['run_attempt']}/jobs?per_page=100&page=1"
            ] = page("jobs", [gate()])
        for item in self.active + self.waiting:
            run_id = item["id"]
            payloads[f"pulls/{run_id}"] = {
                "user": {
                    "login": "svcnemo-autobot" if self.contributor == "internal" else "external"
                },
                "base": {"ref": self.branches.get(run_id, "main")},
                "labels": [{"name": label} for label in self.labels.get(run_id, [])],
            }
            payloads[f"actions/runs/{run_id}/pending_deployments"] = [
                {"id": run_id, "environment": {"id": 10}}
            ]
        payloads.update(self.overrides)
        calls, approvals = [], []

        def get(url, **kwargs):
            endpoint = url.split("/repos/NVIDIA/Megatron-LM/", 1)[1]
            calls.append(endpoint)
            self.assertIn(endpoint, payloads, f"Unexpected API lookup: {endpoint}")
            return types.SimpleNamespace(
                status_code=200, json=lambda: payloads[endpoint], raise_for_status=lambda: None
            )

        def post(url, **kwargs):
            approvals.append(int(url.split("/runs/", 1)[1].split("/", 1)[0]))
            return types.SimpleNamespace(
                status_code=200, json=lambda: {"ok": True}, raise_for_status=lambda: None
            )

        requests = types.ModuleType("requests")
        requests.get, requests.post = get, post
        requests.exceptions = types.SimpleNamespace(
            ConnectionError=ConnectionError, Timeout=TimeoutError, RequestException=OSError
        )
        error = None
        with tempfile.TemporaryDirectory() as directory:
            users = Path(directory) / "users.json"
            users.write_text(json.dumps({}))
            environment = {
                "GITHUB_TOKEN": "offline-test-token",
                "GITHUB_REPOSITORY": "NVIDIA/Megatron-LM",
                "CONTRIBUTOR_TYPE": self.contributor,
                "MAX_CONCURRENCY": "2",
                "MAX_CONCURRENCY_EXTERNAL": "1",
                "ALLOW_PR_RUN_ALL_TESTS": "true",
                "SSO_USERS_FILE": str(users),
            }
            with (
                patch.dict(os.environ, environment),
                patch.dict(sys.modules, {"requests": requests}),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                try:
                    exec(PROGRAM, {"__name__": "__main__", "exit": sys.exit})
                except SystemExit as exception:
                    if exception.code != 0:
                        error = exception
                except RuntimeError as exception:
                    error = exception
        return approvals, calls, error

    def test_cpu_builds_do_not_occupy_regular_or_functional_slots(self):
        self.labels = {1: ["Run functional tests"], 2: ["Run functional tests"]}
        for jobs in (
            [],
            [gate("waiting", None)],
            [gate("completed", "failure")],
            [gate("completed", "skipped")],
        ):
            with self.subTest(jobs=jobs):
                self.overrides["actions/runs/1/attempts/1/jobs?per_page=100&page=1"] = page(
                    "jobs", jobs
                )
                approvals, _, error = self.execute()
                self.assertIsNone(error)
                self.assertEqual(approvals, [2])

    def test_approved_gate_occupies_slot_even_while_queued_for_runner(self):
        for status in ("queued", "in_progress"):
            for job in (gate("queued", None), gate("in_progress", None), gate()):
                with self.subTest(status=status, gate=job):
                    self.active = [run(1, status)]
                    self.overrides["actions/runs/1/attempts/1/jobs?per_page=100&page=1"] = page(
                        "jobs", [job]
                    )
                    approvals, _, error = self.execute()
                    self.assertIsNone(error)
                    self.assertEqual(approvals, [])

    def test_functional_slot_is_global_but_does_not_block_ordinary_tests(self):
        self.branches[1] = "dev"
        self.labels = {1: ["Run functional tests"], 2: ["Run functional tests"]}
        self.waiting.append(run(3, "waiting"))
        approvals, _, error = self.execute()
        self.assertIsNone(error)
        self.assertEqual(approvals, [3])

    def test_external_contributors_keep_one_global_slot(self):
        self.contributor = "external"
        self.branches[1] = "dev"
        approvals, _, error = self.execute()
        self.assertIsNone(error)
        self.assertEqual(approvals, [])

    def test_gate_on_later_job_page_is_counted(self):
        self.overrides.update(
            {
                "actions/runs/1/attempts/1/jobs?per_page=100&page=1": page(
                    "jobs", [{"name": "CPU build"}] * 100, 101
                ),
                "actions/runs/1/attempts/1/jobs?per_page=100&page=2": page("jobs", [gate()], 101),
            }
        )
        approvals, calls, error = self.execute()
        self.assertIsNone(error)
        self.assertEqual(approvals, [])
        self.assertIn("actions/runs/1/attempts/1/jobs?per_page=100&page=2", calls)

    def test_active_run_on_later_workflow_page_is_counted(self):
        self.overrides.update(
            {
                "actions/runs?status=in_progress&per_page=100&page=1": page(
                    "workflow_runs", [{"name": "Other workflow"}] * 100, 101
                ),
                "actions/runs?status=in_progress&per_page=100&page=2": page(
                    "workflow_runs", self.active, 101
                ),
            }
        )
        approvals, _, error = self.execute()
        self.assertIsNone(error)
        self.assertEqual(approvals, [])

    def test_rerun_reuses_successful_gate_unless_current_attempt_has_new_gate(self):
        self.active = [run(1, attempt=2)]
        self.overrides["actions/runs/1/attempts/1/jobs?per_page=100&page=1"] = page(
            "jobs", [gate()]
        )
        for jobs, expected in (([], []), ([gate("waiting", None)], [2])):
            with self.subTest(jobs=jobs):
                self.overrides["actions/runs/1/attempts/2/jobs?per_page=100&page=1"] = page(
                    "jobs", jobs
                )
                approvals, calls, error = self.execute()
                self.assertIsNone(error)
                self.assertEqual(approvals, expected)
                self.assertEqual(
                    "actions/runs/1/attempts/1/jobs?per_page=100&page=1" in calls, not jobs
                )

    def test_incomplete_lookups_prevent_all_approvals(self):
        for endpoint in (
            "actions/runs?status=in_progress&per_page=100&page=1",
            "actions/runs/1/attempts/1/jobs?per_page=100&page=1",
            "pulls/1",
        ):
            with self.subTest(endpoint=endpoint):
                self.overrides = {endpoint: None}
                approvals, _, error = self.execute()
                self.assertIsInstance(error, RuntimeError)
                self.assertEqual(approvals, [])

    def test_failed_or_truncated_later_job_page_prevents_approval(self):
        for response in (None, page("jobs", [], 101)):
            with self.subTest(response=response):
                self.overrides = {
                    "actions/runs/1/attempts/1/jobs?per_page=100&page=1": page(
                        "jobs", [{"name": "CPU build"}] * 100, 101
                    ),
                    "actions/runs/1/attempts/1/jobs?per_page=100&page=2": response,
                }
                approvals, _, error = self.execute()
                self.assertIsInstance(error, RuntimeError)
                self.assertEqual(approvals, [])


if __name__ == "__main__":
    unittest.main()
