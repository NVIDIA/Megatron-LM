# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Exercise the queue approver's inline Python without network access or dependencies."""

import contextlib
import io
import os
import sys
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, mock_open, patch
from urllib.parse import parse_qs, urlparse


def workflow(run_id, *, name="CICD Megatron-LM", created_at="2026-10-07T00:00:00Z"):
    return {
        "id": run_id,
        "name": name,
        "head_branch": f"pull-request/{run_id}",
        "display_title": f"PR {run_id}",
        "created_at": created_at,
    }


class TestApproveTestQueue(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        workflow_path = Path(
            os.environ.get(
                "WORKFLOW", Path(__file__).parents[1] / "workflows/cicd-approve-test-queue.yml"
            )
        )
        block = workflow_path.read_text().split("        shell: python\n        run: |\n", 1)[1]
        lines = []
        for line in block.splitlines():
            if line and not line.startswith("          "):
                break
            lines.append(line)
        source = textwrap.dedent("\n".join(lines)).replace("${{ matrix.branch }}", "main")
        cls.approver = compile(source, str(workflow_path), "exec")

    def setUp(self):
        self.pages = {status: [[]] for status in ("queued", "in_progress", "waiting")}
        self.labels = {}
        self.base_branches = {}
        self.page_requests = []
        self.approvals = []
        self.unrelated_runs = [workflow(i, name="Other workflow") for i in range(1000, 1100)]

    def get(self, url, **kwargs):
        parsed = urlparse(url)
        endpoint = parsed.path.removeprefix("/repos/NVIDIA/Megatron-LM/")
        response = Mock(status_code=200, headers={}, links={})
        if endpoint == "actions/runs":
            query = parse_qs(parsed.query)
            status = query["status"][0]
            page = int(query.get("page", ["1"])[0])
            self.page_requests.append((status, page))
            runs = self.pages[status][page - 1]
            if runs is None:
                response.status_code = 503
            else:
                response.json.return_value = {"workflow_runs": runs}
            if page < len(self.pages[status]):
                response.links = {
                    "next": {
                        "url": f"{url.split('?')[0]}?status={status}&per_page=100&page={page + 1}"
                    }
                }
        elif endpoint.startswith("pulls/"):
            pr_number = int(endpoint.split("/")[1])
            response.json.return_value = {
                "base": {"ref": self.base_branches.get(pr_number, "main")},
                "user": {"login": "svcnemo-autobot"},
                "labels": [{"name": label} for label in self.labels.get(pr_number, [])],
            }
        elif endpoint.endswith("/pending_deployments"):
            response.json.return_value = [{"id": 1, "environment": {"id": 2}}]
        else:
            self.fail(f"Unexpected GET request: {url}")
        return response

    def post(self, url, **kwargs):
        self.approvals.append(int(url.split("/")[-2]))
        response = Mock(status_code=200)
        response.json.return_value = {"state": "approved"}
        return response

    def run_approver(self, *, limit=1, expected_exit=0):
        requests = SimpleNamespace(
            get=self.get,
            post=self.post,
            exceptions=SimpleNamespace(
                ConnectionError=ConnectionError, Timeout=TimeoutError, RequestException=RuntimeError
            ),
        )
        environment = {
            "GITHUB_TOKEN": "test-token",
            "GITHUB_REPOSITORY": "NVIDIA/Megatron-LM",
            "CONTRIBUTOR_TYPE": "internal",
            "MAX_CONCURRENCY": str(limit * 2),
            "ALLOW_PR_RUN_ALL_TESTS": "true",
            "SSO_USERS_FILE": "users_sso.json",
        }
        exit_code = 0
        with (
            patch.dict(os.environ, environment),
            patch.dict(sys.modules, requests=requests),
            patch("builtins.open", mock_open(read_data="{}")),
            patch("time.sleep"),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            try:
                exec(self.approver, {})
            except SystemExit as error:
                exit_code = error.code
        self.assertEqual(exit_code, expected_exit)

    def test_oldest_waiting_run_on_later_page_is_approved_first(self):
        self.pages["waiting"] = [
            [workflow(i) for i in range(2, 102)],
            [workflow(1, created_at="2026-10-06T00:00:00Z")],
        ]
        self.run_approver()
        self.assertEqual(self.approvals, [1])
        self.assertIn(("waiting", 2), self.page_requests)

    def test_active_runs_on_later_pages_consume_concurrency(self):
        for status in ("queued", "in_progress"):
            with self.subTest(status=status):
                self.setUp()
                self.pages[status] = [self.unrelated_runs, [workflow(1)]]
                self.pages["waiting"] = [[workflow(2)]]
                self.run_approver()
                self.assertEqual(self.approvals, [])
                self.assertIn((status, 2), self.page_requests)

    def test_functional_run_on_later_pages_blocks_global_slot(self):
        for status in ("queued", "in_progress"):
            with self.subTest(status=status):
                self.setUp()
                self.pages[status] = [self.unrelated_runs, [workflow(1)]]
                self.pages["waiting"] = [[workflow(2), workflow(3)]]
                self.labels = {1: ["Run functional tests"], 2: ["Run functional tests"]}
                self.base_branches[1] = "dev"
                self.run_approver()
                self.assertEqual(self.approvals, [3])

    def test_later_page_failure_never_approves_partial_results(self):
        for status in ("queued", "in_progress", "waiting"):
            with self.subTest(status=status):
                self.setUp()
                self.pages["waiting"] = [[workflow(2)]]
                self.pages[status] = [[workflow(1)], None]
                self.run_approver(limit=3, expected_exit=1)
                self.assertEqual(self.approvals, [])
                self.assertIn((status, 2), self.page_requests)

    def test_duplicate_waiting_run_is_approved_once(self):
        self.pages["waiting"] = [[workflow(1)], [workflow(1), workflow(2)]]
        self.run_approver(limit=3)
        self.assertEqual(self.approvals, [1, 2])

    def test_duplicate_active_runs_do_not_consume_extra_slots(self):
        for status in ("queued", "in_progress"):
            with self.subTest(status=status):
                self.setUp()
                self.pages[status] = [[workflow(1)], [workflow(1)]]
                self.pages["waiting"] = [[workflow(2)]]
                self.run_approver(limit=2)
                self.assertEqual(self.approvals, [2])

    def test_empty_and_single_page_results_terminate(self):
        for waiting_runs in ([], [workflow(1)]):
            with self.subTest(waiting_runs=waiting_runs):
                self.setUp()
                self.pages["waiting"] = [waiting_runs]
                self.run_approver()
                self.assertEqual(
                    self.page_requests, [("queued", 1), ("in_progress", 1), ("waiting", 1)]
                )
                self.assertEqual(self.approvals, [run["id"] for run in waiting_runs])


if __name__ == "__main__":
    unittest.main()
