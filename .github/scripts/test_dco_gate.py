#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import contextlib
import io
import re
import unittest
from pathlib import Path
from unittest import mock

import dco_gate
from dco_gate import GateError, gate_conclusion, select_latest_dco

SHA = "a" * 40


def _dco(check_run_id: int, conclusion: str | None, *, app_slug: str = "dco", sha: str = SHA):
    return {
        "id": check_run_id,
        "name": "DCO",
        "head_sha": sha,
        "status": "completed",
        "conclusion": conclusion,
        "app": {"slug": app_slug},
        "html_url": f"https://example.test/checks/{check_run_id}",
        "output": {"summary": "All commits are signed off!"},
    }


class TestDcoGate(unittest.TestCase):
    """Validate DCO verdict trust, the fail-closed mapping, and workflow topology."""

    def test_selects_the_newest_trusted_verdict(self) -> None:
        old_failure = _dco(10, "action_required")
        later_approval = _dco(20, "success")
        spoofed = _dco(30, "success", app_slug="github-actions")
        self.assertIs(
            select_latest_dco([later_approval, spoofed, old_failure], SHA), later_approval
        )

    def test_ignores_a_same_named_check_from_another_app(self) -> None:
        spoofed = _dco(30, "success", app_slug="github-actions")
        with self.assertRaisesRegex(GateError, "no completed DCO App check"):
            select_latest_dco([spoofed], SHA)

    def test_ignores_a_verdict_for_another_head(self) -> None:
        with self.assertRaisesRegex(GateError, "no completed DCO App check"):
            select_latest_dco([_dco(10, "success", sha="b" * 40)], SHA)

    def test_ignores_an_unfinished_verdict(self) -> None:
        pending = {**_dco(10, None), "status": "in_progress"}
        with self.assertRaisesRegex(GateError, "no completed DCO App check"):
            select_latest_dco([pending], SHA)

    def test_all_non_success_conclusions_remain_blocking(self) -> None:
        for conclusion in [
            "action_required",
            "cancelled",
            "failure",
            "neutral",
            "stale",
            "timed_out",
            None,
        ]:
            with self.subTest(conclusion=conclusion):
                self.assertEqual(gate_conclusion(_dco(20, conclusion)), "failure")
        self.assertEqual(gate_conclusion(_dco(20, "success")), "success")

    def test_waits_for_a_late_verdict(self) -> None:
        source = _dco(20, "success")
        clock = iter([0.0, 1.0, 2.0, 3.0, 4.0])
        with (
            mock.patch.object(dco_gate, "_list_check_runs", side_effect=[[], [], [source]]),
            mock.patch.object(dco_gate, "time") as patched_time,
        ):
            patched_time.monotonic.side_effect = lambda: next(clock)
            self.assertIs(
                dco_gate.await_dco_verdict(
                    "https://api", "NVIDIA/Megatron-LM", SHA, "token", timeout=100, poll=0
                ),
                source,
            )

    def test_fails_closed_when_no_verdict_arrives(self) -> None:
        with (
            mock.patch.object(dco_gate, "_list_check_runs", return_value=[]),
            mock.patch.object(dco_gate, "time") as patched_time,
        ):
            patched_time.monotonic.side_effect = [0.0, 100.0]
            with self.assertRaisesRegex(GateError, "no completed DCO App verdict"):
                dco_gate.await_dco_verdict(
                    "https://api", "NVIDIA/Megatron-LM", SHA, "token", timeout=10, poll=0
                )

    def test_verdict_becomes_the_job_exit_code(self) -> None:
        for conclusion, expected in (("success", 0), ("action_required", 1)):
            with self.subTest(conclusion=conclusion):
                with (
                    mock.patch.dict("os.environ", {"GITHUB_SHA": SHA}, clear=False),
                    mock.patch.object(
                        dco_gate, "await_dco_verdict", return_value=_dco(20, conclusion)
                    ),
                    contextlib.redirect_stdout(io.StringIO()),
                    contextlib.redirect_stderr(io.StringIO()),
                ):
                    self.assertEqual(
                        dco_gate.verify("NVIDIA/Megatron-LM", "https://api", "token"), expected
                    )

    def test_rejects_a_malformed_workflow_head(self) -> None:
        with mock.patch.dict("os.environ", {"GITHUB_SHA": "not-a-sha"}, clear=False):
            with self.assertRaisesRegex(GateError, "invalid head SHA"):
                dco_gate.verify("NVIDIA/Megatron-LM", "https://api", "token")

    def test_main_reports_a_failure_without_raising(self) -> None:
        environment = {
            "GITHUB_REPOSITORY": "NVIDIA/Megatron-LM",
            "GITHUB_TOKEN": "token",
            "GITHUB_SHA": "not-a-sha",
        }
        with (
            mock.patch.dict("os.environ", environment, clear=False),
            contextlib.redirect_stderr(io.StringIO()) as stderr,
        ):
            self.assertEqual(dco_gate.main(), 1)
        self.assertIn("DCO gate failed", stderr.getvalue())

    def test_pull_request_gate_runs_as_a_job_on_the_mirror_push(self) -> None:
        workflow = Path(".github/workflows/dco-gate-pull-request.yml").read_text()
        self.assertIn("name: DCO gate", workflow)
        self.assertIn('- "pull-request/[0-9]+"', workflow)
        self.assertNotIn("ref: ${{ github.event.repository.default_branch }}", workflow)
        self.assertIn("sparse-checkout: .github/scripts", workflow)
        self.assertIn("checks: read", workflow)
        self.assertNotIn("checks: write", workflow)

    def test_merge_queue_keeps_its_own_gate_and_legacy_dco_placeholder(self) -> None:
        merge_group = Path(".github/workflows/dco-gate-merge-group.yml").read_text()
        main = Path(".github/workflows/cicd-main.yml").read_text()
        self.assertIn("on:\n  merge_group:", merge_group)
        self.assertNotIn("push:", merge_group)
        self.assertIn("name: DCO gate", merge_group)
        self.assertIn("DCO_merge_group:", main)
        self.assertIn("name: DCO", main)

    def test_the_checks_api_publisher_is_gone(self) -> None:
        # A second producer of this context would race the job, and branch protection reads
        # whichever check run is newest — so an API-published gate landing last would put the
        # unbindable run back in front of the merge box.
        self.assertFalse(Path(".github/workflows/dco-gate.yml").exists())
        self.assertFalse(Path(".github/workflows/dco-gate-reconcile.yml").exists())
        source = Path(".github/scripts/dco_gate.py").read_text()
        self.assertEqual(re.findall(r'method="(\w+)"', source), ["GET"])


if __name__ == "__main__":
    unittest.main()
