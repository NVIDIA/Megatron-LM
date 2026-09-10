#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import unittest
from pathlib import Path

from cancel_superseded_merge_group_runs import _pr_number, _select_superseded_runs


def _run(
    run_id: int, head_branch: str, created_at: str, name: str = "CICD Megatron-LM"
) -> dict[str, object]:
    return {
        "id": run_id,
        "event": "merge_group",
        "head_branch": head_branch,
        "created_at": created_at,
        "name": name,
    }


class TestCancelSupersededMergeGroupRuns(unittest.TestCase):
    def setUp(self) -> None:
        self.old_head = "gh-readonly-queue/main/pr-123-a1d1234"
        self.current_head = "gh-readonly-queue/main/pr-123-b2e5678"
        self.current_run = _run(200, self.current_head, "2026-08-13T22:00:00Z")
        self.trigger_runs = [self.current_run, _run(100, self.old_head, "2026-08-13T21:00:00Z")]

    def test_extracts_pr_number_only_from_merge_group_refs(self) -> None:
        self.assertEqual(_pr_number(self.current_head), 123)
        self.assertIsNone(_pr_number("pull-request/123"))
        self.assertIsNone(_pr_number("feature/pr-123-not-a-merge-group"))

    def test_selects_all_active_runs_from_the_older_generation(self) -> None:
        old_cicd = _run(100, self.old_head, "2026-08-13T21:00:00Z")
        old_release = _run(250, self.old_head, "2026-08-13T22:01:00Z", name="Release")
        current_release = _run(201, self.current_head, "2026-08-13T22:00:01Z", name="Release")
        other_pr = _run(50, "gh-readonly-queue/main/pr-456-a7e1234", "2026-08-13T20:00:00Z")

        selected = _select_superseded_runs(
            [old_cicd, old_release, self.current_run, current_release, other_pr],
            self.trigger_runs,
            self.current_run,
        )

        self.assertEqual([run["id"] for run in selected], [100, 250])

    def test_stale_cleanup_does_not_cancel_a_newer_generation(self) -> None:
        newer_head = "gh-readonly-queue/main/pr-123-c3f9abc"
        newer_run = _run(300, newer_head, "2026-08-13T23:00:00Z")

        selected = _select_superseded_runs(
            [self.current_run, newer_run], [newer_run, *self.trigger_runs], self.current_run
        )

        self.assertEqual(selected, [])

    def test_rerun_does_not_change_the_generation_order(self) -> None:
        old_rerun = _run(400, self.old_head, "2026-08-13T23:00:00Z")

        selected = _select_superseded_runs(
            [old_rerun], [old_rerun, *self.trigger_runs], self.current_run
        )

        self.assertEqual([run["id"] for run in selected], [400])

    def test_skips_an_active_generation_without_trusted_ordering_data(self) -> None:
        unknown_head = "gh-readonly-queue/main/pr-123-d4a1111"
        unknown_run = _run(50, unknown_head, "2026-08-13T20:00:00Z")

        selected = _select_superseded_runs([unknown_run], self.trigger_runs, self.current_run)

        self.assertEqual(selected, [])

    def test_workflow_runs_from_the_default_branch_with_minimal_write_permission(self) -> None:
        workflow = Path(".github/workflows/cicd-cancel-superseded-merge-groups.yml").read_text()

        self.assertIn('workflows: ["CICD Megatron-LM"]', workflow)
        self.assertIn("types: [requested]", workflow)
        self.assertIn("github.event.workflow_run.event == 'merge_group'", workflow)
        self.assertIn("actions: write", workflow)
        self.assertIn("ref: ${{ github.event.repository.default_branch }}", workflow)
        self.assertIn("persist-credentials: false", workflow)


if __name__ == "__main__":
    unittest.main()
