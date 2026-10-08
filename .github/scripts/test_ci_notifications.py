#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Validate nightly testing notifications without model or YAML dependencies."""

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class TestScheduledNotification(unittest.TestCase):
    """Keep nightly test alerts complete, immutable, and out of cache population."""

    def setUp(self) -> None:
        self.workflow = (ROOT / ".github/workflows/cicd-main.yml").read_text()
        self.notify = self.workflow.split("  notify-nightly-failure:\n", 1)[1].split(
            "\n  Coverage_Fake:", 1
        )[0]

    def test_reports_every_testing_job_result(self) -> None:
        jobs = set(re.findall(r"^  ([\w-]+):$", self.workflow.split("\njobs:\n", 1)[1], re.M))
        dependencies = re.search(r"^    needs: \[([^]]+)\]$", self.notify, re.M)
        self.assertIsNotNone(dependencies)
        assert dependencies is not None
        testing_jobs = {
            "cicd-unit-tests-latest",
            "cicd-unit-tests-latest-gb200",
            "cicd-integration-tests-latest-h100",
            "cicd-integration-tests-latest-gb200",
        }
        self.assertTrue(testing_jobs <= jobs)
        self.assertEqual(
            {
                job
                for job in jobs
                if re.fullmatch(r"cicd-(unit|integration)-tests-latest(?:-.*)?", job)
            },
            testing_jobs,
        )
        self.assertEqual(set(dependencies[1].split(", ")), testing_jobs)
        self.assertIn("runs-on: ubuntu-latest", self.notify)
        self.assertIn("permissions: {}", self.notify)
        self.assertIn(
            "uses: NVIDIA-NeMo/FW-CI-templates/.github/actions/notify-ci-failure@"
            "631c404d00a9e60afc591cd071d35d2e18f82fc6",
            self.notify,
        )
        self.assertIn("needs-json: ${{ toJSON(needs) }}", self.notify)
        self.assertIn("webhook: ${{ secrets.SLACK_WEBHOOK }}", self.notify)
        self.assertNotIn("checkout", self.notify)
        self.assertNotIn("run:", self.notify)

    def test_only_alerts_on_uncancelled_nightly_test_failures(self) -> None:
        condition = self.notify.split("    if: >-\n", 1)[1].split("    steps:", 1)[0]
        self.assertIn("always() && !cancelled() && contains(needs.*.result, 'failure')", condition)
        self.assertNotIn("failure()", condition)
        self.assertIn("github.repository == 'NVIDIA/Megatron-LM'", condition)
        self.assertIn(
            "(github.event_name == 'schedule' || github.event_name == 'workflow_dispatch')",
            condition,
        )
        self.assertIn("github.ref_name == github.event.repository.default_branch", condition)
        self.assertNotIn("outputs", condition)
        self.assertNotIn("push", condition)
        self.assertNotIn("merge_group", condition)

    def test_cache_population_does_not_send_testing_notifications(self) -> None:
        cache = (ROOT / ".github/workflows/populate-build-cache.yml").read_text()
        self.assertNotIn("notify-nightly-failure:", cache)
        self.assertNotIn("/notify-ci-failure@", cache)

    def test_does_not_restore_daily_gpu_test_schedule(self) -> None:
        triggers = self.workflow.split("\non:\n", 1)[1].split("\nconcurrency:", 1)[0]
        self.assertNotIn("schedule:", triggers)
        self.assertIn("      - name: Test nightly testing notification wiring", self.workflow)
        self.assertIn("run: python3 .github/scripts/test_ci_notifications.py", self.workflow)


if __name__ == "__main__":
    unittest.main()
