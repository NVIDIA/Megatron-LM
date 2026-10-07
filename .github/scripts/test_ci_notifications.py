#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Validate the scheduled-cache notifier without model or YAML dependencies."""

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class TestScheduledNotification(unittest.TestCase):
    """Keep nightly alerts complete, immutable, and schedule-only."""

    def setUp(self) -> None:
        self.workflow = (ROOT / ".github/workflows/populate-build-cache.yml").read_text()
        self.notify = self.workflow.split("  notify-nightly-failure:\n", 1)[1]

    def test_reports_every_direct_job_result(self) -> None:
        jobs = set(re.findall(r"^  ([\w-]+):$", self.workflow.split("\njobs:\n", 1)[1], re.M))
        dependencies = re.search(r"^    needs: \[([^]]+)\]$", self.notify, re.M)
        self.assertIsNotNone(dependencies)
        assert dependencies is not None
        self.assertEqual(set(dependencies[1].split(", ")), jobs - {"notify-nightly-failure"})
        self.assertIn("runs-on: ubuntu-latest", self.notify)
        self.assertIn("permissions: {}", self.notify)
        self.assertIn(
            "uses: NVIDIA-NeMo/FW-CI-templates/.github/actions/notify-ci-failure@"
            "631c404d00a9e60afc591cd071d35d2e18f82fc6",
            self.notify,
        )
        self.assertIn("needs-json: ${{ toJSON(needs) }}", self.notify)
        self.assertIn("webhook: ${{ secrets.SLACK_CI_CHANNEL_WEBHOOK }}", self.notify)
        self.assertNotIn("checkout", self.notify)
        self.assertNotIn("run:", self.notify)

    def test_only_alerts_on_uncancelled_scheduled_failures(self) -> None:
        condition = self.notify.split("    if: >-\n", 1)[1].split("    steps:", 1)[0]
        self.assertIn("always() && !cancelled() && failure()", condition)
        self.assertIn("github.repository == 'NVIDIA/Megatron-LM'", condition)
        self.assertIn("github.event_name == 'schedule'", condition)
        self.assertIn("github.ref_name == github.event.repository.default_branch", condition)
        self.assertNotIn("needs.", condition)
        self.assertNotIn("workflow_dispatch", condition)
        self.assertNotIn("push", condition)

    def test_does_not_restore_daily_gpu_test_schedule(self) -> None:
        main = (ROOT / ".github/workflows/cicd-main.yml").read_text()
        triggers = main.split("\non:\n", 1)[1].split("\nconcurrency:", 1)[0]
        self.assertNotIn("schedule:", triggers)
        self.assertIn("      - name: Test scheduled CI notification wiring", main)
        self.assertIn("run: python3 .github/scripts/test_ci_notifications.py", main)


if __name__ == "__main__":
    unittest.main()
