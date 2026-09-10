#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import unittest
from pathlib import Path

from dco_gate import (
    GateError,
    gate_payload,
    select_existing_gate,
    select_latest_dco,
    validate_trigger,
)

SHA = "a" * 40


def _dco(
    check_run_id: int, conclusion: str, *, app_slug: str = "dco", sha: str = SHA
) -> dict[str, object]:
    return {
        "id": check_run_id,
        "name": "DCO",
        "head_sha": sha,
        "status": "completed",
        "conclusion": conclusion,
        "app": {"slug": app_slug},
        "html_url": f"https://example.test/checks/{check_run_id}",
        "output": {"summary": "Commit sign-off was manually approved."},
    }


class TestDcoGate(unittest.TestCase):
    """Validate DCO event trust, mirroring, and workflow topology."""

    def test_validates_trusted_completed_dco_event(self) -> None:
        self.assertEqual(validate_trigger({"check_run": _dco(10, "success")}), SHA)

    def test_rejects_untrusted_app(self) -> None:
        with self.assertRaisesRegex(GateError, "trusted DCO App"):
            validate_trigger({"check_run": _dco(10, "success", app_slug="github-actions")})

    def test_rejects_invalid_sha(self) -> None:
        with self.assertRaisesRegex(GateError, "invalid head SHA"):
            validate_trigger({"check_run": _dco(10, "success", sha="not-a-sha")})

    def test_selects_new_manual_approval_over_old_failure(self) -> None:
        old_failure = _dco(10, "action_required")
        manual_approval = _dco(20, "success")
        spoofed = _dco(30, "failure", app_slug="other-app")
        self.assertIs(
            select_latest_dco([manual_approval, spoofed, old_failure], SHA), manual_approval
        )

    def test_rejects_unsupported_conclusion(self) -> None:
        with self.assertRaisesRegex(GateError, "unsupported DCO conclusion"):
            select_latest_dco([_dco(10, "skipped")], SHA)

    def test_mirrors_source_conclusion_and_summary(self) -> None:
        payload = gate_payload(_dco(20, "success"), SHA, include_head=True)
        self.assertEqual(payload["name"], "DCO gate")
        self.assertEqual(payload["head_sha"], SHA)
        self.assertEqual(payload["conclusion"], "success")
        self.assertEqual(payload["external_id"], f"dco-gate:{SHA}")
        self.assertIn("manually approved", payload["output"]["summary"])

    def test_non_success_dco_result_remains_blocking(self) -> None:
        payload = gate_payload(_dco(20, "neutral"), SHA, include_head=True)
        self.assertEqual(payload["conclusion"], "failure")
        self.assertIn("`neutral`", payload["output"]["summary"])

    def test_workflows_connect_pr_and_merge_group_gates(self) -> None:
        publisher = Path(".github/workflows/dco-gate.yml").read_text()
        main = Path(".github/workflows/cicd-main.yml").read_text()
        self.assertIn("check_run:", publisher)
        self.assertIn("github.event.check_run.app.slug == 'dco'", publisher)
        self.assertIn("ref: ${{ github.event.repository.default_branch }}", publisher)
        self.assertIn("checks: write", publisher)
        self.assertIn("DCO_gate_merge_group:", main)
        self.assertIn("name: DCO gate", main)
        self.assertNotIn("DCO_merge_group:", main)

    def test_selects_only_gate_for_exact_sha(self) -> None:
        own_gate = {"id": 12, "name": "DCO gate", "head_sha": SHA, "external_id": f"dco-gate:{SHA}"}
        stale_gate = {**own_gate, "id": 13, "head_sha": "b" * 40}
        self.assertIs(select_existing_gate([stale_gate, own_gate], SHA), own_gate)


if __name__ == "__main__":
    unittest.main()
