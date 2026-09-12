#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import unittest
from pathlib import Path
from unittest import mock

import dco_gate
from dco_gate import (
    GateError,
    gate_payload,
    select_existing_gate,
    select_latest_dco,
    validate_trigger,
)

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
        "output": {"summary": "Commit sign-off was manually approved."},
    }


class TestDcoGate(unittest.TestCase):
    """Validate DCO event trust, mirroring, HTTP paths, and workflow topology."""

    def test_validates_trusted_completed_dco_run(self) -> None:
        event = {"check_run": _dco(10, "success")}
        self.assertEqual(validate_trigger(event), SHA)

    def test_rejects_untrusted_app(self) -> None:
        event = {"check_run": _dco(10, "success", app_slug="github-actions")}
        with self.assertRaisesRegex(GateError, "trusted DCO App"):
            validate_trigger(event)

    def test_validates_manual_reconcile_sha(self) -> None:
        self.assertEqual(validate_trigger({}, requested_sha=SHA), SHA)
        with self.assertRaisesRegex(GateError, "invalid head SHA"):
            validate_trigger({}, requested_sha="not-a-sha")

    def test_selects_new_manual_approval_over_old_failure(self) -> None:
        old_failure = _dco(10, "action_required")
        manual_approval = _dco(20, "success")
        spoofed = _dco(30, "failure", app_slug="other-app")
        self.assertIs(
            select_latest_dco([manual_approval, spoofed, old_failure], SHA), manual_approval
        )

    def test_all_non_success_conclusions_remain_blocking(self) -> None:
        for source_conclusion in [
            "action_required",
            "cancelled",
            "failure",
            "neutral",
            "stale",
            "timed_out",
        ]:
            with self.subTest(source_conclusion=source_conclusion):
                payload = gate_payload(_dco(20, source_conclusion), SHA, include_head=True)
                self.assertEqual(payload["conclusion"], "failure")
                self.assertIn(f"`{source_conclusion}`", payload["output"]["summary"])

    def test_mirrors_manual_approval_success(self) -> None:
        payload = gate_payload(_dco(20, "success"), SHA, include_head=True)
        self.assertEqual(payload["name"], "DCO gate")
        self.assertEqual(payload["head_sha"], SHA)
        self.assertEqual(payload["conclusion"], "success")
        self.assertEqual(payload["external_id"], f"dco-gate:{SHA}")
        self.assertIn("manually approved", payload["output"]["summary"])

    def test_selects_only_gate_for_exact_sha(self) -> None:
        own_gate = {"id": 12, "name": "DCO gate", "head_sha": SHA, "external_id": f"dco-gate:{SHA}"}
        stale_gate = {**own_gate, "id": 13, "head_sha": "b" * 40}
        self.assertIs(select_existing_gate([stale_gate, own_gate], SHA), own_gate)

    def test_publish_gate_creates_then_updates(self) -> None:
        source = _dco(20, "success")
        post_result = {"id": 100}
        with (
            mock.patch.object(dco_gate, "_list_check_runs", side_effect=[[source], []]),
            mock.patch.object(dco_gate, "_request_json", return_value=post_result) as request,
        ):
            self.assertEqual(
                dco_gate.publish_gate({}, "NVIDIA/Megatron-LM", "https://api", "token", SHA),
                post_result,
            )
            method, url, _, payload = request.call_args.args
            self.assertEqual(
                (method, url), ("POST", "https://api/repos/NVIDIA/Megatron-LM/check-runs")
            )
            self.assertEqual(payload["head_sha"], SHA)

        gate = {"id": 100, "name": "DCO gate", "head_sha": SHA, "external_id": f"dco-gate:{SHA}"}
        with (
            mock.patch.object(dco_gate, "_list_check_runs", side_effect=[[source], [gate]]),
            mock.patch.object(dco_gate, "_request_json", return_value={"id": 100}) as request,
        ):
            dco_gate.publish_gate({}, "NVIDIA/Megatron-LM", "https://api", "token", SHA)
            method, url, _, payload = request.call_args.args
            self.assertEqual(
                (method, url), ("PATCH", "https://api/repos/NVIDIA/Megatron-LM/check-runs/100")
            )
            self.assertNotIn("head_sha", payload)

    def test_workflows_isolate_merge_group_and_preserve_legacy_dco(self) -> None:
        publisher = Path(".github/workflows/dco-gate.yml").read_text()
        merge_group = Path(".github/workflows/dco-gate-merge-group.yml").read_text()
        main = Path(".github/workflows/cicd-main.yml").read_text()
        self.assertIn("check_run:", publisher)
        self.assertIn("workflow_dispatch:", publisher)
        self.assertIn("github.event.check_run.app.slug == 'dco'", publisher)
        self.assertIn("ref: ${{ github.event.repository.default_branch }}", publisher)
        self.assertIn("on:\n  merge_group:", merge_group)
        self.assertNotIn("push:", merge_group)
        self.assertIn("name: DCO gate", merge_group)
        self.assertIn("DCO_merge_group:", main)
        self.assertIn("name: DCO", main)


if __name__ == "__main__":
    unittest.main()
