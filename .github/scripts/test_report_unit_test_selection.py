#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU-only regression tests for archived unit-test selection measurements."""

import copy
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from report_unit_test_selection import build_report, render_summary


class TestUnitTestSelectionReport(unittest.TestCase):
    def setUp(self) -> None:
        self.buckets = ["tests/unit_tests/models/**/*.py", "tests/unit_tests/transformer/**/*.py"]
        self.selection = {
            "mode": "selective",
            "reason": "impacted files plus baseline",
            "selected_count": 10,
            "total_count": 500,
            "selector_duration_seconds": 1.2,
            "planning_duration_seconds": 3.5,
            "total_bucket_count": 35,
            "matrix": [{"bucket": bucket, "unit_test_files": "encoded"} for bucket in self.buckets],
        }
        self.metadata = {
            "repository": "NVIDIA/Megatron-LM",
            "run_id": "123",
            "run_attempt": 2,
            "sha": "abcd",
            "event_name": "push",
            "pr_number": "6969",
            "run_url": "https://github.com/NVIDIA/Megatron-LM/actions/runs/123/attempts/2",
            "selective_label_present": True,
        }
        self.jobs = [
            self._job(0, "2026-09-09T12:00:00Z", "2026-09-09T12:01:00Z"),
            self._job(1, "2026-09-09T12:00:30Z", "2026-09-09T12:02:30Z"),
        ]

    def _job(self, index: int, start: str, end: str) -> dict:
        return {
            "databaseId": index + 100,
            "name": f"{self.buckets[index]} - latest",
            "status": "completed",
            "conclusion": "success",
            "startedAt": start,
            "completedAt": end,
            "url": f"https://github.com/NVIDIA/Megatron-LM/actions/runs/123/job/{index + 100}",
        }

    def _report(self, jobs: list | None = None) -> dict:
        return build_report(
            self.selection, {"jobs": self.jobs if jobs is None else jobs}, self.metadata
        )

    def test_parallel_job_duration_sum_is_distinct_from_execution_span(self) -> None:
        report = self._report()
        self.assertTrue(report["timing_complete"])
        self.assertEqual(report["sum_job_execution_seconds"], 180)
        self.assertEqual(report["job_execution_span_seconds"], 150)
        self.assertEqual(report["observed_job_execution_seconds"], 180)
        self.assertEqual(report["unit_test_outcome"], "success")
        self.assertEqual(report["selected_file_count"], 10)
        self.assertEqual(report["total_file_count"], 500)
        self.assertEqual(report["selected_bucket_count"], 2)
        self.assertEqual(report["total_bucket_count"], 35)
        self.assertEqual(report["selector_duration_seconds"], 1.2)
        self.assertEqual(report["planning_duration_seconds"], 3.5)

    def test_only_exact_manifest_h100_job_names_are_measured(self) -> None:
        other_names = [
            f"{self.buckets[0]} - gb200 latest",
            f"{self.buckets[0]} - legacy",
            f"{self.buckets[0]} - latest / nested job",
            "cicd-parse-unit-tests",
            "gpt3 - latest",
            "tests/unit_tests/not_planned.py - latest",
        ]
        extras = [{**self.jobs[0], "name": name} for name in other_names]
        report = self._report([*self.jobs, *extras])
        self.assertTrue(report["timing_complete"])
        self.assertEqual(report["observed_job_count"], 2)
        self.assertEqual(report["sum_job_execution_seconds"], 180)

    def test_label_presence_is_independent_of_full_fallback_mode(self) -> None:
        for label in (True, False, None):
            with self.subTest(label=label):
                self.selection.update(mode="full", reason="safe full fallback", selected_count=500)
                self.metadata["selective_label_present"] = label
                report = self._report()
                self.assertIs(report["selective_label_present"], label)
                self.assertEqual(report["mode"], "full")
                self.assertEqual(report["selection_reason"], "safe full fallback")
                self.assertEqual(report["selected_file_count"], 500)
                self.assertTrue(report["timing_complete"])

    def test_missing_job_preserves_partial_measurements_without_complete_totals(self) -> None:
        report = self._report(self.jobs[:1])
        self.assertFalse(report["timing_complete"])
        self.assertEqual(report["missing_job_count"], 1)
        self.assertEqual(report["unit_test_outcome"], "incomplete")
        self.assertIsNone(report["sum_job_execution_seconds"])
        self.assertIsNone(report["job_execution_span_seconds"])
        self.assertEqual(report["observed_job_execution_seconds"], 60)

    def test_invalid_and_incomplete_timestamps_are_not_zero_duration_successes(self) -> None:
        for update in (
            {"startedAt": None},
            {"completedAt": "0001-01-01T00:00:00Z"},
            {"completedAt": "2026-09-09T11:59:59Z"},
            {"startedAt": "2026-09-09T12:00:00"},
            {"completedAt": "invalid"},
            {"status": "in_progress", "completedAt": None},
            {"status": "queued", "startedAt": None, "completedAt": None},
        ):
            with self.subTest(update=update):
                jobs = [{**self.jobs[0], **update}, self.jobs[1]]
                report = self._report(jobs)
                self.assertFalse(report["timing_complete"])
                self.assertIsNone(report["sum_job_execution_seconds"])
                self.assertIsNone(report["jobs"][0]["duration_seconds"])
                self.assertEqual(report["timed_job_count"], 1)

    def test_failed_and_cancelled_jobs_keep_real_durations_and_outcomes(self) -> None:
        for conclusion, expected_field, outcome in (
            ("failure", "failed_job_count", "failure"),
            ("timed_out", "failed_job_count", "failure"),
            ("cancelled", "cancelled_job_count", "cancelled"),
        ):
            with self.subTest(conclusion=conclusion):
                jobs = [{**self.jobs[0], "conclusion": conclusion}, self.jobs[1]]
                report = self._report(jobs)
                self.assertTrue(report["timing_complete"])
                self.assertEqual(report["unit_test_outcome"], outcome)
                self.assertEqual(report[expected_field], 1)
                self.assertEqual(report["successful_job_count"], 1)
                self.assertEqual(report["sum_job_execution_seconds"], 180)

    def test_skipped_jobs_are_incomplete_even_with_equal_timestamps(self) -> None:
        jobs = [
            {**job, "conclusion": "skipped", "completedAt": job["startedAt"]} for job in self.jobs
        ]
        report = self._report(jobs)
        self.assertFalse(report["timing_complete"])
        self.assertEqual(report["unit_test_outcome"], "skipped")
        self.assertEqual(report["skipped_job_count"], 2)
        self.assertIsNone(report["observed_job_execution_seconds"])
        self.assertIsNone(report["sum_job_execution_seconds"])

    def test_duplicate_job_names_cannot_be_mistaken_for_one_attempt(self) -> None:
        report = self._report([*self.jobs, {**self.jobs[0], "databaseId": 999}])
        self.assertFalse(report["timing_complete"])
        self.assertEqual(report["duplicate_job_count"], 1)
        self.assertEqual(report["unit_test_outcome"], "incomplete")
        self.assertIsNone(report["sum_job_execution_seconds"])

    def test_missing_inputs_produce_incomplete_data_without_zero_totals(self) -> None:
        report = build_report(None, None, self.metadata)
        self.assertEqual(report["mode"], "unknown")
        self.assertEqual(report["unit_test_outcome"], "incomplete")
        self.assertFalse(report["timing_complete"])
        self.assertIsNone(report["expected_job_count"])
        self.assertIsNone(report["selected_file_count"])
        self.assertIsNone(report["sum_job_execution_seconds"])
        self.assertIsNone(report["observed_job_execution_seconds"])
        self.assertTrue(report["issues"])

    def test_malformed_manifest_and_job_records_are_incomplete(self) -> None:
        for key, value in (
            ("selected_count", True),
            ("total_count", 0),
            ("matrix", []),
            ("mode", []),
        ):
            with self.subTest(key=key):
                selection = {**self.selection, key: value}
                report = build_report(selection, {"jobs": self.jobs}, self.metadata)
                self.assertFalse(report["timing_complete"])
                self.assertIsNone(report["sum_job_execution_seconds"])
        report = self._report([None, {"name": []}])
        self.assertFalse(report["timing_complete"])
        self.assertEqual(report["missing_job_count"], 2)

    def test_timezone_offsets_are_normalized(self) -> None:
        jobs = copy.deepcopy(self.jobs)
        jobs[1]["startedAt"] = "2026-09-09T05:00:30-07:00"
        jobs[1]["completedAt"] = "2026-09-09T05:02:30-07:00"
        report = self._report(jobs)
        self.assertEqual(report["sum_job_execution_seconds"], 180)
        self.assertEqual(report["job_execution_span_seconds"], 150)

    def test_summary_explains_observed_execution_and_incomplete_data(self) -> None:
        summary = render_summary(self._report(self.jobs[:1]))
        self.assertIn("unavailable", summary)
        self.assertIn("possibly partial", summary)
        self.assertIn("neither metric measures queue time", summary)
        self.assertIn("not collected pytest cases", summary)
        self.assertIn("1 planned H100 job(s) are missing", summary)

    def test_cli_writes_json_csv_markdown_and_preserves_unknown_label(self) -> None:
        script = Path(__file__).with_name("report_unit_test_selection.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selection = root / "selection.json"
            selection.write_text(json.dumps(self.selection))
            jobs = root / "jobs.json"
            jobs.write_text(json.dumps({"jobs": self.jobs}))
            args = [
                sys.executable,
                str(script),
                "--selection",
                str(selection),
                "--jobs",
                str(jobs),
                "--repository",
                "NVIDIA/Megatron-LM",
                "--run-id",
                "123",
                "--run-attempt",
                "2",
                "--sha",
                "abcd",
                "--event-name",
                "push",
                "--selective-label-present",
                "unknown",
                "--output-dir",
                str(root / "metrics"),
            ]
            for missing_inputs in (False, True):
                with self.subTest(missing_inputs=missing_inputs):
                    if missing_inputs:
                        selection.unlink()
                        jobs.unlink()
                    subprocess.run(args, check=True, capture_output=True, text=True)
                    report = json.loads((root / "metrics/unit-test-metrics.json").read_text())
                    self.assertIsNone(report["selective_label_present"])
                    self.assertEqual(report["run_attempt"], 2)
                    self.assertEqual(report["timing_complete"], not missing_inputs)
                    with (root / "metrics/unit-test-metrics.csv").open() as stream:
                        records = list(csv.DictReader(stream))
                    self.assertEqual(len(records), 1)
                    self.assertEqual(records[0]["selective_label_present"], "")
                    self.assertEqual(
                        records[0]["sum_job_execution_seconds"], "" if missing_inputs else "180.0"
                    )
                    self.assertTrue((root / "metrics/unit-test-metrics.md").is_file())


if __name__ == "__main__":
    unittest.main()
