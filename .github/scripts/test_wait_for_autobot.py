# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Exercise result authority, attempt isolation and bounded waiting without network calls."""

import json
import unittest
import urllib.error
import urllib.parse
from copy import deepcopy

from wait_for_autobot import CHECK_NAME, REQUEST_JOB, receive


class AutobotResultTest(unittest.TestCase):
    def setUp(self):
        self.repo, self.sha = "NVIDIA/Megatron-LM", "a" * 40
        self.clock = 0
        self.request = {
            "repository": self.repo,
            "run_id": 10,
            "run_attempt": 2,
            "job_id": 30,
            "head_sha": self.sha,
            "base_sha": "b" * 40,
            "pull_request_number": 123,
            "request_id": f"autobot-ci:v1:{self.repo}:10:2:30",
        }
        self.payload = {
            "schema_version": 1,
            "kind": "hello",
            "request": self.request,
            "result": {"message": "Hello from Autobot"},
        }
        self.job = {
            "id": 30,
            "run_id": 10,
            "run_attempt": 2,
            "head_sha": self.sha,
            "head_branch": "pull-request/123",
            "name": REQUEST_JOB,
            "status": "completed",
            "conclusion": "success",
        }
        self.check = {
            "name": CHECK_NAME,
            "head_sha": self.sha,
            "app": {"id": 50},
            "external_id": self.request["request_id"],
            "status": "completed",
            "conclusion": "success",
            "output": {"text": json.dumps(self.payload)},
        }
        self.checks = [self.check]

    def sleep(self, seconds):
        self.clock += seconds

    def get(self, path):
        if "/jobs?" in path:
            self.assertIn("/attempts/2/", path)
            return {"jobs": [self.job]}
        self.assertIn("filter=all", path)
        return {"check_runs": self.checks}

    def receive(self, get=None):
        return receive(
            get or self.get,
            self.repo,
            10,
            2,
            self.sha,
            50,
            timeout=30,
            now=lambda: self.clock,
            sleep=self.sleep,
        )

    def test_late_result_is_received_after_waiting(self):
        def get(path):
            if "/check-runs?" in path and self.clock < 20:
                return {"check_runs": []}
            return self.get(path)

        self.assertEqual(self.receive(get), self.payload)
        self.assertEqual(self.clock, 20)

    def test_foreign_app_previous_attempt_and_wrong_sha_cannot_satisfy_wait(self):
        for field, value in (
            ("app", {"id": 99}),
            ("external_id", f"autobot-ci:v1:{self.repo}:10:1:30"),
            ("head_sha", "c" * 40),
        ):
            with self.subTest(field=field):
                self.clock = 0
                check = deepcopy(self.check)
                check[field] = value
                self.checks = [check]
                with self.assertRaises(TimeoutError):
                    self.receive()
                self.assertEqual(self.clock, 30)

    def test_matching_check_must_carry_the_same_request_payload(self):
        self.payload["request"]["run_attempt"] = 1
        self.check["output"]["text"] = json.dumps(self.payload)
        with self.assertRaisesRegex(RuntimeError, "payload"):
            self.receive()

    def test_failed_and_duplicate_results_fail_immediately(self):
        self.check["conclusion"] = "failure"
        with self.assertRaisesRegex(RuntimeError, "failed"):
            self.receive()
        self.checks.append(deepcopy(self.check))
        with self.assertRaisesRegex(RuntimeError, "Multiple"):
            self.receive()
        self.assertEqual(self.clock, 0)

    def test_old_attempt_job_cannot_define_the_request(self):
        self.job["run_attempt"] = 1
        with self.assertRaisesRegex(RuntimeError, "identity"):
            self.receive()

    def test_result_on_later_page_is_found(self):
        def get(path):
            query = urllib.parse.parse_qs(urllib.parse.urlsplit(path).query)
            if "/check-runs?" in path and query["page"] == ["1"]:
                return {"check_runs": [{"name": "other"}] * 100}
            return self.get(path)

        self.assertEqual(self.receive(get), self.payload)

    def test_rate_limit_retry_after_is_respected_within_the_deadline(self):
        for status in (403, 429):
            with self.subTest(status=status):
                self.clock = 0

                def get(path):
                    if self.clock == 0:
                        raise urllib.error.HTTPError(
                            path, status, "throttled", {"Retry-After": "5"}, None
                        )
                    return self.get(path)

                self.assertEqual(self.receive(get), self.payload)
                self.assertEqual(self.clock, 5)

    def test_pagination_cannot_keep_reading_after_the_deadline(self):
        calls = []

        def get(path):
            calls.append(path)
            self.clock += 20
            if "/check-runs?" in path:
                return {"check_runs": [{"name": "other"}] * 100}
            return self.get(path)

        with self.assertRaises(TimeoutError):
            self.receive(get)
        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
