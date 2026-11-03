# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Regression tests for review reminders to human and automated reviewers."""

import unittest
from unittest import mock

import re_review_reminder as reminder


def review_event(login="reviewer", user_type="User"):
    return {
        "action": "review_requested",
        "repository": {"full_name": "NVIDIA/Megatron-LM"},
        "number": 7380,
        "pull_request": {"base": {"ref": "main"}, "updated_at": "2026-09-18T20:23:00Z"},
        "sender": {"login": "requester"},
        "requested_reviewer": {"login": login, "type": user_type},
    }


class TestReviewReminder(unittest.TestCase):
    def test_automated_reviewers_skip_api_and_slack_calls(self):
        for login, user_type in [
            ("svcnvidia-nemo-ci", "User"),
            ("Svcnvidia-NeMo-CI", "User"),
            ("review-app[bot]", "Bot"),
        ]:
            with (
                self.subTest(login=login),
                mock.patch.object(
                    reminder,
                    "_request_json",
                    side_effect=AssertionError("Unexpected GitHub request"),
                ),
                mock.patch.object(reminder, "_send_slack_dm") as send,
            ):
                self.assertFalse(reminder.process_event(review_event(login, user_type)))
                send.assert_not_called()

    def test_human_review_request_still_sends_reminder(self):
        event = review_event()
        with (
            mock.patch.object(
                reminder, "_get_mcore_engineers", return_value={"requester", "reviewer"}
            ),
            mock.patch.object(reminder, "_reviewer_is_still_requested", return_value=True),
            mock.patch.object(reminder, "_has_prior_review", return_value=True),
            mock.patch.object(reminder, "_send_slack_dm") as send,
        ):
            self.assertTrue(reminder.process_event(event))
            send.assert_called_once_with(reminder._extract_review_request(event))

    def test_human_slack_lookup_failure_remains_an_error(self):
        event = review_event()
        with (
            mock.patch.object(
                reminder, "_get_mcore_engineers", return_value={"requester", "reviewer"}
            ),
            mock.patch.object(reminder, "_reviewer_is_still_requested", return_value=True),
            mock.patch.object(reminder, "_has_prior_review", return_value=True),
            mock.patch.object(reminder, "get_user_email", return_value="reviewer@nvidia.com"),
            mock.patch.object(reminder, "get_slack_client"),
            mock.patch.object(reminder, "get_slack_user_id", return_value=None),
        ):
            with self.assertRaisesRegex(reminder.NotifierError, "Could not resolve.*Slack user"):
                reminder.process_event(event)


if __name__ == "__main__":
    unittest.main()
