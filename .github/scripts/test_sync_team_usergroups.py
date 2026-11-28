#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Regression tests for GitHub-to-Slack team exclusions; no API calls are made."""

import io
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import Mock, call, patch

# Team-selection tests do not need the HTTP client installed in the sync workflow.
with patch.dict(sys.modules, {"requests": Mock()}):
    import sync_team_usergroups as sync


class TestTeamExclusions(unittest.TestCase):
    def test_child_mapping_excludes_ci_and_preserves_other_teams(self):
        with patch.object(
            sync, "get_child_teams", return_value=["ci", "training", "core-ci-tools"]
        ):
            mapping = sync.get_team_to_usergroup_mapping("mcore-reviewers")

        self.assertEqual(
            mapping, {"training": "megatron-training", "core-ci-tools": "mcore-ci-tools"}
        )

    def test_sync_excludes_ci_from_child_and_direct_teams(self):
        with (
            patch.object(sync, "get_child_teams", return_value=["ci", "training"]),
            patch.object(sync, "sync_team_to_usergroup", return_value=True) as sync_team,
            redirect_stdout(io.StringIO()),
        ):
            success = sync.sync_all_teams(
                parent_teams=["mcore-reviewers"], direct_teams=["ci", "mcore-engineers"]
            )

        self.assertTrue(success)
        self.assertEqual(
            sync_team.call_args_list,
            [
                call("training", "megatron-training", dry_run=False),
                call("mcore-engineers", "mcore-engineers", dry_run=False),
            ],
        )

    def test_dry_run_also_excludes_ci(self):
        with (
            patch.object(sync, "sync_team_to_usergroup", return_value=True) as sync_team,
            redirect_stdout(io.StringIO()),
        ):
            success = sync.sync_all_teams(
                dry_run=True, parent_teams=[], direct_teams=["ci", "training"]
            )

        self.assertTrue(success)
        sync_team.assert_called_once_with("training", "megatron-training", dry_run=True)

    def test_only_excluded_teams_do_not_call_sync(self):
        with patch.object(sync, "sync_team_to_usergroup") as sync_team:
            self.assertFalse(sync.sync_all_teams(parent_teams=[], direct_teams=["ci"]))
        sync_team.assert_not_called()

    def test_list_excludes_ci_from_child_and_direct_teams(self):
        output = io.StringIO()
        with (
            patch("sys.argv", ["sync_team_usergroups.py", "--list", "--team", "ci"]),
            patch.object(sync, "get_child_teams", return_value=["ci", "training"]),
            patch.object(sync, "sync_team_to_usergroup") as sync_team,
            redirect_stdout(output),
        ):
            sync.main()

        self.assertIn("@megatron-training", output.getvalue())
        self.assertNotIn("@mcore-ci", output.getvalue())
        sync_team.assert_not_called()


if __name__ == "__main__":
    unittest.main()
