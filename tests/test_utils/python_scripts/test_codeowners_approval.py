# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Exercise author exemptions through the approval gate and review labels."""

import contextlib
import io
import unittest
from types import SimpleNamespace
from unittest import mock

import check_codeowners_approval as checker
import swap_pr_labels

CODEOWNERS = """\
* @NVIDIA/training
megatron/training/ @NVIDIA/training
megatron/rl/ @NVIDIA/reinforcement-learning
examples/rl/ @NVIDIA/reinforcement-learning
tests/unit_tests/rl/ @NVIDIA/reinforcement-learning
train_rl.py @NVIDIA/reinforcement-learning
megatron/core/ @NVIDIA/core-adlr @NVIDIA/core-nemo
"""


def user(login):
    return SimpleNamespace(login=login)


class ApprovalFixture(unittest.TestCase):
    def setUp(self):
        self.pr = mock.Mock(number=42, draft=False, user=user("tdene"), labels=[])
        self.pr.get_files.return_value = [SimpleNamespace(filename="megatron/rl/actor.py")]
        self.pr.get_reviews.return_value = []
        self.repo = mock.Mock(full_name="NVIDIA/Megatron-LM", organization=user("NVIDIA"))
        self.repo.get_pull.return_value = self.pr
        self.repo.get_contents.return_value = SimpleNamespace(
            decoded_content=CODEOWNERS.encode("utf-8")
        )
        self.team_members = {
            "reinforcement-learning": {"rl-reviewer"},
            "training": {"trainer"},
            "core-adlr": {"adlr-reviewer"},
            "core-nemo": {"nemo-reviewer"},
        }
        self.org = mock.Mock()
        self.org.get_team_by_slug.side_effect = self.team
        self.github = mock.Mock()
        self.github.get_repo.return_value = self.repo
        self.github.get_organization.return_value = self.org
        patches = contextlib.ExitStack()
        self.addCleanup(patches.close)
        patches.enter_context(mock.patch.object(checker, "Github", return_value=self.github))
        patches.enter_context(mock.patch.dict("os.environ", {"GITHUB_TOKEN": "test-token"}))

    def team(self, slug):
        return SimpleNamespace(get_members=lambda: [user(name) for name in self.team_members[slug]])

    def set_files(self, *paths):
        self.pr.get_files.return_value = [SimpleNamespace(filename=path) for path in paths]

    def run_gate(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            result = checker.main(["--pr-number", "42", "--repo-name", "NVIDIA/Megatron-LM"])
        return result, output.getvalue()

    def tracker(self):
        with (
            mock.patch.object(swap_pr_labels, "Github", return_value=self.github),
            mock.patch("builtins.open", mock.mock_open(read_data=CODEOWNERS)),
        ):
            return swap_pr_labels.PRReviewTracker("test-token", "NVIDIA/Megatron-LM", 42)

    def swap_labels(self, tracker, reviews=None):
        with mock.patch.object(tracker, "_get_latest_reviews", return_value=reviews or {}):
            tracker.swap_labels()

    def assert_no_label_changes(self):
        self.pr.add_to_labels.assert_not_called()
        self.pr.remove_from_labels.assert_not_called()


class TestCodeownersApproval(ApprovalFixture):
    def test_tdene_needs_no_rl_approval_for_each_owned_path(self):
        for path in (
            "megatron/rl/actor.py",
            "examples/rl/train.sh",
            "tests/unit_tests/rl/test_actor.py",
            "train_rl.py",
        ):
            with self.subTest(path=path):
                self.set_files(path)
                self.assertEqual(self.run_gate()[0], 0)
        self.org.get_team_by_slug.assert_not_called()

    def test_author_login_is_case_insensitive(self):
        self.pr.user.login = "TdEnE"
        self.assertEqual(self.run_gate()[0], 0)

    def test_other_teams_remain_required_on_mixed_changes(self):
        self.set_files("megatron/rl/actor.py", "megatron/training/training.py")
        result, output = self.run_gate()
        self.assertEqual(result, 1)
        self.assertIn("@NVIDIA/training", output)
        self.assertNotIn("Missing approvals for megatron/rl/actor.py", output)

        self.pr.get_reviews.return_value = [SimpleNamespace(user=user("trainer"), state="APPROVED")]
        self.assertEqual(self.run_gate()[0], 0)

    def test_other_owners_on_the_same_file_are_still_required(self):
        self.repo.get_contents.return_value.decoded_content = (
            b"megatron/rl/ @NVIDIA/reinforcement-learning @NVIDIA/training @tdene\n"
        )
        result, output = self.run_gate()
        self.assertEqual(result, 1)
        self.assertIn("@NVIDIA/training", output)
        self.assertIn("@tdene", output)
        self.assertNotIn("@NVIDIA/reinforcement-learning", output)

    def test_tdene_as_actor_or_reviewer_does_not_exempt_another_author(self):
        self.pr.user.login = "someone-else"
        self.pr.get_reviews.return_value = [SimpleNamespace(user=user("tdene"), state="APPROVED")]
        with mock.patch.dict("os.environ", {"GITHUB_ACTOR": "tdene"}):
            result, output = self.run_gate()
        self.assertEqual(result, 1)
        self.assertIn("@NVIDIA/reinforcement-learning", output)

    def test_other_authors_can_still_pass_with_a_team_approval(self):
        self.pr.user.login = "someone-else"
        self.pr.get_reviews.return_value = [
            SimpleNamespace(user=user("rl-reviewer"), state="APPROVED")
        ]
        self.assertEqual(self.run_gate()[0], 0)

    def test_exemption_does_not_match_another_organizations_team(self):
        self.repo.get_contents.return_value.decoded_content = (
            b"megatron/rl/ @another-org/reinforcement-learning\n"
        )
        result, output = self.run_gate()
        self.assertEqual(result, 1)
        self.assertIn("@another-org/reinforcement-learning", output)

    def test_required_github_data_failures_fail_the_gate(self):
        # Training approval remains required even when the author is RL-exempt.
        self.set_files("megatron/rl/actor.py", "megatron/training/training.py")
        for target, method in (
            (self.repo, "get_contents"),
            (self.pr, "get_files"),
            (self.pr, "get_reviews"),
            (self.org, "get_team_by_slug"),
        ):
            with self.subTest(method=method):
                with mock.patch.object(target, method, side_effect=RuntimeError("API unavailable")):
                    result, output = self.run_gate()
                self.assertEqual(result, 1)
                self.assertIn("::error::", output)

    def test_missing_token_fails_the_gate(self):
        with mock.patch.dict("os.environ", {"GITHUB_TOKEN": ""}):
            result, output = self.run_gate()
        self.assertEqual(result, 1)
        self.assertIn("GITHUB_TOKEN", output)


class TestReviewLabels(ApprovalFixture):
    def test_rl_only_tdene_pr_advances_to_approved(self):
        self.swap_labels(self.tracker())
        self.pr.add_to_labels.assert_called_once_with("Approved")
        self.assertNotIn(
            mock.call("reinforcement-learning"), self.org.get_team_by_slug.call_args_list
        )

    def test_other_authors_still_need_rl_approval(self):
        self.pr.user.login = "someone-else"
        with mock.patch.dict("os.environ", {"GITHUB_ACTOR": "tdene"}):
            self.swap_labels(self.tracker(), {"tdene": "APPROVED"})
        self.assert_no_label_changes()

    def test_other_expert_teams_remain_pending(self):
        self.set_files("megatron/rl/actor.py", "megatron/training/training.py")
        self.swap_labels(self.tracker())
        self.assert_no_label_changes()

    def test_final_teams_remain_required_after_rl_exemption(self):
        self.set_files("megatron/rl/actor.py", "megatron/core/config.py")
        self.swap_labels(self.tracker())
        self.pr.add_to_labels.assert_called_once_with("Final Review")

    def test_rl_member_requesting_changes_still_blocks(self):
        self.swap_labels(self.tracker(), {"rl-reviewer": "CHANGES_REQUESTED"})
        self.assert_no_label_changes()

    def test_core_member_requesting_changes_requires_final_review(self):
        for reviewer in ("nemo-reviewer", "adlr-reviewer"):
            with self.subTest(reviewer=reviewer):
                self.pr.reset_mock()
                self.swap_labels(self.tracker(), {reviewer: "CHANGES_REQUESTED"})
                self.pr.add_to_labels.assert_called_once_with("Final Review")
                self.pr.remove_from_labels.assert_not_called()

    def test_core_member_approval_allows_advancement(self):
        for reviewer in ("nemo-reviewer", "adlr-reviewer"):
            for stage in (None, "Final Review"):
                with self.subTest(reviewer=reviewer, stage=stage):
                    self.pr.reset_mock()
                    self.pr.labels = [] if stage is None else [SimpleNamespace(name=stage)]
                    self.swap_labels(self.tracker(), {reviewer: "APPROVED"})
                    self.pr.add_to_labels.assert_called_once_with("Approved")
                    if stage:
                        self.pr.remove_from_labels.assert_called_once_with("Final Review")
                    else:
                        self.pr.remove_from_labels.assert_not_called()

    def test_core_member_requesting_changes_reverts_approved_to_final_review(self):
        self.pr.labels = [SimpleNamespace(name="Approved")]
        for reviewer in ("nemo-reviewer", "adlr-reviewer"):
            with self.subTest(reviewer=reviewer):
                self.pr.reset_mock()
                self.swap_labels(self.tracker(), {reviewer: "CHANGES_REQUESTED"})
                self.pr.remove_from_labels.assert_called_once_with("Approved")
                self.pr.add_to_labels.assert_called_once_with("Final Review")

    def test_core_member_requesting_changes_blocks_final_review_advancement(self):
        self.pr.labels = [SimpleNamespace(name="Final Review")]
        for reviewer in ("nemo-reviewer", "adlr-reviewer"):
            with self.subTest(reviewer=reviewer):
                self.pr.reset_mock()
                self.swap_labels(self.tracker(), {reviewer: "CHANGES_REQUESTED"})
                self.assert_no_label_changes()

    def test_changes_requested_reverts_an_already_approved_pr(self):
        self.pr.labels = [SimpleNamespace(name="Approved")]
        self.swap_labels(self.tracker(), {"rl-reviewer": "CHANGES_REQUESTED"})
        self.pr.remove_from_labels.assert_any_call("Approved")
        self.pr.add_to_labels.assert_not_called()

    def test_draft_pr_is_not_advanced(self):
        self.pr.draft = True
        self.swap_labels(self.tracker())
        self.assert_no_label_changes()

    def test_review_api_failure_cannot_approve_an_exempt_pr(self):
        tracker = self.tracker()
        with mock.patch.object(
            swap_pr_labels.requests, "post", side_effect=RuntimeError("offline")
        ):
            with self.assertRaises(RuntimeError):
                tracker.swap_labels()
        self.assert_no_label_changes()

    def test_graphql_errors_cannot_approve_an_exempt_pr(self):
        tracker = self.tracker()
        response = mock.Mock()
        response.json.return_value = {"errors": [{"message": "Forbidden"}]}
        with mock.patch.object(swap_pr_labels.requests, "post", return_value=response):
            with self.assertRaises(RuntimeError):
                tracker.swap_labels()
        self.assert_no_label_changes()

    def test_changed_files_failure_does_not_change_labels(self):
        tracker = self.tracker()
        self.pr.get_files.side_effect = RuntimeError("offline")
        with self.assertRaises(RuntimeError):
            self.swap_labels(tracker)
        self.assert_no_label_changes()

    def test_team_lookup_failure_does_not_approve_mixed_changes(self):
        self.set_files("megatron/rl/actor.py", "megatron/training/training.py")
        tracker = self.tracker()
        self.org.get_team_by_slug.side_effect = RuntimeError("offline")
        with self.assertRaises(RuntimeError):
            self.swap_labels(tracker)
        self.assert_no_label_changes()


if __name__ == "__main__":
    unittest.main()
