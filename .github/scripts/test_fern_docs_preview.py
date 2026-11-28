# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the trusted Fern preview resolver and its command-line entry points."""

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import fern_docs_preview as preview

HEAD = "a" * 40
SOURCE = "b" * 40
REPOSITORY = "NVIDIA/Megatron-LM"


def fixture():
    state = {
        "run": {
            "workflow_id": 42,
            "event": "push",
            "status": "completed",
            "conclusion": "success",
            "head_repository": {"full_name": REPOSITORY},
            "head_branch": "pull-request/123",
            "head_sha": SOURCE,
        },
        "pr": {
            "state": "open",
            "draft": False,
            "head": {"sha": HEAD, "repo": {"full_name": "contributor/Megatron-LM"}},
            "base": {"repo": {"full_name": REPOSITORY}},
        },
        "ref": {"object": {"sha": SOURCE}},
        "comparison": {"status": "ahead", "merge_base_commit": {"sha": HEAD}},
        "files": [{"filename": "docs/page.mdx"}],
        "comments": [],
    }
    endpoints = {
        "actions/runs/100": "run",
        "pulls/123": "pr",
        "git/ref/heads/pull-request/123": "ref",
        f"compare/{HEAD}...{SOURCE}": "comparison",
        "pulls/123/files": "files",
        "issues/123/comments": "comments",
    }

    def get(endpoint):
        if endpoint == "actions/workflows/fern-docs-preview-request.yml":
            return {"id": 42}
        return state[endpoints[endpoint]]

    api = mock.Mock(spec=preview.GitHub)
    api.get.side_effect = get
    api.paginate.side_effect = get
    return state, api


class TestFernDocsPreview(unittest.TestCase):
    def test_accepts_current_fork_pr_synchronized_with_base(self):
        _, api = fixture()
        self.assertEqual(
            preview.resolve(api, REPOSITORY, 100), preview.PreviewRequest(123, SOURCE, HEAD)
        )

    def test_skips_ineligible_requests(self):
        cases = [
            ("failed", ("run", "conclusion"), "failure"),
            ("different workflow", ("run", "workflow_id"), 99),
            ("fork run", ("run", "head_repository", "full_name"), "other/repo"),
            ("not push", ("run", "event"), "pull_request"),
            ("not PR branch", ("run", "head_branch"), "main"),
            ("invalid SHA", ("run", "head_sha"), "main"),
            ("closed", ("pr", "state"), "closed"),
            ("draft", ("pr", "draft"), True),
            ("different base", ("pr", "base", "repo", "full_name"), "other/repo"),
            ("superseded", ("ref", "object", "sha"), "c" * 40),
            ("new fork commit", ("comparison", "status"), "diverged"),
            ("unrelated history", ("comparison", "merge_base_commit", "sha"), "c" * 40),
            ("no docs changes", ("files",), [{"filename": "megatron/core/model.py"}]),
        ]
        for label, keys, value in cases:
            with self.subTest(label=label):
                state, api = fixture()
                target = state
                for key in keys[:-1]:
                    target = target[key]
                target[keys[-1]] = value
                self.assertIsNone(preview.resolve(api, REPOSITORY, 100))

    def test_recognizes_docs_renamed_out_of_tree_and_python_helper_changes(self):
        for filename, previous in [
            ("old-page.md", "docs/page.mdx"),
            (".github/scripts/fern_docs_preview.py", ""),
            (".github/scripts/test_fern_docs_preview.py", ""),
        ]:
            with self.subTest(filename=filename):
                state, api = fixture()
                state["files"] = [{"filename": filename, "previous_filename": previous}]
                self.assertIsNotNone(preview.resolve(api, REPOSITORY, 100))

    def test_pins_trusted_config_and_rejects_symlinks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for directory in ("fern", "preview-source/fern", "preview-source/docs"):
                (root / directory).mkdir(parents=True)
            config = root / "fern/fern.config.json"
            config.write_text('{"version":"5.217.2","organization":"nvidia"}')
            source_config = root / "preview-source/fern/fern.config.json"
            source_config.write_text('{"version":"file:malicious"}')
            preview.prepare(root)
            self.assertEqual(source_config.read_text(), config.read_text())
            (root / "preview-source/docs/leak").symlink_to(config)
            with self.assertRaisesRegex(ValueError, "symlink"):
                preview.prepare(root)

    def test_creates_comment_and_updates_only_actions_bot_comment(self):
        state, api = fixture()
        request = preview.PreviewRequest(123, SOURCE, HEAD)
        state["comments"] = [{"id": 1, "user": {"login": "contributor"}, "body": preview.MARKER}]
        preview.comment(api, request, "https://preview.buildwithfern.com/")
        method, endpoint, body = api.write.call_args.args
        self.assertEqual((method, endpoint), ("POST", "issues/123/comments"))
        self.assertIn(HEAD, body)
        self.assertIn(SOURCE, body)
        state["comments"].append(
            {"id": 2, "user": {"login": "github-actions[bot]"}, "body": preview.MARKER}
        )
        preview.comment(api, request, "https://preview.buildwithfern.com/")
        self.assertEqual(api.write.call_args.args[:2], ("PATCH", "issues/comments/2"))

    def test_accepts_only_valid_fern_preview_urls(self):
        url = "https://preview.buildwithfern.com/megatron"
        self.assertEqual(preview.preview_url(f"Published docs to {url} (preview)"), url)
        for message in [
            "No URL returned",
            "Published docs to https://example.com/",
            "Published docs to https://user:pass@preview.buildwithfern.com/",
            "Published docs to https://preview.buildwithfern.com:invalid/",
        ]:
            with self.subTest(message=message), self.assertRaises(ValueError):
                preview.preview_url(message)

    @mock.patch.object(preview.subprocess, "run")
    def test_api_transport_reads_all_pages_and_writes_json_via_stdin(self, run):
        api = preview.GitHub(REPOSITORY)
        run.return_value = SimpleNamespace(stdout='{"id":42}')
        self.assertEqual(api.get("actions/runs/100"), {"id": 42})
        self.assertEqual(
            run.call_args.args[0], ["gh", "api", f"repos/{REPOSITORY}/actions/runs/100"]
        )
        run.return_value = SimpleNamespace(stdout='[[{"id":1}],[{"id":2}]]')
        self.assertEqual(api.paginate("issues/123/comments"), [{"id": 1}, {"id": 2}])
        command = run.call_args.args[0]
        self.assertIn("--paginate", command)
        self.assertIn("--slurp", command)
        self.assertTrue(command[-1].endswith("?per_page=100"))
        api.write("PATCH", "issues/comments/2", 'Body with "quotes" and\na newline')
        self.assertEqual(run.call_args.args[0][-2:], ["--input", "-"])
        self.assertEqual(
            json.loads(run.call_args.kwargs["input"]), {"body": 'Body with "quotes" and\na newline'}
        )
        run.side_effect = preview.subprocess.CalledProcessError(1, ["gh", "api"])
        with self.assertRaises(preview.subprocess.CalledProcessError):
            api.get("actions/runs/100")

    def test_cli_reports_resolution_and_rechecks_approved_head(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            event = root / "event.json"
            event.write_text('{"workflow_run":{"id":100}}')
            env = {
                "GITHUB_REPOSITORY": REPOSITORY,
                "GITHUB_EVENT_PATH": str(event),
                "GITHUB_OUTPUT": str(root / "output"),
                "GITHUB_STEP_SUMMARY": str(root / "summary"),
                "EXPECTED_HEAD_SHA": HEAD,
                "PREVIEW_URL": "https://preview.buildwithfern.com/",
            }
            request = preview.PreviewRequest(123, SOURCE, HEAD)
            with (
                mock.patch.dict(os.environ, env),
                mock.patch.object(preview, "resolve", return_value=request) as resolve,
                mock.patch.object(preview, "comment") as comment,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                with mock.patch("sys.argv", ["preview", "resolve"]):
                    preview.main()
                self.assertEqual(
                    (root / "output").read_text(),
                    f"pr-number=123\nsource-sha={SOURCE}\nhead-sha={HEAD}\n",
                )
                self.assertIn(SOURCE, (root / "summary").read_text())
                with mock.patch("sys.argv", ["preview", "current"]):
                    preview.main()
                self.assertTrue((root / "output").read_text().endswith("current=true\n"))
                with mock.patch("sys.argv", ["preview", "comment"]):
                    preview.main()
                comment.assert_called_once()
                comment.reset_mock()
                resolve.return_value = preview.PreviewRequest(123, SOURCE, "c" * 40)
                with mock.patch("sys.argv", ["preview", "current"]):
                    preview.main()
                self.assertTrue((root / "output").read_text().endswith("current=false\n"))
                with mock.patch("sys.argv", ["preview", "comment"]):
                    preview.main()
                comment.assert_not_called()


if __name__ == "__main__":
    unittest.main()
