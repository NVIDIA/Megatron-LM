#!/usr/bin/env bash
set -euo pipefail

readonly WORKFLOW=${WORKFLOW:-.github/workflows/cicd-approve-test-queue.yml}
readonly EXTERNAL_QUEUE='          - branch: all'
readonly WORKFLOW_RUN_EVENT='  workflow_run:'
readonly CICD_COMPLETION_TRIGGER='    workflows: ["CICD Megatron-LM"]'
readonly COMPLETED_ACTIVITY='    types: [completed]'
readonly CONCURRENCY_GROUP='  group: approve-test-queue'
readonly SERIALIZED_RUNS='  cancel-in-progress: false'
readonly INTERNAL_SERVICE_ACCOUNT='          INTERNAL_SERVICE_ACCOUNTS = {"svcnemo-autobot"}'
readonly INTERNAL_SERVICE_ACCOUNT_CHECK='              return login in INTERNAL_SERVICE_ACCOUNTS or any('
readonly RUN_ALL_TESTS_VARIABLE='          ALLOW_PR_RUN_ALL_TESTS: ${{ vars.ALLOW_PR_RUN_ALL_TESTS }}'
readonly RUN_ALL_TESTS_LABELS='          RUN_ALL_TESTS_LABELS = {FUNCTIONAL_TEST_LABEL, RUN_TESTS_LABEL}'
readonly RUN_ALL_TESTS_GUARD='              if not ALLOW_PR_RUN_ALL_TESTS and workflow_labels & RUN_ALL_TESTS_LABELS:'

if [[ $(/usr/bin/grep -c -F "$EXTERNAL_QUEUE" "$WORKFLOW") -ne 1 ]]; then
  echo "Approve Test Queue must define exactly one global external worker" >&2
  exit 1
fi

if /usr/bin/grep -q -F 'contributor_type: [internal, external]' "$WORKFLOW"; then
  echo "Approve Test Queue must not race multiple workers against the global external queue" >&2
  exit 1
fi

if [[ $(/usr/bin/grep -c -F 'if CONTRIBUTOR_TYPE == "external":' "$WORKFLOW") -ne 3 ]]; then
  echo "Approve Test Queue must filter queued, running, and waiting external runs globally" >&2
  exit 1
fi

for expected in \
  "$WORKFLOW_RUN_EVENT" \
  "$CICD_COMPLETION_TRIGGER" \
  "$COMPLETED_ACTIVITY" \
  "$CONCURRENCY_GROUP" \
  "$SERIALIZED_RUNS"; do
  if [[ $(/usr/bin/grep -c -F "$expected" "$WORKFLOW") -ne 1 ]]; then
    echo "Approve Test Queue must wake on completed CICD runs without racing approvals: $expected" >&2
    exit 1
  fi
done

if [[ $(/usr/bin/grep -c -F "$INTERNAL_SERVICE_ACCOUNT" "$WORKFLOW") -ne 1 ]]; then
  echo "Approve Test Queue must classify svcnemo-autobot as internal without relying on the SSO export" >&2
  exit 1
fi

if [[ $(/usr/bin/grep -c -F "$INTERNAL_SERVICE_ACCOUNT_CHECK" "$WORKFLOW") -ne 1 ]]; then
  echo "Approve Test Queue must apply its internal service-account allowlist" >&2
  exit 1
fi

for expected in \
  "$RUN_ALL_TESTS_VARIABLE" \
  "$RUN_ALL_TESTS_LABELS" \
  "$RUN_ALL_TESTS_GUARD"; do
  if [[ $(/usr/bin/grep -c -F "$expected" "$WORKFLOW") -ne 1 ]]; then
    echo "Approve Test Queue must skip PRs requesting all tests when disabled: $expected" >&2
    exit 1
  fi
done

guard_line=$(/usr/bin/grep -n -F "$RUN_ALL_TESTS_GUARD" "$WORKFLOW" | /usr/bin/cut -d: -f1)
next_approval_line=$(/usr/bin/awk -v start="$guard_line" 'NR > start && /print\(f"Approving workflow/ { print NR; exit }' "$WORKFLOW")
if [[ -z "$next_approval_line" ]] || ! /usr/bin/sed -n "${guard_line},${next_approval_line}p" "$WORKFLOW" \
  | /usr/bin/grep -q -F '                  continue'; then
  echo "Approve Test Queue must continue past disabled full-test PRs before approving another workflow" >&2
  exit 1
fi

echo "Approve Test Queue preserves concurrency while skipping disabled full-test PRs"

python3 - "$WORKFLOW" <<'PY'
import contextlib
import io
import json
import os
import re
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import mock_open, patch

source = Path(sys.argv[1]).read_text()
source = textwrap.dedent(source.split('        shell: python\n        run: |\n', 1)[1].split('\n  notify:', 1)[0])


def approvals(prs, queued=(), running=(), waiting=(), branch="main", contributor="internal", limit=20):
    runs = {}
    approved = []
    by_status = {}
    for status, numbers in (("queued", queued), ("in_progress", running), ("waiting", waiting)):
        by_status[status] = []
        for number in numbers:
            run_id = len(runs) + 1
            run = {"id": run_id, "name": "CICD Megatron-LM", "display_title": str(number),
                   "head_branch": f"pull-request/{number}", "created_at": f"2026-01-01T00:00:{run_id:02d}Z"}
            runs[run_id] = number
            by_status[status].append(run)

    def response(data):
        return SimpleNamespace(status_code=200, links={}, json=lambda: data, raise_for_status=lambda: None)

    def get(url, **kwargs):
        if "actions/runs?" in url:
            status = re.search(r"status=([^&]+)", url).group(1)
            return response({"workflow_runs": by_status[status]})
        if "/pulls/" in url:
            number = int(url.rsplit("/", 1)[1])
            author, base = prs[number]
            return response({"number": number, "user": {"login": author}, "base": {"ref": base}, "labels": []})
        if url.endswith("/pending_deployments"):
            return response([{"id": 1, "environment": {"id": 1}}])
        raise AssertionError(url)

    def post(url, **kwargs):
        approved.append(runs[int(url.split("/runs/")[1].split("/")[0])])
        return response([{"state": "approved"}])

    sso = {author: {"org_roles": ["NVIDIA:Member"]} for author, _ in prs.values()} if contributor == "internal" else {}
    env = {"GITHUB_TOKEN": "test", "GITHUB_REPOSITORY": "NVIDIA/Megatron-LM",
           "CONTRIBUTOR_TYPE": contributor, "MAX_CONCURRENCY": str(limit * 2),
           "MAX_CONCURRENCY_EXTERNAL": str(limit), "SSO_USERS_FILE": "sso.json",
           "ALLOW_PR_RUN_ALL_TESTS": "true"}
    with patch.dict(os.environ, env), patch.dict(sys.modules, {"requests": SimpleNamespace(get=get, post=post)}), \
         patch("builtins.open", mock_open(read_data=json.dumps(sso))), contextlib.redirect_stdout(io.StringIO()):
        try:
            exec(compile(source.replace("${{ matrix.branch }}", branch), "queue", "exec"), {})
        except SystemExit as exc:
            assert exc.code == 0, exc.code
    return approved


prs = {1: ("alice", "main"), 2: ("alice", "main"), 3: ("alice", "main"),
       4: ("bob", "main"), 5: ("alice", "dev")}
# Existing active work and approvals in this pass both consume the author's slots.
assert approvals(prs, queued=[1], running=[2], waiting=[3, 4]) == [4]
assert approvals(prs, waiting=[1, 2, 3, 4]) == [1, 2, 4]
assert approvals(prs, running=[1], waiting=[2, 3, 4]) == [2, 4]
# A PR appearing in both status snapshots still consumes only one author slot.
assert approvals(prs, queued=[1], running=[1], waiting=[2, 3]) == [2]
# Different branches remain independent, including the global external queue.
assert approvals(prs, running=[1, 2], waiting=[3, 4, 5], contributor="external") == [4, 5]
assert approvals(prs, running=[1, 2], waiting=[5], branch="dev") == [5]
release_prs = {n: (author, "release/a" if base == "main" else "release/b")
               for n, (author, base) in prs.items()}
assert approvals(release_prs, running=[1, 2], waiting=[3, 4, 5], branch="others") == [4, 5]
# The existing queue-wide limit still applies.
assert approvals(prs, waiting=[1, 2, 4], limit=2) == [1, 2]
print("Approve Test Queue limits each author to two active PRs per base branch")
PY
