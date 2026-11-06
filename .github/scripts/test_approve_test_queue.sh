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

python3 - "$WORKFLOW" <<'PY'
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

workflow = Path(sys.argv[1]).read_text()
assert 'MAX_CONCURRENCY_DEV: ${{ vars.MAX_CONCURRENCY_DEV }}' in workflow
assert 'MAX_CONCURRENCY_OTHERS: ${{ vars.MAX_CONCURRENCY_OTHERS }}' in workflow
assert 'TARGET_BRANCH: ${{ matrix.branch }}' in workflow
start = workflow.index('          if CONTRIBUTOR_TYPE == "external":')
end = workflow.index('          API_BASE =', start)
limit_code = compile(textwrap.dedent(workflow[start:end]), sys.argv[1], "exec")


def limit(branch, contributor="internal", dev=None, others=None, shared="8"):
    env = {
        "TARGET_BRANCH": branch,
        "MAX_CONCURRENCY": shared,
        "MAX_CONCURRENCY_EXTERNAL": "3",
    }
    for name, value in (("MAX_CONCURRENCY_DEV", dev), ("MAX_CONCURRENCY_OTHERS", others)):
        if value is not None:
            env[name] = value
    namespace = {"os": SimpleNamespace(environ=env), "CONTRIBUTOR_TYPE": contributor}
    exec(limit_code, namespace)
    return namespace["MAX_CONCURRENCY"]


for branch in ("dev", "others"):
    for override, expected in ((None, 4), ("", 4), (" ", 4), ("0", 0), ("1", 1), ("2", 2)):
        assert limit(branch, **{branch: override}) == expected
    assert limit(branch, shared="9") == 4
    for override in ("-1", "invalid", "1.5"):
        try:
            limit(branch, **{branch: override})
        except ValueError:
            pass
        else:
            raise AssertionError(f"Accepted invalid {branch} concurrency: {override!r}")
assert limit("dev", dev="1", others="invalid") == 1
assert limit("others", dev="invalid", others="2") == 2
for override in ("0", "1", "invalid"):
    assert limit("main", dev=override, others=override) == 4
    assert limit("all", contributor="external", dev=override, others=override) == 3
PY

python3 - "$WORKFLOW" <<'PY'
import re
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

workflow = Path(sys.argv[1]).read_text()
start = workflow.index("          # Headers for GitHub API")
end = workflow.index("          # Get current running and queued workflows", start)
helpers_code = compile(textwrap.dedent(workflow[start:end]), sys.argv[1], "exec")


class RequestException(Exception):
    def __init__(self, response=None) -> None:
        super().__init__("request failed")
        self.response = response


class Response:
    def __init__(self, status, headers=None, body=None) -> None:
        self.status_code = status
        self.headers = headers or {}
        self.links = {}
        self.text = ""
        self._body = body

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RequestException(self)


def load(responses):
    """Run the workflow helpers against a scripted sequence of API responses."""
    state = SimpleNamespace(requests=[], sleeps=[])

    def send(*args, **kwargs):
        state.requests.append(args[0])
        return responses.pop(0)

    def fail_exit(code=0):
        raise SystemExit(code)

    fake_requests = SimpleNamespace(
        get=send,
        post=send,
        exceptions=SimpleNamespace(
            ConnectionError=ConnectionError, Timeout=TimeoutError, RequestException=RequestException
        ),
    )
    fake_time = SimpleNamespace(time=lambda: 1000, sleep=state.sleeps.append)
    state.ns = {
        "GITHUB_TOKEN": "token",
        "API_BASE": "https://api.example",
        "INTERNAL_SERVICE_ACCOUNTS": set(),
        "sso_users": {},
        "re": re,
        "requests": fake_requests,
        "time": fake_time,
        "exit": fail_exit,
    }
    exec(helpers_code, state.ns)
    return state


def limited(remaining="0", reset="1009"):
    return Response(403, {"X-RateLimit-Remaining": remaining, "X-RateLimit-Reset": reset})


# Primary rate limit sleeps until the reset, then returns the successful result.
state = load([limited(), Response(200, body={"ok": True})])
assert state.ns["make_request"]("x") == {"ok": True}
assert state.sleeps == [10] and len(state.requests) == 2

# Waiting exactly 60s retries; 61s fails the job without any further request.
state = load([limited(reset="1059"), Response(200, body={"ok": True})])
assert state.ns["make_request"]("x") == {"ok": True} and state.sleeps == [60]
state = load([limited(reset="1060"), Response(200, body={"ok": True})])
try:
    state.ns["make_request"]("x")
except SystemExit as e:
    assert e.code == 1
else:
    raise AssertionError("Long rate-limit wait must fail the job")
assert state.sleeps == [] and len(state.requests) == 1

# Ordinary errors are not treated as rate limits.
assert load([Response(403)]).ns["make_request"]("x") is None
assert load([Response(403, {"X-RateLimit-Remaining": "5"})]).ns["make_request"]("x") is None

# 429 honors Retry-After, and keeps exponential backoff when the header is missing.
state = load([Response(429, {"Retry-After": "3"}), Response(200, body={"ok": True})])
assert state.ns["make_request"]("x") == {"ok": True} and state.sleeps == [3]
state = load([Response(429), Response(429), Response(200, body={"ok": True})])
assert state.ns["make_request"]("x") == {"ok": True} and state.sleeps == [1, 2]

# Successful GET/POST results and return_response are preserved.
response = Response(200, body={"ok": True})
assert load([response]).ns["make_request"]("x", return_response=True) is response
assert load([Response(200, body=[1])]).ns["make_request"]("x", method="POST", data={}) == [1]


def pr(number, base="main", labels=()):
    return Response(
        200,
        body={"base": {"ref": base}, "labels": [{"name": n} for n in labels], "user": {"login": "u"}},
    )


# PR lookups are reused across branch, label, and contributor checks.
state = load([pr(1, labels=("Run tests",)), pr(2, base="dev")])
run_1 = {"head_branch": "pull-request/1"}
run_2 = {"head_branch": "pull-request/2"}
for _ in range(3):
    assert state.ns["get_pr_base_branch"](run_1)[0] == "main"
    assert state.ns["get_pr_labels"](run_1) == {"Run tests"}
    assert state.ns["matches_contributor"](run_1, "external")
assert len(state.requests) == 1
assert state.ns["get_pr_base_branch"](run_2)[0] == "dev"
assert len(state.requests) == 2

# A failed lookup is not cached, and non-PR runs make no request.
state = load([Response(404), pr(1)])
assert state.ns["get_pr_base_branch"](run_1) == (None, None)
assert state.ns["get_pr_base_branch"](run_1)[0] == "main"
assert len(state.requests) == 2
assert state.ns["get_pr_base_branch"]({"head_branch": "main"}) == (None, None)
assert len(state.requests) == 2
PY

echo "Approve Test Queue preserves branch-specific concurrency while skipping disabled full-test PRs"

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
           "TARGET_BRANCH": branch,
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
