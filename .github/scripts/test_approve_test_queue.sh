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

echo "Approve Test Queue preserves branch-specific concurrency while skipping disabled full-test PRs"
