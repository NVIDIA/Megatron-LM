---
name: triage-issue
description: Investigate a failing GitHub Actions run or job and create a GitHub issue for the failure. Use when the user shares a GitHub Actions URL and wants to file a bug report for the CI failure.
user_invocable: true
argument: "<github-actions-run-or-job-url>"
---

# Triage CI Failure into a GitHub Issue

Investigate a failing GitHub Actions job, extract the root cause, and file a
well-structured bug issue against `NVIDIA/Megatron-LM`.

## Workflow

### 1. Parse the URL

The argument is a GitHub Actions URL. It will be one of:

- **Job URL**: `https://github.com/<owner>/<repo>/actions/runs/<run_id>/job/<job_id>`
- **Run URL**: `https://github.com/<owner>/<repo>/actions/runs/<run_id>`

Extract `run_id` and, if present, `job_id`.

### 2. Identify failed jobs

- If a `job_id` was provided, use that job directly.
- If only a `run_id` was provided, list all failed jobs in the run:

  ```bash
  gh run view <run_id> --repo NVIDIA/Megatron-LM --json jobs \
    --jq '[.jobs[] | select(.conclusion == "failure") | {id: .databaseId, name: .name, url: .url}]'
  ```

  If multiple jobs failed, ask the user which one to triage, or triage all of them if they say so.

### 3. Fetch the failure logs

For each failed job, retrieve the logs and narrow them down to the failure:

```bash
# Pull the raw log and keep only error-bearing lines
gh api repos/NVIDIA/Megatron-LM/actions/jobs/<job_id>/logs 2>&1 \
  | grep -E "(FAILED|ERROR|\bError\b|assert|Traceback|Exception|##\[error\])" \
  | head -200
```

Also capture the full job name:

```bash
gh run view --job <job_id> --repo NVIDIA/Megatron-LM --json name --jq .name
```

If the grep output is sparse, download the full logs and look for the pytest
`FAILURES` section or the last non-zero exit signal.

### 4. Extract the root cause

From the logs, identify:

- **Failed test(s)**: lines matching `FAILED tests/...::...` give the exact pytest node IDs.
- **Error message**: the assertion failure, exception type, or first meaningful
  traceback frame — keep it under ~30 lines.
- **Job name**: the GitHub Actions job name (e.g. `tests/unit_tests/transformer/moe/**/*.py - latest`).
- **Run / job URLs**: for linking in the issue.

### 5. Check for duplicate issues

Search for open issues that already cover the same test:

```bash
gh issue list --repo NVIDIA/Megatron-LM \
  --state open \
  --search "<failed-test-filename>" \
  --json number,title,url \
  --limit 10
```

- If a matching open issue exists, **do not create a new one**. Report the
  existing issue to the user and stop.
- If no match is found, proceed to file a new issue.

### 6. Create the issue

```bash
gh issue create \
  --repo NVIDIA/Megatron-LM \
  --title "🐛 CI failure: <failed-test-node-id>" \
  --label "bug" \
  --body "..."
```

Use the bug-report template body structure:

```markdown
**Describe the bug**

CI test `<failed-test-node-id>` failed in job [`<job-name>`](<job-url>).
Tag the [@mcore-oncall](https://github.com/orgs/NVIDIA/teams/mcore-oncall) to get oncall's attention to this issue.

**Failing run**

| Field | Value |
|-------|-------|
| Run   | [<run_id>](<run_url>) |
| Job   | [<job_name>](<job_url>) |

**Error**

```
<core error message / traceback — 30 lines max>
```

**Steps/Code to reproduce bug**

Re-run the failing CI job linked above, or locally inside the dev container:

```bash
pytest <failed-test-node-id>
```

**Additional context**

Triaged automatically via `/triage-issue`.
```

If multiple tests failed in the same job, list each one as a separate bullet
under "Describe the bug" and include the combined error snippets.

### 7. Report back to the user

Print the URL of the newly created issue (or the duplicate, if found) so the
user can review or share it.

## Important guidelines

- Never create an issue if a duplicate already exists — link the existing one instead.
- Keep the error snippet concise (≤30 lines). Truncate long tracebacks and note that the full log is available via the job URL.
- Do not guess the root cause — quote the actual log output verbatim.
- If the job is still in progress or the logs are unavailable, say so and ask the user to retry once the run completes.
