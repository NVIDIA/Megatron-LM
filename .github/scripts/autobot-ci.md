# Autobot CI request/result smoke test

The jobs in `cicd-main.yml` exercise an asynchronous request to Autobot
through GitHub. The request job finishes; GitHub delivers its signed
`workflow_job.completed` webhook; Autobot writes an `Autobot CI selection` check
on that exact commit; the waiting job reads its JSON and uploads it as an
`autobot-ci-result-<run_id>-<attempt>` Actions artifact. Neither job connects to
an Autobot endpoint or holds an Autobot credential.

## Enable and run

Deploy the matching CI-Events-Workflows hello responder first and confirm its
GitHub integration receives Megatron-LM `workflow_job` events. The responder
admits only `NVIDIA/Megatron-LM`, the `cicd-main.yml` workflow, and canonical
`pull-request/<number>` mirror refs. Running this workflow in a personal fork
does not invoke the responder.

Set the repository variable `AUTOBOT_CI_APP_ID` to the numeric GitHub App ID
of Autobot's result publisher. This is public identity metadata, not a secret.
It must be independently configured: the waiter never trusts an arbitrary App
that publishes a check with the expected name.

To exercise an unmerged draft PR, push its signed commit to the personal fork,
open the PR against NVIDIA/Megatron-LM, and request the normal copy-pr-bot
mirror with `/ok to test <full head SHA>`. Drafts are not automatically synced.
The mirror push automatically runs the hello integration. To repeat the test,
manually dispatch the existing `cicd-main.yml` on that mirror ref:

```sh
gh workflow run cicd-main.yml --repo NVIDIA/Megatron-LM \
  --ref "pull-request/<PR-number>"
```

This draft runs only the request and waiter jobs. The real CI jobs, including
Docker builds, linting, downstream/GPU tests, coverage and the final test gate,
are temporarily commented out in `cicd-main.yml`. Restore them before using this
workflow for production CI.

## Request and result identity

The request is the successful job named `Autobot CI request (hello)`. Its identity
is `autobot-ci:v1:<repository>:<run_id>:<run_attempt>:<job_id>`. Both endpoints
derive it from GitHub job metadata. The check's `external_id` carries that
identity, and its `output.text` is plain JSON:

```jsonc
// Autobot CI selection check output.text (wire example)
{
  "schema_version": 1,
  "kind": "hello",
  "request": {
    "repository": "NVIDIA/Megatron-LM",
    "run_id": 123,
    "run_attempt": 1,
    "job_id": 456,
    "request_id": "autobot-ci:v1:NVIDIA/Megatron-LM:123:1:456",
    "head_sha": "<40-character commit SHA>",
    "base_sha": "<40-character PR base SHA>",
    "pull_request_number": 789
  },
  "result": {"message": "Hello from Autobot"}
}
```

The waiter requires the configured publisher App, exact request/commit, successful
check conclusion, supported schema/kind and matching payload. It paginates with
`filter=all` so an older or concurrent same-SHA attempt cannot replace the result.
The wait has a ten-minute deadline inside a twelve-minute CPU job. Missing,
ambiguous, malformed or failed results fail this smoke-test job; they never
silently count as a response. This draft does not run the real CI test suite.

The result is also available as `needs.wait-for-autobot.outputs.result`, but no
existing job consumes it. A future test-recommendation implementation can keep
the event, identity, Checks transport and artifact path while adding a supported
result kind/schema and review-Agent reasoning over the pinned base/head. Its
consumer must validate test identifiers against the actual test inventory and
fall back to normal selection on unavailable/invalid results. Never evaluate
model output as shell commands or workflow expressions.

## Local verification

```sh
python3 -m unittest discover -s .github/scripts -p test_wait_for_autobot.py -v
```
