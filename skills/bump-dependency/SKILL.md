---
name: bump-dependency
description: Bump a pinned dependency (TransformerEngine, FlashMLA, nemo-run, nvidia-resiliency-ext, Emerging-Optimizers, NGC base image, etc.), regenerate `uv.lock`, open a draft PR from a fork, and drive it to green by attaching a watchdog to the `CICD Megatron-LM` workflow and quarantining flaky functional tests via the `-broken` scope suffix until the run is green.
when_to_use: Bumping a dependency pin in `pyproject.toml` `[tool.uv.sources]` or `docker/.ngc_version.{dev,lts}` and shepherding the PR to green. 'bump TE', 'bump transformer-engine', 'update TE pin', 'bump nemo-run', 'bump flash-mla', 'bump base image', 'update lock file', 'bump dependency PR', 'watch CI for a bump', 'quarantine flaky tests after bump', 'run all tests for this bump'.
---

# Bump Dependency

End-to-end workflow for shipping a dependency bump in Megatron-LM.
Optimised for the case where TE, FlashMLA, nemo-run, or the NGC base
image moves forward — bumps that often surface flakes which have to be
quarantined before the PR can land.

The pipeline is always: **edit → relock in container → push to fork →
draft PR → /ok to test → watchdog on `CICD Megatron-LM` → quarantine on
red (`scope: [...-broken]`) → re-trigger → repeat until `Nemo_CICD_Test`
is green**.

## When to reach for this skill

- Bumping a git-source pin in `pyproject.toml` `[tool.uv.sources]`
  (e.g. `transformer-engine = { git = ..., rev = "<new-sha>" }`).
- Bumping the NGC base image pin in `docker/.ngc_version.dev` or
  `docker/.ngc_version.lts`.
- Any change that touches `pyproject.toml` or `uv.lock` and needs the
  full functional-test matrix to prove out before merge.

For pure dep additions/removals without a CI loop, the
`build-and-dependency` skill is enough.

## Required context

Read first, then follow the steps below:

- @AGENTS.md — PRs must be drafts; push to a personal fork, never `NVIDIA/Megatron-LM` directly.
- @skills/build-and-dependency/SKILL.md — `uv lock` mechanics, container choice, `dev` vs `lts`.
- @skills/cicd/SKILL.md — pipeline structure, scope labels (`Run tests` / `Run functional tests` / `container::lts`), `pull-request/<N>` branch convention, `Nemo_CICD_Test` final gate, `copy-pr-bot` trust + log/artifact retrieval.
- @skills/testing/SKILL.md — recipe YAML layout, the `-broken` scope suffix for disabling without deleting, flaky markers (`flaky_in_dev`, `flaky`).

## Step 1 — Worktree and edit

Branch off `github/main` and push to your fork per @AGENTS.md. Edit the
pin. For TE the canonical knob is the `[tool.uv.sources]` entry in
`pyproject.toml`:

```toml
[tool.uv.sources]
transformer-engine = { git = "https://github.com/NVIDIA/TransformerEngine.git", rev = "<new-sha>" }
```

Always pin to a **full commit SHA** for reproducibility, not a branch
name. TE branches use `release_vX.Y` (underscore) — verify any branch
ref with `git ls-remote https://github.com/NVIDIA/TransformerEngine.git`
and resolve to a SHA before locking.

For a base-image bump, edit `docker/.ngc_version.dev` (and/or `.lts`) — a
single line containing the NGC PyTorch image tag.

## Step 2 — Regenerate the lockfile

Run `uv lock` inside the project container per
@skills/build-and-dependency/SKILL.md "Regenerating uv.lock". If the
bump is for the `lts` stack, also re-resolve against the `lts` image —
`uv.lock` is shared but the resolution must satisfy the `lts` extra too.

Confirm only the intended packages moved:

```bash
git diff --stat pyproject.toml uv.lock
```

If the diff carries changes you didn't ask for (transitive movements you
can't explain), stop and investigate before pushing. Floor constraints
in `pyproject.toml` will float unrelated transitives — accept those,
don't try to revert them.

## Step 3 — Commit and push to your fork

Conventional Commits + sign-off + signed commit per @AGENTS.md and
@skills/cicd/SKILL.md "Commit and PR Workflow". For a bump:

```bash
git add pyproject.toml uv.lock         # or docker/.ngc_version.* for base image
git commit -S -s -m "build: bump <package> to <ref>"
git push -u origin <your-branch>
```

## Step 4 — Open a draft PR

Title and labels per @skills/cicd/SKILL.md. Two bump-specific requirements:

- The PR **must be a draft** (`--draft`) and pushed cross-repo from your
  fork (per @AGENTS.md): `--head <your-username>:<branch>`.
- Apply **`Run functional tests`** — the matrix-expand label. Add
  **`container::lts`** for any high-blast-radius bump (TE, FlashMLA,
  base image, anything that touches CUDA / NCCL) so the LTS stack is
  exercised in the same run.

The PR body template — this is the durable record of the bump:

```markdown
<details><summary>Claude summary</summary>

## What
- Bump `<package>` to `<ref>`.
- Regenerate `uv.lock` inside the dev container.

## Lockfile delta
```
Updated <package> <old-sha> -> <new-sha>
```

## Test plan
- [ ] `Nemo_CICD_Test` gate green
- [ ] `Run functional tests` label applied (full matrix, golden-value compare)
- [ ] `container::lts` label applied if the bump can affect LTS

## Quarantined tests (this bump)
_None yet — will be appended as flakes are identified during CI iteration._

</details>
```

To update the PR title or body later, use `gh api -X PATCH
"repos/NVIDIA/Megatron-LM/pulls/<N>" -F "body=@/tmp/pr-body.md"` —
never `gh pr edit`.

## Step 5 — Trigger CI on the exact SHA

Trigger mechanics + `pull-request/<N>` mirror branch live in
@skills/cicd/SKILL.md "How CI Is Triggered". For this loop the rule is
simple: **on every new SHA you push, post `/ok to test
$(git rev-parse HEAD)`** as a PR comment. Use the **full** SHA — the
short form silently fails to match.

## Step 6 — Attach the watchdog (always; never a cronjob)

For a bump PR you want a single live process that emits per-job state
changes for the **`CICD Megatron-LM`** workflow only. Other workflows
(docs, copyright, claude-review, sync-skills, mbridge-trigger) are noise
here.

The final pass/fail gate inside that workflow is the **`Nemo_CICD_Test`**
job — when it goes green, the bump can merge.

**Always attach a watchdog with the Monitor tool. Never schedule wakeups
or cronjobs for this loop.** A watchdog gives you:

- Sub-minute reaction time on every job transition.
- A single live process — no scattered scheduled-wakeup state to reason
  about.
- Natural early termination via `TaskStop` once the gate is green.

### Watchdog script

The `pull-request/<N>` branch is created by `copy-pr-bot` after the first
`/ok to test`. Save to `/tmp/watchdog-<PR>.sh` and chmod +x:

```bash
#!/usr/bin/env bash
# Watchdog: monitor "CICD Megatron-LM" runs on pull-request/<PR> and emit
# per-job state changes. Stays alive across re-runs (new commits).
set -u
PR=<PR>
REPO=NVIDIA/Megatron-LM
BRANCH="pull-request/$PR"
GATE_JOB="Nemo_CICD_Test"

prev_run_id=""
declare -A prev_state

emit() { echo "[$(date -u +%H:%M:%SZ)] $*"; }

while true; do
  run_json=$(gh run list --repo "$REPO" --workflow "CICD Megatron-LM" \
    --branch "$BRANCH" --limit 1 \
    --json databaseId,status,conclusion,headSha 2>/dev/null || echo "[]")
  run_id=$(echo "$run_json" | jq -r '.[0].databaseId // empty')
  run_status=$(echo "$run_json" | jq -r '.[0].status // empty')
  run_conclusion=$(echo "$run_json" | jq -r '.[0].conclusion // empty')
  run_sha=$(echo "$run_json" | jq -r '.[0].headSha // empty')

  if [[ -z "$run_id" ]]; then
    sleep 30; continue
  fi

  if [[ "$run_id" != "$prev_run_id" ]]; then
    emit "RUN ${run_id} STARTED sha=${run_sha:0:8} status=${run_status}"
    prev_run_id="$run_id"
    unset prev_state
    declare -A prev_state
  fi

  jobs_json=$(gh run view "$run_id" --repo "$REPO" --json jobs 2>/dev/null || echo "{}")
  while IFS=$'\t' read -r name status conclusion; do
    [[ -z "$name" ]] && continue
    cur="${status}/${conclusion}"
    if [[ "${prev_state[$name]:-}" != "$cur" ]]; then
      case "$status" in
        completed)
          emit "JOB ${name} -> ${conclusion}"
          if [[ "$name" == "$GATE_JOB" ]]; then
            emit "GATE ${GATE_JOB} -> ${conclusion}"
          fi ;;
        in_progress)
          if [[ -z "${prev_state[$name]:-}" || "${prev_state[$name]}" == "queued/" ]]; then
            emit "JOB ${name} -> in_progress"
          fi ;;
      esac
      prev_state[$name]="$cur"
    fi
  done < <(echo "$jobs_json" | jq -r '.jobs[]? | [.name, .status, (.conclusion // "")] | @tsv')

  if [[ "$run_status" == "completed" ]]; then
    emit "RUN ${run_id} COMPLETED conclusion=${run_conclusion}"
  fi

  sleep 60
done
```

### Arming the watchdog

```text
Monitor(
  description="CICD Megatron-LM run state changes on PR <N>",
  command="bash /tmp/watchdog-<N>.sh",
  persistent=true,
  timeout_ms=3600000
)
```

`persistent: true` keeps it alive across re-runs (you'll push more
commits when quarantining flakes). Stop it with `TaskStop(<task-id>)`
once `GATE Nemo_CICD_Test -> success` fires.

### Why never a cronjob / scheduled wakeup

- Cronjobs run blind — they fire on a clock, not on an event. You'll
  either over-poll (cache miss every wake-up) or miss long stalls.
- Wakeups can't easily fan out to "tell me whenever a job transitions"
  — they only resume the agent on a fixed interval.
- A persistent Monitor surfaces every job edge in real time and exits
  cleanly when the work is done.

## Step 7 — Quarantine on red, then iterate

When a `JOB <name> -> failure` event fires:

1. **Triage the failure — is it the bump or a flake?** Pull logs (and
   per-rank artifacts if needed) per @skills/cicd/SKILL.md "CI Failure
   Investigation". The runner already retries known transients up to
   3 times — if it still failed, the issue is real. Only quarantine if
   the failure reproduces on `main` or is clearly unrelated
   infrastructure. If the failure is caused by the bump itself
   (real regression), **stop quarantining** — fix the underlying issue
   or revert the bump. Quarantining a real regression hides the very
   signal the bump PR exists to surface.

2. **Quarantine via the `-broken` scope suffix** per
   @skills/testing/SKILL.md "Disabling a recipe". Map a failing job
   (e.g. `cicd-integration-tests-latest-h100 / gpt_<test_case>_dev_dgx_h100`)
   to its recipe by `model: gpt` and `test_case: <test_case>` in
   `tests/test_utils/recipes/{h100,gb200}/*.yaml`. **Never delete the
   entry** — the suffix preserves it for trivial re-enable. For unit-test
   buckets, mark `@pytest.mark.flaky_in_dev` (or `@pytest.mark.flaky`
   if it only flakes on `lts`).

3. **Append to the PR body's Quarantined tests section** with a one-line
   reason and a follow-up tracking link if you have one. This is the
   durable record of what this bump deferred — the section exists
   precisely so a reviewer can see at a glance which flakes were
   side-stepped to land the bump.

4. **Commit, push, retrigger**:

   ```bash
   git commit -S -s -m "test: quarantine flaky <test_case> for <package> bump"
   git push
   gh pr comment <N> --repo NVIDIA/Megatron-LM \
     --body "/ok to test $(git rev-parse HEAD)"
   ```

5. **Update the PR body** via `gh api PATCH` so the quarantine list
   stays current.

The watchdog is persistent — it picks up the new run automatically and
emits `RUN <id> STARTED` for the new attempt. Loop back to step 1.

## Step 8 — Stop when green, mark non-draft

`GATE Nemo_CICD_Test -> success` is the exit condition. Then:

```bash
gh pr checks <N> --repo NVIDIA/Megatron-LM | awk '{print $2}' | sort | uniq -c
TaskStop(<watchdog-task-id>)
gh api -X PATCH "repos/NVIDIA/Megatron-LM/pulls/<N>" -F "body=@/tmp/pr-body.md"
gh pr ready <N> --repo NVIDIA/Megatron-LM
```

The final `gh pr ready` flips the PR out of draft so reviewers can pick
it up.

## Common pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| Wrong TE branch ref (`release/v2.15`) silently resolves nothing | TE uses `release_vX.Y` with an underscore | Verify with `git ls-remote` and pin a SHA, not a branch |
| Lockfile diff includes unrelated transitive bumps | Floor constraints in `pyproject.toml` floated when re-resolving | Re-run lock and accept; don't try to revert transitives |
| `gh pr create` fails with `head must be in the format owner:branch` | PR opened from a fork without the fork prefix | Use `--head <your-username>:<branch>` |
| Watchdog reports `RUN id STARTED` but no jobs | First poll happened before jobs were enumerated | Wait one cycle (60s); watchdog will fill in |

## Anti-patterns

- **Cron / scheduled wakeups for this loop.** Always Monitor.
- **Polling all workflows.** Filter to `CICD Megatron-LM` — the rest are
  noise for a bump.
- **Quarantining a real regression** to "make CI green." That defeats
  the purpose of the bump PR. Only quarantine if the failure reproduces
  on `main` or is clearly unrelated infrastructure.
- **Deleting a recipe entry to disable a test.** Always suffix the
  `scope` with `-broken` so the entry stays in the file and is trivial
  to re-enable.
- **`gh pr edit`** for title/body. Use `gh api PATCH`.
- **HEREDOC in `gh pr create --body`.** Always go through a tmpfile +
  `--body-file`.
- **Pushing the bump branch to `NVIDIA/Megatron-LM` directly.** Push to
  your fork; open the PR cross-repo as a draft (per @AGENTS.md).
- **Opening the PR non-draft.** `--draft` is mandatory in this repo.
- **Bundling unrelated changes** (feature work, refactors) into a bump
  PR. Bumps should stay surgical so CI failures attribute cleanly.
