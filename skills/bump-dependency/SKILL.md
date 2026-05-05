---
name: bump-dependency
description: Bump a pinned dependency (TransformerEngine, FlashMLA, nemo-run, nvidia-resiliency-ext, Emerging-Optimizers, container base image, etc.), regenerate `uv.lock`, open a draft PR from a fork, and drive it to green by attaching a watchdog to the `CICD Megatron-LM` workflow and quarantining flaky functional tests by suffixing their `scope` with `-broken` until the run is green.
when_to_use: Bumping a dependency pin in `pyproject.toml` `[tool.uv.sources]` or `docker/.ngc_version.{dev,lts}` and shepherding the PR to green. 'bump TE', 'bump transformer-engine', 'update TE pin', 'bump nemo-run', 'bump flash-mla', 'bump base image', 'update lock file', 'bump dependency PR', 'watch CI for a bump', 'quarantine flaky tests after bump', 'run all tests for this bump'.
---

# Bump Dependency

End-to-end workflow for shipping a dependency bump in Megatron-LM. Optimised
for the case where TE, FlashMLA, nemo-run, or the NGC base image moves
forward — bumps that often surface flakes which have to be quarantined
before the PR can land.

The pipeline is always: **edit → relock in container → push to fork →
draft PR → /ok to test → watchdog on `CICD Megatron-LM` → quarantine on
red (`scope: [...-broken]`) → re-trigger → repeat until green**.

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
- @skills/cicd/SKILL.md — pipeline structure, scope labels (`Run tests` / `Run functional tests` / `container::lts`), `pull-request/<N>` branch convention, `Nemo_CICD_Test` final gate.
- @skills/testing/SKILL.md — recipe YAML layout, the `-broken` scope suffix for disabling without deleting, flaky markers (`flaky_in_dev`, `flaky`).

## Step 1 — Worktree and edit

Branch off `github/main`; push to your personal fork (this repo's `origin`
points to `ko3n1g/Megatron-LM` — adjust if your fork is named differently):

```bash
# From the Megatron-LM repo root
git fetch github
git worktree add .claude/worktrees/<slug> -b ko3n1g/<tag>/<desc> github/main
```

Edit the pin. For TE the canonical knob is the `[tool.uv.sources]` entry
in `pyproject.toml`:

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

## Step 2 — Regenerate the lockfile (inside the CI container)

`uv.lock` is Linux + CUDA only and must be resolved **inside the CI
container** — never on the host. Acquire the image first via
@skills/build-and-dependency/SKILL.md, then:

```bash
docker run --rm \
  -v $(pwd):/workspace \
  -v $HOME/.cache/uv:/root/.cache/uv \
  -w /workspace \
  megatron-lm:local \
  bash -c 'uv lock'
```

If the bump is for the `lts` stack, also re-resolve against the lts image
(`megatron-lm:local-lts`) — `uv.lock` is shared across both groups but
the resolution must satisfy the `lts` extra too.

Confirm only the intended packages moved:

```bash
git diff --stat pyproject.toml uv.lock
git diff pyproject.toml | head
```

If the diff carries changes you didn't ask for (transitive movements you
can't explain), stop and investigate before pushing.

## Step 3 — Commit and push to your fork

Conventional Commits with the Megatron-LM scope convention. Sign the
commit (`-S`) and sign off (`-s`):

```bash
git add pyproject.toml uv.lock          # or docker/.ngc_version.* for base image
git commit -S -s -m "build: bump <package> to <ref>"
git push -u origin ko3n1g/<tag>/<desc>
```

Per @AGENTS.md, **never push branches to `NVIDIA/Megatron-LM` directly** —
push to your fork (`origin` here) and open the PR cross-repo.

## Step 4 — Open a draft PR

PR body goes through a tmpfile to preserve formatting. **Wrap it in a
`<details>` block.** PRs in this repo must be **drafts** (per @AGENTS.md):

```bash
cat > /tmp/pr-body.md <<'EOF'
<details><summary>Claude summary</summary>

## What
- Bump `<package>` to `<ref>`.
- Regenerate `uv.lock` inside the dev container.

## Lockfile delta
```
Updated <package> <old-sha> -> <new-sha>
```

## Test plan
- [ ] `CICD Megatron-LM` green (`Nemo_CICD_Test` gate)
- [ ] `Run functional tests` label applied (full matrix, golden-value compare)
- [ ] `container::lts` label applied if the bump can affect LTS

## Quarantined tests (this bump)
_None yet — will be appended as flakes are identified during CI iteration._

</details>
EOF

gh pr create \
  --repo NVIDIA/Megatron-LM \
  --base main \
  --head ko3n1g:ko3n1g/<tag>/<desc> \
  --draft \
  --title "build: bump <package> to <ref>" \
  --body-file /tmp/pr-body.md \
  --label "Run functional tests"
```

For a high-blast-radius bump (TE, FlashMLA, base image, anything that
touches CUDA / NCCL), also add **`container::lts`** so the LTS stack is
exercised in the same run. See the scope-label decision tree in
@skills/cicd/SKILL.md.

`gh pr edit` is unreliable. To update a PR's title or body later, use
the REST API directly (and always pass `--repo`):

```bash
gh api -X PATCH "repos/NVIDIA/Megatron-LM/pulls/<N>" \
  -F "body=@/tmp/pr-body.md"

gh api -X PATCH "repos/NVIDIA/Megatron-LM/pulls/<N>" \
  -f "title=build: bump <package> to <ref>"
```

## Step 5 — Trigger CI on the exact SHA

Megatron-LM's GitHub Actions pipeline runs on a `pull-request/<N>` branch
synced by `copy-pr-bot` after a maintainer comment. **For every new
commit you push to the PR, post `/ok to test <full-SHA>`**:

```bash
SHA=$(git rev-parse HEAD)
gh pr comment <N> --repo NVIDIA/Megatron-LM --body "/ok to test $SHA"
```

Use the **full** SHA (`git rev-parse HEAD`), never the short form. The
short form silently fails to match.

## Step 6 — Attach the watchdog (always; never a cronjob)

For a bump PR you want a single live process that emits per-job state
changes for the **`CICD Megatron-LM`** workflow only — that's the gate
that decides green-or-red. Other workflows (docs, copyright, claude-review,
sync-skills, mbridge-trigger) are noise here.

The final pass/fail gate inside that workflow is the **`Nemo_CICD_Test`**
job — when it goes green, the bump can merge.

**Always attach a watchdog with the Monitor tool. Never schedule wakeups
or cronjobs for this loop.** A watchdog gives you:

- Sub-minute reaction time on every job transition.
- A single live process — no scattered scheduled-wakeup state to reason
  about.
- Natural early termination via `TaskStop` once the run is green.

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

1. Pull the failing logs and confirm it's a flake / pre-existing issue,
   not the bump itself. Per @skills/cicd/SKILL.md, full per-rank logs
   are uploaded as artifacts; the runner stdout only has top-level
   output. The runner already retries known transients up to 3 times
   (NCCL timeout, ECC, segfault, HuggingFace) — if it still failed,
   investigate:

   ```bash
   RUN_ID=<from "RUN ... STARTED" event>
   gh run view "$RUN_ID" --repo NVIDIA/Megatron-LM --log-failed > /tmp/run.log
   tail -200 /tmp/run.log

   # If you need full per-rank logs, list and download artifacts:
   gh run view "$RUN_ID" --repo NVIDIA/Megatron-LM --json artifacts \
     --jq '.artifacts[].name'
   gh run download "$RUN_ID" --repo NVIDIA/Megatron-LM \
     --name "logs-<test_case>-<run_id>-<uuid>" -D /tmp/ci-logs
   ```

   If the failure is caused by the bump (real regression, not a flake),
   **stop quarantining** — fix the underlying issue or revert the bump.
   Quarantining a real regression hides the very signal the bump PR
   exists to surface.

2. **Quarantine by suffixing `-broken`, never delete the entry.** The
   functional-test recipes live in `tests/test_utils/recipes/{h100,gb200}/`
   and the parser skips any `scope` ending in `-broken`:

   ```yaml
   # before (test runs in CI)
   scope: [mr, mr-github]

   # after (test is skipped; entry preserved for easy re-enable)
   scope: [mr-broken, mr-github-broken]
   ```

   Map a failing CI job (e.g. `cicd-integration-tests-latest-h100 / gpt_<test_case>_dev_dgx_h100`)
   to its recipe by `model: gpt` and `test_case: <test_case>` in the YAML.
   For a unit-test bucket failure, add `@pytest.mark.flaky_in_dev` to the
   specific test (or `@pytest.mark.flaky` if it only flakes on `lts`) per
   @skills/testing/SKILL.md.

3. Append the test to the PR description's **Quarantined tests**
   section, with a one-line reason and a follow-up tracking link if you
   have one. This is the durable record of what this bump deferred.

4. Commit, push, retrigger:

   ```bash
   git commit -S -s -m "test: quarantine flaky <test_case> for <package> bump"
   git push
   SHA=$(git rev-parse HEAD)
   gh pr comment <N> --repo NVIDIA/Megatron-LM --body "/ok to test $SHA"
   ```

5. Update the PR body via `gh api PATCH` so the quarantine list stays
   current.

The watchdog is persistent — it will pick up the new run automatically
and emit `RUN <id> STARTED` for the new attempt.

## Step 8 — Stop when green

`GATE Nemo_CICD_Test -> success` is the exit condition. Then:

```bash
# Sanity check
gh pr checks <N> --repo NVIDIA/Megatron-LM | awk '{print $2}' | sort | uniq -c

# Tear down
TaskStop(<watchdog-task-id>)

# Tick the boxes in the PR body, then move out of draft
gh api -X PATCH "repos/NVIDIA/Megatron-LM/pulls/<N>" -F "body=@/tmp/pr-body.md"
gh pr ready <N> --repo NVIDIA/Megatron-LM
```

## Common pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| `uv lock` fails on host with CUDA / Linux marker errors | Resolved outside the container | Run `uv lock` inside `megatron-lm:local` (see @skills/build-and-dependency/SKILL.md) |
| CI never starts on a new push | `copy-pr-bot` hasn't synced — the `pull-request/<N>` branch doesn't exist yet, or you forgot `/ok to test` for the new SHA | Post `/ok to test $(git rev-parse HEAD)` for every new commit |
| Watchdog reports `RUN id STARTED` but no jobs | First poll happened before jobs were enumerated | Wait one cycle (60s); watchdog will fill in |
| Watchdog goes silent for 30+ min | `gh` rate-limited or auth expired | `gh auth status`; restart Monitor |
| Quarantine commit doesn't trigger a new run | Pushed but didn't post `/ok to test` for the new SHA | Always re-post on the new SHA |
| Wrong TE branch ref (`release/v2.15`) silently resolves nothing | TE uses `release_vX.Y` with an underscore | Verify with `git ls-remote` and pin a SHA, not a branch |
| Lockfile diff includes unrelated transitive bumps | Floor constraints in `pyproject.toml` floated when re-resolving | Re-run lock and accept; don't try to revert transitives |
| `gh pr create` fails with `head must be in the format owner:branch` | PR opened from a fork without the fork prefix | Use `--head ko3n1g:ko3n1g/<tag>/<desc>` |
| Push rejected to `NVIDIA/Megatron-LM` | Direct push to the canonical repo is not allowed | Push to your fork (`origin`) and open a cross-repo PR |

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
