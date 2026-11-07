---
name: batch-and-test-mergequeue-prs
description: Select queued or pending Megatron-LM PRs, combine their changes into an attributed squash batch, and validate it with functional CI. Also covers landing the original PRs and stopping obsolete CI when requested.
license: Apache-2.0
---

# Batch and Test Merge-Queue PRs

Use one batch PR to validate a specific set of source PR heads together. Keep the
source PRs independently mergeable so GitHub can record their eventual merges.
Follow repository `AGENTS.md` and the user's requested scope throughout.

## Select the sources

- Read the live `NVIDIA/Megatron-LM` main merge queue through GraphQL
  `repository.mergeQueue(branch: "main").entries`; paginate the connection.
  Keep `AWAITING_CHECKS`, `QUEUED`, and blocked entries distinct.
- PRs with `autoMergeRequest` but no `mergeQueueEntry` are pending auto-merges,
  not actual queue entries. An `Approved` label alone does not imply either state.
  Report that distinction when counting or widening the candidate pool.
- Honor supplied PRs and exclusions. If selecting a batch, inspect the diffs,
  review state, current CI, and dependencies. Prefer small fixes with regression
  coverage, limited default-behavior changes, and little file overlap. A small
  diff or complexity label is not sufficient evidence of low risk.
- Record each source PR number, full head SHA, author/coauthors, relevant labels,
  and changed files. Recheck that the PR is open and its head is unchanged before
  publishing the batch. Do not silently substitute a later revision.

## Assemble an attributed batch

- Verify the remotes and fetch upstream `main` and the source heads. Create a
  dedicated worktree and branch from upstream main. In the usual local setup,
  `github` is upstream and `origin` is the personal fork; inspect before writing.
- Apply each PR's complete diff relative to its upstream merge base, retaining
  binary files. Use one squash commit per source PR, signed and signed off with
  `git commit -S -s`. Preserve original authorship, coauthors, and signoff trailers;
  include the source PR URL and full source head in the commit body.
- Preserve source patches when resolving integration. Record any necessary
  adjustments beyond them in a separate signed integration-fix commit. Explain
  the affected sources and resulting behavior in the batch description.
- Run the repository formatter/lint checks before committing. Verify commit
  signatures, `git diff --check`, and the final tree against independent patch
  application. Record the upstream base and source heads used for validation.
- Check determinism coverage. An existing source exemption can explain that
  source's gate result; verify other sources separately so a batch-wide label
  does not conceal missing coverage for a new kernel change. Do not add an
  exemption merely to obtain a green batch.

## Open and run the batch

Push only to the personal fork and create a draft PR against upstream `main`.
Add the `Batch PR` label. Describe the source PRs and exact heads, upstream base,
conflict adjustments, and concrete validation commands/results. Keep internal
hosts out of public descriptions, comments, and commit messages. Update the
description after pushes; pass bodies through files or structured API payloads.

Use these scope labels according to the requested validation:

| Label | Functional scope | Repetitions | Mode |
| --- | --- | --- | --- |
| `Run functional tests once` | Full L1 | 1 | Normal steps and golden comparisons |
| `Run functional tests` | Full L1 | 5 | Normal steps and golden comparisons |
| `Run tests` | L1 | 1 | Lightweight, four steps without golden comparisons |

For ordinary batch validation, prefer `Run functional tests once` unless the
user requests repetition. It takes precedence when another scope label is also
present. Add downstream labels only for the requested coverage or relevant
integration; LTS remains opt-in. See [mcore-cicd](../mcore-cicd/SKILL.md) for scope,
image, and downstream controls.

Apply labels before CI starts. Labels alone do not trigger the workflow. After
pushing, use `/ok to test` when required by the contribution workflow; draft
auto-sync may be disabled. Verify that a `CICD Megatron-LM` run appears on
`pull-request/<batch PR>` for the pushed head.

- Inspect the configure job's actual outputs: expect `scope=L1`,
  `lightweight=false`, the requested `n_repeat`, `run_functional=true`,
  `run_h100_functional=true`, `cadence_bypass=true`, and
  `unit_testmon_eligible=false` for a full batch run. `run_functional` controls
  GB200 functional selection; platform eligibility still applies.
- CI may test a synthetic merge SHA rather than the source head. Record it and
  compare its tree with the expected source patches on its actual upstream base.
  Do not call a different or stale revision validated.
- Check the `test` environment gate. Honor the approval queue's full-test switch,
  concurrency limits, and any user authorization to override them. Report a
  pending gate or queued build accurately; requesting CI is not a passing result.
- When monitoring, report failure and cancellation as terminal states alongside
  success. Identify downstream runs by their exact parent/run metadata; a parent
  finishing or timing out does not guarantee its downstream CI stopped.
- For failures, inspect the failing case and per-rank artifacts, compare against
  main or a matching baseline, and separate assertions from cancellations and
  environment failures. A previously passing test does not prove a new failure
  is unrelated. A targeted retry can check reproducibility; avoid repeated
  retries of the same deterministic failure or unrequested changes to tolerances.

## Land and clean up when requested

Do not infer permission to merge, cancel unrelated runs, or notify a team from
permission to create and test a batch.

When the user asks to land the sources:

- Recheck each source's current head against the validated head and review/merge
  requirements. Inspect any upstream-main drift. Resolve dependencies before
  merging: open PRs based on `pull-request/<source PR>` must be retargeted to
  `main`, as required by repository guidance.
- Remove a source from the merge queue when needed for a requested direct merge.
  Squash-merge each original PR through GitHub with an exact head guard, such as
  `gh pr merge <number> --repo NVIDIA/Megatron-LM --squash --match-head-commit <SHA>`.
  Use an authorized administrative bypass if direct landing requires it.
- Verify every original PR is marked merged and each resulting single-parent
  commit contains the expected patch. Merging only the batch does not mark the
  originals merged in the GitHub UI. Close the redundant batch after landing.
- Stop obsolete PR, merge-group, and downstream runs within the authorized scope.
  Match PR numbers, refs, full heads, and downstream parent IDs; exclude unrelated
  main/release workflows. Verify both run and job states are terminal after
  cancellation, rather than relying on the cancellation request alone.
- Report the landed PRs, merge commits, CI result and any remaining validation
  limits. Keep an audit of cancelled runs when cleanup is requested.
