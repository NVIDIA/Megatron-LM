---
name: batch-and-test-mergequeue-prs
description: Batch fully approved Megatron-LM PRs that are actual merge-queue entries, approve their batch CI gate, and validate with Run Merge-Queue jobs. Squash-merge the originals after Nemo_CICD_Test succeeds and queue membership and approvals are rechecked. Excludes PRs that only have auto-merge enabled.
license: Apache-2.0
---

# Batch and Test Merge-Queue PRs

Use one batch PR to validate a specific set of source PR heads together, then
squash-merge the originals so GitHub records each PR as merged. A full invocation
includes approving the batch's CI gate and conditionally landing the sources
after `Nemo_CICD_Test` succeeds. Honor narrower requests such as selection or
testing only. Follow repository `AGENTS.md` and the user's requested scope
throughout.

## Select the sources

- Read the live `NVIDIA/Megatron-LM` main merge queue through GraphQL
  `repository.mergeQueue(branch: "main").entries`; paginate the connection.
  Keep `AWAITING_CHECKS`, `QUEUED`, and blocked entries distinct.
- Select sources only from that connection. Query each PR through GraphQL and
  require `isInMergeQueue: true`, with a non-null `mergeQueueEntry` in
  `AWAITING_CHECKS` or `QUEUED` state. Record the flag, entry ID, state, and
  observed source head. A false or missing flag is not verified membership.
  Do not select blocked entries or widen the pool when it is small.
- PRs with `autoMergeRequest` but no `mergeQueueEntry` are pending auto-merges,
  not actual queue entries. An `Approved` label alone does not imply either state.
  Exclude them from batching, even if they are mergeable or have native approvals.
- Require full source approval on the selected head: native
  `reviewDecision=APPROVED` and successful `codeowners-approval` and
  `multi-approval-bot-summary` checks. Missing, pending, skipped, or failed
  approval checks make a source ineligible. Native `APPROVED`, an `Approved`
  label, and a green `Nemo_CICD_Test` do not substitute for these checks.
- Honor supplied PRs, counts, and exclusions. If selecting a batch, inspect the diffs,
  review state, current CI, and dependencies. Prefer small fixes with regression
  coverage, limited default-behavior changes, and little file overlap. A small
  diff or complexity label is not sufficient evidence of low risk.
- Recommend **5 PRs per batch**, normally **3–7**, weighing each PR's risk,
  interactions, file overlap, and test coverage. Favor the lower end when risk
  or uncertainty increases; approach 7 only for independent, well-covered changes.
  Exceed 7 only for very trivial changes. If fewer than 3 suitable candidates are
  available, report the shortage rather than adding risky PRs to reach a count.
- Record each source PR number, full head SHA, author/coauthors, relevant labels,
  changed files, queue-entry evidence, and exact-head approval-check results.
  Before publishing, recheck that the PR is open, its head is unchanged,
  GraphQL still reports `isInMergeQueue: true` with an eligible queue entry,
  and its approval checks still pass.
  If any check fails, stop and report it; do not substitute pending auto-merges.

Use this per-PR GraphQL query to verify membership and the source head during
selection, before publishing, and before dequeueing for an authorized merge:

```graphql
query ($number: Int!) {
  repository(owner: "NVIDIA", name: "Megatron-LM") {
    pullRequest(number: $number) {
      number
      state
      baseRefName
      headRefOid
      isInMergeQueue
      reviewDecision
      mergeQueueEntry { id state position }
    }
  }
}
```

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

Use only `Run Merge-Queue jobs` as the batch's test-scope label. It selects the
same configuration branch as an actual merge-group event: L1 functional tests
with one repetition, normal training steps and golden comparisons, merge-group
cadence filtering, full unit buckets, dev images, AWS H100 routing, MBridge L1,
and no NeMo RL downstream run. Do not inherit source PR scope, image, or
downstream labels. Platform eligibility and PR approval controls still apply.
See [mcore-cicd](../mcore-cicd/SKILL.md) for the shared configuration reference.

Apply labels before CI starts. Labels alone do not trigger the workflow. After
pushing, use `/ok to test` when required by the contribution workflow; draft
auto-sync may be disabled. Verify that a `CICD Megatron-LM` run appears on
`pull-request/<batch PR>` for the pushed head.

- Inspect the configure job's actual outputs: expect `merge_queue_jobs=true`,
  `scope=L1`, `lightweight=false`, `n_repeat=1`, `cadence=mergegroup`,
  `cadence_bypass=false`, `unit_testmon_eligible=false`, `lts=false`,
  `run_functional=true`, `run_h100_functional=true`, `run_mbridge=true`,
  `mbridge_suite=L1`, and `run_nemo_rl=false`. GB200 functional jobs retain their
  existing approval, unit-test, maintainer, and feature gates; downstream feature
  gates still apply.
- CI may test a synthetic merge SHA rather than the source head. Record it and
  compare its tree with the expected source patches on its actual upstream base.
  Do not call a different or stale revision validated.
- Approve the batch's `cicd-wait-in-queue` job by approving its pending `test`
  environment deployment. Verify the exact run ID, `pull-request/<batch PR>`
  ref, and full batch head first; use the live pending deployment's environment
  ID and review state `approved`. This approval is part of the full workflow.
  Keep repository-wide full-test switches, concurrency limits, environment
  protections, and other runs' pending deployments unchanged. If approval is
  unavailable, report the gate and the reason. Verify that the job completes
  successfully and subsequent CI starts; an approval request alone is not test
  success.
- Monitor that exact batch run through `Nemo_CICD_Test`. Its completed `success`
  result triggers the fresh source checks and squash merges below. Keep
  monitoring through this decision and conditional landing; a queued build,
  pending approval, or skipped core job is not a passing result.
- When monitoring, report failure and cancellation as terminal states alongside
  success. Identify downstream runs by their exact parent/run metadata; a parent
  finishing or timing out does not guarantee its downstream CI stopped.
- For failures, inspect the failing case and per-rank artifacts, compare against
  main or a matching baseline, and separate assertions from cancellations and
  environment failures. A previously passing test does not prove a new failure
  is unrelated. A targeted retry can check reproducibility; avoid repeated
  retries of the same deterministic failure or unrequested changes to tolerances.

## Land and clean up after core CI succeeds

Once the exact batch run's `Nemo_CICD_Test` completes successfully, re-read the
live main merge queue and source PRs before landing. Failure, cancellation, or a
skipped core job stops landing. A narrower user request to prepare or test only
takes precedence. Team notifications require an explicit request.

For each original source:

- If a source has already merged independently, verify and record its expected
  patch, then continue with the remaining open sources.
- Recheck that every remaining source is OPEN, non-draft, based on `main`, and
  still at its validated head with native `reviewDecision=APPROVED`. Require
  membership in the live GraphQL main queue connection, `isInMergeQueue: true`,
  and a matching `mergeQueueEntry` in `AWAITING_CHECKS` or `QUEUED` state.
  Require successful exact-head `codeowners-approval` and
  `multi-approval-bot-summary` checks again immediately before dequeueing or
  merging. If any remaining source loses membership, changes head, or lacks
  approval, stop and report it. Batch CI success does not waive source review
  requirements.
- Treat upstream-main drift as informational. Record the tested base and current
  main, then continue landing the approved source patches on current main.
  Watchers must continue when main advances, without path/blob allowlists,
  additional approval, or a CI rerun solely for that drift. Refresh the merge
  plan if main advances during preparation. Verify each resulting patch against
  its actual merge parent; keep the frozen batch's CI tree proof separate from
  verification of the landed patch. Source head, approval, queue membership,
  and actual merge-conflict checks still apply.
- Resolve dependencies before merging: open PRs based on
  `pull-request/<source PR>` must be retargeted to
  `main`, as required by repository guidance.
- Remove a verified source from the merge queue when needed for a direct merge.
  Squash-merge each original PR through GitHub with an exact head guard, such as
  `gh pr merge <number> --repo NVIDIA/Megatron-LM --squash --match-head-commit <SHA>`.
  An authorized administrative bypass may handle queue mechanics or redundant
  source CI only after the full source approval requirements above are verified.
  Never use it to bypass missing or failing source approval checks.
- Verify every original PR is marked merged and each resulting single-parent
  commit contains the expected patch. Merging only the batch does not mark the
  originals merged in the GitHub UI. Close the redundant batch after landing.
- Stop obsolete PR, merge-group, and downstream runs within the authorized scope.
  Match PR numbers, refs, full heads, and downstream parent IDs; exclude unrelated
  main/release workflows. Verify both run and job states are terminal after
  cancellation, rather than relying on the cancellation request alone.
- Report the landed PRs, merge commits, CI result and any remaining validation
  limits. Keep an audit of cancelled runs when cleanup is requested.
