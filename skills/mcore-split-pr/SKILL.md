---
name: mcore-split-pr
description: Split a PR into multiple PRs to reduce the number of required CODEOWNERS reviewer groups, and manage the resulting stack through review and merge.
license: Apache-2.0
when_to_use: User asks to split a PR, reduce reviewer groups, break up a large PR, or stack dependent PRs; 'too many CODEOWNERS', 'split this PR', 'break up PR', 'reduce reviewers needed', 'stacked PRs', 'clean per-PR diffs'.
user_invocable: true
argument: "PR URL or number"
metadata:
  author: Philip Petrakian <ppetrakian@nvidia.com>
---

# Split PR by CODEOWNERS Groups

Split a large pull request into multiple smaller PRs, where each PR touches
the fewest possible CODEOWNERS reviewer groups. The goal is to reduce review
burden: a PR that only touches `megatron/core/` needs only the core reviewers,
while a PR that also touches `examples/`, `tools/`, and `megatron/training/`
pulls in many additional groups.

## Answer-First Constraints

For split-planning questions, lead with these constraints before the full
workflow:

- Minimize CODEOWNERS reviewer groups per PR, but each resulting PR must still
  be independently mergeable and reviewable.
- Tests travel with the production code they validate; do not split tests into a
  separate PR just to reduce reviewer groups.
- If PR B depends on symbols renamed in PR A, call out the dependency and put
  backward-compatible aliases, re-exports, or shims in PR A when needed.
- **Create every PR with base `main`.** Stacked bases come later: the
  `pull-request/<N>` mirror refs used for stacking do not exist until a vetter
  runs `/ok to test`, so creating a PR with such a base fails.
- **Never merge a PR whose base is a `pull-request/*` ref.** GitHub merges into
  the base branch: merging while mirror-based writes the commits into the
  bot's scratch ref (which gets force-pushed away), lands nothing on `main`,
  and leaves an unreopenable MERGED PR. Retarget to `main` first, then merge.
- Wait for user approval before execution.
- Execution creates draft PRs, applies file-scoped diffs with
  `git diff upstream/main..<source-branch> -- <paths> | git apply`, pushes
  to the user's fork, and never pushes directly to upstream.

## How Megatron-LM's CI shapes stacked PRs

Understand this model before touching PR bases; every stacking step below
derives from it.

- CI runs on self-hosted runners, so fork PRs get **no CI** until a trusted
  vetter comments `/ok to test <head-sha>` (copy-pr-bot). The bot then copies
  the vetted SHA into a real upstream branch, `refs/heads/pull-request/<N>`,
  and workflows run against that trusted copy — never against the fork ref.
- That mirror branch is the **only** upstream ref containing a fork PR's
  commits, and GitHub requires a PR's base to be an upstream branch — which is
  why `pull-request/<parent-N>` is the one legal way to get stacked
  (per-layer) diffs in a fork-only repo.
- The mirror is a **vetted snapshot, not a live mirror**: it refreshes only on
  the next `/ok to test`. After pushing to a parent, child diffs go slightly
  stale until the parent is re-vetted. This is the security model working as
  intended, not a bug.
- The `linting` job checks the PR **merged with current main**, using main's
  tool pins (e.g. black version from main's `pyproject.toml`), over changed
  `.py` files under `megatron/core` and `tests/` only. A touched file must be
  fully clean under main's formatter version — pre-existing lines can fail
  after a formatter bump on main, and a local older formatter will not
  reproduce the complaint.
- Merges are **squash** merges: a child's diff does not collapse automatically
  when its parent merges. The child must be rebased onto the new `main`
  (the parent's commits drop out as already-applied) and force-pushed.

## Workflow

### 1. Analyze the PR

1. Fetch the PR details: `gh pr view <number> --repo NVIDIA/Megatron-LM --json title,body,headRefName,author` and `gh pr diff <number> --repo NVIDIA/Megatron-LM --stat`. Also determine the current GitHub user with `gh api user --jq .login`.
2. Parse `.github/CODEOWNERS` to build a mapping from file path patterns to owner groups.
3. For each changed file in the PR, determine which CODEOWNERS groups would be required to review it.
4. Build a summary table grouped by CODEOWNERS group, showing which files pull in which groups.
5. Count the total number of distinct reviewer groups the PR currently requires.

### 2. Propose a split that minimizes reviewer groups per PR

The primary optimization goal: **minimize the number of CODEOWNERS reviewer groups required for each resulting PR**.

Strategy:
1. Cluster files by their CODEOWNERS groups. Files owned by the same set of groups naturally belong together.
2. Identify the largest cluster — this becomes the first (and usually largest) PR.
3. Remaining files form one or more additional PRs, each ideally requiring only one or two reviewer groups.
4. If a split creates a dependency (e.g., PR B uses symbols renamed in PR A), the dependent PR must be merged after the first. Note this explicitly.
5. Each PR must be independently mergeable to main — no broken imports, no missing symbols. Backward-compatible aliases and re-export stubs in the first PR can make this possible.
6. A PR with **two dependencies** cannot have a clean stacked diff (a git branch
   has one parent). Linearize: stack it on one parent's branch and cherry-pick a
   copy of the other parent's commit beneath it. Its diff shows the copied
   commit until that other parent merges to `main`, after which the next rebase
   drops the copy and the diff collapses — note this in the PR body.

Present the proposed split as a table:
- PR name/description
- Files included
- CODEOWNERS groups required
- Dependencies on other PRs (if any)

Wait for user approval before proceeding.

### 3. Execute the split (after user approval)

For each new PR:
1. Create a new branch from the appropriate local base (`main`, or a dependency PR's branch).
2. Extract the relevant changes: `git diff upstream/main..<source-branch> -- <file paths> | git apply`.
3. Stage, commit with `-s -S` and a clear message, and push to the user's fork.
4. Create the PR as a **draft** with base `main` (per repo contributing
   guidelines; the mirror refs for stacking do not exist yet).
5. If the original PR needs to be narrowed in scope, confirm with the user before force-pushing.
6. Report all PR URLs when done.

### 4. Stack for review (once a vetter is available)

1. Have a vetter comment `/ok to test <head-sha>` on every PR in the series —
   this both unblocks CI and creates the `pull-request/<N>` mirrors.
2. Confirm the refs exist: `git ls-remote origin 'refs/heads/pull-request/<N>'`.
3. Retarget each child onto its parent's mirror:
   `gh pr edit <child> --base pull-request/<parent-N>`.
   Roots keep base `main`.
4. Each PR's Files-changed now shows only its own layer. Reviews proceed in
   parallel across the whole stack.
5. While reviewing: pushes to a child require rebasing its descendants
   (cascade + force-push). Do **not** use GitHub's "Update branch" button on
   stacked branches — it injects merge commits that fight the rebase cascade.

### 5. Merge (bottom-up through the dependency DAG)

Reviews are parallel; merges are strictly parents-before-children. For each
PR whose dependencies have all merged:

1. **Retarget to `main`:** `gh pr edit <N> --base main`. Never click merge
   while the header says "into `pull-request/...`".
2. Rebase the branch onto latest `origin/main` if needed; push; re-vet for CI.
3. Squash-merge into `main`.
4. Immediately rebase every descendant onto the new `main` and force-push
   (the merged parent's commits drop out as already-applied).
5. The next child's diff has now collapsed to its own layer; retarget it to
   `main` when its turn comes, and repeat.

Independent roots can merge at any time, in any order — the ordering
constraint is the dependency DAG, not the PR-number sequence.

## Important guidelines

- Always create PRs as **drafts** and push to the user's fork, never directly to upstream.
- Backward-compatible changes (aliases, re-exports, deprecation shims) should go in the first PR so subsequent PRs can depend on them.
- Test files should go with the production code they test, not in a separate PR.
- Prefer a single clean commit per split PR over replaying the original commit history.
- If a file is hard to categorize (e.g., it touches two groups), ask the user which PR it should go in.
- If the current GitHub user is not the author of the original PR, each new PR's description must explicitly credit the original author (e.g., "Original changes by @<author> in #<number>").
- Consider a watchdog for the two stacking failure modes: a PR reaching
  approved state while still mirror-based (merge hazard), and a parent merging
  (a rebase cascade is now due).
