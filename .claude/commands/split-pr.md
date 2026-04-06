---
description: Split a PR into multiple PRs based on CODEOWNERS expert groups
argument-hint: <pr-url-or-number> [repo]
---

## Context

- CODEOWNERS file: @.github/CODEOWNERS

## Your task

The user wants to split a pull request into multiple, smaller PRs to reduce the number of required reviewer groups (as defined in CODEOWNERS).

Given the PR `$ARGUMENTS`:

### Step 1: Analyze the PR

1. Fetch the PR details and list all changed files using `gh pr view` and `gh pr diff --stat`
2. Parse the CODEOWNERS file to map each changed file to its owner group(s)
3. Build a table showing: file path, owner groups, and which "split group" it falls into
4. Identify the distinct CODEOWNERS groups involved and count how many files each group owns
5. Present this analysis to the user

### Step 2: Propose a split strategy

Based on the analysis:
1. Propose logical groupings that minimize the number of expert groups per PR
2. Each proposed PR must be independently mergeable to main (no broken imports, no missing dependencies)
3. If PR B depends on PR A's changes, note that PR B should target PR A's branch or be merged after PR A
4. Consider: changes within `megatron/core/` are often backward-compatible via aliases/stubs, making them natural candidates for a standalone PR
5. Present the proposed split to the user for approval before proceeding

### Step 3: Execute the split (after user approval)

For each new PR:
1. Create a new branch from the appropriate base (main, or a dependency branch)
2. Apply only the relevant file changes using `git diff` filtered by path
3. Verify the branch compiles/imports correctly if possible
4. Stage, commit, and push
5. Create the PR as a draft (per repo contributing guidelines)
6. Report the PR URL

### Important considerations

- Always create PRs as drafts
- Push to the user's fork, never directly to upstream
- Backward-compatible changes (aliases, re-exports) should go in the first PR so subsequent PRs can depend on them
- Test files should go with the production code they test
- Keep commits clean and logical - prefer a single well-described commit per split PR over replaying the original commit history
- If the original PR needs to be modified (e.g., reduced in scope), confirm with the user before force-pushing
