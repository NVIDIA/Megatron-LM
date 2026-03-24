<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# Claude Commands

This repository uses [Claude Code Action](https://github.com/anthropics/claude-code-action) to provide AI-assisted automation via PR comments. **All commands are triggered by commenting on a pull request.** At this time, all Claude commands can only be triggered by NVIDIA-internal contributors.

## Code Review

### `/claude review`

Performs a light code review on the current PR.

**Where to use:** Any PR.

**What it does:**
- Reviews the PR diff for critical bugs, logic errors, and typos
- Checks for missing or insufficient test coverage
- Flags outdated documentation affected by the changes
- Posts inline comments for specific suggestions and top-level comments for general observations
- Posts "LGTM" if no issues are found

## Branch Sync

The following commands automate the process of porting PRs from non-main branches (e.g., `dev`, `release/*`) to `main`. This eliminates the need to manually create and maintain duplicate PRs across branches.

:warning: These commands can **only** be used on PRs targeting non-main branches. If used on a PR targeting `main`, Claude will post a comment explaining why.

### `/claude copy`

Creates a new draft PR targeting `main` that ports the changes from the current PR.

**Where to use:** PRs targeting non-main branches.

**What it does:**
1. Checks if a port of this PR already exists (open or closed) targeting `main`
2. Creates a new branch (`claude/port-pr-<number>`) from `main`
3. Applies the PR's diff to the new branch, resolving conflicts where possible
4. Opens a **draft** PR targeting `main` with:
   - A summary of the ported changes
   - A conflict resolution summary (if any conflicts were resolved)
5. Posts a sticky comment on the source PR linking to the new PR

### `/claude copy and track`

Same as `/claude copy`, but also enables automatic tracking. When tracking is enabled, future changes to the source PR can be synced to the port PR via `/claude sync` or automatically when new commits are pushed.

**Where to use:** PRs targeting non-main branches.

### `/claude track <pr_number>`

Links the current PR to an existing PR targeting `main` and enables automatic tracking. Use this when the `main` PR was already created manually or by another process.

**Where to use:** PRs targeting non-main branches.

**Requirements:**
- The target PR must exist and be open
- The target PR must target `main`

**Example:**
```
/claude track 456
```

### `/claude sync`

Syncs the latest changes from the source PR to the linked port PR. This can be triggered manually or automatically when tracking is enabled.

**Where to use:** PRs targeting non-main branches that have a linked port PR (via `/claude copy`, `/claude copy and track`, or `/claude track`).

**What it does:**
1. Reads the sticky comment to find the linked port PR
2. Applies the latest changes to the port PR's branch, resolving conflicts where possible
3. Posts a summary on both the source and target PRs

**Automatic syncing:** When tracking is enabled (via `/claude copy and track` or `/claude track`), `/claude sync` is posted automatically whenever new commits are pushed to the source PR.

### `/claude untrack`

Disables automatic tracking for the current PR. The linked port PR is not affected — it remains open but will no longer receive automatic syncs.

**Where to use:** PRs targeting non-main branches with tracking enabled.
