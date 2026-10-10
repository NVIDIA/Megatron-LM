<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# Oncall Overview

The oncall's primary responsibility is:

1. Helping community contributors and users
2. Helping the CI team resolve regressions from nightly or weekly runs

## Community Issues

**Goal: triage, assign, and ensure assignees respond in a timely manner.**

3-4 times per working day you should check if there are any new issues with the 
[community-request](https://github.com/NVIDIA/Megatron-LM/issues?q=is%3Aissue%20state%3Aopen%20label%3Acommunity-request) 
label. You should also check for issues that are out-of-SLA with the 
[waiting-on-maintainers](https://github.com/NVIDIA/Megatron-LM/issues?q=is%3Aissue%20state%3Aopen%20label%3Awaiting-on-maintainers%20sort%3Aupdated-desc) 
label.

We have a useful Claude tool that will send a Slack DM with context to the assignee:

- if you know who to assign: comment `/claude assign @gh-username`
- if you do not know who to assign: comment `/claude assign` and Claude will figure it out for you
  - the assignee may reach out to you if there is a mistake, do your best to find another assignee

## PR Responsibilities

**Goal: maintain our high-quality bar, launch CI, get approvals, and merge PRs.**

### PR Checklist

- Should the PR remain a single PR?
  - Each PR should have at most 1 expert reviewer, although there will be some outlier cases
- Does this PR have proper testing coverage?
  - If new logic is added, is the new logic tested?
- Should the PR add documentation for any new features?
- Does the PR conform to our style guidelines?
  - Code structure
  - Cleanliness
  - Comments
  - File structure

### Launch CI

Community contributors are unable to launch CI. If there is a basic merge conflict or lint error,
it is acceptable to fix it and re-launch CI (to reduce iteration time).

### Approvals and Merging

You may have to reach out to reviewers to help get approvals. Once the PR is fully-approved, 
please merge the PR! Community contributors are unable to do so.

## CI Regressions

**Goal: resolve nightly and weekly CI errors.**

Nightly and weekly CI tests do occasionally fail, typically due to a large divergence in loss, 
iteration time, or memory usage. Even improvements will cause CI to fail!

### Steps

1. Monitor CI Slack channel (#megatron-core-pipeline-alerts-main)
2. Work with CI to find root cause
3. Resolve
  - If it's a low-hanging fruit, try to fix immediately
  - If it's a severe blocker, revert and inform author
  - If not, we reach out to the author

### Tips

- Leverage the CI Dashboard
  - Find link in #megatron-core-pipeline-alerts-main channel description
  - You will need to join the `nemo-fw-eng` DL
- Setup the GitLab MCP server with Codex
