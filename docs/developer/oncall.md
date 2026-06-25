<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# Oncall Overview

The oncall's primary responsibility is helping community contributors and users.

## Community Issues

**Goal: triage, assign, and ensure assignees respond in a timely manner.**

3-4 times per working day you should check if there are any new issues with the 
[community-request](https://github.com/NVIDIA/Megatron-LM/issues?q=is%3Aissue%20state%3Aopen%20label%3Acommunity-request) 
label.

We have a useful Claude tool that will send a Slack DM with context to the assignee:

- if you know who to assign: comment `/claude assign @gh-username`
- if you do not know who to assign: comment `/claude assign` and Claude will figure it out for you
  - the assignee may reach out to you if there is a mistake, do your best to find another assignee

## PR Responsibilities

**Goal: maintain our high-quality bar, launch CI, get approvals, and merge PRs.**

### PR Checklist

- [ ] Should the PR remain a single PR?
  - Each PR should have at most 1 expert reviewer, although there will be some outlier cases
- [ ] Does this PR have proper testing coverage?
  - If new logic is added, is the new logic tested?
- [ ] Should the PR add documentation for any new features?
- [ ] Does the PR conform to our style guidelines?
  - Code structure
  - Cleanliness
  - Comments
  - File structure

### Launch CI

Community contributors are unable to launch CI. If there is a basic merge conflict or lint errror,
it is acceptable to fix it and re-launch CI (to reduce iteration time).

### Approvals and Merging

You may have to reach out to reviewers to help get approvals. Once the PR is fully-approved, please
merge the PR! Community contributors are unable to do so.

## Out-of-SLA

On a daily basis, track the [out-of-SLA list](https://github.com/NVIDIA/Megatron-LM/issues?q=is%3Aissue%20state%3Aopen%20label%3Awaiting-on-maintainers).
