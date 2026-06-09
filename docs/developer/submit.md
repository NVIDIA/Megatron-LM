<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# How to Submit a PR

All PRs start as **draft**. If you open a non-draft PR, the system automatically converts it to draft.

## Step 1: Mark PR as "Ready for Review"

1. When your PR is ready, click **Ready for Review**.
2. The system auto-assigns the oncall reviewer and notifies expert reviewers based on your changes.

:warning: Only mark as ready after all merge-conflicts are resolved and the CI is passing.
Reviewers may decline the Final Review if you have not met these requirements.

## Step 2: Final Review (`megatron/core` only)

For PRs that change `megatron/core`, after all expert reviewers approve, the system **automatically** applies the `Final Review` label and assigns final reviewers.

For PRs outside `megatron/core`, this step is skipped.

## Step 3: Approved

After all required reviewers approve, the system **automatically** applies the `Approved` label. The PR is now ready to merge.

## Step 4: Merge

Any member of [mcore-engineers](https://github.com/orgs/NVIDIA/teams/mcore-engineers) will be able to merge your PR.
