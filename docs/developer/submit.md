<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# How to Submit a PR

All PRs start as **draft**. If you open a non-draft PR, it will be automatically converted to draft.

## Step 1: Mark PR as "Ready for Review"

1. When your PR is ready, click **Ready for Review**.
2. The oncall reviewer is auto-assigned and expert reviewers are notified based on your changes. They will get notified and pick up your PR soon.

:warning: Only mark as ready once all merge-conflicts are resolved and the CI is passing.
Final Review might get declined if these requirements are not fulfilled.

## Step 2: Final Review (`megatron/core` only)

For PRs that change `megatron/core`, once all expert reviewers have approved, the `Final Review` label is applied **automatically** and final reviewers are assigned.

For PRs outside `megatron/core`, this step is skipped.

## Step 3: Approved

Once all required reviewers have approved, the `Approved` label is applied **automatically**. The PR is now ready to merge.

## Step 4: Merge

Any member of [mcore-engineers](https://github.com/orgs/NVIDIA/teams/mcore-engineers) will be able to merge your PR.
