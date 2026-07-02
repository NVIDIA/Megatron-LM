<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# How to Submit a PR

All PRs start as **draft**. If you open a non-draft PR, it will be automatically converted to
draft.

## Step 1: Mark PR as "Ready for Review"

1. When your PR is ready, click **Ready for Review**.
2. Expert reviewers are notified based on your changes. They will get notified and pick up your
PR soon.

:warning: Only mark as ready once all merge-conflicts are resolved and the CI is passing.
Final Review might get declined if these requirements are not fulfilled.

## Step 2: Final Review (`megatron/core` only)

For PRs that change `megatron/core`, once all expert reviewers have approved, the `Final Review`
label is applied **automatically** and final reviewers are expected to review. This is intended to
be a more lightweight review to ensure the repository's standard is upheld.

For PRs outside `megatron/core`, this step is skipped.

## Step 3: Approved

Once all required reviewers have approved, the `Approved` label is applied **automatically**. The
PR is now ready to merge.

## Step 4: Merge

Any member of [mcore-engineers](https://github.com/orgs/NVIDIA/teams/mcore-engineers) will be able
to merge your PR.

## FAQ

### How does an expert review group get assigned?

The mapping from directory or file to GitHub team is set in
[.github/CODEOWNERS](https://github.com/NVIDIA/Megatron-LM/blob/main/.github/CODEOWNERS).

### What is the difference between expert reviewers and final reviewers?

Final review groups are [core-nemo](https://github.com/orgs/NVIDIA/teams/core-nemo) and
[core-adlr](https://github.com/orgs/NVIDIA/teams/core-adlr). All other groups are considered
expert groups.

### What should I do if my PR is not getting reviewed?

#### Internal Contributors

1. Mention review groups (e.g. @mcore-hybrid-model) in the #megatron-core-developments Slack
channel
2. DM a maintainer of the review group asking for a review
3. Schedule a meeting with a maintainer to go review the PR together
4. DM the [mcore-oncall](https://github.com/orgs/NVIDIA/teams/mcore-oncall) in Slack.

Any other questions? Reach out to the
[mcore-oncall](https://github.com/orgs/NVIDIA/teams/mcore-oncall)!

#### External Contributors

Mention the [mcore-oncall](https://github.com/orgs/NVIDIA/teams/mcore-oncall) in your PR or issue.
The oncall's main priority is helping external contributors and users!
