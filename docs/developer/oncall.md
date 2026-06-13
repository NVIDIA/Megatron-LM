<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# Oncall Overview

During your oncall week, you are assigned to all PRs marked “Ready for 
Review”. At a high level, your responsibilities include:

- Review all new PRs
- Accelerate the review process
- Ensure issues and discussion questions are answered

## PR Responsibilities

Go through the following checklist for each PR:

- Determine whether the PR should remain a single PR.
  - Each PR should have at most one expert reviewer, although there will be some outlier cases.
- Label PR as “complexity: low”, “complexity: medium”, or “complexity: high” depending on complexity.
  - Expert reviewers have final say, but oncall sets the initial complexity level.
  - Initial complexity level guidelines:
    - Low: <100 lines changed
    - Medium: 100 < lines changed < 500
    - High: > 500 lines changed
- Verify proper testing coverage.
  - If new logic is added, confirm it is tested.
- Verify the PR includes documentation for any new features.
- Verify the PR conforms to the project style guidelines.
  - Code structure
  - Cleanliness
  - Comments
  - File structure
- Confirm all tests pass.
  - Oncall kicks off the testing suite for external reviewers.
  - Comment “/ok to test commit_id” to kick off the testing suite.
- GitHub notifies expert reviewers after you mark the PR “Ready for Review”.
  - **Expert reviewers should review within one business day.** Message the assigned reviewer if it is taking longer. The reviewer either needs to review the PR or suggest an alternate reviewer.
  - If the reviewer is not responding after two business days, escalate to the reviewer’s manager.
- For `megatron/core` PRs, the “Final Review” label applies automatically after all expert reviewers approve.
  - Final reviewers should review within one business day. Message the assigned reviewer if it is taking longer.
  - If the reviewer is not responding after two business days, escalate to the reviewer’s manager.
- The “Approved” label applies automatically after all required reviewers approve.

## Issues and Discussion Questions

If you do not know the answer to an issue or discussion question, that is ok. **Delegate to someone who does.**

On a daily basis, track the following:

- [Dashboard for out of SLA issues](https://github.com/NVIDIA/Megatron-LM/issues?q=is%3Aissue%20state%3Aopen%20label%3Awaiting-on-maintainers).


