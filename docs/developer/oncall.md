# Oncall Overview

During your oncall week, you will be assigned to all PRs marked “Ready for 
Review”. From a high-level, your responsibilities include:

- Review all new PRs
- Accelerate the review process
- Ensure issues and discussion questions are answered

## PR Responsibilities

Below is the checklist that the oncall needs to go through for each PR.

- Should the PR remain a single PR?
  - Each PR should have at most 1 expert reviewer, although there will be some outlier cases
- Label PR as “complexity: low”, “complexity: medium”, or “complexity: high” depending on complexity
  - Expert reviewers have final say, oncall just sets the initial complexity level
  - Initial complexity level guideline
    - Low: <100 lines changed
    - Medium: 100 < lines changed < 500
    - High: > 500 lines changed
- Does this PR have proper testing coverage?
  - If new logic is added, is the new logic tested?
- Should the PR add documentation for any new features?
- Does the PR conform to our style guidelines?
  - Code structure
  - Cleanliness
  - Comments
  - File structure
- Do all tests pass?
  - Oncall will need to kick off testing suite for external reviewers
  - Comment “/ok to test commid_id” to kick off testing suite
- Add the “Expert Review” label
  - Select an expert reviewer from each expert group as a reviewer. If you’re unsure who to select, pick a “maintainer” or manager.
  - **Expert reviewers should review within 1 business day.** Message the assigned reviewer if it is taking longer. The reviewer either needs to review the PR or suggest an alternate reviewer.
  - If the reviewer is not responding after 2 business days, escalate to the reviewer's manager.
- Add the “Final Review” label after experts approve
  - Final reviewers should review within 1 business day. Message the assigned reviewer if it is taking longer.
  - If the reviewer is not responding after 2 business days, escalate to the reviewer's manager.

## Issues and Discussion Questions

On a daily basis, check for new [issues](https://github.com/NVIDIA/Megatron-LM/issues) 
and [discussions](https://github.com/NVIDIA/Megatron-LM/discussions). If you 
do not know how to answer that's ok! Delegate the issue or discussion to someone who does.