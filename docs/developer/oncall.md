# Oncall Overview

During your oncall week, you will be assigned to all PRs marked “Ready for Review”. From a high-level, your responsibilities include:

  - Review all new PRs
  - Uphold and enforce the Megatron coding standard
  - Accelerate the review process for expert reviewers (when necessary)

## Checklist

Below is the checklist that the oncall needs to go through for each PR.

- [ ] Should the PR remain a single PR?
  - Each PR should have at most 1 expert reviewer, although there will be some outlier cases
- [ ] Label PR as “complexity: low”, “complexity: medium”, or “complexity: high” depending on complexity
  - Low: <100 lines changed
  - Medium: 100 < lines changed < 500
  - High: > 500 lines changed
- [ ] Does this PR have proper testing coverage?
  - If new logic is added, is the new logic tested?
- [ ] Should the PR add documentation for any new features?
- [ ] Does the PR conform to our style guidelines?
  - Code structure
  - Cleanliness
  - Comments
  - File structure
- [ ] Do all tests pass?
  - Oncall will need to kick off testing suite for external reviewers
  - Comment “/ok to test commid_id” to kick off testing suite
- [ ] Add the “Expert Review” label
  - Expert reviewers should review within 1 business day. Message the assigned reviewer if it is taking longer.
  - After 2 business days, the expert reviewer waives the right to review.
- [ ] Add the “Final Review” label after experts approve
