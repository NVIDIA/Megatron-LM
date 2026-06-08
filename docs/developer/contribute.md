<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# Contributing to Megatron-LM

This document outlines the processes and policies for issues and pull requests by non-NVIDIA contributors to the Megatron-LM GitHub repository.

Everyone is welcome to contribute. The project recently migrated from an internal repository to doing all development directly on GitHub.

When contributing, it is important to ensure that changes are in line with the project direction. Small bug fixes are always welcome. **If proposing large architectural changes or changes for stylistic reasons, open an issue first for discussion.**

## Issue Policy

File any bugs you find, keeping the following in mind:

- If filing a bug, that is, you have found something that does not work as expected, use the `BUG` template.
- If you've found a regression in speed or accuracy, use the `REGRESSION` template.
- If you are requesting a new feature or modification of an existing feature, use the `ENHANCEMENT` template.
- If opening an issue to ask a question, you do not need a template, but make your question as clear and concise as possible.
- One issue per bug. Putting multiple things in the same issue makes both discussion and completion unnecessarily complicated.
- Reproducible bugs get the fastest attention from the development team.
- Use proper spelling, grammar, and punctuation.
- Write in an authoritative and technical tone.

## Code Submission Policy

### Do

- Format new code in a style that is consistent with the file being changed. Megatron-LM doesn't (yet) have a style guide or enforced formatting.
- Split your changes into separate, atomic commits, that is, a commit per feature or fix.
- Make sure your commits are rebased on the `main` branch.
- Write the commit message subject line in the imperative mood ("Change the default argument for X", not "Changed the default argument for X").
- Write your commit messages in proper English, with care and punctuation.
- Check the spelling of your code, comments, and commit messages.

### Don't

- Submit code that's incompatible with the project license.
- Touch anything outside the stated scope of the PR. This includes formatting changes to code not relevant to the PR.
- Iterate excessively on your design across multiple commits.
- Include commented-out code.
- Attempt large architectural changes without first opening an issue to discuss.

## Issue and Pull Request Q&A

### Response Time for Issues and PRs

You should receive a response within two business days.

### Getting Help

Use `@NVIDIA/mcore-oncall`.

### Escalating Unresponsive Issues or PRs

After two business days, tag `@NVIDIA/mcore-oncall`.

### Stale Issue and PR Policy

A bot marks untouched PRs as "stale" after 60 days.

A long backlog of issues and PRs dates back years, and triage works backwards through it. Older issues that may still be relevant receive a request to re-test with the latest code. Without a response, the issue may be closed. To request reopening, respond with a comment.

Thank you.
