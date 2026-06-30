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

## Signing Your Work

### What is the DCO?

The Developer Certificate of Origin (DCO) is a lightweight, per-commit
declaration that you have the right to submit the contribution under the
project's open source license. It is **not** a copyright assignment and does
not transfer any rights; it simply certifies the origin of your work. The full
text is reproduced [below](#full-text-of-the-dco). By signing off on a commit,
you agree to its terms (a) through (d).

We require that **every** commit in a pull request is signed off. A pull
request that contains one or more commits without a valid `Signed-off-by`
trailer **will not be accepted**, and the automated DCO check will block it
from merging until all commits are signed.

### How to sign off on a commit

To sign off on a commit, add the `--signoff` (or `-s`) option when committing
your changes:

```bash
git commit -s -m "Add cool feature."
```

This appends a `Signed-off-by` trailer to your commit message:

```
Signed-off-by: Your Name <your@email.com>
```

The name and email in the trailer must match the `user.name` and `user.email`
configured in Git, and the email must be a real address you can be reached at
(do not use anonymized/no-reply addresses for the sign-off). Configure them
once with:

```bash
git config --global user.name "Your Name"
git config --global user.email "your@email.com"
```

### Fixing commits that are missing a sign-off

If the DCO check fails because one or more existing commits are not signed off,
amend or rebase them to add the trailer:

- To fix the most recent commit:

  ```bash
  git commit --amend --signoff
  git push --force-with-lease
  ```

- To sign off **every** commit on your branch (replace `main` with your base
  branch as needed):

  ```bash
  git rebase --signoff main
  git push --force-with-lease
  ```

After force-pushing, the DCO check re-runs automatically and should pass once
all commits carry a valid `Signed-off-by` trailer.

### Full text of the DCO

  ```
  Developer Certificate of Origin
  Version 1.1

  Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

  Everyone is permitted to copy and distribute verbatim copies of this
  license document, but changing it is not allowed.


  Developer's Certificate of Origin 1.1

  By making a contribution to this project, I certify that:

  (a) The contribution was created in whole or in part by me and I
      have the right to submit it under the open source license
      indicated in the file; or

  (b) The contribution is based upon previous work that, to the best
      of my knowledge, is covered under an appropriate open source
      license and I have the right under that license to submit that
      work with modifications, whether created in whole or in part
      by me, under the same open source license (unless I am
      permitted to submit under a different license), as indicated
      in the file; or

  (c) The contribution was provided directly to me by some other
      person who certified (a), (b) or (c) and I have not modified
      it.

  (d) I understand and agree that this project and the contribution
      are public and that a record of the contribution (including all
      personal information I submit with it, including my sign-off) is
      maintained indefinitely and may be redistributed consistent with
      this project or the open source license(s) involved.
  ```

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
