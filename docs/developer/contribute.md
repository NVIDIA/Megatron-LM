# Contributing to Megatron-LM

This document outlines the processes and policies for issues and pull requests by non-NVIDIA contributors to the Megatron-LM GitHub repository.

Everyone is welcome to contribute to the project! We recently migrated from using an internal repo to doing all development directly from the GitHub repository.

When contributing it is important to ensure that changes are in line with the project direction. Small changes to fix bugs are welcomed and appreciated. If proposing large architectural changes or changes for stylistic reasons open an issue first so we can discuss it.

## Issue policy

Please do file any bugs you find, keeping the following in mind:

- If filing a bug, i.e. you have found something that doesn't work as expected, use the BUG template.
- If you've found a regression in speed or accuracy use the REGRESSION template.
- If you are requesting a new feature or modification of an existing feature use the ENHANCEMENT template.
- If opening an issue to ask a question no template is needed but please make your question as clear and concise as possible.
- One issue per bug. Putting multiple things in the same issue makes both discussion and completion unnecessarily complicated.
- Your bug is mostly likely to get attention from the development team quickly if we can easily reproduce it.
- Use proper spelling, grammar, and punctuation.
- Write in an authoritative and technical tone.

## Code submission policy

### Do

- Format new code in a style that is consistent with the file being changed. Megatron-LM doesn't (yet) have a style guide or enforced formatting.
- Split your changes into separate, atomic commits i.e. A commit per feature or fix.
- Make sure your commits are rebased on the master branch.
- Write the commit message subject line in the imperative mood ("Change the default argument for X", not "Changed the default argument for X").
- Write your commit messages in proper English, with care and punctuation.
- Check the spelling of your code, comments and commit messages.

### Don't

- Submit code that's incompatible with the project licence.
- Touch anything outside the stated scope of the PR. This includes formatting changes to code not relevant to the PR.
- Iterate excessively on your design across multiple commits.
- Include commented-out code.
- Attempt large architectural changes without first opening an issue to discuss.

## Issue and Pull Request Q&A

### I've submitted an issue and PR. When can I expect to get some feedback?

You should receive a response within 2 business days.

### I need help, who should I ping?

Use [@mcore-oncall](https://github.com/orgs/NVIDIA/teams/mcore-oncall).

### If my issue or PR isn't getting attention, what should I do?

After 2 business days, tag the user [@mcore-oncall](https://github.com/orgs/NVIDIA/teams/mcore-oncall).

### Is there a policy for issues and PRs that haven't been touched in X days? Should they be closed?

Yes, we have a bot that will mark untouched PRs as "stale" after 60 days.

We have a long backlog of issues and PRs dating back years. We are trying to triage these now by working backwards. Older issues we believe may still be relevant may recieve a request to re-test them with the latest code. If there's no response they may be closed. Again, if you they should be re-opened then just respond with a comment to that effect.

Thank you!