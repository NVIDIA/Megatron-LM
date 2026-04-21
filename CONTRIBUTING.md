# Contributing to Megatron-LM

This document outlines the processes and policies for issues and pull requests by non-NVIDIA contributors to the Megatron-LM github repository.

Everyone is welcome to contribute to the project but development of Megatron-LM continues internally at NVIDIA. When contributing it important to ensure that changes are in line with the project direction. Small changes to fix bugs are welcomed and appreciated. If proposing large architectural changes or changes for stylistic reasons open an issue first so we can discuss it.

PRs will first be pulled into NVIDIA's internal Megatron-LM repo and then pushed back out to the open github repo with proper credit given to the committers.

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

Here are some dos & don'ts to try and stick to:

### Do:

- Format new code in a style that is consistent with the file being changed. Megatron-LM doesn't (yet) have a style guide or enforced formatting.
- Split your changes into separate, atomic commits i.e. A commit per feature or fix.
- Make sure your commits are rebased on the master branch.
- Write the commit message subject line in the imperative mood ("Change the default argument for X", not "Changed the default argument for X").
- Write your commit messages in proper English, with care and punctuation.
- Check the spelling of your code, comments and commit messages.

### Don't:

- Submit code that's incompatible with the project licence.
- Touch anything outside the stated scope of the PR. This includes formatting changes to code not relevant to the PR.
- Iterate excessively on your design across multiple commits.
- Include commented-out code.
- Attempt large architectural changes without first opening an issue to discuss.

## Issue and Pull Request Q&A (Updated Jul 2023)

### I've submitted an issue and PR. When can I expect to get some feedback?

Megatron-LM is developed and maintained by a small team of researchers. We will endeavour to read and acknowledge all new issues and PRs within a week. A few rules of thumb:
- Reproducible bugs/regressions and bug/regression fixes are likely to get the attention of maintainers the quickest.
- Issues requesting an enhancement may only recieve acknowlegement that they've been read and may be closed with a "wontfix" label if they're not inline with the project direction. If they are acknowledged and remain open you can assume the maintainers agree they're a desirable feature.
- Support requests, i.e. requests for help running the code, have the lowest priority and will be responded to as maintainer time permits.

### If my issue or PR isn't getting attention, how long should I wait before pinging one of the project maintainers?

One week if there is no acknowledgement of the intial request.

### Who are the project maintainers I should ping?

The corresponding maintainers at this time are @jaredcasper and @jon-barker.

### Is there a policy for issues and PRs that haven't been touched in X days? Should they be closed?

Yes, starting in July 2023 we have a bot that will mark untouched PRs as "stale" after 60 days.

We have a long backlog of issues and PRs dating back 3.5 years. We are trying to triage these now by working backwards. Older issues we believe may still be relevant may recieve a request to re-test them with the latest code. If there's no response they may be closed. Again, if you they should be re-opened then just respond with a comment to that effect.

Thank-you!