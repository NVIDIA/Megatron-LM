# What does this PR do ?
<!-- Add a one line overview of what this PR aims to accomplish. -->

:warning: For major changes (either in lines of code or in its impact), please make sure to first share a design doc with the team. If you're unsure what's the best way to do so, contact the @mcore-oncall.

## Contribution process

```mermaid
flowchart LR
    A[Pre-checks] --> B[PR Tests]
    subgraph Code Review/Approval
        C1[Expert Review] --> C2[Final Review]
    end
    B --> C1
    C2 --> D[Merge]
```

> **Note:** All PRs start as drafts. Click **"Ready for Review"** when your PR is ready to begin the review process.

### Pre-checks

- [ ] I want this PR in a versioned release and have added the appropriate Milestone (e.g., `Core 0.8`)
- [ ] I have added relevant unit tests
- [ ] I have added relevant functional tests
- [ ] I have added proper typing to my code [Typing guidelines](https://docs.python.org/3/library/typing.html)
- [ ] I have added relevant documentation
- [ ] I have run the [autoformatter.sh](https://github.com/NVIDIA/Megatron-LM/blob/main/tools/autoformat.sh) on my PR

### Code review

The following process is enforced via the CODEOWNERS file for changes into `megatron/core`. For changes outside of `megatron/core`, it is up to the PR author whether or not to tag the Final Reviewer team.

<details>
<summary>For MRs into `main` branch</summary>

Feel free to message or comment the @mcore-oncall to help accelerate your merge into main. The less complex your PR is, the faster it will be approved and merged!

#### (Step 1): Mark PR as "Ready for Review"

When your PR is ready, click **"Ready for Review"**. GitHub will auto-assign expert reviewers based on your changes. They will get notified and pick up your PR soon.

:warning: Only proceed to the next step once all reviewers have approved, merge conflicts are resolved, and CI is passing.
Final Review might get declined if these requirements are not fulfilled.

#### (Step 2): Final Review (automatic)

Once all expert reviewers approve, the `Final Review` label is added automatically. GitHub auto-assigns final reviewers based on your changes. They will get notified and pick up your PR soon.

#### (Optional Step 4): Cherry-pick into release branch

If this PR also needs to be merged into `core_r*` release branches, after this PR has been merged, select `Cherry-pick` to open a new PR into the release branch.

</details>

<details>
<summary>For MRs into `dev` branch</summary>
The proposed review process for `dev` branch is under active discussion.

MRs are mergable after one approval by either `eharper@nvidia.com` or `zijiey@nvidia.com`.
</details>

### Merging your PR

Any member of [core-adlr](https://github.com/orgs/teams/NVIDIA/core-adlr) and [`core-nemo`](https://github.com/orgs/teams/NVIDIA/core-nemo) will be able to merge your PR.
