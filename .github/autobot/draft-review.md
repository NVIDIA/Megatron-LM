<!-- megatron-lm-draft-review -->
👉 Please work with Autobot to get your draft PR approved before asking the engineering team for a review. Comment `/review-approve` for a findings-only review or `/review-approve reset=true` for a full review. If you’re outside `mcore-engineers`, ask a member to post the command for you.

Address any findings in draft. After you push fixes, ask Autobot to take another look. Once it approves your latest commit, click **Ready for review**. GitHub will request reviews from the teams that own the files you changed.

To go straight to engineering review, add `skip-bot-approval` first. If you mark the PR ready without approval or the label, automation will move it back to draft.

```mermaid
flowchart LR
    D[Work in draft] -->|Request a review| B[Autobot review]
    B -->|Address findings and push fixes| D
    B -->|Latest commit approved| R[Mark ready for review]
    D -->|Apply skip-bot-approval| R
    R --> G{Approved or label-exempt?}
    G -->|No · return to draft| D
    G -->|Yes| E[Engineering review by code owners]
```
