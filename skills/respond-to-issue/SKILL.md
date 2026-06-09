---
name: respond-to-issue
description: Maintainer triage and response for GitHub issues and PRs in NVIDIA/Megatron-LM. Use when asked to reply to, classify, route, assign, scan open or out-of-SLA issues, batch-draft responses, ask for reproducibility/scenario details, or identify owners from CODEOWNERS, changed files, history, and similar reports.
when_to_use: User shares a GitHub issue or PR URL/number, asks a maintainer or on-call to triage, route, assign, respond, draft a reply, scan issues needing response, find out-of-SLA community issues, ask for more information, or design automation for GitHub community intake.
user_invocable: true
argument: "<github-issue-or-pr-url-or-number>"
---

# GitHub Issue and PR Triage

Help maintainers and on-call responders reply quickly, route to the right owner, and ask for the minimum information needed to make progress on GitHub issues and PRs.

Default posture: helpful coordinator first, technical diagnoser second. The reply should either unblock the reporter, route the item to a likely owner, or ask for the exact missing data needed to reproduce and diagnose.

## Workflow

### 1. Read the artifact first

Do not infer or draft before reading the live issue/PR and comments.

For issues:

```bash
gh issue view <number> --repo NVIDIA/Megatron-LM --json title,body,comments,labels,state,author,createdAt,updatedAt,url
```

For PRs:

```bash
gh pr view <number> --repo NVIDIA/Megatron-LM --json title,body,comments,labels,state,isDraft,author,files,reviewRequests,reviews,statusCheckRollup,baseRefName,headRefName,headRefOid,url,mergeable
gh pr diff <number> --repo NVIDIA/Megatron-LM
```

Also search for prior art before routing:

```bash
gh issue list --repo NVIDIA/Megatron-LM --search "<keywords>" --limit 10
gh pr list --repo NVIDIA/Megatron-LM --search "<keywords>" --limit 10
```

### 2. Classify the intake

Choose one primary queue:

- **Bug**: something fails or produces wrong behavior.
- **Regression**: behavior, speed, memory, or accuracy got worse between two commits/releases.
- **Question**: usage, expected behavior, configuration, or design clarification.
- **Feature request**: new capability or changed behavior.
- **PR review/routing**: contributor opened a PR and needs owner review, scope guidance, CI help, or missing context.
- **CI/release/infrastructure**: GitHub Actions, GitLab, containers, wheels, release, docs publishing.
- **Security/private data**: do not request public repro details; route to the appropriate private security/reporting channel.

Apply or suggest labels if available: `bug`, `enhancement`, `community-request`, `needs-info`, `needs-repro`, `regression`, and area labels. If labels do not exist, mention the intended label rather than inventing it silently.

### 3. Check reproducibility and scenario completeness

For bugs and regressions, verify whether the report contains enough to reproduce. Ask only for missing items.

Minimum useful repro packet:

- Megatron-LM commit or release, and whether it reproduces on latest `main`.
- Exact command, script, config, or minimal code snippet.
- Expected behavior and actual behavior.
- Full stack trace or relevant log excerpt.
- Environment: container/OS, PyTorch, CUDA, NCCL, GPU type/count, node count.
- Scenario: model family, training vs inference, dataset/checkpoint assumptions, precision, and parallelism topology (TP/PP/DP/CP/EP/FSDP).
- For regressions: known-good commit, first-bad or current-bad commit, old/new metric values, run-to-run variance if known.

For questions, ask for the user scenario instead of a full bug repro: what they are trying to do, the config/command, and what behavior is unclear.

For PRs, check for:

- Linked issue or clear problem statement.
- Affected area and likely owner/team.
- Tests, docs, and validation commands.
- Backward compatibility risk: API/config defaults, checkpoint format, training numerics, performance, distributed behavior.
- CI status and whether failures are related to the PR.

### 4. Route to an owner

Use evidence in this order:

1. Referenced files, changed files, stack traces, commands, and labels.
2. `.github/CODEOWNERS` for path/team ownership.
3. Recent file history: `git log --oneline -20 -- <path>` and, when useful, `git log -S "<symbol>" -- <path>`.
4. Similar issues/PRs and maintainers who already handled the same area.
5. Current on-call or maintainer coordinator when the report lacks enough information to route.

Prefer routing to teams from CODEOWNERS when possible. Assign or request an individual only when they are clearly the code-change writer, active owner, or already engaged. If ownership is uncertain, keep a maintainer/on-call responder as coordinator and ask for the missing scenario/repro details.

Common owner evidence:

- `megatron/core/transformer/moe/` -> MoE teams from CODEOWNERS.
- `megatron/core/distributed/`, FSDP, optimizer, checkpointing, pipeline, datasets, tokenizers, inference, RL, CI, docs, and training paths -> use the matching CODEOWNERS entry.
- PRs touching multiple areas -> identify primary area plus secondary reviewers; ask author to split only if review ownership or blast radius is too broad.

### 5. Research before technical claims

If the response will make a technical claim, verify it against code or CI first.

- Read relevant source files and surrounding context.
- Re-read exact file/line references before citing them.
- Confirm commit hashes with `git show <hash> --stat`.
- If claiming code is unused or missing, use `rg` and, if needed, `git log -S`.
- If a fix already exists, link the issue/PR and state the relationship precisely.

### 6. Draft the maintainer reply

Every reply should contain:

1. Acknowledge the report or PR.
2. Summarize the maintainer understanding in one sentence.
3. State the route or next action.
4. Ask for exactly the missing information, or point to the likely owner/workaround/guide.

Keep the tone respectful and concise. Do not promise a fix, timeline, or owner action unless already confirmed.

Missing repro template:

```markdown
Thanks for the report. To route this to the right maintainer and reproduce it, could you add the missing details below?

- Megatron-LM commit or release, and whether this reproduces on latest `main`
- Exact command/config or a minimal script
- Expected behavior vs actual behavior
- Full stack trace or relevant logs
- Environment: container/OS, PyTorch, CUDA, NCCL, GPU type/count, node count
- Scenario: model family, training/inference path, precision, and parallelism setup (TP/PP/DP/CP/EP/FSDP)

Once we have that, a maintainer can route it to the right area owner.
```

Owner routing template:

```markdown
Thanks for the details. This looks related to `<path-or-feature>`, so routing to @NVIDIA/<team> for area-owner input. My understanding is: <one-sentence summary>. The current repro/validation signal is: <brief evidence>.

If you can also add <one missing item>, that will make it easier for the owner to reproduce quickly.
```

Question template:

```markdown
Thanks for the question. My understanding is that you are trying to <scenario>. The relevant path appears to be <feature/config/doc>. Could you share the command/config and the parallelism/model setup you are using? With that, a maintainer can either answer directly or route to the owner of that path.
```

PR routing template:

```markdown
Thanks for the PR. To accelerate review, could you link the issue or describe the user scenario this fixes, list the validation commands/results, and note the affected area/owners if known? Based on the changed files, this likely needs review from @NVIDIA/<team>. Maintainers will keep this routed once the PR is ready for review and CI is passing.
```

### 7. Automation guidance

An automated GitHub agent is useful, but keep it constrained:

- Start with triage/draft mode: classify, label, identify likely owners, and produce a proposed comment in a job summary or Slack/on-call notification.
- Allow auto-comments only for low-risk cases: missing repro/template fields, obvious PR checklist gaps, or deterministic CODEOWNERS routing.
- Require human approval for closing issues, declaring duplicates, making unverified technical diagnoses, assigning specific individuals, or promising fixes/timelines.
- Add an `agent-triaged` marker/label or hidden comment marker so the bot does not repeat itself on every edit.
- Skip automation if a maintainer has already replied after the latest reporter update.
- Use least-privilege permissions. For `pull_request_target`, do not checkout or execute untrusted PR code; inspect metadata/diff via GitHub APIs or checkout only the trusted base repo.
- Keep a dry-run/manual trigger such as `/oncall triage` for sensitive issues or early rollout.

Recommended event model:

- Issues: `opened`, `reopened`, `edited`, and `labeled` with `community-request`.
- PRs: `opened`, `ready_for_review`, `reopened`, and `synchronize` only for routing/checklist refreshes.
- Outputs: labels, team review requests, a concise public ask-for-info comment, and a maintainer-facing summary with owner evidence.

### 8. Batch SLA scan

When asked to scan issues needing maintainer response, produce drafts first; never post during the scan.

If the user gives an SLA, use it. If not, use age buckets instead of claiming an issue is out-of-SLA: `new` (<1 day), `watch` (1-2 days), `at-risk` (2-5 days), `stale` (>5 days), based on the latest reporter/community update that still needs maintainer action.

Candidate search:

```bash
gh issue list --repo NVIDIA/Megatron-LM --state open --limit 100 \
  --json number,title,labels,assignees,author,createdAt,updatedAt,url
gh issue list --repo NVIDIA/Megatron-LM --state open --search "label:community-request" --limit 100 \
  --json number,title,labels,assignees,author,createdAt,updatedAt,url
```

For each candidate, fetch full context:

```bash
gh issue view <number> --repo NVIDIA/Megatron-LM --json title,body,comments,labels,state,author,createdAt,updatedAt,url
```

Classify as **needs reply** when:

- no maintainer has replied yet;
- the latest substantive comment is from the reporter/community and asks a new question or adds requested data;
- the issue has `community-request`, `needs-info`, `needs-repro`, or no clear owner/label and is aging;
- the issue was routed but no owner has engaged after the requested SLA;
- a prior maintainer/on-call reply asked for information and the reporter has now provided it.

Do **not** draft a new reply when:

- a maintainer already replied after the latest reporter update;
- the issue is clearly waiting on reporter info and the reporter has not responded;
- the latest activity is bot-only, label-only, or project-board churn;
- the issue is security/private-data sensitive.

For each issue that needs response, apply this skill's single-issue workflow and output a review packet:

```markdown
## #<issue-number> <title>

- URL:
- SLA state:
- Intake type:
- Latest action needed:
- Likely owner/team:
- Owner evidence:
- Suggested labels:
- Suggested assignee:
- Risk/uncertainty:

Draft reply:
> <editable public reply>

Approval:
- [ ] Post reply
- [ ] Apply labels
- [ ] Assign owner
```

After presenting the batch, wait for explicit approval. Accept maintainer edits in chat or in a checked/generated draft file. Post only approved items, and post exactly the edited text. Use `gh issue comment <number> --repo NVIDIA/Megatron-LM --body-file <file>` for comments and `gh issue edit <number> --repo NVIDIA/Megatron-LM --add-label <label>` for labels.

### 9. Present to the maintainer

Unless the user explicitly asks you to post or the repository automation is intentionally running in auto-comment mode, show the draft response and recommended labels/owners to the maintainer first. Format the draft as a quoted Markdown block so it is easy to review.

## Important guidelines

- Never post comments to GitHub without explicit approval from the user, except inside a deliberately configured automation path with clear guardrails.
- Ask for the smallest useful missing information, not a generic wall of questions.
- Do not route publicly to a person/team based only on a guess; state the evidence.
- If the issue is outside what can be determined from code, say what remains unclear.
- If the issue identifies a cleanly actionable bug or missing feature, tell the maintainer and offer to create a branch/PR after triage.
