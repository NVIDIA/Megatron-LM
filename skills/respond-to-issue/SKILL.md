---
name: respond-to-issue
description: Research and draft a response to a GitHub issue or question from an external contributor.
license: Apache-2.0
when_to_use: User shares a GitHub issue URL or asks to respond to a community question; 'respond to this issue', 'draft a reply', 'answer this GitHub question'.
user_invocable: true
argument: "GitHub issue URL or number"
---

# Respond to GitHub Issue

Help a maintainer draft a high-quality response to a GitHub issue from an external contributor.

## Answer-First Pattern: Unused-Code Claims

When asked how to answer an issue claiming code is unused, give the maintainer a
checklist rather than guessing:

- Read the issue with `gh issue view <number> --repo NVIDIA/Megatron-LM --json title,body,comments,labels,state`.
- Classify the issue, then search/read the referenced files or symbols.
- Before saying code is unused or missing, run a thorough reference search,
  inspect recent file history with `git log --oneline -20 -- <relevant-files>`,
  and trace symbol history with `git log -S "<symbol>" --oneline`.
- Search related PRs and issues with `gh pr list --repo NVIDIA/Megatron-LM --search "<keywords>" --limit 5`
  and `gh issue list --repo NVIDIA/Megatron-LM --search "<keywords>" --limit 5`.
- Re-read cited lines and verify cited commits with `git show <hash> --stat`.
- Present the maintainer draft as a quoted markdown block; never post it without
  explicit approval.

## Workflow

### 1. Understand the issue

- Fetch the issue using `gh issue view <number> --repo NVIDIA/Megatron-LM --json title,body,comments,labels,state`.
- Read the title, body, and all existing comments to understand the full context.
- Identify the type: bug report, feature request, question, or discussion.

### 2. Research the codebase

- Based on what the issue is asking, search the Megatron-LM codebase for the relevant code.
- Read the relevant source files to understand the current behavior.
- If the issue references specific files or functions, read those directly.
- Check `git log --oneline -20 -- <relevant-files>` to see if there have been recent changes that address or relate to the issue.
- Use `git log -S "<symbol>" --oneline` to trace when code was added or removed — this is especially useful for questions about unused/deprecated code or missing features.
- If the issue is about a bug, try to confirm whether the reported behavior matches the code.
- Check whether an existing PR already addresses the issue: `gh pr list --repo NVIDIA/Megatron-LM --search "<keywords>" --limit 5`.

### 3. Verify before citing

Before including specific details in the response, verify them:
- If citing a commit hash, confirm the commit message and diff match what you're claiming (`git show <hash> --stat`).
- If citing a file path and line number, re-read the file to confirm the line content is correct.
- If claiming code is unused or missing, do a thorough grep to make sure you haven't missed a reference.

### 4. Draft the response

Write a response that:
- Is technically accurate and grounded in the actual code (cite file paths and line numbers where helpful).
- Is respectful and welcoming to external contributors.
- Directly addresses the question or concern raised.
- If the contributor identified a real gap or bug, acknowledge it clearly.
- If there's a workaround, mention it.
- If work is planned or a fix would be welcome, say so and suggest next steps (e.g., "a PR to address this would be welcome").
- Keeps the tone professional but friendly.
- Is concise -- contributors appreciate direct answers, not walls of text.

### 5. Suggest follow-up actions

If the issue identifies something cleanly actionable (dead code to remove, a small bug fix, a missing feature), tell the maintainer and offer to create a branch and PR to address it — don't just draft a comment.

### 6. Present to the maintainer

Show the drafted response to the user (the maintainer) for review. Do NOT post it to GitHub automatically. The maintainer will decide whether to post it, edit it, or ask for changes.

Format the draft as a quoted markdown block so it's easy to copy.

## Important guidelines

- Never post comments to GitHub without explicit approval from the user.
- If you're unsure about the answer, say so clearly in your draft and flag the uncertainty for the maintainer.
- If the issue is outside the scope of what you can determine from the code, tell the maintainer what you found and what remains unclear.
- Check whether similar issues exist that might be relevant: `gh issue list --repo NVIDIA/Megatron-LM --search "<keywords>" --limit 5`.
