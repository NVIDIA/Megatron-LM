---
name: pr-review
description: Prompt asset for the Claude Code Review GitHub Action. It is read as a file by .github/workflows/claude_review.yml and is not an interactive skill — do not load it to answer questions or to review code outside that workflow.
license: Apache-2.0
disable-model-invocation: true
user_invocable: false
---

# Claude PR Review

This is the review prompt behind `.github/workflows/claude_review.yml`. The
workflow supplies `REPO`, `PR NUMBER`, `REVIEW DEPTH` and — for strict reviews —
`BASE REF`, then tells the reviewer to read this file.

It lives in `skills/` so the rubric can be diffed, reviewed and evolved like
code instead of being buried in YAML, but it is deliberately inert: the
frontmatter carries `disable-model-invocation: true`, so Claude Code drops it
from the advertised skill list and refuses to auto-invoke it. Reading it by
path, which is exactly what the workflow does, still works. Do not add a
`when_to_use:` field — that is the trigger text that would make it activate on
its own.

## Pick the depth

Read only the reference for the depth the caller passed. Each one is a complete
rubric, so loading the other adds nothing but noise:

| `REVIEW DEPTH` | Comment trigger | Read |
| -------------- | --------------- | ---- |
| `light` | `/claude review` | `skills/pr-review/references/light.md` |
| `strict` | `/claude strict-review` | `skills/pr-review/references/strict.md` |

## Mandatory workflow — never skip or reorder

1. Read the PR diff first: `gh pr diff $PR_NUMBER --repo $REPO`.
2. From the changed files and areas, identify the relevant domain skills. Use
   Glob on `skills/*/SKILL.md` to see what exists rather than assuming names —
   the set changes over time, and Megatron-LM domain guides carry an `mcore-`
   prefix (`mcore-testing`, `mcore-cicd`, `mcore-build-and-dependency`,
   `mcore-linting-and-formatting`, `mcore-run-on-slurm`, `mcore-split-pr`,
   `mcore-onboard-gb200-1node-tests`, …).
3. Read those `SKILL.md` files with the Read tool.
4. Read the depth reference from the table above.
5. Only then review.

The order is what makes the review worth reading. A reviewer who forms an
opinion before loading `mcore-testing` will invent a test convention that this
repo does not use, and a confidently wrong review comment costs the author more
time than no review at all.

## Posting findings

Use inline ` ```suggestion ` blocks only for simple, self-contained line
replacements — typos, renames, single-line fixes. For structural changes that
add, remove or reorganize blocks of code (a new function, an inserted YAML
step, reordered logic), post a top-level PR comment with a fenced code block
showing the proposed change instead. GitHub's suggestion blocks can only
replace the exact lines they are anchored to, so an insertion or a multi-block
restructuring applied via `suggestion` silently corrupts the author's file.

Findings that deeper analysis invalidates should be dropped entirely rather
than hedged. A hedged comment transfers the work of disproving it to the author.

Completion — what to post at the end, and when to approve — is depth-specific
and covered in the reference file.
