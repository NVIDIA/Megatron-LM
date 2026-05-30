---
name: mcore-training-task-triage
description: Classify Megatron training tasks and route to root skills plus specialist experts.
---

# Skill: MCore Training Task Triage

## Purpose
Classify a Megatron-LM training request into the smallest useful set of root skills and specialist experts while preserving the original repository harness as source of truth.

## Trigger Conditions
- A user asks for a training review, launch audit, test plan, loss diagnosis, or experiment design.
- The task includes logs, configs, recipes, patches, SLURM scripts, or benchmark output.
- Multiple expert domains may be involved.

## Method
1. Read the user request and every referenced artifact.
2. Read root `AGENTS.md` and `CLAUDE.md` when present.
3. Decide whether existing root skills cover the task before using specialist expertise.
4. Apply mandatory delegation:
   - SLURM/distributed launch: `skills/mcore-run-on-slurm/SKILL.md`.
   - Tests/golden values/recipes: `skills/mcore-testing/SKILL.md`.
   - Dependency/container issues: `skills/mcore-build-and-dependency/SKILL.md`.
   - CI failures: `skills/mcore-cicd/SKILL.md`.
   - Formatting/import edits: `skills/mcore-linting-and-formatting/SKILL.md`.
5. Select experts only after the relevant skill is loaded.
6. Write artifacts reviewed, skills loaded, selected experts, and assumptions.

## Output
`jobs/current/working/training-review-routing.md`.

## Edge Cases
- If an artifact is missing, name the missing path and continue with available evidence.
- If a root skill conflicts with specialist guidance, the root skill wins.
- If no concrete artifact exists, produce a bounded plan and ask for the missing evidence.
