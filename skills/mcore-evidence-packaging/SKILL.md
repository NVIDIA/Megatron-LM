---
name: mcore-evidence-packaging
description: Package findings with cited artifacts, loaded skills, validation, and residual risk.
---

# Skill: MCore Evidence Packaging

## Purpose
Package Megatron training conclusions into reviewable reports with evidence, loaded skills, assumptions, and validation steps.

## Trigger Conditions
- A training review is ready for final delivery.
- Specialist notes must be merged into `jobs/current/outputs/training-review.md`.
- QA needs to verify that conclusions are supported.

## Method
1. List artifacts reviewed with paths.
2. List root skills loaded with paths.
3. List experts consulted.
4. Order findings by severity.
5. For each finding, include evidence, reasoning, recommendation, validation, and residual risk.
6. Separate facts from assumptions.
7. Include exact commands when they are known from repo context.
8. Put final output in `jobs/current/outputs/training-review.md`.

## Output
A concise final report suitable for a maintainer or training operator.

## Edge Cases
- If a finding lacks evidence, downgrade it to an assumption or remove it.
- If a report omits loaded skills, QA must block it.
