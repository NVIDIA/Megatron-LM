# Evaluation Report

Evaluation of the `mcore-bump-base-image` skill before publication to the NVCARPS ecosystem.

This benchmark summarizes NV-BASE validation and Tier 3 live agent evaluation results for the skill. The goal is to document whether the skill is safe, discoverable, effective, and useful for agents before it is published for broader workflow use.

## Evaluation Summary

- Skill: `mcore-bump-base-image`
- Evaluation date: 2026-05-29
- NV-BASE profile: `internal`
- Overall verdict: FAIL
- Tier 3 live agent evaluation: not available in this report

## Agents Used

- Tier 3 agent details were not available in this report.

## Metrics Used

Reported benchmark dimensions:

- Security: checks whether skill-assisted execution avoids unsafe behavior such as secret leakage, destructive commands, or unauthorized access.
- Correctness: checks whether the agent follows the expected workflow and produces the correct final output.
- Discoverability: checks whether the agent loads the skill when relevant and avoids using it when irrelevant.
- Effectiveness: checks whether the agent performs measurably better with the skill than without it.
- Efficiency: checks whether the agent uses fewer tokens and avoids redundant work.

Underlying evaluation signals used in this run:

- No Tier 3 evaluation signal details were available in this report.

## Test Tasks

Tier 3 evaluation task details were not available in this report.

## Results

Tier 3 dimension rollup was not available in this report.

## Tier 1: Static Validation Summary

Tier 1 validation reported findings. NV-BASE ran 9 checks and found 9 total findings.

Top findings:

- MEDIUM QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.tags' (`skills/mcore-bump-base-image/SKILL.md`)
- MEDIUM SCHEMA/body_recommended_section: Missing recommended section: '## Instructions' (`skills/mcore-bump-base-image/SKILL.md`)
- MEDIUM SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`skills/mcore-bump-base-image/SKILL.md`)
- LOW QUALITY/quality_discoverability: Description very long (329 chars, recommend 50-150) (`skills/mcore-bump-base-image/SKILL.md`)
- LOW QUALITY/quality_discoverability: No '## Purpose' section (`skills/mcore-bump-base-image/SKILL.md`)

## Tier 2: Deduplication Summary

This tier was not run or did not produce findings in this report.

## Publication Recommendation

The skill should be reviewed before NVCARPS publication. Skill owners should address the findings above and rerun NV-BASE to refresh this benchmark.
