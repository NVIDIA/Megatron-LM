# Evaluation Report

Evaluation of the `mcore-testing` skill before publication to the NVCARPS ecosystem.

This benchmark summarizes NV-BASE validation and Tier 3 live agent evaluation results for the skill. The goal is to document whether the skill is safe, discoverable, effective, and useful for agents before it is published for broader workflow use.

## Evaluation Summary

- Skill: `mcore-testing`
- Evaluation date: 2026-05-29
- NV-BASE profile: `internal`
- Overall verdict: PASS
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

Tier 1 validation passed with observations. NV-BASE ran 9 checks and found 8 total findings.

Top findings:

- MEDIUM QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.tags' (`skills/mcore-testing/SKILL.md`)
- MEDIUM SCHEMA/body_recommended_section: Missing recommended section: '## Instructions' (`skills/mcore-testing/SKILL.md`)
- MEDIUM SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`skills/mcore-testing/SKILL.md`)
- LOW QUALITY/quality_discoverability: No '## Purpose' section (`skills/mcore-testing/SKILL.md`)
- LOW QUALITY/quality_reliability: No prerequisites/requirements documented (`skills/mcore-testing/SKILL.md`)

## Tier 2: Deduplication Summary

This tier was not run or did not produce findings in this report.

## Publication Recommendation

The skill is suitable to proceed toward NVCARPS publication based on this benchmark. Skill owners should keep this file with the skill and refresh it when the evaluation dataset, skill behavior, or target agents materially change.
