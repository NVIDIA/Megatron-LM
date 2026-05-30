# Codex Command: training-review

Use this file as the Codex-side entrypoint for the hybrid Megatron training review harness. It intentionally mirrors `.claude/commands/training-review.md` without editing root `AGENTS.md`.

## Invocation

When the user writes either of these forms, run the workflow below:

```text
training review: <training task, diff, log, config, recipe, or question>
codex training-review: <training task, diff, log, config, recipe, or question>
```

## Workflow

1. Read the user request and every concrete artifact named in it.
2. Scan `jobs/current/` for logs, configs, recipes, patches, and notes.
3. Read root `AGENTS.md` and `CLAUDE.md` when present.
4. Load `skills/mcore-training-task-triage/SKILL.md`.
5. Load original Megatron root skills required by the task:
   - SLURM/distributed launch: `skills/mcore-run-on-slurm/SKILL.md`
   - tests/golden values/recipes: `skills/mcore-testing/SKILL.md`
   - dependencies/container/build: `skills/mcore-build-and-dependency/SKILL.md`
   - CI failures: `skills/mcore-cicd/SKILL.md`
   - formatting/imports: `skills/mcore-linting-and-formatting/SKILL.md`
   - loss spikes/NaNs/training signals: `skills/mcore-training-signal-diagnosis/SKILL.md`
6. Route through `.codex/agents/training-orchestrator.toml` and the relevant specialist mirrors.
7. Use `.codex/agents/qa-editor.toml` to block reports that do not cite artifacts or loaded skills.
8. Write the final report to `jobs/current/outputs/training-review.md`.

## Output Contract

The final report must include artifacts reviewed, skills loaded, experts consulted, severity-ordered findings, recommendations, validation plan, and assumptions.
