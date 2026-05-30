# /project:training-review

Run a Megatron-LM training review with the original repo harness as source of truth and this specialist panel as the routing layer.

## Invocation

```text
/project:training-review <training task, diff, log, config, recipe, or question>
```

## Required Workflow

1. Read the user request and every concrete artifact named in it.
2. Scan `jobs/current/` for added logs, configs, recipes, patches, and notes.
3. Read root `AGENTS.md` and `CLAUDE.md` if present.
4. Load relevant root `skills/*/SKILL.md` before reasoning.
5. Route to selected experts through `training-orchestrator`.
6. Ask `qa-editor` to validate evidence, loaded skills, and output policy.
7. Write the final report to `jobs/current/outputs/training-review.md`.

## Mandatory Delegation Rules

- SLURM, node allocation, distributed launch, rank mapping, NCCL, checkpoint resume: load `skills/mcore-run-on-slurm/SKILL.md` and route to `distributed-systems-expert`.
- Tests, golden values, recipes, CI parity, or benchmark evidence: load `skills/mcore-testing/SKILL.md` and route to `evaluation-benchmark-expert`.
- Dependency, container, build, or import-environment issues: load `skills/mcore-build-and-dependency/SKILL.md` and route to `distributed-systems-expert` plus `qa-editor`.
- CI failures: load `skills/mcore-cicd/SKILL.md` and route to `evaluation-benchmark-expert` plus `qa-editor`.
- Formatting or import edits: load `skills/mcore-linting-and-formatting/SKILL.md` and route to `qa-editor`.
- Loss spikes, NaNs, divergence, precision, optimizer, or throughput regression: load `skills/mcore-training-signal-diagnosis/SKILL.md` and route to `optimization-stability-expert`.
- Model topology, MoE, attention, parallelism compatibility, memory, or checkpoint shape: route to `model-architecture-expert`; add `evaluation-benchmark-expert` when tests are affected.
- Dataset blend, tokenization, packing, curriculum, or validation leakage: route to `data-curriculum-expert`.

## Output Contract

Primary output: `jobs/current/outputs/training-review.md`.

The report must include task summary, artifacts reviewed, root skills loaded, experts consulted, severity-ordered findings, recommended changes or experiments, validation plan, and assumptions. If no artifact is available, write a bounded review plan and name the missing files.
