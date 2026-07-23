# Agent Compose Skills

This directory defines agent-agnostic operational contracts for upstreaming
Megatron Lite through Agent Compose. Skills describe how to make and validate a
change; they do not replace executable tests.

## Format

Each skill is one Markdown file with three parts:

1. A short human-facing title.
2. A schema between `MLITE_SKILL_SCHEMA_BEGIN` and
   `MLITE_SKILL_SCHEMA_END`.
3. A finite Python-like pseudocode body with a declared exit.

Schema names map to paths by replacing underscores with hyphens and dots with
directories. For example, `runtime.validate` maps to
`runtime/validate.md`.

## Initial Registry

- `basic.constitution`: global design and validation constraints.
- `basic.lint_skill`: structural validation for skills.
- `primitive.contract`: required contract for a primitive.
- `model.compose`: compose a model only from contracted primitives.
- `runtime.validate`: validate the runtime lifecycle end to end.

Load this file, then exactly one leaf skill for the current work type and every
skill named by that leaf's `imports`.
