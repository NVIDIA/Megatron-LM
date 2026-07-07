---
name: mcore-migrate-gpt-to-hybrid
description: Migrate Megatron Core GPTModel checkpoints, model providers, training commands, and layer mappings to HybridModel. Use for GPTModel-to-HybridModel migration, gpt_hybrid_conversion.py, hybrid layer patterns, converted-checkpoint training, or migration validation. Reads the canonical migration document instead of duplicating it.
---

# Migrate GPTModel to HybridModel

## Canonical source

Read [`docs/user-guide/hybrid-model-migration.md`](../../docs/user-guide/hybrid-model-migration.md)
completely before forming a migration answer or plan, changing code, converting
a checkpoint, or launching training.

Treat that document as the sole source of truth for migration behavior,
commands, mappings, prerequisites, limitations, and validation. Do not copy
those details into this skill or add a separate reference file.

If the document and implementation disagree, report the discrepancy before
continuing. When the task authorizes a correction, update the canonical document
first so future human and skill-driven workflows remain aligned.

## Workflow

1. Pull the task artifact first, such as the checkpoint metadata, model
   provider, training command, conversion log, or failure output.
2. Read the canonical migration document completely.
3. Follow the relevant document section without inventing unsupported migration
   paths or silently changing architecture.
4. Validate the result in proportion to the change, using the repository's
   build and testing skills when applicable.
5. Report the outcome and link the canonical document for human readers.
