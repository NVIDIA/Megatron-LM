---
name: mcore-migrate-gpt-to-hybrid
description: Migration guide for moving Megatron Core GPTModel checkpoints, model providers, training commands, and layer mappings to HybridModel.
license: Apache-2.0
when_to_use: Migrating or reviewing a GPTModel checkpoint or training workflow for HybridModel; choosing or reviewing a hybrid layer pattern; running gpt_hybrid_conversion.py; loading a converted checkpoint; diagnosing GPT-to-Hybrid migration issues; 'migrate GPTModel to HybridModel', 'convert GPT checkpoint to HybridModel', 'hybrid layer pattern'.
metadata:
  author: Philip Petrakian <ppetrakian@nvidia.com>
---

# GPTModel to HybridModel Migration

## Answer-First Migration Guidance

- The canonical source is
  [`docs/user-guide/hybrid-model-migration.md`](../../docs/user-guide/hybrid-model-migration.md).
- Read the canonical document completely before answering, planning, reviewing,
  editing, converting, or training.
- Keep migration behavior, commands, mappings, prerequisites, limitations, and
  validation in the canonical document only. Do not duplicate them in this
  skill.

---

## Workflow

1. Pull the task artifact first: checkpoint metadata, model provider or config,
   training command, conversion log, diff, or failure output.
2. Read the canonical migration document completely.
3. Follow only the relevant document sections. Do not invent an unsupported
   migration path or silently change the target architecture.
4. Validate the result proportionately, invoking the relevant repository build
   and testing skills when applicable.
5. Report the outcome and link the canonical document for human readers.

---

## Documentation Drift

If the implementation and migration guide disagree:

1. Report the discrepancy before continuing.
2. If the task authorizes a correction, update the canonical document first.
3. Do not add a competing migration rule to this skill.
