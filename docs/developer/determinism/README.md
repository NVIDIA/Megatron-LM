---
orphan: true
---

# Determinism developer reference

> **Audience:** Megatron developers and reviewers working on deterministic
> training. For setup instructions and supported configurations, use the
> [Deterministic Training user guide](../../user-guide/deterministic-training.md) —
> this reference does not repeat it.

Bit-exact determinism means two runs with identical configuration, data, seeds,
software, and hardware produce identical results. The precise definition and
the terms used across these pages are in the [glossary](./glossary.md).

## Contents

1. **[`status.md`](./status.md)** — where the effort stands: what deterministic
   mode enforces today, how it is validated, performance cost, and a pointer to
   the live roadmap.
2. **[`op-catalog.md`](./op-catalog.md)** — the per-operation picture, in two
   buckets: operations with a deterministic code path, and operations
   deterministic mode does not support yet. The goal is to shrink the second
   bucket and make the first one fast.
3. **[`glossary.md`](./glossary.md)** — definitions and abbreviations.

The roadmap itself is tracked dynamically in
[issue #5785](https://github.com/NVIDIA/Megatron-LM/issues/5785) rather than in
these documents, so the docs describe what is merged and the issue describes
what is moving.
