---
orphan: true
---

# Determinism Developer Reference

> This reference is for Megatron developers and reviewers working on
> deterministic training. For setup instructions and supported configurations,
> use the [Deterministic Training user guide](../../user-guide/deterministic-training.md).
> This reference does not repeat it.

Bit-exact determinism means two runs with identical configuration, data, seeds,
software, and hardware produce identical results. The precise definition and
the terms used across these pages are in the [glossary](./glossary.md).

## Contents

- [`status.md`](./status.md): what deterministic mode enforces, how it is
  validated, performance cost, and a pointer to the live roadmap.
- [`op-catalog.md`](./op-catalog.md): discusses operations with a deterministic
  code path and operations that deterministic mode does not support. The goal is
  to shrink the second bucket and make the first one fast.
- [`glossary.md`](./glossary.md): definitions and abbreviations.

The roadmap is tracked dynamically in
[issue #5785](https://github.com/NVIDIA/Megatron-LM/issues/5785) rather than in
these documents, so the docs describe what is merged and the issue tracks what
is in progress.
