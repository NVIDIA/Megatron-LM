---
orphan: true
---

# Determinism Developer Reference

> This reference is for Megatron developers and reviewers working on
> deterministic training. For setup instructions and supported configurations,
> use the [Deterministic Training user guide](../../user-guide/deterministic-training.md).
> This reference does not repeat it.

Bit-exact determinism means two runs with identical configuration, data, seeds,
software, and hardware produce identical results.

## Contents

This reference includes:

- [`status.md`](./status.md): deterministic-mode enforcement, validation,
  performance cost, and a pointer to the live roadmap
- [`op-catalog.md`](./op-catalog.md): operations with a deterministic code path,
  operations that deterministic mode does not support, and the goal to shrink
  the unsupported set while speeding up the supported set
- [`glossary.md`](./glossary.md): definitions and abbreviations

The roadmap is tracked dynamically in
[issue #5785](https://github.com/NVIDIA/Megatron-LM/issues/5785).
