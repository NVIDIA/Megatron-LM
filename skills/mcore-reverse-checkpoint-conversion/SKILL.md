---
name: mcore-reverse-checkpoint-conversion
description: Convert and validate Megatron-FSDP fsdp_dtensor checkpoints back to torch_dist so a classic N-D-parallel (TP/PP/EP) job can resume from them.
license: Apache-2.0
when_to_use: Converting a Megatron-FSDP fsdp_dtensor checkpoint to torch_dist; resuming a classic (TP/PP/EP) job from an FSDP-trained checkpoint; running, validating, or extending the reverse checkpoint converter or its validation harness; adding a model or transform to that harness; the convert-fsdp-dtensor-to-torch-dist CLI; the --dist-ckpt-optim-fully-reshardable resume flag; 'convert fsdp_dtensor to torch_dist', 'resume classic from a Megatron-FSDP checkpoint', 'reverse checkpoint converter'.
metadata:
  author: Ahmad Kiswani <akiswani@nvidia.com>
---

# Megatron-FSDP → torch_dist reverse checkpoint conversion

The offline reverse converter rewrites a Megatron-FSDP `fsdp_dtensor` checkpoint
as a native `torch_dist` checkpoint, so a classic (non-FSDP) N-D-parallel job can
resume from a model trained with Megatron-FSDP — weights **and** full
distributed-optimizer state, reshardable into any target TP/PP/EP/CP/VPP on load.

## Answer-First Guidance

- **The tool:** `tools/checkpoint/checkpoint_inspector.py
  convert-fsdp-dtensor-to-torch-dist <in>/iter_XXXXXXX <out>/iter_XXXXXXX`
  (CPU-only, model-free, parallelism-agnostic). Run
  `python tools/checkpoint/checkpoint_inspector.py convert-fsdp-dtensor-to-torch-dist --help`
  for the flags (`--swiglu-modules`, `--stack-layers/--non-homogeneous-layers`,
  `--no-optimizer`, …).
- **Resuming with optimizer state requires** on the classic job:
  `--dist-ckpt-optim-fully-reshardable --ckpt-format torch_dist
  --use-distributed-optimizer --dist-ckpt-strictness log_all --no-load-rng`.
- **Canonical docs** — read the relevant one completely before answering,
  planning, converting, or editing; do not duplicate their content here:
  - User guide:
    [`docs/user-guide/features/megatron_fsdp.md`](../../docs/user-guide/features/megatron_fsdp.md)
    → "Converting Megatron-FSDP (`fsdp_dtensor`) to N-D Parallel (`torch_dist`)"
    (resume flags, supported architectures, memory ceiling, multi-process convert).
  - End-to-end suite:
    [`tests/integration_tests/tools/checkpoint/fsdp_dtensor_to_torch_dist/README.md`](../../tests/integration_tests/tools/checkpoint/fsdp_dtensor_to_torch_dist/README.md)
    (the opt-in pytest suite — what is tested, how, and how to extend it).
  - Unit tests: `tests/unit_tests/tools/checkpoint/test_reverse_convert.py` and
    `test_reverse_convert_roundtrip.py`.

---

## Workflow

1. **Pull the task artifact first** — the checkpoint's saved `args`, the convert
   log's `[Convert] …` line, the classic-load log, a failing diff, or the model
   config. Do not reason about it yet.
2. **Set up the environment.** GPU work runs in the mcore dev container — see the
   [build & dependency skill](../mcore-build-and-dependency/SKILL.md). On older
   images, clear a stale `nvidia-resiliency-ext` (<0.6.0) that breaks
   `import megatron.core` before running anything.
3. **Convert / validate.** For a one-off conversion, use the CLI above. To prove a
   conversion is correct, run the opt-in end-to-end suite as **plain pytest** (it
   spawns its own torchrun — do not launch it under `torch.distributed.run`):
   `MCORE_CHECKPOINT_E2E=1 uv run pytest
   tests/integration_tests/tools/checkpoint/fsdp_dtensor_to_torch_dist --run-e2e`.
   Scope with `-k` by family and by check (`test_resume` / `test_bitexact` [1 GPU],
   `test_reshard` / `test_source_sharding` [≥2 GPUs]).
4. **Read the PASS criteria from the suite README**, not from intuition: loads at the
   right iter, first resumed `lm loss` ≈ FSDP loss within bf16 tolerance, LR exact,
   and (bit-exact) a clean per-tensor diff.
5. **Report** the `[Convert]` line, the load-at-iter confirmation, and the numeric
   verdict; link the canonical doc for human readers.

---

## Key facts & gotchas

- **Single source of truth for arch flags** is the `ModelFamily` registry in the
  e2e suite (`tests/integration_tests/tools/checkpoint/fsdp_dtensor_to_torch_dist/registry.py`);
  the same entries feed every check and the bit-exact worker. Add a family = one
  `ModelFamily(...)` entry.
- **Dropped by design:** RNG state, rerun-state, and all `_extra_state` (incl. FP8
  amax history). FP8 resume re-initializes amax, so it tracks ~1% looser than bf16
  tolerance — expected, not a bug.
- **`gdn_hybrid` needs `flash-linear-attention`** (the image's `fla` stub is
  insufficient); it is pinned in the `dev` extra, and the e2e suite skips the family
  if `fla` is not importable (it never pip-installs at test time).
- **Under GPU contention**, prefer the single-rank resume + bit-exact checks; the
  2-GPU reshard sweep can deadlock on a shared GPU.
- **Suite outputs** land in a tmp dir (or `RESULTS_DIR` if set) and are safe to
  delete between runs.
- **Scale is out of scope for the harness** (1–8 GPUs). The CPU-memory ceiling and
  the multi-process (`torchrun -N`) convert for large checkpoints are in the user
  guide.

---

## Adding tests

Pure-logic and round-trip coverage belong in
`tests/unit_tests/tools/checkpoint/test_reverse_convert*.py` — see the
[testing skill](../mcore-testing/SKILL.md). The GPU end-to-end coverage lives in the
opt-in pytest suite at
`tests/integration_tests/tools/checkpoint/fsdp_dtensor_to_torch_dist/` (add a family
= one `ModelFamily` entry in its `registry.py`); it is a real-checkpoint complement
to the unit tests, gated behind `--run-e2e` / `MCORE_CHECKPOINT_E2E=1`.

## Documentation drift

If the converter code and a canonical doc disagree, trust the code, fix the doc,
and note the correction — do not silently follow stale guidance.
