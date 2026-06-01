<!---
   Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# Weighted Checkpoint Merge

`tools/checkpoint/weighted_merge.py` merges Megatron distributed checkpoints by
weighted averaging of model tensors. It is intended for checkpoint utilities
such as warmup-stable-merge (WSM) experiments, where several compatible model
checkpoints are merged into a new checkpoint that can be loaded for evaluation
or resumed with `--no-load-optim --no-load-rng`.

The utility uses Megatron's distributed-checkpointing API with the current
model's sharded state dict. It does not gather a full production model on rank 0.
Existing full-tensor checkpoint helpers under `tools/checkpoint` are useful for
small conversion tests and debugging, but should not be used as the primary path
for large model merging.

The tool runs Megatron initialization to build the sharded state-dict template,
so launch it in a normal Megatron runtime with the required distributed and CUDA
dependencies. The execution mode controls *where merged output is staged and how
it is written*; it is not a CPU-only execution guarantee.

## Execution Modes

The tool supports exactly two execution modes, selected with
`--merge-execution-mode`:

- **`cpu-resident`** (default): loads each input shard, accumulates in fp32 on
  CPU, and saves the merged checkpoint through the public DCP save path. This
  keeps GPU residency low (observed GPU peak ~691 MiB on a real Super-scale
  count=2 run); host memory holds the rank-local input shards plus the fp32
  accumulators.
- **`direct-dcp-streaming`**: writes merged output chunks directly through a
  tool-local public PyTorch DCP `SavePlanner` and public `FileSystemWriter`,
  avoiding a full-output-shard staging tensor. This bounds *output* residency.
  When no process group is initialized, or the initialized world size is one,
  it calls public DCP save with `no_dist=True`; when a process group with world
  size greater than one is initialized, it uses public distributed DCP save.
  Rank 0 writes the Megatron sidecars and latest marker through the normal
  publication path after public DCP save succeeds. This mode passed a real
  Super-scale TP=8/PP=1/count=2 run with `--verify-load`, landing near the
  I/O-volume lower bound (1.24-1.31x of a serial-I/O proxy).
  `--merge-streaming-chunk-bytes` controls the approximate fp32 accumulator
  budget per output tensor chunk.

Both modes produce identical merge arithmetic and are independent of merge-time
world size. The `direct-dcp-streaming` mode requires atomic publication and
rejects `--no-atomic-merge-output`.

## Manual Weighted Merge

Manual mode takes explicit `PATH:WEIGHT` inputs:

```bash
python tools/checkpoint/weighted_merge.py \
  --merge-inputs \
    /checkpoints/run_a/iter_0001000:0.25 \
    /checkpoints/run_a/iter_0002000:0.75 \
  --merge-output /checkpoints/merged/manual \
  --output-iteration 2000 \
  --ckpt-format torch_dist \
  --model-builder gpt
```

Use `--normalize` when the manual weights should be normalized before merging.
Without `--normalize`, the weights are used exactly as provided (including
unnormalized and negative weights).

## Range And Window Merge

Range mode selects `iter_*` directories from a checkpoint root and computes
weights from a schedule:

```bash
python tools/checkpoint/weighted_merge.py \
  --merge-inputs /checkpoints/run_a \
  --start-checkpoint 1000 \
  --end-checkpoint 5000 \
  --merge-style minus-sqrt \
  --min-iteration-interval 1000 \
  --merge-output /checkpoints/merged/minus_sqrt \
  --ckpt-format torch_dist \
  --model-builder gpt
```

Token-window mode derives the start iteration from checkpoint args:

```bash
python tools/checkpoint/weighted_merge.py \
  --merge-inputs /checkpoints/run_a \
  --end-checkpoint 5000 \
  --merge-window-btoks 125 \
  --merge-style linear \
  --merge-output /checkpoints/merged/linear_125btok \
  --ckpt-format torch_dist \
  --model-builder gpt
```

`--merge-window-btoks` uses `ceil(window_tokens / (seq_length *
global_batch_size))` and always requires the target `--end-checkpoint` to exist.
Minimum-interval filtering walks backward from the target checkpoint, preserving
the target iteration. Add `--min-checkpoints` to fail when filtering selects too
few checkpoints for a meaningful merge.

## Coefficients

Supported base schedules are:

- `linear`: uniform average over the selected checkpoint set.
- `minus-sqrt`: discrete difference of `1 - sqrt(x)` over the selected
  checkpoint positions.

The supported deterministic modifiers are `__reverse` and `__scramble`, for
example `minus-sqrt__reverse`. Scramble uses `--coefficient-seed` and is
deterministic by default.

## Dtype And Metadata Rules

Floating model tensors are accumulated in fp32 on CPU. `--merge-save-dtype`
controls the saved dtype:

- `same`: preserve the dtype observed through the requested model state-dict
  template. In normal `--use-checkpoint-args` usage this should match the source
  checkpoint dtype. All input dtypes for a tensor must match.
- `float32`, `float16`, `bfloat16`: cast averaged tensors to the requested dtype
  before save.

Transformer Engine `_extra_state` entries are copied from a single input
checkpoint (selected by `--extra-state-source-index`, default 0) and are not
averaged. Common checkpoint metadata is copied from the first input checkpoint.
When `--output-iteration` is set, the output checkpoint is written under
`--merge-output/iter_XXXXXXX`, `iteration` metadata is updated, and
`latest_checkpointed_iteration.txt` is written in the output root.

Optimizer state and RNG state are not averaged. Load the merged checkpoint for
evaluation or resume with:

```bash
--no-load-optim --no-load-rng
```

## Compatibility And Fail-Closed Validation

All input checkpoints must use the same distributed checkpoint format and must
be compatible with the model built by `--model-builder` (`gpt`, `hybrid`, or
`mamba`). The merge fails early when requested model keys are absent, shapes
differ, checkpoint formats differ, input dtypes for a tensor disagree under
`--merge-save-dtype=same`, or non-floating model tensors cannot be merged.

By default `--strict` is `raise_unexpected`, which tolerates extra sharded
checkpoint entries such as optimizer state while still requiring every requested
model tensor to exist. Use `--strict raise_all` when debugging an exact
model-only checkpoint.

Support is for `torch_dist` model checkpoints; `fsdp_dtensor` is rejected with
an explicit unsupported-format error.

Data-parallel merge-time world sizes above one are blocked by default and
require `--allow-data-parallel-merge` after validating the run shape.

`--merge-preflight-only` builds the merge template, prints memory estimates,
enforces preflight guards, and exits without loading, saving, verifying, or
writing output. `--merge-max-projected-cpu-bytes` optionally fails before load
when the projected per-rank CPU peak exceeds a byte limit.

## Output Publication And Verification

By default, output is published atomically: the tool writes to a hidden
temporary directory, then renames it into place. It is fail-closed on an
existing output directory; pass `--overwrite-merge-output` to allow replacement.
Publication orders the latest marker after metadata and uses best-effort fsync
of checkpoint sidecars, directories, and the latest-marker replacement.

`--no-atomic-merge-output` writes directly to `--merge-output` (not supported
with `direct-dcp-streaming`).

Add `--verify-load` to reload the merged checkpoint with the same sharded-state
template after saving. For `direct-dcp-streaming`, verification loads the hidden
temporary checkpoint *before* final publication, so a verification failure
leaves the final `iter_*` directory and latest marker unpublished.

The script prints wall-clock timing for checkpoint discovery, model
initialization, checkpoint load, accumulation, output save, and optional
post-save verification, along with input bytes read, output bytes written,
effective read/write bandwidth, and per-rank peak host/GPU memory. Use
`--merge-byte-accounting` and `--merge-resource-log` to control byte accounting
and resource-checkpoint logging granularity.

## Performance And Limitations

- **Load-dominated walltime.** End-to-end time is dominated by checkpoint load
  and storage bandwidth; fp32 accumulation is only a few percent of walltime.
  The tool approaches but does not reach the physical I/O lower bound. The
  residual gap is dominated by storage bandwidth and DCP planning overhead, not
  the merge arithmetic.
- **Source reads are not bounded below the stored-record size.** PyTorch DCP's
  reader deserializes the full stored tensor payload before narrowing, so the
  per-source read volume follows the source checkpoint's storage-record size,
  not any requested chunk size. This is a known limitation that applies to both
  modes; `--merge-streaming-chunk-bytes` bounds output accumulator chunking, not
  source read volume.
- **Crash atomicity is partial.** The tool uses a temporary directory plus
  atomic rename, is fail-closed on an existing output, runs a hidden-temp
  verify-load gate before publication, orders the latest marker after metadata,
  and performs best-effort fsync. It does **not** claim crash atomicity across a
  kill during the final rename, or against filesystem/power-loss faults.
- **Large-scale scaling is not separately proven.** Real evidence covers a
  Super-scale count=2 merge at 8 GPUs (TP=8/PP=1) for both modes. 1T+ model
  sizes and 32-64-GPU scaling are not separately validated; treat them as open
  caveats rather than supported claims.
- **Format.** Only `torch_dist` model checkpoints are supported.
