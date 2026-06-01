<!---
   Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# Weighted Checkpoint Merge

`tools/checkpoint/weighted_merge.py` produces an exact weighted average of the
model tensors in several compatible Megatron `torch_dist` checkpoints, writing a
new `torch_dist` checkpoint. It is intended for checkpoint utilities such as
warmup-stable-merge (WSM) experiments, where several compatible model
checkpoints are merged into a new checkpoint that can be loaded for evaluation
or resumed with `--no-load-optim --no-load-rng`.

The utility uses Megatron's distributed-checkpointing API with the current
model's sharded state dict. It builds a model only to obtain the sharded-state
template; it does not gather a full production model on rank 0. Existing
full-tensor checkpoint helpers under `tools/checkpoint` are useful for small
conversion tests and debugging, but should not be used as the primary path for
large model merging.

## CPU-Only Execution

The tool has a single, CPU-only execution path. It always:

- initializes a gloo process group (never NCCL) and runs with no GPU residency;
- loads each input shard, accumulates floating model tensors in **fp32 on CPU**,
  and writes the merged output **directly through the public PyTorch DCP save
  path** (a tool-local `SavePlanner` plus public `FileSystemWriter`), without
  staging a full merged output shard;
- uses **no GPU memory** for the merge — host memory holds the rank-local input
  shards plus the fp32 accumulators.

When no process group is initialized, or the initialized world size is one, the
output is saved through public DCP with `no_dist=True`; with a world size
greater than one it uses public distributed DCP save. Rank 0 writes the Megatron
sidecars and latest marker through the normal publication path after the DCP
save succeeds. The merge arithmetic is independent of merge-time world size.

`--merge-streaming-chunk-bytes` controls the approximate fp32 accumulator budget
per output tensor chunk during the save.

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
  --use-checkpoint-args \
  --model-builder gpt \
  --tensor-model-parallel-size 8
```

Use `--normalize` when the manual weights should be normalized before merging.
Without `--normalize`, the weights are used exactly as provided, including
unnormalized and negative weights.

## Range And Window Merge

Range mode selects `iter_*` directories from a single checkpoint root and
computes weights from a schedule:

```bash
python tools/checkpoint/weighted_merge.py \
  --merge-inputs /checkpoints/run_a \
  --start-checkpoint 1000 \
  --end-checkpoint 5000 \
  --merge-style minus-sqrt \
  --min-iteration-interval 1000 \
  --merge-output /checkpoints/merged/minus_sqrt \
  --ckpt-format torch_dist \
  --use-checkpoint-args \
  --model-builder gpt \
  --tensor-model-parallel-size 8
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
  --use-checkpoint-args \
  --model-builder gpt \
  --tensor-model-parallel-size 8
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

- `same` (default): preserve the dtype observed through the requested model
  state-dict template. In normal `--use-checkpoint-args` usage this should match
  the source checkpoint dtype. All input dtypes for a tensor must match.
- `float32`, `float16`, `bfloat16`: cast averaged tensors to the requested dtype
  before save.

Transformer Engine `_extra_state` entries are **copied** from a single input
checkpoint (selected by `--extra-state-source-index`, default 0) and are not
averaged. Common checkpoint metadata is copied from the first input checkpoint.
When `--output-iteration` is set, the output checkpoint is written under
`--merge-output/iter_XXXXXXX`, `iteration` metadata is updated, and
`latest_checkpointed_iteration.txt` is written in the output root. In
range/window mode `--output-iteration` defaults to `--end-checkpoint`.

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
`--merge-save-dtype=same`, required checkpoint metadata is missing, or
non-floating model tensors cannot be merged.

By default `--strict` is `raise_unexpected`, which tolerates extra sharded
checkpoint entries such as optimizer state while still requiring every requested
model tensor to exist. Use `--strict raise_all` when debugging an exact
model-only checkpoint.

Support is for `torch_dist` model checkpoints; other formats such as
`fsdp_dtensor` are rejected with an explicit unsupported-format error.

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

Add `--verify-load` to reload the merged checkpoint with the same sharded-state
template after saving. Verification loads the hidden temporary checkpoint
*before* final publication, so a verification failure leaves the final `iter_*`
directory and latest marker unpublished.

Atomic publication is always required: `--no-atomic-merge-output` is rejected.

The script prints wall-clock timing for checkpoint discovery, model
initialization, checkpoint load, accumulation, output save, and optional
post-save verification, along with input bytes read, output bytes written,
effective read/write bandwidth, and per-rank peak host memory. Use
`--merge-byte-accounting` and `--merge-resource-log` to control byte accounting
and resource-checkpoint logging granularity.

## Execution Environment And Resources

The merge is I/O-bound: end-to-end time is dominated by checkpoint load and
storage bandwidth, while fp32 accumulation is only a few percent of walltime.
Because of this, GPUs provide no speedup, which is why the tool is CPU-only.

Evidence for the I/O-bound, CPU-only design:

- A 10-checkpoint merge scaled with rank parallelism, with aggregate read
  bandwidth rising from roughly 1 GiB/s at 1 rank to roughly 3.8 GiB/s at 4
  ranks.
- GPU versus CPU runs at equal world size were within measurement noise.

Parallelism is therefore controlled by the merge-time world size: more ranks
means more parallel reads, saturating at storage bandwidth. Set parallelism by
launching the tool with more ranks (e.g. via `torchrun`), not by adding GPUs.

### Where it can run

"CPU-only" means CPU compute and CPU memory everywhere — there is never any GPU
compute or GPU residency during the merge. Whether you also need a GPU *node* is
determined solely by the model-template import:

- **Pure-PyTorch model builders** (for example GPT with
  `--transformer-impl local`) can run end-to-end on a GPU-less CPU partition.
- **Model builders that import CUDA-only kernels** — Mamba/hybrid builders that
  pull in `mamba_ssm`/`causal_conv1d`, or any builder using Transformer-Engine
  layers — require a GPU *node* so the model template can be imported. The merge
  itself still runs entirely CPU-side on that node (no GPU compute, no GPU
  residency).

## Limitations

- **Crash atomicity is partial.** The tool uses a temporary directory plus
  atomic rename, is fail-closed on an existing output, runs a hidden-temp
  verify-load gate before publication, orders the latest marker after metadata,
  and performs best-effort fsync. It does **not** claim crash atomicity across a
  kill during the final rename, or against filesystem/power-loss faults.
- **Large-scale scaling is not separately proven.** 1T+ model sizes and large
  multi-node scaling are not separately validated; treat them as open caveats
  rather than supported claims.
- **Format.** Only `torch_dist` model checkpoints are supported.
