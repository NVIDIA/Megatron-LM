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
model tensors in several Megatron `torch_dist` checkpoints, writing a new
`torch_dist` checkpoint. It is intended for checkpoint utilities such as
warmup-stable-merge (WSM) experiments, where several checkpoints of the **same
model at different iterations** are merged into a new checkpoint that can be
loaded for evaluation or resumed with `--no-load-optim --no-load-rng`.

The tool is **metadata-driven**: it derives the entire merge structure from each
source checkpoint's public PyTorch DCP metadata — tensor FQNs, global shapes,
dtypes, chunk/sharding layout, and byte/object `_extra_state` entries. It
**never builds a Megatron model or sharded-state template** and imports no model
code. Because it constructs no model, it never imports the CUDA-only kernels that
model builders need (for example `mamba_ssm`/`causal_conv1d` for Mamba/hybrid, or
Transformer-Engine layers), so it runs **CPU-only on a GPU-less CPU partition for
every model family** — GPT, Mamba, hybrid, and Transformer Engine alike. This is
the key advantage over a model-template approach, which needs CUDA merely to
construct Mamba/TE model templates.

## How It Runs

The merge is pure Python over the public DCP APIs. It performs **no Megatron model
bootstrap** — no model-parallel state setup, no model construction. It only:

- initializes a **gloo** process group (never NCCL) and runs with no GPU residency;
- reads each source checkpoint's public DCP metadata to derive the merge layout;
- loads each source's tensor chunks, accumulates floating model tensors in
  **fp32 on CPU**, and writes the merged output **directly through the public
  PyTorch DCP `SavePlanner`** (a tool-local planner plus the public
  `FileSystemWriter`), without staging a full merged output shard.

Parallelism is controlled entirely by the launched world size: more `torchrun`
ranks means more parallel chunk reads, saturating at storage bandwidth. The merge
is **I/O-bound** — end-to-end time is dominated by checkpoint load and storage
bandwidth, while fp32 accumulation is a small fraction of walltime. This is why
GPUs provide no speedup and the tool is CPU-only. A single process is the simplest
invocation; launching multiple ranks reshards the read/write work through DCP. The
merge arithmetic is independent of merge-time world size.

When no process group is initialized, or the world size is one, the output is
saved through public DCP with `no_dist=True`; with a world size greater than one
it uses public distributed DCP save. Rank 0 writes the Megatron sidecars
(`common.pt`, `metadata.json`) and the latest marker after the DCP save succeeds.

## Checkpoint Selection

### Manual `PATH:WEIGHT`

Manual mode takes explicit `PATH:WEIGHT` inputs:

```bash
python tools/checkpoint/weighted_merge.py \
  --merge-inputs \
    /checkpoints/run_a/iter_0001000:0.25 \
    /checkpoints/run_a/iter_0002000:0.75 \
  --merge-output /checkpoints/merged/manual \
  --output-iteration 2000
```

Use `--normalize` when the manual weights should be normalized before merging.
Without `--normalize`, the weights are used exactly as provided, including
unnormalized and **negative** weights (allowed for subtractive merges; the tool
warns that negative or non-unit-sum weights can produce outputs outside the input
range).

### Range and iteration-window selection

Range mode selects `iter_*` directories from a single checkpoint root between an
inclusive `--start-checkpoint` and `--end-checkpoint`, and computes weights from a
schedule:

```bash
python tools/checkpoint/weighted_merge.py \
  --merge-inputs /checkpoints/run_a \
  --start-checkpoint 1000 \
  --end-checkpoint 5000 \
  --merge-style minus-sqrt \
  --min-iteration-interval 1000 \
  --merge-output /checkpoints/merged/minus_sqrt
```

The target `--end-checkpoint` must exist. Minimum-interval filtering
(`--min-iteration-interval`) walks backward from the target checkpoint,
preserving the target iteration. Add `--min-checkpoints` to fail when filtering
selects too few checkpoints for a meaningful merge. In range mode
`--output-iteration` defaults to `--end-checkpoint`.

> **Token-window selection is not supported.** The `--merge-window-btoks` flag
> still exists but is **rejected** in the metadata-driven path: because no model
> is built, the tool cannot read `seq_length`/`global_batch_size` to derive a
> token window. Use explicit `PATH:WEIGHT` inputs or `--start-checkpoint` /
> `--end-checkpoint` selection instead.

### Coefficient schedules

Supported base schedules (`--merge-style`, default `linear`):

- `linear`: uniform average over the selected checkpoint set.
- `minus-sqrt`: discrete difference of `1 - sqrt(x)` over the selected
  checkpoint positions.

The supported deterministic modifiers are `__reverse` and `__scramble`, for
example `minus-sqrt__reverse`. Scramble uses `--coefficient-seed` and is
deterministic by default.

## Merge Semantics

Floating model tensors are accumulated in **fp32 on CPU**, applying each
checkpoint's weight. `--merge-save-dtype` controls the saved dtype:

- `same` (default): preserve the dtype observed in the source DCP metadata. All
  input dtypes for a tensor must match.
- `float32`, `float16`, `bfloat16`: cast averaged tensors to the requested dtype
  before save.

Transformer Engine `_extra_state` entries (including byte/object entries) are
**copied** from a single source checkpoint — selected by
`--extra-state-source-index` (default 0) — and are **not** averaged. Common
checkpoint metadata is copied from the first input checkpoint. When
`--output-iteration` is set, the output is written under
`--merge-output/iter_XXXXXXX`, the `iteration` metadata is updated, and
`latest_checkpointed_iteration.txt` is written in the output root.

Optimizer state and RNG state are not averaged. Load the merged checkpoint for
evaluation or resume with:

```bash
--no-load-optim --no-load-rng
```

## Fail-Closed Validation

The metadata-driven merge requires its inputs to share an **identical layout**.
This is automatically true for the WSM case — averaging checkpoints of the *same*
model saved at different iterations. The tool validates every source's metadata
against the first and fails early when:

- the model tensor key sets differ across inputs (missing or unexpected keys);
- the byte/object `_extra_state` key sets differ across inputs;
- a tensor's global shape or chunk/sharding layout differs across inputs;
- a tensor's dtype differs across inputs under `--merge-save-dtype=same`;
- a model tensor has a non-floating dtype (averaging is only defined for floating
  tensors);
- a byte/object DCP entry appears **outside the recognized model roots** (allowed
  prefixes are `model.`, `model0.`, `model1.`; allowed unprefixed roots are
  `decoder.`, `embedding.`, `output_layer.`, `mtp.`);
- a non-tensor model entry other than a byte `_extra_state` is present;
- the input checkpoint formats differ, or are not `torch_dist`.

Only `torch_dist` model checkpoints are supported; other formats (for example
`fsdp_dtensor`) are rejected with an explicit unsupported-format error.

`--strict` and `--merge-resource-log` are accepted only at their defaults; the
metadata-driven merge does not use these options and rejects non-default values.

## Output Publication

Output is published **atomically**: the tool writes to a hidden temporary
directory, then renames it into place. It is **fail-closed on an existing output
directory**; pass `--overwrite-merge-output` to allow replacement (atomic
overwrite of a non-empty checkpoint is itself refused, so remove the existing
checkpoint or target a new path/iteration). Publication orders the latest marker
**after** the metadata and uses best-effort fsync of checkpoint sidecars,
directories, and the latest-marker replacement. Atomic publication is always
required — `--no-atomic-merge-output` is rejected.

The script prints wall-clock timing for discovery, load, accumulation, and save,
along with input bytes read, output bytes written, effective read/write
bandwidth, and per-rank peak host memory. `--merge-byte-accounting` controls byte
accounting granularity (default `rank0`, which avoids multiplying filesystem
metadata traffic by rank).

## Limitations and Non-Claims

- **Same-layout inputs only.** The metadata-driven path requires every input to
  share an identical tensor/chunk layout — the WSM case. It cannot reshape or
  reconcile differently-sharded or differently-shaped checkpoints; mismatches are
  rejected (see Fail-Closed Validation).
- **No template verify-load step.** There is no model-template reload after save;
  verification of the merged checkpoint is out of scope and should be done
  separately.
- **Crash atomicity is partial.** Temporary-directory-plus-rename, fail-closed on
  existing output, and marker-after-metadata cover the common case. The tool does
  **not** claim atomicity against a kill during the final rename or against
  filesystem/power-loss faults.

## Examples

Manual two-checkpoint blend, written as a standalone output directory:

```bash
python tools/checkpoint/weighted_merge.py \
  --merge-inputs \
    /checkpoints/run_a/iter_0004000:0.5 \
    /checkpoints/run_a/iter_0005000:0.5 \
  --merge-output /checkpoints/merged/blend
```

WSM-style iteration-window merge with a `minus-sqrt` schedule, published as an
`iter_*` checkpoint with a latest marker:

```bash
python tools/checkpoint/weighted_merge.py \
  --merge-inputs /checkpoints/run_a \
  --start-checkpoint 1000 \
  --end-checkpoint 5000 \
  --merge-style minus-sqrt \
  --min-iteration-interval 1000 \
  --merge-output /checkpoints/merged/wsm
```

Multi-rank merge for faster reads on large checkpoints (parallelism comes from the
launched world size, not from GPUs):

```bash
torchrun --nproc_per_node 4 tools/checkpoint/weighted_merge.py \
  --merge-inputs /checkpoints/run_a \
  --start-checkpoint 1000 \
  --end-checkpoint 5000 \
  --merge-style linear \
  --merge-output /checkpoints/merged/linear
```
