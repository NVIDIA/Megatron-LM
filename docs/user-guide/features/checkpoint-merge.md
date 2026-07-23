<!---
   Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# Weighted Checkpoint Merge

`tools/checkpoint/weighted_merge.py` applies either exact user-provided weights
or weights computed by range mode to the model tensors in several Megatron
`torch_dist` checkpoints, writing a new `torch_dist` checkpoint. Range mode
produces normalized coefficients; manual `PATH:WEIGHT` mode applies weights
exactly unless `--normalize` is passed. It is intended for checkpoint utilities
such as warmup-stable-merge (WSM) experiments, where several checkpoints of the
**same model at different iterations** are merged into a new checkpoint that can
be loaded for evaluation or resumed with `--no-load-optim --no-load-rng`.

The tool is **metadata-driven**: it derives the entire merge structure from each
source checkpoint's public PyTorch DCP metadata — tensor FQNs, global shapes,
dtypes, chunk/sharding layout, and byte/object `_extra_state` entries. It
**never builds a Megatron model or sharded-state template** and imports no model
code. Because it constructs no model, it never imports the CUDA-only kernels that
model builders need (for example `mamba_ssm`/`causal_conv1d` for Mamba/hybrid, or
Transformer-Engine layers), so supported same-layout `torch_dist` checkpoints run
**CPU-only on a GPU-less CPU partition** regardless of model family — GPT, Mamba,
hybrid, and Transformer Engine alike. This is the key advantage over a
model-template approach, which needs CUDA merely to construct Mamba/TE model
templates.

The metadata-driven design is a layout mechanism, not a security boundary. Treat
source checkpoints as trusted inputs: the merge loads common checkpoint state
from the first input and deserializes byte/object `_extra_state` entries from the
selected extra-state source when copying them into the output.

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

Large checkpoints can have source DCP chunk layouts that assign more work to one
rank than the others. Pass `--merge-balance-rank-work` to greedily bin-pack
source DCP chunks across ranks. This preserves the original DCP chunk boundaries
and source storage locality while avoiding rank-0 singleton-chunk skew. It does
not add a chunk-size tuning parameter; the merge still writes the same logical
checkpoint layout, only with a more even rank assignment during the save. Leave
it off for source layouts that are already evenly assigned across ranks: in that
case it cannot improve the planned byte balance and may be neutral or slower due
to changed read/write ordering.

When no process group is initialized, or the world size is one, the output is
saved through public DCP with `no_dist=True`; with a world size greater than one
it uses public distributed DCP save. Rank 0 writes the Megatron sidecars
(`common.pt`, `metadata.json`) and the latest marker after the DCP save succeeds.

## Recommended Workflow

Use the metadata-driven merge for checkpoints from the same run, model, and
parallelism layout:

1. Pick either explicit `PATH:WEIGHT` inputs or an iteration range under one
   checkpoint root.
2. Write to a new output path or a new `--output-iteration`.
3. Run the merge on CPU ranks; increase ranks to add parallel readers until the
   filesystem read bandwidth saturates.
4. Load the merged checkpoint separately for evaluation or resume with
   `--no-load-optim --no-load-rng`.

Do not use this tool to merge checkpoints from different architectures, tensor
layouts, or model-parallel sharding. Those cases are rejected instead of being
reshaped or reconciled.

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

Manual paths may point either at a concrete distributed checkpoint directory
containing `metadata.json`, or at a checkpoint root containing
`latest_checkpointed_iteration.txt`. A `latest_checkpointed_iteration.txt` value
of `release` resolves to the `release/` checkpoint; an integer value resolves to
the corresponding `iter_XXXXXXX/` directory.

### Range mode

Range mode requires both `--start-checkpoint` and `--end-checkpoint`. It selects
`iter_*` directories from a single checkpoint root between those inclusive
iterations, and computes weights from a schedule:

```bash
python tools/checkpoint/weighted_merge.py \
  --merge-inputs /checkpoints/run_a \
  --start-checkpoint 1000 \
  --end-checkpoint 5000 \
  --merge-style minus-sqrt \
  --min-iteration-interval 1000 \
  --merge-output /checkpoints/merged/minus_sqrt
```

The target `--end-checkpoint` must exist. Passing `--end-checkpoint` without
`--start-checkpoint` does not enter range mode; the CLI will treat
`--merge-inputs` as manual `PATH:WEIGHT` entries. Minimum-interval filtering
(`--min-iteration-interval`) walks backward from the target checkpoint,
preserving the target iteration. Add `--min-checkpoints` to fail when filtering
selects too few checkpoints for a meaningful merge. In range mode
`--output-iteration` defaults to `--end-checkpoint`.

### Coefficient schedules

Supported base schedules (`--merge-style`, default `linear`):

- `linear`: uniform average over the selected checkpoint set.
- `minus-sqrt`: discrete difference of `1 - sqrt(x)` over the selected
  checkpoint positions.

The full valid style set is `linear`, `minus-sqrt`, `linear__reverse`,
`linear__scramble`, `minus-sqrt__reverse`, and `minus-sqrt__scramble`.
Modifiers reorder the generated coefficient list before assigning coefficients
to the selected checkpoints. Scramble uses `--coefficient-seed` and is
deterministic by default.

## Merge Semantics

Floating model tensors are accumulated in **fp32 on CPU**, applying each
checkpoint's weight. `--merge-save-dtype` controls the saved dtype:

- `same` (default): preserve the dtype observed in the source DCP metadata. All
  input dtypes for a tensor must match.
- `float32`, `float16`, `bfloat16`: cast averaged tensors to the requested dtype
  before save.

Transformer Engine `_extra_state` entries (including byte/object entries) are
**copied** from a single source checkpoint — selected by the zero-based
`--extra-state-source-index` over the resolved input list (default 0) — and are
**not** averaged. In range mode, the resolved input list is sorted by iteration,
so the default extra-state source is the earliest selected checkpoint. When
`--output-iteration` is set, the output is written under
`--merge-output/iter_XXXXXXX`, the `iteration` metadata is updated, and
`latest_checkpointed_iteration.txt` is written in the parent of the concrete
`iter_*` output directory.

If `--merge-output` already names the matching `iter_XXXXXXX` directory, the tool
writes that directory directly. If `--merge-output` names a different `iter_*`
directory than `--output-iteration`, the merge is rejected to avoid publishing an
iteration under the wrong path.

Common checkpoint state is copied from the first input checkpoint, with the
output iteration updated when requested. Both common state loading and byte/object
`_extra_state` copying use checkpoint deserialization, so the input checkpoints
must be trusted.

The output `common.pt` includes `weighted_merge_provenance`: input paths, source
iterations when they can be inferred from `iter_*` directory names, weights,
normalization policy, merge style, output dtype, extra-state source, implementation
mode (`dcp-metadata-same-layout`), rank-work balancing policy, and the git
revision when available.

Only recognized model-root tensors and `_extra_state` entries are supported.
Optimizer/RNG sharded DCP tensor entries outside the recognized model roots must
not be present; the tool rejects them rather than ignoring or copying them. Load
the merged checkpoint for evaluation or resume with:

```bash
--no-load-optim --no-load-rng
```

## Fail-Closed Validation

The metadata-driven merge requires its inputs to share an **identical layout**.
This is normally true for WSM checkpoints from the same run with unchanged
checkpointing, parallelism, dtype, and extra-state layout. The tool still
validates every source's metadata against the first and fails early when:

- the model tensor key sets differ across inputs (missing or unexpected keys);
- the byte/object `_extra_state` key sets differ across inputs;
- a tensor's global shape or chunk/sharding layout differs across inputs;
- a tensor's dtype differs across inputs;
- a model tensor has a non-floating dtype (averaging is only defined for floating
  tensors);
- a byte/object DCP entry appears **outside the recognized model roots** (allowed
  prefixes are `model.` plus all numbered roots matching `model\d+\.`; allowed
  unprefixed roots are `decoder.`, `embedding.`, `output_layer.`, `mtp.`);
- a non-tensor model entry other than a byte `_extra_state` is present;
- the input checkpoint formats differ, or are not `torch_dist`.

Only `torch_dist` model checkpoints are supported; other formats (for example
`fsdp_dtensor`) are rejected with an explicit unsupported-format error.

## Output Publication

Output publication is **fail-closed** and **atomic**: the tool writes to a hidden
temporary directory, validates the temporary checkpoint metadata, then renames it
into place. Existing output directories are rejected before output writing starts
and again at publish time. The tool never replaces an existing output; remove the
existing checkpoint explicitly or target a new path/iteration.

Publication orders the latest marker **after** checkpoint metadata is published.
The latest marker is written through a unique temporary file and replacement, and
the tool uses best-effort fsync of checkpoint sidecars, directories, and the
latest-marker replacement.

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
- **Trusted inputs required.** The merge derives tensor layout from public DCP
  metadata, but common checkpoint state and byte/object `_extra_state` entries
  are loaded through checkpoint deserialization.

## Examples

Manual two-checkpoint blend, written as a standalone output directory:

```bash
python tools/checkpoint/weighted_merge.py \
  --merge-inputs \
    /checkpoints/run_a/iter_0004000:0.5 \
    /checkpoints/run_a/iter_0005000:0.5 \
  --merge-output /checkpoints/merged/blend
```

Range mode merge with a `minus-sqrt` schedule, published as an
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

Source-chunk-balanced multi-rank manual merge for large same-layout checkpoints
with skewed source chunk assignment:

```bash
torchrun --nproc_per_node 8 tools/checkpoint/weighted_merge.py \
  --merge-inputs \
    /checkpoints/run_a/iter_0004000:0.5 \
    /checkpoints/run_a/iter_0005000:0.5 \
  --merge-output /checkpoints/merged/balanced \
  --merge-balance-rank-work
```

Multi-node CPU merge with `torchrun`. Each process is one gloo rank/reader; no
GPU allocation or NCCL setup is required:

```bash
torchrun \
  --nnodes 2 \
  --nproc_per_node 8 \
  --rdzv_backend c10d \
  --rdzv_endpoint host0.example.com:29500 \
  tools/checkpoint/weighted_merge.py \
    --merge-inputs /checkpoints/run_a \
    --start-checkpoint 1000 \
    --end-checkpoint 5000 \
    --merge-style minus-sqrt \
    --min-iteration-interval 1000 \
    --merge-output /checkpoints/merged/wsm \
    --output-iteration 5000
```

Slurm `sbatch` construction for a multi-node CPU merge. Each Slurm task is one
gloo rank/reader; adjust the account, partition, node count, and task count for
the target cluster:

```bash
#!/bin/bash
#SBATCH --job-name=wsm-merge
#SBATCH --partition=cpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --time=01:00:00
#SBATCH --output=wsm-merge-%j.out

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=${MASTER_PORT:-29500}

srun bash -lc '
  export RANK=$SLURM_PROCID
  export WORLD_SIZE=$SLURM_NTASKS
  export LOCAL_RANK=$SLURM_LOCALID
  python tools/checkpoint/weighted_merge.py \
    --merge-inputs /checkpoints/run_a \
    --start-checkpoint 1000 \
    --end-checkpoint 5000 \
    --merge-style minus-sqrt \
    --min-iteration-interval 1000 \
    --merge-output /checkpoints/merged/wsm \
    --output-iteration 5000'
```

## Library API

The same metadata-driven path is callable from Python. It initializes a gloo
process group from `RANK`, `WORLD_SIZE`, `MASTER_ADDR`, and `MASTER_PORT` if one
is not already active; in a plain single-process script it runs as rank 0 with
world size 1.
Pass `balance_rank_work=True` for the same source-chunk rank balancing exposed by
the CLI's `--merge-balance-rank-work` flag.

```python
from tools.checkpoint.weighted_merge import (
    checkpoint_coefficients,
    merge_same_layout_dcp_metadata_checkpoints,
)

iters = [1000, 2000, 3000, 4000, 5000]
weights_by_iter = checkpoint_coefficients(iters, "minus-sqrt")
input_dirs = [f"/checkpoints/run_a/iter_{iteration:07d}" for iteration in iters]

result = merge_same_layout_dcp_metadata_checkpoints(
    input_dirs,
    [weights_by_iter[iteration] for iteration in iters],
    "/checkpoints/merged/wsm",
    output_iteration=5000,
    save_dtype="same",
    merge_style="minus-sqrt",
)

print(result.output_dir)
print(result.averaged_tensors, result.copied_extra_states)
print(result.timings.total, result.bytes_read, result.bytes_written)
```

## Validation

The implementation is covered by CPU unit tests for coefficient schedules and
modifiers, manual weight warnings, range selection, fp32 accumulation, dtype
policy, tensor and byte/object `_extra_state` copy, metadata fail-closed cases,
atomic-publication rejection, provenance, CLI parsing, and generated
`torch_dist` round trips.

The merge does not perform a model-template verify-load after save. Validate a
merged checkpoint by loading or evaluating it in the target workflow.
