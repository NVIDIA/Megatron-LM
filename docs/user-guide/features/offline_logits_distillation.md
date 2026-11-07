<!---
   Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# Offline Logits Knowledge Distillation

Offline logits knowledge distillation (KD) trains a student model against a teacher's
cached top-K token distribution instead of running the teacher live. A separate
teacher run saves reference logits and a tail log-sum-exp once; later student runs stream
them back and add a forward-KL loss term, without ever loading teacher weights. This
decouples teacher and student compute (different job, different cluster, different
time) and removes the teacher's memory/compute footprint from the student job entirely.

This is a different mechanism from [ModelOpt's knowledge distillation](https://github.com/NVIDIA/Megatron-LM/blob/dev/examples/post_training/modelopt/distillation.md),
which runs teacher and student together in the same process via module-swap. Use offline
logits KD when the teacher is too large/expensive to co-locate with the student, or when
you want to distill from the same cached teacher run into several independent student
configurations.

Implemented in [`megatron/training/distillation/`](https://github.com/NVIDIA/Megatron-LM/tree/dev/megatron/training/distillation):
- `logits_saver.py` — `LogitsSaverHooks`, attached to the teacher's output layer.
- `cached_logits_loss.py` — `LossFuncCallable` (student-side loss), `CachedLogitsKDLoss`,
  `StudentLogitsCapture`.
- `utils.py` — shared tar/storage format, index bit-packing, and the DP/MBS/GBS
  resharding plan.

## How It Works

**Saving (teacher run):** pass `--logits-save-dir`. `LogitsSaverHooks` attaches a
forward hook to the model's output layer on the last pipeline stage
(`megatron/training/training.py`, gated on `args.logits_save_dir is not None`). Each
microbatch's logits are converted to reference logits (TP-aware: local top-K is computed
per rank first, then gathered to TP rank 0 for the global top-K, avoiding a
full-vocab-sized tensor), optionally narrowed further with top-P nucleus masking, and
buffered until the whole iteration is collected. Buffered iterations are flushed
asynchronously as `.tar` shards (zstd-compressed, one shard per DP rank per flush)
through Megatron's async checkpoint queue: the flush is enqueued from
`save_checkpoint()` (`megatron/training/checkpointing.py`), so **buffered logits are
written to disk at the same cadence as regular checkpoints, i.e. every
`--save-interval` iterations** — not every iteration. This requires `--async-save`
and `--use-persistent-ckpt-worker` to be set, since the write itself piggybacks on
the async checkpoint worker rather than running inline. A dedicated teacher-dump run
should also pass `--freeze-all-layers`; without it, the model keeps training (LM loss
computed, gradients applied) while logits are saved on the side.

**Loading (student run):** pass `--logits-load-dir`. `StudentLogitsCapture` attaches a
capture hook to the student's output layer; `LossFuncCallable` (constructed once per
process, wired into `loss_func` in `pretrain_gpt.py` / `pretrain_hybrid.py`) streams the
matching tar shard(s) for the current DP rank and training iteration via a
`torch.utils.data.DataLoader` (`pin_memory=True`, decode overlapped with GPU compute),
and computes the sparse forward KL divergence `KL(teacher ‖ student)` between the
teacher's top-K positions plus one category for all omitted vocabulary entries. The
total loss is `alpha * kd_loss + (1 - alpha) * lm_loss`.

Both hooks are attached automatically based on the CLI flags below — no application
code changes are needed beyond invoking `pretrain_gpt.py` / `pretrain_hybrid.py` with
the relevant `--logits-save-*` or `--logits-load-*` flags.

## On-Disk Format

Shards are named `dp{D}__{start}-{end}.tar`, keyed by the **global sample index range**
`[start, end)` they cover (not by iteration number), so shard identity is independent of
global batch size. Each tar's first member is a JSON metadata blob (`_meta.json`,
including a dataset-identity hash used to catch teacher/student data mismatches) followed
by one compressed payload per saved iteration. Vocab indices are stored as 16 low bits
(`uint16`) plus a separately bit-packed 17th bit (1 bit/element via
`numpy.packbits`/`unpackbits`, not `torch.bool`'s 1 byte/element), reconstructed as
`(bit_17 << 16) | low_16_bits`.

New dumps use **format version 4**, selecting an encoding separately for every token
when the teacher distribution is available. The saver rounds both reference logits
and normalized top-K log-probabilities to FP16, decodes them using training's FP32
arithmetic, and chooses the lower `KL(teacher || decoded_cache)` over the K selected
tokens plus the aggregate tail. FP64 is used only to score these K+1 categories;
the full vocabulary and persistent caches are never expanded to FP64. Candidates
whose rounded top-K probability mass exceeds one are rejected. Reference logits win
ties. This minimizes the measured coarse KL, which can differ from minimizing
tail-only absolute or relative error.

To select the codec by tail reconstruction error instead, pass
`--logits-save-encoding-metric tail`. This minimizes absolute tail error per token
(equivalently relative tail error when the true tail is positive), and can increase
the full coarse KL. The default `kl` is intended for distillation; `tail` matches
tail-reconstruction-only numerical comparisons.

For K >= 2, the second value's sign records the choice: positive (including `+0`
and `+inf`) means reference logits, negative means normalized log-probabilities.
Both representations' second untagged value is nonpositive, so its sign carries no
numerical information. The decoder restores it before use. No additional mode
array, stored value, index, or per-token byte is needed. K=1 keeps reference logits.

Use the format-aware decoder on version-4 caches:

```python
from megatron.training.distillation.reference_logits import adaptive_logprobs

tail_probability = adaptive_logprobs(values)[..., -1].exp()
```

The adaptive selector changes cache creation only; the student always uses the
same cancellation-resistant coarse KL and gradient. The teacher remains independent
of the live student. Version-3 caches continue to use their fixed reference encoding.

For **format version 3**, K selected indices are represented by
exactly K FP16 numbers per token by default:

- `indices[..., 0]` is the highest teacher logit, used as the reference (implicit zero).
- `values[..., 1:]` stores `z_i - z_reference` for the other selected indices.
- `values[..., 0]` stores `logsumexp(z_j - z_reference)` over **all omitted** indices,
  including entries excluded by top-P. This tail is computed directly, never as
  `1 - sum(top-K probabilities)`.

Masked selected slots use `-inf`; an empty tail is also represented by `-inf` in slot 0.
The number of indices, floating values, and uncompressed tensor bytes is unchanged.
The compressed size can differ. FP32 vocabulary workspaces are processed in bounded
chunks; accumulated cache tensors stay in the selected on-disk dtype.

To reconstruct the tail probability from version-3 `values`:

```python
import torch
import torch.nn.functional as F

v = values.float()  # [..., K]; slot 0 is the tail's relative log-sum-exp
other_lse = torch.logsumexp(v[..., 1:], dim=-1)
log_Z = F.softplus(torch.logaddexp(other_lse, v[..., 0]))  # implicit anchor exp(0)
tail_probability = torch.exp(v[..., 0] - log_Z)
```

`reference_logits.reference_logprobs(values)` returns all K+1 normalized log
probabilities in the order anchor, other selected tokens, tail. The version-3 loss
uses `expm1`/`log1p` near matching distributions and reconstructs the anchor gradient
from the other vocabulary gradients. Teacher FP16 quantization still changes the
target distribution: zero gradient is guaranteed at the decoded target, rather than
at every unquantized teacher value that rounds to it.

Versions 1 and 2 remain readable with their original loss semantics. Versions 3 and 4
require the corresponding updated reader and loss. Use a fresh cache directory when regenerating
an older dump; mixed payload formats are rejected. An old cache cannot recover tail
information already lost during normalization or FP16 rounding.

Superseded shards from a crash-and-resume are quarantined (renamed with a `.stale`
suffix, not deleted) rather than overwritten in place, so a crash mid-write never leaves
two overlapping live shards for the same range.

## Resharding

The student run's parallelism does not need to match the teacher run's:

- **Data parallelism** may differ freely between save and load, including non-power-of-2
  ratios (e.g. 6 saved DP ranks → 4 loaded DP ranks).
- **Micro-batch size** and **global batch size** may each differ from the teacher run,
  provided the new value is an integer multiple (in either direction) of the saved
  value.
- **Context parallelism** is CP-agnostic: shards store the full sequence, and each rank
  slices out its own CP zigzag chunk locally, so CP size can change freely as long as
  sequence length stays divisible by `2 * context_parallel_size`.

A `LogprobsReshardPlan` (`megatron/training/distillation/utils.py`) computes, for each
load-side microbatch, which saved DP-rank shard(s) and row ranges to read.

## Related Arguments

**Teacher saving:**

| Argument | Description |
| --- | --- |
| `--logits-save-dir` | Directory (local path or `msc://` object-store URL) to save top-K log-probs to. Requires `--async-save` and `--use-persistent-ckpt-worker`; buffered logits are flushed as an async request on the checkpoint queue, so they hit disk on the same `--save-interval` cadence as regular checkpoints. Incompatible with `--rampup-batch-size` (the sample↔iteration mapping assumes a fixed global batch size). |
| `--logits-save-top-k` | Number of top log-probs to save per token. Default: `128`. |
| `--logits-save-top-p` | Optional top-P (nucleus) threshold applied after top-K selection — keeps only the smallest prefix of entries whose cumulative probability mass reaches this value. Must be in `(0, 1]`. Default: `None` (disabled). |
| `--logits-save-top-p-min-k` | Minimum entries kept per token when top-P masking is active, regardless of cumulative mass. Default: `1`. |
| `--logits-save-dtype` | On-disk dtype for saved log-probabilities: `fp16`, `bf16`, or `fp32`. Default: `fp16`. |
| `--logits-save-encoding-metric` | Per-token codec selection by round-trip `kl` (default) or `tail` error. The choice uses only the teacher distribution at save time and does not change cache size. |
| `--freeze-all-layers` | Freezes all model parameters. Not required by `--logits-save-dir`, but recommended for a dedicated teacher-dump run: without it, the LM loss is still computed and gradients still update the model while logits are being saved. |

**Student loading:**

| Argument | Description |
| --- | --- |
| `--logits-load-dir` | Directory to load cached teacher top-K log-probs from. |
| `--logits-load-decode-threads` | Number of threads used to parallelize zstd decompression and `torch.load` inside the DataLoader. Default: `4`. |
| `--logits-load-msc-prefetch-depth` | For remote (MSC/object-store) shards, how many whole tar shards to prefetch ahead of sequential consumption. Default: `2`. |
| `--logits-load-kd-loss-alpha` | Weight `alpha` in `alpha * kd_loss + (1 - alpha) * lm_loss`. Default: `1.0`. |
| `--logits-load-ignore-errors` | If set, KD loss errors are logged as warnings and training falls back to LM-only loss instead of crashing. |
| `--logits-load-ignore-hash` | If set, skips the dataset-identity hash check on loaded shards. Use only when intentionally loading logits saved under a different (but known-compatible) dataset configuration. |

## Assumptions and Restrictions

- The student run must use the **same random seed and data pipeline** as the teacher run
  that produced the cached log-probs, so the global sample ordering matches.
- Teacher data must be saved with payload `format_version >= 2` for full DP/MBS/GBS
  resharding support. Older `format_version == 1` shards are still readable but are
  restricted to DP-only resharding, via a legacy reader slated for removal once existing
  caches are regenerated.
- The loader enforces sequential, non-overlapping shard consumption per saved DP rank,
  and checks every iteration that the cached teacher data starts at the same global
  sample as the student's batch (`consumed_train_samples`). Teacher data that is behind
  (duplicate/overlapping shards) is skipped with a warning; teacher data that is ahead
  (missing shards, or a global batch size that changes mid-run such as
  `--rampup-batch-size`) raises a clear error instead of training on misaligned targets.
  These checks verify sample positions, not token contents.
- If the teacher's saved sequence length is longer than the student's, the loader trims
  the teacher tensors to match; the reverse (teacher shorter than student) is not
  currently handled.
- PP > 1 support relies on a per-pipeline-stage collective for remote shard discovery
  (all last-PP-stage ranks participate) rather than a world-group broadcast, to avoid
  deadlocking with the loader's restriction to the last pipeline stage.
