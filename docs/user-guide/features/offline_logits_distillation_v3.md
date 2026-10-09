# Offline KD V3: Paired Inputs and Targets

V3 stores the teacher's processed training inputs beside its sparse log probabilities.
The student replaces its training dataloader with cache replay. It needs the tokenizer
and model configuration, but does not need the original training dataset or a matching
dataset seed. Existing V1/V2 dumps and training paths retain their behavior when the
new input flags are absent.

## Dumping

Add these flags to a normal `pretrain_gpt.py` or `pretrain_hybrid.py` teacher launch:

```sh
--logits-save-dir /data/teacher-v3 \
--logits-save-inputs \
--logits-save-top-k 128 \
--logits-save-dtype fp16 \
--freeze-all-layers \
--async-save --use-persistent-ckpt-worker \
--save /data/teacher-progress --save-interval 100
```

The teacher still uses its original dataset flags. `--freeze-all-layers` is optional;
omit it only when intentionally collecting targets while training the teacher.
`--save` and `--save-interval` are required: the save interval is the flush interval,
and each flush writes one inputs tar and one targets tar per teacher DP rank.
Teacher batch-size ramp-up, `--iterations-to-skip` (including iterations added from
`--result-rejected-tracker-filename`), and `--allow-ambiguous-pad-tokens` are rejected.

A frozen teacher resumes from what the cache has actually published, not from its
progress tracker, so a failed or interrupted flush never leaves a resubmitted job past
its data. Partial flushes beyond that point are deleted on resume.

Each raw CPU microbatch is captured before TP field filtering, packed flattening, or
CP partitioning. An explicit token mapping gathers CP outputs back into sample order.
The saver checks that every unmasked token has a target. Rerun attempts remain
provisional until the training loop accepts an attempt.

### Parallel dump jobs

Several frozen teacher jobs can dump disjoint iteration ranges of one cache at the same
time. Give every job the same `--train-iters` (so the dataset order is identical) and the
same `--exit-interval E`, and start job `k` with `--override-ckpt-iteration k*E`:

```sh
# job 0: iterations [0, 10000)
--train-iters 50000 --exit-interval 10000 --override-ckpt-iteration 0
# job 1: iterations [10000, 20000)
--train-iters 50000 --exit-interval 10000 --override-ckpt-iteration 10000
```

Megatron exits at the next multiple of `E`, so each job owns
`[override, next multiple of E)`. `--exit-interval` and `--override-ckpt-iteration` must
be multiples of `--save-interval`, so every job's flushes line up with a single
sequential dump. Give each job its own `--save` directory. Resubmitting a job with an
unchanged script resumes inside its own range; a job whose range is already complete
exits before running a step. Without `--override-ckpt-iteration`, a job is the cache's
single sequential writer and `--exit-interval` keeps its usual periodic meaning (for
example, to requeue every N iterations).

Every tar records the cache settings. A teacher checks them against an existing tar at
startup, and students check every tar before using its data, so a job launched with
different settings is detected rather than silently mixed in.

## Loading

Add these flags to a student launch, with ordinary student model/checkpoint arguments:

```sh
--logits-load-dir /data/teacher-v3 \
--logits-load-inputs \
--logits-load-kd-loss-alpha 1.0 \
--dataloader-type single \
--eval-iters 0
```

Do not provide a training data source just to align the targets. Set `--train-samples`
or `--train-iters` to the desired student budget. Replay consumes each selected saved
sample once; it does not repeat the cache automatically. The available cache must cover
the complete final student global batch. If validation is enabled, provide an explicit
ordinary validation dataset (or mock data). Validation uses LM loss without cached KD
targets, and does not build the original training split.

Use the same tokenizer vocabulary and packing/attention isolation settings as the
teacher. The tokenizer fingerprint includes vocabulary IDs and relevant special tokens.
Position IDs, labels, and loss weights are replayed exactly as saved. Dataset identity
is provenance rather than a student alignment constraint; `--logits-load-ignore-hash`
is unnecessary for V3. Member checksums and pairing identities are always checked;
`--logits-load-ignore-errors` is rejected.

### Reordering, subsets, and growing dumps

Enable deterministic shard-group reordering with:

```sh
--logits-load-shuffle-shards 123
```

Omitting this option keeps sequential loading; `--logits-load-shuffle-shards 0` enables
shuffling with seed zero. The shuffle seed is independent of the teacher dataset seed.
One reorder unit is one flush range (all teacher DP tars for it). Samples inside a group
retain their global saved order. A smaller student budget consumes a prefix of the
selected order, and may stop within a group at a student iteration boundary. It does not
require the teacher's global batch size.

Shuffling uses the groups published when the student first starts; that extent is saved
with the student checkpoint and reused on resume, even if the teacher keeps publishing.
Whether a dump is finished before shuffling it is up to you.

Sequential loading can read a teacher dump that is still growing without extra flags.
When the next student iteration would cross the known published prefix, world rank 0
lists the cache once and shares any newly published groups with every rank. If there are
still too few samples, all ranks raise the same exhaustion error; replay does not sleep
or poll. Start the student after the teacher has published enough data to stay ahead of
consumption. A flush range is published once its inputs and targets tars exist for every
teacher DP rank. Replay never crosses a hole in the published prefix; when later ranges
are already published (for example, parallel dump jobs that are still writing), it logs
a warning naming the gap, and only fails if the student actually needs those samples.

### Batch and parallelism changes

Students may change DP, microbatch size, global batch size, and use batch-size ramp-up.
Replay derives the next iteration's consumption from Megatron's current microbatch
calculator. Each student rank streams the teacher DP tars it needs, decoding a bounded
window of teacher iterations ahead.

When the student DP size does not divide the teacher's, or the student microbatch is
smaller than the teacher's, several student ranks read the same teacher tar, so the
bytes read exceed the dump size by that factor (as in V2).

The full saved sequence is partitioned for the student's CP configuration. Only CP rank
0 of each data-parallel replica reads storage; it maps targets for every CP rank and
scatters them. Non-packed prefix shortening slices inputs and targets together
**before** this partitioning. Increasing CP is allowed when the student sequence length
satisfies the partition's divisibility requirements. Packed sequences cannot be
shortened. Their original padding is retained; a CP/layout change is rejected when saved
document padding is incompatible. Replay cannot recover tokens discarded by the teacher's
dataset builder. Changing TP or PP also requires a compatible model checkpoint and
Megatron configuration. The student must retain the teacher's padded vocabulary size;
adjust vocabulary padding when changing TP so cached probabilities over padded vocabulary
slots remain meaningful.

## Storage

Local paths and configured `msc://` storage are supported; remote paths require
`--enable-msc`. A cache is a flat directory with one inputs tar and one targets tar per
teacher DP rank and flush range (global sample range):

```text
dp0__0-51200.inputs.tar
dp0__0-51200.targets.tar
dp1__0-51200.inputs.tar
dp1__0-51200.targets.tar
...
```

Discovery only lists tar names. A tar is published once its object exists: local writes
are staged and renamed, and remote objects become visible only when their upload
completes. Published tars are immutable. Keeping one tar per teacher DP rank means a
missing or corrupt shard points directly at the rank (and node) that wrote it. Ranks that
only need inputs (the first pipeline stage, and middle stages for packed data) never
download targets.

Each tar begins with `_meta.json`, followed by one member per teacher iteration:

| Member | Contents |
| --- | --- |
| `_meta.json` | Format version, kind (`inputs`/`targets`), DP rank, sample range, writing job's generation, `first_sample`, and the shared cache settings: teacher DP/MBS/GBS/CP, save interval, train budget, sequence length, packing and attention flags, tokenizer fingerprint, padded vocabulary, target settings, dataset identity, teacher checkpoint (provenance) |
| `START-END.inputs.pt.zst` | This rank's samples for that iteration (see below) |
| `START-END.targets.pt.zst` | Sparse targets for the same tokens, with a matching `record_id` |

Each data member is one zstd frame whose built-in content checksum is verified on read.

| Input field | Encoding | Purpose |
| --- | --- | --- |
| `tokens`, `labels`, `position_ids` | Flat int32 tensors | Exact processed inputs, including negative labels |
| `loss_mask` | Bool when binary; otherwise float32 | Preserve independent loss weighting |
| `sample_offsets` | Int64 offsets into flat tensors | Retain each original training sample |
| `cu_seqlens` | Per-sample int32 boundary rows, or absent | Preserve packed document attention boundaries |
| `cu_seqlens_padded` | Per-sample int32 boundary rows, when supplied | Preserve the dataset's padded boundary convention |
| `record_id` | `generation:dpR:START-END` | Bind inputs to targets and to the writing job |

Sample IDs are derived from the saved DP/microbatch layout rather than stored.
Targets are canonical `[total_tokens, K]` tensors using the existing log probability
dtype and 17-bit vocabulary indices. Sample offsets address the same flattened token
space. Storage is independent of the teacher's microbatch tensor shape. Boundary
metadata records the dataset's processed, potentially padded layout; it does not claim
to recover original unpadded documents.

The uncompressed input overhead is approximately **13 bytes per token** with a binary
mask, or 16 with fractional weights, plus sample/document offsets and container overhead.
FP16/BF16 targets with 17-bit indices take about `4.125 * K` bytes per token, so binary
inputs add about 2.46% at K=128 before compression. Compression ratios depend on the data.

### Object storage

Remote tars are streamed sequentially with large ranged reads, one request per
`--logits-load-read-chunk-mb` (default 64) chunk, independent of the MSC cache
configuration. `--logits-load-msc-prefetch-depth` sets how many decoded teacher
iterations each teacher-DP-rank stream keeps ahead, and `--logits-load-decode-threads`
how many decode concurrently. Only one rank per pipeline stage lists the cache, and only
when the student reaches the known frontier.

## Checkpoint resume

Student checkpoints use Megatron's normal consumed-sample cursor and saved CLI settings,
plus the shuffled extent when shuffling. Sequential loading resumes at the saved cursor
and discovers new published groups. Prefetched microbatches do not advance the
checkpoint cursor. Students need only read access to cache storage.

You may change `--logits-load-dir` on resume to use a replacement dump. Resume uses the
same logical sample cursor in the replacement directory; it continues identically when
the replacement was dumped identically (for shuffled runs this includes the same teacher
DP size and save interval, which define the shuffle groups). Changing shuffle settings or
sequence length on resume is rejected because it changes the replay plan. Start a new
student run with `--finetune` to select a different plan. A training teacher whose
checkpoint cursor exceeds its published range must resume an earlier checkpoint or create
a new cache directory.

## Initial scope and validation

The adapters support ordinary fixed-length GPT pretraining and fixed-size packed SFT,
including the corresponding hybrid-model entrypoint. Runtime packing, variable-length
dataset adapters, hybrid CP, MTP, CUDA graphs, schedule-plan overlap, dataset phase
transitions, custom dense attention masks, and teacher full-layer recomputation are
rejected. Top-P filtering retains uniform K and its value sentinels, and the accepted
teacher attempt reports `avg-logprobs-kept` just as V2 does.

The CPU suite exercises codec integrity, exact input/target pairing, reshuffling and its
checkpointed extent, DP/MBS changes, ramp-up, prefix shortening, CP token maps and
per-CP-rank target shards, publication and holes from parallel jobs, the collective
replay frontier, chunked remote reads, frozen-teacher range resume and cleanup,
mismatched-settings detection, and differentiable sparse KL:

```sh
uv run --no-project --with torch --with numpy --with zstandard --with pytest \
  python -m pytest -q -o addopts='' tests/offline_kd_cpu
```

GPU training, distributed collectives, and MSC integration still require validation.
When allocation is available, compare a deterministic V2/V3 unpacked run, exercise
TP/CP/PP (including virtual stages), test packed CP compatibility, run two parallel
frozen teacher jobs with a trailing sequential student, and interrupt/resume across range
boundaries.

## Implementation map

The new modules in `megatron/training/distillation/` isolate the format, publication,
replay and token mappings, saver, runtime hooks, and paired loss (`v3_*.py`).
Small hooks in `arguments.py`, `training.py`, `checkpointing.py`, `pretrain_gpt.py`, and
`pretrain_hybrid.py` select the V3 path. V3 reuses existing sparse top-K extraction,
student output capture, the shared LM/sparse-KD loss and reporting, and the persistent
asynchronous writer queue.
