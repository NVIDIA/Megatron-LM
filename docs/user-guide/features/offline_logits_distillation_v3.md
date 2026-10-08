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
Teacher batch-size ramp-up is currently rejected. A dedicated frozen teacher resumes
its data cursor from `/data/teacher-progress` while continuing to load the original
weights. Progress is recorded after the existing async queue completes writes across
all ranks.

Each raw CPU microbatch is captured before TP field filtering, packed flattening, or
CP partitioning. An explicit token mapping gathers CP outputs back into sample order.
The saver checks that every unmasked token has a target. Rerun attempts remain
provisional until the training loop accepts an attempt.

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
`--logits-load-ignore-errors` does not bypass structural V3 errors.

### Reordering, subsets, and growing dumps

For a completed, immutable cache, enable deterministic shard-group reordering:

```sh
--logits-load-shuffle-shards 123
```

Omitting this option keeps sequential loading; `--logits-load-shuffle-shards 0` enables
shuffling with seed zero. The shuffle seed is independent of the teacher dataset seed.

One reorder unit comprises all teacher DP tar shards for the same flush range. Samples
inside that group retain their global saved order. A smaller student budget consumes
a prefix of this selected order, and may stop within a group at a student iteration
boundary. It does not require the teacher's global batch size.

Sequential loading can read a teacher dump that is still growing without extra flags.
When the known published prefix cannot fill the next student iteration, the reader
checks once for newly published shard groups. If there are still too few samples, it
raises an exhaustion error immediately; it does not sleep or poll again. Start the
student after the teacher has published enough data to stay ahead of consumption.
Readers expose a group only after every teacher DP descriptor is published, and never
cross a hole in the published prefix.

Shuffling requires a completion marker matching the cache generation and the complete
published range. Shuffling a growing cache is not supported.

### Batch and parallelism changes

Students may change DP, microbatch size, global batch size, and use batch-size ramp-up.
Replay derives the next iteration's consumption from Megatron's current microbatch
calculator. CPU decoding runs in parallel, with one microbatch of look-ahead within
the current iteration; it does not predict the next ramp-up boundary.

The full saved sequence is partitioned for the student's CP configuration. Non-packed
prefix shortening slices inputs and targets together **before** this partitioning.
Increasing CP is allowed when the student sequence length satisfies the partition's
divisibility requirements. Packed sequences cannot be shortened. Their original
padding is retained; a CP/layout change is rejected when saved document padding is
incompatible. Replay cannot recover tokens discarded by the teacher's dataset builder.
Changing TP or PP also requires a compatible model checkpoint and Megatron configuration.
The student must retain the teacher's padded vocabulary size; adjust vocabulary padding
when changing TP so cached probabilities over padded vocabulary slots remain meaningful.

## Storage

Local paths and configured `msc://` storage are supported; remote paths require
`--enable-msc`. Initialization lists once on world rank zero. At a sequential replay
frontier, one TP-zero rank per pipeline stage lists and reads new descriptors, then
shares the result with that stage's participating DP/CP readers. Listing results are
refreshed when needed rather than retained in a stale glob cache. These collectives
run before iteration decoding; background prefetch never performs distributed refresh.
A cache contains:

```text
_v3_cache.json
dp0__0-800.tar
dp0__0-800.tar.ready.json
dp1__0-800.tar
dp1__0-800.tar.ready.json
...
_v3_complete.json                 # written only after successful teacher completion
```

Each tar contains `_meta.json` and paired, independently compressed members per teacher
iteration: `START-END.inputs.pt.zst` and `START-END.targets.pt.zst`. A ready descriptor
records sample IDs, pairing identities, member sizes, and SHA-256 checksums. The writer
publishes it only after closing the tar. Published descriptors and records are immutable.
Local files are staged and renamed; remote readers rely on completed object writes plus
descriptor publication. On teacher resume, descriptors and tar files beyond an incomplete
group are moved aside with an `.aborted.*` suffix before regenerating that unreadable tail.
The complete published prefix remains immutable. Only one teacher job should write a
cache directory. A resuming teacher removes the completion marker before continuing;
it publishes a new marker only after successful completion. Already-running shuffled
students still require their selected dump to remain immutable.

| Input field | Encoding | Purpose |
| --- | --- | --- |
| `tokens`, `labels`, `position_ids` | Flat int32 tensors | Exact processed inputs, including negative labels |
| `loss_mask` | Bool when binary; otherwise float32 | Preserve independent loss weighting |
| `sample_offsets` | Int64 offsets into flat tensors | Retain each original training sample |
| `cu_seqlens` | Per-sample int32 boundary rows, or absent | Preserve packed document attention boundaries |
| `cu_seqlens_padded` | Per-sample int32 boundary rows, when supplied | Preserve the dataset's padded boundary convention |
| `sample_ids`, `record_id` | IDs | Bind inputs, targets, and descriptors |

Targets are canonical `[total_tokens, K]` tensors using the existing log probability
dtype and 17-bit vocabulary indices. Sample offsets address the same flattened token
space. Storage is independent of the teacher's microbatch tensor shape. Boundary
metadata currently records the dataset's processed, potentially padded layout; it
does not claim to recover original unpadded documents.

The uncompressed input overhead is approximately **13 bytes per token** with a binary
mask, or 16 with fractional weights, plus sample/document offsets and container overhead.
FP16/BF16 targets with 17-bit indices take about `4.125 * K` bytes per token, so binary
inputs add about 2.46% at K=128 before compression. Compression ratios depend on the data.

## Checkpoint resume

Student checkpoints use Megatron's normal consumed-sample cursor and saved CLI settings.
Completed caches reconstruct shard order from
the shuffle seed. Sequential loading resumes at the saved cursor and discovers new
published groups. Checkpoints do not contain shard descriptors, a duplicate cursor, or
loader worker state. Prefetched microbatches do not advance the checkpoint cursor.
Students need only read access to cache storage.

You may change `--logits-load-dir` on resume to use a replacement dump. Resume uses the
same logical sample cursor in the replacement directory without requiring the original
dump's generation ID. For continuity, provide the corresponding sample order and records
in the replacement dump; shuffled runs also depend on matching shard-group boundaries.
The selected directory's format, applicable completion-marker checks, and input/target
integrity checks still apply.

Changing shuffle settings or sequence length on resume is rejected because it changes
the replay plan. Start a new student run with `--finetune` to select a different plan.
A training teacher whose checkpoint cursor exceeds the complete published prefix must
resume an earlier checkpoint or create a new cache directory.

## Initial scope and validation

The adapters support ordinary fixed-length GPT pretraining and fixed-size packed SFT,
including the corresponding hybrid-model entrypoint. Runtime packing, variable-length
dataset adapters, hybrid CP, MTP, CUDA graphs, schedule-plan overlap, dataset phase
transitions, custom dense attention masks, and teacher full-layer recomputation are
rejected. The flattened format leaves room for future adapters without changing the
current replay semantics. The legacy whole-tar MSC prefetch-depth flag does not control
V3's bounded microbatch prefetch. V3 does not implement V2's whole-tar look-ahead
prefetch, so remote throughput still needs measurement. `--logits-load-ignore-errors`
is rejected: paired input/target loading cannot use the legacy loss-only fallback.
Top-P filtering retains uniform K and its value sentinels, and the accepted teacher
attempt reports `avg-logprobs-kept` just as V2 does.

The CPU suite exercises codec integrity, exact input/target pairing, reshuffling,
DP/MBS changes, ramp-up, prefix shortening, CP token maps, incomplete publication,
single publication refresh and exhaustion, completed-cache resume, teacher reruns, and differentiable sparse KL:

```sh
uv run --no-project --with torch --with numpy --with zstandard --with pytest \
  python -m pytest -q -o addopts='' tests/offline_kd_cpu
```

GPU training, distributed collectives, and MSC integration still require validation.
When allocation is available, compare a deterministic V2/V3 unpacked run, exercise
TP/CP/PP (including virtual stages), test packed CP compatibility, and interrupt/resume
a teacher and a student reading a growing dump across publication boundaries.

## Implementation map

The new modules in `megatron/training/distillation/` isolate the format, publication,
replay and token mappings, saver, runtime hooks, and paired loss (`v3_*.py`).
Replay collation and CP mappings live together in `v3_replay.py`. Small hooks
in `arguments.py`, `training.py`, `checkpointing.py`, `pretrain_gpt.py`, and
`pretrain_hybrid.py` select the V3 path. V3 reuses existing sparse top-K extraction,
student output capture, sparse KL, and the persistent asynchronous writer queue.
