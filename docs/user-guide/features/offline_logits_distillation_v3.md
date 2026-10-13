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

Use the same tokenizer as the teacher (checked by fingerprint, see
[Tokenizer fingerprint](#tokenizer-fingerprint)) and settings that reproduce the cache's
`attention_mode`. The student's padded vocabulary must be at least the tokenizer size; it
need not equal the teacher's. Target entries whose index is beyond the student vocabulary
(teacher padding rows) are masked into the ghost bucket, counted, and reported when they
exceed 0.01% of entries. Stored labels, loss weights and positions are replayed exactly.
Dataset identity is provenance rather than a student alignment constraint;
`--logits-load-ignore-hash` is unnecessary for V3. Member checksums and pairing identities
are always checked; `--logits-load-ignore-errors` is rejected.

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

Dumps store no padding (see [Stored records](#stored-records)); the student lays each
sample out again for its own configuration. Only CP rank 0 of each data-parallel replica
reads storage; it maps targets for every CP rank and scatters them.

- Non-packed samples keep their prefix up to the student sequence length (prefix
  shortening) and pad the tail; the student length cannot exceed the teacher's.
- Packed samples (`isolated_segments`) pad each segment to the student's own SFT divisor
  (`2·CP` when CP > 1, times TP with sequence parallelism) and extend the last segment to
  the student sequence length, as Megatron SFT does. The student divisor must divide the
  cache's `padding_divisor`, which guarantees every pack still fits; a smaller divisor
  yields less padding than the teacher used. A packed student sequence length may exceed
  the teacher's but not fall below it.

Replay cannot recover tokens discarded by the teacher's dataset builder. Changing TP or PP
also requires a compatible model checkpoint and Megatron configuration.

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

Each tar begins with `_meta.json`, followed by one member per teacher iteration
(`START-END.inputs.pt.zst` or `START-END.targets.pt.zst`, with a matching `record_id`).
Each data member is one zstd frame whose built-in content checksum is verified on read,
holding a `torch.save` dictionary of CPU tensors.

The rest of this section is the producer-neutral specification. Megatron is one producer;
other producers (for example a vLLM-based dumper) and consumers (for example a Hugging
Face loader) implement the same rules. `v3_format.py` has no Megatron dependencies and
can be reused directly. Golden dumps under `tests/offline_kd_cpu/fixtures/` must decode
identically in every implementation.

### Metadata

```json
{"format_version": 3, "producer": "megatron",
 "kind": "inputs", "dp_rank": 0, "range": [0, 51200], "generation": "<job id>",
 "contract": {...},
 "provenance": {...}}
```

`contract` must be identical across every tar of a cache, whichever producer wrote it:

| Contract key | Meaning |
| --- | --- |
| `first_sample` | Global sample index where replay starts |
| `dp_size_save`, `mbs_save`, `gbs_save` | Writer layout: sample `x` of an iteration starting at `s` lives in rank `((x-s)//mbs) % dp`, row `((x-s)//(mbs·dp))·mbs + (x-s)%mbs`; one record per iteration of `gbs` samples |
| `seq_length` | Sample (pack) capacity |
| `padding_divisor` | Per-segment padding granularity the writer used (1 when segments were not padded) |
| `padding_policy` | Always `"strip"` (see [Stored records](#stored-records)) |
| `attention_mode` | `causal`, `causal_eod_reset` or `isolated_segments` (see [Context](#context)) |
| `tokenizer` | `{"sha256", "size", "special_ids", "eod_id"}` (see [Tokenizer fingerprint](#tokenizer-fingerprint)) |
| `logprob_vocab_size` | Size of the vocabulary the teacher normalized over; every valid index is below it |
| `temperature` | Always `1.0` |
| `targets` | `{"k", "p", "min_k", "save_dtype", "index_encoding"}` |

`provenance` is producer-specific and never compared across producers. Megatron records
`cp_size_save`, `save_interval`, `train_budget`, `sft`, `inter_document_masking`,
`reset_attention_mask`, `dataset_identity` and `teacher_checkpoint`; Megatron dump jobs
additionally require matching `save_interval` and `train_budget` among themselves so
parallel jobs share flush boundaries and dataset order.

### Stored records

Within each segment, the writer drops the trailing run of positions with zero loss
weight. Under causal attention no kept position attends to them and they are never
scored, so this is lossless; it removes dataset padding (per-document and pack tails) and
trailing masked turns alike. Leading or inner zero-weight positions (prompts) are kept.
Targets drop the same rows. Consequently every stored segment ends on a scored position.

| Inputs field | Required | Encoding / default |
| --- | --- | --- |
| `record_id` | yes | `generation:dpR:START-END`, also in the targets member |
| `sample_offsets` | yes | int64 `[samples + 1]` offsets into the flat token tensors (empty samples allowed) |
| `tokens` | yes | int32 flat token IDs |
| `segment_lengths`, `segment_counts` | for `isolated_segments` | int32 positive lengths of each sample's segments, and segments per sample; otherwise each non-empty sample is one segment |
| `position_ids` | no | int32; default counts from 0 within each segment |
| `labels` | no | int32; default is the next token within the segment, and `segment_final_labels[i]` at segment `i`'s last position |
| `segment_final_labels` | when `labels` is omitted | int32, one per segment |
| `loss_mask` | no | bool, or float32 for fractional weights; default all ones |

Writers may omit an optional field only when it equals its default; Megatron does so
automatically (RoPE models never read `position_ids`, so most Megatron dumps omit them).
Readers fill defaults before use. A target row with no valid entry (all sentinel)
contributes zero KD loss; its stored loss weight still controls CE supervision.
Student-inserted padding has zero loss weight for both losses.

### Target contract

- Target row `t` stores the distribution from the output-layer logits at input position
  `t`, predicting the next token. Producers using prompt-token scores (e.g. vLLM
  `prompt_logprobs`, which score token `t` given the prefix) shift by one.
- Values are full-vocabulary-normalized log probabilities at **T=1** over
  `logprob_vocab_size` entries, followed by top-K selection and optional top-P masking.
  They are never renormalized over K. Temperature-scaled dumps are rejected; any
  temperature belongs in the student loss.
- Every row has exactly K entries. A logprob of `-1000` marks a masked or unavailable
  entry; its decoded index must be ignored. Rows with fewer than K available entries are
  filled with it.
- The loss derives ghost residual mass as `1 - sum(exp(retained_logprobs))`; no residual
  tensor is stored.

| Targets field | Encoding |
| --- | --- |
| `values` | `[tokens, K]`, `save_dtype` |
| `indices_low` | `[tokens, K]`, uint16 lower 16 bits |
| `indices_high` | `[tokens, K]`, uint8 bits 16–23; present only for `index_encoding: "u16+u8"` |
| `index_encoding` | `"u16"` when `logprob_vocab_size <= 65536`, else `"u16+u8"` (up to `2**24`) |

The `-1` selection sentinel encodes as all ones. Readers validate dtypes, shapes, the
encoding against `logprob_vocab_size`, and that valid entries index below it. TP gathering
in the Megatron saver uses FP32 transport, exact for every supported index. V1/V2 keep
their 17-bit encoding.

### Context

| `attention_mode` | Context construction |
| --- | --- |
| `causal` | Causal attention over each sample |
| `causal_eod_reset` | Causal attention reset after `eod_id` tokens |
| `isolated_segments` | Causal attention within each stored segment |

Megatron derives the mode from its dataset path: `reset_attention_mask` without a
dataloader attention mask still yields `causal`. Students must reproduce the mode.
`loss_mask` controls scoring, not attention; a zero-weight prompt still supplies context.

### Tokenizer fingerprint

`sha256` hashes this canonical object, serialized with
`json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)` as UTF-8:

```json
{"vocab": [[token_string, id], ...],
 "special_ids": {"bos": .., "eos": .., "pad": .., "unk": ..},
 "size": n}
```

For Hugging Face tokenizers, `vocab` is `get_vocab()` (including added tokens) sorted by
ID, `special_ids` are the `*_token_id` attributes (null when absent), and `size` is
`len(tokenizer)` (`v3_format.hf_tokenizer_identity`). Megatron fingerprints its wrapped
Hugging Face tokenizer the same way. Normalizers, pre-tokenizers and chat templates are
excluded because inputs are stored as IDs. Megatron's EOD is recorded as `eod_id` for
`causal_eod_reset` and is not hashed.

### Writing V3 from vLLM

A non-Megatron teacher writes the same tars with `producer` set accordingly. The writer
owns tokenization and packing; vLLM only scores token IDs.

**Scoring existing documents (default).**

1. Tokenize each document with the teacher's Hugging Face tokenizer (apply the chat
   template, if any) and append EOS. Pass the IDs to vLLM as `prompt_token_ids` with
   `SamplingParams(max_tokens=1, prompt_logprobs=K)`, configuring vLLM to return raw
   (unprocessed) logprobs (the `logprobs_mode` option in recent versions). vLLM requires
   at least one generated token; discard it.
2. Row `t` takes `prompt_logprobs[t + 1]`, so the last content token's row is the
   distribution over EOS. Keep exactly the top K entries (vLLM may add the prompt token's
   own logprob); fill missing entries with `-1000`.
3. Give the EOS position zero loss weight (it has no next token); the stripping rule
   removes it. Omit `labels` and store `segment_final_labels` = EOS: the stored segment
   then matches what a Megatron SFT dataset produces.
4. Store `loss_mask` only when some tokens are not trained on, e.g. from
   `apply_chat_template(..., return_assistant_tokens_mask=True)` for assistant-only loss.
   Omit `position_ids`.

**Teacher-generated responses.** To distill on responses the teacher writes (sequence-level
KD), make each segment prompt plus completion: completion rows come from the generated
tokens' `logprobs=K`, prompt rows from `prompt_logprobs` or sentinel rows with zero loss
weight. Targets must still be raw T=1 logprobs whatever the sampling temperature or top-P;
record sampling settings under `provenance`. A completion ending in EOS gets EOS as its
final label; one cut at `max_tokens` gets zero weight on its last position, which is
stripped. Completions are fixed at dump time; on-policy distillation (student-generated
data) needs a live teacher and is outside V3.

Without appending EOS, the single generated token's `logprobs=K` also provides the last
row, but its final label is then the sampled token rather than a data label.

**Layout.**

- Pack segments into samples of `seq_length` tokens in an `isolated_segments` cache, one
  writer per `dp_rank`, `mbs_save=1`, `gbs_save` samples per record; store no padding.
- `padding_divisor` promises that the samples still fit when each segment is padded to a
  multiple of it. Packing with headroom for, say, 64 lets a Megatron student use any SFT
  divisor (`2·CP`, times TP with sequence parallelism) that divides 64; 1 restricts packed
  students to CP 1 without sequence parallelism.
- Set `logprob_vocab_size` to the model's `config.vocab_size` and fingerprint the
  tokenizer with `hf_tokenizer_identity`.

### Size

Inputs cost about 4 bytes per token for `tokens` plus whichever optional fields differ
from their defaults (labels 4, positions 4, loss mask 1 or 4). FP16/BF16 targets take
about `4·K` bytes per token with `u16` indices, or `5·K` with `u16+u8`, before zstd; the
high byte compresses to a small fraction of a byte per entry for typical token IDs.

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

The CPU suite exercises the producer-neutral format (index encodings, padding stripping and
re-padding, input defaults, metadata contract, tokenizer fingerprint, golden fixtures),
codec integrity, exact input/target pairing, reshuffling and its
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
