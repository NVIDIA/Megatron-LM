# Offline KD V3: Paired Inputs and Targets

V3 stores the teacher's processed training inputs beside its sparse log probabilities.
The student replaces its training dataloader with cache replay. It needs the tokenizer
and model configuration, but does not need the original training dataset or a matching
dataset seed. Existing V1/V2 dumps and training paths retain their behavior when the
new input flags are absent.

See [Storage](#storage) for the directory layouts, Arrow columns, JSON sidecar fields,
and metadata contract. The [student corpus manifest](#student-corpus-manifest) is a
separate, in-memory loading plan.

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
and each flush writes one data file and its index per teacher DP rank.
Teacher batch-size ramp-up, `--iterations-to-skip` (including iterations added from
`--result-rejected-tracker-filename`), and `--allow-ambiguous-pad-tokens` are rejected.
A frozen teacher must score deterministically: it still runs in train mode, so set
`--hidden-dropout 0 --attention-dropout 0` (both default to 0.1); MoE input jitter and
token dropping (`--moe-expert-capacity-factor`, `--moe-token-dropping`) are rejected too.

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

Every file records the cache settings (see [Metadata](#metadata)). A teacher checks them
against an existing file at startup, and students check every file before using its data:
a file whose required settings differ is rejected, and differences in recorded settings
(teacher, data, split, seed, sample budget, ...) are reported by name but don't block.
Corpus sources additionally enforce the sample-order invariants described below.

### Corpus dumps

A corpus dump scores one whole source dataset into its own folder, so that dumps of
several sources can later be blended like ordinary dataset files instead of replayed as
one run. A corpus source is generic, like the files it comes from: complete, unweighted
and unsplit. Weights, repetition and train/validation choices belong to blends. It is a
frozen-teacher dump with three restrictions: one unweighted `--data-path` prefix (or SFT
file), all of it exactly once as Megatron builds it (GPT windows or SFT rows, at the
teacher's `--seq-length`; `--split` is ignored), and a seeded permutation (`--seed`), so
every block is a random sample of the source:

```sh
for source in web_crawl math_sft code; do
  torchrun ... pretrain_gpt.py <frozen teacher args> --eval-iters 0 \
    --data-path "$DATA/$source" --kd-corpus-dump "$source" \
    --logits-save-dir "$CORPUS" --logits-save-top-k 128 \
    --save "$PROGRESS/$source" --save-interval 100
done
```

Each job writes `<corpus>/<source>/` and sizes itself to the source, ignoring
`--train-iters`. Jobs for
different sources can run in parallel, and each source can itself be split into parallel
range jobs as above. Sources of one corpus must share a tokenizer (a job refuses to start
otherwise); a different teacher is reported, not refused.

The epoch is padded to a whole number of global batches by repeating its first samples;
the source's epoch size is recorded as the required `num_samples` setting, and samples at
or beyond it are filler that corpus readers ignore. A source is complete when its files
cover `[0, num_samples)` padded to whole batches. Megatron's own dataset builder decides
the samples, so SFT truncation and pretraining windows are exactly as in training.
The writer first builds the ordinary GPT/SFT dataset and then applies the one-epoch
wrapper. The dump seed controls both underlying dataset construction and the wrapper's
permutation; student blend ordering has its own seed.

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
or `--train-iters` to the desired student budget. Sequential replay consumes each saved
sample once. Shuffled replay and corpus blends can repeat samples as described below.
The available cache must cover the complete final student global batch. If validation
is enabled, provide an explicit ordinary validation dataset (or mock data).
Validation uses LM loss without cached KD
targets, and does not build the original training split.

Use the same tokenizer as the teacher (checked by fingerprint, see
[Tokenizer fingerprint](#tokenizer-fingerprint)) and settings that reproduce the cache's
`attention_mode`. The student's padded vocabulary must be at least the tokenizer size; it
need not equal the teacher's. Target entries whose index is beyond the student vocabulary
(teacher padding rows) are masked into the ghost bucket, counted, and reported when they
exceed 0.01% of entries. Stored labels, loss weights and positions are replayed exactly.
The teacher's dataset settings are recorded rather than checked against the student;
`--logits-load-ignore-hash` is unnecessary for V3. Readers check file sizes, schemas and
input/target row alignment. V3 does not store content checksums or recompute hashes of
the original dataset; `--logits-load-ignore-errors` is rejected.

### Training on a corpus blend

A student trains on a blend of corpus sources (see [Corpus dumps](#corpus-dumps)) with
`--kd-corpus-blend blend.json` instead of `--logits-load-inputs`:

```json
{"corpus": "/data/corpus", "sources": {"web_crawl": 0.7, "math_sft": 0.3},
 "seed": 1234, "lanes": 64}
```

Weights are in samples, as in Megatron's `BlendedDataset`. At startup, world rank 0
plans the schedule for the training budget (`--train-samples`, or `--train-iters` times
the global batch size) and logs each source's realized share of samples and of supervised
tokens:

- Each source contributes whole seeded passes over its blocks, then a seeded subset.
  An allocation larger than the source's epoch repeats it; a smaller allocation selects
  a subset. Weights are relative proportions, not epoch counts. Filler samples are
  skipped, and a source whose dump is not finished is refused.
- Every source is spread evenly along one stream, and the stream is cut into `lanes`
  equal lanes. Global sample `p` is position `p // lanes` of lane `p % lanes`, so the
  schedule does not depend on the student's DP size, and every global batch draws from
  up to `lanes` places in the stream.
- When `lanes` is a multiple of DP × microbatch size, each lane is read by exactly one
  student rank, one ranged read per block. Otherwise ranks share lanes and read the same
  blocks. Each rank keeps about `lanes / DP × --logits-load-msc-prefetch-depth` blocks
  (16 MiB uncompressed each) in flight.

Weights affect the student's consumption only. Every real sample is dumped once,
regardless of any later blend; final-batch fillers are excluded by `contract.num_samples`.
The manifest does not store another `weight` field per source: requested weights remain
in its copy of the blend configuration.

The sources must share tokenizer identity, attention mode and prefix stability. Their
normalization vocabulary sizes, index encodings and top-K widths may differ: decoding
uses each file's vocabulary size, and collation pads K to the batch maximum with masked
entries. EOD IDs must agree only for EOD-reset attention. Teacher identity is recorded
per source. The checkpoint pins the plan's hash: a resume re-plans and refuses a different
file table or sample schedule. Changing DP or microbatch size on resume is fine.
Shuffling flags do not apply. Validation works as with replay, from `--data-path` with
LM loss only; to hold data out of a blend, dump it as its own source and leave it out.

Outside Megatron, `format/blend.py` plans the same schedule and `format/hf.py` is a
torch-only reference consumer: `CorpusStream` yields one rank's samples in Megatron's
order, `collate` pads them into `input_ids`, `attention_mask`, `position_ids`, `labels`
and the teacher's top-K targets, and `topk_kl` computes the same sparse KL with the tail
bucket as Megatron. Stored labels are already shifted to line up with the logits at the
same position.

`CorpusStream` already assigns samples to DP ranks; do not wrap it in another distributed
dataset sharder (including automatic Trainer/Accelerate sharding). Use a DataLoader with
`num_workers=0` or `1`; more workers are rejected rather than duplicating the stream.
The reader attaches attention mode and EOD identity to each sample. For isolated segments
and EOD-reset attention, the collator constructs a 4D additive block-causal mask, independent
of loss weights. Use a standard-attention HF model/backend that honors custom masks
(`eager` or `sdpa`); FlashAttention and recurrent/linear attention require a dedicated
boundary adapter. The reference mask uses quadratic memory. Plain causal samples retain
the usual 2D padding mask. Hand-built samples must supply equivalent attention metadata
or pass `attention_mode` and, when needed, `eod_id` to `collate`.

The HF loss uses the actual student logit width to move unsupported teacher target IDs
into the residual bucket, even if `student_vocab_size` was not supplied to the reader.

### Reordering, subsets, and growing dumps

Enable deterministic shard-group reordering with:

```sh
--logits-load-shuffle-shards 123
```

Omitting this option keeps sequential loading; `--logits-load-shuffle-shards 0` enables
shuffling with seed zero. The shuffle seed is independent of the teacher dataset seed.
One reorder unit is one flush range (all teacher DP files for it). Samples inside a group
retain their global saved order. A smaller student budget consumes a prefix of the
selected order, and may stop within a group at a student iteration boundary. It does not
require the teacher's global batch size.

Shuffling uses the groups published when the student first starts; that extent is saved
with the student checkpoint and reused on resume, even if the teacher keeps publishing.
Whether a dump is finished before shuffling it is up to you.

With shuffling, a student budget (`--train-samples`, or `--train-iters` times the global
batch size) larger than the shuffled extent replays it in whole passes, each group once per
pass and each pass in a fresh order from the same seed (the first pass is the order a
smaller budget would see). Raising the budget on resume only appends passes. Sequential
loading never repeats: it reads a possibly growing dump and fails when the student runs
past what is published.

Sequential loading can read a teacher dump that is still growing without extra flags.
When the next student iteration would cross the known published prefix, world rank 0
lists the cache once and shares any newly published groups with every rank. If there are
still too few samples, all ranks raise the same exhaustion error; replay does not sleep
or poll. Start the student after the teacher has published enough data to stay ahead of
consumption. A flush range is published once its index exists for every
teacher DP rank. Replay never crosses a hole in the published prefix; when later ranges
are already published (for example, parallel dump jobs that are still writing), it logs
a warning naming the gap, and only fails if the student actually needs those samples.

### Batch and parallelism changes

Students may change DP, microbatch size, global batch size, and use batch-size ramp-up.
Replay derives the next iteration's consumption from Megatron's current microbatch
calculator. Each student rank streams the teacher DP files it needs, decoding a bounded
window of blocks ahead.

When the student DP size does not divide the teacher's, or the student microbatch is
smaller than the teacher's, several student ranks read the same teacher block, so the
bytes read exceed the dump size by that factor (as in V2).

Dumps store no padding (see [Stored records](#stored-records)); the student lays each
sample out again for its own configuration. Only CP rank 0 of each data-parallel replica
reads storage; it maps targets for every CP rank and scatters them.

Lengths are checked per sample, not against the producer's recorded `seq_length` or
`padding_divisor`:

- Non-packed samples pad their tail to the student sequence length. A longer sample
  keeps its prefix (prefix shortening), which requires a `prefix_stable` cache (see
  [Metadata](#metadata)); otherwise it is an error naming the sample. A sample whose
  supervised tokens all lie beyond the student length trains on nothing; the first one
  is reported and the rest are counted.
- Packed samples (`isolated_segments`) pad each segment to the student's own SFT divisor
  (`2·CP` when CP > 1, times TP with sequence parallelism) and extend the last segment to
  the student sequence length, as Megatron SFT does. A pack that no longer fits is an
  error naming the sample, except a single segment, which is prefix-shortened like a
  non-packed sample.

Replay cannot recover tokens discarded by the teacher's dataset builder. Changing TP or PP
also requires a compatible model checkpoint and Megatron configuration.

## Storage

Local paths and configured `msc://` storage are supported; remote paths require
`--enable-msc`.

### Directory layouts

A replay dump is one flat directory. For two teacher DP ranks and two flushes:

```text
/data/teacher-v3/                      # --logits-save-dir / --logits-load-dir
├── dp0__0-51200.arrow                 # inputs and targets together
├── dp0__0-51200.index.json            # publishes this writer's file
├── dp1__0-51200.arrow
├── dp1__0-51200.index.json
├── dp0__51200-102400.arrow
├── dp0__51200-102400.index.json
├── dp1__51200-102400.arrow
└── dp1__51200-102400.index.json
```

The naming rule is `dp{writer_rank}__{start}-{end}.{arrow|index.json}`. `start` and
`end` are global **sample counts**, with an exclusive end; they are not iterations or
token offsets. A sample is a GPT window or an SFT dataset row, which may contain several
isolated segments. With two writers, each file for `[0, 51200)` holds 25,600 samples.
The files share a range name because together they cover that global range.

A corpus root contains one such directory per source. The source directory name comes
from `--kd-corpus-dump`, and the student blend configuration refers to that name:

```text
/data/corpus/                         # teacher --logits-save-dir; blend "corpus"
├── web_crawl/                        # --kd-corpus-dump web_crawl
│   ├── dp0__0-51200.arrow
│   ├── dp0__0-51200.index.json
│   ├── dp1__0-51200.arrow
│   ├── dp1__0-51200.index.json
│   └── ...
└── math_sft/                         # --kd-corpus-dump math_sft
    ├── dp0__0-51200.arrow
    ├── dp0__0-51200.index.json
    ├── dp1__0-51200.arrow
    ├── dp1__0-51200.index.json
    └── ...

/data/student-blend.json              # user-authored configuration, outside the dump
/data/teacher-progress/<job>/         # normal Megatron checkpoints from --save
/data/student-progress/               # normal student checkpoints from --save
```

Each source has its own writer layout and epoch size. Teacher progress checkpoints are
separate from the Arrow dump: each parallel dump job needs its own progress directory.
The student blend JSON is an input configuration, not an automatically generated dump
file. V3 creates no separate input/target files, `metadata.json`, completion marker, or
persisted corpus-manifest file. Required settings are embedded in every Arrow file and
also available through the footer copy in its index.

### Arrow files and publication

A data file is an [Arrow IPC file](https://arrow.apache.org/docs/format/Columnar.html#ipc-file-format)
with zstd-compressed buffers: one row per sample, the input columns first and then the
target columns, grouped into record batches ("blocks") of at most 16 MiB uncompressed
(a larger sample gets a block of its own). PyArrow opens the file directly; V3 adapters
interpret its sample columns and the settings in schema metadata under `offline_kd`.

The index is written after its data file is complete, so its existence publishes the
range for that writer: local writes are staged and renamed, and remote objects become
visible only when their upload completes. Discovery only lists index names. A complete
flush group has an index for every writer rank `0..writers-1`. Missing writers or a gap
prevent replay from advancing through that group; overlapping ranges are rejected.
A published range is never replaced; a data file without an index (an interrupted attempt) is
ignored and overwritten by a retry. Temporary local writes may leave
`<filename>.<uuid>.tmp` files after interruption; they are not published files.

### JSON index

The sidecar is a range-read directory and publication marker. Its footer and block
locations duplicate information obtainable from Arrow so that a reader can start with
one small index read and then fetch only the blocks it needs.

| Index key | Meaning and purpose |
| --- | --- |
| `index_version` | Currently `1`; versions the custom sidecar representation independently of V3 sample semantics. Unsupported versions fail validation. |
| `size` | Arrow file size in bytes. Supplies the virtual range reader's file length and detects a size mismatch between index and data. It is not a content checksum. |
| `footer` | Base64 copy of the Arrow IPC footer and trailer, including schema metadata. Avoids a separate footer fetch from the data file. |
| `blocks` | Ordered array describing the file's Arrow record batches. Enables sample lookup and bounded range reads. |
| `blocks[].first_row` | Zero-based sample row where the block starts within this writer's file. Enables locating a sample without reading preceding blocks; derivable from preceding row counts. |
| `blocks[].rows` | Number of sample rows in the block. Used for planning and range/row-count validation. |
| `blocks[].offset` | Byte offset of the record-batch message in the Arrow file. |
| `blocks[].length` | Byte length to fetch for the complete block, including inputs and targets. |
| `blocks[].input_length` | Byte length of the block prefix needed for its input columns. Lets pipeline stages that need no targets avoid downloading them. |
| `blocks[].supervised` | Number of target rows, equal to the count of nonzero loss weights. Used to estimate supervised-token blend shares without decoding inputs; not used to compute the training loss. |

A reader fetches each block with one ranged read, or only its input columns (the first
pipeline stage, and middle stages for packed data, never download targets). Keeping one
file per writer means a missing or corrupt file points directly at the rank (and node)
that wrote it.

The size checks detect truncation and many accidental file replacements. Same-size
content changes are not guaranteed to be detected. Supervised-token reporting for a
partially selected block uses its average per-sample count and is approximate.

The rest of this section is the producer-neutral specification. Megatron is one producer;
other producers (for example a vLLM-based dumper) and consumers (for example a Hugging
Face loader) implement the same rules. `v3_format.py` has no Megatron dependencies and
can be reused directly, as can `format/contract.py` and `format/arrow.py`, which import
nothing from Megatron. Golden dumps under `tests/offline_kd_cpu/fixtures/` must decode
identically in every implementation.

### Metadata

The Arrow schema metadata key `offline_kd` contains a JSON object. The following is
an illustrative **corpus** file's metadata; the fingerprint placeholder represents the
canonical digest described under [Tokenizer fingerprint](#tokenizer-fingerprint):

```json
{
  "format_version": 3,
  "contract": {
    "tokenizer": {
      "sha256": "<canonical tokenizer SHA-256>",
      "size": 128256,
      "eod_id": 128001
    },
    "logprob_vocab_size": 128512,
    "attention_mode": "causal",
    "prefix_stable": true,
    "writers": 2,
    "writer_mbs": 2,
    "first_sample": 0,
    "num_samples": 100000
  },
  "recorded": {
    "teacher": "/checkpoints/teacher@2000",
    "seq_length": 4096,
    "padding_divisor": 1,
    "data": {"kind": "corpus", "path": "/datasets/web_crawl"},
    "seed": 1234,
    "builder": "GPTDataset"
  },
  "provenance": {
    "producer": "megatron",
    "teacher_precision": "bf16",
    "top_k": 128,
    "top_p": null,
    "min_k": 1,
    "save_dtype": "fp16",
    "cp_size": 1,
    "save_interval": 8
  }
}
```

| Top-level key | Purpose and validation |
| --- | --- |
| `format_version` | Required, currently `3`. Selects V3's interpretation rules: aligned next-token targets, full-vocabulary T=1 logprobs, sentinels and stripped padding. Other versions are rejected. |
| `contract` | Required file interpretation and compatibility settings. Missing required fields or incompatible values fail validation. |
| `recorded` | Optional dataset/run description. Replay differences produce warnings. Corpus writing and planning additionally enforce the sample-order fields listed below within each source. |
| `provenance` | Optional, free-form diagnostics. Its values never select a decoder or impose student compatibility constraints. |

#### Contract fields

Every file stores its inputs next to its targets, so a file is self-consistent on its
own. Settings needed to interpret a file are required (`contract`). Files in one replay
cache or corpus source must agree on them. Across sources in a corpus blend, only token
identity, attention mode and prefix stability must agree; normalization domain and writer
layout remain per source/file. The blend manifest records its shared contract separately
and pins each file's normalization size:

| Key | Meaning and reason for storing it |
| --- | --- |
| `tokenizer` | Object describing the token-ID space through the three fields below. Stored token IDs and targets must have the same meaning for the student. |
| `tokenizer.sha256` | Fingerprint of the complete token-to-ID mapping, true vocabulary size and special-token assignments. Prevents accepting different tokenizers merely because their vocabulary sizes happen to match. |
| `tokenizer.size` | True, non-padded vocabulary size. Lets a reader check that the student's output vocabulary can represent every real token; this number cannot be recovered from the fingerprint. |
| `tokenizer.eod_id` | Document-boundary ID for `causal_eod_reset`; required to reconstruct that attention mode and must match the student's EOD ID. Unused for the other modes and excluded from cross-source comparisons for them. |
| `logprob_vocab_size` | Size of the teacher's normalization domain, including any padding entries. Bounds valid target IDs and determines/checks `u16` versus `u16+u8` index encoding. Sources in a blend may differ. It is not used to recompute or adjust the stored probabilities. |
| `attention_mode` | `causal`, `causal_eod_reset` or `isolated_segments` (see [Context](#context)). Prevents changing the context available to each teacher/student position. |
| `prefix_stable` | Whether a retained prefix preserves the teacher's distributions. False for length-dependent position scaling such as dynamic NTK RoPE or LongRoPE's short/long switch. Megatron currently writes true. Shortening a longer sample requires true. |
| `writers` | Teacher DP writer count. Defines which ranks must publish each complete group and participates in mapping global sample IDs to writer files. The student may use a different DP size. |
| `writer_mbs` | Teacher microbatch size. Defines how successive microbatches were interleaved across the writer files; writer count alone cannot reconstruct that order. |
| `first_sample` | Global sample ID where the intended replay dump begins. Separates a deliberate nonzero start from a missing initial range. Corpus and frozen-teacher dumps use zero. |
| `num_samples` | Required only for corpus files: number of real samples in the source's epoch, before final-batch filling. Defines completeness and excludes repeated filler rows. It counts prepared GPT windows or SFT rows, not necessarily raw documents or conversations. |

For a flush starting at global sample `s`, sample `x` lives in writer
`((x-s)//writer_mbs) % writers`, at row
`((x-s)//(writer_mbs*writers))*writer_mbs + (x-s)%writer_mbs` within that writer's file.
Corpus planning uses this layout to recognize fillers; the student manifest then points
directly to selected file/block rows.

For example, if the only available ranges are `[100, 200)` and `[200, 300)`,
`first_sample=0` means the initial `[0, 100)` range is missing. `first_sample=100`
means those same files form a valid start. File names alone cannot distinguish the two
intentions. Student replay positions start at zero even for a nonzero teacher start.

`logprob_vocab_size` is a validation and interpretation field, not a student vocabulary
matching requirement. A Megatron source normalized over 128,512 entries and a vLLM
source normalized over 128,256 entries can share a blend if their true tokenizer
identities match. Their original distributions are retained; V3 performs no correction
for the different normalization domains. Retained IDs outside the student's output
vocabulary join the ghost bucket. Arrow's schema also exposes the index encoding, but
the declared vocabulary size additionally detects IDs outside the teacher's domain.

`format_version` 3 implies T=1 log probabilities and stripped padding (see
[Stored records](#stored-records)).

#### Recorded settings

`recorded` settings describe how the data was produced. They are optional for other
producers. Replay reports differences by name. Within one corpus source, `data`, `seed`,
`builder`, `seq_length` and `padding_divisor` must remain identical across files and resumed
or parallel dump jobs: splicing different sample streams would break one-epoch coverage.
These fields may differ freely between separate sources. Changing source contents in
place is not detected; use a new source directory for a revised dataset.

| Key | Meaning and purpose |
| --- | --- |
| `teacher` | Checkpoint identity, e.g. `/ckpts/teacher@2000`. Helps diagnose mixed teachers; not used to decode targets or require a matching student. |
| `seq_length` | Teacher dataset window/pack length. A corpus guard because changing it changes sample construction. Student lengths are checked against actual stored samples rather than this recorded value. |
| `padding_divisor` | Teacher per-segment padding granularity, or `1` for non-SFT data. A corpus guard because padding can affect which segments fit before truncation. Students calculate their own padding divisor. |
| `data` | Replay's path-agnostic blend description, or corpus's full source path/URI, or mock identity. A corpus guard against appending another source under the same source name. Students read the stored inputs rather than reopening this path. |
| `split` | Replay-only original dataset partition (`--split`), for warnings and provenance. Not needed to load saved samples. Omitted from corpus files, whose source is unsplit. |
| `seed` | Teacher dataset construction and corpus-permutation seed. A corpus guard: using another permutation for a later range can duplicate some original samples and omit others. Students use their own scheduling seed. |
| `num_samples` | Replay-only requested teacher sample budget, for provenance/warnings. It does not determine published availability. Omitted from corpus `recorded`; the essential corpus epoch size lives in `contract.num_samples`. |
| `builder` | Producer's dataset-construction label. Megatron currently writes the hard-coded labels `GPTDataset`, `MockGPTDataset`, or `SFTDataset` (including mock SFT). A coarse corpus guard within one source; other producers need not use Megatron class names, and separate sources may differ. It does not fingerprint every preprocessing option. |

The `data` object has a `kind` tag and one of these representations:

| `data.kind` | Additional fields | Purpose |
| --- | --- | --- |
| `corpus` | `path`: canonical local path or unchanged storage URI | Identifies the single source being dumped. Contains no weight. |
| `blend` | `blend`: `[[weight, source_basename_without_extension], ...]` | Describes the original replay teacher's blend for diagnostics. These weights are not a corpus sampling policy. |
| `blend_per_split` | `train`: the same weighted-name representation for the training split | Describes replay's original split-specific training blend. |
| `mock` | `mock: true` | Identifies synthetic source data; the boolean duplicates the tag and is diagnostic. |

Recorded settings describe the source but do not checksum it. They cannot establish
identity after an in-place data change, or detect every change in tokenizer templates
or preprocessing code.

#### Producer provenance

`provenance` is free-form and never compared. Omitting these fields does not change
decoding or student training; they explain how the stored targets were produced.

| Key | Meaning and purpose |
| --- | --- |
| `producer` | Writer implementation, currently `megatron`. Does not select a framework-specific decoder. |
| `teacher_precision` | Teacher execution numerics, e.g. `bf16` or `bf16+fp8:e4m3/delayed`. Useful when comparing target distributions. |
| `top_k` | Requested maximum retained entries per target row. Actual K is already in Arrow's fixed-size-list type and can be smaller if the requested K exceeds the vocabulary. |
| `top_p` | Optional nucleus-filtering threshold, or null. Stored sentinels and tail fully describe the resulting masked rows without this setting. |
| `min_k` | Minimum entries retained by nucleus filtering. Records the production policy; no additional loading behavior. |
| `save_dtype` | Stored target value precision, `fp16` or `fp32`. Duplicates the actual Arrow value type for inspection. |
| `cp_size` | Teacher context-parallel size. Diagnostic only: V3 reconstructs canonical samples before storage, so the student need not match it. |
| `save_interval` | Teacher flush interval in iterations. Readers discover actual sample ranges rather than deriving them from this value. |

The saver also keeps `published_through` in memory to avoid republishing an existing
prefix after resume. It is stripped from file metadata. There is no `range_end` metadata
field; the running job derives its end from the current dump-range arguments.

### Stored records

Within each segment, the writer drops the trailing run of positions with zero loss
weight. Under causal attention no kept position attends to them and they are never
scored, so this is lossless; it removes dataset padding (per-document and pack tails) and
trailing masked turns alike. Leading or inner zero-weight positions (prompts) are kept as
context. Consequently every stored segment ends on a scored position.

Targets are stored only for positions with nonzero loss weight: target row `i` of a
sample belongs to its `i`-th such position, in token order, and the row count must equal
the number of nonzero weights. Context-only positions such as prompts and earlier
conversation turns carry tokens but no target rows, so a long conversation supervised
only on its final answer stores rows for the answer alone.

Every column is a `large_list` with one entry per sample; all input columns are always
present:

| Input column | Encoding | Purpose |
| --- | --- | --- |
| `tokens` | int32 token IDs; an empty sample is allowed | The exact teacher inputs, equivalent to HF `input_ids`. Replaces the original input dataset during student training. |
| `labels` | int32, next-token label at each position | Preserves CE supervision, including a final label that may not occur in the stored input sequence. Already shifted relative to inputs. |
| `position_ids` | int32 | Preserves teacher positioning and resets when reconstructing the student batch. |
| `loss_mask` | bool, or float32 for fractional weights; one type per file | Defines scored positions, target-row alignment and loss weighting. Zero-weight prompts remain valid context. It is authoritative even when labels alone do not reveal the scoring mask. |
| `segments` | int32 positive lengths summing to the sample's token count; one segment for a non-packed sample | Preserves isolated attention boundaries and allows each segment to be re-padded for the student. |

Arrow's outer list offsets preserve sample boundaries; no separate stored
`sample_offsets` tensor is needed. There is no `cu_seqlens` column: cumulative segment
lengths reconstruct it after applying the student's padding. Dense attention masks are
not stored; the reader reconstructs the declared `attention_mode` from segment lengths
or EOD tokens. The loss mask describes scoring independently of those boundaries.

A target row with no valid entry (all sentinel)
contributes zero KD loss; its stored loss weight still controls CE supervision.
Student-inserted padding has zero loss weight for both losses. The student places target
rows back at their positions; positions without a row get sentinel targets.

### Target contract

- Target row `t` stores the distribution from the output-layer logits at input position
  `t`, predicting the next token. Producers using prompt-token scores (e.g. vLLM
  `prompt_logprobs`, which score token `t` given the prefix) shift by one.
- Values are full-vocabulary-normalized log probabilities at **T=1** over
  `logprob_vocab_size` entries, followed by top-K selection and optional top-P masking.
  They are never renormalized over K. Producers must supply raw T=1 targets; V3 has no
  separate temperature field and cannot identify arbitrary temperature scaling from the
  probabilities alone.
- Every row has exactly K entries. A logprob of `-1000` marks a masked or unavailable
  entry; its decoded index must be ignored. Rows with fewer than K available entries are
  filled with it. K is fixed within an Arrow file; files from different sources may use
  different K. Readers pad narrower rows with masked entries when constructing a batch,
  without changing their stored tail.
- `tail` (optional, but always written by Megatron) is the log of the
  probability mass outside each row's retained entries, including entries masked by
  top-P: `log(1 - sum(exp(retained)))`. Writers compute it from full-precision log
  probabilities **before** casting values to `save_dtype` (`v3_format.tail_logprob`);
  computing it afterwards from rounded values distorts small tails of confident rows
  (with fp16 values, a 0.1% tail can be off by about 2%). A row whose
  distribution is fully retained stores `-1000`. Readers reject a stored tail unless
  `sum(exp(retained)) + exp(tail)` is within 0.05 of 1 for every row. Without the field,
  readers derive the tail from the stored values.
- The loss uses the tail as the teacher's ghost bucket against the student's mass outside
  the same entries. Entries a student cannot represent (indices beyond its vocabulary)
  are moved into the tail.

| Target column | Encoding (per sample, one entry per target row) | Purpose |
| --- | --- | --- |
| `values` | `fixed_size_list<float16 or float32, K>` | Retained log probabilities; `-1000` marks entries the loss ignores. Actual K and dtype are carried by this Arrow type. |
| `indices_low` | `fixed_size_list<uint16, K>` | Lower 16 bits of the vocabulary IDs corresponding to `values`. |
| `indices_high` | `fixed_size_list<uint8, K>`; present for `u16+u8` encoding | Bits 16–23, required to represent IDs above 65535. Sentinel entries' decoded IDs are ignored. |
| `tail` | float32, optional; always written by Megatron | Teacher probability mass outside retained entries for the mandatory ghost bucket. Preserves the residual calculated before value rounding; absent tails are reconstructed from stored values. |

The index encoding is derived from `logprob_vocab_size`: `u16` up to 65536, else `u16+u8`
(up to `2**24`). There is no separately stored `index_encoding` metadata key.

The `-1` selection sentinel encodes as all ones. Readers validate the row counts, the
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

`sha256` hashes the complete vocabulary and special-token assignments, serialized with
`json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)` as UTF-8.
For example, a three-token tokenizer would use this object:

```json
{"vocab": [["<bos>", 0], ["<eos>", 1], ["hello", 2]],
 "special_ids": {"bos": 0, "eos": 1, "pad": null, "unk": null},
 "size": 3}
```

For Hugging Face tokenizers, `vocab` is `get_vocab()` (including added tokens) sorted by
ID, `special_ids` are the `*_token_id` attributes (null when absent), and `size` is
`len(tokenizer)` (`v3_format.hf_tokenizer_identity`). Megatron fingerprints its wrapped
Hugging Face tokenizer the same way. Normalizers, pre-tokenizers and chat templates are
excluded because inputs are stored as IDs. Megatron's EOD is recorded as `eod_id` for
`causal_eod_reset` and is not hashed.
The canonical `special_ids` object participates in the fingerprint but is not separately
stored in the tokenizer metadata.

### Writing V3 from vLLM

A non-Megatron teacher writes the same files, filling the required settings and whatever
recorded settings and provenance (including `producer` and `teacher_precision`, e.g.
vLLM's `dtype` and `quantization`) it can. The writer owns tokenization and packing; vLLM
only scores token IDs.

**Scoring existing documents (default).**

1. Tokenize each document with the teacher's Hugging Face tokenizer (apply the chat
   template, if any) and append EOS. Pass the IDs to vLLM as `prompt_token_ids` with
   `SamplingParams(max_tokens=1, prompt_logprobs=K)`, configuring vLLM to return raw
   (unprocessed) logprobs (the `logprobs_mode` option in recent versions). vLLM requires
   at least one generated token; discard it.
2. Row `t` takes `prompt_logprobs[t + 1]`, so the last content token's row is the
   distribution over EOS. Keep exactly the top K entries (vLLM may add the prompt token's
   own logprob); fill missing entries with `-1000`. Compute `tail` from these
   float32 values before casting them to `save_dtype`, and keep only the rows of
   positions with nonzero loss weight.
3. Give the EOS position zero loss weight (it has no next token); the stripping rule
   removes it. Labels are the next tokens, with EOS as the last kept position's label:
   the stored segment then matches what a Megatron SFT dataset produces.
4. Set `loss_mask` to zero on tokens that are not trained on, e.g. from
   `apply_chat_template(..., return_assistant_tokens_mask=True)` for assistant-only loss.
   Shift this token mask with the labels: weight at input position `t` describes whether
   its label (token `t + 1`) is supervised, not whether input token `t` is an assistant token.
   Positions count from 0 within each segment.

**Teacher-generated responses.** To distill on responses the teacher writes (sequence-level
KD), make each segment prompt plus completion: completion rows come from the generated
tokens' `logprobs=K`; prompt positions have zero loss weight and need no rows, so
`prompt_logprobs` can be skipped. Targets must still be raw T=1 logprobs whatever the sampling temperature or top-P;
record sampling settings under `provenance`. A completion ending in EOS gets EOS as its
final label; one cut at `max_tokens` gets zero weight on its last position, which is
stripped. Completions are fixed at dump time; on-policy distillation (student-generated
data) needs a live teacher and is outside V3.

Without appending EOS, the single generated token's `logprobs=K` also provides the last
row, but its final label is then the sampled token rather than a data label.

**Layout.**

- Pack segments into samples in an `isolated_segments` cache with `writers=1` and
  `writer_mbs=1` (or one file per writer, following the writer layout); store no padding.
  `format/arrow.py` (`samples_batch`, `write_file`) and `v3_storage.publish` write the
  files and index.
- A student checks each pack against its own length after re-padding every segment to
  its SFT divisor (`2·CP`, times TP with sequence parallelism), so leave headroom for the
  students you expect; record the headroom you used as `padding_divisor`.
- Set `logprob_vocab_size` to the actual number of vocabulary entries included in the
  raw log-softmax (normally `config.vocab_size`, not a rounded embedding allocation),
  and fingerprint the tokenizer with `hf_tokenizer_identity`. A Megatron source may
  normalize over additional padding entries; this does not prevent blending the sources
  when their true tokenizer identities match. Unsupported retained IDs join the student's
  residual bucket rather than being reassigned to real tokens.
- Set `prefix_stable` to false when the model's `rope_scaling` depends on the sequence
  length (`"dynamic"`, or `"longrope"` when sequences may exceed
  `original_max_position_embeddings`), true otherwise.

### Size

Inputs cost about 12 bytes per token (tokens, labels and positions) plus the loss mask
(1 bit or 4 bytes), before zstd. FP16 targets take about `4·K + 4` bytes per scored token
with `u16` indices, or `5·K + 4` with `u16+u8`, before zstd; the high byte compresses to a small fraction of a byte per entry for typical
token IDs. Unscored context costs only its input bytes: a 60K-token conversation trained
on a 1K-token answer stores 1K target rows.

### Object storage

Each block is one ranged read (`StorageClient.read` with a byte range), independent of
the MSC cache configuration; a file's index is one small read. Reading starts at the
block a resume needs, without fetching earlier ones. `--logits-load-msc-prefetch-depth`
sets how many blocks each writer stream reads and decodes ahead, and
`--logits-load-decode-threads` how many decode concurrently. Only one rank per pipeline stage lists the cache, and only
when the student reaches the known frontier.

## Student corpus manifest

The manifest is the student's materialized consumption schedule, built on world rank 0
from the corpus indexes and the user-authored blend configuration, then shared with
other ranks. It selects/repeats rows for the student budget, orders blocks and assigns
them to lanes. It does not change the source dump or add weights to its files.

The current implementation keeps the full manifest in memory and regenerates it on
resume. It writes no `manifest.json`. The student checkpoint stores only its hash in
`args.kd_corpus_manifest_hash`, alongside the normal consumed-sample cursor. Comparing
this digest avoids checkpointing the potentially large old schedule solely to compare
it field by field. A mismatch reports a changed plan without identifying each changed
field.

### Manifest fields

| Key | Meaning and purpose |
| --- | --- |
| `manifest_version` | Currently `1`. Identifies this internal plan representation; the reader does not currently enforce the value. |
| `config` | Copy of the resolved blend configuration; field meanings are below. Megatron uses its corpus root to locate files. |
| `budget` | Requested student sample count before rounding up to whole lanes. Descriptive once the schedule has been built. |
| `lanes` | Number of streams, `L`. Global sample position `p` maps to lane `p % L` and offset `p // L`, independently of student DP. |
| `lane_samples` | Equal number of samples assigned to each lane. Total planned capacity is `lanes * lane_samples`, at least the requested budget. |
| `contract` | Shared tokenizer, attention mode and prefix stability. Per-file normalization size and writer layout are excluded. EOD ID is retained only for EOD-reset attention. |
| `files` | Array resolving scheduled file IDs to source folders, writer/range names and planned file properties. |
| `sources` | Per-source reporting and provenance summaries. Not needed to execute an already built schedule. |
| `segments` | Integer array of schedule pieces `(lane, file_id, block, first_row, end_row)`. These pieces select sample rows; they are unrelated to the attention `segments` column stored inside Arrow. |
| `hash` | SHA-256 over the canonical shared contract, file table, lane count, lane length and schedule-piece bytes. Checkpointed to detect a changed plan on resume. |

The hash covers file names, sizes and declared normalization domains, but not file
contents. It is not a corruption checksum. The absolute corpus root and descriptive
reporting fields are excluded, allowing the same file layout and schedule to be read
from a replacement root. Resume still requires equivalent data in that replacement.

### Blend configuration in `config`

| Key | Meaning and purpose |
| --- | --- |
| `corpus` | Root path/URI containing the source directories. |
| `sources` | Mapping of source directory names to positive relative weights in samples, e.g. `{"web_crawl": 3, "math_sft": 1}`. Used to allocate student consumption, without changing source files. |
| `seed` | Student seed for selecting/repeating and ordering source blocks. Independent of the teacher's recorded dataset seed. Defaults to `1234`. |
| `lanes` | Requested stream count, default `64`; duplicated at manifest top level after planning. Ideally a multiple of student DP times microbatch size for exclusive lane ownership. |

### File table in `files[]`

| Key | Meaning and purpose |
| --- | --- |
| `folder` | Source directory name taken directly from a key in `config.sources`, e.g. `math_sft`. Relative to `config.corpus`; matches the teacher's `--kd-corpus-dump` name. |
| `dp_rank` | Writer rank; combines with `start` and `end` to reconstruct the Arrow/index file names. |
| `start` | Inclusive global sample-range start in the original source dump. |
| `end` | Exclusive global sample-range end in the original source dump. |
| `size` | Expected Arrow file length when planning. Checked again when consuming the file. |
| `logprob_vocab_size` | Expected per-file normalization domain. Duplicates the file contract to pin/check that decoding setting in the planned snapshot. |

Within a schedule piece, `file_id` is an index into this table and `block` is an index
into that file's `index.blocks`. `first_row` is inclusive and `end_row` exclusive,
relative to the selected block. The reader therefore fetches a block and slices its
sample rows without reconstructing teacher DP order during consumption.

### Reporting summaries in `sources[]`

| Key | Meaning and purpose |
| --- | --- |
| `name` | Source name, also the corresponding directory/configuration key. |
| `samples` | Realized number of student samples allocated to this source. |
| `available` | Number of real source samples available before any repetition, excluding fillers. |
| `passes` | Number of source passes needed for the allocation, including a final partial pass. |
| `supervised` | Estimated supervised-token count of the selected allocation, using block statistics. |
| `teacher` | Copy of `recorded.teacher`, or null, retained for source provenance. |
| `share` | Realized sample fraction of the entire plan. |
| `supervised_share` | Estimated supervised-token fraction of the entire plan. |

Requested weights are read from `config.sources` for logging; no additional `weight`
copy is stored in each source summary. These summaries can be reconstructed or omitted
without changing the already planned sample sequence.

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
re-padding, the Arrow file and index layout, metadata contract, tokenizer fingerprint,
golden fixtures), exact input/target pairing, reshuffling and its checkpointed extent,
DP/MBS changes, ramp-up, prefix shortening, CP token maps and per-CP-rank target shards,
publication and holes from parallel jobs, the collective replay frontier, one ranged read
per block (and input-only prefixes), frozen-teacher range resume and cleanup,
mismatched-settings detection, and differentiable sparse KL:

```sh
uv run --no-project --with torch --with numpy --with pyarrow --with pytest \
  python -m pytest -q -o addopts='' tests/offline_kd_cpu
```

GPU training, distributed collectives, and MSC integration still require validation.
When allocation is available, compare a deterministic V2/V3 unpacked run, exercise
TP/CP/PP (including virtual stages), test packed CP compatibility, run two parallel
frozen teacher jobs with a trailing sequential student, and interrupt/resume across range
boundaries.

## Implementation map

The new modules in `megatron/training/distillation/` isolate the format, publication,
replay and token mappings, saver, runtime hooks, and paired loss (`v3_*.py`). The
producer-neutral parts (target encodings, tokenizer fingerprint, metadata validation,
the Arrow file and index layout, and the corpus blend planner) live in `format/`, which
imports nothing from Megatron. Corpus dumps and blends add `corpus_runtime.py` and
`CorpusReader` in `v3_replay.py`.
Small hooks in `arguments.py`, `training.py`, `checkpointing.py`, `pretrain_gpt.py`, and
`pretrain_hybrid.py` select the V3 path. V3 reuses existing sparse top-K extraction,
student output capture, the shared LM/sparse-KD loss and reporting, and the persistent
asynchronous writer queue.
