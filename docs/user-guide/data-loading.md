<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# Data Loading at Scale

This guide covers how Megatron's data pipeline works and how to configure it for efficient training at 256 nodes and beyond. At this scale, the primary bottlenecks are **index building** and **barrier synchronization** -- not raw data bandwidth.

## Observing packed schedules and dynamic context parallelism

Add `--log-dcp-stats` to a training run using `--sequence-packing-scheduler`
(including `--dynamic-context-parallel`). It defaults to off. The final packed
schedule is observed **after VPP alignment**, without changing packing or
scheduling decisions. At the existing `--log-interval`, stdout, TensorBoard and
W&B emit the metrics below through their existing rank/writer ownership. No new
monitoring dependency is required. The first-iteration stdout diagnostic retains
the current window until the regular interval boundary.

For microbatch $m$, let $S_{m,r}$ be the IDs assigned to DP×CP rank $r$,
$L_i$ the dataset's **padded** sequence length, and
$c_{m,i}=|\{r:i\in S_{m,r}\}|$ its actual CP size in that microbatch. Then

$$
W_{m,r}=\sum_{i\in S_{m,r}}\frac{L_i^2}{c_{m,i}},\qquad
E=\frac{\sum_m\sum_r W_{m,r}}{R\sum_m\max_r W_{m,r}},
$$

where $R$ is the number of DP×CP ranks, excluding TP and PP replicas. For
example, $W=[8,8,4,4]$ gives $E=24/(4\times8)=0.75$. Multiple packed samples
contribute their **individual squares**, not the square of their total length.
Taking the maximum before summing over microbatches preserves alternating rank
hotspots. The sum of rank work in each legal microbatch equals the sum of $L_i^2$
over its unique logical IDs.

- `dcp/estimated_attention_balance_efficiency` reports $E$, closer to 1 for
  more balanced estimated work. Empty or zero-work windows omit this undefined
  ratio rather than reporting 100%.
- `dcp/cp_size_<c>_sample_fraction` reports the fraction of logical samples
  assigned CP size $c$. Each ID is counted once per microbatch, including when
  replicated across CP ranks. Equal-length samples with different IDs remain
  distinct. Only observed CP sizes are emitted.

Logging windows accumulate the efficiency numerator and denominator and CP
counts before division. TP/PP copies of the global schedule are never added
together. Reruns reuse one pending summary; only a successful training step
commits it. Skipped steps, exceptions, early exits and evaluation do not
contribute. A restart begins a fresh logging window.

Example output (illustrative):

```text
iteration 100/1000 | dcp/estimated_attention_balance_efficiency: 0.750000 | dcp/cp_size_1_sample_fraction: 0.666667 | dcp/cp_size_2_sample_fraction: 0.333333 | elapsed time per iteration (ms): ...
```

`L²/CP` is the scheduler's **attention-work estimate**, not measured GPU
utilization or elapsed-time efficiency. It excludes communication, MLP work,
pipeline bubbles, and some additional THD padding applied after scheduling.
Interpret it alongside existing step timing and effective (unmasked) tokens/s.
Enabling it performs CPU statistics and extends the existing TP scalar metadata
broadcast using float64 for squared-length precision. There are no additional
DP/CP collectives; this still has observation and communication overhead.
The complete microbatch-by-rank work matrix is available from
`compute_schedule_stats` in `data_schedule_utils.py`, and on TP source ranks via
the optional `schedule_stats` output dict of `wrap_data_iterator`. The existing
four-element return value is preserved. No metrics are computed when the dict
is `None`.

For a small dense BF16/THD DCP run (requires a backend supporting the selected
GPU and THD padding metadata):

```bash
python - <<'PY'
from pathlib import Path
# Current MockVarlenDataset shifts away one token. These CSV entries yield
# aligned valid lengths 1024, 256, 128, 512 for the Ampere FA2 path.
Path('/tmp/dcp-lengths.csv').write_text(('1025\n257\n129\n129\n513\n257\n129\n129\n') * 24)
PY
COMMON=(
  --use-mcore-models --num-layers 4 --hidden-size 128 --ffn-hidden-size 512
  --num-attention-heads 4 --seq-length 1024 --max-position-embeddings 1024
  --transformer-impl transformer_engine --attention-backend flash --bf16
  --attention-dropout 0 --hidden-dropout 0 --no-gradient-accumulation-fusion
  --micro-batch-size 1 --global-batch-size 8 --calculate-per-token-loss
  --context-parallel-size 2 --max-seqlen-per-dp-cp-rank 512
  --use-varlen-dataset --mock-data --tokenizer-type NullTokenizer --vocab-size 256
  --null-tokenizer-pad-id 0 --num-workers 0 --dataloader-type single
  --varlen-mock-dataset-config-json '{"mode":"file","format":"thd","path":"/tmp/dcp-lengths.csv"}'
  --train-iters 40 --lr-decay-iters 100 --lr 0.00015 --min-lr 0.00001
  --lr-warmup-iters 0 --seed 1234 --log-interval 1 --eval-iters 0
)
python -m torch.distributed.run --standalone --nproc-per-node 2 pretrain_gpt.py \
  "${COMMON[@]}" --dynamic-context-parallel --log-dcp-stats
```

For an observation-overhead comparison, run the same command with the flag
off/on from the same checkpoint, data and seed, exclude warmup, alternate the
order and repeat measurements. Verify schedules, microbatch counts, losses,
gradients and parameter updates before interpreting timing. For fixed CP use
`--sequence-packing-scheduler dp_balanced` in place of
`--dynamic-context-parallel`. The existing
`examples/dynamic_context_parallel/benchmark_dcp.sh` also compares fixed CP and
DCP with steady-state training timers. These are small-model acceptance checks,
not production-scale convergence or scaling evidence.

Four-GPU topology variants of the command above are TP2×CP2 (add
`--tensor-model-parallel-size 2 --sequence-parallel`) and PP2×CP2/VPP2 (add
`--pipeline-model-parallel-size 2 --num-layers-per-virtual-pipeline-stage 1
--microbatch-group-size-per-virtual-pipeline-stage 2`). The latter has **two DP×CP ranks**,
not four. Confirm that the workload actually triggers VPP alignment; requesting
VPP alone does not prove that microbatches were split.

### Local A6000 acceptance (2026-10-22)

The observation change was validated on dev `a5aa637fa` plus an independent eager
THD padding-metadata prerequisite, tracked by [#7417](https://github.com/NVIDIA/Megatron-LM/pull/7417).
That prerequisite is **outside the metrics change**. Unmodified dev marks every
packed batch as internally padded, disabling FA2 on Ampere; TE's fused BF16 THD
backend requires SM90+. The acceptance dataset above uses aligned valid lengths,
so the accurate eager padding flag permits FA2. Internally padded Ampere THD
workloads are not qualified by these runs.

Environment: RTX A6000/SM86, driver 580.178.04, Python 3.12.4, PyTorch
2.11.0+cu130, CUDA 13.0, Transformer Engine 2.20.2, FlashAttention 2.8.3,
cuDNN 9.19.0, NCCL 2.28.9. The dense model has four layers, hidden size 128,
four attention heads, FFN size 512 and vocabulary size 256. Dropout is disabled.
All runs execute real BF16 forward/backward and Adam updates with synthetic,
length-controlled tokens; they do not establish production-model convergence.

Each correctness pair restored the same topology-specific checkpoint and RNG
state, with seed 1234 and deterministic execution. Five subsequent updates
matched bitwise in final schedules, microbatch counts, losses, gradients and
updated parameters. Evaluation ran between training steps. Every enabled
training step produced exactly one stdout record and one TensorBoard scalar per
metric; TP peers received the same summary. W&B routing is covered by the logger
unit test, without an external W&B service in these runs.

| GPUs | TP / PP / CP / VPP | Microbatches before → after alignment | CP1 / CP2 logical samples | Estimated balance |
|---|---|---|---|---|
| 2 | 1 / 1 / 2 / 1 | 3 → 3 | 7 / 1 | 0.92 |
| 4 | 2 / 1 / 2 / 1 | 3 → 3 | 7 / 1 | 0.92 |
| 4 | 1 / 2 / 2 / 2 | 3 → 4 | 3 / 5 | 0.92 |

The standalone metrics branch passed 84 focused tests per rank on four GPUs;
the integration with the padding prerequisite passed 49 scheduling tests per
rank. Existing tests requiring other topology assumptions and four unrelated
model-builder cases with an optional-import failure were excluded. Black,
isort, Ruff and the core formatter/Pylint checks passed. The optional mypy stage
reported existing baseline diagnostics; no new diagnostic messages remained.

Timing reused the existing training interval timer, with TensorBoard and
`--log-interval 1`. Each arm restored the same checkpoint, ran 100 updates,
discarded ten warmup updates and retained 90 steady updates, repeated three
times in rotated order. Only the measurement processes were bound to fixed CPU
sets. Each step trained 2560 effective tokens; these aligned workloads have
equal padded and valid lengths. Correctness tensor hashing was disabled.

| Arm | Mean step ± between-run std (ms) | Median of run medians (ms) | Mean effective tokens/s |
|---|---|---|---|
| Fixed CP2 | 145.38 ± 3.26 | 144.65 | 17,615 |
| DCP, metrics off | 113.55 ± 4.18 | 109.95 | 22,566 |
| DCP, metrics on | 116.22 ± 8.27 | 109.20 | 22,100 |

The on/off mean difference was **+2.35%**; paired differences ranged from
−1.35% to +6.34%, while the central step times were similar. This shared-host
measurement does not resolve small observation costs and is not evidence of
zero overhead. An additional CPU diagnostic measured schedule statistics at
45–48 µs/step and window updates at about 12 µs/step. On the emitting rank, the
whole existing `training_log` call averaged 1.565 ms off versus 1.604 ms on;
Python garbage-collection events did not explain the multi-millisecond tail variation.
These diagnostics are separate from the primary timing results.

[Validation records and reproduction scripts](https://github.com/Pinoeer-kingxi/Megatron-LM/tree/evidence/dcp-stats-a6000/validation/dcp-stats-a6000)
include source hashes, checkpoint hashes, exact commands, raw logs, TensorBoard
events, per-rank audits, initial unbound timing results, CPU diagnostics and the
standalone compatibility patch. The metrics PR does not contain that patch or
a new benchmark framework.

## How Data Loading Works

Understanding the architecture helps explain why specific flags matter.

Megatron builds three index arrays for each dataset: a **document index** (shuffled document order), a **sample index** (mapping samples to document offsets), and a **shuffle index** (final sample permutation). This happens once during initialization:

1. **Rank 0** builds all three indices and writes them to a cache directory as `.npy` files.
2. All ranks synchronize at a `torch.distributed.barrier()`.
3. **All other ranks** load the cached indices via memory-mapped reads (`numpy.load(mmap_mode='r')`).

After initialization, data access is **read-only and lock-free**. Each data-parallel rank consumes a disjoint subset of samples, and no cross-rank coordination is needed during training because all ranks derive the same deterministic permutation from a shared random seed.

## The Problem at 256+ Nodes

Three things break down at large node counts:

1. **Barrier synchronization**: All ranks block while rank 0 builds indices. On a 512-node job, this means 4,095 GPUs sit idle.
2. **Simultaneous memory-mapping**: All ranks `mmap` three large `.npy` files at once after the barrier, causing a burst of page faults and I/O.

## Baseline: Establish Maximum Achievable Performance

Before tuning data loading, establish a performance ceiling by running with `--mock-data`. This bypasses the data pipeline entirely and shows the maximum throughput your configuration can achieve without any dataloader overhead. The gap between `--mock-data` performance and real-data performance tells you exactly how much time the dataloader is costing you.

## Recommended Configuration

### Step 1: Consolidate dataset files

A common issue at scale is having datasets split across many small file prefixes. Thousands of 100 MB files perform significantly worse than tens of 10 GB+ files, both for building dataset caches and for runtime file access.

Use the merge tool to consolidate datasets stored as many small prefixes in one directory:

```bash
python tools/merge_datasets.py \
    --input /path/to/input-directory \
    --output-prefix /path/to/output/merged
```

**Target at least 10 GB per file.** This reduces the number of file descriptors, metadata lookups, and index-building work at initialization.

### Step 2: Pre-build the dataset cache

Build the GPT dataset cache as a separate step before training. This avoids the usual "rank 0 builds, everyone else waits" startup path and is the recommended workflow for large jobs:

```bash
python tools/prepare_cache.py \
    --data-path <your-data-config> \
    --split 99,1,0 \
    --data-cache-path /path/to/cache \
    --global-batch-size <global-batch-size> \
    --seq-length <seq-length> \
    ...
```

If your later training job does not set `--global-batch-size`, or you are preparing the cache on a machine that does not match the future training topology, also pass:

```bash
--prepare-cache-world-size <future-world-size>
```

This keeps the prepared cache aligned with the sample counts expected by training.

> **Unsupported configurations:** `tools/prepare_cache.py` does not support `--mock-data`, `--sft`, `--fim-data`, or `--step-batch-size-schedule`. Using any of these will cause the script to exit with an error.

### Step 3: Optionally pre-build per-dataset metadata

When blending many datasets, generate the `--per-dataset-sequences-path` JSON ahead of time to avoid one metadata read per file prefix at startup:

```bash
python tools/build_sequences_per_dataset.py \
    --data-path <your-data-config> \
    --per-dataset-sequences-path sequences.json
```

### Step 4: Launch training with optimized data loading

Once the cache is ready, enable the fast-path flags:

```bash
torchrun --nproc_per_node=8 --nnodes=512 ... pretrain_gpt.py \
    --dataloader-fast-cache-load \
    --dataloader-defer-npy-index-mmap \
    --per-dataset-sequences-path sequences.json \
    --data-cache-path /path/to/cache \
    --num-workers 2 \
    ...
```

### Flag reference

| Flag | Default | Recommendation | What it does |
|------|---------|----------------|-------------|
| `--dataloader-fast-cache-load` | off | **On** | Skips the rank-0 barrier by assuming the cache already exists. All ranks build their dataset views in parallel. This is the single biggest win at scale. |
| `--dataloader-defer-npy-index-mmap` | off | **On** | Defers memory-mapping of `.npy` index files until first access. When combined with `--num-workers > 0`, index loading is overlapped with the training iteration rather than blocking startup. |
| `--per-dataset-sequences-path` | None | **Set when blending many datasets** | Points to a JSON file mapping each dataset path to its `(sequence_count, document_count)`. Replaces per-file metadata reads with a single JSON lookup. Generate with `tools/build_sequences_per_dataset.py`. |
| `--data-cache-path` | None | **Set** | Directory where index `.npy` files are cached. Must be on shared storage for multi-node jobs so all ranks can read it. |
| `--num-workers` | 2 | **Keep as small as necessary** | Number of DataLoader worker processes. The goal is to satisfy: *time to process a batch > time to prepare a batch*. This hides dataloader work behind the training step. Increasing beyond what's needed wastes CPU and memory. |
| `--no-mmap-bin-files` | mmap on | **Test both** | Memory-mapping `.bin` files leverages the OS page cache, but the optimal setting is filesystem-dependent. Some large-scale production configurations disable mmap. Test with and without to determine what works best for your storage. |

### Object storage (S3 / Multi-Storage Client)

When data lives on S3 or MSC rather than a POSIX filesystem:

- **Index files** (`.idx`) are cached locally under `object_storage_cache_path`.
- **Binary data files** (`.bin`) are streamed on-demand in 256 MB chunks, avoiding the need to download entire files.
- Set `--no-mmap-bin-files` since memory-mapping doesn't apply to object storage.
- Ensure the index-cache path is visible wherever the later dataset construction will run.

## Scaling Characteristics

| Aspect | Behavior | Why it works |
|--------|----------|-------------|
| **Cross-rank contention** | None after init | All index files are read-only; `numpy.memmap` uses OS page cache with no locking |
| **Sampling determinism** | All ranks produce the same permutation | Shared `numpy.random.RandomState(seed)` with epoch-based seed variation |
| **Data-parallel sharding** | Each DP rank gets a disjoint subset of samples | No overlap during training; assignment happens in the sampler rather than via extra dataset coordination |
| **Index broadcast** | Via shared filesystem, not collectives | Rank 0 writes `.npy` files; other ranks read them. No explicit `torch.distributed.broadcast` |

## Troubleshooting

**Symptom: Training hangs at startup for minutes**
- Likely cause: Rank 0 is building indices while all other ranks wait at the barrier.
- Fix: Pre-build the cache with `tools/prepare_cache.py` and enable `--dataloader-fast-cache-load`.

**Symptom: Spike in I/O at training start, then normal**
- Likely cause: All ranks simultaneously memory-mapping index files after the barrier.
- Fix: Enable `--dataloader-defer-npy-index-mmap` to overlap index loading with training.

**Symptom: Slow data loading during training (not just startup)**
- Run with `--mock-data` to confirm the dataloader is the bottleneck.
- If startup, not steady-state throughput, is the main issue, try `--dataloader-defer-npy-index-mmap`.
- If you are blending many dataset prefixes, try `--per-dataset-sequences-path`.
- Test with `--no-mmap-bin-files` -- the optimal setting depends on your filesystem.

## Related Resources

- [PR #2445](https://github.com/NVIDIA/Megatron-LM/pull/2445): Original implementation of fast cache load, deferred mmap, and per-dataset sequences optimizations.
- [PR #4080](https://github.com/NVIDIA/Megatron-LM/pull/4080): Adds `tools/prepare_cache.py` for offline GPT dataset cache preparation.
- [`tools/prepare_cache.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/tools/prepare_cache.py): Pre-build GPT dataset caches ahead of training.
- [`tools/merge_datasets.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/tools/merge_datasets.py): Merge multiple small dataset files into larger ones.
- [`tools/build_sequences_per_dataset.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/tools/build_sequences_per_dataset.py): Generate the `--per-dataset-sequences-path` JSON file.
