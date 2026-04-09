<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# Data Loading at Scale

This guide covers how Megatron's data pipeline works and how to configure it for efficient training at 256 nodes and beyond. At this scale, the primary bottlenecks are **index building**, **filesystem metadata operations**, and **barrier synchronization** -- not raw data bandwidth.

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
2. **Filesystem metadata storms**: When blending many datasets, thousands of simultaneous `open()` and `stat()` calls from all ranks can overwhelm NFS/Lustre metadata servers.
3. **Simultaneous memory-mapping**: All ranks `mmap` three large `.npy` files at once after the barrier, causing a burst of page faults and I/O.

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

**Symptom: Metadata server errors or slow `open()` calls**
- Likely cause: Thousands of ranks simultaneously opening per-dataset files for metadata.
- Fix: Use `--per-dataset-sequences-path` to consolidate metadata into a single JSON file.

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
