# Data Pipeline

Packed training and evaluation support `--pad-packed-seq-alignment N` (a CP-local
token multiple before SP) or `max` (pad to `--max-seqlen-per-dp-cp-rank`). MXFP8
automatically requires a multiple of 32 tokens after SP; the effective alignment
combines that requirement with the requested multiple. Padding is added before
CP layout construction, with masked dummy sequences and consistent THD metadata.
The scheduler reserves an aligned capacity; padding never truncates valid tokens.
This eager path does not require CUDA graphs.

When unpacking SFT data, prompt-only sub-sequences are retained so packing can
still fill all DP ranks; their zero loss masks exclude them from the loss. A DP
rank can contribute no samples and still receive scheduled work. A global batch
with no sequences is rejected on all ranks.

Dynamic CP does not currently support `--deterministic-mode`, CUDA graphs, or
router z-loss. Configuration validation rejects these combinations before training.
Static sequence packing without dynamic CP retains its existing deterministic-mode
support. Dynamic CP for Hybrid models follows in PR #7048.

Batch helpers use `dynamic_cp` and `dynamic_cp_group_func`. The old `is_hybrid_cp`
and `hybrid_cp_group_func` keyword arguments remain accepted with a deprecation
warning for downstream callers. Positional arguments and the legacy
`hybrid_cp_group` batch metadata are unchanged during this compatibility window.
The deprecated keywords also remain visible to signature-based feature detection
used by downstream consumers such as MBridge.

The `gpt_packed_train_eval` functional test runs the real GPT entrypoint on eight
GPUs with TP2/PP2, synthetic variable-length data, and static or dynamic packing.
It checks finite training/evaluation losses and gradient norms, a completed
checkpoint, and CP1/CP2 coverage in both training and evaluation. It does not
claim deterministic or golden-loss equivalence. Run it from the repository root:

```bash
python tests/functional_tests/test_cases/gpt/gpt_packed_train_eval/run_test.py \
  --packing-mode dynamic --output-path /path/to/a/new/output-directory
```

## Data pre-processing

Data preprocessing is built around the following classes:

1. `IndexedDatasetBuilder`
2. `IndexedDataset`

At the moment, an end-to-end data preprocessing implementation is left to the user. See the class docstring(s) for more details.

### IndexedDatasetBuilder

The `IndexedDatasetBuilder` is capable of building and merging `IndexedDataset` instances.

### IndexedDataset

The `IndexedDataset` class is the lowest-level data interface in Megatron Core. Internally, an `IndexedDataset` instance references two binaries: the data file (`.bin`) contains document/sequence data and the index file (`.idx`) contains document/sequence metadata.

The index file stores dataset-level metadata first:
- The index header, for backward compatibility
- The index version, for backward compatibility
- A numeric code corresponding to the data type used to write data to the data file
- The number of sequences in the dataset
- The number of documents in the dataset

The index file stores document-level and sequence-level metadata second:
- In order, the number of elements per sequence
- In order, the byte offset (pointer) per sequence
- In order, the consecutive sequence index range `[...)` per document
- In order, the mode per sequence (in the multimodal case)

## Data loading: construction

Building the data loaders is a distributed-aware process built around the following classes:

1. `BlendedMegatronDatasetConfig`
2. `BlendedMegatronDatasetBuilder`
3. `IndexedDataset`
3. `MegatronDataset`
4. `BlendedDataset`

See the class docstrings for more details.

### BlendedMegatronDatasetConfig (extendable)

The `BlendedMegatronDatasetConfig` class parameterizes the `BlendedMegatronDatasetBuilder` and in turn the `MegatronDataset` and `BlendedDataset`.

Different training/inference regimes will require different extensions e.g. the `GPTDatasetConfig`

### BlendedMegatronDatasetBuilder

The `BlendedMegatronDatasetBuilder` class builds the highest-level data interfaces in Megatron Core.

**NB:** All ranks should attempt to build the dataset via the `BlendedMegatronDatasetBuilder` or the program will hang. Which ranks follow through on their attempts can be controlled via the `BlendedMegatronDatasetConfig`.

### IndexedDataset

The `IndexedDataset` class is the lowest-level data interface in Megatron Core.

The `IndexedDataset` should already exist on disk before attempting to build any of the high-level data interfaces.


### MegatronDataset (extendable)

The `MegatronDataset` abstract class is a high-level data interface in Megatron Core. It is an abstraction built upon the `IndexedDataset`.

Different training/inference regimes will require different extensions e.g. the `GPTDataset`

### BlendedDataset

The `BlendedDataset` class is a high-level data interface in Megatron Core. It is an abstraction built upon the `MegatronDataset`.

The `BlendedDataset` is only necessary when a blend multiple data distributions, i.e. multiple `MegatronDataset` instances, should contribute to a certain dataset split. The blend can be controlled via the `BlendedMegatronDatasetConfig`.

## Deterministic training subsets

GPT pretraining can select a fixed subset of each training source before building
sample indexes. A per-split blend accepts flattened **weight, prefix, probability**
triples for `train`; validation and test continue to use ordinary prefixes or
weight/prefix pairs. For example:

```json
{
  "train": ["0.7", "/data/source_a", "0.5", "0.3", "/data/source_b", "0.25"],
  "valid": ["/data/validation"],
  "test": null
}
```

Add these options to a GPT or Hybrid pretraining command, along with its model,
tokenizer, batch and training-duration settings:

```bash
--per-split-data-args-path /path/to/blend.json \
--train-data-down-sampling-seed 1234 \
--data-cache-path /path/to/cache
```

For each source containing `N` eligible indexed sequences, probability `p` retains
`ceil(p * N)` entries from a NumPy `RandomState` permutation. Probabilities must be
finite and in `(0, 1]`. The subset seed is independent of `--seed`, which controls
sample shuffling and other training randomness. Use the same subset seed and
unchanged source indexes to obtain nested subsets as probabilities increase.
Probability `1` keeps the legacy order and cache identity. Unweighted blends and
weight/prefix pairs keep their existing behavior.

Selection operates on **indexed sequence IDs**. It selects whole documents only
when preprocessing stored exactly one sequence per document; it does not remove
duplicate text. Blend weights still determine sampling frequency, while
probabilities determine which source records are eligible. The selected pool's
token count is separate from the training horizon: a longer run can revisit
selected records.

Training subsets require a per-split training blend. They are not supported with
`--data-path`, validation/test triples, SFT/variable-length datasets, or cached-logit
distillation. An optional
`--train-data-down-sampling-prob-override` applies one probability to every training
source; using it replaces the per-source probabilities, including calibrated ones.

### Cache preparation

Use the same blend, subset seed, tokenizer, sequence length and requested sample
counts for `tools/prepare_cache.py` and training. Probability and subset seed are
part of the dataset cache identity whenever downsampling is active; changing them
requires matching caches. `--dataloader-fast-cache-load` consumes prepared caches
and does not construct a new subset from missing source-index arrays. Prepare
caches before enabling it, and retain the corresponding per-source sequence counts
and dtype settings used by the fast-cache workflow.
`tools/build_sequences_per_dataset.py` accepts the training triples and records
the full source sequence counts; subset probabilities are applied later by GPTDataset.

The feature is wired through GPT, Hybrid and cache-preparation configuration.
GPT fill-in-the-middle datasets also retain the selected source subset.

### Prepare nested pools by token budget

[`tools/prepare_data_subsets.py`](../../../tools/prepare_data_subsets.py) converts a
training-only weighted blend into nested subsets using the token lengths recorded
in native Megatron indexes. It preserves source weights and paths, supports native
token storage dtypes, and does not copy token binaries. Start with a JSON such as:

```json
{
  "train": ["0.7", "/data/source_a", "0.3", "/data/source_b"],
  "valid": null,
  "test": null
}
```

Run from the repository root, using an output directory that does not exist:

```bash
python -m tools.prepare_data_subsets \
  --input-blend /path/to/sources.json \
  --output-dir /path/to/subsets \
  --token-budgets 1000000000 500000000 250000000 \
  --seed 1234
```

Each source receives a token quota proportional to its blend weight. The tool
selects the shortest prefix of the same seeded sequence permutation that meets
that quota. Whole-sequence selection can exceed the requested token budget; the
outputs record both requested and actual counts. Source quotas above available
capacity are rejected. The tool refuses to overwrite an existing output directory.

The outputs are `subset_<token_budget>.json` and `summary.json`. The summary
records source sizes, quotas, selected sequence/token counts and the NumPy version.
Relative source prefixes are interpreted from the working directory and remain
unchanged in the generated blends.

Use a generated training blend, add your existing validation/test paths if needed,
and pass the recorded subset seed to both cache preparation and training. Keep the
original source indexes and a matching NumPy version for reproducible selection.
The helper uses memory proportional to the largest source's sequence count.

## Data loading: implementation

### GPTDataset

The `GPTDataset` is parameterized by the following variables: the underlying `IndexedDataset` instance `indexed_dataset`, the split indices `indexed_indices` (the congituous subset of document or sequence indices used for training, validation, and testing), the number of samples `N`, the sequence length `S`, and the random seed `R`.

The `GPTDataset` creates three index mappings to facilitate lookup: (1) the document index, (2) the sample index, and (3) the shuffle index.

1. The document index _Do_idx_ is a 1-D array mapping from _i_ to document index of length `E * |indexed_indices|` where `E` corresponds to the minimum number of epochs such that `E * |indexed_indices| >= N`. The document index is shuffled according to `R`.

    ```
    Given:

    N = 15
    indexed_indices = [5, 6, 7, 8, 9]
    E = 3

    Then, for example:

    Do_idx = [8, 8, 9, 6, 7, 5, 8, 5, 6, 6, 5, 9, 7, 7, 9]
    ```

2. The sample index _Sa_idx_ is a 2-D array mapping from _j_ to pairs of (_i_, _Do_idx_[ _i_ ] offset) of shape `[N + 1, 2]`. The rows _j_ and _j_ + 1 serve as the left and right bounds for the _j_-th sample. 

    ```
    Given:

    S = 1024

    Then, for example:

    Sa_idx[0] = (0, 0)
    Sa_idx[1] = (0, 1024)       => Do_idx[0] has length greater than S
    Sa_idx[2] = (1, 512)        => Do_idx[0] has length 1536
    Sa_idx[3] = (2, 0)          => Do_idx[1] has length 1536
    Sa_idx[4] = (5, 300)        => Do_idx[2:5] are shorter documents relative to Do_idx[0:2]
    Sa_idx[5] = (6, 24)         => Do_idx[5] has length 1300
    ```

3. The shuffle index _Sh_idx_ is a 1-D array mapping from _k_ to _j_ of length `N`. The shuffle index is shuffled according to `R`.

    ```
    Given

    N = 10

    Then, for example:

    Sh_idx = [4, 0, 2, 6, 1, 9, 5, 8, 7, 3]
    ```

To query the `GPTDataset` for the _k_-th sample we do the following

-  Use the shuffle index to get the index _j_ into the sample index.

    ```
    j = Sh_idx[k]
    ```
- Use the sample index to get the left and right sample-bounding indices into the document index and the starting token offset for each document.

    ```
    i, offset = Sa_idx[j]
    i_next, offset_next = Sa_idx[j + 1]
    ```
- Use the document index to retrieve `S` tokens from consecutive (in the document index) documents.

    ```
    sample = []
    sample += indexed_dataset[Do_idx[i]][offset:]
    if i != i_next:
        sample += indexed_dataset[Do_idx[i + 1:i_next]]
    sample += indexed_dataset[Do_idx[i_next]][:offset_next]
    ```

To save time during initialization, each index is built/cached sequentially on one process rank and subsequently loaded in parallel on other process ranks. The cached indices are unique to a hash generated in the `MegatronDataset.__init__` function.

### BlendedDataset

The `BlendedDataset` is parameterized by the following variables: the underlying `MegatronDataset` instances `D`, the weights `W` (one per dataset), and the size `S`. The `BlendedDataset` will draw samples from contributing datasets in proportion to the weights until achieving a composite dataset of the desired size. During each sampling step, we draw a single sample from the dataset which has the greatest sampling error.

The `BlendedDataset` creates two "blending" indices to facilitate lookup: (1) the dataset index and (2) the dataset sample index.

1. The dataset index _Da_idx_ is a 1-D array mapping from _i_ to dataset index of length `S`.

    ```
    Given

    D = [d0, d1, d2]
    W = [1/2, 1/4, 1/4]
    S = 4

    Then, for example:

    Da_idx = [0, 1, 2, 0]

    ```

2. The dataset sample index _Sa_idx_ is a 1-D mapping from _i_ to the sample index for dataset _Da_idx[i]_ of length `S`.

    ```
    Given

    Da_idx = [0, 1, 2, 0]

    Then, for example:

    Sa_idx = [0, 0, 0, 1]
    ```

To query the `BlendedDataset` for the _k_-th sample we do the following

- Use the dataset index to retrieve the corresponding dataset from `D` and the dataset sample index to retrieve the corresponding sample from that dataset.

    ```
    sample = D[Da_idx[k]][Sa_idx[k]]
    ```

To save time during initialization, each index is built/cached sequentially on one process rank and subsequently loaded in parallel on other process ranks. The cached indices are unique to a hash generated in the `BlendedDataset.__init__` function.

## Offline cache preparation

For GPT-style training, the dataset caches described above can be prepared ahead of time with `tools/prepare_cache.py` instead of waiting for rank 0 to build them during training startup.

The script reuses the normal dataset construction path used by `pretrain_gpt.py` and `pretrain_mamba.py`, including `GPTDataset`, `BlendedDataset`, and `BlendedMegatronDatasetBuilder`. It accepts the usual dataset arguments, supports blends and per-split dataset definitions, and requires `--data-cache-path` so the generated cache can later be reused by training.

This is especially useful for large blends or many file prefixes, where building the document, sample, and shuffle indices can take several minutes and leave all GPUs idle while rank 0 performs CPU-only work.

If the later training job does not specify `--global-batch-size` (which is needed to determine the dataset size and splits), you should specify `--prepare-cache-world-size` to explicitly set the world size used during cache preparation.

`tools/prepare_cache.py` does not support `--mock-data`, `--sft`, `--fim-data`, or `--step-batch-size-schedule`.

## Packing Scheduler

The packing scheduler re-schedules variable-length sequences across DP×CP ranks to improve GPU utilization. It is built around the following modules:

### `data_schedule`

This module contains the high-level scheduling logic and entry points:

- **`BasePackingScheduler`**: Abstract base class for packing schedulers. Defines the interface for `get_groups_and_subsamples()` (scheduling algorithm) and `run()` (full scheduling pipeline including fetch, schedule, reroute, pack, broadcast, and VPP handling).

- **`DpBalancedScheduler`**: A concrete scheduler that packs sequences in their original order until reaching the max sequence length limit per DPxCP rank. Supports aligning the number of microbatches to DP size and VPP stage multiples.

- **`DefaultDynamicCPScheduler`**: Selects a valid runtime CP group for each packed microbatch while balancing sequences across the DP×CP domain.

- **`wrap_data_iterator()`**: Top-level entry point that wraps an existing `data_iterator`. It creates the appropriate scheduler, runs the scheduling pipeline, broadcast metadata and new num_microbatches, returns a new data iterator along with the updated number of microbatches and FLOPs statistics.

- **`get_batch_on_this_rank_for_sequence_packing()`**: Fetches and broadcasts a single packed microbatch for the current rank. Handles TP/PP broadcasting, constructs `PackedSeqParams` (with `cu_seqlens`, `max_seqlen`, `qkv_format=thd`), and optionally partitions sequences across CP ranks using Transformer Engine's `thd_get_partitioned_indices`.

Both entry points require a caller-owned `ProcessGroupCollection`. Scheduling uses its
`tp`, `pp`, `dp`, and `dp_cp` groups; batch preparation uses `tp`, `pp`, and `cp`, plus
`tp_cp` for static CP layout conversion with sequence parallelism. Dynamic CP selects
pre-created groups from `dynamic_cp` and `dynamic_tp_cp`, whose keys are runtime CP sizes
(the groups in `dynamic_tp_cp` have `TP size * CP size` ranks). The scheduler only selects
sizes present in `dynamic_cp`; the static `cp` group remains unchanged. The standard GPT
and Hybrid forward steps pass their model's collection. Legacy training entrypoints may
bootstrap a collection with `ProcessGroupCollection.use_mpu_process_groups()` before
calling Core, while independent parallel grids supply their own groups directly.

### `data_schedule_utils.py`

This module contains the utility functions used by the schedulers.

## Fast DataLoader initialization

Especially for large-scale runs, DataLoader initialization can take several minutes, since it involves opening and memory-mapping multiple files and can significantly stress the filesystem. To speed up this process, we have developed the following three optimizations, controlled by configuration flags:

  - `--dataloader-fast-cache-load`: This option assumes that the dataset cache already exists in the specified `--data-cache-path`. When enabled, it speeds up the creation process by removing synchronization points and file check assertions.

  - `--dataloader-defer-npy-index-mmap`: This option also assumes that the dataset cache already exists in the specified `--data-cache-path`. When enabled, it defers the memory mapping of the dataset indexes (.npy files) until their first access. We recommend using this configuration together with `--num-workers` > 0 so that the DataLoader prefetches the next batches of data, thereby hiding the cost of index memory mapping.

  - `--per-dataset-sequences-path`: With this configuration, we specify the JSON file generated by the `tools/build_sequences_per_dataset.py` script. This script generates a single file containing the required metadata from all the specified file prefixes. This configuration is especially useful when dealing with hundreds to thousands of file prefixes, since it requires only a single `open` operation instead of one per file prefix.
