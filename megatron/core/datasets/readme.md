# Data Pipeline

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

## Packing Scheduler

The packing scheduler re-schedules variable-length sequences across DP×CP ranks to improve GPU utilization. It is built around two modules: `data_schedule.py` (high-level logic and entry points) and `data_schedule_utils.py` (utility functions).

### Call Hierarchy

The scheduling pipeline has two phases connected by the data iterator: `wrap_data_iterator` consumes the **original** data iterator, performs global-batch scheduling, and produces a **wrapped** (packed) data iterator; `get_batch_on_this_rank_for_sequence_packing` then consumes this **wrapped** data iterator to fetch individual packed microbatches during training.

```
                          original                              wrapped (packed)
                       data_iterator                             data_iterator
                            │                                        │
                            ▼                                        ▼
               ┌────────────────────────┐               ┌────────────────────────────────────┐
               │  wrap_data_iterator()  │               │ get_batch_on_this_rank_for_        │
Phase 1        │  (once per global      │   ────────►   │       sequence_packing()            │  Phase 2
(scheduling)   │       batch)           │   returns     │  (once per microbatch,              │  (fetching)
               │                        │   wrapped     │   called by training loop)          │
               └───────────┬────────────┘   iterator    └──────────────┬─────────────────────┘
                           │                                           │
                           ▼                                           ▼
          DpBalancedScheduler.run()                   next(wrapped_data_iterator)
          │                                           ├─ get_thd_partitioned_indices()  [TE]
          ├─ get_batch_and_global_seqlens()  [utils]  ├─ broadcast_tensor()             [utils]
          ├─ get_groups_and_subsamples()              └─ PackedSeqParams(...)
          ├─ reroute_samples_to_dcp_ranks()  [utils]
          ├─ build_packed_microbatches()     [utils]
          ├─ broadcast_to_pp_group()         [utils]
          ├─ broadcast_scalars()             [utils]
          └─ create_data_iterator()          [utils]
```

### `data_schedule.py`

#### Entry Points

- **`wrap_data_iterator(original_data_iterator) → wrapped_data_iterator`** — Top-level entry point called once per global batch. Takes the **original** data iterator as input, resolves the scheduler class from `scheduler_map`, instantiates it, and delegates to `scheduler.run()` which consumes all microbatches from the original iterator, re-schedules them, and produces a **wrapped** (packed) data iterator along with the updated `num_microbatches` and FLOPs statistics.

- **`get_batch_on_this_rank_for_sequence_packing(wrapped_data_iterator)`** — Per-microbatch entry point called by the training loop. Takes the **wrapped** data iterator returned by `wrap_data_iterator` as input. Fetches one packed microbatch via `next(wrapped_data_iterator)`, broadcasts batch fields across TP ranks, optionally partitions sequences across CP ranks using Transformer Engine's `thd_get_partitioned_indices`, and constructs `PackedSeqParams` (with `cu_seqlens`, `max_seqlen`, `qkv_format=thd`).

#### Scheduler Classes

- **`BasePackingScheduler`** — Abstract base class. Defines the interface:
  - `get_groups_and_subsamples()` — pure scheduling algorithm (must be overridden).
  - `run()` — full pipeline: fetch → schedule → reroute → pack → broadcast → VPP handling.

- **`DpBalancedScheduler(BasePackingScheduler)`** — Concrete scheduler that packs sequences in their original order until reaching `max_seqlen_per_dp_cp_rank × cp_size`. Aligns the number of microbatches to `dp_size` (and VPP stage multiples when applicable).

### `data_schedule_utils.py`

Utility functions consumed by the schedulers above:

| Function | Role |
|---|---|
| `get_batch_and_global_seqlens()` | Fetch `num_microbatches` batches from the data iterator and all-gather sequence lengths across DP ranks. |
| `reroute_samples_to_dcp_ranks()` | All-to-all communication to transfer sub-samples to their scheduled DP×CP rank. |
| `build_packed_microbatches()` | Concatenate sub-samples within each microbatch group and produce `cu_seqlens`. |
| `broadcast_to_pp_group()` | Broadcast packed samples and metadata from the first/last PP stage to middle stages. |
| `broadcast_scalars()` | Broadcast scalar values (e.g. `num_microbatches`, FLOPs stats) across a process group. |
| `broadcast_tensor()` | Broadcast a single tensor within a process group. |
| `create_data_iterator()` | Wrap packed sample lists into a data iterator; handles VPP stage splitting. |

## Fast DataLoader initialization

Especially for large-scale runs, DataLoader initialization can take several minutes, since it involves opening and memory-mapping multiple files and can significantly stress the filesystem. To speed up this process, we have developed the following three optimizations, controlled by configuration flags":

  - `--dataloader-fast-cache-load`: This option assumes that the dataset cache already exists in the specified `--data-cache-path`. When enabled, it speeds up the creation process by removing synchronization points and file check assertions.

  - `--dataloader-defer-npy-index-mmap`: This option also assumes that the dataset cache already exists in the specified `--data-cache-path`. When enabled, it defers the memory mapping of the dataset indexes (.npy files) until their first access. We recommend using this configuration together with `--num-workers` > 0 so that the DataLoader prefetches the next batches of data, thereby hiding the cost of index memory mapping.

  - `--per-dataset-sequences-path`: With this configuration, we specify the JSON file generated by the `tools/build_sequences_per_dataset.py` script. This script generates a single file containing the required metadata from all the specified file prefixes. This configuration is especially useful when dealing with hundreds to thousands of file prefixes, since it requires only a single `open` operation instead of one per file prefix.