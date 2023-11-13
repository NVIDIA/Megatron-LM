# Data Pipeline

## Data pre-processing

Data preprocessing is built around the following classes:

1. `MMapIndexedDatasetBuilder`
2. `MMapIndexedDataset`

At the moment, an end-to-end data preprocessing implementation is left to the user. See the class docstring(s) for more details.

#### MMapIndexedDatasetBuilder

The `MMapIndexedDatasetBuilder` is capable of building and merging `MMapIndexedDataset` instances.

#### MMapIndexedDataset

The `MMapIndexedDataset` class is the lowest-level data interface in Megatron Core. Internally, an `MMapIndexedDataset` instance references two binaries: the data file (`.bin`) contains document/sequence data and the index file (`.idx`) contains document/sequence metadata.

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
3. `MMapIndexedDataset`
3. `MegatronDataset`
4. `BlendedDataset`

See the class docstrings for more details.

#### BlendedMegatronDatasetConfig (extendable)

The `BlendedMegatronDatasetConfig` class parameterizes the `BlendedMegatronDatasetBuilder` and in turn the `MegatronDataset` and `BlendedDataset`.

Different training/inference regimes will require different extensions e.g. the `GPTDatasetConfig`

#### BlendedMegatronDatasetBuilder

The `BlendedMegatronDatasetBuilder` class builds the highest-level data interfaces in Megatron Core.

**NB:** All ranks should attempt to build the dataset via the `BlendedMegatronDatasetBuilder` or the program will hang. Which ranks follow through on their attempts can be controlled via the `BlendedMegatronDatasetConfig`.

#### MMapIndexedDataset

The `MMapIndexedDataset` class is the lowest-level data interface in Megatron Core.

The `MMapIndexedDataset` should already exist on disk before attempting to build any of the high-level data interfaces.


#### MegatronDataset (extendable)

The `MegatronDataset` abstract class is a high-level data interface in Megatron Core. It is an abstraction built upon the `MMapIndexedDataset`.

Different training/inference regimes will require different extensions e.g. the `GPTDataset`

#### BlendedDataset

The `BlendedDataset` class is a high-level data interface in Megatron Core. It is an abstraction built upon the `MegatronDataset`.

The `BlendedDataset` is only necessary when a blend multiple data distributions, i.e. multiple `MegatronDataset` instances, should contribute to a certain dataset split. The blend can be controlled via the `BlendedMegatronDatasetConfig`.

## Data loading: implementation

### GPTDataset

The `GPTDataset` is parameterized by the following variables: the underlying `MMapIndexedDataset` instance `indexed_dataset`, the split indices `indexed_indices` (the congituous subset of document or sequence indices used for training, validation, and testing), the number of samples `N`, the sequence length `S`, and the random seed `R`.

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
