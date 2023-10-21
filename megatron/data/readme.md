# Data Pipeline

## GPT

The GPT data pipeline is built around the following three classes. Each successive class is an abstraction built upon the preceding class.

1. `MMapIndexedDataset`
2. `GPTDataset`
3. `BlendableDataset`

### Indexed Dataset

The `MMapIndexedDataset` is the lowest-level data interface in Megatron-LM. For each dataset prefix mapping to a pair of `.bin` and `.idx` files (provided via `--data-path` or `--[train|valid|test]-data-path`), one MMapIndexedDataset will be created.
- The `.bin` file is a binary which contains document and token data
- The `.idx` file is a binary which contains document and token metadata for indexing into the `.bin` file

Inside the `.idx` file are found the following information in the following order:
- The index header, for backward compatibility
- The index version, for backward compatibility
- A numeric code corresponding to the data type used to write the `.bin` file
- The number of sequences in the dataset
- The number of documents in the dataset
- The number of tokens per sequence
- The byte offsets for all sequences
- The sequence indices marking the end of each document
- The mode per sequence (in the multimodal case)

### GPTDataset

The `GPTDataset` is an abstraction built upon `MMapIndexedDataset` and is parameterized by the following variables: the contributing `MMapIndexedDataset` class instance `indexed_dataset`, the split `Split` (the congituous subset of document indices used for training, validation, and testing), the number of samples `N`, the sequence length `Seqlen`, and the random seed `Seed`.

The `GPTDataset` creates three index mappings to facilitate lookup: (1) the document index, (2) the sample index, and (3) the shuffle index.

1. The document index _Do_idx_ is a 1-D array mapping from _i_ to document index of length `Epochs * |Split|` where `Epochs` corresponds to the minimum number of epochs such that `Epochs * |Split| >= N`. The document index is shuffled according to `Seed`.

    ```
    Given:

    N = 15
    Split = [5, 6, 7, 8, 9]
    Epochs = 3

    Then, for example:

    Do_idx = [8, 8, 9, 6, 7, 5, 8, 5, 6, 6, 5, 9, 7, 7, 9]
    ```

2. The sample index _Sa_idx_ is a 2-D array mapping from _j_ to pairs of (_i_, _Do_idx_[ _i_ ] offset) of shape `[N + 1, 2]`. The rows _j_ and _j_ + 1 serve as the left and right bounds for the _j_-th sample. 

    ```
    Given:

    Seqlen = 1024

    Then, for example:

    Sa_idx[0] = (0, 0)
    Sa_idx[1] = (0, 1024)       => Do_idx[0] has length greater than Seqlen
    Sa_idx[2] = (1, 512)        => Do_idx[0] has length 1536
    Sa_idx[3] = (2, 0)          => Do_idx[1] has length 1536
    Sa_idx[4] = (5, 300)        => Do_idx[2:5] are shorter documents relative to Do_idx[0:2]
    Sa_idx[5] = (6, 24)         => Do_idx[5] has length 1300
    ```

3. The shuffle index _Sh_idx_ is a 1-D array mapping from _k_ to _j_ of length `N`. The shuffle index is shuffled according to `Seed`.

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
- Use the document index to retrieve `Seqlen` tokens from consecutive (in the document index) documents.

    ```
    sample = []
    sample += indexed_dataset[Do_idx[i]][offset:]
    if i != i_next:
        sample += indexed_dataset[Do_idx[i + 1:i_next]]
    sample += indexed_dataset[Do_idx[i_next]][:offset_next]
    ```

To save time during initialization (we don't want to build these indices again), each index is saved and cached (see `--data-cache-path`). The cached indices are unique to a hash which is determined by the parameters used to initialize the `GPTDataset`. They are `<hash>_doc_idx.npy`, `<hash>_sample_idx.npy`, and `<hash>_shuffle_idx.npy`.

### BlendableDataset

The `BlendableDataset` is an abstraction built upon single distribution dataset classes, e.g. `GPTDataset`, and is parameterized by the following variables: the contributing class instances `datasets`, the weights `Weights` (one per dataset), and the size `Size`. The `BlendableDataset` will draw samples from contributing datasets in proportion to the weights until achieving a composite dataset of the desired size. At each sampling step, we draw a single sample from the dataset which has the greatest sampling error.

The `BlendableDataset` creates two "blending" indices to facilitate lookup: (1) the datasat index and (2) the dataset sample index.

1. The dataset index _Da_idx_ is a 1-D array mapping from _i_ to dataset index of length `Size`.

    ```
    Given

    datasets = [d0, d1, d2]
    Weights = [1/2, 1/4, 1/4]
    Size = 4

    Then, for example:

    Da_idx = [0, 1, 2, 0]

    ```

2. The dataset sample index _Sa_idx_ is a 1-D mapping from _i_ to the sample index for dataset _Da_idx[i]_ of length `Size`.

    ```
    Given

    Da_idx = [0, 1, 2, 0]

    Then, for example:

    Sa_idx = [0, 0, 0, 1]
    ```

To query the `BlendableDataset` for the _k_-th sample we do the following

- Use the dataset index to retrieve the corresponding dataset from `datasets` and the dataset sample index to retrieve the corresponding sample from that dataset.

    ```
    sample = datasets[Da_idx[k]][Sa_idx[k]]
    ```

To save time during initialization (we don't want to build these indices again), each index is saved and cached (see `--data-cache-path`). The cached indices are unique to a hash which is determined by the parameters used to initialize the `BlendableDataset`. They are `<hash>_index.npy` and `<hash>_sample_index.npy`.