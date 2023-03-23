This directory contains a collection of tools for building the retrieval database and pretraining neighbors for Retro. This preprocessing pipeline is broken into 3 main stages:

1. **Build retrieval chunk database** : Used for retrieving neighbors and continuation chunks, which are then passed through the retrieval encoder.
2. **Build index for similarity search** : Train and build a search index for querying chunk neighbors.
3. **Query pretraining neighbors** : For matching pretraining samples to database chunks. Neighbors are generated separately for training, validation, and test datasets.

The following overview goes into more detail on the pipeline, code structure, usage, and pretraining.

<!-- ################ contents ################ -->
# Contents

  * [Quick start](#quick-start)
  * [Stages](#stages)
  * [Code structure](#code-structure)
  * [Arguments](#arguments)
  <!-- * [Pretraining](#pretraining) -->

<!-- ################ quick start ################ -->
# Quick start

See `examples/get_preprocess_cmd.sh` for example arguments.

Key files:

- `main.py` : Entry point.
- `examples/get_preprocess_cmd.sh` : Build preprocessing command (for `main.py`).
- `examples/preprocess_data.sh` : Run preprocessing (calls `get_preprocess_cmd.sh`, `main.py`).

Use `--retro-tasks` to move through the preprocessing pipeline.

- Simplest setup (builds everything): `--retro-tasks build`
- Alternatively, for tuning compute resources, run stages independently:
  - Build retrieval database: `--retro-tasks db-build`
  - Build search index: `--retro-tasks index-build`
  - Query neighbors: `--retro-tasks pretraining-query-neighbors`

Sample code flow:

- `main.py` : Entry point (e.g., using `--retro-tasks X`).
- `db/build.py` : Build retrieval database.
- `index/build.py` : Build search index. Calls the following two files:
  - `index/train.py` : Train index on subset of database.
  - `index/add.py` : Add database chunks to index.
- `pretraining/query.py` : Query pretraining samples for database neighbors (saved to disk and used during pretraining).

<!-- ################ stages ################ -->
# Stages

### Build retrieval chunk database

This *database* (stored as a 2-D array, NOT a relational database) consists of a list of chunks (traditionally length 64) extracted from the original GPT token dataset. This is simply a consecutive, non-overlapping chunking of the token dataset. Chunking only takes place within a document, and therefore the final chunk of each document has length: 1 <= chunk_length <= max_chunk_length.

We discard chunks that would convert to an empty Bert sequence (rare case, happens ~1/100,000 chunks in our case), since we use Bert embeddings for building our index. Thus, the total number of chunks in the database will be slightly less than a naive calculation.

### Build index for similarity search

To match pretraining chunks to database chunks, a search index must be built to perform this querying. We use Faiss (https://github.com/facebookresearch/faiss) for training and building this index. Generally, the index is trained on a subset of all chunks in the database (specified via `--retro-nchunks-sampled`). After training, all chunks are added into the index, to be available during querying.

Indexes only accept 1-D floating point vectors for training and adding, so each chunk must first be embedded before passing to the index for either training or adding. We use Bert embeddings for this purpose, and the embeddings are generated automatically within the pipeline.

### Query pretraining neighbors

To ensure fast Retro pretraining, the database neighbors for pretraining samples are pre-computed and saved to disk, for efficient access within the Retro dataset. In this stage, the pretraining datasets (training, validation, and test) are iterated, each sample is broken into chunks, and the chunks are used for querying the index. Similar to when building the index, each chunk is embedded (via Bert) before querying the index.

The saved neighbors are labeled with unique dataset properties (i.e., seed, sequence length, number of samples, etc.) to ensure the neighbors generated during preprocessing match the neighbors requested during pretraining.

<!-- ################ code structure ################ -->
# Code structure

### `tools/retro/main.py`

This is the main entry point for Retro preprocessing. Call `main.py --help` to see arguments. Additionally, some Retro arguments are in Megatron's core arguments, so also see `add_retro_args()` section of `megatron/arguments.py` for additional arguments. Two of the most important arguments to customize are `--retro-workdir` and `--retro-tasks`.

- **`--retro-workdir`** : Set the directory in which the preprocessing pipeline saves its datasets and configuration files. This argument should remain consistent for a full pass through the pipeline, and for pretraining.

- **`--retro-tasks`** : Set the stages of preprocessing to perform. As mentioned previously, the three high-level stages are: 1) build retrieval database, 2) build search index, and 3) query pretraining neighbors. `--retro-tasks` can be used to either run the full pipeline, or run each of these stages in isolation. The latter case is useful for tuning compute resources for each stage. For example, index training utilizes GPUs and requires relatively less time, while querying neighbors uses the CPU and is a relatively slow process. Example tasks include:

  - **`--retro-tasks build`** : Run entire preprocessing pipeline.
  - **`--retro-tasks db-build`** : Build retrieval database.
  - **`--retro-tasks index-build`** : Train and build search index.
  - **`--retro-tasks pretraining-query-neighbors`** : Query pretraining neighbors.

Multiple tasks can be specified by separating with commas (e.g., `--retro-tasks db-build,index-build`). Additionally, various 'miscellaneous' tasks are currently including, primarily for validating data for each stage; these task names can be seen in `main.py`.

### `tools/retro/examples`

Example scripts for setting arguments and launch Retro preprocessing. The key files here are:

- **`get_preprocess_cmd.sh`** : Sets up arguments and command for preprocessing. **Important note**: this script assumes a few environment variables are already set before it is called. Please see the `Environment vars.` section at the top of this file. Generally, environment variables must be set to determine the location of Retro workdirs, input datasets, and GPT and Bert model information.
- **`preprocess_data.sh`** : Calls `get_preprocess_cmd.sh` to get arguments, and then calls `main.py` to launch preprocessing.
- **`pretrain_model.sh`** : Example script for pretraining on Wikipedia data, after preprocessing is complete.

### `tools/retro/db`

Build the retrieval chunk database. The key files here are:

- **`build.py`** : Entry point for building the database. This code is responsible for iterating the input datasets (i.e., `--data-path`), parsing each dataset into consecutive chunks, checking for empty Bert (Wordpiece) conversions, and storing this information to disk. Two databases are created: 1) the retrieval database, and 2) a sampled database used for training the search index.
- **`dataset.py`** : Defines database class, for iterating or accessing chunks in the database. Each chunk contains its tokens, Bert conversion length, and dataset index.

Input data:

<!-- - Token datasets, as generated by `tools/preprocess_data.py`. Each dataset should include a `.bin` and `.idx` file. Multiple datasets can be specified by using a blended configuration (see `--data-path` in `megatron/arguments.py`). -->
- Token datasets, as loaded by `gpt_dataset.py`. Multiple datasets can be specified by using a blended configuration (see `--data-path` in `megatron/arguments.py`).

Output data:

- **`<RETRO_WORKDIR>/db/merged/train.hdf5`** : The main retrieval database. (*Database* here is used to denote a list of indexed chunks, rather than a *relational database*.) The chunks in this database are added to the search index, and are used for retrieval during pretraining. This file contains a single dataset `'chunks'`, which contains 5 columns:

  - `dataset_idx` : Dataset index, from list of blended indexed datasets.
  - `document_idx` : Document index within dataset.
  - `chunk_start_idx` : Chunk's starting token index within document.
  - `chunk_end_idx` : Chunk's ending token index (exclusive) within document.
  - `bert_chunk_length` : Length of Bert token sequence, after converting from GPT.

- **`<RETRO_WORKDIR>/db/merged/sampled.hdf5`** : Subset of training database that is used for training the search index. This file has the same structure as detailed above. In general, this database is significanly smaller than the `train.hdf5` database, since the search index only needs a relatively small number of samples to understand the data's structure. After training, all chunks in the main database (`train.hdf5`) are *added* to the search index.

### `tools/retro/index`

Build the search index. The key files here are:

- `build.py` : Entry point for building the search index. First, the index is trained on the sampled chunk database (see above) by calling `train.py`, and then all chunks for the full database are added to the index by calling `add.py`. Note that training requires first embedding (using Bert) all chunks (a parallel operation), and then loading these embeddings and training the index (a sequential operation), so it's best to change one's compute setup after all chunks have been embedded and saved to disk.
- `indexes/faiss_base.py` : Wrapper class for building a Faiss index, following the standard `train()` and `add()` operations.
- `indexes/faiss_par_add.py` : Similar to above, except it uses an embarrassingly parallel (multi-node, multi-process) `add()` operation. Vectors are first added to separate index copies, and then merged together.

Input data:

- **`<RETRO_WORKDIR>/db/merged/sampled.hdf5`** : Chunks used for training the search index.
- **`<RETRO_WORKDIR>/db/merged/train.hdf5`** : Chunks used for adding to the *trained* search index.

Output data:

- **`<RETRO_WORKDIR>/index/<RETRO_INDEX_TYPE>/<RETRO_INDEX_STR>/added.faissindex`** : The final index, which has been trained and has had all database chunks added to it. This index is ready for querying neighbors. Here, `RETRO_INDEX_TYPE` and `RETRO_INDEX_STR` correspond to the same-name arguments `--retro-index-type` (e.g., `faiss-par-add`) and `--retro-index-str` (e.g., `OPQ32_256,IVF4194304_HNSW32,PQ32`).
- **`<RETRO_WORKDIR>/index/<RETRO_INDEX_TYPE>/<RETRO_INDEX_STR>/empty.faissindex`** : Generally can be discarded once `added.faissindex` has been built, but this file contains the *post-training*, *pre-adding* index. Useful for debugging or building other indexes.

### `tools/retro/pretraining`

Query the pretraining datasets (training, validation, test) for their neighbors within the database. Neighbors are queried during preprocessing -- rather than during pretraining -- because querying is a fairly slow operation, so it would be a bottleneck if performed during pretraining. Queried neighbors are tagged with their unique identifying information (e.g., `train_indexmap_27662746ns_2048sl_1234s`), so as to avoid incorrect references during pretraining. The key files here are:

- **`query.py`** : Entry point for querying. The pretraining datasets are iterated, and each chunk within each sample is queried using the search index. These neighbors are filtered by discarding any database chunks that fall within the same document as any chunk within a pretraining sample.
- **`chunk_dataset.py`** : This creates an iterable 'chunk' dataset form of a pretraining dataset. This is just a light wrapper, but makes it easier to deterministically iterate and assign IDs to each chunk in a sample dataset.
- **`retro_dataset.py`** : The Retro dataset used for pretraining (not used in preprocessing). Each sample returns the sample tokens, along with neighbor tokens for each chunk within the sample.

Input data:

- Token datasets, as loaded by `gpt_dataset.py`.
- **`<RETRO_WORKDIR>/index/<RETRO_INDEX_TYPE>/<RETRO_INDEX_STR>/added.faissindex`** : The trained index, with all database chunks added to it (see previous section for details).

Output data:

- **`<RETRO_WORKDIR>/{train,valid,test}_XXns_YYsl_ZZs/WW.hdf5`** : These directories/files contain the indexes of neighbors for each chunk within each sample of the pretraining datasets. Each directory (e.g., `train_indexmap_2047435ns_2048sl_1234s`) contains a list of HDF5 files (e.g., one file might be called `0075700000-0075800000.hdf5`). Each HDF5 file contains a consecutive subset of neighbor IDs for a given chunk, for indexing into the main retrieval database. All HDF5 files taken together within a given directory, represent the entire set of neighbors for a dataset. The size of these HDF5 files is determined by the argument `--retro-block-size`. The `XX`, `YY`, `ZZ`, `WW` notation above denotes the dataset properties that are used for uniquely tagging the neighbor files, to ensure compatibility during model pretraining. These neighbor files are ultimated used by `retro_dataset.py` during pretraining, for building Retro samples.

### `tools/retro/cli`

Inspect preprocessed data. To use the CLI, open a Python terminal via the `python` command, and then load a Retro workdir with the following:

```
from tools.retro.cli import retro
retro.init("/path/to/retro/workdir")
```

This initializes Megatron, and prepares the Retro data for inspection. See the printed usage for available functions. Several routines are included for viewing data in the retrieval database and viewing pretraining samples and neighbors. For example:

```python
retro.get_db_num_indexed_datasets() # 15
retro.get_db_chunk_text(92874113) # 'research project at ...  and philosophy'
retro.get_pt_sample('train', 62005) # '[16084, 26158, 25387 ..., 6898, 9568]'
```

Most methods within the CLI are prefixed to denote the data being inspected:

- **'db'** : Retrieval database (i.e., chunk tokens, document IDs, and dataset IDs)
- **'pt'** : Pretraining datasets (i.e., sample tokens and neighbor tokens)

### `tools/retro/utils.py`

A collection of utility methods. Most importantly, this contains:

- **`def get_gpt_tokenizer()`** : Get the GPT tokenizer.
- **`def get_bert_tokenizer()`** : Get the Bert tokenizer.
- **`class GPTToTextDataset`** : Wrapper class that converts GPT (BPE) samples to raw text.

### `tools/bert_embedding`

Generate Bert embeddings. The main files here are:

- **`embed.py`** : Entry point for generating embeddings, and contains the two main embedding classes, `BertEmbedder` and `DiskDataParallelBertEmbedder` (more below). This file contains code for generating Megatron embeddings, while the file below contains code for Huggingface embeddings.
- **`huggingface.py`** : Used by `embed.py` when the embedder is configured (see below) to output Huggingface embeddings.
- **`dataset.py`** : Wrapper class for converting a raw-text dataset to Bert (Wordpiece) tokens.

The Bert embeddings can be configured along two axes. The first axis is the output type:

- **`class BertEmbedder`** : This class takes a raw-text dataset as input, generates its embeddings, and returns a Numpy array. The main functions are `embed_text_dataset` (accepts a raw-text dataset) and `embed_text` (accepts a string).
- **`class DiskDataParallelBertEmbedder`** : This class wraps `BertEmbedder`, and rather than returning a Numpy array, it saves the embeddings to disk. Additionally, this class automatically splits data across data parallel ranks (using interleaving), and also processes data in a specified `block_size` (e.g., 1,000,000).

The second axis is the type of embedding model to use, controlled by the argument `--bert-embedder-type`:

- **`--bert-embedder-type megatron`** : Use Megatron's Bert model. The specific model used is dependent on the loaded checkpoint, vocab file, and tokenizer.
- **`--bert-embedder-type huggingface`** : Use Huggingface's `bert-large-cased`. (*Note*: Huggingface's inclusion is likely to be deprecated; and there is no ability to configure cased/uncased.)

### Pretraining

- **`pretrain_retro.py`** : Launch script for pretraining Retro. Similar to `pretrain_gpt.py`, except this script handles loading neighbor tokens and setting up the neighbor attention mask.
<!-- - `megatron/data/gpt_dataset.py` : ? -->
- **`megatron/model/retro_transformer.py`** : Implementation of Retro model, including the main transformer, the retrieval encoder, and chunked cross-attention layers. Note that currently, `retro_transformer.py` contains several classes that are nearly identical to `transformer.py`, except for 1 or 2 lines, due to code changes that are yet to be integrated.
- **`tools/retro/pretraining/retro_dataset.py`** : The Retro dataset used for pretraining (not used in preprocessing). Each sample returns the sample tokens, along with neighbor tokens for each chunk within the sample.


<!-- ################ arguments ################ -->
# Arguments

See `tools/retro/main.py`'s `add_retro_args()` and `megatron/arguments.py`'s `_add_retro_args()` for details and descriptions. Here we list some particularly important arguments:

- `--retro-workdir` : Mentioned previously, this argument determines the directory in which a set of Retro data is stored (during preprocessing) and loaded (during pretraining). Any change in this directory during preprocessing may result in preprocessing starting over from scratch, and any change before pretraining will result in pretraining throwing an error.
- Preprocessing
  - `--retro-gpt-chunk-length` : Retro chunk length (e.g., 64 in original paper).
  - `--retro-tasks` : Comma-separated list of preprocessing tasks. Generally, the `build` task is the simplest way to run the preprocessing pipeline. For finer control, individual stages can be run by using tasks (in order): `db-build`, `index-build`, and `pretraining-query-neighbors`.
  - `--retro-index-str` : Faiss index string that defines the index configuration. This will vary based on data size, compute/disk setup, and user needs. For example, this string looks something like `IVF262144_HNSW32,Flat` or `OPQ32_256,IVF4194304_HNSW32,PQ32`.
- Pretraining
  - `--retro-add-retriever` : Must be used to select Retro model.
  - `--retro-num-neighbors` : Number of neighbors to retrieve from the retrieval database (defaults to 2).
  - `--retro-num-retrieved-chunks` : For each neighbor, the number consecutive chunks to retrieve, including the initial neighbor (defaults to 2).

<!-- ################ pretraining ################ -->
<!-- # Pretraining -->
<!-- - New retro args in arguments.py (add_retro_args). -->
<!-- - Most important arg is `--retro-add-retriever`. -->
