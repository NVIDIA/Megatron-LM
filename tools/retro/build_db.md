This directory contains a collection of tools for building the retrieval database and pretraining neighbors for Retro. This preprocessing pipeline is broken into 3 main stages:

1. **Build retrieval chunk database** : Used for retrieving neighbors and continuation chunks, which are then passed through the retrieval encoder.
2. **Build index for similarity search** : Train and build a search index for querying chunk neighbors.
3. **Query pretraining neighbors** : For matching pretraining samples to database chunks. Neighbors are generated separately for training, validation, and test datasets.

The following overview goes into more detail on the pipeline, code structure, usage, and pretraining.

<!-- ################ contents ################ -->
# Contents

  * [Quick start](#quick-start)
  * [Tutorial](#tutorial)
  * [Code structure](#code-structure)
  * [Arguments](#arguments)
  <!-- * [Pretraining](#pretraining) -->

<!-- ################ quick start ################ -->

# Quick Start
Key files:

- `main.py` : Entry point for processing.
- `examples/preprocess_data.sh` : Example preprocessing launch (calls `main.py`).
- `examples/pretrain_data.sh` : Example pretraining launch (calls `pretrain_retro.py`).

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

<!-- ################ tutorial ################ -->

# Tutorial

In this tutorial example, we use Wikipedia corpus to demonstrate how we build a retrieval database and index for this corpus, and then query the pretraining datasets for their neighbors.

## Step 1: Prepare your retrieval text corpus

The format of text corpus follows the same format as in Megatron training. See [data precessing](https://github.com/NVIDIA/Megatron-LM/tree/main#data-preprocessing) for more details on how to convert your json dataset into the mmap format.

Assume we have the Wikipedia corpus in the following format:

```
<retrieval/db/path>/Wikipedia_shuf_text_document.bin
<retrieval/db/path>/Wikipedia_shuf_text_document.idx
```

We note that the retrieval database can also be a blend of multiple text corpus.

## Step 2: Build retrieval chunk database

This *database* (stored as a 2-D array, NOT a relational database) consists of a list of chunks (traditionally length 64) extracted from the original GPT token dataset. This is simply a consecutive, non-overlapping chunking of the token dataset. Chunking only takes place within a document, and therefore the final chunk of each document has length: 1 <= chunk_length <= max_chunk_length.

We discard chunks that would convert to an empty Bert sequence (rare case, happens ~1/100,000 chunks in our case), since we use Bert embeddings for building our index. Thus, the total number of chunks in the database will be slightly less than a naive calculation.

Take the Wikipedia corpus as an example to build the retrieval chunk database:

Prepare the following arguments and update our templates in `tools/retro/examples/preprocess_data.sh`:
- `--retro-workdir`: The directory in which the preprocessing pipeline saves its datasets and configuration files. 
  **This argument should remain consistent for a full pass through the pipeline, and for pretraining.**
- `--data-path`: text corpus path to build retrieval database. In the case of Wikipedia corpus, it could be
```bash
WIK="${DATA_HOME}/Wikipedia_shuf_text_document"

DATA_BLEND=" \
  1 ${WIK} \
"
```
- `--load`: bert path to load bert embedder
- `--vocab-file` and `--retro-bert-vocab-file`: bert vocab file
- `--retro-gpt-tokenizer-model`: gpt tokenizer model file

Then launch the script:
```bash
bash tools/retro/examples/preprocess_data.sh db-build
```

After the `db-build` is finished, the output includes:
- The launching args will be saved in your `<retro-workdir>/args.json` for the following steps. 
- The retrieval chunk database will be saved in your `<retro-workdir>/db/` with your dataset information in `<retro-workdir>/db/indexed_dataset_infos.json`.  

## Step 3: Build index for similarity search

To match pretraining chunks to database chunks, a search index must be built to perform this querying. We use Faiss (https://github.com/facebookresearch/faiss) for training and building this index. Generally, the index is trained on a subset of all chunks in the database (specified via `--retro-nchunks-sampled`). After training, all chunks are added into the index, to be available during querying.

Indexes only accept 1-D floating point vectors for training and adding, so each chunk must first be embedded before passing to the index for either training or adding. We use Bert embeddings for this purpose, and the embeddings are generated automatically within the pipeline.

Take the Wikipedia corpus as an example to build the retrieval chunk database:

```bash
bash tools/retro/examples/preprocess_data.sh index-train
```
The `index-train` step is expected to take less than 4-hour on a single DGX-A100 node given the template index configuration. 
To scale up for larger retrieval database, please carefully tune the faiss hyper-parameters specified in `--retro-index-str`. Please refer to [Faiss](https://github.com/facebookresearch/faiss/wiki/The-index-factory) to learn more about the index configuration.  

After the index is trained, the centroids, HNSW graph, and product quantizer is determined. However, the index is still empty, as there is no chunk added.

Take the example of the Wikipedia corpus, with the default template, the output of `index-train` includes:
- The embedded Bert embeddings of the sampled chunks for `index-train` is saved in `<retro-workdir>/index/train_emb/`.  
- The empty index is saved in `<retro-workdir>/index/faiss-par-add/OPQ32_64,IVF65536_HNSW8,PQ32/empty_0.970.faissindex`.

Then we add all chunks in the retrieval database into the index so that we perform fast query over the whole retrieval database:
```bash
bash tools/retro/examples/preprocess_data.sh index-add
```

We note that this step can be time-consuming as it will go through the whole retrieval database, embed chunk tokens  to BERT embeddings, and add them into the index. Please make sure you successfully add the whole retrieval database before moving on to the next stage.

*In case your job is interrupted in the middle, you can just run the script again, and it will automatically skip the chunks that have been added into the index and start from the chunk where it is interrupted.*


Following the Wikipedia configuration, an example output of the step `index-add` includes:
- The index with retrieval data chunks added is saved in `<retro-workdir>/index/faiss-par-add/OPQ32_64,IVF65536_HNSW8,PQ32/added_0.970_0.950.faissindex`, which can be used to query the neighbors for pretraining.

## Step 4: Query pretraining neighbors

To ensure fast Retro pretraining, the database neighbors for pretraining samples are pre-computed and saved to disk, for efficient access within the Retro dataset. In this stage, the pretraining datasets (training, validation, and test) are iterated, each sample is broken into chunks, and the chunks are used for querying the index. Similar to when building the index, each chunk is embedded (via Bert) before querying the index.

The saved neighbors are labeled with unique dataset properties (i.e., seed, sequence length, number of samples, etc.) to ensure the neighbors generated during preprocessing match the neighbors requested during pretraining. Please also make sure the pretraining configuration is the same as this step so that the neighbors are aligned.

There are query-time hyper-parameters that can be tuned to improve the quality of the neighbors. These are specified in `RETRO_QUERY_EF_SEARCH` and `RETRO_QUERY_NPROBE`. The most important parameter is `RETRO_QUERY_NPROBE`, which controls the number of clusters to search during querying. This parameter can be tuned to improve the quality of the neighbors, but will also increase the query time. 
We recommend following the tutorial of [faiss](https://github.com/facebookresearch/faiss/wiki/Index-IO,-cloning-and-hyper-parameter-tuning) to tune the hyper-parameters for your own retrieval database. 

Take the Wikipedia corpus as an example to query the neighbors in the retrieval database:

```bash
bash tools/retro/examples/preprocess_data.sh query-pretraining-neighbors
```

The output of `query-pretraining-neighbors` on the Wikipedia corpus includes:
- `<retro-workdir>/wiki/query/train_855ab50e05151610301e2a74c4030fbc`, which contains the pre-retrieved neighbors for the pretraining dataset. 
- `<retro-workdir>/wiki/query/valid_40bc7330318d64accec28e1e63c59bad`, which contains the pre-retrieved neighbors for the validation set of the pretraining corpus.

## Step 5: Visualization of retrieval neighbors

We also provide cli tools to help visualize and inspect the quality of your retrieved neighbors. 

To use the CLI, open a Python terminal via the `python` command, and then load a Retro workdir with the following:

```
from tools.retro.cli import retro
retro.init("/path/to/retro/workdir")
```

This initializes Megatron, and prepares the Retro data for inspection. We also print out some example commands to help you get familiar with the command lines.   

An example output for the Wikipedia Corpus:

```text
setting number of micro-batches to constant 32
> building BertWordPieceLowerCase tokenizer ...
> initializing torch distributed ...
> initialized tensor model parallel with size 1
> initialized pipeline model parallel with size 1
> compiling dataset index builder ...
...
...
 > sample ratios:
   dataset 0, input: 1, achieved: 1
> size of blendable dataset: 201000 samples
> elapsed time for building blendable dataset indices: 0.00 (sec)
> building indices for blendable datasets ...
 > sample ratios:
   dataset 0, input: 1, achieved: 1
> size of blendable dataset: 12864 samples
> finished creating pretrained GPT datasets ...

+++++++++++++++++++++++++++++++++++++++++++++++++++
examples ... [ *note*: 'db' = chunk db; 'pt' = pretraining corpus. ]
+++++++++++++++++++++++++++++++++++++++++++++++++++

~~~~ indexed datasets ~~~~
retro.get_db_num_indexed_datasets() : 1
retro.get_db_indexed_dataset_infos() :
  [(1.000000, Wikipedia_shuf_text_document)]

~~~~ counts ~~~~
retro.get_db_num_chunks : 68104992.

retro.get_pt_num_samples('train') : 201000.
retro.get_pt_num_samples('valid') : 12864.
retro.get_pt_num_chunks('train') : 1608000.
retro.get_pt_num_chunks('valid') : 102912.

~~~~ tokens, text ~~~~
retro.get_db_chunk_gpt(chunk_id) : [46809, 218340, 716, 647, ... , 251525, 872, 692, 4042]
retro.get_db_chunk_bert(chunk_id) : [10680, 16216, 4313, 1745 ... , 8117, 1007, 1012, 1997]
retro.get_db_chunk_text(chunk_id) : Jonas Geirnaert\n\nJonas  ... ort Flatlife (11 min). Of
retro.get_db_chunk_and_continuation_text(chunk_id) :
  ['Jonas Geirnaert  Jonas Ge ... ort Flatlife (11 min). Of',
   'the copy he sent in for s ... abet, clearly has one. On']

retro.get_pt_sample('train', sample_id) :
  {
    'dataset_idx' : 0
    'text' : [   676     14  40656 184 ... 4\n    276  17361 251542]
    'doc_ids' : [1246422 1596948 2403969]
    'neighbor_chunks' : [[[  657380   657381]\n   ... \n  [34108760 34108761]]]
    'neighbor_tokens' : [[[   276   9596 251511 . ... .    889    646   1723]]]
  }

(e.g., sample = retro.get_pt_sample(...))

  sample['text'].shape : (513,)
  sample['neighbor_tokens'].shape : (8, 20, 128)
  sample['text'] : [   676     14  40656 184 ... 4\n    276  17361 251542]
  sample['neighbor_tokens'][17][1] : [    14     14  30291   1 ... 682    328    379 251527]
  retro.gpt_to_text(sample['text']) : also\nLatgalians (modern) ... ission criticised the AVN
  retro.gpt_to_text(sample['neighbor_tokens']) : \n\nHis second marriage o ... Augusta Eardley-Wilmot (2
+++++++++++++++++++++++++++++++++++++++++++++++++++
```

We can also directly call the function `retro.print_neighbor_texts(sample_id, chunk_id)` to inspect the retrieval neighbors for a specific sample and chunk within the pretraining corpus. For example,  

```text
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
PRETRAINING CHUNK:
  - also\nLatgalians (modern)\n\nReferences\n\nCategory:Defunct political parti ... e.\n\nAbout \nThe company was established established in 1997. It is listed
NEIGHBOR_CHUNKS:
  - the sides.\n\nNotes\n\nReferences\n\nCategory:Obaku Zen\n*\nCategory:Japane ... 2, 2008. It was founded by Anand Jagannathan, CEO of parent company Kriyari
  - 2007).\n\nSee also\n Satellite Communications\n Tonga\n\nReferences\n\nExte ... y Procter & Gamble (P&G) in 1985 in order for P&G to compete in the "beauty
  - Japan\nCategory:Fish of Russia\nCategory:Fish described in 1845 Mareco Inde ... lic Opinion (WAPOR)\n European Society for Opinion and Marketing Research (
  - The current director of the company is Albert Bosch.\n\nSee also\n Coupon\n ...  some articles in Basque. Deia is the main product of the Editorial Iparrag
  - A.Ş have been traded on the Istanbul Stock Exchange since 2000.\n\nReferenc ... with stores in California, New York City, and London.\n\nHistory \nSnapette
  - \nCategory:Hawaiian mythology\nCategory:Hawaiian religion\nCategory:Religio ... crative state contracts. In 2008 Prokom became a part of the Asseco capital
  - , and the Baltic countries, as well as an online store.\n\nReferences\n\nEx ... nd are involved in intracellular trafficking. This protein does not contain
  - juice producer\nFood industry of Russia\n\nReferences\n\nExternal links\nWi ... panies formerly listed on the New York Stock Exchange General Grant's March
  - is in private ownership.\n\nReferences\n\nExternal links\n\nCategory:Online ... ten and directed by Brent Hodge. The film stars Aubrey Plaza, Molly Hawkey,
  - company's display technology to manufacture and sell display-only engines.\ ... for a group of naval vessels (a division in naval usage).\n\nUsage\n Russia
  - .\n\nCarrols also operated a chain of outlets in neighbouring Estonia from  ... rama film directed by Raajeev Walia. It is produced by Aman Mehta and Bijal
  - \n\nExternal links\nHightail website\nThe Next Web on YouSendIt rebrand to  ... eptember 2014, sitting mainly in the criminal division of that court.\n\nBe
  - American television seasons\nCategory:2014 American television seasons\nCat ...  Canada and larger European cities.\n\nIn 2010, advertising in New Zealand,
  - .\n\nNotes\n\nCategory:Trade unions\nCategory:Industrial Workers of the Wor ... x people, some of whom may have been working on a part-time basis. Its head
  - \n List of podcasting companies\n\nReferences\n\nExternal links\n \n\nCateg ... ct.\n\nCategory:Populated places in the Ashanti Region Nkeirouka Ezekh\n\nN
  - \n\nReferences\n\nExternal links\n ADESE official website\n\nCategory:Compa ...  State Street, and UBS Warburg. Its first CEO was Ian M. Drachman. The firm
  - Hotel\n Sulake Corporation\n Sulake Press Room\n Habbo Hotel - Blog\n\nCate ... l: 김진태; born December 19, 1980), better known by his stage name Verbal Jint
  - hockey player\n Ruutu.fi, a Finnish television streaming service operated b ...  from the bottom, a BDSM term\n Topping cycle, a cycle used in power plants
  - of Surakarta\nCategory:Indonesian names\nCategory:Indonesian families\nCate ... mber 13, 2013 in Izhevsk on Universitetskaya Street (later it was given the
  - facilities are also in Ankara and the company HQ is in Istanbul.\n\nReferen ... is currently a World Wide Web Consortium Working Draft.\n\nSee also\n Voice
```

The code snippet for the above example is also equivalent to
```python
tokens = retro.get_pt_sample('train', 0)
for token_ids in tokens["neighbor_tokens"][0]:
    print("- %s" % (retro.gpt_to_text(token_ids)))
    print("-" * 20)
```
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

- **`preprocess_data.sh`** : Example launch script for preprocessing retro data.
- **`pretrain_model.sh`** : Example launch script for pretraining a retro model.

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
