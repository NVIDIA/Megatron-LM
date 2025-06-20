# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.

"""Retro preprocessing config."""

from dataclasses import dataclass

from megatron.core.transformer import TransformerConfig

from .bert_embedders import RetroBertEmbedders
from .gpt_chunk_datasets import RetroGPTChunkDatasets
from .tokenizers import RetroTokenizers


@dataclass
class RetroPreprocessingConfig(TransformerConfig):
    """Configuration object for Retro preprocessing.

    *Note* : Arguments prefixed with '--retro-gpt-*' or '--retro-bert-*' are
    included and named as such to more easily handle managing both models
    running at the same time. Megatron is not optimized to run two models at
    once, so this naming convention makes it clearer.

    Args:

        retro_project_dir (str): Retro project directory, which contains the preprocessed data for for pretraining. This directory is built during preprocessing (see tools/retro/README.md), and contains subdirectories for the chunk database and pretraining neighbors.
        retro_tasks (str): Comma-separated list of tasks to run. Run entire preprocesing pipeline by using '--retro-tasks build'. Alternatively, run individual stages with tasks (in this order) 'db-build', 'index-build', or 'query-pretraining-neighbors'. For example, '--retro-tasks db-build,index-build,query-pretraining-neighbors' is equivalent to '--retro-tasks build'; or the argument can contain a subset of these tasks. Stages must always be run in the correct order (listed above).
        retro_task_validate (float): If defined, validate a randomly sampled subset of the existing results of the given task. Each task implements a 'validate' method that is responsible for sampling a `retro_task_validate` fraction of the existing results, and then checking for bitwise equality with the current code base. (E.g., `--retro-task-validate 0.01`.)
        retro_block_size (int): Number of chunks to process at a time when generating Bert embeddings and querying the search index. Partial results for each block are generally saved to disk in separate files.
        retro_doc_block_size (int): Number of documents to processe at time when processing token datasets into chunk databases. The partial chunk database for each block is saved into a separate file.
        retro_gpt_seed (int): Random seed used for python, numpy, pytorch, and cuda.
        retro_gpt_data_path (str): Path to the training dataset. Accepted format: 1) a single data path, 2) multiple datasets in the form: dataset1-weight dataset1-path dataset2-weight dataset2-path ... It is used with --split when a single dataset used for all three: train, valid and test. It is exclusive to the other --*-data-path args.
        retro_gpt_data_cache_path (str): Path to a directory to hold cached index files.
        retro_gpt_split (str): Comma-separated list of proportions for training, validation, and test split. For example the split `90,5,5` will use 90%% of data for training, 5%% for validation and 5%% for test.
        retro_gpt_train_samples (int): Total number of samples to train over all training runs.
        retro_gpt_eval_interval (int): GPT evaluation interval.
        retro_gpt_eval_iters (int): GPT evaluation iterations.
        retro_gpt_tokenizer_type (str): GPT tokenizer type.
        retro_gpt_tokenizer_model (str): GPT tokenizer model file.
        retro_gpt_vocab_file (str): GPT vocab file.
        retro_gpt_merge_file (str): GPT merge file.
        retro_gpt_seq_length (int): GPT sequence length.
        retro_gpt_global_batch_size (int): GPT global batch size.
        retro_gpt_chunk_length (int): GPT chunk length.
        retro_bert_tokenizer_type (str): Bert tokenizer type (for when using '--bert-embedder-type megatron').
        retro_bert_vocab_file (str): Bert vocab file.
        retro_bert_batch_size (int): Micro-batch size for processing Bert embeddings.
        retro_bert_max_chunk_length (int): Maximum sequence length for Bert embeddings. (Named 'chunk' here in reference to these Bert sequences being converted from GPT chunks.)
        retro_index_type (str): A 'faiss-base' index is a simple, un-optimized wrapper around a Faiss index. A 'faiss-par-add' index optimizes the 'add()' method by making it multi-node and multi-process, but with bit-wise equivalent results.
        retro_index_str (str): Index string used for calling faiss.index_factory(). For example, 'IVF262144_HNSW32,Flat' or 'OPQ32_256,IVF4194304_HNSW32,PQ32'.
        retro_index_ntrain (int): Number of database chunks to use for training the index. This value must be less or equal to the total number of chunks in the database.
        retro_index_train_load_fraction (float): Fraction of sampled chunks to use for training the index. Useful when our total sampled embeddings use too much memory; lowering the load fraction is less costly than re-embedding a new sampled dataset from scratch.
        retro_index_add_load_fraction (float): Fraction of database chunks to use for adding to the index. Useful when our total index size would use too much memory; lowering the load fraction is less costly than re-designing our token datasets.
        retro_index_delete_training_embeddings (bool): Delete training embeddings for the search index. Useful for debugging.
        retro_index_delete_added_codes (bool): Delete added codes for the search index. Useful for debugging.
        retro_query_ef_search (int): Index ef-search parameter for Hierarchical Navigable Small Worlds (HNSW) during querying.
        retro_query_nprobe (int): Index nprobe parameter for Inverted File (IVF) during querying.
        retro_query_num_neighbors_query (int): Number of neighbors to retrieve when calling index.search().
        retro_query_num_neighbors_save (int): Number of neighbors to save to disk after the index's returned neighbors. If longer than target value, neighbors truncated; and if shorter than target value, neighbors are padded with -1's.
        retro_bert_embedders (RetroBertEmbedders): Set of Bert embedders used for embedding chunks. Contains entries: 1) 'mem' for an in-memory embedder, and 2) 'disk' for an embedder that saves results in blocks to disk.
        retro_gpt_chunk_datasets (RetroGPTChunkDatasets): GPT datasets for 'train', 'valid', and 'test'.
        retro_tokenizers (RetroTokenizers): GPT ('gpt') and Bert ('bert') tokenizers.
    """

    # Basic.
    retro_project_dir: str = None
    retro_tasks: str = 'build'
    retro_task_validate: float = None
    retro_block_size: int = 100000
    retro_doc_block_size: int = 100000

    # GPT.
    retro_gpt_seed: int = 1234
    retro_gpt_data_path: list = None  # basic list here, for parsing purposes
    retro_gpt_data_cache_path: str = None
    retro_gpt_split: str = '969,30,1'
    retro_gpt_train_samples: int = None
    retro_gpt_eval_interval: int = None
    retro_gpt_eval_iters: int = None
    retro_gpt_tokenizer_type: str = None
    retro_gpt_tokenizer_model: str = None
    retro_gpt_vocab_file: str = None
    retro_gpt_merge_file: str = None
    retro_gpt_seq_length: int = None
    retro_gpt_global_batch_size: int = None
    retro_gpt_chunk_length: int = 64

    # Bert.
    retro_bert_tokenizer_type: str = None
    retro_bert_vocab_file: str = None
    retro_bert_batch_size: int = 128
    retro_bert_max_chunk_length: int = 256

    # Index.
    retro_index_type: str = 'faiss-par-add'
    retro_index_str: str = None
    retro_index_ntrain: int = None
    retro_index_train_load_fraction: float = 1.0
    retro_index_add_load_fraction: float = 1.0
    retro_index_delete_training_embeddings: bool = True
    retro_index_delete_added_codes: bool = True

    # Query.
    retro_query_ef_search: int = 256
    retro_query_nprobe: int = 65536
    retro_query_num_neighbors_query: int = 200
    retro_query_num_neighbors_save: int = 20

    # Tools.
    retro_bert_embedders: RetroBertEmbedders = None
    retro_gpt_chunk_datasets: RetroGPTChunkDatasets = None
    retro_tokenizers: RetroTokenizers = None

    def __post_init__(self) -> None:
        """Validate Retro config."""

        # Validate required attributes.
        assert self.retro_project_dir is not None
        assert self.retro_tasks is not None
        assert self.retro_gpt_data_path is not None or self.retro_gpt_data_cache_path is not None
        assert self.retro_gpt_train_samples is not None
        assert self.retro_gpt_eval_interval is not None
        assert self.retro_gpt_eval_iters is not None
        assert self.retro_gpt_tokenizer_type is not None
        assert self.retro_gpt_tokenizer_model is not None or (
            self.retro_gpt_vocab_file is not None and self.retro_gpt_merge_file is not None
        )
        assert self.retro_gpt_seq_length is not None
        assert self.retro_gpt_global_batch_size is not None
        assert self.retro_bert_tokenizer_type is not None
        assert self.retro_bert_vocab_file is not None
        assert self.retro_index_str is not None
        assert self.retro_index_ntrain is not None

        # Split retro tasks.
        self.retro_tasks = self.retro_tasks.split(",")
