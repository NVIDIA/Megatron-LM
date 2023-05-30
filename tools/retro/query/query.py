# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

import numpy as np
import os
import psutil
import time
import torch
from tqdm import tqdm

from megatron import get_retro_args, print_rank_0
from tools.bert_embedding import BertEmbedder
from tools.bert_embedding.utils import get_missing_blocks_by_rank
from tools.retro.db.utils import \
    get_merged_train_dataset as get_db_merged_train_dataset
from tools.retro.external_libs import faiss, h5py
from tools.retro.index.factory import IndexFactory
from tools.retro.index.utils import get_index_dir
from tools.retro.utils import GPTToTextDataset

from .chunk_dataset import get_chunk_dataset_map as get_query_dataset_map


def get_index(ondisk=False):
    '''Read index from disk.'''

    args = get_retro_args()

    # Load index.
    index_wrapper = IndexFactory.get_index(args.retro_index_type)
    index_dir = get_index_dir()
    added_index_path = index_wrapper.get_added_index_path()
    if ondisk:
        index = faiss.read_index(added_index_path, faiss.IO_FLAG_MMAP)
    else:
        index = faiss.read_index(added_index_path)

    # Search parameters.
    faiss.ParameterSpace().set_index_parameter(index, "efSearch",
                                               args.retro_query_ef_search)
    faiss.ParameterSpace().set_index_parameter(index, "nprobe",
                                               args.retro_query_nprobe)

    return index


def embed_block(gpt_dataset, block, embedder):
    '''Embed block of chunks.'''
    text_block_dataset = torch.utils.data.Subset(
        GPTToTextDataset(gpt_dataset),
        range(*block["range"]),
    )
    return embedder.embed_text_dataset(text_block_dataset)


def query_embeddings(db_dataset, index,
                     embeddings, chunk_id_range,
                     sample_map, n_chunks_per_sample,
                     verbose=True):
    '''Query neighbors of a block of embeddings.'''

    args = get_retro_args()

    # Query neighbor ids.
    if verbose: print_rank_0("search.")
    t = time.time()
    assert index.ntotal > 0, "check we don't accidentally have an empty index."
    _, query_neighbor_ids = \
        index.search(embeddings, args.retro_query_num_neighbors_query)
    if verbose: print_rank_0("  time : %.3f sec." % (time.time() - t))

    # Filter banned neighbor ids.
    if verbose: print_rank_0("filter banned neighbor ids.")
    filtered_neighbor_ids = np.full(
        shape=(len(query_neighbor_ids), args.retro_query_num_neighbors_save),
        fill_value=-1,
        dtype="int64",
    )
    min_chunk_id, max_chunk_id = chunk_id_range
    for chunk_id in range(min_chunk_id, max_chunk_id):

        sample_id = chunk_id // n_chunks_per_sample
        sample = sample_map[sample_id]
        sample_dataset_idx = sample["dataset_idx"].item()
        sample_doc_ids = sample["doc_ids"].tolist()
        sample_doc_tuples = [(sample_dataset_idx, d) for d in sample_doc_ids]
        
        # Get valid neighbors (!= -1).
        query_row = [ i for i in query_neighbor_ids[chunk_id-min_chunk_id]
                      if i >= 0 ]

        # Filter row.
        filtered_row = [ i for i in query_row
                         if tuple(db_dataset.doc_tuples[i].tolist())
                         not in sample_doc_tuples ]
        filtered_row = filtered_row[:args.retro_query_num_neighbors_save]
        filtered_row += \
            [-1] * (args.retro_query_num_neighbors_save - len(filtered_row))
        filtered_neighbor_ids[chunk_id-min_chunk_id] = filtered_row

    return query_neighbor_ids, filtered_neighbor_ids


def query_embedding_block(db_dataset, index,
                          embeddings, chunk_id_range,
                          sample_map, n_chunks_per_sample):

    query_neighbor_ids = []
    filtered_neighbor_ids = []

    # Query in sub-blocks.
    partial_block_size = 1000
    for partial_start_idx in tqdm(
            range(0, len(embeddings), partial_block_size),
            "search",
    ):
        partial_end_idx = min(len(embeddings),
                              partial_start_idx + partial_block_size)
        partial_embeddings = embeddings[partial_start_idx:partial_end_idx]
        partial_chunk_id_range = (
            chunk_id_range[0] + partial_start_idx,
            chunk_id_range[0] + partial_end_idx,
        )
        partial_query_neighbor_ids, partial_filtered_neighbor_ids = \
            query_embeddings(db_dataset, index,
                             partial_embeddings, partial_chunk_id_range,
                             sample_map, n_chunks_per_sample,
                             verbose=False)
        query_neighbor_ids.append(partial_query_neighbor_ids)
        filtered_neighbor_ids.append(partial_filtered_neighbor_ids)

    # Concatenate.
    query_neighbor_ids = np.concatenate(query_neighbor_ids, axis=0)
    filtered_neighbor_ids = np.concatenate(filtered_neighbor_ids, axis=0)

    return query_neighbor_ids, filtered_neighbor_ids


def query_block_neighbors(db_dataset, query_dataset,
                          index, embedder,
                          block):
    '''Query neighbors of a dataset block (i.e., range).'''

    args = get_retro_args()
    n_chunks_per_sample = query_dataset.n_chunks_per_sample

    # Sample map.
    sample_ids = sorted(list(set(chunk_id // n_chunks_per_sample
                                 for chunk_id in range(*block["range"]))))
    sample_map = {}
    for i in sample_ids:
        sample = query_dataset.sample_dataset[i]
        sample_map[i] = {
            "dataset_idx" : sample["dataset_idx"],
            "doc_ids" : sample["doc_ids"],
        }

    # Embed block.
    embeddings = embed_block(query_dataset, block, embedder)

    # Query embeddings.
    _, filtered_neighbor_ids = query_embedding_block(
        db_dataset, index,
        embeddings, block["range"],
        sample_map, n_chunks_per_sample)

    # Save neighbors.
    print_rank_0("save neighbors.")
    os.makedirs(os.path.dirname(block["path"]), exist_ok=True)
    f = h5py.File(block["path"], "w")
    f.create_dataset("neighbors", data=filtered_neighbor_ids)
    f.close()


def query_dataset_neighbors(db_dataset, query_dataset,
                            prefix, neighbor_dir,
                            index, embedder):
    '''Query neighbors of each chunk within a dataset.'''

    args = get_retro_args()

    def validate(f):
        assert f["neighbors"].shape[1] == args.retro_query_num_neighbors_save, \
            "neighbors.shape == %s; num_neighbors_target == %d." % (
                str(f["neighbors"].shape),
                args.retro_num_neighbors_target,
            )
    n_missing_blocks, missing_neighbor_blocks = get_missing_blocks_by_rank(
        neighbor_dir,
        len(query_dataset),
        args.retro_block_size,
        validate=validate,
    )

    # Query each block.
    for block_index, block in enumerate(missing_neighbor_blocks):

        if block is not None:

            # Progress.
            print_rank_0("query '%s' block %d / %d ... %s ... mem %.3f gb, %.1f%%." % (
                prefix,
                block_index,
                len(missing_neighbor_blocks),
                os.path.basename(block["path"]),
                psutil.virtual_memory()[3] / 1024**3,
                psutil.virtual_memory()[2],
            ))

            # Query block neighbors.
            query_block_neighbors(db_dataset, query_dataset,
                                  index, embedder,
                                  block)

        # Synchronize progress across all ranks. (for easier observation)
        print_rank_0(" > waiting for other ranks to finish block.")
        torch.distributed.barrier()


def query_pretraining_neighbors():
    '''Query pretraining datasets (train & valid).'''

    args = get_retro_args()

    # Num threads.
    faiss.omp_set_num_threads(64)

    # Load chunk db dataset.
    print_rank_0("load chunk db dataset.")
    db_dataset = get_db_merged_train_dataset()
    db_dataset.load_doc_tuples()

    # Load index.
    print_rank_0(" > get index.")
    index = get_index()

    # Load datasets.
    print_rank_0(" > get dataset map.")
    query_dataset_map = get_query_dataset_map()

    # Bert embedder.
    embedder = BertEmbedder(args.retro_bert_batch_size,
                            args.retro_bert_max_chunk_length,
                            args.bert_embedder_type)

    # Query each (i.e., train, valid, test) dataset.
    print_rank_0(" > query.")
    for prefix, info in query_dataset_map.items():
        print_rank_0(" > query '%s' dataset ... %d samples." %
                     (prefix, len(info["data"])))
        query_dataset_neighbors(db_dataset, info["data"],
                                prefix, info["neighbor_dir"],
                                index, embedder)
