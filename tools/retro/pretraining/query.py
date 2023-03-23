# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

from collections import defaultdict
import numpy as np
import os
import time
import torch
from tqdm import tqdm

from megatron import get_retro_args, mpu, print_rank_0
from tools.bert_embedding import BertEmbedder
from tools.bert_embedding.utils import get_missing_blocks_by_rank
from tools.retro.db.utils import (
    get_merged_train_dataset as get_db_merged_train_dataset,
    get_train_doc_chunk_map,
)
from tools.retro.external_libs import faiss, h5py
from tools.retro.index.factory import IndexFactory
from tools.retro.index.utils import get_index_dir, num_samples_to_block_ranges
from tools.retro.utils import GPTToTextDataset

from .chunk_dataset import get_chunk_dataset_map


def get_index(chunk_db_dataset, ondisk=False):
    '''Read index from disk.'''

    args = get_retro_args()

    # Chunk db block ranges.
    n_db_chunks = len(chunk_db_dataset)
    dataset_block_ranges = num_samples_to_block_ranges(n_db_chunks)

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
                                               args.retro_ef_search)
    faiss.ParameterSpace().set_index_parameter(index, "nprobe",
                                               args.retro_nprobe)

    return index


def embed_block(gpt_dataset, block, embedder):
    '''Embed block of chunks.'''
    text_block_dataset = torch.utils.data.Subset(
        GPTToTextDataset(gpt_dataset),
        range(*block["range"]),
    )
    return embedder.embed_text_dataset(text_block_dataset)


def query_embeddings(index, banned_chunk_map, chunk_id_range,
                     embeddings, sample_map, n_chunks_per_sample,
                     verbose=True):
    '''Query neighbors of a block of embeddings.'''

    args = get_retro_args()

    # Query neighbor ids.
    if verbose: print_rank_0("search.")
    t = time.time()
    assert index.ntotal > 0, "check we don't accidentally have an empty index."
    _, query_neighbor_ids = \
        index.search(embeddings, args.retro_num_neighbors_query)
    if verbose: print_rank_0("  time : %.3f sec." % (time.time() - t))

    # Banned neighbor ids.
    if verbose: print_rank_0("get banned neighbor ids.")
    sample_banned_chunk_id_map = {}
    for sample_id, sample in sample_map.items():
        dataset_idx = sample["dataset_idx"].item()
        doc_ids = sample["doc_ids"].tolist()
        banned_chunk_ids = set()
        for doc_id in doc_ids:
            banned_chunk_ids.update(banned_chunk_map[(dataset_idx, doc_id)])
        sample_banned_chunk_id_map[sample_id] = banned_chunk_ids

    # Filter banned neighbor ids.
    if verbose: print_rank_0("filter banned neighbor ids.")
    filtered_neighbor_ids = np.full(
        shape=(len(query_neighbor_ids), args.retro_num_neighbors_target),
        fill_value=-1,
        dtype="int64",
    )
    min_chunk_id, max_chunk_id = chunk_id_range
    for chunk_id in range(min_chunk_id, max_chunk_id):

        sample_id = chunk_id // n_chunks_per_sample

        # Get valid neighbors (!= -1).
        query_row = [ i for i in query_neighbor_ids[chunk_id-min_chunk_id]
                      if i >= 0 ]

        # Filter row.
        filtered_row = [i for i in query_row
                        if i not in sample_banned_chunk_id_map[sample_id]]
        filtered_row = filtered_row[:args.retro_num_neighbors_target]
        filtered_row += \
            [-1] * (args.retro_num_neighbors_target - len(filtered_row))
        filtered_neighbor_ids[chunk_id-min_chunk_id] = filtered_row

    return query_neighbor_ids, filtered_neighbor_ids


def query_embedding_block(index, banned_chunk_map, chunk_id_range,
                          embeddings, sample_map, n_chunks_per_sample):

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
            query_embeddings(index, banned_chunk_map, partial_chunk_id_range,
                             partial_embeddings, sample_map, n_chunks_per_sample,
                             verbose=False)
        query_neighbor_ids.append(partial_query_neighbor_ids)
        filtered_neighbor_ids.append(partial_filtered_neighbor_ids)

    # Concatenate.
    query_neighbor_ids = np.concatenate(query_neighbor_ids, axis=0)
    filtered_neighbor_ids = np.concatenate(filtered_neighbor_ids, axis=0)

    return query_neighbor_ids, filtered_neighbor_ids


def query_block_neighbors(index, banned_chunk_map, chunk_dataset,
                          block, embedder):
    '''Query neighbors of a dataset block (i.e., range).'''

    args = get_retro_args()
    n_chunks_per_sample = chunk_dataset.n_chunks_per_sample

    # Sample map.
    sample_ids = sorted(list(set(chunk_id // n_chunks_per_sample
                                 for chunk_id in range(*block["range"]))))
    sample_map = {i:chunk_dataset.sample_dataset[i] for i in sample_ids}

    # Embed block.
    embeddings = embed_block(chunk_dataset, block, embedder)

    # Query embeddings.
    _, filtered_neighbor_ids = query_embedding_block(
        index, banned_chunk_map, block["range"],
        embeddings, sample_map,
        n_chunks_per_sample)

    # Save neighbors.
    print_rank_0("save neighbors.")
    os.makedirs(os.path.dirname(block["path"]), exist_ok=True)
    f = h5py.File(block["path"], "w")
    f.create_dataset("neighbors", data=filtered_neighbor_ids)
    f.close()


def query_dataset_neighbors(index, banned_chunk_map,
                            prefix, chunk_dataset, neighbor_dir,
                            embedder):
    '''Query neighbors of each chunk within a dataset.'''

    args = get_retro_args()

    def validate(f):
        assert f["neighbors"].shape[1] == args.retro_num_neighbors_target, \
            "neighbors.shape == %s; num_neighbors_target == %d." % (
                str(f["neighbors"].shape),
                args.retro_num_neighbors_target,
            )
    n_missing_blocks, missing_neighbor_blocks = get_missing_blocks_by_rank(
        neighbor_dir,
        len(chunk_dataset),
        args.retro_block_size,
        validate=validate,
    )

    # Query each block.
    for block_index, block in enumerate(missing_neighbor_blocks):

        if block is not None:

            # Progress.
            print_rank_0("query '%s' block %d / %d ... %s." % (
                prefix,
                block_index,
                len(missing_neighbor_blocks),
                block["path"],
            ))

            # Query block neighbors.
            query_block_neighbors(index, banned_chunk_map,
                                  chunk_dataset, block, embedder)

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
    chunk_db_dataset = get_db_merged_train_dataset()

    # Load index, banned chunk ids, datasets.
    print_rank_0(" > get index.")
    index = get_index(chunk_db_dataset)

    print_rank_0(" > get banned doc-chunk id map.")
    banned_chunk_map = get_train_doc_chunk_map()

    print_rank_0(" > get dataset map.")
    chunk_dataset_map = get_chunk_dataset_map()

    # Bert embedder.
    embedder = BertEmbedder(args.retro_bert_batch_size,
                            args.retro_bert_max_chunk_length,
                            args.bert_embedder_type)

    # Query each (i.e., train, valid, test) dataset.
    print_rank_0(" > query.")
    for prefix, info in chunk_dataset_map.items():
        print_rank_0(" > query '%s' dataset ... %d samples." %
                     (prefix, len(info["data"])))
        query_dataset_neighbors(index, banned_chunk_map,
                                prefix, info["data"], info["neighbor_dir"],
                                embedder)
