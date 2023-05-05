# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

from collections import defaultdict
from concurrent.futures import as_completed, ProcessPoolExecutor
from functools import reduce
import glob
import json
import numpy as np
import os
from pathlib import Path
import threading
import torch
from tqdm import tqdm
import types

from megatron import get_retro_args, print_rank_0
from megatron.data.indexed_dataset import make_dataset as make_indexed_dataset
from megatron.tokenizer.tokenizer import (
    _BertWordPieceTokenizer,
    _GPT2BPETokenizer,
)
from tools.bert_embedding.utils import get_missing_blocks_by_rank
from tools.retro.external_libs import h5py
from tools.retro.utils import get_gpt_tokenizer, get_bert_tokenizer

from .utils import (
    get_individual_db,
    get_individual_db_dir,
    get_merged_dataset,
    get_merged_db_path_map,
    get_train_doc_chunk_map_dir,
    save_indexed_dataset_infos,
)


def init_indexed_dataset_infos():
    '''Gather meta-info about each indexed dataset.

    The returned info array allows for easy access to the configuration, and
    helps remove ambiguity.
    '''

    args = get_retro_args()

    assert len(args.data_path) % 2 == 0, \
        "currently, only blendable dataset is supported."

    # Dataset infos.
    infos = []
    for i in range(0, len(args.data_path), 2):
        ratio = float(args.data_path[i])
        prefix = args.data_path[i + 1]
        path = prefix + ".bin"
        name = os.path.basename(prefix)
        assert os.path.exists(path)
        infos.append({
            "ratio" : ratio,
            "prefix" : prefix,
            "path" : path,
            "name" : name,
            "db_dir" : get_individual_db_dir(name),
            "dataset" : make_indexed_dataset(prefix, "mmap", True),
        })

    return infos


def build_partial_db(
        dataset_idx,
        n_datasets,
        indexed_dataset,
        block_id,
        n_blocks,
        block,
        proc_id,
        n_procs,
        tokenizers,
):
    '''Process a document index range of the indexed dataset.

    The chunk database is built in parallel blocks, since de-tokenizing &
    re-tokenizing for Bert-length computation is expensive. This method
    iterates each document and extracts sequential 'chunk-length' sequences
    from each document.
    '''

    args = get_retro_args()

    # Document start/end indexes.
    doc_range = block["range"]
    n_docs = doc_range[1] - doc_range[0]
    n_docs_per_proc = int(np.ceil(n_docs / n_procs))
    doc_start_id = doc_range[0] + proc_id * n_docs_per_proc
    doc_end_id = min(doc_range[1], doc_start_id + n_docs_per_proc)

    # Print progress.
    progress_proc_ids = set(range(n_procs)) \
        if torch.distributed.get_rank() == 0 else set()
    if proc_id in progress_proc_ids:
        print(" > building partial chunk db, proc %d / %d, docs %d:%d / %d."%(
            proc_id,
            n_procs,
            doc_start_id,
            doc_end_id,
            n_docs,
        ))

    # Progress bars (snapshot of overall progress).
    doc_id_iter = range(doc_start_id, doc_end_id)
    pbar = tqdm(doc_id_iter) \
        if proc_id in progress_proc_ids else \
           doc_id_iter

    # Iterate documents & parse chunks.
    chunk_db_valid = []
    chunk_db_invalid = []
    for doc_id in pbar:

        # Progress description.
        try:
            pbar.set_description("ds %d / %d, block %d / %d, proc %d / %d." % (
                dataset_idx,
                n_datasets,
                block_id,
                n_blocks,
                proc_id,
                n_procs))
        except:
            pass

        # Remove EOD token.
        doc = indexed_dataset.get(doc_id)
        if doc[-1].item() == tokenizers.gpt.eod_id:
            doc = doc[:-1]
        doc_len = len(doc)

        # Chunk start/end indexes.
        chunk_start_idxs = list(range(0, doc_len, args.retro_gpt_chunk_length))
        chunk_end_idxs = [min(doc_len, s + args.retro_gpt_chunk_length)
                          for s in chunk_start_idxs]

        # Re-tokenize each chunk to Bert/Wordpiece (empty bert -> 'invalid').
        for i, chunk_start_idx in enumerate(chunk_start_idxs):

            # Re-tokenize.
            chunk_end_idx = chunk_end_idxs[i]
            gpt_token_ids = indexed_dataset.get(
                idx=doc_id,
                offset=chunk_start_idx,
                length=chunk_end_idx - chunk_start_idx,
            )
            text = tokenizers.gpt.detokenize(gpt_token_ids)
            bert_token_ids = tokenizers.bert.tokenize(text)

            # 'Valid' for non-empty Bert chunks; 'invalid' otherwise.
            _chunk_db = chunk_db_invalid \
                if len(bert_token_ids) == 0 else \
                   chunk_db_valid
            _chunk_db.append((
                doc_id,
                chunk_start_idx,
                chunk_end_idx,
                len(bert_token_ids),
            ))

    return proc_id, chunk_db_valid, chunk_db_invalid


def build_individual_db(dataset_idx, n_datasets, dataset_info, tokenizers):
    '''Process a single indexed dataset & extract chunks.'''

    args = get_retro_args()

    # Make directory.
    db_dir = dataset_info["db_dir"]
    os.makedirs(db_dir, exist_ok=True)

    # Indexed dataset.
    indexed_dataset = dataset_info["dataset"]

    # Missing db blocks.
    n_missing_world, missing_db_blocks = get_missing_blocks_by_rank(
        db_dir,
        len(indexed_dataset.doc_idx) - 1,
        args.retro_doc_block_size,
        validate=lambda f : f["chunks_valid"].shape[1] == 4)

    # Prevent missing-path-write race condition.
    torch.distributed.barrier()

    if not missing_db_blocks:
        return

    # Num processes.
    if n_missing_world == 1:
        n_procs = 128
    elif n_missing_world <= 2:
        n_procs = 64
    elif n_missing_world <= 4:
        n_procs = 32
    elif n_missing_world <= 8:
        n_procs = 16
    else:
        n_procs = 8

    # Process documents in parallel.
    with ProcessPoolExecutor(max_workers=n_procs) as executor:
        for block_idx, block in enumerate(missing_db_blocks):

            if block is not None:

                # Build partial dbs.
                print_rank_0(' > build partial dbs.')
                futures = []
                for proc_id in range(n_procs): # not true process id
                    futures.append(executor.submit(
                        build_partial_db,
                        dataset_idx,
                        n_datasets,
                        indexed_dataset,
                        block_idx,
                        len(missing_db_blocks),
                        block,
                        proc_id,
                        n_procs,
                        tokenizers,
                    ))
                partial_chunk_dbs = []
                for future in as_completed(futures):
                    partial_chunk_dbs.append(future.result())

                # Concatenate chunks.
                partial_chunk_dbs.sort(key=lambda item:item[0]) # sort by proc_id
                chunk_db_valid = [item
                                  for partial_chunk_db in partial_chunk_dbs
                                  for item in partial_chunk_db[1]]
                chunk_db_invalid = [item
                                    for partial_chunk_db in partial_chunk_dbs
                                    for item in partial_chunk_db[2]]

                # Convert to numpy.
                print_rank_0(' > converting chunk db to numpy.')
                chunk_db_valid = np.array(chunk_db_valid)
                chunk_db_invalid = np.array(chunk_db_invalid)

                # Save DB.
                print_rank_0(" > saving individual db.")
                f = h5py.File(block["path"], "w")
                dset = f.create_dataset("chunks_valid", data=chunk_db_valid)
                dset = f.create_dataset("chunks_invalid", data=chunk_db_invalid)
                f.close()

            # Wait for all ranks to finish block.
            print_rank_0(" > waiting for all ranks to finish block.")
            torch.distributed.barrier()

    print_rank_0(" > finished saving individual db.")


def build_individual_dbs(indexed_dataset_infos):
    '''Iterate each indexed dataset & process its chunks.'''

    args = get_retro_args()

    # Tokenizers.
    tokenizers = types.SimpleNamespace(
        gpt=get_gpt_tokenizer(),
        bert=get_bert_tokenizer(),
    )

    # Build individual DBs.
    print_rank_0(" > build individual chunk dbs.")
    for ds_idx, ds_info in enumerate(indexed_dataset_infos):

        # Progress.
        print_rank_0(" > building individual db, dataset %d / %d ... '%s'." % (
            ds_idx,
            len(indexed_dataset_infos),
            ds_info["name"],
        ))

        # Process single dataset.
        build_individual_db(ds_idx, len(indexed_dataset_infos),
                            ds_info, tokenizers)


def update_chunk_counts(indexed_dataset_infos):
    '''Set n_chunks_train & n_chunks sampled for each individual DB.'''

    args = get_retro_args()

    if torch.distributed.get_rank() != 0:
        return

    # Training split size (split at document level).
    train_fraction = float(args.split.split(",")[0]) / 100
    assert train_fraction > 0 and train_fraction <= 1

    # Set n_chunks (including n_chunks_sampled for unambiguity).
    print_rank_0(" > compute n_chunks.")
    for ds_index, ds_info in \
        enumerate(tqdm(indexed_dataset_infos, "count_chunks")):

        db_dir = ds_info["db_dir"]
        db_paths = sorted(glob.glob(db_dir + "/*.hdf5"))

        # Update counts.
        ds_info["n_docs"] = len(ds_info["dataset"].doc_idx) - 1
        ds_info["n_docs_train"] = int(train_fraction * ds_info["n_docs"])
        ds_info["n_chunks"] = 0 # previously, 'n_chunks_valid'
        ds_info["n_chunks_train"] = 0
        ds_info["n_chunks_invalid"] = 0
        for db_path in db_paths:
            with h5py.File(db_path, "r") as f:
                ds_info["n_chunks"] += len(f["chunks_valid"])
                ds_info["n_chunks_invalid"] += len(f["chunks_invalid"])
                ds_info["n_chunks_train"] += \
                    (np.copy(f["chunks_valid"][:, 0]) < ds_info["n_docs_train"]) \
                    .sum().item()

        ds_info["n_chunks_sampled"] = \
            int(round(args.retro_nchunks_sampled * ds_info["ratio"]))

        # Verify counts.
        assert ds_info["n_chunks_train"] <= ds_info["n_chunks"], \
            "n_train (%d) > n_total (%d)." % (
                ds_info["n_chunks_train"], ds_info["n_chunks"])
        assert ds_info["n_chunks_sampled"] <= ds_info["n_chunks_train"], \
            "n_sampled (%d) > n_train (%d)." % (
                ds_info["n_chunks_sampled"], ds_info["n_chunks_train"])


def merge_dbs(indexed_dataset_infos, db_type):
    '''Merge individual DBs into single DB.'''

    if torch.distributed.get_rank() != 0:
        return

    print(" > build %s chunk db." % db_type)

    # Count chunks.
    if db_type == "full":
        raise Exception("deprecated; use 'train' or 'sampled'.")
        n_chunks_key = "n_chunks"
    elif db_type == "sampled":
        n_chunks_key = "n_chunks_sampled"
    elif db_type == "train":
        n_chunks_key = "n_chunks_train"
    elif db_type == "valid":
        pass
    else:
        raise Exception("handle db_type '%s'." % db_type)

    if db_type == "valid":
        n_chunks = sum(m["n_chunks"] - m["n_chunks_train"]
                       for m in indexed_dataset_infos)
    else:
        n_chunks = sum(m[n_chunks_key] for m in indexed_dataset_infos)

    # DB path.
    db_path = get_merged_db_path_map()[db_type]

    # Delete existing chunk db if incorrect size.
    if os.path.exists(db_path):

        try:

            f = h5py.File(db_path)
            n_alloc = len(f["chunks"])           # total allocated
            n_written = f["n_written"][0].item() # total written
            f.close()

            if n_chunks != n_alloc or n_chunks != n_written:
                os.remove(db_path)

        except Exception as e:
            if isinstance(e, OSError):
                os.remove(full_db_path)
            elif isinstance(e, KeyError):
                f.close()
                os.remove(full_db_path)
            else:
                raise e

    # Build merged chunk db.
    if not os.path.exists(db_path):

        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        f = h5py.File(db_path, "w")

        # Initialize output arrays.
        merged_db = f.create_dataset("chunks", (n_chunks, 5), dtype="i8")
        n_written = f.create_dataset("n_written", (1,), dtype="uint64")
        n_written[0] = 0

        # Iterate indexed datasets & collect chunks.
        start_index = 0
        for ds_idx, ds_info in enumerate(indexed_dataset_infos):
            print(" > merging dbs; '%s', dataset %d / %d ... '%s'." %
                  (db_type, ds_idx, len(indexed_dataset_infos), ds_info["name"]))
            individual_db = get_individual_db(ds_idx, ds_info)

            if db_type == "valid":
                individual_db = individual_db[ds_info["n_chunks_train"]:]
            else:
                individual_db = individual_db[:ds_info[n_chunks_key]]

            merged_db[start_index:start_index+len(individual_db)] = individual_db
            start_index += len(individual_db)
            n_written[0] = start_index

        f.close()


def get_partial_banned_chunk_map(proc_id, db_path, chunk_range_info):
    '''Build partial mapping of {(dataset_id,doc_id):[chunk_ids]}.

    In this method, only chunks within the range (start_chunk_id, end_chunk_id]
    are processed.'''

    start_chunk_id = chunk_range_info["start"]
    end_chunk_id = chunk_range_info["end"]
    output_path = chunk_range_info["path"]

    # Skip, if output file exists.
    if os.path.exists(output_path):
        return

    # Chunk subset.
    with h5py.File(db_path) as f:
        sub_chunk_db = np.copy(f["chunks"][start_chunk_id:end_chunk_id, :2])

    # Map docs to chunks.
    banned_chunk_map = defaultdict(list)
    for rel_chunk_id, (dataset_id, doc_id) in enumerate(tqdm(
            sub_chunk_db,
            "map banned docs, proc %d" % proc_id,
            total=sub_chunk_db.shape[0],
    )):
        chunk_id = start_chunk_id + rel_chunk_id
        banned_chunk_map["%d,%d" % (dataset_id.item(), doc_id.item())] \
            .append(chunk_id)

    # Save output.
    with open(output_path, "w") as f:
        json.dump(banned_chunk_map, f)


def build_doc_chunk_map(indexed_dataset_infos, db_type):
    '''Build mapping of {(dataset_id,doc_id):[chunk_ids]}.'''

    if torch.distributed.get_rank() != 0:
        return

    print(" > build %s doc-chunk map." % db_type)

    n_procs = 128

    # Get dataset.
    db_dataset = get_merged_dataset(db_type, indexed_dataset_infos)

    # Sub-ranges for parallel processing.
    n_chunks = db_dataset.chunks.shape[0]
    n_chunks_per_proc = max(1, int(np.ceil(n_chunks / n_procs)))
    chunk_id_starts = list(range(0, n_chunks, n_chunks_per_proc))
    chunk_id_ranges = [(s, min(n_chunks, s + n_chunks_per_proc))
                       for s in chunk_id_starts]

    # Wrap range info with output path.
    n_digits = int(np.ceil(np.log(n_chunks) / np.log(10)) + 1)
    output_dirname = get_train_doc_chunk_map_dir()
    chunk_range_infos = [{
        "start" : start_id,
        "end" : end_id,
        "path" : os.path.join(output_dirname, "%s-%s.json" % (
            str(start_id).zfill(n_digits),
            str(end_id).zfill(n_digits),
        )),
    } for start_id, end_id in chunk_id_ranges ]

    # Build doc-chunk map.
    print_rank_0("build doc-chunk-map.")
    with ProcessPoolExecutor(max_workers=n_procs) as executor:

        # Build partial chunk maps.
        futures = []
        for proc_id, chunk_range_info in enumerate(chunk_range_infos):

            if os.path.exists(chunk_range_info["path"]):
                continue

            # Submit job.
            futures.append(executor.submit(
                get_partial_banned_chunk_map,
                proc_id,
                db_dataset.db_path,
                chunk_range_info,
            ))

        # Wait for processes to finish.
        banned_chunk_paths = []
        for finished_idx, future in enumerate(as_completed(futures)):
            print("finished %d / %d." % (finished_idx, n_procs))
            future.result()


def build_db():
    '''Extract token chunks from each indexed dataset.

    Iterate each document of each indexed dataset, extract that document's
    chunks, and save to a 'DB' (hdf5 file).
    '''

    # Indexed dataset info.
    indexed_dataset_infos = init_indexed_dataset_infos()

    # Build dbs.
    build_individual_dbs(indexed_dataset_infos)

    # Single-process going forward.
    if torch.distributed.get_rank() != 0:
        return

    # Update n_chunks.
    update_chunk_counts(indexed_dataset_infos)

    # Merge dbs.
    merge_dbs(indexed_dataset_infos, "sampled")
    merge_dbs(indexed_dataset_infos, "train")
    merge_dbs(indexed_dataset_infos, "valid")
    build_doc_chunk_map(indexed_dataset_infos, "train")

    # Save (fully annotated) indexed dataset infos.
    save_indexed_dataset_infos(indexed_dataset_infos)
