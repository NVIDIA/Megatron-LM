# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.

"""Build a chunk database from a list of indexed datasets.

Building a chunk database consists of.

  - Breaking each document of each indexed dataset into consecutive
      retro_gpt_chunk_length chunks.
  - Re-tokenize each chunk into Bert, and discard any chunks with empty Bert
      tokens.
  - Save chunk offsets to disk for each indexed dataset.
"""

import glob
import os
import types
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from megatron.core.datasets.indexed_dataset import IndexedDataset
from megatron.core.datasets.retro.config import RetroPreprocessingConfig
from megatron.core.datasets.retro.external_libs import h5py
from megatron.core.datasets.retro.utils import (
    extract_data_config,
    get_blocks_by_rank,
    log_retro_rank_0,
    retro_makedir,
)

from .utils import (
    get_indexed_dataset_infos,
    get_indexed_dataset_infos_path,
    get_individual_chunk_db,
    get_individual_db_dir,
    get_individual_db_paths,
    get_individual_doc_offsets,
    get_merged_db_path_map,
    init_indexed_dataset_infos,
    load_indexed_datasets,
    save_indexed_dataset_infos,
)


def build_partial_db(
    config: types.SimpleNamespace,
    dataset_idx: int,
    n_datasets: int,
    indexed_dataset: IndexedDataset,
    block_id: int,
    n_blocks: int,
    block: dict,
    proc_id: int,
    n_procs: int,
) -> Tuple[int, list, list, dict]:
    """Process a document index range of the indexed dataset.

    The chunk database is built in parallel blocks, since de-tokenizing &
    re-tokenizing for Bert-length computation is expensive. This method
    iterates each document and extracts sequential 'chunk-length' sequences
    from each document.

    Args:
        config (types.SimpleNamespace): Subset of Retro config, containing 'chunk_length', 'gpt_eod', 'gpt_detokenize', 'bert_tokenize', and 'task_validate'.
        dataset_idx (int): Index of this dataset out of all blended datasets.
        n_datasets (int): Total number of blended datasets.
        indexed_dataset (IndexedDataset): Indexed dataset to be chunked.
        block_id (int): Block index out of all blocks to be processed.
        n_blocks (int):  Total number of blocks to be processed.
        block (dict): Range information such as start/end points for chunking idnexed dataset.
        proc_id (int): Process ID for tracking parallel process order.
        n_procs (int): Total number of parallel processes.

    Returns:
        A tuple containing:

        - Process ID.
        - List of valid chunks.
        - List of invalid chunks (i.e., chunks that converted to empty Bert embeddings.).
        - Dict mapping document ID to number of valid chunks.
    """

    # Document start/end indexes.
    doc_range = block["range"]
    n_docs = doc_range[1] - doc_range[0]
    n_docs_per_proc = int(np.ceil(n_docs / n_procs))
    doc_start_id = doc_range[0] + proc_id * n_docs_per_proc
    doc_end_id = min(doc_range[1], doc_start_id + n_docs_per_proc)

    # Print progress.
    progress_proc_ids = set(range(n_procs)) if torch.distributed.get_rank() == 0 else set()
    if proc_id in progress_proc_ids:
        log_retro_rank_0(
            " > building partial chunk db, proc %d / %d, docs %d:%d / %d."
            % (
                proc_id,
                n_procs,
                doc_start_id,
                doc_end_id,
                n_docs,
            )
        )

    # Progress bars (snapshot of overall progress).
    doc_id_iter = range(doc_start_id, doc_end_id)
    pbar = (
        tqdm(
            doc_id_iter,
            "parse doc chunks",
            miniters=len(doc_id_iter) // 20,
        )
        if proc_id in progress_proc_ids
        else doc_id_iter
    )

    # Iterate documents & parse chunks.
    chunk_db_valid: List[Tuple] = []
    chunk_db_invalid: List[Tuple] = []
    doc_size_map = {}
    for doc_id in pbar:

        # Progress description.
        try:
            pbar.set_description(
                "%sds %d / %d, block %d / %d, proc %d / %d."
                % (
                    "" if config.task_validate is None else "[validate] ",
                    dataset_idx,
                    n_datasets,
                    block_id,
                    n_blocks,
                    proc_id,
                    n_procs,
                )
            )
        except Exception:
            pass

        # Remove EOD token.
        doc = indexed_dataset.get(doc_id)
        if doc[-1].item() == config.gpt_eod:
            doc = doc[:-1]
        doc_len = len(doc)

        # Chunk start/end indexes.
        chunk_start_idxs = list(range(0, doc_len, config.chunk_length))
        chunk_end_idxs = [min(doc_len, s + config.chunk_length) for s in chunk_start_idxs]

        # Re-tokenize each chunk to Bert/Wordpiece (empty bert -> 'invalid').
        doc_size_map[doc_id] = 0
        for i, chunk_start_idx in enumerate(chunk_start_idxs):

            # Re-tokenize.
            chunk_end_idx = chunk_end_idxs[i]
            gpt_token_ids = indexed_dataset.get(
                idx=doc_id,
                offset=chunk_start_idx,
                length=chunk_end_idx - chunk_start_idx,
            )
            text = config.gpt_detokenize(gpt_token_ids.tolist())
            bert_token_ids = config.bert_tokenize(text)

            # 'Valid' for non-empty Bert chunks; 'invalid' otherwise.
            if len(bert_token_ids) == 0:
                _chunk_db = chunk_db_invalid
            else:
                _chunk_db = chunk_db_valid
                doc_size_map[doc_id] += 1
            _chunk_db.append(
                (
                    doc_id,
                    chunk_start_idx,
                    chunk_end_idx,
                    len(bert_token_ids),
                )
            )

    return proc_id, chunk_db_valid, chunk_db_invalid, doc_size_map


def build_block_db(
    config: RetroPreprocessingConfig,
    dataset_idx: int,
    n_datasets: int,
    indexed_dataset: IndexedDataset,
    n_procs: int,
    executor: ProcessPoolExecutor,
    n_missing_blocks: int,
    block_idx: int,
    block: dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split each document within block into consecutive retro_gpt_chunk_length size chunks.

    Args:
        config (RetroPreprocessingConfig): For DB building, we make use of attributes 'chunk_length', 'gpt_eod', 'gpt_detokenize', 'bert_tokenize', and 'task_validate'.
        dataset_idx (int): Index of this dataset out of all blended datasets.
        n_datasets (int): Total number of blended datasets.
        indexed_dataset (IndexedDataset): Indexed dataset to be chunked.
        n_procs (int): Total number of parallel processes.
        executor (ProcessPoolExecutor): Executor for launching parallel processes.
        n_missing_blocks (int):  Total number of blocks to be processed.
        block_idx (int): Block index out of all blocks to be processed.
        block (dict): Range information such as start/end points for chunking idnexed dataset.

    Returns:
        A tuple containing:

        - List of valid chunks.
        - List of invalid chunks (i.e., chunks that converted to empty Bert embeddings.).
        - Dict mapping document ID to number of valid chunks.
    """

    # Build partial dbs.
    log_retro_rank_0(' > build partial dbs.')
    futures = []
    for proc_id in range(n_procs):  # not true process id
        futures.append(
            executor.submit(
                build_partial_db,
                types.SimpleNamespace(
                    chunk_length=config.retro_gpt_chunk_length,
                    gpt_eod=config.retro_tokenizers.gpt.eod,
                    gpt_detokenize=config.retro_tokenizers.gpt.detokenize,
                    bert_tokenize=config.retro_tokenizers.bert.tokenize,
                    task_validate=config.retro_task_validate,
                ),
                dataset_idx,
                n_datasets,
                indexed_dataset,
                block_idx,
                n_missing_blocks,
                block,
                proc_id,
                n_procs,
            )
        )
    partial_chunk_dbs = []
    for future in as_completed(futures):
        partial_chunk_dbs.append(future.result())

    # Concatenate chunks.
    partial_chunk_dbs.sort(key=lambda item: item[0])  # sort by proc_id
    chunk_db_valid = [
        item for partial_chunk_db in partial_chunk_dbs for item in partial_chunk_db[1]
    ]
    chunk_db_invalid = [
        item for partial_chunk_db in partial_chunk_dbs for item in partial_chunk_db[2]
    ]

    # Convert to numpy.
    log_retro_rank_0(' > converting chunk db to numpy.')
    chunk_db_valid = np.array(chunk_db_valid, dtype="uint32")
    chunk_db_invalid = np.array(chunk_db_invalid, dtype="uint32")

    # Document offsets.
    doc_sizes = [
        (d, s) for partial_chunk_db in partial_chunk_dbs for d, s in partial_chunk_db[3].items()
    ]
    doc_sizes.sort(key=lambda item: item[0])
    doc_offsets = np.cumsum([item[1] for item in doc_sizes]).astype("uint64")
    doc_offsets = np.stack(
        (np.array([item[0] for item in doc_sizes], dtype="uint64"), doc_offsets), axis=1
    )

    return chunk_db_valid, chunk_db_invalid, doc_offsets


def save_block_db(
    block: dict,
    chunk_db_valid: np.ndarray,
    chunk_db_invalid: np.ndarray,
    doc_offsets: np.ndarray,
) -> None:
    """Save block of chunked tokens to disk. These blocks are later used for
    training and adding to the vector index.

    Args:
        block (dict): Range information such as start/end points for chunking idnexed dataset.
        chunk_db_valid (np.ndarray): Array of valid chunk indexes.
        chunk_db_invalid (np.ndarray): Array of invalid chunk indexes.
        doc_offsets (np.ndarray): Array of document offsets by chunks.
    """
    log_retro_rank_0(" > saving individual db.")
    with h5py.File(block["path"], "w") as f:
        dset = f.create_dataset("chunks_valid", data=chunk_db_valid)
        dset = f.create_dataset("chunks_invalid", data=chunk_db_invalid)
        dset = f.create_dataset("doc_offsets", data=doc_offsets)


def build_individual_db(
    config: RetroPreprocessingConfig,
    dataset_idx: int,
    n_datasets: int,
    dataset_info: dict,
) -> None:
    """Process a single indexed dataset & extract chunks.

    Args:
        config (RetroPreprocessingConfig): Retro preprocessing config.
        dataset_idx (int): Dataset index within blended dataset.
        n_datasets (int): Total number of datasets within blended dataset.
        dataset_info (dict): Metadata for dataset (see `save_indexed_dataset_infos()` in `utils.py` for more detail).
    """

    # Make directory.
    db_dir = get_individual_db_dir(config.retro_project_dir, dataset_info["prefix"])
    retro_makedir(config, db_dir)

    # Indexed dataset.
    indexed_dataset = dataset_info["dataset"]

    # Missing DB blocks (split by documents).
    blocks = get_blocks_by_rank(
        db_dir,
        len(indexed_dataset),
        config.retro_doc_block_size,
        validate=lambda f: f["chunks_valid"].shape == (0,) or f["chunks_valid"].shape[1] == 4,
        sample=config.retro_task_validate,
    )
    if config.retro_task_validate is None:
        active_blocks = blocks.missing
    else:
        assert blocks.n_missing_world == 0
        active_blocks = blocks.existing

    # Prevent missing-path-write race condition.
    torch.distributed.barrier()

    # Nothing to do?
    if config.retro_task_validate is None and not active_blocks:
        return

    # Num processes.
    if blocks.n_missing_world == 1:
        n_procs = 128
    elif blocks.n_missing_world <= 2:
        n_procs = 64
    elif blocks.n_missing_world <= 4:
        n_procs = 32
    elif blocks.n_missing_world <= 8:
        n_procs = 16
    else:
        n_procs = 8

    # Process documents in parallel.
    with ProcessPoolExecutor(max_workers=n_procs) as executor:
        for block_idx, block in enumerate(active_blocks):

            if block is not None:

                # Build block DB.
                chunk_db_valid, chunk_db_invalid, doc_offsets = build_block_db(
                    config=config,
                    dataset_idx=dataset_idx,
                    n_datasets=n_datasets,
                    indexed_dataset=indexed_dataset,
                    n_procs=n_procs,
                    executor=executor,
                    n_missing_blocks=len(active_blocks),
                    block_idx=block_idx,
                    block=block,
                )

                if config.retro_task_validate is None:
                    # Save block DB.
                    save_block_db(
                        block=block,
                        chunk_db_valid=chunk_db_valid,
                        chunk_db_invalid=chunk_db_invalid,
                        doc_offsets=doc_offsets,
                    )

                else:

                    # Load existing block DB.
                    with h5py.File(block["path"]) as f:
                        existing_chunks_valid = np.copy(f["chunks_valid"])
                        existing_chunks_invalid = np.copy(f["chunks_invalid"])
                        existing_doc_offsets = np.copy(f["doc_offsets"])

                    # Check equality.
                    log_retro_rank_0(" > validate.")
                    assert np.array_equal(existing_chunks_valid, chunk_db_valid)
                    assert np.array_equal(existing_chunks_invalid, chunk_db_invalid)
                    assert np.array_equal(existing_doc_offsets, doc_offsets)

            # Wait for all ranks to finish block.
            log_retro_rank_0(" > waiting for all ranks to finish block.")
            torch.distributed.barrier()

    log_retro_rank_0(" > finished saving individual db.")


def build_individual_dbs(
    config: RetroPreprocessingConfig,
    indexed_dataset_infos: List[Dict],
) -> None:
    """Iterate each indexed dataset & process its chunks.

    Args:
        config (RetroPreprocessingConfig): Retro preprocessing config.
        indexed_dataset_infos (List[Dict]): Preprocessing metadata for each dataset.
    """

    # Build individual DBs.
    log_retro_rank_0(" > build individual chunk dbs.")
    for ds_idx, ds_info in enumerate(indexed_dataset_infos):

        # Progress.
        log_retro_rank_0(
            " > building individual db, dataset %d / %d ... '%s'."
            % (
                ds_idx,
                len(indexed_dataset_infos),
                ds_info["prefix"],
            )
        )

        # Process single dataset.
        build_individual_db(config, ds_idx, len(indexed_dataset_infos), ds_info)


def update_chunk_counts(
    config: RetroPreprocessingConfig, indexed_dataset_infos: List[Dict]
) -> None:
    """Set n_chunks_train & n_chunks sampled for each individual DB.

    Args:
        config (RetroPreprocessingConfig): Retro preprocessing config.
        indexed_dataset_infos (List[Dict]): Preprocessing metadata for each dataset (i.e., 'prefix', 'ratio', 'n_chunks', etc.).
    """

    if torch.distributed.get_rank() != 0:
        return

    # Data ratio sum (for setting index training chunks).
    data_ratio_sum = sum([d["ratio"] for d in indexed_dataset_infos])

    # Training split size (split at document level).
    train_fraction = float(extract_data_config(config).split.split(",")[0]) / 100
    assert train_fraction > 0 and train_fraction <= 1

    # Set n_chunks (including n_chunks_sampled for unambiguity).
    log_retro_rank_0(" > compute n_chunks.")
    for ds_index, ds_info in enumerate(indexed_dataset_infos):

        db_paths = get_individual_db_paths(config.retro_project_dir, ds_info["prefix"])

        # Update counts.
        ds_info["n_docs"] = len(ds_info["dataset"].document_indices) - 1
        ds_info["n_docs_train"] = int(train_fraction * ds_info["n_docs"])
        ds_info["n_chunks"] = 0  # previously, 'n_chunks_valid'
        ds_info["n_chunks_train"] = 0
        ds_info["n_chunks_invalid"] = 0
        for db_path in tqdm(
            db_paths, "%d/%d, %s" % (ds_index, len(indexed_dataset_infos), ds_info["prefix"])
        ):
            with h5py.File(db_path, "r") as f:
                ds_info["n_chunks"] += len(f["chunks_valid"])
                ds_info["n_chunks_invalid"] += len(f["chunks_invalid"])
                ds_info["n_chunks_train"] += (
                    (np.copy(f["chunks_valid"][:, 0]) < ds_info["n_docs_train"]).sum().item()
                )

        ds_info["n_chunks_sampled"] = int(
            config.retro_index_ntrain * ds_info["ratio"] / data_ratio_sum
        )

        # Verify counts.
        assert ds_info["n_chunks_train"] <= ds_info["n_chunks"], "n_train (%d) > n_total (%d)." % (
            ds_info["n_chunks_train"],
            ds_info["n_chunks"],
        )
        assert (
            ds_info["n_chunks_sampled"] <= ds_info["n_chunks_train"]
        ), "n_sampled (%d) > n_train (%d)." % (
            ds_info["n_chunks_sampled"],
            ds_info["n_chunks_train"],
        )


def merge_dbs(project_dir: str, indexed_dataset_infos: List[Dict], db_type: str) -> None:
    """Merge individual DBs into single DB.

    Args:
        project_dir (str): Retro project dir.
        indexed_dataset_infos (List[Dict]): Preprocessing metadata for each dataset (i.e., 'prefix', 'ratio', 'n_chunks', etc.).
        db_type (str): DB type (e.g., 'sampled', 'train', or 'valid').
    """

    if torch.distributed.get_rank() != 0:
        return

    log_retro_rank_0(" > build %s chunk db." % db_type)

    # Count chunks.
    if db_type == "sampled":
        n_chunks_key = "n_chunks_sampled"
        n_docs_key = None
    elif db_type == "train":
        n_chunks_key = "n_chunks_train"
        n_docs_key = "n_docs_train"
    elif db_type == "valid":
        n_docs_key = None
    else:
        raise Exception("handle db_type '%s'." % db_type)

    if db_type == "valid":
        n_chunks = sum(m["n_chunks"] - m["n_chunks_train"] for m in indexed_dataset_infos)
    else:
        n_chunks = sum(m[n_chunks_key] for m in indexed_dataset_infos)
        n_docs = None if n_docs_key is None else sum(m[n_docs_key] for m in indexed_dataset_infos)

    # DB path.
    db_path = get_merged_db_path_map(project_dir)[db_type]

    # Delete existing chunk db if incorrect size.
    if os.path.exists(db_path):

        try:

            f = h5py.File(db_path)
            n_alloc = len(f["chunks"])  # total allocated
            n_written = f["n_written"][0].item()  # total written
            f.close()

            if n_chunks != n_alloc or n_chunks != n_written:
                os.remove(db_path)

        except Exception as e:
            if isinstance(e, OSError):
                os.remove(db_path)
            elif isinstance(e, KeyError):
                f.close()
                os.remove(db_path)
            else:
                raise e

    # Build merged chunk db.
    if not os.path.exists(db_path):

        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        f = h5py.File(db_path, "w")

        # Initialize output arrays.
        merged_chunk_db: np.ndarray = f.create_dataset("chunks", (n_chunks, 5), dtype="uint32")
        merged_doc_offsets: np.ndarray = (
            None
            if n_docs_key is None
            else f.create_dataset("doc_offsets", (n_docs, 3), dtype="uint64")
        )
        n_written = f.create_dataset("n_written", (1,), dtype="uint64")
        n_written[0] = 0

        # Iterate indexed datasets & collect chunks.
        chunk_start_index = 0
        doc_start_index = 0
        doc_start_offset = 0
        for ds_idx, ds_info in enumerate(indexed_dataset_infos):
            log_retro_rank_0(
                " > merging dbs; '%s', dataset %d / %d ... '%s'."
                % (db_type, ds_idx, len(indexed_dataset_infos), ds_info["prefix"]),
            )
            individual_chunk_db: np.ndarray = get_individual_chunk_db(project_dir, ds_idx, ds_info)
            individual_doc_offsets: np.ndarray = (
                None
                if n_docs_key is None
                else get_individual_doc_offsets(project_dir, ds_idx, ds_info)
            )

            if db_type == "valid":
                individual_chunk_db = individual_chunk_db[ds_info["n_chunks_train"] :]
                if n_docs_key is None:
                    individual_doc_offsets = None
                else:
                    train_doc_offset = individual_doc_offsets[ds_info["n_docs_train"] - 1, 2]
                    individual_doc_offsets = np.copy(
                        individual_doc_offsets[ds_info["n_docs_train"] :]
                    )
                    individual_doc_offsets[:, 2] -= train_doc_offset

                    log_retro_rank_0("~~~")
                    log_retro_rank_0(individual_doc_offsets)
                    log_retro_rank_0(train_doc_offset)
                    raise Exception("test me.")
            else:
                individual_chunk_db = individual_chunk_db[: ds_info[n_chunks_key]]
                individual_doc_offsets = (
                    None
                    if n_docs_key is None
                    else np.copy(individual_doc_offsets[: ds_info[n_docs_key]])
                )

            merged_chunk_db[chunk_start_index : chunk_start_index + len(individual_chunk_db)] = (
                individual_chunk_db
            )
            chunk_start_index += len(individual_chunk_db)
            n_written[0] = chunk_start_index
            if n_docs_key is not None:
                individual_doc_offsets[:, 2] += doc_start_offset
                doc_end_index = doc_start_index + individual_doc_offsets.shape[0]
                merged_doc_offsets[doc_start_index:doc_end_index] = individual_doc_offsets
                doc_start_index = doc_end_index
                doc_start_offset = individual_doc_offsets[-1, 2].item()

        f.close()


def build_merged_dbs(project_dir: str, indexed_dataset_infos: List[Dict]) -> None:
    """Merge individual dataset components into single database.

    This method merges databases for DB types:
    - 'sampled': used for training the vector index.
    - 'train': used for adding to the trained vector index.
    - 'valid': can be used for validating/testing the vector index.

    Args:
        project_dir (str): Retro project dir.
        indexed_dataset_infos (List[Dict]): Preprocessing metadata for each dataset (i.e., 'prefix', 'ratio', 'n_chunks', etc.).
    """
    merge_dbs(project_dir, indexed_dataset_infos, "sampled")
    merge_dbs(project_dir, indexed_dataset_infos, "train")
    merge_dbs(project_dir, indexed_dataset_infos, "valid")


def build_db(config: RetroPreprocessingConfig) -> None:
    """Extract token chunks from each indexed dataset.

    Iterate each document of each indexed dataset, extract that document's chunks, and save to a 'DB' (hdf5 file).

    Args:
        config (RetroPreprocessingConfig): Retro preprocessing config.
    """

    project_dir = config.retro_project_dir

    # Indexed dataset info.
    if config.retro_task_validate is None:
        indexed_dataset_infos = init_indexed_dataset_infos(config)
    else:
        indexed_dataset_infos = get_indexed_dataset_infos(config.retro_project_dir)
    # Build individual dbs.
    build_individual_dbs(config, indexed_dataset_infos)

    # If validating, return here.
    if config.retro_task_validate is not None:
        return

    # Single-process going forward.
    if torch.distributed.get_rank() != 0:
        return

    # Update n_chunks & save indexed dataset infos.
    if not os.path.exists(get_indexed_dataset_infos_path(project_dir)):
        update_chunk_counts(config, indexed_dataset_infos)
        save_indexed_dataset_infos(project_dir, indexed_dataset_infos)
    indexed_dataset_infos = get_indexed_dataset_infos(project_dir)

    # Builded merged dbs.
    build_merged_dbs(project_dir, indexed_dataset_infos)
