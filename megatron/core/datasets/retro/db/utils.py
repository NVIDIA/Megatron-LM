# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.

"""Utilities for building a chunk database."""

import glob
import json
import os
from typing import Dict, List, Optional

import numpy as np

from megatron.core.datasets.indexed_dataset import IndexedDataset
from megatron.core.datasets.retro.config import RetroPreprocessingConfig
from megatron.core.datasets.retro.external_libs import h5py
from megatron.core.models.retro.utils import get_gpt_data_dir

from .dataset import DBDataset


def get_db_dir(project_dir: str) -> str:
    """Sub-directory for DB data.

    Args:
        project_dir (str): Path to Retro project dir.

    Returns:
        Path of the DB sub-directory within the project.
    """
    return os.path.join(project_dir, "db")


def init_indexed_dataset_infos(config: RetroPreprocessingConfig) -> List[Dict]:
    """Gather meta-info about each indexed dataset.

    The returned info array allows for easy access to the configuration, and
    helps remove ambiguity.

    Args:
        config (RetroPreprocessingConfig): Retro preprocessing config.

    Returns:
        List of processing metadata for each dataset, including:
        - ratio: Data split weight.
        - prefix: Relative path to dataset under DB sub-directory.
    """

    data_dir = get_gpt_data_dir(config.retro_project_dir)
    data_blend: List[str] = config.retro_gpt_data_path
    assert len(data_blend) % 2 == 0, "currently, only blended dataset is supported."

    # Dataset infos.
    infos = []
    for i in range(0, len(data_blend), 2):
        ratio = float(data_blend[i])
        prefix = data_blend[i + 1]
        path = os.path.join(data_dir, prefix + ".bin")
        assert os.path.exists(path), "couldn't find '%s'." % path
        infos.append({"ratio": ratio, "prefix": prefix})

    # Load indexed datasets.
    load_indexed_datasets(config.retro_project_dir, infos)

    return infos


def get_indexed_dataset_infos_path(project_dir: str) -> str:
    """Path to indexed dataset meta-infos.

    Args:
        project_dir (str): Path to Retro project dir.

    Returns:
        Path to the `indexed_dataset_infos.json` file.
    """
    return os.path.join(get_db_dir(project_dir), "indexed_dataset_infos.json")


def save_indexed_dataset_infos(project_dir: str, indexed_dataset_infos: List[Dict]) -> None:
    """Save dataset order & meta-info.

    Args:
        project_dir (str): Path to Retro project dir.
        indexed_dataset_infos (List[Dict]): List of metadata for each dataset, with each entry containing:

        - ratio: Data split weight.
        - prefix: Relative path to dataset under DB sub-directory.
        - n_docs: Number of documents.
        - n_docs_train: Number of documents used for pretraining.
        - n_chunks: Number of valid chunks.
        - n_chunks_train: Number of valid chunks used for pretraining.
        - n_chunks_invalid: Number of invalid chunks.
        - n_chunks_sampled: Number of valid chunks used for vector index training.
    """

    # Remove 'dataset' field.
    clean_infos = []
    for info in indexed_dataset_infos:
        info = dict(info)
        del info["dataset"]
        clean_infos.append(info)

    # Save.
    with open(get_indexed_dataset_infos_path(project_dir), "w") as f:
        json.dump(clean_infos, f, indent=4)


def load_indexed_datasets(project_dir: str, indexed_dataset_infos: List[Dict]) -> None:
    """Loaded indexed datasets into memory-mapped datasets.

    Args:
        project_dir (str): Path to Retro project dir.
        indexed_dataset_infos (List[Dict]): List of metadata for each dataset (see `save_indexed_dataset_infos()` for more details.
    """
    data_dir = get_gpt_data_dir(project_dir)
    for info in indexed_dataset_infos:
        info["dataset"] = IndexedDataset(os.path.join(data_dir, info["prefix"]), mmap=True)


def get_indexed_dataset_infos(project_dir: str) -> List[Dict]:
    """Load indexed dataset meta-infos.

    Args:
        project_dir (str): Path to Retro project dir.

    Returns:
        List of metadata for each dataset (see `save_indexed_dataset_infos()` for more details.
    """

    # Load json.
    path = get_indexed_dataset_infos_path(project_dir)
    with open(path) as f:
        infos = json.load(f)

    # Load indexed datasets.
    load_indexed_datasets(project_dir, infos)

    return infos


def get_individual_db_dir(project_dir: str, prefix: str) -> str:
    """Individual DB's directory.

    Args:
        project_dir (str): Path to Retro project dir.
        prefix (str): Unique relative path to dataset within project dir.

    Returns:
        Path to the given datasets's chunk database.
    """
    return os.path.join(get_db_dir(project_dir), "individual", prefix)


def get_individual_db_paths(project_dir: str, prefix: str) -> List[str]:
    """Get paths of all database blocks of an individual dataset.

    Args:
        project_dir (str): Path to Retro project dir.
        prefix (str): Unique relative path to dataset within project dir.

    Returns:
        Paths to each HDF5 chunk database files that comprises this datasets full chunk database.
    """
    return sorted(glob.glob(get_individual_db_dir(project_dir, prefix) + "/*hdf5"))


def get_individual_chunk_db(project_dir: str, ds_id: int, ds_info: dict) -> np.ndarray:
    """Load individual dataset's chunk DB.

    Args:
        project_dir (str): Path to Retro project dir.
        ds_id (int): Index of dataset within blended dataset.
        ds_info (dict): Preprocessing metadata for dataset (see `save_indexed_dataset_infos()` for more detail).

    Returns:
        Array of chunk start/end indexes for this dataset, where the chunk indexes can be used for indexing into the corresponding indexed dataset.
    """
    paths = get_individual_db_paths(project_dir, ds_info["prefix"])
    # *Note*: convert to dataset, rather than copying to memory.
    db = np.zeros((ds_info["n_chunks"], 5), dtype="uint32")
    db[:, 0] = ds_id
    start_idx = 0
    for path in paths:
        f = h5py.File(path, "r")
        n_chunks_current = f["chunks_valid"].shape[0]
        db[start_idx : (start_idx + n_chunks_current), 1:] = f["chunks_valid"]
        start_idx += n_chunks_current
        f.close()

    assert start_idx == ds_info["n_chunks"]

    return db


def get_individual_doc_offsets(project_dir: str, ds_id: int, ds_info: dict) -> np.ndarray:
    """Load individual dataset's document offsets.

    Args:
        project_dir (str): Path to Retro project dir.
        ds_id (int): Index of dataset within blended dataset.
        ds_info (dict): Preprocessing metadata for dataset (see `save_indexed_dataset_infos()` for more detail).

    Returns:
        Array of document offsets by chunk index for this dataset.
    """
    paths = get_individual_db_paths(project_dir, ds_info["prefix"])
    # *Note*: convert to dataset, rather than copying to memory.
    doc_offsets = np.zeros((ds_info["n_docs"], 3), dtype="uint64")
    doc_offsets[:, 0] = ds_id
    start_idx = 0
    start_offset = 0
    for path in paths:
        with h5py.File(path) as f:
            current_doc_offsets = np.copy(f["doc_offsets"])
            current_doc_offsets[:, 1] += start_offset
            current_ndocs = current_doc_offsets.shape[0]
            doc_offsets[start_idx : (start_idx + current_ndocs), 1:] = current_doc_offsets
            start_idx += current_ndocs
            start_offset = current_doc_offsets[-1, 1].item()

    return doc_offsets


def get_merged_db_path_map(project_dir: str) -> dict:
    """Paths to merged datasets.

    Args:
        project_dir (str): Path to Retro project dir.

    Returns:
        A dict of chunk databases, one for each of:
        - sampled: Chunks used for training the vector index.
        - train: Chunks used for pretraining 'train' dataset.
        - valid: Chunks used for pretraining 'valid' dataset.
    """
    base_dir = get_db_dir(project_dir)
    return {
        "sampled": os.path.join(base_dir, "merged", "sampled.hdf5"),
        "train": os.path.join(base_dir, "merged", "train.hdf5"),
        "valid": os.path.join(base_dir, "merged", "valid.hdf5"),
    }


def get_merged_dataset(
    project_dir: str,
    chunk_length: int,
    eod_token_id: int,
    db_type: str,
    indexed_dataset_infos: Optional[List[Dict]] = None,
) -> DBDataset:
    """Get merged dataset.

    Args:
        project_dir (str): Path to Retro project dir.
        chunk_length (int): GPT chunk length (e.g., 64).
        eod_token_id (int): EOD token ID.
        db_type (str): DB type (e.g., 'sampled', 'train', or 'valid').
        indexed_dataset_infos (Optional[List[Dict]]): Optionally, pre-loaded list of dataset metadata (see `save_indexed_dataset_infos()` for more detail). If not provided, the indexed dataset infos will be loaded from disk.

    Returns:
        A DBDataset, which is a dataset that wraps the HDF5 chunk index array.
    """

    if not indexed_dataset_infos:
        indexed_dataset_infos = get_indexed_dataset_infos(project_dir)

    # Load chunks.
    db_path = get_merged_db_path_map(project_dir)[db_type]
    f = h5py.File(db_path, "r")
    chunks = f["chunks"]

    # DB dataset.
    indexed_datasets = [info["dataset"] for info in indexed_dataset_infos]
    dataset = DBDataset(
        db_path=db_path,
        indexed_datasets=indexed_datasets,
        chunks=chunks,
        chunk_length=chunk_length,
        eod_token_id=eod_token_id,
    )

    return dataset


def get_merged_sampled_dataset(
    project_dir: str,
    chunk_length: int,
    eod_token_id: int,
    indexed_dataset_infos: Optional[List[Dict]] = None,
) -> DBDataset:
    """Get sampled dataset (for training the vector index).

    Args:
        project_dir (str): Path to Retro project dir.
        chunk_length (int): GPT chunk length (e.g., 64).
        eod_token_id (int): EOD token ID.
        indexed_dataset_infos (Optional[List[Dict]]): Optionally, pre-loaded list of dataset metadata (see `save_indexed_dataset_infos()` for more detail). If not provided, the indexed dataset infos will be loaded from disk.

    Returns:
        A DBDataset, which is a dataset that wraps the HDF5 chunk index array.
    """
    return get_merged_dataset(
        project_dir, chunk_length, eod_token_id, "sampled", indexed_dataset_infos
    )


def get_merged_train_dataset(
    project_dir: str,
    chunk_length: int,
    eod_token_id: int,
    indexed_dataset_infos: Optional[List[Dict]] = None,
) -> DBDataset:
    """Get training dataset (for adding to the vector index).

    Args:
        project_dir (str): Path to Retro project dir.
        chunk_length (int): GPT chunk length (e.g., 64).
        eod_token_id (int): EOD token ID.
        indexed_dataset_infos (Optional[List[Dict]]): Optionally, pre-loaded list of dataset metadata (see `save_indexed_dataset_infos()` for more detail). If not provided, the indexed dataset infos will be loaded from disk.

    Returns:
        A DBDataset, which is a dataset that wraps the HDF5 chunk index array.
    """
    return get_merged_dataset(
        project_dir, chunk_length, eod_token_id, "train", indexed_dataset_infos
    )


def get_merged_valid_dataset(
    project_dir: str,
    chunk_length: int,
    eod_token_id: int,
    indexed_dataset_infos: Optional[List[Dict]] = None,
) -> DBDataset:
    """Get validation dataset (for testing the vector index).

    Args:
        project_dir (str): Path to Retro project dir.
        chunk_length (int): GPT chunk length (e.g., 64).
        eod_token_id (int): EOD token ID.
        indexed_dataset_infos (Optional[List[Dict]]): Optionally, pre-loaded list of dataset metadata (see `save_indexed_dataset_infos()` for more detail). If not provided, the indexed dataset infos will be loaded from disk.

    Returns:
        A DBDataset, which is a dataset that wraps the HDF5 chunk index array.
    """
    return get_merged_dataset(
        project_dir, chunk_length, eod_token_id, "valid", indexed_dataset_infos
    )


def get_merged_datasets(project_dir: str, chunk_length: int, eod_token_id: int) -> dict:
    """Get all merged datasets.

    Args:
        project_dir (str): Path to Retro project dir.
        chunk_length (int): GPT chunk length (e.g., 64).
        eod_token_id (int): EOD token ID.

    Returns:
        A dict mapping DB type ('sampled', 'train', or 'valid') to the corresponding DBDataset, which is a dataset that wraps the HDF5 chunk index array.
    """
    fns = {
        "sampled": get_merged_sampled_dataset,
        "train": get_merged_train_dataset,
        "valid": get_merged_valid_dataset,
    }
    datasets = {key: fn(project_dir, chunk_length, eod_token_id) for key, fn in fns.items()}
    return datasets
