# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

from collections import defaultdict
import glob
import json
import numpy as np
import os
from tqdm import tqdm

from megatron import get_retro_args, print_rank_0
from megatron.core.datasets.indexed_dataset import MMapIndexedDataset
from tools.retro.external_libs import h5py

from .dataset import DBDataset


def get_base_db_workdir():
    '''Sub-directory for DB data.'''
    args = get_retro_args()
    return os.path.join(args.retro_workdir, "db")


def get_indexed_dataset_infos_path():
    '''Path to indexed dataset meta-infos.'''
    return os.path.join(get_base_db_workdir(), "indexed_dataset_infos.json")


def save_indexed_dataset_infos(indexed_dataset_infos):
    '''Save dataset order & meta-info.'''

    # Remove 'dataset' field.
    clean_infos = []
    for info in indexed_dataset_infos:
        info = dict(info)
        del info["dataset"]
        clean_infos.append(info)

    # Save.
    with open(get_indexed_dataset_infos_path(), "w") as f:
        json.dump(clean_infos, f, indent=4)


def get_indexed_dataset_infos():
    '''Load indexed dataset meta-infos.'''

    # Load json.
    path = get_indexed_dataset_infos_path()
    with open(path) as f:
        infos = json.load(f)

    # Add indexed datasets.
    for info in infos:
        info["dataset"] = MMapIndexedDataset(info["prefix"])

    return infos


def get_individual_db_dir(name):
    '''Individual DB's directory.'''
    return os.path.join(get_base_db_workdir(), "individual", name)


def get_individual_chunk_db(ds_id, ds_info):
    '''Load individual dataset's chunk DB.'''
    db_paths = sorted(glob.glob(ds_info["db_dir"] + "/*hdf5"))
    # *Note*: convert to dataset, rather than copying to memory.
    db = np.zeros((ds_info["n_chunks"], 5), dtype="uint32")
    db[:, 0] = ds_id
    start_idx = 0
    for db_path in db_paths:
        f = h5py.File(db_path, "r")
        n_chunks_current = f["chunks_valid"].shape[0]
        db[start_idx:(start_idx+n_chunks_current), 1:] = f["chunks_valid"]
        start_idx += n_chunks_current
        f.close()

    assert start_idx == ds_info["n_chunks"]

    return db


def get_individual_doc_offsets(ds_id, ds_info):
    '''Load individual dataset's chunk DB.'''
    paths = sorted(glob.glob(ds_info["db_dir"] + "/*hdf5"))
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
            doc_offsets[start_idx:(start_idx+current_ndocs), 1:] = \
                current_doc_offsets
            start_idx += current_ndocs
            start_offset = current_doc_offsets[-1, 1].item()

    return doc_offsets


def get_merged_db_path_map():
    '''Paths to merged datasets.'''
    base_dir = get_base_db_workdir()
    return {
        "sampled" : os.path.join(base_dir, "merged", "sampled.hdf5"),
        "train" : os.path.join(base_dir, "merged", "train.hdf5"),
        "valid" : os.path.join(base_dir, "merged", "valid.hdf5"),
    }


def get_merged_dataset(db_type, indexed_dataset_infos=None):
    '''Get merged dataset.'''

    args = get_retro_args()

    if not indexed_dataset_infos:
        indexed_dataset_infos = get_indexed_dataset_infos()

    # Load chunks.
    db_path = get_merged_db_path_map()[db_type]
    f = h5py.File(db_path, "r")
    chunks = f["chunks"]

    # DB dataset.
    indexed_datasets = [ info["dataset"] for info in indexed_dataset_infos ]
    dataset = DBDataset(db_path, indexed_datasets, chunks,
                        args.retro_gpt_chunk_length)

    return dataset


def get_merged_sampled_dataset(indexed_dataset_infos=None):
    return get_merged_dataset("sampled", indexed_dataset_infos)


def get_merged_train_dataset(indexed_dataset_infos=None):
    return get_merged_dataset("train", indexed_dataset_infos)


def get_merged_valid_dataset(indexed_dataset_infos=None):
    return get_merged_dataset("valid", indexed_dataset_infos)
