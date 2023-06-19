# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

import concurrent
import gc
import glob
import numpy as np
import os
import psutil
import time
import torch
from tqdm import tqdm

from megatron import get_retro_args, print_rank_0
from tools.retro.db.utils import get_indexed_dataset_infos
from tools.retro.external_libs import h5py


def get_index_dir():
    """Create sub-directory for this index."""

    args = get_retro_args()

    # Directory path.
    index_dir_path = os.path.join(
        args.retro_workdir,
        "index",
        args.retro_index_type,
        args.retro_index_str,
    )

    # Make directory.
    os.makedirs(index_dir_path, exist_ok=True)

    return index_dir_path


def num_samples_to_block_ranges(num_samples):
    '''Split a range (length num_samples) into sequence of block ranges
    of size block_size.'''
    args = get_retro_args()
    block_size = args.retro_block_size
    start_idxs = list(range(0, num_samples, block_size))
    end_idxs = [min(num_samples, s + block_size) for s in start_idxs]
    ranges = list(zip(start_idxs, end_idxs))
    return ranges


def get_training_data_root_dir():
    args = get_retro_args()
    return os.path.join(args.retro_workdir, "index", "train_emb")


def get_training_data_block_dir():
    return os.path.join(get_training_data_root_dir(), "blocks")


def get_training_data_block_paths():
    return sorted(glob.glob(get_training_data_block_dir() + "/*.hdf5"))


def get_training_data_merged_path():
    args = get_retro_args()
    return os.path.join(get_training_data_root_dir(),
                        "train_%.3f.bin" % args.retro_index_train_load_fraction)


def get_added_codes_dir():
    return os.path.join(get_index_dir(), "add_codes")


def get_added_code_paths():
    return sorted(glob.glob(get_added_codes_dir() + "/*.hdf5"))
