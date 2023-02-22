# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

from collections import defaultdict
import glob
import numpy as np
import os
import torch
from tqdm import tqdm

from megatron import print_rank_0
from megatron.core import parallel_state

from .external_libs import h5py


def save_data(data_map, *args):
    '''Save map of numpy arrays to hdf5 file.'''

    # Parse args.
    if len(args) == 1:
        path = args[0]
    elif len(args) == 2:
        dir_path, file_name = args
        path = os.path.join(dir_path, file_name)
    else:
        raise Exception("specialize for len(args) == %d." % len(args))

    # Save data.
    if not os.path.isfile(path):
        f = h5py.File(path, "w")
        for k, v in data_map.items():
            f.create_dataset(k, data=v)
        f.close()

    return path


def load_data(paths):
    '''Load multiple hdf5 files to single numpy array.'''

    # Read data shapes.
    shape_map = defaultdict(lambda : (0, None))
    for p in paths:
        f = h5py.File(p, "r")
        for k in f.keys():
            shape = tuple(f[k].shape)
            shape_map[k] = (shape_map[k][0] + shape[0], shape[1])
        f.close()

    # Allocate output array.
    data_map = { k : np.empty(s, dtype="f4") for k, s in shape_map.items() }
    start_map = { k : 0 for k in shape_map }

    # Load files.
    for pi, p in enumerate(tqdm(paths, "load data")):
        f = h5py.File(p, "r")
        for k in f.keys():
            i0 = start_map[k]
            i1 = i0 + len(f[k])
            data_map[k][i0:i1] = f[k]
            start_map[k] += len(f[k])
        f.close()

    return data_map


def get_missing_blocks(workdir, n_samples, block_size,
                       validate=lambda f : None):
    '''Divide range [0, num_samples) to sequence of block ranges.

    This is a core method within the concept of block processing. The idea
    is to divide a range (size n_samples) into a sequence of blocks. Each
    block corresponds to a file within 'workdir' with name
    '{start_idx}-{end_idx}.hdf5'. This method checks for the existence of
    these files, and returns a list of the ones that are missing.
    '''

    # Block ranges.
    block_start_idxs = list(range(0, n_samples, block_size))
    block_end_idxs = [ min(n_samples, i + block_size) for i in block_start_idxs ]
    block_ranges = list(zip(block_start_idxs, block_end_idxs))

    # All block files (existing + missing).
    n_digits = int(np.ceil(np.log(n_samples) / np.log(10)) + 1)
    all_blocks = [{
        "range" : r,
        "path" : os.path.join(
            workdir,
            "%s-%s.hdf5" % tuple([ str(i).zfill(n_digits) for i in r ]),
        )
    } for r in block_ranges]
    all_block_path_set = set(block["path"] for block in all_blocks)

    # Delete corrupt files.
    if torch.distributed.get_rank() == 0:
        existing_block_paths = [block["path"]
                                for block in all_blocks
                                if os.path.exists(block["path"])]
        for index, path in enumerate(
                tqdm(existing_block_paths, "validating block.")):

            assert path in all_block_path_set, "unexpected filename, '%s'." % path

            try:
                f = h5py.File(path, "r")
            except:
                raise Exception("unable to open/validate '%s'." % path)
                os.remove(path)
                continue

            try:
                validate(f)
            except:
                raise Exception("delete block file.")
                os.remove(path)
            finally:
                f.close()

    # Wait for files to be deleted.
    torch.distributed.barrier()

    # Filter missing files.
    missing_blocks = [block
                      for block in all_blocks
                      if not os.path.exists(block["path"])]

    return missing_blocks


def get_missing_blocks_by_rank(workdir, n_samples, block_size,
                               validate=lambda f : None):
    '''Divide missing blocks evenly across all ranks.

    See 'get_missing_blocks()' above for description. The returned list of
    missing blocks is split evenly across ranks via interleaving. This way,
    each rank has a roughly equal number of blocks to process for a
    downstream operation.
    '''

    missing_blocks = get_missing_blocks(workdir, n_samples, block_size,
                                        validate)

    # This rank's missing files.
    data_parallel_rank = parallel_state.get_data_parallel_rank()
    data_parallel_world_size = parallel_state.get_data_parallel_world_size()
    rank_missing_blocks = missing_blocks[data_parallel_rank:len(missing_blocks):data_parallel_world_size]

    # Extend rank's missing blocks (with None) such that all ranks have equal
    # length lists. This allows for easier tracking of global progress.
    n_missing_tensor = torch.cuda.LongTensor([len(rank_missing_blocks)])
    torch.distributed.all_reduce(n_missing_tensor,
                                 op=torch.distributed.ReduceOp.MAX)
    max_n_missing = n_missing_tensor.item()
    rank_missing_blocks += [None] * (max_n_missing - len(rank_missing_blocks))

    return len(missing_blocks), rank_missing_blocks


class IdPathMap:
    '''Maps indexes to the containing block path.

    This class optimizing the mapping of a large number of indexes to the
    path of its containing block. For example, with block_size 1M, this class
    stores 1/1M as many (long) path strings, saving memory.
    '''

    def __init__(self, paths):
        self.paths = paths
        self.path_index_map = {p:i for i,p in enumerate(paths)}
        self.id_index_map = {}

    def __str__(self):
        return "%d paths; %d ids" % (len(self.paths), len(self.id_index_map))

    def add(self, id, path):
        '''Map index to a path.'''
        self.id_index_map[id] = self.path_index_map[path]

    def __contains__(self, idx):
        '''Index added to this object?'''
        return idx in self.id_index_map

    def __getitem__(self, idx):
        '''Get path from index.'''
        return self.paths[self.id_index_map[idx]]


def path_to_range(path):
    '''Parse start/end indexes from block path name (e.g., 00010-00011.hdf5 ->
    (10, 11).'''
    return tuple([
        int(i) for i in os.path.splitext(
            os.path.basename(path))[0].split("-")])


def get_index_path_map(_dir):
    '''Map contained indexes to block file path (on disk).'''

    paths = sorted(glob.glob(_dir + "/*.hdf5"))

    # Build index-path map.
    idx_path_map = IdPathMap(paths)
    for path in paths:
        start_idx, end_idx = path_to_range(path)
        for idx in range(start_idx, end_idx):
            idx_path_map.add(idx, path)

    return idx_path_map
