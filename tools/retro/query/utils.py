# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

import hashlib
import os

from megatron import get_retro_args


def get_query_workdir():
    args = get_retro_args()
    return os.path.join(args.retro_workdir, "query")


def get_neighbor_dirname(key, dataset):
    hashes = ",".join([ d.desc_hash for d in dataset.datasets ])
    hash = hashlib.md5(hashes.encode()).hexdigest()
    return os.path.join(get_query_workdir(), os.path.basename(f"{key}_{hash}"))
