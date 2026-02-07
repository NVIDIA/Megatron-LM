# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""
Utility helpers for broadcasting nested dictionaries of tensors across tensor-parallel ranks.

"""

from typing import Any, Dict, List, Tuple

import torch

from megatron.core import mpu, tensor_parallel


def flatten(
    nested: Dict[str, Any], prefix: Tuple[str, ...] = ()
) -> List[Tuple[Tuple[str, ...], torch.Tensor]]:
    """Recursively flatten nested dict into [(key_path, tensor), …]."""
    flat = []
    for k, v in nested.items():
        path = prefix + (k,)
        if isinstance(v, dict):
            flat.extend(flatten(v, path))
        elif isinstance(v, torch.Tensor):
            flat.append((path, v))        # v is a tensor
        else:
            raise ValueError(f"Unsupported value type: {type(v)} for key {k}"
                             f"In nested dictionary,leaf nodes must contain tensors")
    return flat


def regroup(flat: List[Tuple[Tuple[str, ...], torch.Tensor]]) -> Dict[str, Any]:
    """Rebuild the nested dict from [(key_path, tensor), …]."""
    root = {}
    for path, tensor in flat:
        cur = root
        for k in path[:-1]:
            cur = cur.setdefault(k, {})
        cur[path[-1]] = tensor
    return root


def broadcast_nested_data_batch(nested_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively broadcast nested dictionaries of tensors using each tensor's own dtype."""
    
    tp_group = mpu.get_tensor_model_parallel_group()
    src      = mpu.get_tensor_model_parallel_src_rank()

    # ---------- rank-0 prepares metadata ----------
    if mpu.get_tensor_model_parallel_rank() == 0:
        flat = flatten(nested_dict)                # [(path,tensor), …]
        paths, tensors = zip(*flat) if flat else ([], [])
        dtypes = [t.dtype for t in tensors]
    else:
        paths, dtypes = [], []
        tensors = []

    # ---------- 1. broadcast schema (paths + dtypes) ----------
    meta = [paths, dtypes]                         # small, picklable
    obj_list = [meta]
    torch.distributed.broadcast_object_list(obj_list, src=src, group=tp_group)
    paths, dtypes = obj_list[0]                    # now identical on all ranks

    # ---------- 2. group tensors by dtype and broadcast ----------
    # build maps keyed by dtype for convenience
    dtype_to_keys = {}
    for p, dt in zip(paths, dtypes):
        dtype_to_keys.setdefault(dt, []).append(".".join(p))  # join for key strings

    # On src rank: make a dict {joined_path: tensor}
    if mpu.get_tensor_model_parallel_rank() == 0:
        data_dict = {".".join(p): t.cuda() for p, t in zip(paths, tensors)}
    else:
        data_dict = {}

    flat_out = []
    for dt, keys in dtype_to_keys.items():
        out = tensor_parallel.broadcast_data(keys, data_dict, dt)
        flat_out.extend([(tuple(k.split(".")), out[k]) for k in keys])

    # ---------- 3. rebuild nested structure ----------
    return regroup(flat_out)