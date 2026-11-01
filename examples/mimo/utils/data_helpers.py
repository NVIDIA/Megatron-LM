# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""
Utility helpers for broadcasting nested dictionaries of tensors across tensor-parallel ranks.

"""

from typing import Any, Dict, List, Optional, Tuple

import torch

from megatron.core import tensor_parallel
from megatron.core.process_groups_config import resolve_process_groups


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


def broadcast_nested_data_batch(
    nested_dict: Dict[str, Any], tp_group: Optional[torch.distributed.ProcessGroup] = None
) -> Dict[str, Any]:
    """Recursively broadcast nested dictionaries of tensors using each tensor's own dtype.

    Args:
        nested_dict: The batch on the first rank of ``tp_group``; ignored on the other ranks.
        tp_group: Tensor-parallel group of the model, whose first rank holds the batch. None
            falls back to the global tensor-parallel group with a ProcessGroupFallbackWarning;
            that fallback is deprecated.
    """
    if tp_group is None:
        tp_group = resolve_process_groups(
            None,
            owner="examples.mimo.utils.data_helpers.broadcast_nested_data_batch",
            required=("tp",),
            argument="tp_group",
            deprecated_in="0.21",
            removed_in="0.23",
        ).tp
    src = torch.distributed.get_global_rank(tp_group, 0)
    is_src = tp_group.rank() == 0

    # ---------- rank-0 prepares metadata ----------
    if is_src:
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
    if is_src:
        data_dict = {".".join(p): t.cuda() for p, t in zip(paths, tensors)}
    else:
        data_dict = {}

    flat_out = []
    for dt, keys in dtype_to_keys.items():
        out = tensor_parallel.broadcast_data(keys, data_dict, dt, tp_group=tp_group)
        flat_out.extend([(tuple(k.split(".")), out[k]) for k in keys])

    # ---------- 3. rebuild nested structure ----------
    return regroup(flat_out)