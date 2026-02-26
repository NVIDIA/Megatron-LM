# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""
Utility helpers for broadcasting nested dictionaries of tensors across tensor-parallel ranks.

"""

from typing import Any, Dict, List, Tuple, Union

import torch

from megatron.core import mpu, tensor_parallel

# Special suffixes to mark original types
_LIST_MARKER = "__was_list__"
_INT_MARKER = "__was_int__"
_FLOAT_MARKER = "__was_float__"
_BOOL_MARKER = "__was_bool__"

# Types that should be skipped during tensor broadcast (they'll be broadcast via object_list)
_SKIP_TENSOR_BROADCAST_TYPES = (type(None),)

# Try to import BlockMask for flex attention (optional)
try:
    from torch.nn.attention.flex_attention import BlockMask
    _SKIP_TENSOR_BROADCAST_TYPES = (type(None), BlockMask)
except ImportError:
    BlockMask = None


def flatten(
    nested: Dict[str, Any], prefix: Tuple[str, ...] = ()
) -> Tuple[List[Tuple[Tuple[str, ...], torch.Tensor]], Dict[str, Any]]:
    """Recursively flatten nested dict into [(key_path, tensor), …] and non-tensor items.
    
    Returns:
        Tuple of (flat_tensors, non_tensor_items)
        - flat_tensors: List of (path, tensor) tuples for tensorizable items
        - non_tensor_items: Dict of items that cannot be converted to tensors
    """
    flat = []
    non_tensor = {}
    
    for k, v in nested.items():
        path = prefix + (k,)
        path_str = ".".join(path)
        
        if isinstance(v, dict):
            sub_flat, sub_non_tensor = flatten(v, path)
            flat.extend(sub_flat)
            non_tensor.update(sub_non_tensor)
        elif isinstance(v, torch.Tensor):
            flat.append((path, v))
        elif isinstance(v, list):
            # Convert list to tensor and mark with special suffix
            try:
                tensor_v = torch.tensor(v)
                marked_path = prefix + (k + _LIST_MARKER,)
                flat.append((marked_path, tensor_v))
            except (ValueError, TypeError):
                # List contains non-numeric items, store as non-tensor
                non_tensor[path_str] = v
        elif isinstance(v, int):
            # Convert int to tensor and mark
            tensor_v = torch.tensor([v], dtype=torch.long)
            marked_path = prefix + (k + _INT_MARKER,)
            flat.append((marked_path, tensor_v))
        elif isinstance(v, float):
            # Convert float to tensor and mark
            tensor_v = torch.tensor([v], dtype=torch.float32)
            marked_path = prefix + (k + _FLOAT_MARKER,)
            flat.append((marked_path, tensor_v))
        elif isinstance(v, bool):
            # Convert bool to tensor and mark (must check before int since bool is subclass of int)
            tensor_v = torch.tensor([v], dtype=torch.bool)
            marked_path = prefix + (k + _BOOL_MARKER,)
            flat.append((marked_path, tensor_v))
        elif isinstance(v, _SKIP_TENSOR_BROADCAST_TYPES):
            # These types will be broadcast via object_list
            non_tensor[path_str] = v
        else:
            # Unknown type - try to handle, or store as non-tensor
            non_tensor[path_str] = v
    
    return flat, non_tensor


def regroup(flat: List[Tuple[Tuple[str, ...], torch.Tensor]], non_tensor: Dict[str, Any]) -> Dict[str, Any]:
    """Rebuild the nested dict from [(key_path, tensor), …] and non-tensor items."""
    root = {}
    
    # First, add all tensor items
    for path, tensor in flat:
        cur = root
        for k in path[:-1]:
            cur = cur.setdefault(k, {})
        final_key = path[-1]
        
        # Check for type markers and convert back
        if final_key.endswith(_LIST_MARKER):
            final_key = final_key[:-len(_LIST_MARKER)]
            cur[final_key] = tensor.tolist()
        elif final_key.endswith(_INT_MARKER):
            final_key = final_key[:-len(_INT_MARKER)]
            cur[final_key] = tensor.item()
        elif final_key.endswith(_FLOAT_MARKER):
            final_key = final_key[:-len(_FLOAT_MARKER)]
            cur[final_key] = tensor.item()
        elif final_key.endswith(_BOOL_MARKER):
            final_key = final_key[:-len(_BOOL_MARKER)]
            cur[final_key] = tensor.item()
        else:
            cur[final_key] = tensor
    
    # Then, add all non-tensor items
    for path_str, value in non_tensor.items():
        path = path_str.split(".")
        cur = root
        for k in path[:-1]:
            cur = cur.setdefault(k, {})
        cur[path[-1]] = value
    
    return root


def broadcast_nested_data_batch(nested_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively broadcast nested dictionaries of tensors using each tensor's own dtype.
    
    Handles:
    - Tensors: broadcast via tensor_parallel.broadcast_data
    - Lists: converted to tensors, broadcast, converted back
    - Scalars (int, float, bool): converted to tensors, broadcast, converted back
    - Other types (None, BlockMask, etc.): broadcast via object_list
    """
    
    tp_group = mpu.get_tensor_model_parallel_group()
    src = mpu.get_tensor_model_parallel_src_rank()

    # ---------- rank-0 prepares metadata ----------
    if mpu.get_tensor_model_parallel_rank() == 0:
        flat, non_tensor = flatten(nested_dict)
        paths, tensors = zip(*flat) if flat else ([], [])
        dtypes = [t.dtype for t in tensors]
    else:
        paths, dtypes = [], []
        tensors = []
        non_tensor = {}

    # ---------- 1. broadcast schema (paths + dtypes + non_tensor keys) ----------
    meta = [paths, dtypes, non_tensor]
    obj_list = [meta]
    torch.distributed.broadcast_object_list(obj_list, src=src, group=tp_group)
    paths, dtypes, non_tensor = obj_list[0]

    # ---------- 2. group tensors by dtype and broadcast ----------
    dtype_to_keys = {}
    for p, dt in zip(paths, dtypes):
        dtype_to_keys.setdefault(dt, []).append(".".join(p))

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
    return regroup(flat_out, non_tensor)
