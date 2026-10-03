# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Reuse packed-sequence validation while tensor identity and versions are unchanged."""

import weakref
from collections import OrderedDict

import torch

_VALIDATED = OrderedDict()
_CACHE_SIZE = 16


def _version(tensor):
    try:
        return tensor._version
    except RuntimeError:
        # Inference tensors do not track mutations, so always revalidate them.
        return None


def resolve_packed_sequences(
    q: torch.Tensor, kv: torch.Tensor, total_seq_len: int, cp_size: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate packed metadata, reusing validation only for unchanged live tensors."""
    q_version, kv_version = _version(q), _version(kv)
    key = (id(q), q_version, id(kv), kv_version, total_seq_len, cp_size)
    cacheable = q_version is not None and kv_version is not None
    cached = _VALIDATED.get(key) if cacheable else None
    if cached is not None and cached[0]() is q and cached[1]() is kv:
        _VALIDATED.move_to_end(key)
        return q, kv

    q_values = q.detach().cpu().tolist()
    kv_values = q_values if kv is q else kv.detach().cpu().tolist()
    for name, values in (("cu_seqlens_q", q_values), ("cu_seqlens_kv", kv_values)):
        if not values or values[-1] != total_seq_len:
            raise ValueError(f"GDN: {name} does not end at total_sequence_length={total_seq_len}")
        if cp_size != 1 and any((end - start) % cp_size for start, end in zip(values, values[1:])):
            raise ValueError(f"All per-sequence lengths must be divisible by cp_size={cp_size}")
    if q_values != kv_values:
        raise AssertionError("Currently only support cu_seqlens_q equals to cu_seqlens_kv")

    if cacheable:
        _VALIDATED[key] = (weakref.ref(q), weakref.ref(kv))
        _VALIDATED.move_to_end(key)
        while len(_VALIDATED) > _CACHE_SIZE:
            _VALIDATED.popitem(last=False)
    return q, kv
