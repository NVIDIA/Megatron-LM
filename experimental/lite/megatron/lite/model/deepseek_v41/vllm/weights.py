# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Deployment weight encodings, computed once per model forward.

Masters change only between model forwards (optimizer steps, weight loads), never
between a forward, its checkpoint replay and its backward. Every encoding of a
parameter is therefore made once per :meth:`DeepseekV41Model.forward` and reused by
the replay and the backward. The cache is cleared at each model forward entry, after
the parameters are ready.
"""
import weakref

import torch

_CACHE = {}
_ENABLED = True


def clear(*, enabled=True):
    """Start a model forward; inference has no replay/backward to reuse encodings."""
    global _ENABLED
    _CACHE.clear()
    _ENABLED = enabled


def encoded(params, kind, produce):
    """``produce()`` once per model forward for these parameters and encoding ``kind``.

    Only ``nn.Parameter`` owners are cached (keyed by identity, guarded by weak
    references); anything else is encoded on every call."""
    if not _ENABLED or not params or not all(isinstance(p, torch.nn.Parameter) for p in params):
        return produce()
    key = (kind, *map(id, params))
    hit = _CACHE.get(key)
    if hit is None or any(ref() is not p for ref, p in zip(hit[0], params)):
        hit = _CACHE[key] = ([weakref.ref(p) for p in params], produce())
    return hit[1]
