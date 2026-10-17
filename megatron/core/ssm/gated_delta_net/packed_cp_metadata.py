# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Reuse eager packed-sequence CP metadata within one microbatch."""

from collections import OrderedDict

import torch


def _tensor_version(tensor):
    try:
        return tensor._version
    except RuntimeError:
        # Inference tensors have no version counter, so do not reuse their contents.
        return None


def _cpu_cu_seqlens(packed_seq_params, cu_seqlens):
    """Copy offsets once per microbatch, refreshing after tracked tensor mutations."""
    if cu_seqlens.is_cuda and torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "Packed chunkwise CP metadata must be prepared outside CUDA graph capture."
        )
    version = _tensor_version(cu_seqlens)
    cache = getattr(packed_seq_params, '_gdn_cpu_offsets_cache', None)
    if cache is None:
        cache = OrderedDict()
        packed_seq_params._gdn_cpu_offsets_cache = cache
    key = id(cu_seqlens)
    cached = cache.get(key)
    if (
        version is not None
        and cached is not None
        and cached[0] is cu_seqlens
        and cached[1] == version
    ):
        cache.move_to_end(key)
        return cached[2]
    cpu_offsets = cu_seqlens.detach().cpu().clone()
    if version is not None:
        # Hold the source tensor itself so allocator/pointer reuse cannot alias an entry.
        cache[key] = (cu_seqlens, version, cpu_offsets)
        cache.move_to_end(key)
        while len(cache) > 4:
            cache.popitem(last=False)
    return cpu_offsets


def _packed_cp_context(packed_seq_params, cu_seqlens, group, conv_kernel_size, builder):
    """Reuse FLA's metadata-only CP context across GDN/KDA layers in a microbatch."""
    cpu_offsets = _cpu_cu_seqlens(packed_seq_params, cu_seqlens)
    version = _tensor_version(cu_seqlens)
    cache = getattr(packed_seq_params, '_gdn_cp_context_cache', None)
    if cache is None:
        cache = OrderedDict()
        packed_seq_params._gdn_cp_context_cache = cache
    key = (id(cu_seqlens), group, conv_kernel_size, builder)
    cached = cache.get(key)
    if (
        version is not None
        and cached is not None
        and cached[0] is cu_seqlens
        and cached[1] == version
    ):
        cache.move_to_end(key)
        return cached[2]
    context = builder(
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cpu_offsets,
        group=group,
        conv1d_kernel_size=conv_kernel_size,
    )
    if version is not None:
        cache[key] = (cu_seqlens, version, context)
        cache.move_to_end(key)
        while len(cache) > 8:
            cache.popitem(last=False)
    return context
