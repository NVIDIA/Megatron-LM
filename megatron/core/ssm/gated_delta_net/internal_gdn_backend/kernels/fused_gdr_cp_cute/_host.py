# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Derived from flash-linear-attention; see this package's __init__.py for the
# inline MIT license notice and upstream contributor link.

"""Host-side helpers shared by the forward and backward CP wrapper classes.

Everything here needs the CuTe DSL runtime, so only the GPU wrapper modules
(:mod:`.fused_ws`, :mod:`.bwd_ws`) import it.  ``_raw_stream`` lives in
:mod:`.backend` -- the dispatcher has to import without CUTLASS for its CPU
unit tests, and those tests monkeypatch the name there -- and is re-exported
from this module for the wrappers.
"""

from collections import OrderedDict

import cuda.bindings.driver as cuda
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_ptr

from .backend import _raw_stream  # noqa: F401  (re-export for the wrappers)

# CuTeDSL pointer objects memoized by device address. THD gives most arguments
# a fresh address every step (the producer window starts at a moving `bos`), so
# this must evict rather than fill and stop -- see `CPHostCaches._p`.
PTR_CACHE_SIZE = 4096


def _ptr(tensor: torch.Tensor, dtype):
    return make_ptr(dtype, tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)


def _normalize_gate(gate, dummy):
    """fp32-contiguous view of a gate slot, without a no-op ATen call.

    The dummies are already fp32 and contiguous, and the live gate arrives fp32
    from `chunk_local_cumsum` and contiguous by the caller's contract, so the
    unconditional `.float().contiguous()` this replaces was two ATen dispatches
    per gate slot that could not do anything but still cost host time on the
    per-step path.  The predicates are cheap C calls and keep the old behaviour
    for a caller that does hand over a bf16 or strided gate.
    """
    if gate is None:
        return dummy
    if gate.dtype is not torch.float32:
        gate = gate.float()
    return gate if gate.is_contiguous() else gate.contiguous()


class CPHostCaches:
    """Per-wrapper memoization of CuTeDSL pointer objects and CUstream wrappers.

    Wrappers call :meth:`_init_host_caches` from their constructor.
    """

    def _init_host_caches(self):
        self._ptr_cache: OrderedDict[tuple[int, object], object] = OrderedDict()
        self._stream_cache: dict[int, object] = {}

    def _p(self, tensor: torch.Tensor, dtype):
        """Memoize the CuTeDSL pointer object for a device address.

        LRU, not fill-and-stop.  Under THD the producer window starts at a
        different `bos` every step, so most arguments arrive with an address
        never seen before; a cache that stops inserting once full ends up
        holding only the addresses from the first thousand steps and misses on
        every call thereafter.  Evicting keeps the entries that recur (the
        symmetric buffers, the descriptors, the dummy gates) resident and bounds
        the retained pointer objects at the same time.
        """
        key = (tensor.data_ptr(), dtype)
        pointer = self._ptr_cache.get(key)
        if pointer is not None:
            self._ptr_cache.move_to_end(key)
            return pointer
        pointer = _ptr(tensor, dtype)
        self._ptr_cache[key] = pointer
        if len(self._ptr_cache) > PTR_CACHE_SIZE:
            self._ptr_cache.popitem(last=False)
        return pointer

    def _cu_stream(self, handle=None):
        """`cuda.CUstream` for the current torch stream, memoized by handle.

        The handle still has to be read every call -- the caller may have
        switched streams -- but `torch.cuda.current_stream()` builds a
        `torch.cuda.Stream` object to then throw it away, which costs ~3.1 us
        against 0.09 us for the raw accessor. Wrapping the handle allocates a
        ctypes object too, and a training run uses a handful of distinct
        streams for its whole life, so that is cached on the handle.
        """
        if handle is None:
            handle = _raw_stream(self.device.index)
        stream = self._stream_cache.get(handle)
        if stream is None:
            stream = cuda.CUstream(handle)
            self._stream_cache[handle] = stream
        return stream
