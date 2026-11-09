# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""FP32 copy of a contiguous range of a BF16 or FP16 buffer.

``Tensor.float()`` on such a range reaches only a fraction of the HBM bandwidth on recent GPUs.
This kernel converts with 16-byte loads and streaming stores. The conversion is exact, so the
result equals ``Tensor.float()`` bit for bit.
"""

from unittest.mock import MagicMock

import torch

from megatron.core.utils import null_decorator

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False

if not HAVE_TRITON:
    triton = MagicMock()
    triton.jit = null_decorator
    tl = MagicMock()

_BLOCK = 4096
# Ranges are widened to multiples of this many elements, so that the kernel's accesses are
# unmasked 16-byte vectors.
_ALIGN = 16


@triton.jit
def _upcast_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, eviction_policy="evict_first")
    tl.store(y_ptr + offsets, x.to(tl.float32), mask=mask, cache_modifier=".cs")


def upcast_storage_range_to_fp32(tensor: torch.Tensor, start: int, end: int) -> torch.Tensor:
    """Return an FP32 tensor equal to elements ``[start, end)`` of ``tensor``'s storage.

    ``tensor`` is a BF16 or FP16 CUDA tensor; ``start`` and ``end`` are element offsets into
    its storage (not into the tensor). When the storage allows, the kernel converts the range
    widened to multiples of 16 elements, reading but not changing up to 15 neighbouring
    elements on each side, and a view of the requested range is returned.
    """
    assert tensor.dtype in (torch.bfloat16, torch.float16)
    if not HAVE_TRITON or end <= start or not tensor.is_cuda:
        return tensor.as_strided((end - start,), (1,), start).float()
    storage_numel = tensor.untyped_storage().nbytes() // tensor.element_size()
    lo = start - start % _ALIGN
    hi = -(-end // _ALIGN) * _ALIGN
    if hi > storage_numel or tensor.untyped_storage().data_ptr() % 16 != 0:
        lo, hi = start, end
    src = tensor.as_strided((hi - lo,), (1,), lo)
    out = torch.empty(hi - lo, dtype=torch.float32, device=tensor.device)
    grid = (triton.cdiv(hi - lo, _BLOCK),)
    _upcast_kernel[grid](src, out, hi - lo, BLOCK=_BLOCK, num_warps=4)
    return out[start - lo : end - lo]
