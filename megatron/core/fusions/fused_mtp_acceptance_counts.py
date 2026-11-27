# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""MTP acceptance counts (correct and valid positions) in one kernel.

The MTP acceptance metric compares the ``[s, b]`` argmax predictions with the ``[b, s]`` labels
at the positions the ``[b, s]`` loss mask keeps. In PyTorch that is a chain of small kernels per
microbatch: two transposes to contiguous ``[s, b]`` copies, a cast of the mask to bool, a
comparison, a logical and, two sums and two casts to float. This kernel reads the three tensors
through their strides in a single program and writes both counts as float32 scalars.

The counts are integers accumulated in int32 and converted once to float32. Integer addition
is associative, so the result does not depend on the reduction order, and every count the
kernel produces is far below 2**24, where float32 holds every integer exactly. The result is
bit-identical to the PyTorch reference.

The kernel is one program, which is the right shape for the per-microbatch sizes it targets
(a few thousand to tens of thousands of positions) but not for very large inputs, so inputs
above ``_MAX_KERNEL_NUMEL`` positions take the PyTorch path.
"""

from typing import Tuple
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

# Largest number of positions handled by the single-program kernel. Beyond roughly 64K
# positions the multi-block PyTorch reductions are faster. Must stay below 2**24 so the
# float32 counts are exact.
_MAX_KERNEL_NUMEL = 1 << 15
_BLOCK = 1024

_INDEX_DTYPES = (torch.int32, torch.int64)
_MASK_DTYPES = (
    torch.bool,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
)


@triton.jit
def _mtp_acceptance_counts_kernel(
    preds_ptr,
    labels_ptr,
    mask_ptr,
    correct_ptr,
    total_ptr,
    seq_len,
    batch,
    preds_stride_s,
    preds_stride_b,
    labels_stride_b,
    labels_stride_s,
    mask_stride_b,
    mask_stride_s,
    BLOCK: tl.constexpr,
):
    """Count the valid positions and the valid positions where preds == labels.

    preds is [seq_len, batch]; labels and mask are [batch, seq_len]. One program walks all
    seq_len * batch positions in BLOCK-sized chunks.
    """
    correct_acc = tl.zeros([BLOCK], dtype=tl.int32)
    total_acc = tl.zeros([BLOCK], dtype=tl.int32)
    numel = seq_len * batch
    for start in range(0, numel, BLOCK):
        idx = start + tl.arange(0, BLOCK)
        in_range = idx < numel
        s_idx = (idx // batch).to(tl.int64)
        b_idx = (idx % batch).to(tl.int64)
        pred = tl.load(
            preds_ptr + s_idx * preds_stride_s + b_idx * preds_stride_b, mask=in_range, other=0
        )
        label = tl.load(
            labels_ptr + b_idx * labels_stride_b + s_idx * labels_stride_s, mask=in_range, other=0
        )
        keep = tl.load(
            mask_ptr + b_idx * mask_stride_b + s_idx * mask_stride_s, mask=in_range, other=0
        )
        valid = (keep != 0) & in_range
        correct_acc += ((pred == label) & valid).to(tl.int32)
        total_acc += valid.to(tl.int32)
    tl.store(correct_ptr, tl.sum(correct_acc, axis=0).to(tl.float32))
    tl.store(total_ptr, tl.sum(total_acc, axis=0).to(tl.float32))


def mtp_acceptance_counts_reference(
    preds: torch.Tensor, labels: torch.Tensor, loss_mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """PyTorch implementation of :func:`fused_mtp_acceptance_counts`."""
    labels_match = labels.transpose(0, 1).contiguous()  # [b, s] => [s, b]
    mask_match = loss_mask.transpose(0, 1).contiguous()  # [b, s] => [s, b]
    valid_positions = mask_match.bool()
    correct = ((preds == labels_match) & valid_positions).sum().float()
    total = valid_positions.sum().float()
    return correct, total


def _kernel_supports(preds: torch.Tensor, labels: torch.Tensor, loss_mask: torch.Tensor) -> bool:
    if not HAVE_TRITON:
        return False
    if preds.dim() != 2 or labels.dim() != 2 or loss_mask.dim() != 2:
        return False
    seq_len, batch = preds.shape
    if tuple(labels.shape) != (batch, seq_len) or tuple(loss_mask.shape) != (batch, seq_len):
        return False
    if not (preds.is_cuda and labels.device == preds.device and loss_mask.device == preds.device):
        return False
    if preds.dtype not in _INDEX_DTYPES or labels.dtype not in _INDEX_DTYPES:
        return False
    if loss_mask.dtype not in _MASK_DTYPES:
        return False
    return 0 < preds.numel() <= _MAX_KERNEL_NUMEL


def fused_mtp_acceptance_counts(
    preds: torch.Tensor, labels: torch.Tensor, loss_mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Number of correct and of valid MTP positions.

    Args:
        preds (torch.Tensor): ``[s, b]`` predicted token ids.
        labels (torch.Tensor): ``[b, s]`` label token ids.
        loss_mask (torch.Tensor): ``[b, s]`` mask; nonzero entries are valid positions.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: ``(correct, total)`` as float32 0-dim tensors on the
        device of ``preds``, equal to :func:`mtp_acceptance_counts_reference`. Inputs the kernel
        does not handle (CPU tensors, other shapes or dtypes, more than ``_MAX_KERNEL_NUMEL``
        positions, or Triton unavailable) take the PyTorch path.
    """
    if not _kernel_supports(preds, labels, loss_mask):
        return mtp_acceptance_counts_reference(preds, labels, loss_mask)
    seq_len, batch = preds.shape
    correct = torch.empty((), dtype=torch.float32, device=preds.device)
    total = torch.empty((), dtype=torch.float32, device=preds.device)
    # bool tensors are loaded through a uint8 view; strides are unchanged.
    mask = loss_mask.view(torch.uint8) if loss_mask.dtype == torch.bool else loss_mask
    _mtp_acceptance_counts_kernel[(1,)](
        preds,
        labels,
        mask,
        correct,
        total,
        seq_len,
        batch,
        preds.stride(0),
        preds.stride(1),
        labels.stride(0),
        labels.stride(1),
        mask.stride(0),
        mask.stride(1),
        BLOCK=_BLOCK,
        num_warps=8,
    )
    return correct, total
