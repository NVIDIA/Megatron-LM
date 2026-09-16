# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Forked from `fla/utils.py`, `fla/ops/utils/op.py`, `fla/ops/utils/index.py` and
# `fla/modules/l2norm.py` in flash-linear-attention v0.5.1
# (https://github.com/fla-org/flash-linear-attention).
#
# Licensed under the MIT license; see the LICENSE file in the repository root.

"""Shared helpers for the Gated Delta Product kernels.

The kernel modules in this package reference a handful of names that do not
belong to any one of them: the hardware-capability probes that select autotune
configurations, the chunk-descriptor builders, `exp` / `exp2`, and the L2
normalization applied to the queries and keys. Keeping them here makes the
package self-contained, with no `fla` import at run time.

Take care when changing the probes: they choose which autotune configurations
exist, so editing one changes which kernel variants get benchmarked and picked.
Every GDP autotune config list must pass through `autotune_configs`: timing-based
selection can otherwise choose numerically different reduction tilings between
deterministic-mode processes.
"""

import functools
import os

import torch

from megatron.core.ssm.ops.common.determinism import autotune_configs

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:
    from unittest.mock import MagicMock

    from megatron.core.utils import null_decorator

    triton = MagicMock()
    triton.jit = null_decorator
    tl = MagicMock()
    HAVE_TRITON = False


# The Gated Delta Product chunk length is fixed at 64: `solve_tril`
# merges 16x16 blocks up to 64x64, the WY representation is built on 64-wide
# blocks, and the h kernel stores one state block per `num_householder`
# expanded chunks.
CHUNK_SIZE = 64

# 1/ln(2), used to convert natural-log decays into the base-2 space the chunked
# kernels exponentiate in. Best fp32 approximation (hex 0x3FB8AA3B).
RCP_LN2 = 1.4426950216


# ----------------------------------------------------------------------------
# Hardware capability probes (`fla.utils`).
# ----------------------------------------------------------------------------
def _is_nvidia() -> bool:
    return torch.cuda.is_available() and torch.version.hip is None


IS_NVIDIA = _is_nvidia()
IS_NVIDIA_HOPPER = IS_NVIDIA and (
    'NVIDIA H' in torch.cuda.get_device_name(0) or torch.cuda.get_device_capability()[0] >= 9
)
IS_NVIDIA_BLACKWELL = IS_NVIDIA and torch.cuda.get_device_capability()[0] in (10, 12)
IS_TMA_SUPPORTED = (
    IS_NVIDIA
    and torch.cuda.get_device_capability(0)[0] >= 9
    and os.environ.get('FLA_USE_TMA', '0') == '1'
    and (
        hasattr(tl, '_experimental_make_tensor_descriptor') or hasattr(tl, 'make_tensor_descriptor')
    )
)

if hasattr(tl, '_experimental_make_tensor_descriptor'):
    # Triton 3.3.x
    make_tensor_descriptor = tl._experimental_make_tensor_descriptor
elif hasattr(tl, 'make_tensor_descriptor'):
    # Triton 3.4.x and later
    make_tensor_descriptor = tl.make_tensor_descriptor
else:

    @triton.jit
    def make_tensor_descriptor(base, shape, strides, block_shape, _builder=None):
        """No-op stand-in for Triton builds without TMA support."""
        return None


# Shared-memory thresholds per architecture, from `fla.utils.Backend`.
_SHARED_MEM_BY_ARCH = {
    'ada': 101376,  # RTX 4090
    'ampere': 166912,  # A100
    'hopper': 232448,  # H100
    'none': 102400,  # default
}


@functools.cache
def check_shared_mem(arch: str = "none") -> bool:
    """Whether the current device has at least `arch`'s shared memory budget."""
    try:
        return torch.cuda.get_device_properties(
            0
        ).shared_memory_per_block_optin >= _SHARED_MEM_BY_ARCH.get(
            arch.lower(), _SHARED_MEM_BY_ARCH['none']
        )
    except Exception:
        return False


# ----------------------------------------------------------------------------
# `fla.ops.utils.op`
# ----------------------------------------------------------------------------
@triton.jit
def exp(x):
    """Exponentiate in fp32 regardless of the input dtype."""
    return tl.exp(x.to(tl.float32))


@triton.jit
def exp2(x):
    """Base-2 exponentiate in fp32 regardless of the input dtype."""
    return tl.math.exp2(x.to(tl.float32))


# ----------------------------------------------------------------------------
# `fla.ops.utils.index`
# ----------------------------------------------------------------------------
def _segmented_arange(counts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand per-segment counts into flat per-slot index tensors.

    Given segment sizes `counts = [c0, c1, ...]`, return two 1-D tensors of
    length `counts.sum()` labelling every slot with its segment and its
    position within that segment. For `counts = [2, 3]`::

        seg_id    = [0, 0, 1, 1, 1]
        intra_idx = [0, 1, 0, 1, 2]
    """
    seg_id = torch.repeat_interleave(
        torch.arange(counts.numel(), device=counts.device, dtype=counts.dtype), counts
    )
    seg_start = torch.nn.functional.pad(counts.cumsum(0), (1, 0))[:-1]
    intra_idx = (
        torch.arange(seg_id.shape[0], device=seg_id.device, dtype=counts.dtype) - seg_start[seg_id]
    )
    return seg_id, intra_idx


def prepare_chunk_indices(
    cu_seqlens: torch.Tensor, chunk_size: int, cu_seqlens_cpu: torch.Tensor | None = None
) -> torch.Tensor:
    """Flattened `(sequence, chunk-within-sequence)` pairs, one per chunk.

    Note that this synchronizes on the device when the counts live on the GPU:
    `repeat_interleave` reads `counts.sum()` on the host to size its output.
    Passing `cu_seqlens_cpu` avoids the sync.
    """
    src = cu_seqlens_cpu if cu_seqlens_cpu is not None else cu_seqlens
    counts = triton.cdiv(torch.diff(src), chunk_size)
    seg_id, intra_idx = _segmented_arange(counts)
    return torch.stack([seg_id, intra_idx], 1).to(cu_seqlens)


def prepare_chunk_offsets(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """Per-sequence prefix sum of chunk counts, with a leading zero."""
    lens = torch.diff(cu_seqlens)
    return torch.nn.functional.pad(triton.cdiv(lens, chunk_size), (1, 0), value=0).cumsum(-1)


# ----------------------------------------------------------------------------
# `fla.modules.l2norm`
# ----------------------------------------------------------------------------
_BT_LIST = [8, 16, 32, 64, 128]


@triton.autotune(
    configs=autotune_configs(
        [triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4, 8, 16, 32]]
    ),
    key=["D"],
)
@triton.jit
def l2norm_fwd_kernel1(x, y, rstd, eps, D, BD: tl.constexpr):
    """Row-per-program L2 normalization, used when `D > 512`."""
    i_t = tl.program_id(0)
    x += i_t * D
    y += i_t * D
    cols = tl.arange(0, BD)
    mask = cols < D

    b_x = tl.load(x + cols, mask=mask, other=0.0).to(tl.float32)
    b_rstd = 1 / tl.sqrt(tl.sum(b_x * b_x) + eps)
    b_y = b_x * b_rstd
    tl.store(y + cols, b_y, mask=mask)
    tl.store(rstd + i_t, b_rstd)


@triton.autotune(
    configs=autotune_configs(
        [
            triton.Config({"BT": BT}, num_warps=num_warps)
            for num_warps in [1, 2, 4, 8, 16]
            for BT in _BT_LIST
        ]
    ),
    key=["D", "NB"],
)
@triton.jit(do_not_specialize=["T"])
def l2norm_fwd_kernel(
    x, y, rstd, eps, T, D: tl.constexpr, BD: tl.constexpr, NB: tl.constexpr, BT: tl.constexpr
):
    """Block-of-rows L2 normalization, used when `D <= 512`."""
    i_t = tl.program_id(0)
    p_x = tl.make_block_ptr(x, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    p_y = tl.make_block_ptr(y, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    p_rstd = tl.make_block_ptr(rstd, (T,), (1,), (i_t * BT,), (BT,), (0,))

    b_x = tl.load(p_x, boundary_check=(0, 1)).to(tl.float32)
    b_rstd = 1 / tl.sqrt(tl.sum(b_x * b_x, 1) + eps)
    b_y = b_x * b_rstd[:, None]

    tl.store(p_y, b_y.to(p_y.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_rstd, b_rstd.to(p_rstd.dtype.element_ty), boundary_check=(0,))


def l2norm_fwd(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Row-wise L2 normalization over the last dimension, computed in fp32.

    The kernel also writes `rstd`, which only a backward pass would consume, so
    it is not returned.
    """
    x_shape_og = x.shape
    x = x.view(-1, x.shape[-1])
    y = torch.empty_like(x)
    assert y.stride(-1) == 1
    T, D = x.shape[0], x.shape[-1]
    # Less than 64KB per feature: enqueue the fused kernel.
    max_fused_size = 65536 // x.element_size()
    BD = min(max_fused_size, triton.next_power_of_2(D))
    if D > BD:
        raise RuntimeError("This layer doesn't support feature dim >= 64KB.")

    rstd = torch.empty((T,), dtype=torch.float32, device=x.device)
    if D <= 512:
        # Tolerate a wide range of T before recompiling, to limit autotuning.
        NB = triton.cdiv(T, 2048 * 32)

        def grid(meta):
            return (triton.cdiv(T, meta["BT"]),)

        l2norm_fwd_kernel[grid](x=x, y=y, rstd=rstd, eps=eps, T=T, D=D, BD=BD, NB=NB)
    else:
        l2norm_fwd_kernel1[(T,)](x=x, y=y, rstd=rstd, eps=eps, D=D, BD=BD)
    return y.view(x_shape_og)
