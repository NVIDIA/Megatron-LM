# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Fixed-threshold rounding for FP8 primary weights with the blockwise recipe.

With ``--fp8-param-gather`` and the blockwise recipe, the distributed optimizer casts each FP32
master shard to E4M3 after every step (``quantize_param_shard``), and Transformer Engine rounds
``master * scale`` to nearest. RL learning rates move a weight by about 1e-6 per step, while an
E4M3 cell is about ``|w| / 8`` wide, so two problems follow.

If the master is built from the dequantized FP8 parameter (the default fine-tune load), every weight
starts on a grid point. Its update is rounded away until the weight has drifted half a cell, and
learning stalls. If the master is built from bf16 checkpoint values with power-of-two scales, 1/16
of the weights sit exactly on a rounding tie. The first update resolves all of them in the direction
of its own sign, which moves the FP8 policy by a whole cell on about 3% of weights at once, whatever
the learning rate.

The fixed-threshold cast gives each weight a threshold ``1/2 + u`` with ``u`` in ``[-1/2, 1/2)``,
fixed for the run. The magnitude ``|master * scale|`` rounds up from its lower E4M3 neighbour iff
its position in the cell is at least that threshold. Averaged over ``u`` the cast is unbiased, grid
points stay put, and nothing sits on a tie. A weight changes code with probability
``|drift| / cell``, against 0 until half a cell under round to nearest. ``u`` is a hash of the
weight's flat index in its full tensor, so nothing is stored, the cast is deterministic, and every
data-parallel shard agrees.

The cast is applied by giving TE's existing kernel a temporary copy of the master shard snapped to
the chosen grid value, so the kernel's round to nearest returns that value unchanged. The master
itself is never modified.
"""

import contextlib
from typing import Dict, List

import torch

E4M3_MAX = 448.0


def _hash_base(seed: int, rows: int, cols: int) -> int:
    return (seed * 0x2C1B3C6D + rows * 0x297A2D39 + cols * 0x1B873593) & 0x7FFFFFFF


def _threshold_offsets(seed: int, rows: int, cols: int, flat_index: torch.Tensor) -> torch.Tensor:
    """Per-weight offsets u in [-1/2, 1/2), an integer hash of (seed, tensor shape, flat index).

    The hash runs in wrapping int32 arithmetic with constants below 2^31, so eager PyTorch and
    torch.compile agree.
    """
    base = torch.tensor(_hash_base(seed, rows, cols), dtype=torch.int32, device=flat_index.device)
    return _offsets_from_base(base, flat_index)


def _offsets_from_base(base: torch.Tensor, flat_index: torch.Tensor) -> torch.Tensor:
    low = (flat_index & 0x7FFFFFFF).to(torch.int32)
    high = (flat_index >> 31).to(torch.int32)
    x = low * 0x61C88647 + high * 0x5BD1E995 + base
    for multiplier, shift in ((0x7FEB352D, 15), (0x2C1B3C6D, 13)):
        x = x ^ ((x >> 16) & 0xFFFF)
        x = x * multiplier
        x = x ^ ((x >> shift) & ((1 << (32 - shift)) - 1))
    return ((x >> 1) & 0x7FFFFFFF).to(torch.float64).div(2.0**31).sub(0.5).to(torch.float32)


def _e4m3_cell(magnitude: torch.Tensor):
    """Lower E4M3 neighbour and cell width for magnitudes in [0, 448]. Subnormal spacing is 2^-9."""
    exponent = torch.floor(torch.log2(magnitude.clamp_min(2.0**-6)))
    width = torch.exp2(exponent - 3)
    return torch.floor(magnitude / width) * width, width


def fixed_threshold_e4m3(scaled: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """Round scaled values to E4M3 with per-element thresholds 1/2 + offsets."""
    magnitude = scaled.abs().clamp(max=E4M3_MAX)
    lower, width = _e4m3_cell(magnitude)
    position = (magnitude - lower) / width
    rounded = torch.where(position >= 0.5 + offsets, lower + width, lower).clamp(max=E4M3_MAX)
    return torch.sign(scaled) * rounded


def snap_master_shard(
    master: torch.Tensor,
    scale: torch.Tensor,
    rows: int,
    cols: int,
    start_offset: int,
    block_len: int,
    seed: int = 0,
) -> torch.Tensor:
    """Return a copy of a flat FP32 master shard whose round-to-nearest E4M3 cast is the
    fixed-threshold cast.

    ``master`` holds flat elements ``[start_offset, start_offset + numel)`` of a ``rows x cols``
    weight, and ``scale`` holds TE's per-block multipliers (``max_fp8 / amax``) with shape
    ``(ceil(rows / block_len), ceil(cols / block_len))``.
    """
    flat = master.reshape(-1).float()
    snapped = _snap_flat(flat, scale.float(), rows, cols, start_offset, block_len, seed)
    return snapped.to(master.dtype).reshape(master.shape)


def _snap_flat(flat, scale, rows: int, cols: int, start_offset: int, block_len: int, seed: int):
    base = torch.tensor(_hash_base(seed, rows, cols), dtype=torch.int32, device=flat.device)
    snap = _snap_compiled() if flat.is_cuda else _snap_with_base
    return snap(flat, scale, base, cols, start_offset, block_len)


def _snap_with_base(flat, scale, base, cols: int, start_offset: int, block_len: int):
    index = torch.arange(
        start_offset, start_offset + flat.numel(), device=flat.device, dtype=torch.int64
    )
    block_scale = scale[(index // cols) // block_len, (index % cols) // block_len]
    offsets = _offsets_from_base(base, index)
    return fixed_threshold_e4m3(flat * block_scale, offsets) / block_scale


_SNAP_COMPILED = None


def _snap_compiled():
    """The snap fused into one kernel with torch.compile (about 7x faster than eager on an H100)."""
    global _SNAP_COMPILED
    if _SNAP_COMPILED is None:
        _SNAP_COMPILED = torch.compile(_snap_with_base, dynamic=True)
    return _SNAP_COMPILED


@contextlib.contextmanager
def fixed_threshold_blockwise_cast(seed: int = 0):
    """Within this context, TE's blockwise master-to-FP8 cast uses fixed-threshold rounding."""
    import transformer_engine.pytorch.tensor.utils as te_tensor_utils

    tex = te_tensor_utils.tex
    nearest_cast = tex.fp8_block_scaling_partial_cast

    def cast(master, out, scale, rows, cols, start_offset, block_len, fp8_dtype):
        snapped = snap_master_shard(master, scale, rows, cols, start_offset, block_len, seed)
        return nearest_cast(snapped, out, scale, rows, cols, start_offset, block_len, fp8_dtype)

    tex.fp8_block_scaling_partial_cast = cast
    try:
        yield
    finally:
        tex.fp8_block_scaling_partial_cast = nearest_cast


def _blockwise_scales(weight: torch.Tensor, block_len: int, power_of_two: bool) -> torch.Tensor:
    rows, cols = weight.shape
    pad_rows, pad_cols = (-rows) % block_len, (-cols) % block_len
    padded = torch.nn.functional.pad(weight.abs().float(), (0, pad_cols, 0, pad_rows))
    amax = padded.view(
        (rows + pad_rows) // block_len, block_len, (cols + pad_cols) // block_len, block_len
    ).amax(dim=(1, 3))
    scale = torch.where(amax > 0, E4M3_MAX / amax.clamp_min(1e-30), torch.ones_like(amax))
    return torch.exp2(torch.floor(torch.log2(scale))) if power_of_two else scale


@torch.no_grad()
def fp8_master_census(
    weights: List[torch.Tensor], block_len: int = 128, power_of_two: bool = True
) -> Dict[str, float]:
    """Fractions of FP32 master weights on the E4M3 grid and on round-to-nearest ties under
    blockwise scales.

    A grid fraction near 1 means the master was built from the dequantized FP8 parameter, where
    round to nearest stalls small updates. A tie fraction near 1/16 means bf16-valued weights under
    power-of-two scales, where the first update moves every tie at once.
    """
    total = on_grid = on_tie = 0
    for weight in weights:
        if weight.dim() != 2:
            continue
        rows, cols = weight.shape
        scale = _blockwise_scales(weight, block_len, power_of_two)
        full = scale.repeat_interleave(block_len, 0)[:rows].repeat_interleave(block_len, 1)[
            :, :cols
        ]
        magnitude = (weight.float() * full).abs().clamp(max=E4M3_MAX)
        lower, width = _e4m3_cell(magnitude)
        position = (magnitude - lower) / width
        total += weight.numel()
        on_grid += int(((position == 0) & (magnitude > 0)).sum())
        on_tie += int((position == 0.5).sum())
    return {"on_grid": on_grid / max(total, 1), "on_tie": on_tie / max(total, 1)}
