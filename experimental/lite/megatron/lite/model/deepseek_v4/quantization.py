"""DeepSeek-V4 checkpoint quantization policy shared by load and resync."""

from dataclasses import dataclass

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU-only checkpoint tooling
    triton = tl = None

BLOCK_SHAPE = (128, 128)


@dataclass(frozen=True)
class CanonicalBlockFP8Weight:
    qweight: torch.Tensor
    scales: torch.Tensor
    block_shape: tuple[int, int] = BLOCK_SHAPE


def _validate_weight(weight: torch.Tensor) -> None:
    if weight.dtype != torch.bfloat16 or weight.ndim != 2:
        raise TypeError("block-FP8 master weight must be a 2-D BF16 tensor")
    if any(size % block for size, block in zip(weight.shape, BLOCK_SHAPE, strict=True)):
        raise ValueError(f"weight shape {tuple(weight.shape)} must be divisible by {BLOCK_SHAPE}")


if triton is not None:

    @triton.jit
    def _requantize(master, scales, output, columns, scale_columns, elements, BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < elements
        rows = offsets // columns
        columns_in_row = offsets - rows * columns
        scale_offsets = (rows // 128) * scale_columns + columns_in_row // 128
        values = tl.load(master + offsets, mask=mask).to(tl.float32)
        tl.store(output + offsets, values / tl.load(scales + scale_offsets, mask=mask), mask=mask)


def requantize_block_fp8_weight(
    weight: torch.Tensor, scales: torch.Tensor
) -> CanonicalBlockFP8Weight:
    """Recover release FP8 bytes using the release scales, without rescaling."""
    _validate_weight(weight)
    expected = (weight.shape[0] // 128, weight.shape[1] // 128)
    if scales.dtype != torch.float32 or tuple(scales.shape) != expected or scales.device != weight.device:
        raise ValueError(f"fixed scales must be float32 {expected} on the weight device")
    if not bool(torch.all(torch.isfinite(scales) & (scales > 0))):
        raise ValueError("fixed scales must be finite and positive")
    with torch.no_grad():
        if weight.is_cuda and triton is not None:
            qweight = torch.empty_like(weight, dtype=torch.float8_e4m3fn)
            _requantize[(triton.cdiv(weight.numel(), 256),)](
                weight,
                scales.contiguous(),
                qweight,
                weight.shape[1],
                scales.shape[1],
                weight.numel(),
                BLOCK=256,
            )
        else:
            expanded = scales.repeat_interleave(128, 0).repeat_interleave(128, 1)
            qweight = (weight.float() / expanded).to(torch.float8_e4m3fn)
    return CanonicalBlockFP8Weight(qweight, scales)


def is_release_unquantized_weight(name: str) -> bool:
    """Return whether the official V4 release stores ``name`` unscaled."""

    if name in {"embed.weight", "head.weight", "norm.weight"}:
        return True
    if name.endswith("norm.weight") or name.endswith(".ffn.gate.weight"):
        return True
    if ".attn.compressor." in name:
        return True
    return ".attn.indexer." in name and not name.endswith(
        ".attn.indexer.wq_b.weight"
    )


__all__ = [
    "BLOCK_SHAPE",
    "CanonicalBlockFP8Weight",
    "is_release_unquantized_weight",
    "requantize_block_fp8_weight",
]
