"""DeepSeek-V4 BF16-master adapters over vLLM's block-FP8 deployment path."""

from __future__ import annotations

import importlib
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn

from megatron.lite.model.deepseek_v4 import quantization as ds4_quantization
from megatron.lite.model.deepseek_v4.quantization import (
    BLOCK_SHAPE,
    CanonicalBlockFP8Weight,
    requantize_block_fp8_weight,
)

triton = ds4_quantization.triton
tl = ds4_quantization.tl


if triton is not None:

    @triton.jit
    def _dynamic_block_fp8_quantize(
        weight,
        qweight,
        scales,
        columns: tl.constexpr,
        scale_columns: tl.constexpr,
        BLOCK_ELEMENTS: tl.constexpr,
    ):
        block = tl.program_id(0)
        block_row = block // scale_columns
        block_col = block - block_row * scale_columns
        local = tl.arange(0, BLOCK_ELEMENTS)
        row = block_row * 128 + local // 128
        column = block_col * 128 + local % 128
        offsets = row * columns + column
        values = tl.load(weight + offsets).to(tl.float32)
        amax = tl.maximum(tl.max(tl.abs(values), axis=0), 1.0e-4)
        scale = amax / 448.0
        tl.store(scales + block, scale)
        tl.store(qweight + offsets, values * (1.0 / scale))

    @triton.jit
    def _requantize_block_fp8_to_ue8m0(
        qweight,
        scales,
        ue8m0_scales,
        columns: tl.constexpr,
        scale_columns: tl.constexpr,
        BLOCK_ELEMENTS: tl.constexpr,
    ):
        block = tl.program_id(0)
        block_row = block // scale_columns
        block_col = block - block_row * scale_columns
        local = tl.arange(0, BLOCK_ELEMENTS)
        row = block_row * 128 + local // 128
        column = block_col * 128 + local % 128
        offsets = row * columns + column
        dequantized = tl.load(qweight + offsets).to(tl.float32) * tl.load(
            scales + block
        )
        amax = tl.maximum(tl.max(tl.abs(dequantized), axis=0), 1.0e-4)
        exponent = tl.ceil(tl.log2(amax / 448.0))
        scale = tl.exp2(exponent)
        tl.store(qweight + offsets, dequantized * (1.0 / scale))
        tl.store(ue8m0_scales + block, exponent + 127.0)


@contextmanager
def _weight_nvtx_range(name: str):
    if os.environ.get("MLITE_STEP_NVTX") != "1" or not torch.cuda.is_available():
        yield
        return
    with torch.cuda.nvtx.range(name):
        yield


@dataclass(frozen=True)
class PackedBlockFP8Weight:
    qweight: torch.Tensor
    scales: torch.Tensor
    cache_key: object | None = None


@dataclass(frozen=True)
class PackedBlockFP8Activation:
    qactivation: torch.Tensor
    scales: torch.Tensor


def _key(weight: nn.Parameter):
    return (
        id(weight),
        weight._version,
        weight.device,
        weight.dtype,
        tuple(weight.shape),
    )


def _validate_weight(weight: torch.Tensor) -> None:
    if weight.dtype != torch.bfloat16 or weight.ndim != 2:
        raise TypeError("block-FP8 master weight must be a 2-D BF16 tensor")
    if any(size % block for size, block in zip(weight.shape, BLOCK_SHAPE, strict=True)):
        raise ValueError(
            f"weight shape {tuple(weight.shape)} must be divisible by {BLOCK_SHAPE}"
        )


def _quantize_block_fp8_weight_fused(
    weight: torch.Tensor,
) -> CanonicalBlockFP8Weight:
    qweight = torch.empty_like(weight, dtype=torch.float8_e4m3fn)
    scales = torch.empty(
        (weight.shape[0] // BLOCK_SHAPE[0], weight.shape[1] // BLOCK_SHAPE[1]),
        dtype=torch.float32,
        device=weight.device,
    )
    _dynamic_block_fp8_quantize[(scales.numel(),)](
        weight,
        qweight,
        scales,
        columns=weight.shape[1],
        scale_columns=scales.shape[1],
        BLOCK_ELEMENTS=BLOCK_SHAPE[0] * BLOCK_SHAPE[1],
        num_warps=8,
    )
    return CanonicalBlockFP8Weight(qweight, scales)


def _quantize_block_fp8_weight_fused_ue8m0_out(
    weight: torch.Tensor,
    qweight: torch.Tensor,
    ue8m0_scales: torch.Tensor,
) -> None:
    scales = torch.empty_like(ue8m0_scales, dtype=torch.float32)
    _dynamic_block_fp8_quantize[(scales.numel(),)](
        weight,
        qweight,
        scales,
        columns=weight.shape[1],
        scale_columns=scales.shape[1],
        BLOCK_ELEMENTS=BLOCK_SHAPE[0] * BLOCK_SHAPE[1],
        num_warps=8,
    )
    _requantize_block_fp8_to_ue8m0[(scales.numel(),)](
        qweight,
        scales,
        ue8m0_scales,
        columns=weight.shape[1],
        scale_columns=scales.shape[1],
        BLOCK_ELEMENTS=BLOCK_SHAPE[0] * BLOCK_SHAPE[1],
        num_warps=8,
    )


def _quantize_block_fp8_weight_fused_ue8m0(
    weight: torch.Tensor,
) -> CanonicalBlockFP8Weight:
    qweight = torch.empty_like(weight, dtype=torch.float8_e4m3fn)
    ue8m0_scales = torch.empty(
        (weight.shape[0] // BLOCK_SHAPE[0], weight.shape[1] // BLOCK_SHAPE[1]),
        dtype=torch.uint8,
        device=weight.device,
    )
    _quantize_block_fp8_weight_fused_ue8m0_out(
        weight, qweight, ue8m0_scales
    )
    return CanonicalBlockFP8Weight(qweight, ue8m0_scales)


def quantize_block_fp8_weight(weight: torch.Tensor):
    _validate_weight(weight)
    scales = getattr(weight, "_fp8_source_scales", None)
    if (
        scales is not None
        and getattr(weight, "_fp8_source_scale_version", None) == weight._version
    ):
        if weight.is_cuda and ds4_quantization.triton is not None:
            # Checkpoint loading already validates positivity, finiteness, and
            # exact BF16->FP8 reversibility before binding these source scales.
            # Repeating torch.all(...)->bool for every lazy deployment pack
            # introduces a device-to-host synchronization and can exhaust the
            # CUDA launch path while a full 43-layer EP model is first packed.
            with torch.no_grad(), _weight_nvtx_range("fp8_weight/source_requantize"):
                qweight = torch.empty_like(weight, dtype=torch.float8_e4m3fn)
                ds4_quantization._requantize[
                    ((weight.numel() + 255) // 256,)
                ](
                    weight,
                    scales.contiguous(),
                    qweight,
                    weight.shape[1],
                    scales.shape[1],
                    weight.numel(),
                    BLOCK=256,
                )
            return CanonicalBlockFP8Weight(qweight, scales)
        return requantize_block_fp8_weight(weight, scales)
    with torch.no_grad(), _weight_nvtx_range("fp8_weight/dynamic_quantize"):
        if (
            weight.is_cuda
            and triton is not None
            and os.environ.get("MLITE_VLLM_FUSED_WEIGHT_QUANT", "1") != "0"
        ):
            if (
                os.environ.get(
                    "MLITE_VLLM_FUSED_UE8M0_WEIGHT_QUANT", "1"
                )
                != "0"
            ):
                return _quantize_block_fp8_weight_fused_ue8m0(weight.detach())
            return _quantize_block_fp8_weight_fused(weight.detach())
        from vllm.utils.deep_gemm import per_block_cast_to_fp8

        qweight, scales = per_block_cast_to_fp8(
            weight.detach(), block_size=list(BLOCK_SHAPE), use_ue8m0=False
        )
        return CanonicalBlockFP8Weight(qweight, scales)


def _grouped_checkpoint_weights(
    weights: tuple[nn.Parameter, ...],
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Requantize trusted source weights directly into their final stack."""

    if not weights or not weights[0].is_cuda or ds4_quantization.triton is None:
        return None
    scales = tuple(getattr(weight, "_fp8_source_scales", None) for weight in weights)
    if any(
        scale is None
        or getattr(weight, "_fp8_source_scale_version", None) != weight._version
        for weight, scale in zip(weights, scales, strict=True)
    ):
        return None
    for weight in weights:
        _validate_weight(weight)
        if weight.shape != weights[0].shape:
            raise ValueError("grouped block-FP8 weights must have identical shapes")
    qweight = torch.empty(
        (len(weights), *weights[0].shape),
        dtype=torch.float8_e4m3fn,
        device=weights[0].device,
    )
    with torch.no_grad():
        for index, (weight, scale) in enumerate(
            zip(weights, scales, strict=True)
        ):
            ds4_quantization._requantize[
                ((weight.numel() + 255) // 256,)
            ](
                weight,
                scale,
                qweight[index],
                weight.shape[1],
                scale.shape[1],
                weight.numel(),
                BLOCK=256,
            )
    return qweight, torch.stack(scales)


def _quantize_grouped_block_fp8_weights_direct(
    weights: tuple[nn.Parameter, ...],
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if (
        not weights
        or not weights[0].is_cuda
        or triton is None
        or os.environ.get("MLITE_VLLM_FUSED_WEIGHT_QUANT", "1") == "0"
        or os.environ.get("MLITE_VLLM_FUSED_UE8M0_WEIGHT_QUANT", "1") == "0"
    ):
        return None
    for weight in weights:
        _validate_weight(weight)
        if weight.shape != weights[0].shape:
            raise ValueError("grouped block-FP8 weights must have identical shapes")
    qweight = torch.empty(
        (len(weights), *weights[0].shape),
        dtype=torch.float8_e4m3fn,
        device=weights[0].device,
    )
    scales = torch.empty(
        (
            len(weights),
            weights[0].shape[0] // BLOCK_SHAPE[0],
            weights[0].shape[1] // BLOCK_SHAPE[1],
        ),
        dtype=torch.uint8,
        device=weights[0].device,
    )
    with torch.no_grad():
        for index, weight in enumerate(weights):
            _quantize_block_fp8_weight_fused_ue8m0_out(
                weight.detach(), qweight[index], scales[index]
            )
    return qweight, scales


def _quantize_concatenated_block_fp8_weights_direct(
    weights: tuple[nn.Parameter, ...],
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if (
        not weights
        or not weights[0].is_cuda
        or triton is None
        or os.environ.get("MLITE_VLLM_FUSED_WEIGHT_QUANT", "1") == "0"
        or os.environ.get("MLITE_VLLM_FUSED_UE8M0_WEIGHT_QUANT", "1") == "0"
        or any(
            getattr(weight, "_fp8_source_scales", None) is not None
            and getattr(weight, "_fp8_source_scale_version", None) == weight._version
            for weight in weights
        )
    ):
        return None
    columns = weights[0].shape[1]
    for weight in weights:
        _validate_weight(weight)
        if weight.shape[1] != columns:
            raise ValueError("fused block-FP8 weights must have identical K dimensions")
    qweight = torch.empty(
        (sum(weight.shape[0] for weight in weights), columns),
        dtype=torch.float8_e4m3fn,
        device=weights[0].device,
    )
    scales = torch.empty(
        (
            sum(weight.shape[0] // BLOCK_SHAPE[0] for weight in weights),
            columns // BLOCK_SHAPE[1],
        ),
        dtype=torch.uint8,
        device=weights[0].device,
    )
    row_offset = 0
    scale_row_offset = 0
    with torch.no_grad():
        for weight in weights:
            rows = weight.shape[0]
            scale_rows = rows // BLOCK_SHAPE[0]
            _quantize_block_fp8_weight_fused_ue8m0_out(
                weight.detach(),
                qweight.narrow(0, row_offset, rows),
                scales.narrow(0, scale_row_offset, scale_rows),
            )
            row_offset += rows
            scale_row_offset += scale_rows
    return qweight, scales


def bind_source_scale_to_visible_weight(
    module: nn.Module, parameter_name: str, weight: torch.Tensor
):
    """Apply model-owned checkpoint scale metadata to the visible weight."""

    scales = getattr(module, "_fp8_source_scales_by_parameter", {}).get(parameter_name)
    if scales is None:
        for attribute in ("_fp8_source_scales", "_fp8_source_scale_version"):
            if hasattr(weight, attribute):
                delattr(weight, attribute)
        return weight
    weight._fp8_source_scales = scales
    weight._fp8_source_scale_version = weight._version
    return weight


def _post_process(qweight, scales):
    with torch.no_grad(), _weight_nvtx_range("fp8_weight/post_process"):
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            deepgemm_post_process_fp8_weight_block,
        )

        return deepgemm_post_process_fp8_weight_block(
            wq=qweight, ws=scales, quant_block_shape=BLOCK_SHAPE, use_e8m0=True
        )


def pack_block_fp8_weight(weight: nn.Parameter):
    canonical = quantize_block_fp8_weight(weight)
    qweight, scales = _post_process(canonical.qweight, canonical.scales)
    return PackedBlockFP8Weight(qweight, scales, _key(weight))


def pack_grouped_block_fp8_weight(weights: Iterable[nn.Parameter]):
    weights = tuple(weights)
    if not weights:
        raise ValueError("grouped block-FP8 packing requires at least one expert")
    checkpoint_weights = _grouped_checkpoint_weights(weights)
    if checkpoint_weights is None:
        checkpoint_weights = _quantize_grouped_block_fp8_weights_direct(weights)
        if checkpoint_weights is None:
            canonical = tuple(quantize_block_fp8_weight(weight) for weight in weights)
            checkpoint_weights = (
                torch.stack([item.qweight for item in canonical]),
                torch.stack([item.scales for item in canonical]),
            )
    qweight, scales = _post_process(*checkpoint_weights)
    return PackedBlockFP8Weight(
        qweight, scales, tuple(_key(weight) for weight in weights)
    )


def pack_block_fp8_activation(x: torch.Tensor):
    if (
        x.dtype != torch.bfloat16
        or x.ndim != 2
        or x.shape[1] % 128
        or x.stride(-1) != 1
    ):
        raise ValueError(
            "block-FP8 activation must be contiguous 2-D BF16 with K divisible by 128"
        )
    from vllm.utils.deep_gemm import DeepGemmQuantScaleFMT

    oracle = DeepGemmQuantScaleFMT.from_oracle()
    packed = getattr(oracle, "name", "") == "UE8M0"
    fp8_utils = importlib.import_module(
        "vllm.model_executor.layers.quantization.utils.fp8_utils"
    )
    quantize = getattr(
        fp8_utils,
        "per_token_group_quant_fp8_packed_for_deepgemm"
        if packed
        else "per_token_group_quant_fp8",
    )
    with torch.no_grad():
        if packed:
            qactivation, scales = quantize(x, 128, use_ue8m0=True)
        else:
            from vllm.envs import VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES

            tma = bool(VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES)
            qactivation, scales = quantize(
                x, 128, use_ue8m0=True, column_major_scales=True, tma_aligned_scales=tma
            )
    return PackedBlockFP8Activation(qactivation, scales)


def fp8_gemm_nt(x: torch.Tensor, weight: PackedBlockFP8Weight):
    if x.ndim != 2 or x.shape[1] != weight.qweight.shape[-1]:
        raise ValueError("activation K does not match packed weight K")
    activation = pack_block_fp8_activation(x)
    output = torch.empty(
        (x.shape[0], weight.qweight.shape[-2]), dtype=torch.bfloat16, device=x.device
    )
    with torch.no_grad():
        from vllm.utils.deep_gemm import fp8_gemm_nt as deep_gemm_fp8_gemm_nt

        deep_gemm_fp8_gemm_nt(
            (activation.qactivation, activation.scales),
            (weight.qweight, weight.scales),
            output,
            is_deep_gemm_e8m0_used=True,
        )
    return output


class DeploymentBlockFP8Adapter:
    def __init__(self, *, cache_weight: bool = False):
        self.cache_weight = cache_weight
        self._cached_weight = None

    def clear_cache(self):
        self._cached_weight = None

    def pack_weight(self, weight: nn.Parameter):
        key = _key(weight)
        if (
            self.cache_weight
            and self._cached_weight is not None
            and self._cached_weight.cache_key == key
        ):
            with _weight_nvtx_range("fp8_weight/cache_hit"):
                return self._cached_weight
        with _weight_nvtx_range("fp8_weight/cache_miss"):
            packed = pack_block_fp8_weight(weight)
        if self.cache_weight:
            self._cached_weight = packed
        return packed

    def __call__(self, x, weight):
        return fp8_gemm_nt(x, self.pack_weight(weight))


class DeploymentFusedBlockFP8Adapter(DeploymentBlockFP8Adapter):
    def pack_weight(self, weights):
        weights = tuple(weights)
        key = tuple(_key(weight) for weight in weights)
        if (
            self.cache_weight
            and self._cached_weight is not None
            and self._cached_weight.cache_key == key
        ):
            with _weight_nvtx_range("fp8_weight/fused_cache_hit"):
                return self._cached_weight
        with _weight_nvtx_range("fp8_weight/fused_cache_miss"):
            checkpoint_weights = _quantize_concatenated_block_fp8_weights_direct(
                weights
            )
            if checkpoint_weights is None:
                canonical = tuple(
                    quantize_block_fp8_weight(weight) for weight in weights
                )
                checkpoint_weights = (
                    torch.cat([item.qweight for item in canonical]),
                    torch.cat([item.scales for item in canonical]),
                )
            qweight, scales = _post_process(*checkpoint_weights)
            packed = PackedBlockFP8Weight(qweight, scales, key)
        if self.cache_weight:
            self._cached_weight = packed
        return packed

    def __call__(self, x, *weights):
        return fp8_gemm_nt(x, self.pack_weight(weights))


class DeploymentGroupedBlockFP8Adapter:
    """Versioned per-group cache for routed expert deployment weights."""

    def __init__(self, *, cache_weight: bool = False):
        self.cache_weight = cache_weight
        self._cached_weights: dict[object, PackedBlockFP8Weight] = {}

    def clear_cache(self) -> None:
        self._cached_weights.clear()

    def pack_weight(
        self,
        slot: object,
        weights: Iterable[nn.Parameter],
    ) -> PackedBlockFP8Weight:
        weights = tuple(weights)
        key = tuple(_key(weight) for weight in weights)
        cached = self._cached_weights.get(slot)
        if (
            self.cache_weight
            and cached is not None
            and cached.cache_key == key
        ):
            with _weight_nvtx_range("fp8_weight/grouped_cache_hit"):
                return cached
        with _weight_nvtx_range("fp8_weight/grouped_cache_miss"):
            packed = pack_grouped_block_fp8_weight(weights)
        if self.cache_weight:
            self._cached_weights[slot] = packed
        return packed
