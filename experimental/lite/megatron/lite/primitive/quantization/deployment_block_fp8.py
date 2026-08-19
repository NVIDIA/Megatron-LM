# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Inference-only block-FP8 adapter backed by vLLM's deployment kernels.

This module deliberately keeps the BF16 ``Parameter`` as the only model state.
FP8 weights and scales are ephemeral deployment artifacts (or an explicit
non-state cache) and are never registered as parameters or buffers.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Callable, Iterable, Literal

import torch
from torch import nn

from megatron.lite.primitive.autograd import inference_only

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

BLOCK_SHAPE = (128, 128)

WeightScaleFormat = Literal["ue8m0"]
WeightScaleLayout = Literal["deepgemm_block_tma"]
ActivationScaleFormat = Literal["ue8m0_packed_int32", "float32_ceil_ue8m0"]
ActivationScaleLayout = Literal[
    "deepgemm_packed_k_tma", "deepgemm_column_major"
]


@dataclass(frozen=True)
class WeightCacheKey:
    """Identity and mutation state of a BF16 master parameter."""

    parameter_id: int
    version: int
    device: torch.device
    dtype: torch.dtype
    shape: tuple[int, int]


FusedWeightCacheKey = tuple[WeightCacheKey, ...]


@dataclass(frozen=True)
class PackedBlockFP8Weight:
    """vLLM/DeepGEMM-ready weight and scale tensors."""

    qweight: torch.Tensor
    scales: torch.Tensor
    block_shape: tuple[int, int]
    scale_format: WeightScaleFormat
    scale_layout: WeightScaleLayout
    cache_key: WeightCacheKey | FusedWeightCacheKey


@dataclass(frozen=True)
class CanonicalBlockFP8Weight:
    """Checkpoint/export block-FP8 tensors before runtime layout transforms."""

    qweight: torch.Tensor
    scales: torch.Tensor
    block_shape: tuple[int, int]


@dataclass(frozen=True)
class PackedBlockFP8Activation:
    """vLLM/DeepGEMM-ready per-token-group activation tensors."""

    qactivation: torch.Tensor
    scales: torch.Tensor
    group_size: int
    scale_format: ActivationScaleFormat
    scale_layout: ActivationScaleLayout


@dataclass(frozen=True)
class PackedGroupedBlockFP8Weight:
    """Grouped-expert weight with one jointly transformed scale layout."""

    qweight: torch.Tensor
    scales: torch.Tensor
    block_shape: tuple[int, int]
    scale_format: WeightScaleFormat
    scale_layout: WeightScaleLayout
    cache_keys: tuple[WeightCacheKey, ...]


if triton is not None:

    @triton.jit
    def _requantize_with_scale_kernel(
        master,
        scales,
        qweight,
        columns,
        scale_columns,
        elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < elements
        rows = offsets // columns
        columns_in_row = offsets - rows * columns
        scale_offsets = (rows // 128) * scale_columns + columns_in_row // 128
        values = tl.load(master + offsets, mask=mask).to(tl.float32)
        block_scales = tl.load(scales + scale_offsets, mask=mask)
        tl.store(qweight + offsets, values / block_scales, mask=mask)


def _import_attr(module_name: str, attr_name: str) -> Callable:
    """Load a vLLM entry point lazily and fail closed."""

    try:
        module = importlib.import_module(module_name)
        value = getattr(module, attr_name)
    except (ImportError, AttributeError) as error:
        raise RuntimeError(
            f"vLLM deployment entry point {module_name}.{attr_name} is unavailable"
        ) from error
    if not callable(value):
        raise RuntimeError(
            f"vLLM deployment entry point {module_name}.{attr_name} is not callable"
        )
    return value


def _weight_cache_key(master_weight: nn.Parameter) -> WeightCacheKey:
    return WeightCacheKey(
        parameter_id=id(master_weight),
        version=master_weight._version,
        device=master_weight.device,
        dtype=master_weight.dtype,
        shape=(master_weight.shape[0], master_weight.shape[1]),
    )


def _validate_bf16_weight(weight: torch.Tensor) -> None:
    if not isinstance(weight, torch.Tensor):
        raise TypeError("block-FP8 weight source must be a torch.Tensor")
    if weight.dtype != torch.bfloat16:
        raise TypeError(f"block-FP8 master weight must be BF16, got {weight.dtype}")
    if weight.ndim != 2:
        raise ValueError(
            f"block-FP8 master weight must be 2-D [out, in], got {weight.shape}"
        )
    if any(size % block for size, block in zip(weight.shape, BLOCK_SHAPE)):
        raise ValueError(
            f"weight shape {tuple(weight.shape)} must be divisible by "
            f"block shape {BLOCK_SHAPE}"
        )


def _validate_master_weight(master_weight: nn.Parameter) -> None:
    if not isinstance(master_weight, nn.Parameter):
        raise TypeError(
            "block-FP8 deployment weight must be a BF16 torch.nn.Parameter"
        )
    _validate_bf16_weight(master_weight)


def requantize_block_fp8_weight(
    bf16_weight: torch.Tensor,
    scales: torch.Tensor,
) -> CanonicalBlockFP8Weight:
    """Reconstruct FP8 values with caller-owned scales, without rescaling."""

    _validate_bf16_weight(bf16_weight)
    expected_scale_shape = tuple(
        size // block
        for size, block in zip(bf16_weight.shape, BLOCK_SHAPE, strict=True)
    )
    if scales.dtype != torch.float32 or tuple(scales.shape) != expected_scale_shape:
        raise ValueError(
            f"fixed scales must be float32 with shape {expected_scale_shape}, "
            f"got dtype={scales.dtype} shape={tuple(scales.shape)}"
        )
    if scales.device != bf16_weight.device:
        raise ValueError("fixed scales and BF16 master must share a device")
    if not bool(torch.all(torch.isfinite(scales) & (scales > 0))):
        raise ValueError("fixed scales must be finite and positive")

    with torch.inference_mode():
        if bf16_weight.is_cuda and triton is not None:
            qweight = torch.empty_like(bf16_weight, dtype=torch.float8_e4m3fn)
            block_size = 256
            _requantize_with_scale_kernel[
                (triton.cdiv(bf16_weight.numel(), block_size),)
            ](
                bf16_weight,
                scales.contiguous(),
                qweight,
                bf16_weight.shape[1],
                scales.shape[1],
                bf16_weight.numel(),
                BLOCK_SIZE=block_size,
            )
        else:
            expanded = scales.repeat_interleave(BLOCK_SHAPE[0], dim=0)
            expanded = expanded.repeat_interleave(BLOCK_SHAPE[1], dim=1)
            qweight = (bf16_weight.float() / expanded).to(torch.float8_e4m3fn)
    return CanonicalBlockFP8Weight(qweight, scales, BLOCK_SHAPE)


def quantize_block_fp8_weight(
    bf16_weight: torch.Tensor,
) -> CanonicalBlockFP8Weight:
    """Shared BF16→canonical FP8 path for forward packing and export."""

    _validate_bf16_weight(bf16_weight)
    source_scales = getattr(bf16_weight, "_fp8_source_scales", None)
    source_version = getattr(bf16_weight, "_fp8_source_scale_version", None)
    if source_scales is not None and source_version == bf16_weight._version:
        return requantize_block_fp8_weight(bf16_weight, source_scales)

    per_block_cast_to_fp8 = _import_attr(
        "vllm.utils.deep_gemm", "per_block_cast_to_fp8"
    )
    with torch.inference_mode():
        qweight, scales = per_block_cast_to_fp8(
            bf16_weight.detach(),
            block_size=list(BLOCK_SHAPE),
            use_ue8m0=False,
        )
    expected_scale_shape = tuple(
        size // block
        for size, block in zip(bf16_weight.shape, BLOCK_SHAPE, strict=True)
    )
    if qweight.dtype != torch.float8_e4m3fn:
        raise RuntimeError(
            f"vLLM returned weight dtype {qweight.dtype}, expected float8_e4m3fn"
        )
    if tuple(qweight.shape) != tuple(bf16_weight.shape) or not qweight.is_contiguous():
        raise RuntimeError("vLLM returned an invalid canonical block-FP8 weight")
    if scales.dtype != torch.float32 or tuple(scales.shape) != expected_scale_shape:
        raise RuntimeError(
            "vLLM returned invalid canonical block-FP8 weight scales"
        )
    if qweight.device != bf16_weight.device or scales.device != bf16_weight.device:
        raise RuntimeError("vLLM returned block-FP8 artifacts on the wrong device")
    return CanonicalBlockFP8Weight(qweight, scales, BLOCK_SHAPE)


def pack_block_fp8_weight(master_weight: nn.Parameter) -> PackedBlockFP8Weight:
    """Quantize a BF16 master through vLLM's actual weight deployment path."""

    _validate_master_weight(master_weight)
    canonical = quantize_block_fp8_weight(master_weight)
    post_process = _import_attr(
        "vllm.model_executor.layers.quantization.utils.fp8_utils",
        "deepgemm_post_process_fp8_weight_block",
    )

    with torch.inference_mode():
        qweight, scales = canonical.qweight, canonical.scales
        # This mirrors vLLM's DeepGEMM kernel preparation. With use_e8m0=True,
        # float32 scales are requantized through requant_weight_ue8m0_inplace
        # before vLLM transforms them into the architecture-required layout.
        qweight, scales = post_process(
            wq=qweight,
            ws=scales,
            quant_block_shape=BLOCK_SHAPE,
            use_e8m0=True,
        )

    if qweight.dtype != torch.float8_e4m3fn:
        raise RuntimeError(
            f"vLLM returned weight dtype {qweight.dtype}, expected float8_e4m3fn"
        )
    if tuple(qweight.shape) != tuple(master_weight.shape):
        raise RuntimeError(
            f"vLLM returned weight shape {tuple(qweight.shape)}, expected "
            f"{tuple(master_weight.shape)}"
        )
    if not qweight.is_contiguous():
        raise RuntimeError("vLLM returned a non-contiguous block-FP8 weight")
    if scales.ndim != 2 or scales.numel() == 0:
        raise RuntimeError(
            "vLLM returned invalid DeepGEMM weight-scale dimensions"
        )
    if qweight.device != master_weight.device or scales.device != master_weight.device:
        raise RuntimeError("vLLM returned block-FP8 artifacts on the wrong device")

    return PackedBlockFP8Weight(
        qweight=qweight,
        scales=scales,
        block_shape=BLOCK_SHAPE,
        scale_format="ue8m0",
        scale_layout="deepgemm_block_tma",
        cache_key=_weight_cache_key(master_weight),
    )


def pack_grouped_block_fp8_weight(
    master_weights: Iterable[nn.Parameter],
) -> PackedGroupedBlockFP8Weight:
    """Quantize experts independently, then transform scales as one E×N×K group."""

    master_weights = tuple(master_weights)
    if not master_weights:
        raise ValueError("grouped block-FP8 packing requires at least one expert")
    for weight in master_weights:
        _validate_master_weight(weight)
    first = master_weights[0]
    if any(weight.shape != first.shape for weight in master_weights):
        raise ValueError("all grouped block-FP8 master weights must share one shape")
    if any(weight.device != first.device for weight in master_weights):
        raise ValueError("all grouped block-FP8 master weights must share one device")

    post_process = _import_attr(
        "vllm.model_executor.layers.quantization.utils.fp8_utils",
        "deepgemm_post_process_fp8_weight_block",
    )
    with torch.inference_mode():
        quantized = [
            quantize_block_fp8_weight(weight) for weight in master_weights
        ]
        qweight = torch.stack([item.qweight for item in quantized])
        scales = torch.stack([item.scales for item in quantized])
        qweight, scales = post_process(
            wq=qweight,
            ws=scales,
            quant_block_shape=BLOCK_SHAPE,
            use_e8m0=True,
        )
    if qweight.ndim != 3 or qweight.shape[0] != len(master_weights):
        raise RuntimeError("vLLM returned an invalid grouped FP8 weight shape")
    if qweight.dtype != torch.float8_e4m3fn or not qweight.is_contiguous():
        raise RuntimeError("vLLM returned an invalid grouped FP8 weight layout")
    if scales.ndim < 2 or scales.device != first.device:
        raise RuntimeError("vLLM returned invalid grouped DeepGEMM scales")
    return PackedGroupedBlockFP8Weight(
        qweight=qweight,
        scales=scales,
        block_shape=BLOCK_SHAPE,
        scale_format="ue8m0",
        scale_layout="deepgemm_block_tma",
        cache_keys=tuple(_weight_cache_key(weight) for weight in master_weights),
    )


def pack_block_fp8_activation(x: torch.Tensor) -> PackedBlockFP8Activation:
    """Quantize BF16 activations with vLLM's packed DeepGEMM entry point."""

    if x.dtype != torch.bfloat16:
        raise TypeError(f"block-FP8 activation must be BF16, got {x.dtype}")
    if x.ndim != 2:
        raise ValueError(
            f"block-FP8 activation must be 2-D [tokens, hidden], got {x.shape}"
        )
    if x.shape[1] % BLOCK_SHAPE[1]:
        raise ValueError(
            f"activation hidden size {x.shape[1]} must be divisible by "
            f"group size {BLOCK_SHAPE[1]}"
        )
    if x.stride(-1) != 1:
        raise ValueError("block-FP8 activation groups must be contiguous")

    scale_oracle = _import_attr(
        "vllm.utils.deep_gemm", "DeepGemmQuantScaleFMT"
    ).from_oracle()
    packed_ue8m0 = getattr(scale_oracle, "name", "") == "UE8M0"
    quantize = _import_attr(
        "vllm.model_executor.layers.quantization.utils.fp8_utils",
        (
            "per_token_group_quant_fp8_packed_for_deepgemm"
            if packed_ue8m0
            else "per_token_group_quant_fp8"
        ),
    )
    with torch.inference_mode():
        if packed_ue8m0:
            qactivation, scales = quantize(
                x,
                BLOCK_SHAPE[1],
                use_ue8m0=True,
            )
        else:
            try:
                tma_aligned = bool(
                    getattr(
                        importlib.import_module("vllm.envs"),
                        "VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES",
                    )
                )
            except (ImportError, AttributeError) as error:
                raise RuntimeError(
                    "vLLM DeepGEMM TMA scale-layout configuration is unavailable"
                ) from error
            qactivation, scales = quantize(
                x,
                BLOCK_SHAPE[1],
                use_ue8m0=True,
                column_major_scales=True,
                tma_aligned_scales=tma_aligned,
            )

    if tuple(qactivation.shape) != tuple(x.shape):
        raise RuntimeError(
            f"vLLM returned activation shape {tuple(qactivation.shape)}, "
            f"expected {tuple(x.shape)}"
        )
    if qactivation.dtype != torch.float8_e4m3fn:
        raise RuntimeError(
            f"vLLM returned activation dtype {qactivation.dtype}, "
            "expected float8_e4m3fn"
        )
    if not qactivation.is_contiguous():
        raise RuntimeError("vLLM returned a non-contiguous block-FP8 activation")
    expected_scale_dtype = torch.int32 if packed_ue8m0 else torch.float32
    if scales.dtype != expected_scale_dtype or scales.ndim != 2:
        raise RuntimeError(
            "vLLM returned invalid packed UE8M0 activation scales"
        )
    if qactivation.device != x.device or scales.device != x.device:
        raise RuntimeError("vLLM returned block-FP8 activation artifacts on wrong device")

    return PackedBlockFP8Activation(
        qactivation=qactivation,
        scales=scales,
        group_size=BLOCK_SHAPE[1],
        scale_format=(
            "ue8m0_packed_int32"
            if packed_ue8m0
            else "float32_ceil_ue8m0"
        ),
        scale_layout=(
            "deepgemm_packed_k_tma"
            if packed_ue8m0
            else "deepgemm_column_major"
        ),
    )


def fp8_gemm_nt(
    x: torch.Tensor,
    packed_weight: PackedBlockFP8Weight,
) -> torch.Tensor:
    """Run vLLM's DeepGEMM NT op and return BF16 output."""

    if x.dtype != torch.bfloat16:
        raise TypeError(f"block-FP8 GEMM input must be BF16, got {x.dtype}")
    if x.ndim != 2:
        raise ValueError(f"block-FP8 GEMM input must be 2-D, got {x.shape}")
    if x.shape[1] != packed_weight.qweight.shape[1]:
        raise ValueError(
            f"input K={x.shape[1]} does not match weight K="
            f"{packed_weight.qweight.shape[1]}"
        )
    if x.device != packed_weight.qweight.device:
        raise ValueError("input and packed weight must be on the same device")
    if packed_weight.block_shape != BLOCK_SHAPE:
        raise ValueError(
            f"packed weight block shape must be {BLOCK_SHAPE}, got "
            f"{packed_weight.block_shape}"
        )
    if packed_weight.scale_format != "ue8m0":
        raise ValueError("fp8_gemm_nt requires UE8M0 weight scales")
    if packed_weight.scale_layout != "deepgemm_block_tma":
        raise ValueError("fp8_gemm_nt requires DeepGEMM block-TMA weight scales")
    if packed_weight.qweight.dtype != torch.float8_e4m3fn:
        raise TypeError("fp8_gemm_nt requires an E4M3 quantized weight")
    if packed_weight.qweight.ndim != 2 or not packed_weight.qweight.is_contiguous():
        raise ValueError("fp8_gemm_nt requires a contiguous 2-D quantized weight")
    if packed_weight.scales.ndim != 2:
        raise ValueError("fp8_gemm_nt requires 2-D DeepGEMM weight scales")
    if packed_weight.scales.device != x.device:
        raise ValueError("input and packed weight scales must be on the same device")

    with torch.inference_mode():
        packed_activation = pack_block_fp8_activation(x)
        output = torch.empty(
            (x.shape[0], packed_weight.qweight.shape[0]),
            dtype=torch.bfloat16,
            device=x.device,
        )
        op = _import_attr("vllm.utils.deep_gemm", "fp8_gemm_nt")
        op(
            (packed_activation.qactivation, packed_activation.scales),
            (packed_weight.qweight, packed_weight.scales),
            output,
            is_deep_gemm_e8m0_used=True,
        )
    if output.dtype != torch.bfloat16:
        raise RuntimeError(f"vLLM FP8 GEMM returned non-BF16 output {output.dtype}")
    return output


class DeploymentBlockFP8Adapter:
    """Stateless-by-default adapter from BF16 master weights to vLLM FP8."""

    def __init__(self, *, cache_weight: bool = False) -> None:
        self.cache_weight = cache_weight
        self._cached_weight: PackedBlockFP8Weight | None = None

    def clear_cache(self) -> None:
        self._cached_weight = None

    def pack_weight(self, master_weight: nn.Parameter) -> PackedBlockFP8Weight:
        _validate_master_weight(master_weight)
        key = _weight_cache_key(master_weight)
        if (
            self.cache_weight
            and self._cached_weight is not None
            and self._cached_weight.cache_key == key
        ):
            return self._cached_weight

        packed = pack_block_fp8_weight(master_weight)
        if self.cache_weight:
            self._cached_weight = packed
        return packed

    def __call__(
        self,
        x: torch.Tensor,
        master_weight: nn.Parameter,
    ) -> torch.Tensor:
        """Visible forward is always deployment FP8 and never an STE substitute."""

        packed = self.pack_weight(master_weight)
        output = fp8_gemm_nt(x, packed)
        return inference_only(output, x, master_weight)


class DeploymentFusedBlockFP8Adapter:
    """Pack several release parameters into one vLLM-visible FP8 GEMM."""

    def __init__(self, *, cache_weight: bool = False) -> None:
        self.cache_weight = cache_weight
        self._cached_weight: PackedBlockFP8Weight | None = None

    def clear_cache(self) -> None:
        self._cached_weight = None

    def pack_weight(
        self, master_weights: Iterable[nn.Parameter]
    ) -> PackedBlockFP8Weight:
        weights = tuple(master_weights)
        if not weights:
            raise ValueError("fused block-FP8 packing requires at least one weight")
        for weight in weights:
            _validate_master_weight(weight)
        first = weights[0]
        if any(weight.shape[1] != first.shape[1] for weight in weights):
            raise ValueError("fused block-FP8 weights must share their input width")
        if any(weight.device != first.device for weight in weights):
            raise ValueError("fused block-FP8 weights must share one device")
        key = tuple(_weight_cache_key(weight) for weight in weights)
        if (
            self.cache_weight
            and self._cached_weight is not None
            and self._cached_weight.cache_key == key
        ):
            return self._cached_weight

        canonical = tuple(quantize_block_fp8_weight(weight) for weight in weights)
        qweight = torch.cat([item.qweight for item in canonical], dim=0)
        scales = torch.cat([item.scales for item in canonical], dim=0)
        post_process = _import_attr(
            "vllm.model_executor.layers.quantization.utils.fp8_utils",
            "deepgemm_post_process_fp8_weight_block",
        )
        with torch.inference_mode():
            qweight, scales = post_process(
                wq=qweight,
                ws=scales,
                quant_block_shape=BLOCK_SHAPE,
                use_e8m0=True,
            )
        packed = PackedBlockFP8Weight(
            qweight=qweight,
            scales=scales,
            block_shape=BLOCK_SHAPE,
            scale_format="ue8m0",
            scale_layout="deepgemm_block_tma",
            cache_key=key,
        )
        if self.cache_weight:
            self._cached_weight = packed
        return packed

    def __call__(
        self, x: torch.Tensor, *master_weights: nn.Parameter
    ) -> torch.Tensor:
        packed = self.pack_weight(master_weights)
        return inference_only(fp8_gemm_nt(x, packed), x, *master_weights)


__all__ = [
    "BLOCK_SHAPE",
    "CanonicalBlockFP8Weight",
    "DeploymentBlockFP8Adapter",
    "DeploymentFusedBlockFP8Adapter",
    "PackedBlockFP8Activation",
    "PackedBlockFP8Weight",
    "PackedGroupedBlockFP8Weight",
    "WeightCacheKey",
    "fp8_gemm_nt",
    "pack_block_fp8_activation",
    "pack_block_fp8_weight",
    "pack_grouped_block_fp8_weight",
    "quantize_block_fp8_weight",
    "requantize_block_fp8_weight",
]
