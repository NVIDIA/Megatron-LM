"""DeepSeek-V4 BF16-master adapters over vLLM's block-FP8 deployment path."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn

from megatron.lite.model.deepseek_v4.quantization import (
    BLOCK_SHAPE,
    CanonicalBlockFP8Weight,
    requantize_block_fp8_weight,
)


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


def quantize_block_fp8_weight(weight: torch.Tensor):
    _validate_weight(weight)
    scales = getattr(weight, "_fp8_source_scales", None)
    if (
        scales is not None
        and getattr(weight, "_fp8_source_scale_version", None) == weight._version
    ):
        return requantize_block_fp8_weight(weight, scales)
    with torch.no_grad():
        from vllm.utils.deep_gemm import per_block_cast_to_fp8

        qweight, scales = per_block_cast_to_fp8(
            weight.detach(), block_size=list(BLOCK_SHAPE), use_ue8m0=False
        )
    return CanonicalBlockFP8Weight(qweight, scales)


def bind_source_scale_to_visible_weight(
    module: nn.Module, parameter_name: str, weight: torch.Tensor
):
    """Apply model-owned checkpoint scale metadata to the visible weight."""

    scales = getattr(module, "_fp8_source_scales_by_parameter", {}).get(parameter_name)
    if scales is not None:
        weight._fp8_source_scales = scales
        weight._fp8_source_scale_version = weight._version
    return weight


def _post_process(qweight, scales):
    with torch.no_grad():
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
    canonical = tuple(quantize_block_fp8_weight(weight) for weight in weights)
    qweight, scales = _post_process(
        torch.stack([item.qweight for item in canonical]),
        torch.stack([item.scales for item in canonical]),
    )
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
            return self._cached_weight
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
            return self._cached_weight
        canonical = tuple(quantize_block_fp8_weight(weight) for weight in weights)
        qweight, scales = _post_process(
            torch.cat([item.qweight for item in canonical]),
            torch.cat([item.scales for item in canonical]),
        )
        packed = PackedBlockFP8Weight(qweight, scales, key)
        if self.cache_weight:
            self._cached_weight = packed
        return packed

    def __call__(self, x, *weights):
        return fp8_gemm_nt(x, self.pack_weight(weights))
