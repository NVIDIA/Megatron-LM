"""vLLM grouped-DeepGEMM forward with a BF16-master training backward."""

from __future__ import annotations

import weakref
from collections import OrderedDict
from dataclasses import dataclass

import torch

from megatron.lite.primitive.modules.experts import swiglu_with_probs
from megatron.lite.primitive.kernels.swiglu import swiglu_back
from megatron.lite.model.deepseek_v4.vllm.primitive.block_fp8 import (
    pack_grouped_block_fp8_weight,
)

_M_ALIGNMENT = 128
_MAX_PACK_CACHE_ENTRIES = 32
_MAX_LAYOUT_CACHE_ENTRIES = 64


def _weight_cache_key(weight: torch.Tensor) -> tuple[object, ...]:
    return (
        id(weight),
        weight._version,
        weight.device,
        weight.dtype,
        tuple(weight.shape),
    )


class _GroupedWeightPackCache:
    """Small identity/version cache that does not keep master weights alive."""

    def __init__(self) -> None:
        self._entries: OrderedDict[
            tuple[int, ...], tuple[tuple[weakref.ReferenceType[torch.Tensor], ...], object]
        ] = OrderedDict()

    def clear(self) -> None:
        self._entries.clear()

    def get(self, weights: tuple[torch.Tensor, ...]):
        identities = tuple(id(weight) for weight in weights)
        cache_key = tuple(_weight_cache_key(weight) for weight in weights)
        cached = self._entries.get(identities)
        if cached is not None:
            references, packed = cached
            if (
                all(
                    reference() is weight
                    for reference, weight in zip(
                        references, weights, strict=True
                    )
                )
                and packed.cache_key == cache_key
            ):
                self._entries.move_to_end(identities)
                return packed

        packed = pack_grouped_block_fp8_weight(weights)
        _require_power_of_two_scales("grouped weight", packed.scales)
        self._entries[identities] = (
            tuple(weakref.ref(weight) for weight in weights),
            packed,
        )
        self._entries.move_to_end(identities)
        while len(self._entries) > _MAX_PACK_CACHE_ENTRIES:
            self._entries.popitem(last=False)
        return packed


@dataclass(frozen=True)
class _GroupedForwardLayout:
    padded_counts: tuple[int, ...]
    m_indices: torch.Tensor
    valid_rows: torch.Tensor | None


_PACKED_WEIGHT_CACHE = _GroupedWeightPackCache()
_LAYOUT_CACHE: OrderedDict[
    tuple[torch.device, tuple[int, ...]], _GroupedForwardLayout
] = OrderedDict()


def _require_power_of_two_scales(name: str, scales: torch.Tensor) -> None:
    if scales.dtype != torch.float32:
        return
    bits = scales.contiguous().view(torch.int32)
    invalid = (bits & 0x807FFFFF) != 0
    torch._assert_async(
        torch.all(~invalid),
        f"{name} contains non-UE8M0 FP32 scales",
    )


def _get_forward_layout(
    counts: tuple[int, ...], device: torch.device
) -> _GroupedForwardLayout:
    key = (device, counts)
    cached = _LAYOUT_CACHE.get(key)
    if cached is not None:
        _LAYOUT_CACHE.move_to_end(key)
        return cached

    padded_counts = tuple(
        ((count + _M_ALIGNMENT - 1) // _M_ALIGNMENT) * _M_ALIGNMENT if count else 0
        for count in counts
    )
    valid_rows = None
    if padded_counts != counts:
        valid_ranges = []
        padded_start = 0
        for count, padded_count in zip(counts, padded_counts, strict=True):
            if count:
                valid_ranges.append(
                    torch.arange(
                        padded_start,
                        padded_start + count,
                        device=device,
                        dtype=torch.long,
                    )
                )
            padded_start += padded_count
        valid_rows = (
            torch.cat(valid_ranges)
            if valid_ranges
            else torch.empty(0, device=device, dtype=torch.long)
        )
    layout = _GroupedForwardLayout(
        padded_counts,
        _build_m_indices(padded_counts, device),
        valid_rows,
    )
    _LAYOUT_CACHE[key] = layout
    _LAYOUT_CACHE.move_to_end(key)
    while len(_LAYOUT_CACHE) > _MAX_LAYOUT_CACHE_ENTRIES:
        _LAYOUT_CACHE.popitem(last=False)
    return layout


def _pad_expert_rows(
    value: torch.Tensor, counts: tuple[int, ...]
) -> tuple[torch.Tensor, _GroupedForwardLayout]:
    layout = _get_forward_layout(counts, value.device)
    if layout.valid_rows is None:
        return value, layout
    padded = value.new_zeros((sum(layout.padded_counts), value.shape[1]))
    if layout.valid_rows.numel():
        padded.index_copy_(0, layout.valid_rows, value)
    return padded, layout


def _build_m_indices(counts: tuple[int, ...], device: torch.device) -> torch.Tensor:
    output = torch.empty(sum(counts), dtype=torch.int32, device=device)
    offset = 0
    for expert, count in enumerate(counts):
        if count:
            output.narrow(0, offset, count).fill_(expert)
            offset += count
    return output


def _vllm_quantize_contiguous_input(
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match the vLLM LL dispatch quant contract in contiguous layout."""
    import vllm.envs as envs
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        per_token_group_quant_fp8,
        per_token_group_quant_fp8_packed_for_deepgemm,
    )
    from vllm.utils.deep_gemm import DeepGemmQuantScaleFMT

    scale_format = DeepGemmQuantScaleFMT.from_oracle()
    if scale_format == DeepGemmQuantScaleFMT.UE8M0:
        return per_token_group_quant_fp8_packed_for_deepgemm(
            value,
            128,
            use_ue8m0=True,
        )
    if scale_format not in (
        DeepGemmQuantScaleFMT.FLOAT32,
        DeepGemmQuantScaleFMT.FLOAT32_CEIL_UE8M0,
    ):
        raise RuntimeError(
            "contiguous DS4 input requires FLOAT32 or packed UE8M0 scales, "
            f"got {scale_format}"
        )
    return per_token_group_quant_fp8(
        value,
        128,
        eps=1e-10,
        dtype=torch.float8_e4m3fn,
        column_major_scales=True,
        tma_aligned_scales=bool(envs.VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES),
        use_ue8m0=(
            scale_format == DeepGemmQuantScaleFMT.FLOAT32_CEIL_UE8M0
        ),
    )


def _vllm_silu_mul_quant(
    gate_up: torch.Tensor,
    *,
    output: torch.Tensor,
    swiglu_limit: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        fused_silu_mul_per_token_group_quant_fp8,
    )
    from vllm.utils.deep_gemm import DeepGemmQuantScaleFMT

    scale_format = DeepGemmQuantScaleFMT.from_oracle()
    # The alignment kernel is mandatory; never substitute another quantizer.
    return fused_silu_mul_per_token_group_quant_fp8(
        gate_up,
        output_q=output,
        use_ue8m0=(scale_format == DeepGemmQuantScaleFMT.UE8M0),
        round_scale=(scale_format != DeepGemmQuantScaleFMT.FLOAT32),
        clamp_limit=swiglu_limit,
        masked_m=None,
        group_size=128,
    )


def _vllm_grouped_forward(
    hidden_states: torch.Tensor,
    counts: tuple[int, ...],
    swiglu_limit: float,
    w13: tuple[torch.Tensor, ...],
    w2: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    from vllm.utils.deep_gemm import m_grouped_fp8_gemm_nt_contiguous

    if hidden_states.shape[0] == 0:
        return hidden_states.new_empty((0, w2[0].shape[0]))
    padded, layout = _pad_expert_rows(hidden_states, counts)
    packed_input = _vllm_quantize_contiguous_input(padded)
    packed_w13 = _PACKED_WEIGHT_CACHE.get(w13)
    gate_up = hidden_states.new_empty((padded.shape[0], w13[0].shape[0]))
    m_grouped_fp8_gemm_nt_contiguous(
        packed_input,
        (packed_w13.qweight, packed_w13.scales),
        gate_up,
        layout.m_indices,
    )
    activated_q = torch.empty(
        (padded.shape[0], w2[0].shape[1]),
        device=hidden_states.device,
        dtype=torch.float8_e4m3fn,
    )
    activated_q, activated_scale = _vllm_silu_mul_quant(
        gate_up,
        output=activated_q,
        swiglu_limit=swiglu_limit,
    )
    packed_w2 = _PACKED_WEIGHT_CACHE.get(w2)
    output = hidden_states.new_empty((padded.shape[0], w2[0].shape[0]))
    m_grouped_fp8_gemm_nt_contiguous(
        (activated_q, activated_scale),
        (packed_w2.qweight, packed_w2.scales),
        output,
        layout.m_indices,
    )
    if layout.valid_rows is not None:
        output = output.index_select(0, layout.valid_rows)
    return output


class _TEGroupedGemmAdapter:
    """BF16 grouped GEMMs used to replay the training VJP."""

    def __init__(self) -> None:
        try:
            from transformer_engine.pytorch.cpp_extensions import (
                general_grouped_gemm,
            )
        except (ImportError, ModuleNotFoundError, OSError) as exc:
            raise RuntimeError(
                "mLite.vLLM grouped MoE backward requires Transformer Engine "
                "general_grouped_gemm; no per-expert fallback is allowed"
            ) from exc
        self._gemm = general_grouped_gemm

    @staticmethod
    def _split(value: torch.Tensor, counts: tuple[int, ...]) -> list[torch.Tensor]:
        return list(torch.split(value.contiguous(), counts, dim=0))

    def forward(
        self,
        value: torch.Tensor,
        weights: tuple[torch.Tensor, ...] | list[torch.Tensor],
        counts: tuple[int, ...],
    ) -> torch.Tensor:
        output = value.new_empty((value.shape[0], weights[0].shape[0]))
        self._gemm(
            list(weights),
            self._split(value, counts),
            [output],
            [None] * len(weights),
            value.dtype,
            layout="TN",
            m_splits=list(counts),
            single_output=True,
        )
        return output

    def dgrad(
        self,
        grad_output: torch.Tensor,
        weights: tuple[torch.Tensor, ...] | list[torch.Tensor],
        counts: tuple[int, ...],
    ) -> torch.Tensor:
        output = grad_output.new_empty(
            (grad_output.shape[0], weights[0].shape[1])
        )
        self._gemm(
            list(weights),
            self._split(grad_output, counts),
            [output],
            [None] * len(weights),
            grad_output.dtype,
            layout="NN",
            m_splits=list(counts),
            grad=True,
            single_output=True,
        )
        return output

    def wgrad(
        self,
        value: torch.Tensor,
        grad_output: torch.Tensor,
        weights: tuple[torch.Tensor, ...] | list[torch.Tensor],
        counts: tuple[int, ...],
    ) -> tuple[torch.Tensor, ...]:
        packed_output = torch.empty(
            (len(weights), *weights[0].shape),
            device=weights[0].device,
            dtype=weights[0].dtype,
        )
        output = [packed_output[index] for index in range(len(weights))]
        self._gemm(
            self._split(value, counts),
            self._split(grad_output, counts),
            output,
            [None] * len(weights),
            value.dtype,
            layout="NT",
            m_splits=list(counts),
            grad=True,
        )
        return tuple(output)


def _get_grouped_backward_adapter():
    return _TEGroupedGemmAdapter()


def _swiglu_backward(
    gate_up: torch.Tensor, grad_output: torch.Tensor, swiglu_limit: float
) -> torch.Tensor:
    if swiglu_limit <= 0:
        return swiglu_back(grad_output, gate_up)

    gate, up = gate_up.chunk(2, dim=-1)
    gate_float = gate.float()
    up_float = up.float()
    gate_value = torch.clamp(gate_float, max=swiglu_limit)
    up_value = torch.clamp(
        up_float, min=-swiglu_limit, max=swiglu_limit
    )
    gate_mask = gate_float <= swiglu_limit
    up_mask = (up_float >= -swiglu_limit) & (up_float <= swiglu_limit)
    grad_float = grad_output.float()
    grad_gate = torch.ops.aten.silu_backward.default(
        grad_float * up_value, gate_value
    )
    grad_up = grad_float * torch.nn.functional.silu(gate_value)
    grad_gate = grad_gate * gate_mask
    grad_up = grad_up * up_mask
    return torch.cat((grad_gate, grad_up), dim=-1).to(gate_up.dtype)


class VLLMGroupedMoEWithBF16Backward(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        tokens_per_expert: list[int] | tuple[int, ...],
        swiglu_limit: float,
        *weights: torch.Tensor,
    ) -> torch.Tensor:
        counts = tuple(int(value) for value in tokens_per_expert)
        num_experts = len(counts)
        if not num_experts:
            raise ValueError("grouped MoE requires at least one local expert")
        if any(count < 0 for count in counts):
            raise ValueError("tokens_per_expert cannot contain negative counts")
        if len(weights) != 2 * num_experts:
            raise ValueError("grouped MoE weight count does not match local experts")
        if sum(counts) != hidden_states.shape[0]:
            raise ValueError("tokens_per_expert does not match expert-major rows")
        w13 = tuple(weights[:num_experts])
        w2 = tuple(weights[num_experts:])
        output = _vllm_grouped_forward(hidden_states, counts, float(swiglu_limit), w13, w2)
        ctx.counts = counts
        ctx.swiglu_limit = float(swiglu_limit)
        ctx.save_for_backward(hidden_states, *weights)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        hidden_states, *weights = ctx.saved_tensors
        num_experts = len(ctx.counts)
        w13 = tuple(weights[:num_experts])
        w2 = tuple(weights[num_experts:])
        if hidden_states.shape[0] == 0:
            grad_hidden = (
                torch.empty_like(hidden_states) if ctx.needs_input_grad[0] else None
            )
            grad_weights = tuple(
                torch.zeros_like(weight) if ctx.needs_input_grad[3 + index] else None
                for index, weight in enumerate(weights)
            )
            return grad_hidden, None, None, *grad_weights
        adapter = _get_grouped_backward_adapter()

        gate_up = adapter.forward(hidden_states, w13, ctx.counts)
        activated = swiglu_with_probs(gate_up, None, ctx.swiglu_limit)
        needs_fc1_vjp = ctx.needs_input_grad[0] or any(
            ctx.needs_input_grad[3 : 3 + num_experts]
        )
        needs_w2 = ctx.needs_input_grad[3 + num_experts :]

        grad_w2_all = (
            adapter.wgrad(activated, grad_output, w2, ctx.counts)
            if any(needs_w2)
            else (None,) * num_experts
        )
        if needs_fc1_vjp:
            grad_activated = adapter.dgrad(grad_output, w2, ctx.counts)
            grad_gate_up = _swiglu_backward(
                gate_up, grad_activated, ctx.swiglu_limit
            )
            grad_hidden = (
                adapter.dgrad(grad_gate_up, w13, ctx.counts)
                if ctx.needs_input_grad[0]
                else None
            )
            grad_w13_all = (
                adapter.wgrad(hidden_states, grad_gate_up, w13, ctx.counts)
                if any(ctx.needs_input_grad[3 : 3 + num_experts])
                else (None,) * num_experts
            )
        else:
            grad_hidden = None
            grad_w13_all = (None,) * num_experts

        grad_w13 = tuple(
            grad if ctx.needs_input_grad[3 + index] else None
            for index, grad in enumerate(grad_w13_all)
        )
        grad_w2 = tuple(
            grad if ctx.needs_input_grad[3 + num_experts + index] else None
            for index, grad in enumerate(grad_w2_all)
        )
        return grad_hidden, None, None, *grad_w13, *grad_w2
