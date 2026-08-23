"""vLLM grouped-DeepGEMM forward with a BF16-master training backward."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from megatron.lite.primitive.modules.experts import swiglu_with_probs
from megatron.lite.model.deepseek_v4.vllm.primitive.block_fp8 import (
    pack_grouped_block_fp8_weight,
)

_M_ALIGNMENT = 128
_EXPERTS_PER_FORWARD_GROUP = 4
_BACKWARD_CHUNK_ROWS = 1024


def _pad_expert_rows(
    value: torch.Tensor, counts: tuple[int, ...]
) -> tuple[torch.Tensor, tuple[int, ...], torch.Tensor | None]:
    padded_counts = tuple(
        ((count + _M_ALIGNMENT - 1) // _M_ALIGNMENT) * _M_ALIGNMENT if count else 0
        for count in counts
    )
    if padded_counts == counts:
        return value, padded_counts, None
    valid_ranges = []
    padded_start = 0
    for count, padded_count in zip(counts, padded_counts, strict=True):
        if count:
            valid_ranges.append(
                torch.arange(
                    padded_start,
                    padded_start + count,
                    device=value.device,
                    dtype=torch.long,
                )
            )
        padded_start += padded_count
    valid_rows = torch.cat(valid_ranges) if valid_ranges else torch.empty(0, device=value.device, dtype=torch.long)
    padded = value.new_zeros((sum(padded_counts), value.shape[1]))
    if valid_rows.numel():
        padded.index_copy_(0, valid_rows, value)
    return padded, padded_counts, valid_rows


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
        use_ue8m0=False,
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
        return hidden_states.new_empty((0, hidden_states.shape[1]))
    # Match Slime's bounded groups of four adjacent experts.
    compact_output = hidden_states.new_empty(
        (hidden_states.shape[0], w2[0].shape[0])
    )
    token_offset = 0
    for expert_start in range(0, len(counts), _EXPERTS_PER_FORWARD_GROUP):
        expert_end = min(
            expert_start + _EXPERTS_PER_FORWARD_GROUP,
            len(counts),
        )
        group_counts = counts[expert_start:expert_end]
        group_tokens = sum(group_counts)
        if group_tokens == 0:
            continue
        group_hidden = hidden_states.narrow(0, token_offset, group_tokens)
        padded, padded_counts, valid_rows = _pad_expert_rows(
            group_hidden, group_counts
        )
        m_indices = _build_m_indices(padded_counts, hidden_states.device)
        packed_input = _vllm_quantize_contiguous_input(padded)
        packed_w13 = pack_grouped_block_fp8_weight(w13[expert_start:expert_end])
        gate_up = hidden_states.new_empty((padded.shape[0], w13[0].shape[0]))
        m_grouped_fp8_gemm_nt_contiguous(
            packed_input,
            (packed_w13.qweight, packed_w13.scales),
            gate_up,
            m_indices,
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
        packed_w2 = pack_grouped_block_fp8_weight(w2[expert_start:expert_end])
        group_output = hidden_states.new_empty(
            (padded.shape[0], w2[0].shape[0])
        )
        m_grouped_fp8_gemm_nt_contiguous(
            (activated_q, activated_scale),
            (packed_w2.qweight, packed_w2.scales),
            group_output,
            m_indices,
        )
        if valid_rows is not None:
            group_output = group_output.index_select(0, valid_rows)
        compact_output.narrow(0, token_offset, group_tokens).copy_(group_output)
        token_offset += group_tokens
    if token_offset != hidden_states.shape[0]:
        raise RuntimeError(
            "grouped MoE expert counts do not cover all expert-major rows: "
            f"{token_offset} != {hidden_states.shape[0]}"
        )
    return compact_output


class VLLMGroupedMoEWithBF16Backward(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        swiglu_limit: float,
        *weights: torch.Tensor,
    ) -> torch.Tensor:
        num_experts = tokens_per_expert.numel()
        if len(weights) != 2 * num_experts:
            raise ValueError("grouped MoE weight count does not match local experts")
        counts = tuple(int(value) for value in tokens_per_expert.detach().cpu().tolist())
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
        w13 = weights[:num_experts]
        w2 = weights[num_experts:]
        grad_hidden = torch.empty_like(hidden_states) if ctx.needs_input_grad[0] else None
        grad_w13 = [torch.zeros_like(weight) if ctx.needs_input_grad[3 + i] else None for i, weight in enumerate(w13)]
        grad_w2 = [torch.zeros_like(weight) if ctx.needs_input_grad[3 + num_experts + i] else None for i, weight in enumerate(w2)]
        offset = 0
        for expert, count in enumerate(ctx.counts):
            for start in range(0, count, _BACKWARD_CHUNK_ROWS):
                end = min(start + _BACKWARD_CHUNK_ROWS, count)
                row_slice = slice(offset + start, offset + end)
                with torch.enable_grad():
                    hidden = hidden_states[row_slice].detach().requires_grad_(True)
                    fc1 = w13[expert].detach().requires_grad_(True)
                    fc2 = w2[expert].detach().requires_grad_(True)
                    gate_up = F.linear(hidden, fc1)
                    activated = swiglu_with_probs(gate_up, None, ctx.swiglu_limit)
                    recomputed = F.linear(activated, fc2)
                    grad_h, grad_fc1, grad_fc2 = torch.autograd.grad(
                        recomputed,
                        (hidden, fc1, fc2),
                        grad_output[row_slice],
                    )
                if grad_hidden is not None:
                    grad_hidden[row_slice].copy_(grad_h)
                if grad_w13[expert] is not None:
                    grad_w13[expert].add_(grad_fc1)
                if grad_w2[expert] is not None:
                    grad_w2[expert].add_(grad_fc2)
            offset += count
        return grad_hidden, None, None, *grad_w13, *grad_w2
