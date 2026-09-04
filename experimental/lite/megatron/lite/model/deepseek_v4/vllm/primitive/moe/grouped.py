"""vLLM grouped-DeepGEMM forward with a BF16-master training backward."""

from __future__ import annotations

from contextlib import contextmanager
import os

import torch

from megatron.lite.primitive.modules.experts import swiglu_with_probs
from megatron.lite.model.deepseek_v4.vllm.primitive.block_fp8 import (
    DeploymentGroupedBlockFP8Adapter,
)
from megatron.lite.model.deepseek_v4.vllm.primitive.dense import (
    dynamic_clamped_swiglu_vjp,
)

_M_ALIGNMENT = 128
_PACK_DEBUG_SYNC = os.getenv("MLITE_VLLM_PACK_DEBUG_SYNC") == "1"


def _direct_ceil_ue8m0_activation_packing(value: torch.Tensor) -> bool:
    """Use v9's ceil-UE8M0 activation packing only on Blackwell."""
    if not value.is_cuda:
        return False
    major, _minor = torch.cuda.get_device_capability(value.device)
    return major >= 10


@contextmanager
def _nvtx_range(name: str):
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


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
        use_ue8m0=(
            scale_format == DeepGemmQuantScaleFMT.FLOAT32_CEIL_UE8M0
            and _direct_ceil_ue8m0_activation_packing(value)
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
    weight_cache: DeploymentGroupedBlockFP8Adapter,
    w13: tuple[torch.Tensor, ...],
    w2: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    from vllm.utils.deep_gemm import m_grouped_fp8_gemm_nt_contiguous

    if hidden_states.shape[0] == 0:
        return (
            hidden_states.new_empty((0, hidden_states.shape[1])),
            hidden_states.new_empty((0, w13[0].shape[0])),
        )
    compact_output = hidden_states.new_empty(
        (hidden_states.shape[0], w2[0].shape[0])
    )
    compact_gate_up = hidden_states.new_empty(
        (hidden_states.shape[0], w13[0].shape[0])
    )
    token_offset = 0
    experts_per_group = len(counts)
    for expert_start in range(0, len(counts), experts_per_group):
        expert_end = min(
            expert_start + experts_per_group,
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
        packed_w13 = weight_cache.pack_weight(
            ("w13", expert_start),
            w13[expert_start:expert_end],
        )
        gate_up = hidden_states.new_empty((padded.shape[0], w13[0].shape[0]))
        m_grouped_fp8_gemm_nt_contiguous(
            packed_input,
            (packed_w13.qweight, packed_w13.scales),
            gate_up,
            m_indices,
        )
        if _PACK_DEBUG_SYNC:
            try:
                torch.cuda.synchronize(hidden_states.device)
            except torch.AcceleratorError as error:
                raise RuntimeError(
                    "grouped FC1 failed before FC2 packing: "
                    f"counts={group_counts}, padded_counts={padded_counts}"
                ) from error
        visible_gate_up = (
            gate_up
            if valid_rows is None
            else gate_up.index_select(0, valid_rows)
        )
        compact_gate_up.narrow(0, token_offset, group_tokens).copy_(
            visible_gate_up
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
        if _PACK_DEBUG_SYNC:
            try:
                torch.cuda.synchronize(hidden_states.device)
            except torch.AcceleratorError as error:
                raise RuntimeError(
                    "grouped SwiGLU quant failed before FC2 packing: "
                    f"counts={group_counts}, padded_counts={padded_counts}"
                ) from error
        packed_w2 = weight_cache.pack_weight(
            ("w2", expert_start),
            w2[expert_start:expert_end],
        )
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
    return compact_output, compact_gate_up


def _te_grouped_gemm(
    lhs: tuple[torch.Tensor, ...] | list[torch.Tensor],
    rhs: tuple[torch.Tensor, ...] | list[torch.Tensor],
    output: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor],
    *,
    activation_dtype: torch.dtype,
    layout: str = "TN",
    m_splits: tuple[int, ...] | None = None,
    single_output: bool = False,
    grad: bool = False,
    use_split_accumulator: bool = False,
) -> None:
    """Call TE's grouped GEMM primitive without constructing GroupedLinear."""
    from transformer_engine.pytorch.cpp_extensions import general_grouped_gemm

    outputs = [output] if isinstance(output, torch.Tensor) else list(output)
    general_grouped_gemm(
        list(lhs),
        list(rhs),
        outputs,
        [None] * len(lhs),
        activation_dtype,
        single_output=single_output,
        layout=layout,
        m_splits=list(m_splits) if m_splits is not None else None,
        grad=grad,
        use_split_accumulator=use_split_accumulator,
    )


def _te_grouped_bf16_backward(
    hidden_states: torch.Tensor,
    gate_up: torch.Tensor,
    grad_output: torch.Tensor,
    counts: tuple[int, ...],
    swiglu_limit: float,
    w13: tuple[torch.Tensor, ...],
    w2: tuple[torch.Tensor, ...],
    *,
    needs_hidden: bool,
    needs_w13: tuple[bool, ...],
    needs_w2: tuple[bool, ...],
) -> tuple[
    torch.Tensor | None,
    tuple[torch.Tensor | None, ...],
    tuple[torch.Tensor | None, ...],
]:
    """Run the common BF16 FC dgrad/wgrad path from saved visible intermediates."""
    activation_dtype = hidden_states.dtype
    hidden_mats = tuple(torch.split(hidden_states, counts))
    grad_output_mats = tuple(torch.split(grad_output.contiguous(), counts))

    activated = swiglu_with_probs(
        gate_up,
        None,
        swiglu_limit,
    )

    grad_activated = torch.empty_like(activated)
    with _nvtx_range("moe_bwd/fc2_dgrad"):
        _te_grouped_gemm(
            w2,
            grad_output_mats,
            grad_activated,
            activation_dtype=activation_dtype,
            layout="NN",
            m_splits=counts,
            single_output=True,
            grad=True,
            use_split_accumulator=True,
        )

    grad_w2: list[torch.Tensor | None] = [None] * len(w2)
    if any(needs_w2):
        computed_w2_packed = torch.empty(
            (len(w2), *w2[0].shape),
            dtype=activation_dtype,
            device=hidden_states.device,
        )
        computed_w2 = tuple(computed_w2_packed.unbind(0))
        activated_mats = tuple(torch.split(activated.detach(), counts))
        with _nvtx_range("moe_bwd/fc2_wgrad"):
            _te_grouped_gemm(
                activated_mats,
                grad_output_mats,
                computed_w2,
                activation_dtype=activation_dtype,
                layout="NT",
                m_splits=counts,
                grad=True,
                use_split_accumulator=True,
            )
        grad_w2 = [
            gradient if needed else None
            for gradient, needed in zip(computed_w2, needs_w2, strict=True)
        ]

    with _nvtx_range("moe_bwd/swiglu"):
        grad_gate_up = dynamic_clamped_swiglu_vjp(
            grad_activated,
            gate_up,
            swiglu_limit,
        )
    grad_gate_up_mats = tuple(torch.split(grad_gate_up.contiguous(), counts))

    grad_hidden = None
    if needs_hidden:
        grad_hidden = torch.empty_like(hidden_states)
        with _nvtx_range("moe_bwd/fc1_dgrad"):
            _te_grouped_gemm(
                w13,
                grad_gate_up_mats,
                grad_hidden,
                activation_dtype=activation_dtype,
                layout="NN",
                m_splits=counts,
                single_output=True,
                grad=True,
                use_split_accumulator=True,
            )

    grad_w13: list[torch.Tensor | None] = [None] * len(w13)
    if any(needs_w13):
        computed_w13_packed = torch.empty(
            (len(w13), *w13[0].shape),
            dtype=activation_dtype,
            device=hidden_states.device,
        )
        computed_w13 = tuple(computed_w13_packed.unbind(0))
        with _nvtx_range("moe_bwd/fc1_wgrad"):
            _te_grouped_gemm(
                hidden_mats,
                grad_gate_up_mats,
                computed_w13,
                activation_dtype=activation_dtype,
                layout="NT",
                m_splits=counts,
                grad=True,
                use_split_accumulator=True,
            )
        grad_w13 = [
            gradient if needed else None
            for gradient, needed in zip(computed_w13, needs_w13, strict=True)
        ]

    return grad_hidden, tuple(grad_w13), tuple(grad_w2)


class VLLMGroupedMoEWithBF16Backward(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        tokens_per_expert: list[int] | tuple[int, ...],
        swiglu_limit: float,
        weight_cache: DeploymentGroupedBlockFP8Adapter,
        *weights: torch.Tensor,
    ) -> torch.Tensor:
        counts = tuple(int(value) for value in tokens_per_expert)
        num_experts = len(counts)
        if len(weights) != 2 * num_experts:
            raise ValueError("grouped MoE weight count does not match local experts")
        if sum(counts) != hidden_states.shape[0]:
            raise ValueError("tokens_per_expert does not match expert-major rows")
        w13 = tuple(weights[:num_experts])
        w2 = tuple(weights[num_experts:])
        output, gate_up = _vllm_grouped_forward(
            hidden_states,
            counts,
            float(swiglu_limit),
            weight_cache,
            w13,
            w2,
        )
        ctx.counts = counts
        ctx.swiglu_limit = float(swiglu_limit)
        ctx.save_for_backward(hidden_states, gate_up, *weights)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        hidden_states, gate_up, *weights = ctx.saved_tensors
        num_experts = len(ctx.counts)
        w13 = tuple(weights[:num_experts])
        w2 = tuple(weights[num_experts:])
        grad_hidden, grad_w13, grad_w2 = _te_grouped_bf16_backward(
            hidden_states,
            gate_up,
            grad_output,
            ctx.counts,
            ctx.swiglu_limit,
            w13,
            w2,
            needs_hidden=ctx.needs_input_grad[0],
            needs_w13=tuple(
                ctx.needs_input_grad[4 + expert]
                for expert in range(num_experts)
            ),
            needs_w2=tuple(
                ctx.needs_input_grad[4 + num_experts + expert]
                for expert in range(num_experts)
            ),
        )
        return grad_hidden, None, None, None, *grad_w13, *grad_w2
