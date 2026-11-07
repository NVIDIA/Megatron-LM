# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""FlashInfer routed-MoE support for per-token NVFP4 x NVFP4 squared-ReLU and SwiGLU experts.

Expert weights are quantized from MCore's BF16 expert stacks with one FP32
global scale per expert. Activations are quantized online with one FP32 scale
per token, so no calibration or static input scale is required. The experts
run on FlashInfer's CuTe DSL block-scaled grouped GEMMs.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from megatron.core.inference.moe.flashinfer_mxfp8 import select_routed_mxfp8_active_rows
from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
    is_batch_invariant_mode_enabled,
)

try:
    import flashinfer
    from flashinfer import ActivationType, SfLayout, nvfp4_quantize
    from packaging.version import Version

    # Per-token NVFP4 Relu2 MoE landed in FlashInfer 0.7.0 (flashinfer-ai/flashinfer#3645).
    if Version(flashinfer.__version__).release < (0, 7, 0):
        raise ImportError(
            f"FlashInfer {flashinfer.__version__} lacks the per-token NVFP4 Relu2 MoE kernels"
        )
    from flashinfer import CuteDslMoEWrapper
    from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout

    _FLASHINFER_NVFP4_IMPORT_ERROR: ImportError | None = None
except ImportError as exc:
    _FLASHINFER_NVFP4_IMPORT_ERROR = exc

HAVE_FLASHINFER_ROUTED_NVFP4 = _FLASHINFER_NVFP4_IMPORT_ERROR is None

NVFP4_BLOCK_SIZE = 16
FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = 448.0
# Per-token quantization folds the row amax into the per-token scale, so the base
# global scale is the recipe constant 1 / (E4M3_MAX * E2M1_MAX). FlashInfer rejects
# other values. It must be a Python float: a tensor triggers a host sync.
PER_TOKEN_ACTIVATION_GLOBAL_SCALE = 1.0 / (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX)
# GEMM K/N padding: the CuTe DSL grouped GEMMs need N % 128 and full 256-wide FP4 K tiles.
_ALIGNMENT = 256
# Gated FC1 rows are [up, gate] blocks of this many rows each, alternating.
_GLU_INTERLEAVE_ROWS = 64

_CUTE_DSL_WRAPPERS: dict = {}


def require_flashinfer_routed_nvfp4() -> None:
    """Raise an error when the per-token NVFP4 routed-MoE APIs are unavailable."""
    if not HAVE_FLASHINFER_ROUTED_NVFP4:
        raise RuntimeError(
            "FlashInfer per-token NVFP4 MoE requires FlashInfer >= 0.7.0 with "
            "nvidia-cutlass-dsl. Upgrade flashinfer-python or disable "
            "inference_moe_nvfp4_recipe."
        ) from _FLASHINFER_NVFP4_IMPORT_ERROR


def _round_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


@dataclass(frozen=True)
class FlashInferRoutedNVFP4Weights:
    """FC1/FC2 expert stacks in the CuTe DSL block-scaled grouped-GEMM layout.

    Data is packed E2M1 ``[experts, N, K // 2]``. ``*_scale`` is the 6D MMA view
    of the 128x4-swizzled E4M3 block scales held in ``*_scale_swizzled``.
    ``*_alpha`` are the per-expert FP32 dequantization scales and
    ``fc2_input_scale`` is the per-token recipe constant for the FC2 input.
    Gated (SwiGLU) FC1 has ``2 * padded_intermediate_size`` rows in FlashInfer's
    interleaved up/gate order.
    """

    hidden_size: int
    intermediate_size: int
    padded_hidden_size: int
    padded_intermediate_size: int
    gated: bool
    fc1_data: torch.Tensor
    fc1_scale_swizzled: torch.Tensor
    fc1_scale: torch.Tensor
    fc1_alpha: torch.Tensor
    fc2_data: torch.Tensor
    fc2_scale_swizzled: torch.Tensor
    fc2_scale: torch.Tensor
    fc2_alpha: torch.Tensor
    fc2_input_scale: torch.Tensor

    @property
    def num_local_experts(self) -> int:
        """Number of experts in each stack."""
        return self.fc1_data.shape[0]


def _global_encode_scale(weight: torch.Tensor) -> torch.Tensor:
    """Return the one-element NVFP4 global encode scale for a weight matrix."""
    amax = weight.abs().amax().float().reshape(1)
    encode = (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / amax
    return torch.where(amax > 0, encode, torch.ones_like(encode))


def _zero_pad(weight: torch.Tensor, padded_rows: int, padded_cols: int) -> torch.Tensor:
    rows, cols = weight.shape
    padded = weight.new_zeros(padded_rows, padded_cols, dtype=torch.bfloat16)
    padded[:rows, :cols].copy_(weight)
    return padded


def _pad_fc1(weight: torch.Tensor, gated: bool, padded_i: int, padded_h: int) -> torch.Tensor:
    """Zero-pad one FC1 expert weight into the CuTe DSL row order.

    MCore stores gated FC1 as ``[gate; up]``. FlashInfer expects ``[up; gate]`` with each half
    padded separately and the halves interleaved in ``_GLU_INTERLEAVE_ROWS``-row blocks.
    """
    if not gated:
        return _zero_pad(weight, padded_i, padded_h)
    gate, up = weight.chunk(2, dim=0)
    halves = torch.stack((_zero_pad(up, padded_i, padded_h), _zero_pad(gate, padded_i, padded_h)))
    blocks = halves.view(2, padded_i // _GLU_INTERLEAVE_ROWS, _GLU_INTERLEAVE_ROWS, padded_h)
    return blocks.transpose(0, 1).reshape(2 * padded_i, padded_h)


def _quantize_expert(
    padded: torch.Tensor,
    data_out: torch.Tensor,
    swizzled_scale_out: torch.Tensor,
    alpha_out: torch.Tensor,
) -> None:
    """Quantize one zero-padded ``[rows, cols]`` expert weight into CuTe DSL NVFP4 storage."""
    padded_rows, padded_cols = padded.shape
    encode = _global_encode_scale(padded)
    data, scale = nvfp4_quantize(padded, encode, sfLayout=SfLayout.layout_128x4)
    data_out.copy_(data.view(torch.uint8).reshape(padded_rows, padded_cols // 2))
    swizzled_scale_out.copy_(
        scale.view(torch.uint8).reshape(padded_rows, padded_cols // NVFP4_BLOCK_SIZE)
    )
    alpha_out.copy_(encode.reciprocal())


def _validate_stacks(
    fc1_weight: torch.Tensor, fc2_weight: torch.Tensor
) -> tuple[int, int, int, bool]:
    """Return (experts, intermediate, hidden, gated); gated FC1 has twice FC2's input width."""
    if fc1_weight.ndim != 3 or fc2_weight.ndim != 3:
        raise ValueError(
            "expected [experts, M, K] expert stacks, got "
            f"fc1={tuple(fc1_weight.shape)}, fc2={tuple(fc2_weight.shape)}"
        )
    experts, hidden_size, intermediate_size = fc2_weight.shape
    fc1_rows = fc1_weight.shape[1]
    gated = fc1_rows == 2 * intermediate_size
    if tuple(fc1_weight.shape) != (experts, fc1_rows, hidden_size) or fc1_rows not in (
        intermediate_size,
        2 * intermediate_size,
    ):
        raise ValueError(
            f"FC1 shape {tuple(fc1_weight.shape)} does not match FC2 shape "
            f"{tuple(fc2_weight.shape)}: expected [experts, intermediate or 2 * intermediate, "
            "hidden]"
        )
    return experts, intermediate_size, hidden_size, gated


def _check_refresh_metadata(
    out, experts, hidden_size, intermediate_size, padded_h, padded_i, gated
):
    expected = (experts, hidden_size, intermediate_size, padded_h, padded_i, gated)
    actual = (
        out.num_local_experts,
        out.hidden_size,
        out.intermediate_size,
        out.padded_hidden_size,
        out.padded_intermediate_size,
        out.gated,
    )
    if actual != expected:
        raise ValueError(
            "existing FlashInfer routed NVFP4 weight metadata changed across refit: "
            f"got {actual}, expected {expected}"
        )


@torch.no_grad()
def prepare_routed_nvfp4_weights(
    fc1_weight: torch.Tensor,
    fc2_weight: torch.Tensor,
    out: FlashInferRoutedNVFP4Weights | None = None,
) -> FlashInferRoutedNVFP4Weights:
    """Quantize BF16 squared-ReLU or SwiGLU expert stacks into the per-token NVFP4 kernel layout.

    Args:
        fc1_weight: ``[experts, intermediate, hidden]`` BF16 FC1 weights, or
            ``[experts, 2 * intermediate, hidden]`` as MCore's ``[gate; up]`` for SwiGLU.
        fc2_weight: ``[experts, hidden, intermediate]`` BF16 FC2 weights.
        out: When provided, its tensors are refreshed in place. This preserves
            addresses captured by CUDA graphs across model refits.
    """
    require_flashinfer_routed_nvfp4()
    experts, intermediate_size, hidden_size, gated = _validate_stacks(fc1_weight, fc2_weight)
    padded_h = _round_up(hidden_size, _ALIGNMENT)
    padded_i = _round_up(intermediate_size, _ALIGNMENT)
    fc1_rows = 2 * padded_i if gated else padded_i
    device = fc1_weight.device

    if out is None:

        def _empty(*shape, dtype):
            return torch.empty(*shape, dtype=dtype, device=device)

        # Each expert's row count is a multiple of 128, so per-expert 128x4-swizzled
        # scale blocks concatenate into the swizzled layout of the stacked matrix.
        fc1_scale_swizzled = _empty(
            experts * fc1_rows, padded_h // NVFP4_BLOCK_SIZE, dtype=torch.uint8
        )
        fc2_scale_swizzled = _empty(
            experts * padded_h, padded_i // NVFP4_BLOCK_SIZE, dtype=torch.uint8
        )
        out = FlashInferRoutedNVFP4Weights(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            padded_hidden_size=padded_h,
            padded_intermediate_size=padded_i,
            gated=gated,
            fc1_data=_empty(experts, fc1_rows, padded_h // 2, dtype=torch.uint8),
            fc1_scale_swizzled=fc1_scale_swizzled,
            fc1_scale=convert_sf_to_mma_layout(
                fc1_scale_swizzled, m=fc1_rows, k=padded_h, num_groups=experts
            ),
            fc1_alpha=_empty(experts, dtype=torch.float32),
            fc2_data=_empty(experts, padded_h, padded_i // 2, dtype=torch.uint8),
            fc2_scale_swizzled=fc2_scale_swizzled,
            fc2_scale=convert_sf_to_mma_layout(
                fc2_scale_swizzled, m=padded_h, k=padded_i, num_groups=experts
            ),
            fc2_alpha=_empty(experts, dtype=torch.float32),
            fc2_input_scale=torch.full(
                (1,), PER_TOKEN_ACTIVATION_GLOBAL_SCALE, dtype=torch.float32, device=device
            ),
        )
    else:
        _check_refresh_metadata(
            out, experts, hidden_size, intermediate_size, padded_h, padded_i, gated
        )

    for expert in range(experts):
        _quantize_expert(
            _pad_fc1(fc1_weight[expert], gated, padded_i, padded_h),
            out.fc1_data[expert],
            out.fc1_scale_swizzled[expert * fc1_rows : (expert + 1) * fc1_rows],
            out.fc1_alpha[expert : expert + 1],
        )
        _quantize_expert(
            _zero_pad(fc2_weight[expert], padded_h, padded_i),
            out.fc2_data[expert],
            out.fc2_scale_swizzled[expert * padded_h : (expert + 1) * padded_h],
            out.fc2_alpha[expert : expert + 1],
        )
    for swizzled, mma, m, k in (
        (out.fc1_scale_swizzled, out.fc1_scale, fc1_rows, padded_h),
        (out.fc2_scale_swizzled, out.fc2_scale, padded_h, padded_i),
    ):
        converted = convert_sf_to_mma_layout(swizzled, m=m, k=k, num_groups=experts)
        if converted.data_ptr() != mma.data_ptr():
            mma.copy_(converted)
    return out


def quantize_routed_nvfp4_input(
    hidden_states: torch.Tensor, padded_hidden_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad and quantize routed-MoE input with one FP32 scale per token.

    Returns packed E2M1 data ``[tokens, padded_hidden // 2]``, linear block scales
    ``[tokens, padded_hidden // 16]`` (as returned by FlashInfer), and FP32
    per-token scales ``[tokens]``.
    """
    hidden_padding = padded_hidden_size - hidden_states.shape[-1]
    if hidden_padding < 0:
        raise ValueError(
            f"padded hidden size {padded_hidden_size} is smaller than input hidden size "
            f"{hidden_states.shape[-1]}"
        )
    padded_hidden = F.pad(hidden_states, (0, hidden_padding)) if hidden_padding else hidden_states
    quantized_hidden, hidden_scale, per_token_scale = nvfp4_quantize(
        padded_hidden.contiguous(),
        PER_TOKEN_ACTIVATION_GLOBAL_SCALE,
        sfLayout=SfLayout.layout_linear,
        per_token_activation=True,
    )
    hidden_scale = hidden_scale.reshape(
        hidden_states.shape[0], padded_hidden_size // NVFP4_BLOCK_SIZE
    )
    return quantized_hidden, hidden_scale, per_token_scale


def _get_cute_dsl_wrapper(
    weights: FlashInferRoutedNVFP4Weights,
    num_experts: int,
    top_k: int,
    local_expert_offset: int,
    activation_type: int,
    device: torch.device,
):
    # The fused finalize accumulates top-k routes with output-dtype atomics, whose
    # order is not fixed; the two-stage finalize reduces each token's routes in order.
    use_fused_finalize = not (
        torch.are_deterministic_algorithms_enabled() or is_batch_invariant_mode_enabled()
    )
    key = (
        use_fused_finalize,
        num_experts,
        top_k,
        weights.padded_hidden_size,
        weights.padded_intermediate_size,
        weights.num_local_experts,
        local_expert_offset,
        activation_type,
        str(device),
    )
    wrapper = _CUTE_DSL_WRAPPERS.get(key)
    if wrapper is None:
        # Persistent stream/event resources are created here, outside graph capture.
        wrapper = CuteDslMoEWrapper(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=weights.padded_hidden_size,
            intermediate_size=weights.padded_intermediate_size,
            use_cuda_graph=True,
            num_local_experts=weights.num_local_experts,
            local_expert_offset=local_expert_offset,
            output_dtype=torch.bfloat16,
            device=str(device),
            activation_type=activation_type,
            use_fused_finalize=use_fused_finalize,
            quant_mode="w4a4",
        )
        _CUTE_DSL_WRAPPERS[key] = wrapper
    return wrapper


def flashinfer_routed_nvfp4_moe(
    hidden_states: torch.Tensor,
    routing_map: torch.Tensor,
    probabilities: torch.Tensor,
    weights: FlashInferRoutedNVFP4Weights,
    *,
    num_experts: int,
    local_expert_offset: int,
    activation_type: int,
    out: torch.Tensor | None = None,
    token_capacity: int | None = None,
    use_bounded_rows: bool = False,
) -> torch.Tensor:
    """Run the FlashInfer per-token NVFP4 squared-ReLU or SwiGLU MoE.

    Rows whose routing entries are -1 contribute zero output, so a graph-stable
    dispatcher buffer may be passed in full. When token_capacity is set and the
    controller marks the current graph safe, only that fixed prefix is quantized
    and processed, as for routed MXFP8; rows past it are left untouched in out.
    """
    require_flashinfer_routed_nvfp4()
    expected_activation = ActivationType.Swiglu if weights.gated else ActivationType.Relu2
    if activation_type != expected_activation.value:
        raise ValueError(
            f"{'Gated' if weights.gated else 'Non-gated'} per-token NVFP4 expert weights need "
            f"activation_type={expected_activation.name}; got activation_type={activation_type}"
        )
    if hidden_states.shape[-1] != weights.hidden_size:
        raise ValueError(
            f"hidden size {hidden_states.shape[-1]} does not match expert hidden size "
            f"{weights.hidden_size}"
        )
    if probabilities.dtype != torch.float32:
        raise TypeError(
            f"FlashInfer routed MoE requires FP32 probabilities; got {probabilities.dtype}"
        )

    active_rows, _ = select_routed_mxfp8_active_rows(
        hidden_states.shape[0], token_capacity=token_capacity, use_bounded_rows=use_bounded_rows
    )
    hidden_states = hidden_states[:active_rows]
    routing_map = routing_map[:active_rows]
    probabilities = probabilities[:active_rows]
    quantized_hidden, hidden_scale, per_token_scale = quantize_routed_nvfp4_input(
        hidden_states, weights.padded_hidden_size
    )
    wrapper = _get_cute_dsl_wrapper(
        weights,
        num_experts=num_experts,
        top_k=routing_map.shape[1],
        local_expert_offset=local_expert_offset,
        activation_type=activation_type,
        device=quantized_hidden.device,
    )
    output = wrapper.run(
        quantized_hidden,
        hidden_scale.unsqueeze(-1),
        routing_map.to(torch.int32),
        probabilities,
        weights.fc1_data,
        weights.fc1_scale,
        weights.fc1_alpha,
        weights.fc2_input_scale,
        weights.fc2_data,
        weights.fc2_scale,
        weights.fc2_alpha,
        per_token_scale=per_token_scale,
    )[:, : weights.hidden_size].contiguous()

    if out is not None:
        if out.shape[0] < output.shape[0] or out.shape[1:] != output.shape[1:]:
            raise ValueError(
                f"output buffer shape {tuple(out.shape)} cannot hold {tuple(output.shape)}"
            )
        out[: output.shape[0]].copy_(output)
        return out
    return output
