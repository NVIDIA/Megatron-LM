# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""FlashInfer CuTe-DSL fused MXFP8 MoE for MCore MXFP8 expert weights.

Runs MXFP8 non-gated squared-ReLU experts as two fused grouped GEMMs: FC1 with squared-ReLU
and MXFP8 requantization, FC2 with routing-weight scaling and token scatter. With
inference_moe_locality_domains (Rubin), every expert's intermediate dimension is split across
the two CUDA locality domains: each domain runs FC1 and FC2 over its half on its own
green-context stream and adds its partial sum into the output.

Weights are repacked bit-exactly (no requantization) from MCore's Triton/cuBLAS MXFP8 layout,
with the intermediate dimension zero-padded to a multiple of 128.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Optional

import torch

from megatron.core.inference.moe.flashinfer_mxfp8 import _unshuffle_cublas_scale
from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor

_BLOCK = 32  # MXFP8 scale block (elements per E8M0 scale)
_ROW_TILE = 128  # scale-layout row atom
_COL_ATOM = 4  # scale columns per atom

# FlashInfer's default MXFP8 MoE tactic: tile size 128, FC1 and FC2 with a 128x128 MMA tile
# on a 1x1 cluster, rastered along N.
DEFAULT_TACTIC = (128, ((128, 128), (1, 1), False), ((128, 128), (1, 1), False))
# Locality domains: intermediate split + add, both domains adding BF16 partial sums into the
# output (FlashInfer's default output_memset, "overlap_fc1").
LOCALIZED_OPTIONS = {"fc2_split": "k", "fc2_accumulator": "output"}
# Green-context SM split: "backfill" uses every SM, borrowing remainder SMs across domains.
LOCALITY_DOMAIN_SM_SPLIT = "backfill"
# Each locality domain's FC1 output is preallocated in its own memory for calls of up to this
# many tokens; larger calls let FlashInfer allocate it.
PRIVATE_INTERMEDIATE_MAX_TOKENS = 4096

_REQUIRED_CUTE_DSL_FUSED_MOE_ARGS = (
    "quant_mode",
    "localized_weights",
    "localized_streams",
    "localized_options",
)

try:
    from flashinfer import cute_dsl_fused_moe
    from flashinfer.fp8_quantization import mxfp8_quantize

    HAVE_FLASHINFER_CUTEDSL_MXFP8 = set(_REQUIRED_CUTE_DSL_FUSED_MOE_ARGS) <= set(
        inspect.signature(cute_dsl_fused_moe).parameters
    )
    _FLASHINFER_CUTEDSL_MXFP8_IMPORT_ERROR: Optional[ImportError] = None
except ImportError as exc:
    HAVE_FLASHINFER_CUTEDSL_MXFP8 = False
    _FLASHINFER_CUTEDSL_MXFP8_IMPORT_ERROR = exc


def require_flashinfer_cutedsl_mxfp8() -> None:
    """Raise an error when the FlashInfer CuTe-DSL MXFP8 MoE API is unavailable."""
    if not HAVE_FLASHINFER_CUTEDSL_MXFP8:
        raise RuntimeError(
            "inference_grouped_gemm_backend='flashinfer_cutedsl' requires a FlashInfer build "
            "providing cute_dsl_fused_moe(quant_mode='mxfp8') with "
            f"{', '.join(_REQUIRED_CUTE_DSL_FUSED_MOE_ARGS[1:])} and nvidia-cutlass-dsl."
        ) from _FLASHINFER_CUTEDSL_MXFP8_IMPORT_ERROR


def _round_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


@dataclass
class FlashInferCuteDslMXFP8Weights:
    """FC1/FC2 expert stacks in the CuTe-DSL MXFP8 layout, optionally split across
    locality domains."""

    fc1_data: torch.Tensor  # [E, I_pad, H] float8_e4m3fn
    fc1_scale: torch.Tensor  # 6-D MMA-layout E8M0 bytes (32, 4, I_pad/128, 4, ceil(H/128), E)
    fc2_data: torch.Tensor  # [E, H, I_pad] float8_e4m3fn
    fc2_scale: torch.Tensor  # (32, 4, ceil(H/128), 4, I_pad/128, E)
    alpha: torch.Tensor  # [E] float32 ones (per-expert global scale)
    hidden_size: int
    localized_shards: Optional[list] = None
    localized_streams: Optional[list] = None
    localized_sm_count: Optional[int] = None
    # Per-locality-domain FC1 outputs for calls of <= PRIVATE_INTERMEDIATE_MAX_TOKENS tokens.
    private_intermediates: Optional[dict] = None


def _swizzle_to_mma(logical: torch.Tensor) -> torch.Tensor:
    """[E, R, C] logical scale bytes (R % 128 == 0, C % 4 == 0) -> 6-D CuTe MMA layout.

    Physical order (E, R/128, C/4, 32, 4, 4) maps logical (r, c) to
    (r // 128, c // 4, r % 32, (r % 128) // 32, c % 4), the same 128x4 atom order as
    cuBLAS and flashinfer.cute_dsl.utils.convert_sf_to_mma_layout. Returns the strided
    (32, 4, R/128, 4, C/4, E) view.
    """
    experts, rows, cols = logical.shape
    physical = (
        logical.reshape(experts, rows // _ROW_TILE, 4, 32, cols // _COL_ATOM, _COL_ATOM)
        .permute(0, 1, 4, 3, 2, 5)
        .contiguous()
    )
    return physical.permute(3, 4, 1, 5, 2, 0)


def _padded_logical_scales(
    weight: MXFP8Tensor, rows: int, cols: int, padded_rows: int, padded_cols: int
) -> torch.Tensor:
    """Unswizzle every expert's scales into a zero-padded [E, padded_rows, padded_cols/32].

    padded_rows must be a multiple of 128 and padded_cols of 128 (whole scale atoms)."""
    experts = weight.data.shape[0]
    out = torch.zeros(
        experts, padded_rows, padded_cols // _BLOCK, dtype=torch.uint8, device=weight.data.device
    )
    for expert in range(experts):
        out[expert, :rows, : cols // _BLOCK].copy_(
            _unshuffle_cublas_scale(weight.scale[expert], rows, cols)
        )
    return out


def _place(domains, domain: int, tensor: torch.Tensor) -> torch.Tensor:
    """Copy a tensor (contiguous, same shape) into one locality domain's memory."""
    placed = domains.empty(domain, tuple(tensor.shape), tensor.dtype)
    placed.copy_(tensor)
    return placed


# Shared by all MoE layers of the process: this relies on each layer's FC1 and FC2 running in
# order on the locality domain's stream, so one layer's FC2 has read the buffer before the
# next layer's FC1 overwrites it.
_PRIVATE_INTERMEDIATE_CACHE: dict = {}


def _private_intermediates(domains, shards, max_tokens, *, top_k):
    """FC1 output buffers (FP8 data + MMA-layout scales) in each locality domain's memory,
    sized for max_tokens tokens at any routing tile size."""
    from flashinfer.fused_moe.cute_dsl.moe_utils import get_max_num_permuted_tokens
    from flashinfer.fused_moe.cute_dsl.tuner import VALID_TILE_SIZES

    layout = []
    for shard in shards:
        domain_experts, width = shard["w1_weight"].shape[0], shard["w1_weight"].shape[1]
        rows = max(
            get_max_num_permuted_tokens(max_tokens, top_k, domain_experts, tile)
            for tile in VALID_TILE_SIZES
        )
        layout.append((_round_up(rows, _ROW_TILE), width))
    key = (domains.device, tuple(layout))
    if key not in _PRIVATE_INTERMEDIATE_CACHE:
        outs, scales = [], []
        for domain, (rows, width) in enumerate(layout):
            outs.append(domains.empty(domain, (rows, width), torch.float8_e4m3fn))
            scale = domains.empty(
                domain, (32, 4, rows // _ROW_TILE, 4, -(-width // _ROW_TILE), 1), torch.uint8
            )
            scale.zero_()
            scales.append(scale)
        _PRIVATE_INTERMEDIATE_CACHE[key] = {"gemm1_out": outs, "gemm1_out_scale": scales}
    return _PRIVATE_INTERMEDIATE_CACHE[key]


@torch.no_grad()
def prepare_cutedsl_mxfp8_weights(
    fc1: MXFP8Tensor,
    fc2: MXFP8Tensor,
    *,
    locality_domains: bool = False,
    top_k: Optional[int] = None,
) -> FlashInferCuteDslMXFP8Weights:
    """Repack stacked canonical MXFP8 expert weights for cute_dsl_fused_moe.

    Args:
        fc1: [E, I, H] stacked MCore MXFP8 FC1 weights (Triton/cuBLAS layout).
        fc2: [E, H, I] stacked MCore MXFP8 FC2 weights.
        locality_domains: split every local expert's intermediate dimension across the
            device's two CUDA locality domains (domain d holds FC1 rows and FC2 columns
            [d*I/2, (d+1)*I/2) of every expert), placing every shard in its domain's memory.
        top_k: router top-k, used to size the per-domain private FC1 outputs.
    """
    require_flashinfer_cutedsl_mxfp8()
    for name, weight in (("fc1", fc1), ("fc2", fc2)):
        if weight.backend != "triton" or weight.data.ndim != 3:
            raise ValueError(
                f"{name} must be a stacked [E, M, K] MXFP8Tensor in the Triton/cuBLAS layout; "
                f"got backend={weight.backend!r}, shape={tuple(weight.data.shape)}"
            )
    experts, inter, hidden = fc1.data.shape
    if tuple(fc2.data.shape) != (experts, hidden, inter):
        raise ValueError(
            f"FC2 shape {tuple(fc2.data.shape)} does not match FC1 {tuple(fc1.data.shape)}"
        )
    if hidden % 16:
        raise ValueError(f"CuTe-DSL MXFP8 MoE requires hidden % 16 == 0; got {hidden}")
    inter_pad = _round_up(inter, _ROW_TILE)
    device = fc1.data.device

    # Zero rows of FC1 give zero (ReLU^2) activations; the matching FC2 columns are zero too.
    fc1_data = torch.zeros(experts, inter_pad, hidden, dtype=fc1.data.dtype, device=device)
    fc1_data[:, :inter].copy_(fc1.data)
    fc2_data = torch.zeros(experts, hidden, inter_pad, dtype=fc2.data.dtype, device=device)
    fc2_data[:, :, :inter].copy_(fc2.data)
    hidden_pad = _round_up(hidden, _ROW_TILE)
    fc1_scale = _padded_logical_scales(fc1, inter, hidden, inter_pad, hidden_pad)
    fc2_scale = _padded_logical_scales(fc2, hidden, inter, hidden_pad, inter_pad)

    weights = FlashInferCuteDslMXFP8Weights(
        fc1_data=fc1_data,
        fc1_scale=_swizzle_to_mma(fc1_scale),
        fc2_data=fc2_data,
        fc2_scale=_swizzle_to_mma(fc2_scale),
        alpha=torch.ones(experts, dtype=torch.float32, device=device),
        hidden_size=hidden,
    )
    if not locality_domains:
        return weights

    from flashinfer.locality_domain import get_localized_domains

    domains = get_localized_domains(device, sm_split=LOCALITY_DOMAIN_SM_SPLIT)
    return _intermediate_split(weights, domains, fc1_data, fc2_data, inter_pad, top_k=top_k)


def _intermediate_split(weights, domains, fc1_data, fc2_data, inter_pad, *, top_k):
    """Intermediate split + add (fc2_split="k"): locality domain d holds rows inter_rows[d] of
    every local expert's FC1 and the same columns (K) of its FC2, data and scales sliced
    bit-exactly at a 128-row scale-tile boundary and placed in its own memory."""
    tiles = inter_pad // _ROW_TILE
    if tiles % 2:
        raise ValueError(
            f"the intermediate split needs an even number of 128-row FC1 tiles; got {tiles} "
            f"(padded intermediate size {inter_pad})"
        )
    half = inter_pad // 2
    inter_rows = [(0, half), (half, inter_pad)]
    fc1_scale_phys = weights.fc1_scale.permute(5, 2, 4, 0, 1, 3)  # (E, mt, kt, 32, 4, 4)
    fc2_scale_phys = weights.fc2_scale.permute(5, 2, 4, 0, 1, 3)  # (E, H tiles, I tiles, ...)
    shards = []
    for domain, (lo, hi) in enumerate(inter_rows):
        tiles = slice(lo // _ROW_TILE, hi // _ROW_TILE)
        shards.append(
            {
                "w1_weight": _place(domains, domain, fc1_data[:, lo:hi].contiguous()),
                "w1_weight_sf": _place(
                    domains, domain, fc1_scale_phys[:, tiles].contiguous()
                ).permute(3, 4, 1, 5, 2, 0),
                "w2_weight": _place(domains, domain, fc2_data[:, :, lo:hi].contiguous()),
                "w2_weight_sf": _place(
                    domains, domain, fc2_scale_phys[:, :, tiles].contiguous()
                ).permute(3, 4, 1, 5, 2, 0),
            }
        )
    if top_k is not None:
        # Every locality domain sees every routed row; its FC1 output is its intermediate half.
        weights.private_intermediates = _private_intermediates(
            domains, shards, PRIVATE_INTERMEDIATE_MAX_TOKENS, top_k=top_k
        )
    weights.localized_shards = shards
    weights.localized_streams = list(domains.streams)
    weights.localized_sm_count = int(domains.sm_counts[0])  # SMs per locality domain
    return weights


def flashinfer_cutedsl_mxfp8_moe(
    hidden_states: torch.Tensor,
    routing_map: torch.Tensor,
    probabilities: torch.Tensor,
    weights: FlashInferCuteDslMXFP8Weights,
    *,
    num_experts: int,
    local_expert_offset: int,
    activation_type: int,
    clamped_relu2_limit: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run MXFP8 MoE through FlashInfer's CuTe-DSL fused kernels.

    ``clamped_relu2_limit`` is the (1,) FP32 limit L of ActivationType.ClampedRelu2
    (relu(L * tanh(x / L)) ** 2), required with that activation and None otherwise.

    Rows whose expert IDs are outside this rank's local range (including -1) are skipped
    and produce zero output rows. The BF16 -> MXFP8 activation quantization runs on every
    call.

    ``out``: optional output buffer with at least as many rows as ``hidden_states`` (such as
    the NVLS dispatcher's FP32 combine buffer); its leading rows receive a copy of the BF16
    result and ``out`` is returned.
    """
    if hidden_states.shape[-1] != weights.hidden_size:
        raise ValueError(
            f"hidden size {hidden_states.shape[-1]} does not match expert hidden "
            f"{weights.hidden_size}"
        )
    if probabilities.dtype != torch.float32:
        raise TypeError(
            f"FlashInfer CuTe-DSL MoE requires FP32 probabilities; got {probabilities.dtype}"
        )
    num_tokens = hidden_states.shape[0]
    if out is not None and (out.shape[0] < num_tokens or out.shape[1:] != hidden_states.shape[1:]):
        raise ValueError(
            f"output buffer shape {tuple(out.shape)} cannot hold "
            f"{(num_tokens, *hidden_states.shape[1:])}"
        )
    # alignment=128 zero-pads K (data and scales) to a multiple of 128; FC1 reads the
    # weights' true K and zero-fills its residual tile.
    quantized, scale = mxfp8_quantize(
        hidden_states.contiguous(), is_sf_swizzled_layout=False, alignment=128
    )
    scale = scale.view(torch.uint8).reshape(num_tokens, quantized.shape[1] // _BLOCK)
    localized = {}
    if weights.localized_shards is not None:
        localized_options = LOCALIZED_OPTIONS
        if (
            weights.private_intermediates is not None
            and num_tokens <= PRIVATE_INTERMEDIATE_MAX_TOKENS
        ):
            localized_options = {**localized_options, **weights.private_intermediates}
        localized = dict(
            localized_weights=weights.localized_shards,
            localized_streams=weights.localized_streams,
            localized_sm_count=weights.localized_sm_count,
            localized_options=localized_options,
        )
    output = cute_dsl_fused_moe(
        x=quantized,
        x_sf=scale,
        token_selected_experts=routing_map.to(torch.int32).contiguous(),
        token_final_scales=probabilities.contiguous(),
        w1_weight=weights.fc1_data,
        w1_weight_sf=weights.fc1_scale,
        w1_alpha=weights.alpha,
        fc2_input_scale=None,
        w2_weight=weights.fc2_data,
        w2_weight_sf=weights.fc2_scale,
        w2_alpha=weights.alpha,
        num_experts=num_experts,
        top_k=routing_map.shape[1],
        num_local_experts=weights.fc1_data.shape[0],
        local_expert_offset=local_expert_offset,
        output_dtype=torch.bfloat16,
        activation_type=activation_type,
        clamped_relu2_limit=clamped_relu2_limit,
        quant_mode="mxfp8",
        tactic=DEFAULT_TACTIC,
        **localized,
    )
    if out is not None:
        out[:num_tokens].copy_(output)
        return out
    return output
