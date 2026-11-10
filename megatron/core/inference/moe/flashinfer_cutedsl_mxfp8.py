# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""FlashInfer CuTe-DSL fused MXFP8 MoE for MCore MXFP8 expert weights.

The CuTe-DSL path runs MXFP8 x MXFP8 non-gated squared-ReLU experts as two fused grouped
GEMMs: FC1 with squared-ReLU and MXFP8 requantization in its epilogue, and FC2 with the
routing-weight scaling and token scatter in its epilogue. Optionally, on devices with two
CUDA locality domains (Rubin), every expert's intermediate dimension is split across the
domains ("intermediate split + add"): domain d keeps half of every local expert's FC1 rows
and the matching half of its FC2 columns (the FC2 reduction dimension) in its own memory,
routing runs once, each domain runs FC1 into a private intermediate buffer and FC2 over its
half on a green-context stream of that domain, and both domains add their partial sums into
the token output, so there is no join between FC1 and FC2.

Weights are repacked once from MCore's canonical Triton/cuBLAS MXFP8 layout without
requantization: FP8 codes and E8M0 scales are copied bit-exactly. The FC1 output rows (and
FC2 reduction columns) are zero-padded to a multiple of 128; the hidden (latent) width is
kept unpadded (the activation quantizer zero-pads K itself).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import torch

try:
    from flashinfer import cute_dsl_fused_moe
    from flashinfer.fp8_quantization import mxfp8_quantize

    HAVE_FLASHINFER_CUTEDSL_MXFP8 = True
    _FLASHINFER_CUTEDSL_MXFP8_IMPORT_ERROR: Optional[ImportError] = None
except ImportError as exc:
    HAVE_FLASHINFER_CUTEDSL_MXFP8 = False
    _FLASHINFER_CUTEDSL_MXFP8_IMPORT_ERROR = exc

from megatron.core.inference.moe.flashinfer_mxfp8 import _unshuffle_cublas_scale
from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor

logger = logging.getLogger(__name__)


def _die_streams(domains):
    """(green-context streams, per-die SM count) of the locality domains."""
    return list(domains.streams), int(domains.sm_counts[0])


_BLOCK = 32  # MXFP8 scale block (elements per E8M0 scale)
_ROW_TILE = 128  # scale-layout row atom
_COL_ATOM = 4  # scale columns per atom


def require_flashinfer_cutedsl_mxfp8() -> None:
    """Raise an error when the FlashInfer CuTe-DSL MXFP8 MoE API is unavailable."""
    if not HAVE_FLASHINFER_CUTEDSL_MXFP8:
        raise RuntimeError(
            "inference_grouped_gemm_backend='flashinfer_cutedsl' requires a FlashInfer build "
            "providing cute_dsl_fused_moe(quant_mode='mxfp8') and nvidia-cutlass-dsl."
        ) from _FLASHINFER_CUTEDSL_MXFP8_IMPORT_ERROR


def _round_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


@dataclass(frozen=True)
class FlashInferCuteDslOptions:
    """Execution options parsed from ``inference_flashinfer_cutedsl_options`` (JSON).

    Keys (all optional):
      tactic: fixed CuTe-DSL tactic tuple (JSON list); default: FlashInfer's AutoTuner.
      localized_options: dict forwarded to cute_dsl_fused_moe(localized_options=...) with
          inference_moe_locality_domains; default {"fc2_split": "k", "fc2_accumulator":
          "output"}: the intermediate split + add, both dies adding BF16 partial sums into
          the output (each K half rounded once). fc2_accumulator "float32" sums the halves
          in an fp32 buffer and rounds once. Only fc2_split "k" is supported.
      private_intermediate_max_tokens: preallocate each die's FC1
          output (data + scales) in its own memory, sized for calls of up to this many tokens
          (default 4096; larger calls let FlashInfer allocate; 0 disables).
    """

    tactic: Optional[tuple] = None
    localized_options: dict = field(
        default_factory=lambda: {"fc2_split": "k", "fc2_accumulator": "output"}
    )
    private_intermediate_max_tokens: int = 4096

    @classmethod
    def from_json(cls, text: Optional[str]) -> "FlashInferCuteDslOptions":
        """Parse the JSON option string (None or empty -> defaults)."""
        if not text:
            return cls()
        raw = json.loads(text)
        unknown = set(raw) - {"tactic", "localized_options", "private_intermediate_max_tokens"}
        if unknown:
            raise ValueError(
                f"unknown inference_flashinfer_cutedsl_options keys: {sorted(unknown)}"
            )

        def as_tuple(value):
            return tuple(as_tuple(v) for v in value) if isinstance(value, list) else value

        tactic = raw.get("tactic")
        kwargs = {}
        if "localized_options" in raw:
            kwargs["localized_options"] = dict(raw["localized_options"])
            kwargs["localized_options"].setdefault("fc2_split", "k")
            if kwargs["localized_options"]["fc2_split"] != "k":
                raise ValueError(
                    "inference_flashinfer_cutedsl_options localized_options['fc2_split'] "
                    "must be 'k' (intermediate split + add)"
                )
        return cls(
            tactic=as_tuple(tactic) if tactic is not None else None,
            private_intermediate_max_tokens=int(raw.get("private_intermediate_max_tokens", 4096)),
            **kwargs,
        )


@dataclass
class FlashInferCuteDslMXFP8Weights:
    """FC1/FC2 expert stacks in the CuTe-DSL MXFP8 layout, optionally domain-split."""

    fc1_data: torch.Tensor  # [E, I_pad, H] float8_e4m3fn
    fc1_scale: torch.Tensor  # 6-D MMA-layout E8M0 bytes (32, 4, I_pad/128, 4, ceil(H/128), E)
    fc2_data: torch.Tensor  # [E, H, I_pad] float8_e4m3fn
    fc2_scale: torch.Tensor  # (32, 4, ceil(H/128), 4, I_pad/128, E)
    alpha: torch.Tensor  # [E] float32 ones (per-expert global scale)
    hidden_size: int
    intermediate_size: int
    padded_intermediate_size: int
    options: FlashInferCuteDslOptions
    localized_shards: Optional[list] = None
    localized_streams: Optional[list] = None
    localized_sm_count: Optional[int] = None
    localized_domains: Any = None
    localized_row_splits: Optional[dict] = None
    # Per-die FC1 outputs in die memory for calls of <= private_max_tokens tokens.
    private_intermediates: Optional[dict] = None
    private_max_tokens: int = 0


def _logical_scale(weight: MXFP8Tensor, expert: int, rows: int, cols: int) -> torch.Tensor:
    """Return one expert's unswizzled [rows, cols // 32] E8M0 scale bytes."""
    return _unshuffle_cublas_scale(weight.scale[expert], rows, cols)


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
    """Unswizzle every expert's scales into a zero-padded [E, padded_rows, padded_cols/32]."""
    experts = weight.data.shape[0]
    out = torch.zeros(
        experts, padded_rows, padded_cols // _BLOCK, dtype=torch.uint8, device=weight.data.device
    )
    for expert in range(experts):
        out[expert, :rows, : cols // _BLOCK].copy_(_logical_scale(weight, expert, rows, cols))
    return out


def _mma_scale_from_logical(logical: torch.Tensor) -> torch.Tensor:
    """Pad a [E, R, C] logical scale slice to whole atoms (zeros) and swizzle it."""
    experts, rows, cols = logical.shape
    padded = torch.zeros(
        experts,
        _round_up(rows, _ROW_TILE),
        _round_up(cols, _COL_ATOM),
        dtype=torch.uint8,
        device=logical.device,
    )
    padded[:, :rows, :cols].copy_(logical)
    return _swizzle_to_mma(padded)


def _place(domains, domain: int, tensor: torch.Tensor) -> torch.Tensor:
    """Copy a tensor (contiguous, same shape) into one locality domain's memory."""
    placed = domains.empty(domain, tuple(tensor.shape), tensor.dtype)
    placed.copy_(tensor)
    return placed


_PRIVATE_INTERMEDIATE_CACHE: dict = {}


def _private_intermediates(domains, shards, max_tokens, *, top_k):
    """Per-die FC1 output buffers (FP8 data + MMA-layout scales) in each die's memory.

    Sized for the largest permuted row count of max_tokens tokens over the die's experts at
    any routing tile size. Shared by all MoE layers of the process: each die's FC1 -> FC2 of
    one layer completes on that die's stream before the next layer's FC1 is enqueued there.
    """
    from flashinfer.fused_moe.cute_dsl.moe_utils import get_max_num_permuted_tokens
    from flashinfer.fused_moe.cute_dsl.tuner import VALID_TILE_SIZES

    layout = []
    for shard in shards:
        # The die's experts at full width.
        die_experts, width = shard["w1_weight"].shape[0], shard["w1_weight"].shape[1]
        rows = max(
            get_max_num_permuted_tokens(max_tokens, top_k, die_experts, tile)
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
    options: FlashInferCuteDslOptions,
    locality_domains: bool = False,
    sm_split: str = "backfill",
    top_k: Optional[int] = None,
) -> FlashInferCuteDslMXFP8Weights:
    """Repack stacked canonical MXFP8 expert weights for cute_dsl_fused_moe.

    Args:
        fc1: [E, I, H] stacked MCore MXFP8 FC1 weights (Triton/cuBLAS layout).
        fc2: [E, H, I] stacked MCore MXFP8 FC2 weights.
        options: parsed execution options.
        locality_domains: split every local expert's intermediate dimension across the
            device's two CUDA locality domains (die d holds FC1 rows and FC2 columns
            [d*I/2, (d+1)*I/2) of every expert), placing every shard in its domain's memory.
        sm_split: green-context SM split ("backfill" uses every SM, "strict" only local SMs).
        top_k: router top-k, used to size the per-die private FC1 outputs.
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
    fc1_logical = _padded_logical_scales(fc1, inter, hidden, inter_pad, _round_up(hidden, 128))
    fc2_logical = _padded_logical_scales(
        fc2, hidden, inter, _round_up(hidden, _ROW_TILE), inter_pad
    )
    # Logical scale views limited to the real (unpadded-hidden) extent.
    fc1_logical = fc1_logical[:, :, : -(-hidden // _BLOCK)]
    fc2_logical = fc2_logical[:, :hidden]

    weights = FlashInferCuteDslMXFP8Weights(
        fc1_data=fc1_data,
        fc1_scale=_mma_scale_from_logical(fc1_logical),
        fc2_data=fc2_data,
        fc2_scale=_mma_scale_from_logical(fc2_logical),
        alpha=torch.ones(experts, dtype=torch.float32, device=device),
        hidden_size=hidden,
        intermediate_size=inter,
        padded_intermediate_size=inter_pad,
        options=options,
    )
    if not locality_domains:
        return weights

    from flashinfer.locality_domain import get_localized_domains

    domains = get_localized_domains(device, sm_split=sm_split)
    return _intermediate_split(
        weights, domains, fc1_data, fc2_data, inter_pad, options=options, top_k=top_k
    )


def _intermediate_split(weights, domains, fc1_data, fc2_data, inter_pad, *, options, top_k):
    """Intermediate split + add (fc2_split="k"): die d holds rows inter_rows[d] of every
    local expert's FC1 and the same columns (K) of its FC2, data and scales sliced bit-exactly
    at a 128-row scale-tile boundary and placed in its own memory. Each die's FC1 output goes
    to a private buffer in its own memory."""
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
    for domain, shard in enumerate(shards):
        for key, tensor in shard.items():
            actual = domains.pointer_domain(tensor)
            if actual != domain:
                raise RuntimeError(
                    f"locality-domain shard {domain} {key} was placed in domain {actual}"
                )
    if options.private_intermediate_max_tokens > 0 and top_k is not None:
        # Every die sees every routed row; its FC1 output is its intermediate half.
        weights.private_intermediates = _private_intermediates(
            domains, shards, options.private_intermediate_max_tokens, top_k=top_k
        )
        weights.private_max_tokens = options.private_intermediate_max_tokens
    weights.localized_shards = shards
    weights.localized_streams, weights.localized_sm_count = _die_streams(domains)
    weights.localized_domains = domains
    weights.localized_row_splits = {"fc2_split": "k", "intermediate_rows": inter_rows}
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
    # alignment=128 zero-pads K (data and scales) to a multiple of 128; FC1 reads the
    # weights' true K and zero-fills its residual tile.
    quantized, scale = mxfp8_quantize(
        hidden_states.contiguous(), is_sf_swizzled_layout=False, alignment=128
    )
    scale = scale.view(torch.uint8).reshape(num_tokens, quantized.shape[1] // _BLOCK)
    localized = {}
    if weights.localized_shards is not None:
        localized_options = weights.options.localized_options
        if weights.private_intermediates is not None and num_tokens <= weights.private_max_tokens:
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
        tactic=weights.options.tactic,
        **localized,
    )
    if out is not None:
        if out.shape[0] < output.shape[0] or out.shape[1:] != output.shape[1:]:
            raise ValueError(
                f"output buffer shape {tuple(out.shape)} cannot hold {tuple(output.shape)}"
            )
        out[: output.shape[0]].copy_(output)
        return out
    return output
