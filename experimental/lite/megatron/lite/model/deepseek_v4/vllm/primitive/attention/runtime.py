"""Caller-owned vLLM cache metadata and native CSA/CP lowering."""

from __future__ import annotations

import math
import os
from copy import copy
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch
from vllm import envs
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.models.deepseek_v4.common.ops import save_partial_states
from vllm.models.deepseek_v4.common.rope import build_deepseek_v4_rope
from vllm.transformers_utils.config import get_config

from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.transformer.experimental_attention_variant.csa_utils import cp_utils
from megatron.lite.model.deepseek_v4.config import DeepseekV4Config
from megatron.lite.model.deepseek_v4.vllm.primitive.attention.backward import (
    _rope_and_qnorm,
)
from megatron.lite.model.deepseek_v4.vllm.primitive.attention.host_geometry import (
    compressed_sequence_boundaries,
    padded_sequence_boundaries,
)


def _round_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _top_k_per_row_prefill(
    logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    output: torch.Tensor,
    num_rows: int,
    stride0: int,
    stride1: int,
    topk: int,
) -> None:
    """Use deterministic score-order Top-K whenever BI is enabled."""
    if envs.VLLM_BATCH_INVARIANT:
        # The standalone BI extension has a fixed 4096-column radix-sort
        # capacity.  A 32K C4 request has 8192 compressed columns.  vLLM's
        # official prefill kernel uses its deterministic insertion-sort branch
        # for at most 12288 rows and has the same score/source-index ordering.
        # Keep the authoritative extension for every shape it supports and use
        # that existing vLLM path only for the larger full-model geometry.
        if logits.shape[1] > 4096:
            if num_rows > 12288:
                raise RuntimeError(
                    "DS4 BI Top-K fallback would leave vLLM's deterministic "
                    "insertion-sort range"
                )
            torch.ops._C.top_k_per_row_prefill(
                logits,
                row_starts,
                row_ends,
                output,
                num_rows,
                stride0,
                stride1,
                topk,
            )
            return
        if not hasattr(torch.ops.ds4_bi, "top_k_per_row_prefill"):
            library = os.environ.get("DS4_BI_TOPK_LIB")
            if library:
                torch.ops.load_library(library)
        if not hasattr(torch.ops.ds4_bi, "top_k_per_row_prefill"):
            raise RuntimeError(
                "VLLM_BATCH_INVARIANT requires the loaded DS4 deterministic "
                "Top-K extension"
            )
        torch.ops.ds4_bi.top_k_per_row_prefill(
            logits,
            row_starts,
            row_ends,
            output,
            num_rows,
            stride0,
            stride1,
            topk,
        )
        return

    from vllm import _custom_ops as ops

    ops.top_k_per_row_prefill(
        logits,
        row_starts,
        row_ends,
        output,
        num_rows,
        stride0,
        stride1,
        topk,
    )


@dataclass(frozen=True)
class _PackedBlocks:
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    num_blocks: int


def _packed_blocks(
    token_counts: list[int],
    block_size: int,
    *,
    device: torch.device,
    write_every: int = 1,
) -> _PackedBlocks:
    logical_lengths = [count // write_every for count in token_counts]
    block_counts = [
        max(1, _round_up(max(length, 1), block_size) // block_size)
        for length in logical_lengths
    ]
    offsets = [0]
    for count in block_counts:
        offsets.append(offsets[-1] + count)
    table = torch.full(
        (len(token_counts), max(block_counts)),
        -1,
        dtype=torch.int32,
        device=device,
    )
    slots = []
    for request, count in enumerate(token_counts):
        table[request, : block_counts[request]] = torch.arange(
            offsets[request], offsets[request + 1], dtype=torch.int32, device=device
        )
        positions = torch.arange(count, device=device)
        if write_every == 1:
            mapped = positions.to(torch.int64)
        else:
            mapped = torch.full((count,), -1, dtype=torch.int64, device=device)
            complete = (positions + 1).remainder(write_every).eq(0)
            mapped[complete] = positions[complete].div(
                write_every, rounding_mode="floor"
            )
        valid = mapped >= 0
        mapped[valid] += offsets[request] * block_size
        slots.append(mapped)
    return _PackedBlocks(table, torch.cat(slots).contiguous(), offsets[-1])


def _build_rope(
    hf_config: Any,
    config: DeepseekV4Config,
    *,
    compress_ratio: int,
    device: torch.device,
) -> torch.Tensor:
    with set_current_vllm_config(VllmConfig()), torch.device(device):
        rotary = build_deepseek_v4_rope(
            hf_config,
            head_dim=config.head_dim,
            rope_head_dim=config.qk_rope_head_dim,
            max_position_embeddings=config.max_position_embeddings,
            compress_ratio=compress_ratio,
        )
    return rotary.to(device=device).cos_sin_cache.to(dtype=torch.float32)


@dataclass
class DS4CompressorMetadata:
    state_cache: torch.Tensor
    state_slot_mapping: torch.Tensor
    state_block_table: torch.Tensor
    state_block_size: int
    token_to_req_indices: torch.Tensor
    k_cache: torch.Tensor
    k_slot_mapping: torch.Tensor
    cos_sin_cache: torch.Tensor
    rms_norm_eps: float
    rope_head_dim: int


@dataclass
class AttentionKernelMetadata:
    positions: torch.Tensor
    cos_sin_cache: torch.Tensor
    packed_seq_params: Any
    sequence_boundaries: tuple[int, ...]
    compressed_boundaries: tuple[int, ...]
    cp_size: int
    compressor_metadata: DS4CompressorMetadata | None
    indexer_compressor_metadata: DS4CompressorMetadata | None


def _allocate_k_cache(
    num_blocks: int,
    block_size: int,
    token_bytes: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    block_stride = _round_up(block_size * token_bytes, 32)
    storage = torch.zeros(num_blocks * block_stride, dtype=torch.uint8, device=device)
    return torch.as_strided(
        storage,
        size=(num_blocks, block_size, token_bytes),
        stride=(block_stride, token_bytes, 1),
    )


def _build_compressor_metadata(
    config: DeepseekV4Config,
    *,
    ratio: int,
    rows: int,
    cos_sin_cache: torch.Tensor,
    head_dim: int | None = None,
) -> DS4CompressorMetadata | None:
    if ratio not in (4, 128):
        return None
    device = cos_sin_cache.device
    head_dim = config.head_dim if head_dim is None else head_dim
    d_comp = 8 if ratio == 4 else ratio
    alignment = 32 // math.gcd(32, ratio)
    capacity = _round_up(max(1, (rows + d_comp) // ratio), alignment)
    token_counts = [capacity * ratio]
    coff = 2 if ratio == 4 else 1
    state_block_size = 4 if ratio == 4 else 8
    compressed_block_size = 256 // ratio
    state = _packed_blocks(token_counts, state_block_size, device=device)
    compressed = _packed_blocks(
        token_counts,
        compressed_block_size,
        device=device,
        write_every=ratio,
    )
    return DS4CompressorMetadata(
        state_cache=torch.zeros(
            state.num_blocks,
            state_block_size,
            2 * coff * head_dim,
            dtype=torch.float32,
            device=device,
        ),
        state_slot_mapping=state.slot_mapping,
        state_block_table=state.block_table,
        state_block_size=state_block_size,
        token_to_req_indices=torch.zeros(
            token_counts[0], dtype=torch.int32, device=device
        ),
        k_cache=_allocate_k_cache(
            compressed.num_blocks,
            compressed_block_size,
            584 if head_dim == 512 else 132,
            device=device,
        ),
        k_slot_mapping=compressed.slot_mapping,
        cos_sin_cache=torch.empty(
            token_counts[0],
            cos_sin_cache.shape[1],
            dtype=cos_sin_cache.dtype,
            device=device,
        ),
        rms_norm_eps=config.rms_norm_eps,
        rope_head_dim=config.qk_rope_head_dim,
    )


def compressor_operation(
    *,
    kv_score: torch.Tensor,
    positions: torch.Tensor,
    ape: torch.Tensor,
    norm_weight: torch.Tensor,
    compress_ratio: int,
    head_dim: int,
    metadata: DS4CompressorMetadata,
) -> None:
    coff = 2 if compress_ratio == 4 else 1
    width = coff * head_dim
    kv, score = kv_score.split([width, width], dim=-1)
    save_partial_states(
        kv=kv,
        score=score,
        ape=ape,
        positions=positions,
        state_cache=metadata.state_cache,
        slot_mapping=metadata.state_slot_mapping,
        block_size=metadata.state_block_size,
        state_width=width,
        compress_ratio=compress_ratio,
        pdl_kwargs={"launch_pdl": False},
    )
    use_cutedsl = head_dim == 512 and kv_score.device.type == "cuda"
    if use_cutedsl:
        from vllm.models.deepseek_v4.nvidia.ops.sparse_attn_compress_cutedsl import (
            compress_norm_rope_store_cutedsl as compress,
        )
    else:
        from vllm.models.deepseek_v4.common.ops.fused_compress_quant_cache import (
            compress_norm_rope_store_triton as compress,
        )
    token_stride = (
        head_dim - metadata.rope_head_dim + 2 * metadata.rope_head_dim
        if head_dim == 512
        else head_dim
    )
    compress(
        state_cache=metadata.state_cache,
        num_actual=positions.numel(),
        token_to_req_indices=metadata.token_to_req_indices,
        positions=positions,
        slot_mapping=metadata.state_slot_mapping,
        block_table=metadata.state_block_table,
        block_size=metadata.state_block_size,
        state_width=width,
        cos_sin_cache=metadata.cos_sin_cache,
        kv_cache=metadata.k_cache,
        k_cache_metadata=SimpleNamespace(slot_mapping=metadata.k_slot_mapping),
        pdl_kwargs={"launch_pdl": False},
        head_dim=head_dim,
        rope_head_dim=metadata.rope_head_dim,
        compress_ratio=compress_ratio,
        overlap=compress_ratio == 4,
        use_fp4_cache=False,
        rms_norm_weight=norm_weight,
        rms_norm_eps=metadata.rms_norm_eps,
        quant_block=64 if head_dim == 512 else 128,
        token_stride=token_stride,
        scale_dim=8 if head_dim == 512 else 4,
        **(
            {"store_full_kv": False, "store_full_fp8": False, "fp8_scale": None}
            if use_cutedsl
            else {}
        ),
    )


class AttentionMetadataBuilder:
    def __init__(
        self,
        config: DeepseekV4Config,
        *,
        layer_idx: int,
        cos_sin_cache: torch.Tensor,
    ) -> None:
        self.config = config
        self.layer_idx = layer_idx
        self.ratio = max(1, config.compress_ratios[layer_idx])
        self.cos_sin_cache = cos_sin_cache

    def build(
        self,
        positions: torch.Tensor,
        packed_seq_params: Any,
        sequence_boundaries: tuple[int, ...],
        *,
        cp_size: int,
    ):
        if cp_size < 1:
            raise ValueError("cp_size must be positive")
        if len(sequence_boundaries) < 2 or sequence_boundaries[0] != 0:
            raise ValueError("sequence boundaries must start at zero and be non-empty")
        sequence_lengths = tuple(
            end - start
            for start, end in zip(sequence_boundaries, sequence_boundaries[1:])
        )
        if any(length <= 0 for length in sequence_lengths):
            raise ValueError("sequence boundaries must be strictly increasing")
        total_tokens = sequence_boundaries[-1]
        if total_tokens % cp_size:
            raise ValueError("padded token count must be divisible by cp_size")
        cu_seqlens = getattr(packed_seq_params, "cu_seqlens_q_padded", None)
        if cu_seqlens is None:
            cu_seqlens = getattr(packed_seq_params, "cu_seqlens_q", None)
        if cu_seqlens is None:
            raise ValueError("packed_seq_params must provide cu_seqlens_q")
        if cu_seqlens.ndim != 1 or cu_seqlens.numel() != len(sequence_boundaries):
            raise ValueError(
                "host sequence boundaries do not match packed cu_seqlens shape"
            )
        if positions.numel() != total_tokens // cp_size:
            raise ValueError(
                "host sequence terminal does not match CP-local position rows"
            )
        packed_max = getattr(packed_seq_params, "max_seqlen_q", None)
        if packed_max is not None and int(packed_max) != max(sequence_lengths):
            raise ValueError(
                "host sequence lengths do not match packed max_seqlen_q"
            )
        compressed_boundaries = (
            compressed_sequence_boundaries(sequence_boundaries, ratio=self.ratio)
            if self.ratio > 1
            else (0,) * len(sequence_boundaries)
        )
        return AttentionKernelMetadata(
            positions=positions,
            cos_sin_cache=self.cos_sin_cache,
            packed_seq_params=packed_seq_params,
            sequence_boundaries=sequence_boundaries,
            compressed_boundaries=compressed_boundaries,
            cp_size=cp_size,
            compressor_metadata=_build_compressor_metadata(
                self.config,
                ratio=self.ratio,
                rows=positions.numel(),
                cos_sin_cache=self.cos_sin_cache,
            ),
            indexer_compressor_metadata=(
                _build_compressor_metadata(
                    self.config,
                    ratio=self.ratio,
                    rows=positions.numel(),
                    cos_sin_cache=self.cos_sin_cache,
                    head_dim=self.config.index_head_dim,
                )
                if self.ratio == 4
                else None
            ),
        )


def build_attention_metadata_builders(
    model_path: str,
    config: DeepseekV4Config,
    layer_indices: tuple[int, ...],
    device: torch.device,
) -> dict[int, AttentionMetadataBuilder]:
    hf_config = get_config(model_path, trust_remote_code=True)
    caches = {
        ratio: _build_rope(hf_config, config, compress_ratio=ratio, device=device)
        for ratio in {max(1, config.compress_ratios[i]) for i in layer_indices}
    }
    return {
        layer_idx: AttentionMetadataBuilder(
            config,
            layer_idx=layer_idx,
            cos_sin_cache=caches[max(1, config.compress_ratios[layer_idx])],
        )
        for layer_idx in layer_indices
    }


@dataclass(frozen=True)
class CPCompressionGeometry:
    cu_seqlens_compressed: torch.Tensor
    hidden_compact: torch.Tensor
    compressed_group_ids: torch.Tensor
    seq_to_rank_row: torch.Tensor


def prepare_cp_compression_geometry(
    hidden_local: torch.Tensor,
    boundary_hidden: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    global_start: int,
    cp_size: int,
    ratio: int,
) -> CPCompressionGeometry:
    compressed_lens = torch.div(
        cu_seqlens[1:] - cu_seqlens[:-1], ratio, rounding_mode="floor"
    )
    cu_seqlens_compressed = torch.cat(
        (
            torch.zeros_like(cu_seqlens[:1]),
            torch.cumsum(compressed_lens, dim=0, dtype=torch.int32),
        )
    )
    hidden_compact, compressed_group_ids, seq_to_rank_row = (
        cp_utils.prepare_cp_compressor_input(
            hidden_local,
            boundary_hidden,
            cu_seqlens,
            cu_seqlens_compressed,
            global_start,
            cp_size,
            ratio,
        )
    )
    return CPCompressionGeometry(
        cu_seqlens_compressed,
        hidden_compact,
        compressed_group_ids,
        seq_to_rank_row,
    )


def gather_cp_compressed_rows(
    local_rows: torch.Tensor,
    seq_to_rank_row: torch.Tensor,
    *,
    cp_group,
) -> tuple[torch.Tensor, torch.Tensor]:
    rank_major = gather_from_sequence_parallel_region(local_rows, group=cp_group)
    return rank_major, torch.index_select(
        rank_major, 0, seq_to_rank_row.clamp_min(0).long()
    )


def _packed_cache(
    rows: int, device: torch.device, block_size: int = 64
) -> torch.Tensor:
    blocks = max(1, (int(rows) + block_size - 1) // block_size)
    block_stride = ((block_size * 584 + 575) // 576) * 576
    return torch.empty((blocks, block_stride), dtype=torch.uint8, device=device)


def _dequantize_packed_cache(
    cache: torch.Tensor, rows: int, *, block_size: int = 64
) -> torch.Tensor:
    from vllm.models.deepseek_v4.common.ops.cache_utils import (
        dequantize_and_gather_k_cache_triton,
    )

    output = torch.empty((1, rows, 512), dtype=torch.bfloat16, device=cache.device)
    if rows:
        blocks = (rows + block_size - 1) // block_size
        dequantize_and_gather_k_cache_triton(
            output,
            cache,
            torch.tensor([rows], dtype=torch.int32, device=cache.device),
            None,
            torch.arange(blocks, dtype=torch.int32, device=cache.device).unsqueeze(0),
            block_size,
            0,
        )
    return output.squeeze(0)


def quantized_main_k_visible(functional_k: torch.Tensor) -> torch.Tensor:
    from vllm.models.deepseek_v4.common.ops.cache_utils import (
        quantize_and_insert_k_cache,
    )

    rows = functional_k.shape[0]
    if rows == 0:
        return functional_k
    cache = _packed_cache(rows, functional_k.device)
    slots = torch.arange(rows, dtype=torch.int64, device=functional_k.device)
    quantize_and_insert_k_cache(
        functional_k.detach().contiguous(), cache, slots, block_size=64
    )
    visible = _dequantize_packed_cache(cache, rows)
    return visible + (functional_k - functional_k.detach())


def _dequantize_indexer_k_cache(
    k_cache: torch.Tensor, rows: int
) -> torch.Tensor:
    block_size = k_cache.shape[1]
    storage = k_cache.as_strided(
        (k_cache.shape[0] * k_cache.stride(0),),
        (1,),
    )
    slots = torch.arange(rows, dtype=torch.int64, device=k_cache.device)
    block_ids = torch.div(slots, block_size, rounding_mode="floor")
    block_offsets = slots % block_size
    block_bases = block_ids * k_cache.stride(0)
    columns = torch.arange(128, device=k_cache.device)
    fp8_bytes = storage[
        block_bases[:, None] + block_offsets[:, None] * 128 + columns
    ].contiguous()
    scale_bytes = storage[
        block_bases[:, None]
        + block_size * 128
        + block_offsets[:, None] * 4
        + torch.arange(4, device=k_cache.device)
    ].contiguous()
    visible = fp8_bytes.view(torch.float8_e4m3fn).float()
    visible *= scale_bytes.view(torch.float32)
    return visible.to(torch.bfloat16)


def official_compact_compressed_visible(
    functional_k: torch.Tensor,
    compact_score: torch.Tensor,
    ape: torch.Tensor,
    norm_weight: torch.Tensor,
    compressed_group_ids: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    *,
    operation,
    runtime_metadata,
    ratio: int,
    head_dim: int,
    compressed_boundaries: tuple[int, ...],
) -> torch.Tensor:
    """Evaluate compact CP groups with the official compressor kernel."""
    if (
        len(compressed_boundaries) < 2
        or compressed_boundaries[0] != 0
        or any(
            end < start
            for start, end in zip(
                compressed_boundaries, compressed_boundaries[1:]
            )
        )
    ):
        raise ValueError("compressed boundaries must be monotonic and start at zero")
    groups = compressed_boundaries[-1]
    if (
        groups > compressed_group_ids.numel()
        or groups > functional_k.shape[0]
        or groups * ratio > compact_score.shape[0]
    ):
        raise ValueError("host compressed terminal exceeds compact GPU rows")
    if groups == 0:
        return functional_k[:0]
    functional_k = functional_k[:groups]
    compressed_group_ids = compressed_group_ids[:groups]
    # Each packed request needs an independent compressor state reset.
    from vllm.models.deepseek_v4.common.ops.cache_utils import (
        dequantize_and_gather_k_cache_triton,
    )

    block_size = runtime_metadata.k_cache.shape[1]
    visible_parts = []
    for group_start, group_end in zip(
        compressed_boundaries, compressed_boundaries[1:]
    ):
        segment_groups = group_end - group_start
        if segment_groups == 0:
            continue
        segment_tokens = segment_groups * ratio
        segment = copy(runtime_metadata)
        segment.state_slot_mapping = runtime_metadata.state_slot_mapping[
            :segment_tokens
        ]
        segment.token_to_req_indices = runtime_metadata.token_to_req_indices[
            :segment_tokens
        ]
        segment.k_slot_mapping = runtime_metadata.k_slot_mapping[:segment_tokens]
        segment.state_cache.zero_()
        synthetic_positions = torch.arange(
            segment_tokens, dtype=torch.int64, device=compact_score.device
        )
        synthetic_starts = torch.arange(
            0, segment_tokens, ratio, dtype=torch.int64, device=compact_score.device
        )
        source_positions = (
            compressed_group_ids[group_start:group_end].clamp_min(0).long() * ratio
        )
        segment.cos_sin_cache.index_copy_(
            0,
            synthetic_starts,
            cos_sin_cache.index_select(0, source_positions),
        )
        operation(
            kv_score=compact_score[group_start * ratio : group_end * ratio].detach(),
            positions=synthetic_positions,
            ape=ape.detach(),
            norm_weight=norm_weight.detach(),
            compress_ratio=ratio,
            head_dim=head_dim,
            metadata=segment,
        )
        if head_dim == 128:
            visible_parts.append(
                _dequantize_indexer_k_cache(segment.k_cache, segment_groups)
            )
        else:
            output = torch.empty(
                (1, segment_groups, head_dim),
                dtype=torch.bfloat16,
                device=compact_score.device,
            )
            blocks = (segment_groups + block_size - 1) // block_size
            dequantize_and_gather_k_cache_triton(
                output,
                segment.k_cache,
                torch.tensor(
                    [segment_groups], dtype=torch.int32, device=compact_score.device
                ),
                None,
                torch.arange(
                    blocks, dtype=torch.int32, device=compact_score.device
                ).unsqueeze(0),
                block_size,
                0,
            )
            visible_parts.append(output.squeeze(0))
    visible = torch.cat(visible_parts)
    return visible + (functional_k - functional_k.detach())


def official_local_qk_visible(
    q: torch.Tensor,
    kv: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    kv_insert,
    *,
    eps: float,
    rope_dim: int,
    padded_heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = q.shape[0]
    cache = _packed_cache(rows, q.device)
    slots = torch.arange(rows, dtype=torch.int64, device=q.device)
    q_visible = kv_insert(
        q.detach(),
        kv.detach(),
        cache,
        slots,
        positions,
        cos_sin_cache,
        eps=eps,
        block_size=64,
        padded_heads=padded_heads,
    ).contiguous()
    k_visible = _dequantize_packed_cache(cache, rows)
    q_graph = _rope_and_qnorm(
        q, positions, cos_sin_cache, rope_dim, eps, normalize=True
    )
    k_graph = _rope_and_qnorm(
        kv, positions, cos_sin_cache, rope_dim, eps, normalize=False
    )
    return (
        q_visible + (q_graph - q_graph.detach()),
        k_visible + (k_graph - k_graph.detach()),
    )


def official_indexer_topk(
    index_q: torch.Tensor,
    index_weights: torch.Tensor,
    index_k_seq_major: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    *,
    sequence_boundaries: tuple[int, ...],
    compressed_boundaries: tuple[int, ...],
    global_start: int,
    ratio: int,
    topk: int,
) -> torch.Tensor:
    from vllm.models.deepseek_v4.common.ops.fused_indexer_q import (
        fused_indexer_q_rope_quant,
    )
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        per_token_group_quant_fp8,
    )
    from vllm.utils.deep_gemm import fp8_fp4_mqa_logits

    rows = index_q.shape[0]
    output = torch.full((rows, topk), -1, dtype=torch.int32, device=index_q.device)
    q_quant, weights = fused_indexer_q_rope_quant(
        positions,
        index_q.detach(),
        cos_sin_cache,
        index_weights.detach(),
        index_q.shape[-1] ** -0.5,
        index_q.shape[1] ** -0.5,
        use_fp4=False,
    )
    k_quant, k_scale = per_token_group_quant_fp8(
        index_k_seq_major.detach().contiguous(),
        group_size=index_k_seq_major.shape[-1],
        use_ue8m0=True,
    )
    if len(sequence_boundaries) != len(compressed_boundaries):
        raise ValueError("sequence and compressed boundary counts must match")
    if (
        sequence_boundaries[0] != 0
        or compressed_boundaries[0] != 0
        or compressed_boundaries[-1] != index_k_seq_major.shape[0]
    ):
        raise ValueError("host boundaries do not match indexer GPU rows")
    if index_k_seq_major.shape[0] == 0:
        return output
    k_scale = k_scale.view(torch.float32).squeeze(-1)
    local_end = global_start + rows
    for seq_idx, (seq_start, seq_end) in enumerate(
        zip(sequence_boundaries, sequence_boundaries[1:])
    ):
        query_start = max(global_start, seq_start)
        query_end = min(local_end, seq_end)
        if query_end <= query_start:
            continue
        q_start = query_start - global_start
        q_end = query_end - global_start
        k_start = compressed_boundaries[seq_idx]
        k_end = compressed_boundaries[seq_idx + 1]
        if k_end <= k_start:
            continue

        # vLLM prefill scores each request against its own compressed-K
        # segment.  Keeping N and the logits stride request-local is required
        # for deterministic Top-K tie resolution across packed batches.
        segment_rows = q_end - q_start
        row_starts = torch.zeros(
            segment_rows, dtype=torch.int32, device=index_q.device
        )
        row_ends = torch.div(
            positions[q_start:q_end] + 1, ratio, rounding_mode="floor"
        ).to(torch.int32)
        row_ends.clamp_max_(k_end - k_start)
        segment_k = k_quant[k_start:k_end].clone()
        segment_k_scale = k_scale[k_start:k_end].clone()
        padded_k_rows = ((segment_k.shape[0] + 127) // 128) * 128
        if padded_k_rows != segment_k.shape[0]:
            segment_k = torch.cat(
                (
                    segment_k,
                    segment_k.new_zeros(
                        (padded_k_rows - segment_k.shape[0], segment_k.shape[1])
                    ),
                ),
                dim=0,
            )
            segment_k_scale = torch.cat(
                (
                    segment_k_scale,
                    segment_k_scale.new_ones(
                        padded_k_rows - segment_k_scale.shape[0]
                    ),
                ),
                dim=0,
            )
        logits = fp8_fp4_mqa_logits(
            (q_quant[q_start:q_end].clone(), None),
            (segment_k, segment_k_scale),
            weights[q_start:q_end].clone(),
            row_starts,
            row_ends.contiguous(),
            clean_logits=False,
        )
        segment_output = torch.full(
            (segment_rows, topk), -1, dtype=torch.int32, device=index_q.device
        )
        _top_k_per_row_prefill(
            logits,
            row_starts,
            row_ends,
            segment_output,
            segment_rows,
            logits.stride(0),
            logits.stride(1),
            topk,
        )
        output[q_start:q_end] = segment_output
    return output


def c128_all_visible_topk(
    positions: torch.Tensor,
    *,
    width: int,
    ratio: int,
) -> torch.Tensor:
    columns = torch.arange(width, dtype=torch.int32, device=positions.device)
    counts = torch.div(positions + 1, ratio, rounding_mode="floor").to(torch.int32)
    return torch.where(
        columns.unsqueeze(0) < counts.unsqueeze(1),
        columns.unsqueeze(0),
        torch.full((), -1, dtype=torch.int32, device=positions.device),
    )


def compressed_width(max_seqlen: int, ratio: int, index_topk: int) -> int:
    if ratio == 4:
        return int(index_topk)
    rows = max(1, max_seqlen // max(1, ratio))
    return max(128, math.ceil(rows / 128) * 128)
