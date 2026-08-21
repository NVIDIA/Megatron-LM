from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.models.deepseek_v4.common.ops import save_partial_states
from vllm.models.deepseek_v4.common.rope import build_deepseek_v4_rope
from vllm.transformers_utils.config import get_config

from megatron.lite.model.deepseek_v4.config import DeepseekV4Config


def _round_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


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
    compressor_metadata: DS4CompressorMetadata | None


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
) -> DS4CompressorMetadata | None:
    if ratio not in (4, 128):
        return None
    device = cos_sin_cache.device
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
            2 * coff * config.head_dim,
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
            584,
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

    def build(self, positions: torch.Tensor, packed_seq_params: Any):
        return AttentionKernelMetadata(
            positions=positions,
            cos_sin_cache=self.cos_sin_cache,
            packed_seq_params=packed_seq_params,
            compressor_metadata=_build_compressor_metadata(
                self.config,
                ratio=self.ratio,
                rows=positions.numel(),
                cos_sin_cache=self.cos_sin_cache,
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
        ratio: _build_rope(
            hf_config, config, compress_ratio=ratio, device=device
        )
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
