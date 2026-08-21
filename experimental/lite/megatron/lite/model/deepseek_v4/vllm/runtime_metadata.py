from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.forward_context import (
    ForwardContext,
    override_forward_context,
)
from vllm.model_executor.layers.sparse_attn_indexer import sparse_attn_indexer
from vllm.models.deepseek_v4.attention import _fill_short_context_topk_indices
from vllm.models.deepseek_v4.common.ops import (
    combine_topk_swa_indices,
    dequantize_and_gather_k_cache,
    fused_indexer_q_rope_quant,
    save_partial_states,
)
from vllm.models.deepseek_v4.common.rope import build_deepseek_v4_rope
from vllm.models.deepseek_v4.sparse_mla import build_c128a_topk_metadata
from vllm.transformers_utils.config import get_config
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadata,
    DeepseekV32IndexerPrefillMetadata,
    build_prefill_chunk_metadata,
)

from megatron.lite.model.deepseek_v4.config import DeepseekV4Config


DS4_SWA_BLOCK_SIZE = 256
DS4_FP8_MLA_TOKEN_BYTES = 584
DS4_FLASHMLA_INDEX_ALIGNMENT = 128


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
    """Map one packed training microbatch to caller-owned contiguous blocks.

    This is not a scheduler or cache allocator: all blocks belong to this
    invocation and are discarded after its forward/backward completes.
    """

    logical_lengths = [count // write_every for count in token_counts]
    block_counts = [max(1, _round_up(max(length, 1), block_size) // block_size) for length in logical_lengths]
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
            mapped[complete] = positions[complete].div(write_every, rounding_mode="floor")
        valid = mapped >= 0
        mapped[valid] += offsets[request] * block_size
        slots.append(mapped)
    return _PackedBlocks(table, torch.cat(slots).contiguous(), offsets[-1])


def _build_rope(
    hf_config: Any,
    config: DeepseekV4Config,
    *,
    compress_ratio: int,
    device: torch.device | str,
) -> Any:
    target_device = torch.device(device)
    with set_current_vllm_config(VllmConfig()), torch.device(target_device):
        rotary = build_deepseek_v4_rope(
            hf_config,
            head_dim=config.head_dim,
            rope_head_dim=config.qk_rope_head_dim,
            max_position_embeddings=config.max_position_embeddings,
            compress_ratio=compress_ratio,
        )
    return rotary.to(device=target_device)


@dataclass
class DS4CompressorRuntimeMetadata:
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
class DS4IndexerRuntimeMetadata:
    compressor: DS4CompressorRuntimeMetadata
    attention_metadata: Any
    k_cache_prefix: str
    topk_indices: torch.Tensor
    max_model_len: int
    max_total_seq_len: int


@dataclass
class AttentionKernelMetadata:
    positions: torch.Tensor
    slot_mapping: torch.Tensor
    cos_sin_cache: torch.Tensor
    swa_cache: torch.Tensor
    block_size: int
    indices: torch.Tensor
    topk_length: torch.Tensor
    output: torch.Tensor
    kv_workspace: torch.Tensor | None = None
    kv_workspace_slot_mapping: torch.Tensor | None = None
    compressor_workspace_slot_mapping: torch.Tensor | None = None
    query_start_loc: torch.Tensor | None = None
    swa_block_table: torch.Tensor | None = None
    swa_seq_lens: torch.Tensor | None = None
    prepare_flash: Any | None = None
    compressor_metadata: Any | None = None
    indexer_metadata: Any | None = None


@dataclass
class CPAttentionKernelMetadata:
    positions: torch.Tensor
    cos_sin_cache: torch.Tensor
    packed_seq_params: Any
    compressor_metadata: DS4CompressorRuntimeMetadata | None


def build_native_cp_attention_metadata(
    config: DeepseekV4Config,
    *,
    layer_idx: int,
    cos_sin_cache: torch.Tensor,
    local_positions: torch.Tensor,
    packed_seq_params: Any,
):
    device = local_positions.device
    ratio = max(1, config.compress_ratios[layer_idx])
    cp_compressor_metadata = None
    if ratio in (4, 128):
        local_rows = local_positions.numel()
        d_comp = 8 if ratio == 4 else ratio
        group_alignment = 32 // math.gcd(32, ratio)
        capacity = max(1, (local_rows + d_comp) // ratio)
        capacity = (
            (capacity + group_alignment - 1) // group_alignment
        ) * group_alignment
        synthetic_tokens = capacity * ratio
        synthetic_cos = torch.empty(
            (synthetic_tokens, cos_sin_cache.shape[1]),
            dtype=cos_sin_cache.dtype,
            device=device,
        )
        builder = DS4PrefillMetadataBuilder(
            config,
            layer_idx=layer_idx,
            device=device,
            cos_sin_cache=synthetic_cos,
        )
        cp_compressor_metadata, _ = builder._compressor_metadata_batch(
            [synthetic_tokens],
            head_dim=config.head_dim,
            token_bytes=builder._MAIN_TOKEN_BYTES,
        )
    return CPAttentionKernelMetadata(
        positions=local_positions,
        cos_sin_cache=cos_sin_cache,
        packed_seq_params=packed_seq_params,
        compressor_metadata=cp_compressor_metadata,
    )


class _SWAOnlyMetadataBuilder:
    def __init__(
        self,
        config: DeepseekV4Config,
        *,
        layer_idx: int = 0,
        device: torch.device | str,
        cos_sin_cache: torch.Tensor,
    ) -> None:
        if not 0 <= layer_idx < config.num_hidden_layers:
            raise ValueError(f"layer_idx is outside the model: {layer_idx}")
        self.config = config
        self.layer_idx = layer_idx
        requested_device = torch.device(device)
        self.device = (
            cos_sin_cache.device
            if requested_device.index is None
            and isinstance(cos_sin_cache, torch.Tensor)
            and cos_sin_cache.device.type == requested_device.type
            else requested_device
        )
        self.cos_sin_cache = cos_sin_cache
        self.compress_ratio = max(1, config.compress_ratios[layer_idx])
        if self.compress_ratio not in (1, 4, 128):
            raise ValueError(f"unsupported compress_ratio={self.compress_ratio}")
        if config.head_dim != 512 or config.index_head_dim != 128:
            raise ValueError("DS4 requires head_dim=512/index_head_dim=128")
        if config.sliding_window <= 0:
            raise ValueError("sliding_window must be positive")
        if (
            not isinstance(cos_sin_cache, torch.Tensor)
            or cos_sin_cache.ndim != 2
            or cos_sin_cache.device != self.device
            or cos_sin_cache.dtype != torch.float32
        ):
            raise ValueError(
                "cos_sin_cache must be a 2D float32 tensor on the runtime device"
            )

    def _prefill_indices(self, num_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
        width = _round_up(self.config.sliding_window, DS4_FLASHMLA_INDEX_ALIGNMENT)
        positions = torch.arange(num_tokens, dtype=torch.int32, device=self.device)
        offsets = torch.arange(width, dtype=torch.int32, device=self.device)
        starts = torch.clamp(positions - self.config.sliding_window + 1, min=0)
        lengths = torch.minimum(
            positions + 1,
            torch.tensor(
                self.config.sliding_window, dtype=torch.int32, device=self.device
            ),
        )
        indices = starts[:, None] + offsets[None, :]
        indices.masked_fill_(offsets[None, :] >= lengths[:, None], -1)
        return indices.unsqueeze(1).contiguous(), lengths.contiguous()

    def build_prefill_batch(self, token_counts: list[int]):

        if not token_counts or any(count <= 0 for count in token_counts):
            raise ValueError("token_counts must contain positive sequence lengths")
        blocks = _packed_blocks(
            token_counts, DS4_SWA_BLOCK_SIZE, device=self.device
        )
        cache = torch.empty(
            (
                blocks.num_blocks,
                DS4_SWA_BLOCK_SIZE,
                DS4_FP8_MLA_TOKEN_BYTES,
            ),
            dtype=torch.uint8,
            device=self.device,
        )
        positions = []
        indices = []
        topk_lengths = []
        query_start = [0]
        max_tokens = max(token_counts)
        for batch_index, count in enumerate(token_counts):
            sequence_positions = torch.arange(
                count, dtype=torch.int64, device=self.device
            )
            positions.append(sequence_positions)
            sequence_indices, sequence_lengths = self._prefill_indices(count)
            valid = sequence_indices >= 0
            sequence_indices = sequence_indices.clone()
            sequence_indices[valid] += batch_index * max_tokens
            indices.append(sequence_indices)
            topk_lengths.append(sequence_lengths)
            query_start.append(query_start[-1] + count)

        seq_lens = torch.tensor(token_counts, dtype=torch.int32, device=self.device)
        query_start_loc = torch.tensor(
            query_start, dtype=torch.int32, device=self.device
        )
        workspace_3d = torch.empty(
            (len(token_counts), max_tokens, self.config.head_dim),
            dtype=torch.bfloat16,
            device=self.device,
        )
        output = torch.empty(
            (
                sum(token_counts),
                self.config.num_attention_heads,
                self.config.head_dim,
            ),
            dtype=torch.bfloat16,
            device=self.device,
        )

        def prepare_flash() -> None:
            dequantize_and_gather_k_cache(
                workspace_3d,
                cache,
                seq_lens=seq_lens,
                gather_lens=seq_lens,
                block_table=blocks.block_table,
                block_size=DS4_SWA_BLOCK_SIZE,
                offset=0,
            )

        metadata = AttentionKernelMetadata(
            positions=torch.cat(positions).contiguous(),
            slot_mapping=blocks.slot_mapping,
            cos_sin_cache=self.cos_sin_cache,
            swa_cache=cache,
            block_size=DS4_SWA_BLOCK_SIZE,
            indices=torch.cat(indices).contiguous(),
            topk_length=torch.cat(topk_lengths).contiguous(),
            output=output,
            kv_workspace=workspace_3d.view(-1, 1, self.config.head_dim),
            kv_workspace_slot_mapping=torch.cat(
                [
                    batch_index * max_tokens
                    + torch.arange(count, dtype=torch.int64, device=self.device)
                    for batch_index, count in enumerate(token_counts)
                ]
            ).contiguous(),
            query_start_loc=query_start_loc,
            swa_block_table=blocks.block_table,
            swa_seq_lens=seq_lens,
            prepare_flash=prepare_flash,
        )
        return metadata

class DS4PrefillMetadataBuilder(_SWAOnlyMetadataBuilder):

    _MAIN_TOKEN_BYTES = 584
    _INDEXER_TOKEN_BYTES = 132
    _SWA_BLOCK_SIZE = DS4_SWA_BLOCK_SIZE
    _MLA_BLOCK_SIZE = 256

    def _allocate_k_cache(
        self, num_blocks: int, block_size: int, token_bytes: int
    ) -> torch.Tensor:
        block_stride = _round_up(block_size * token_bytes, 32)
        storage = torch.zeros(
            num_blocks * block_stride, dtype=torch.uint8, device=self.device
        )
        return torch.as_strided(
            storage,
            size=(num_blocks, block_size, token_bytes),
            stride=(block_stride, token_bytes, 1),
        )

    def _compressor_metadata_batch(
        self,
        token_counts: list[int],
        *,
        head_dim: int,
        token_bytes: int,
    ) -> tuple[DS4CompressorRuntimeMetadata, torch.Tensor]:

        ratio = self.compress_ratio
        coff = 2 if ratio == 4 else 1
        state_block_size = 4 if ratio == 4 else 8
        compressed_block_size = self._MLA_BLOCK_SIZE // ratio
        state = _packed_blocks(token_counts, state_block_size, device=self.device)
        compressed = _packed_blocks(
            token_counts,
            compressed_block_size,
            device=self.device,
            write_every=ratio,
        )
        request_indices = [
            torch.full((count,), request, dtype=torch.int32, device=self.device)
            for request, count in enumerate(token_counts)
        ]

        metadata = DS4CompressorRuntimeMetadata(
            state_cache=torch.zeros(
                state.num_blocks,
                state_block_size,
                2 * coff * head_dim,
                dtype=torch.float32,
                device=self.device,
            ),
            state_slot_mapping=state.slot_mapping,
            state_block_table=state.block_table,
            state_block_size=state_block_size,
            token_to_req_indices=torch.cat(request_indices).contiguous(),
            k_cache=self._allocate_k_cache(
                compressed.num_blocks, compressed_block_size, token_bytes
            ),
            k_slot_mapping=compressed.slot_mapping,
            cos_sin_cache=self.cos_sin_cache,
            rms_norm_eps=self.config.rms_norm_eps,
            rope_head_dim=self.config.qk_rope_head_dim,
        )
        return metadata, compressed.block_table

    @staticmethod
    def compressor_operation(
        *,
        kv_score: torch.Tensor,
        positions: torch.Tensor,
        ape: torch.Tensor,
        norm_weight: torch.Tensor,
        compress_ratio: int,
        head_dim: int,
        metadata: DS4CompressorRuntimeMetadata | DS4IndexerRuntimeMetadata,
    ) -> None:

        if isinstance(metadata, DS4IndexerRuntimeMetadata):
            metadata = metadata.compressor
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
        quant_block = 64 if head_dim == 512 else 128
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
            quant_block=quant_block,
            token_stride=token_stride,
            scale_dim=8 if head_dim == 512 else 4,
            **(
                {
                    "store_full_kv": False,
                    "store_full_fp8": False,
                    "fp8_scale": None,
                }
                if use_cutedsl
                else {}
            ),
        )

    def _indexer_attention_metadata_batch(
        self,
        token_counts: list[int],
        block_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> Any:
        query_start = torch.tensor(
            [0, *torch.tensor(token_counts).cumsum(0).tolist()],
            dtype=torch.int32,
            device=self.device,
        )
        seq_lens = torch.tensor(token_counts, dtype=torch.int32, device=self.device)
        compressed = seq_lens // self.compress_ratio
        chunk = build_prefill_chunk_metadata(
            0,
            len(token_counts),
            query_start,
            query_start.cpu(),
            seq_lens,
            compressed,
            compressed.cpu(),
            block_table,
            self.compress_ratio,
        )
        return DeepseekV32IndexerMetadata(
            seq_lens=seq_lens,
            max_seq_len=max(token_counts),
            slot_mapping=slot_mapping,
            num_decodes=0,
            num_decode_tokens=0,
            num_prefills=len(token_counts),
            num_prefill_tokens=sum(token_counts),
            prefill=DeepseekV32IndexerPrefillMetadata([] if chunk is None else [chunk]),
            decode=None,
        )

    @staticmethod
    def indexer_operation(
        *,
        qr: torch.Tensor,
        index_q: torch.Tensor,
        index_weights: torch.Tensor,
        positions: torch.Tensor,
        compress_ratio: int,
        topk: int,
        metadata: DS4IndexerRuntimeMetadata,
    ) -> torch.Tensor:

        if metadata.attention_metadata.max_seq_len // compress_ratio <= topk:
            padded_topk = 1 << (topk - 1).bit_length()
            _fill_short_context_topk_indices[(positions.numel(),)](
                metadata.topk_indices,
                positions,
                TOP_K=topk,
                COMPRESS_RATIO=compress_ratio,
                PADDED_TOP_K=padded_topk,
                num_warps=8,
            )
            return metadata.topk_indices

        q_quant, weights = fused_indexer_q_rope_quant(
            positions,
            index_q,
            metadata.compressor.cos_sin_cache,
            index_weights,
            index_q.shape[-1] ** -0.5,
            index_q.shape[1] ** -0.5,
            use_fp4=False,
        )
        context = ForwardContext(
            no_compile_layers={},
            attn_metadata={metadata.k_cache_prefix: metadata.attention_metadata},
            slot_mapping={},
        )
        with override_forward_context(context):
            return sparse_attn_indexer(
                qr,
                metadata.k_cache_prefix,
                metadata.compressor.k_cache,
                q_quant,
                None,
                None,
                weights,
                128,
                "ue8m0",
                topk,
                index_q.shape[-1],
                metadata.max_model_len,
                metadata.max_total_seq_len,
                metadata.topk_indices,
                True,
                False,
                "",
                False,
            )

    def _c128_prefill_indices(self, num_tokens: int) -> torch.Tensor:
        width = max(128, _round_up(max(num_tokens // 128, 1), 128))
        positions = torch.arange(num_tokens, dtype=torch.int64, device=self.device)
        global_buffer = torch.empty((1, width), dtype=torch.int32, device=self.device)
        decode_lens = torch.empty(1, dtype=torch.int32, device=self.device)
        prefill_buffer = torch.empty(
            (num_tokens, width), dtype=torch.int32, device=self.device
        )
        return build_c128a_topk_metadata(
            positions,
            128,
            0,
            torch.zeros(num_tokens, dtype=torch.int32, device=self.device),
            torch.zeros((1, 1), dtype=torch.int32, device=self.device),
            2,
            torch.arange(num_tokens, dtype=torch.int64, device=self.device),
            global_buffer,
            decode_lens,
            prefill_buffer,
            max_compressed_tokens=width,
        )[2]

    def build_prefill_batch(self, token_counts: list[int]):

        if not token_counts or any(count <= 0 for count in token_counts):
            raise ValueError("token_counts must contain positive sequence lengths")
        if self.compress_ratio == 1:
            return super().build_prefill_batch(token_counts)

        base = super().build_prefill_batch(token_counts)
        main, main_block_table = self._compressor_metadata_batch(
            token_counts,
            head_dim=self.config.head_dim,
            token_bytes=self._MAIN_TOKEN_BYTES,
        )
        topk_parts = [
            (
                torch.full(
                    (count, self.config.index_topk),
                    -1,
                    dtype=torch.int32,
                    device=self.device,
                )
                if self.compress_ratio == 4
                else self._c128_prefill_indices(count)
            )
            for count in token_counts
        ]
        topk = torch.cat(topk_parts).contiguous()

        indexer_runtime = None
        indexer_block_table = None
        if self.compress_ratio == 4:
            indexer_compressor, indexer_block_table = (
                self._compressor_metadata_batch(
                    token_counts,
                    head_dim=self.config.index_head_dim,
                    token_bytes=self._INDEXER_TOKEN_BYTES,
                )
            )
            indexer_runtime = DS4IndexerRuntimeMetadata(
                compressor=indexer_compressor,
                attention_metadata=self._indexer_attention_metadata_batch(
                    token_counts,
                    indexer_block_table,
                    indexer_compressor.k_slot_mapping,
                ),
                k_cache_prefix=f"mlite.layers.{self.layer_idx}.indexer.k_cache",
                topk_indices=topk,
                max_model_len=self.config.max_position_embeddings // 4,
                # SparseAttnIndexer uses this value to allocate the gathered-K
                # workspace for the whole prefill chunk, not one request.  The
                # official vLLM module supplies a scheduler-wide total-token
                # upper bound; this caller-owned runtime can use the exact
                # packed-microbatch total instead.
                max_total_seq_len=sum(
                    count // self.compress_ratio for count in token_counts
                ),
            )

        compressed_lens = [count // self.compress_ratio for count in token_counts]
        # See the single-sequence path above.  Each packed sequence needs its
        # full SWA source because this metadata describes the whole prefill, not
        # only a final decode-style tail.
        gathered_lens = list(token_counts)
        max_compressed = max(compressed_lens)
        max_gathered = max(gathered_lens)
        workspace_width = max_compressed + max_gathered
        workspace = torch.empty(
            (len(token_counts), workspace_width, self.config.head_dim),
            dtype=torch.bfloat16,
            device=self.device,
        )
        combined_width = _round_up(
            topk.shape[-1] + self.config.sliding_window,
            DS4_FLASHMLA_INDEX_ALIGNMENT,
        )
        combined = torch.empty(
            (sum(token_counts), combined_width),
            dtype=torch.int32,
            device=self.device,
        )
        combined_lens = torch.empty(
            sum(token_counts), dtype=torch.int32, device=self.device
        )
        metadata = AttentionKernelMetadata(
            positions=base.positions,
            slot_mapping=base.slot_mapping,
            cos_sin_cache=self.cos_sin_cache,
            swa_cache=base.swa_cache,
            block_size=base.block_size,
            indices=topk.unsqueeze(1),
            topk_length=combined_lens,
            output=base.output,
            kv_workspace=workspace.view(-1, 1, self.config.head_dim),
            kv_workspace_slot_mapping=torch.cat(
                [
                    batch_index * workspace_width
                    + max_compressed
                    + torch.arange(count, dtype=torch.int64, device=self.device)
                    for batch_index, count in enumerate(token_counts)
                ]
            ).contiguous(),
            compressor_workspace_slot_mapping=torch.cat(
                [
                    batch_index * workspace_width
                    + torch.arange(
                        count // self.compress_ratio,
                        dtype=torch.int64,
                        device=self.device,
                    )
                    for batch_index, count in enumerate(token_counts)
                ]
            ).contiguous(),
            query_start_loc=base.query_start_loc,
            swa_block_table=base.swa_block_table,
            swa_seq_lens=base.swa_seq_lens,
            compressor_metadata=main,
            indexer_metadata=indexer_runtime,
        )
        def prepare_flash() -> None:
            if max_compressed:
                dequantize_and_gather_k_cache(
                    workspace,
                    main.k_cache,
                    seq_lens=torch.tensor(
                        compressed_lens, dtype=torch.int32, device=self.device
                    ),
                    gather_lens=None,
                    block_table=main_block_table,
                    block_size=self._MLA_BLOCK_SIZE // self.compress_ratio,
                    offset=0,
                )
            dequantize_and_gather_k_cache(
                workspace,
                base.swa_cache,
                seq_lens=base.swa_seq_lens,
                gather_lens=torch.tensor(
                    gathered_lens, dtype=torch.int32, device=self.device
                ),
                block_table=base.swa_block_table,
                block_size=self._SWA_BLOCK_SIZE,
                offset=max_compressed,
            )
            indices, lengths = combine_topk_swa_indices(
                # ``topk`` is the per-forward source layout.  ``metadata.indices``
                # is the derived FlashMLA layout and is overwritten below; using
                # it as input on activation recompute would combine twice.  The
                # indexer updates ``topk`` in place for C4 layers.
                topk,
                base.query_start_loc,
                base.swa_seq_lens,
                torch.tensor(gathered_lens, dtype=torch.int32, device=self.device),
                self.config.sliding_window,
                self.compress_ratio,
                topk.shape[-1],
                workspace_width,
                max_compressed,
                out=(combined, combined_lens),
            )
            metadata.indices = indices.unsqueeze(1)
            metadata.topk_length = lengths

        metadata.prepare_flash = prepare_flash
        return metadata


compressor_operation = DS4PrefillMetadataBuilder.compressor_operation
indexer_operation = DS4PrefillMetadataBuilder.indexer_operation


def build_prefill_metadata_builders(
    model_path: str,
    config: DeepseekV4Config,
    layer_indices: tuple[int, ...],
    device: torch.device,
) -> dict[int, DS4PrefillMetadataBuilder]:
    hf_config = get_config(model_path, trust_remote_code=True)
    caches = {}
    for ratio in {max(1, config.compress_ratios[i]) for i in layer_indices}:
        caches[ratio] = _build_rope(
            hf_config, config, compress_ratio=ratio, device=device
        ).cos_sin_cache.to(device=device, dtype=torch.float32)
    return {
        i: DS4PrefillMetadataBuilder(
            config, layer_idx=i, device=device, cos_sin_cache=caches[max(1, config.compress_ratios[i])]
        )
        for i in layer_indices
    }
