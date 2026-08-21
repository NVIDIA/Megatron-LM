from __future__ import annotations

from typing import Callable

import torch
from vllm.model_executor.layers.batch_invariant import linear_batch_invariant
from vllm.models.common.ops import fused_q_kv_rmsnorm

from megatron.lite.model.deepseek_v4.config import DeepseekV4Config
from megatron.lite.model.deepseek_v4.deployment_block_fp8 import (
    DeploymentBlockFP8Adapter,
    DeploymentFusedBlockFP8Adapter,
)
from megatron.lite.model.deepseek_v4.vllm.primitive.attention.backward import (
    attach_indexer_aux_loss,
    visible_sparse_attention,
)
from megatron.lite.model.deepseek_v4.vllm.primitive.attention.metadata import (
    AttentionKernelMetadata,
    compressor_operation,
)
from megatron.lite.model.deepseek_v4.vllm.primitive.kernels import (
    insert_qkv,
    o_projection_visible,
    sparse_attention,
)
from megatron.lite.model.deepseek_v4.vllm.primitive.linear import (
    block_fp8_linear,
    fused_block_fp8_linear,
    visible_linear,
)
from megatron.lite.model.deepseek_v4.vllm.primitive.norm import fused_qkv_rms_norm
from megatron.lite.model.deepseek_v4.vllm.primitive.o_proj import o_projection
from megatron.lite.primitive.modules.attention.csa import CompressedSparseAttention
from megatron.lite.primitive.parallel import ParallelState


def _fp32_linear(value: torch.Tensor, *weights: torch.Tensor) -> torch.Tensor:
    return torch.mm(
        value.contiguous(), torch.cat(weights, dim=0).T, out_dtype=torch.float32
    )


class VLLMAttention(CompressedSparseAttention):
    def __init__(
        self,
        config: DeepseekV4Config,
        *,
        ps=None,
        layer_idx: int,
        indexer_loss_coeff: float = 0.0,
        cache_deployment_weights: bool = False,
    ):
        ps = ps or ParallelState()
        super().__init__(config, layer_idx=layer_idx, ps=ps)
        self.config = config
        configured_ratio = (
            config.compress_ratios[layer_idx]
            if layer_idx < len(config.compress_ratios)
            else 0
        )
        self.compress_ratio = max(1, configured_ratio)
        self.indexer_loss_coeff = indexer_loss_coeff
        self.fused_linear = DeploymentFusedBlockFP8Adapter(
            cache_weight=cache_deployment_weights
        )
        self.q_linear = DeploymentBlockFP8Adapter(
            cache_weight=cache_deployment_weights
        )
        self.indexer_q_linear = DeploymentBlockFP8Adapter(
            cache_weight=cache_deployment_weights
        )
        self._projection_streams: list[torch.cuda.Stream] | None = None
        self._projection_events: list[torch.cuda.Event] | None = None

    def clear_deployment_weight_cache(self) -> None:
        self.fused_linear.clear_cache()
        self.q_linear.clear_cache()
        self.indexer_q_linear.clear_cache()

    def _output_projection(
        self,
        result: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
    ) -> torch.Tensor:
        heads_per_group = self.config.num_attention_heads // self.config.o_groups
        nope_dim = self.config.head_dim - self.config.qk_rope_head_dim
        return o_projection(
            lambda value, wa, wb: o_projection_visible(
                value,
                positions,
                cos_sin_cache,
                wa,
                wb,
                n_groups=self.config.o_groups,
                heads_per_group=heads_per_group,
                nope_dim=nope_dim,
                rope_dim=self.config.qk_rope_head_dim,
                o_lora_rank=self.config.o_lora_rank,
            ),
            result,
            self.wo_a.weight,
            self.wo_b.weight,
            positions=positions,
            cos_sin_cache=cos_sin_cache,
            n_groups=self.config.o_groups,
            heads_per_group=heads_per_group,
            nope_dim=nope_dim,
            rope_dim=self.config.qk_rope_head_dim,
            o_lora_rank=self.config.o_lora_rank,
        )

    def _input_projections(self, hidden_states: torch.Tensor):
        def fused_projection():
            return fused_block_fp8_linear(
                self.fused_linear,
                hidden_states,
                self.wq_a.weight,
                self.wkv.weight,
            )

        aux_fns: list[Callable[[], torch.Tensor] | None] = [None, None, None]
        if self.compressor is not None:
            aux_fns[0] = lambda: fused_block_fp8_linear(
                _fp32_linear,
                hidden_states,
                self.compressor.wkv.weight,
                self.compressor.wgate.weight,
            )
        if self.indexer is not None:
            aux_fns[1] = lambda: visible_linear(
                linear_batch_invariant,
                hidden_states,
                self.indexer.weights_proj.weight,
            )
            aux_fns[2] = lambda: fused_block_fp8_linear(
                _fp32_linear,
                hidden_states,
                self.indexer.compressor.wkv.weight,
                self.indexer.compressor.wgate.weight,
            )
        if self._projection_streams is None:
            self._projection_streams = [torch.cuda.Stream() for _ in range(3)]
        if self._projection_events is None:
            self._projection_events = [torch.cuda.Event() for _ in range(4)]
        assert self._projection_events is not None
        from vllm.utils.multi_stream_utils import execute_in_parallel

        default_output, aux_outputs = execute_in_parallel(
            fused_projection,
            aux_fns,
            self._projection_events[0],
            self._projection_events[1:],
            self._projection_streams,
            enable=True,
        )
        # ``execute_in_parallel`` establishes the execution dependency with
        # CUDA events, but the auxiliary outputs were allocated on their
        # respective streams.  Tell the caching allocator that the current
        # stream also owns their lifetime before they are consumed below.
        # Without this, a later allocation can recycle their storage while a
        # current-stream kernel is still reading it; full launch blocking hid
        # that race in the RL integration test.
        current_stream = torch.cuda.current_stream(hidden_states.device)
        for output in aux_outputs:
            if isinstance(output, torch.Tensor):
                output.record_stream(current_stream)
        return default_output, aux_outputs

    def _forward_training_attention(
        self,
        hidden_states: torch.Tensor,
        metadata: AttentionKernelMetadata,
        q: torch.Tensor,
        kv: torch.Tensor,
        compressor_kv_score: torch.Tensor | None,
        index_q: torch.Tensor | None,
        indexer_weights: torch.Tensor | None,
        indexer_kv_score: torch.Tensor | None,
    ) -> torch.Tensor:
        cp_size = 1 if self.ps is None else self.ps.cp_size
        cp_rank = 0 if self.ps is None else self.ps.cp_rank
        cp_group = None if self.ps is None else self.ps.cp_group
        if cp_size > 1 and cp_group is None:
            raise RuntimeError("DS4 CP requires a model-owned CP group")
        if cp_size > 1 and self.indexer_loss_coeff:
            raise NotImplementedError(
                "DS4 CP indexer auxiliary loss is not implemented"
            )

        from megatron.core.transformer.experimental_attention_variant.csa_utils import (
            cp_layout_kernels,
            cp_utils,
        )
        from megatron.lite.model.deepseek_v4.vllm.primitive.attention.cp import (
            c128_all_visible_topk,
            compressed_width,
            official_indexer_topk,
            official_local_qk_visible,
            quantized_main_k_visible,
        )
        from megatron.lite.model.deepseek_v4.vllm.primitive.attention.backward import (
            compressed_compact_graph,
        )
        from megatron.lite.model.deepseek_v4.cp import (
            gather_cp_compressed_rows,
            prepare_cp_compression_geometry,
        )

        psp = metadata.packed_seq_params
        cu_seqlens = (
            psp.cu_seqlens_q_padded
            if psp.cu_seqlens_q_padded is not None
            else psp.cu_seqlens_q
        )
        if cu_seqlens is None:
            raise RuntimeError("DS4 packed attention requires cu_seqlens_q")
        l_local = hidden_states.shape[0]
        global_start = cp_rank * l_local
        positions = metadata.positions.reshape(-1).to(torch.int64)
        if positions.numel() != l_local:
            raise RuntimeError("DS4 CP position rows do not match local tokens")
        q_visible, kv_visible = official_local_qk_visible(
            q,
            kv,
            positions,
            metadata.cos_sin_cache,
            insert_qkv,
            eps=self.config.rms_norm_eps,
            rope_dim=self.config.qk_rope_head_dim,
            padded_heads=self.config.num_attention_heads,
        )

        boundary_hidden = (
            cp_utils.exchange_cp_boundary_hidden(
                hidden_states,
                self.compress_ratio,
                self.config.sliding_window,
                cp_group,
            )
            if cp_size > 1
            else hidden_states[:0]
        )
        d_window = boundary_hidden.shape[0]
        boundary_qr_kv = fused_block_fp8_linear(
            self.fused_linear,
            boundary_hidden,
            self.wq_a.weight,
            self.wkv.weight,
        )
        boundary_qr, boundary_kv = boundary_qr_kv.split(
            [self.config.q_lora_rank, self.config.head_dim], dim=-1
        )
        _, boundary_kv = fused_qkv_rms_norm(
            fused_q_kv_rmsnorm,
            boundary_qr,
            boundary_kv,
            self.q_norm.weight,
            self.kv_norm.weight,
            self.config.rms_norm_eps,
        )
        boundary_positions = cp_utils._thd_cp_position_ids(
            cu_seqlens, global_start - d_window, d_window
        ).to(torch.int64)
        boundary_dummy_q = boundary_kv.new_zeros(
            (
                d_window,
                self.config.num_attention_heads,
                self.config.head_dim,
            )
        )
        _, boundary_k_visible = official_local_qk_visible(
            boundary_dummy_q,
            boundary_kv,
            boundary_positions,
            metadata.cos_sin_cache,
            insert_qkv,
            eps=self.config.rms_norm_eps,
            rope_dim=self.config.qk_rope_head_dim,
            padded_heads=self.config.num_attention_heads,
        )

        compressed_rank_major = hidden_states.new_empty((0, self.config.head_dim))
        cu_seqlens_compressed = None
        seq_to_rank_row = None
        compressed_topk = None
        ratio = max(1, self.compress_ratio)
        if self.compressor is not None and ratio > 1:
            compression_geometry = prepare_cp_compression_geometry(
                hidden_states,
                boundary_hidden,
                cu_seqlens,
                global_start=global_start,
                cp_size=cp_size,
                ratio=ratio,
            )
            cu_seqlens_compressed = compression_geometry.cu_seqlens_compressed
            hidden_compact = compression_geometry.hidden_compact
            group_ids = compression_geometry.compressed_group_ids
            seq_to_rank_row = compression_geometry.seq_to_rank_row
            compact_score = fused_block_fp8_linear(
                _fp32_linear,
                hidden_compact,
                self.compressor.wkv.weight,
                self.compressor.wgate.weight,
            )
            compressed_graph = compressed_compact_graph(
                compact_score,
                self.compressor.ape,
                self.compressor.norm.weight,
                group_ids,
                metadata.cos_sin_cache,
                ratio=ratio,
                head_dim=self.config.head_dim,
                rope_dim=self.config.qk_rope_head_dim,
                eps=self.config.rms_norm_eps,
            )
            if ratio in (4, 128):
                from megatron.lite.model.deepseek_v4.vllm.primitive.attention.cp import (
                    official_compact_compressed_visible,
                )

                if metadata.compressor_metadata is None:
                    raise RuntimeError(
                        "C4/C128 CP requires caller-owned official compressor metadata"
                    )
                compressed_graph = official_compact_compressed_visible(
                    compressed_graph,
                    compact_score,
                    self.compressor.ape,
                    self.compressor.norm.weight,
                    group_ids,
                    metadata.cos_sin_cache,
                    operation=compressor_operation,
                    runtime_metadata=metadata.compressor_metadata,
                    ratio=ratio,
                    head_dim=self.config.head_dim,
                )
                compressed_local = compressed_graph
            else:
                compressed_local = quantized_main_k_visible(compressed_graph)
            if cp_size > 1:
                compressed_rank_major, _ = gather_cp_compressed_rows(
                    compressed_local,
                    seq_to_rank_row,
                    cp_group=cp_group,
                )
            else:
                compressed_rank_major = compressed_local

            width = compressed_width(
                int(psp.max_seqlen_q), ratio, self.config.index_topk
            )
            if self.indexer is not None:
                if indexer_kv_score is None or indexer_weights is None:
                    raise RuntimeError("C4 CP requires indexer projections")
                assert index_q is not None
                compact_index_score = fused_block_fp8_linear(
                    _fp32_linear,
                    hidden_compact.detach(),
                    self.indexer.compressor.wkv.weight,
                    self.indexer.compressor.wgate.weight,
                )
                index_k_local = compressed_compact_graph(
                    compact_index_score,
                    self.indexer.compressor.ape.detach(),
                    self.indexer.compressor.norm.weight.detach(),
                    group_ids,
                    metadata.cos_sin_cache,
                    ratio=ratio,
                    head_dim=self.config.index_head_dim,
                    rope_dim=self.config.qk_rope_head_dim,
                    eps=self.config.rms_norm_eps,
                )
                if cp_size > 1:
                    index_k_rank_major, index_k_seq_major = (
                        gather_cp_compressed_rows(
                            index_k_local,
                            seq_to_rank_row,
                            cp_group=cp_group,
                        )
                    )
                else:
                    index_k_seq_major = torch.index_select(
                        index_k_local,
                        0,
                        seq_to_rank_row.clamp_min(0).long(),
                    )
                compressed_topk = official_indexer_topk(
                    index_q,
                    indexer_weights,
                    index_k_seq_major,
                    positions,
                    metadata.cos_sin_cache,
                    cu_seqlens,
                    cu_seqlens_compressed,
                    global_start=global_start,
                    ratio=ratio,
                    topk=width,
                )
            else:
                compressed_topk = c128_all_visible_topk(
                    positions, width=width, ratio=ratio
                )
        else:
            width = 0

        workspace = torch.cat(
            (boundary_k_visible, kv_visible, compressed_rank_major), dim=0
        ).view(-1, 1, self.config.head_dim)
        indexer_topk_for_loss = compressed_topk
        if compressed_topk is not None:
            # vLLM's BI combine canonicalizes the unordered compressed set in
            # descending logical-index order before appending chronological SWA.
            compressed_topk = torch.sort(
                compressed_topk, dim=-1, descending=True
            ).values
        indices, topk_length, _ = cp_layout_kernels.build_attention_indices(
            cu_seqlens,
            global_start,
            l_local,
            d_window,
            self.config.sliding_window,
            ratio,
            width,
            compressed_topk,
            cu_seqlens_compressed=cu_seqlens_compressed,
            seq_to_rank_row=seq_to_rank_row,
            for_indexer_loss=False,
        )
        if topk_length is None:
            raise RuntimeError("DS4 CP index lowering must return lengths")
        if compressed_topk is not None:
            # MCore's normal selected layout is compact [window | compressed].
            # vLLM FlashMLA consumes compact [compressed | window], so rotate
            # only the valid prefix without changing either selected set.
            window_count = torch.minimum(
                positions + 1,
                torch.tensor(
                    self.config.sliding_window,
                    dtype=positions.dtype,
                    device=positions.device,
                ),
            ).to(torch.int64)
            compressed_count = topk_length.to(torch.int64) - window_count
            columns = torch.arange(indices.shape[-1], device=indices.device).unsqueeze(0)
            source = torch.where(
                columns < compressed_count.unsqueeze(1),
                window_count.unsqueeze(1) + columns,
                columns - compressed_count.unsqueeze(1),
            ).clamp_min(0)
            indices = torch.gather(indices, 1, source)
            indices = torch.where(
                columns < topk_length.to(torch.int64).unsqueeze(1),
                indices,
                torch.full_like(indices, -1),
            )
        indices = indices.unsqueeze(1)
        scale = self.config.head_dim**-0.5
        sink = self.sinks.float().contiguous()
        output_buffer = torch.empty_like(q_visible)

        def visible_attention(q_value, kv_value):
            return sparse_attention(
                q_value,
                kv_value,
                indices,
                sm_scale=scale,
                attn_sink=sink,
                topk_length=topk_length,
                out=output_buffer,
            )

        result = visible_sparse_attention(
            visible_attention,
            q_visible,
            workspace,
            indices,
            topk_length,
            sink,
            softmax_scale=scale,
        )
        result = result[:, : self.config.num_attention_heads, :]
        if self.indexer_loss_coeff:
            assert indexer_topk_for_loss is not None
            assert index_q is not None
            assert indexer_kv_score is not None and indexer_weights is not None
            assert compressor_kv_score is not None and self.compressor is not None
            result = attach_indexer_aux_loss(
                result,
                q,
                index_q,
                indexer_kv_score,
                indexer_weights,
                compressor_kv_score,
                self.indexer.compressor.ape,
                self.indexer.compressor.norm.weight,
                self.compressor.ape,
                self.compressor.norm.weight,
                metadata.positions,
                metadata.cos_sin_cache,
                indexer_topk_for_loss,
                ratio=self.compress_ratio,
                rope_dim=self.config.qk_rope_head_dim,
                eps=self.config.rms_norm_eps,
                softmax_scale=scale,
                loss_coeff=self.indexer_loss_coeff,
            )
        return self._output_projection(
            result,
            positions,
            metadata.cos_sin_cache,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        metadata: AttentionKernelMetadata | None,
    ) -> torch.Tensor:
        if hidden_states.ndim != 2:
            raise ValueError("layer-0 attention requires flat [tokens, hidden]")
        if metadata is None:
            raise NotImplementedError("layer-0 attention requires explicit metadata")
        # FSDP2 mixed precision recursively casts floating forward inputs,
        # including tensors nested in this metadata dataclass. FlashMLA's RoPE
        # kernels require the cache to remain FP32, so restore that boundary
        # after the FSDP pre-forward hook has run.
        if metadata.cos_sin_cache.dtype != torch.float32:
            metadata.cos_sin_cache = metadata.cos_sin_cache.float()
        qr_kv, projection_outputs = self._input_projections(hidden_states)
        compressor_kv_score, indexer_weights, indexer_kv_score = projection_outputs
        qr, kv = qr_kv.split([self.config.q_lora_rank, self.config.head_dim], dim=-1)
        qr, kv = fused_qkv_rms_norm(
            fused_q_kv_rmsnorm,
            qr,
            kv,
            self.q_norm.weight,
            self.kv_norm.weight,
            self.config.rms_norm_eps,
        )
        q = block_fp8_linear(
            self.q_linear, qr, self.wq_b.weight
        ).view(-1, self.config.num_attention_heads, self.config.head_dim)
        index_q = (
            block_fp8_linear(
                self.indexer_q_linear,
                qr,
                self.indexer.wq_b.weight,
            )
            .view(-1, self.config.index_n_heads, self.config.index_head_dim)
            .contiguous()
            if self.indexer is not None
            else None
        )
        return self._forward_training_attention(
            hidden_states,
            metadata,
            q.contiguous(),
            kv,
            compressor_kv_score,
            index_q,
            indexer_weights,
            indexer_kv_score,
        )
