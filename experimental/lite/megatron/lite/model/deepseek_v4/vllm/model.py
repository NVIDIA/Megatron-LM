from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from megatron.lite.model.deepseek_v4.config import DeepseekV4Config
from megatron.lite.model.deepseek_v4.lite.model import (
    DeepseekV4CSAAttention as LiteDeepseekV4CSAAttention,
    DeepseekV4Layer as LiteDeepseekV4Layer,
    DeepseekV4Model as LiteDeepseekV4Model,
)
from megatron.lite.model.deepseek_v4.vllm.moe import DeepseekV4MoE, MoEKernelMetadata
from megatron.lite.model.deepseek_v4.vllm.runtime_metadata import AttentionKernelMetadata
from megatron.lite.model.deepseek_v4.vllm.primitive import (
    attention_core,
    attach_indexer_aux_loss,
    block_fp8_linear,
    fused_block_fp8_linear,
    fused_qkv_rms_norm,
    mhc_head,
    mhc_post,
    mhc_pre_broadcast,
    o_projection,
    rms_norm,
    visible_linear,
    visible_sparse_attention,
)
from megatron.lite.primitive.modules.attention.hca import HyperConnection
from megatron.lite.primitive.modules.attention.csa import CompressedSparseAttention
from megatron.lite.primitive.kernels.vllm_ds4 import (
    DS4KVInsertAdapter,
    FlashMLAAdapter,
    FusedQKVRMSNormAdapter,
    KVCacheLayout,
    MHCKernel,
    MHCTileLangAdapter,
    OProjectionAdapter,
)
from megatron.lite.primitive.parallel import ParallelState
from megatron.lite.primitive.parallel.mhc import (
    fold_mhc_hidden_for_pipeline,
    unfold_mhc_hidden_from_pipeline,
)
from megatron.lite.primitive.quantization.deployment_block_fp8 import (
    DeploymentBlockFP8Adapter,
    DeploymentFusedBlockFP8Adapter,
)


def _rollout_selected_log_probs(
    logits: torch.Tensor, labels: torch.Tensor, temperature: float
) -> torch.Tensor:
    """Evaluate selected tokens with the same reduction used by rollout."""
    from vllm.v1.worker.gpu.sample.logprob import compute_token_logprobs

    rollout_logits = logits if temperature == 1.0 else logits.float() / temperature
    return compute_token_logprobs(
        rollout_logits, labels.unsqueeze(-1)
    ).squeeze(-1)


def _differentiable_log_probs_and_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    *,
    calculate_entropy: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    scaled_logits = logits.float()
    if temperature != 1.0:
        scaled_logits = scaled_logits / temperature
    log_probs = F.log_softmax(scaled_logits, dim=-1)
    selected = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    entropy = None
    if calculate_entropy:
        entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
    return selected, entropy


def _aligned_selected_log_probs(
    hidden_states: torch.Tensor,
    lm_head: nn.Module,
    labels: torch.Tensor,
    temperature: float,
    chunk_size: int,
    *,
    calculate_entropy: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Use the rollout value and the same BF16 LM-head for its training VJP.

    vLLM's selected-logprob Triton kernel intentionally has no autograd.  The
    visible value is evaluated by that exact kernel.  The differentiable path
    is derived from the very same BF16 logits.  Non-reentrant checkpointing
    drops each full-vocabulary chunk and recomputes it during backward.
    """
    if chunk_size <= 0:
        raise ValueError("logprob_chunk_size must be positive")

    selected_chunks = []
    entropy_chunks = []
    grad_enabled = torch.is_grad_enabled()
    # Keep both forward and backward full-vocabulary storage bounded.  Training
    # recomputes one chunk at a time, so the only difference from a one-shot
    # head is floating-point accumulation order for the shared weight gradient.
    for start in range(0, hidden_states.shape[0], chunk_size):
        stop = min(start + chunk_size, hidden_states.shape[0])
        chunk_labels = labels[start:stop]

        def chunk_forward(
            chunk_hidden: torch.Tensor, chunk_labels=chunk_labels
        ):
            logits = lm_head(chunk_hidden)
            differentiable, entropy = _differentiable_log_probs_and_entropy(
                logits,
                chunk_labels,
                temperature,
                calculate_entropy=calculate_entropy,
            )
            with torch.no_grad():
                visible = _rollout_selected_log_probs(
                    logits, chunk_labels, temperature
                )
            selected = visible + (differentiable - differentiable.detach())
            if calculate_entropy:
                assert entropy is not None
                return selected, entropy
            return selected

        chunk_hidden = hidden_states[start:stop]
        if grad_enabled:
            chunk_result = checkpoint(
                chunk_forward, chunk_hidden, use_reentrant=False
            )
        else:
            with torch.no_grad():
                logits = lm_head(chunk_hidden)
                selected = _rollout_selected_log_probs(
                    logits, chunk_labels, temperature
                )
                entropy = None
                if calculate_entropy:
                    _, entropy = _differentiable_log_probs_and_entropy(
                        logits,
                        chunk_labels,
                        temperature,
                        calculate_entropy=True,
                    )
                chunk_result = (selected, entropy) if calculate_entropy else selected

        if calculate_entropy:
            selected, entropy = chunk_result
            selected_chunks.append(selected)
            entropy_chunks.append(entropy)
        else:
            selected_chunks.append(chunk_result)

    selected = torch.cat(selected_chunks, dim=0)
    entropy = torch.cat(entropy_chunks, dim=0) if calculate_entropy else None
    return selected, entropy


def _default_attention_ops() -> SimpleNamespace:
    return SimpleNamespace(
        fused_linear=DeploymentFusedBlockFP8Adapter(cache_weight=True),
        q_linear=DeploymentBlockFP8Adapter(cache_weight=True),
        indexer_q_linear=DeploymentBlockFP8Adapter(cache_weight=True),
        bf16_linear=lambda value, weight: __import__(
            "vllm.model_executor.layers.batch_invariant",
            fromlist=["linear_batch_invariant"],
        ).linear_batch_invariant(value, weight),
        fp32_linear=lambda value, *weights: torch.mm(
            value.contiguous(),
            torch.cat(weights, dim=0).T,
            out_dtype=torch.float32,
        ),
        norm=FusedQKVRMSNormAdapter(),
        kv_insert=DS4KVInsertAdapter(KVCacheLayout.FP8_DS_MLA),
        flash=FlashMLAAdapter(),
        o_project=OProjectionAdapter(),
    )


class _AttentionState(CompressedSparseAttention):
    def __init__(
        self,
        config: DeepseekV4Config,
        *,
        ps=None,
        layer_idx: int,
        indexer_loss_coeff: float = 0.0,
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
        self.adapters = _default_attention_ops()
        self._projection_streams: list[torch.cuda.Stream] | None = None
        self._projection_events: list[torch.cuda.Event] | None = None

    def _input_projections(self, hidden_states: torch.Tensor):
        def fused_projection():
            return fused_block_fp8_linear(
                self.adapters.fused_linear,
                hidden_states,
                self.wq_a.weight,
                self.wkv.weight,
            )

        aux_fns: list[Callable[[], torch.Tensor] | None] = [None, None, None]
        if self.compressor is not None:
            aux_fns[0] = lambda: fused_block_fp8_linear(
                self.adapters.fp32_linear,
                hidden_states,
                self.compressor.wkv.weight,
                self.compressor.wgate.weight,
            )
        if self.indexer is not None:
            aux_fns[1] = lambda: visible_linear(
                self.adapters.bf16_linear,
                hidden_states,
                self.indexer.weights_proj.weight,
            )
            aux_fns[2] = lambda: fused_block_fp8_linear(
                self.adapters.fp32_linear,
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

    def _forward_native_cp(
        self, hidden_states: torch.Tensor, metadata: AttentionKernelMetadata
    ) -> torch.Tensor:
        if self.ps is None or self.ps.cp_size <= 1 or self.ps.cp_group is None:
            raise RuntimeError("native DS4 CP requires a model-owned CP group")
        if metadata.cp_packed_seq_params is None or metadata.cp_positions is None:
            raise RuntimeError("native DS4 CP requires packed sequence geometry")
        if self.indexer_loss_coeff:
            raise NotImplementedError(
                "native DS4 CP indexer auxiliary loss is not implemented"
            )

        from megatron.core.transformer.experimental_attention_variant.csa_utils import (
            cp_layout_kernels,
            cp_utils,
        )
        from megatron.lite.model.deepseek_v4.vllm.native_cp import (
            c128_all_visible_topk,
            compressed_width,
            official_indexer_topk,
            official_local_qk_visible,
            quantized_main_k_visible,
        )
        from megatron.lite.model.deepseek_v4.vllm.primitive.attention import (
            compressed_compact_graph,
        )
        from megatron.lite.primitive.modules.attention.cp_geometry import (
            gather_cp_compressed_rows,
            prepare_cp_compression_geometry,
        )

        psp = metadata.cp_packed_seq_params
        cu_seqlens = (
            psp.cu_seqlens_q_padded
            if psp.cu_seqlens_q_padded is not None
            else psp.cu_seqlens_q
        )
        if cu_seqlens is None:
            raise RuntimeError("native DS4 CP requires cu_seqlens_q")
        l_local = hidden_states.shape[0]
        global_start = self.ps.cp_rank * l_local
        positions = metadata.cp_positions.reshape(-1).to(torch.int64)
        if positions.numel() != l_local:
            raise RuntimeError("native DS4 CP position rows do not match local tokens")
        if metadata.cos_sin_cache.dtype != torch.float32:
            metadata.cos_sin_cache = metadata.cos_sin_cache.float()

        qr_kv, projection_outputs = self._input_projections(hidden_states)
        compressor_kv_score, indexer_weights, indexer_kv_score = projection_outputs
        qr, kv = qr_kv.split([self.config.q_lora_rank, self.config.head_dim], dim=-1)
        qr, kv = fused_qkv_rms_norm(
            self.adapters.norm,
            qr,
            kv,
            self.q_norm.weight,
            self.kv_norm.weight,
            self.config.rms_norm_eps,
        )
        q = block_fp8_linear(self.adapters.q_linear, qr, self.wq_b.weight).view(
            -1, self.config.num_attention_heads, self.config.head_dim
        ).contiguous()
        q_visible, kv_visible = official_local_qk_visible(
            q,
            kv,
            positions,
            metadata.cos_sin_cache,
            self.adapters.kv_insert,
            eps=self.config.rms_norm_eps,
            rope_dim=self.config.qk_rope_head_dim,
            padded_heads=self.config.num_attention_heads,
        )

        boundary_hidden = cp_utils.exchange_cp_boundary_hidden(
            hidden_states,
            self.compress_ratio,
            self.config.sliding_window,
            self.ps.cp_group,
        )
        d_window = boundary_hidden.shape[0]
        boundary_qr_kv = fused_block_fp8_linear(
            self.adapters.fused_linear,
            boundary_hidden,
            self.wq_a.weight,
            self.wkv.weight,
        )
        boundary_qr, boundary_kv = boundary_qr_kv.split(
            [self.config.q_lora_rank, self.config.head_dim], dim=-1
        )
        _, boundary_kv = fused_qkv_rms_norm(
            self.adapters.norm,
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
            self.adapters.kv_insert,
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
                cp_size=self.ps.cp_size,
                ratio=ratio,
            )
            cu_seqlens_compressed = compression_geometry.cu_seqlens_compressed
            hidden_compact = compression_geometry.hidden_compact
            group_ids = compression_geometry.compressed_group_ids
            seq_to_rank_row = compression_geometry.seq_to_rank_row
            compact_score = fused_block_fp8_linear(
                self.adapters.fp32_linear,
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
                from megatron.lite.model.deepseek_v4.vllm.native_cp import (
                    official_compact_compressed_visible,
                )

                if (
                    metadata.cp_compressor_operation is None
                    or metadata.cp_compressor_metadata is None
                ):
                    raise RuntimeError(
                        "C4/C128 native CP requires caller-owned official compressor metadata"
                    )
                compressed_graph = official_compact_compressed_visible(
                    compressed_graph,
                    compact_score,
                    self.compressor.ape,
                    self.compressor.norm.weight,
                    group_ids,
                    metadata.cos_sin_cache,
                    operation=metadata.cp_compressor_operation,
                    runtime_metadata=metadata.cp_compressor_metadata,
                    ratio=ratio,
                    head_dim=self.config.head_dim,
                )
                compressed_local = compressed_graph
            else:
                compressed_local = quantized_main_k_visible(compressed_graph)
            compressed_rank_major, _ = gather_cp_compressed_rows(
                compressed_local,
                seq_to_rank_row,
                cp_group=self.ps.cp_group,
            )

            width = compressed_width(
                int(psp.max_seqlen_q), ratio, self.config.index_topk
            )
            if self.indexer is not None:
                if indexer_kv_score is None or indexer_weights is None:
                    raise RuntimeError("C4 native CP requires indexer projections")
                index_q = block_fp8_linear(
                    self.adapters.indexer_q_linear,
                    qr,
                    self.indexer.wq_b.weight,
                ).view(-1, self.config.index_n_heads, self.config.index_head_dim)
                compact_index_score = fused_block_fp8_linear(
                    self.adapters.fp32_linear,
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
                index_k_rank_major, index_k_seq_major = gather_cp_compressed_rows(
                    index_k_local,
                    seq_to_rank_row,
                    cp_group=self.ps.cp_group,
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
            raise RuntimeError("native DS4 CP index lowering must return lengths")
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
            return self.adapters.flash.sparse(
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
        heads_per_group = self.config.num_attention_heads // self.config.o_groups
        nope_dim = self.config.head_dim - self.config.qk_rope_head_dim
        return o_projection(
            lambda o, wa, wb: self.adapters.o_project(
                o,
                positions,
                metadata.cos_sin_cache,
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
            cos_sin_cache=metadata.cos_sin_cache,
            n_groups=self.config.o_groups,
            heads_per_group=heads_per_group,
            nope_dim=nope_dim,
            rope_dim=self.config.qk_rope_head_dim,
            o_lora_rank=self.config.o_lora_rank,
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
        if self.ps is not None and self.ps.cp_size > 1:
            return self._forward_native_cp(hidden_states, metadata)
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
            self.adapters.norm,
            qr,
            kv,
            self.q_norm.weight,
            self.kv_norm.weight,
            self.config.rms_norm_eps,
        )
        q = (
            block_fp8_linear(self.adapters.q_linear, qr, self.wq_b.weight)
            .view(-1, self.config.num_attention_heads, self.config.head_dim)
            .clone(memory_format=torch.contiguous_format)
        )
        index_q = (
            block_fp8_linear(
                self.adapters.indexer_q_linear,
                qr,
                self.indexer.wq_b.weight,
            )
            .view(-1, self.config.index_n_heads, self.config.index_head_dim)
            .contiguous()
            if self.indexer is not None
            else None
        )
        indexer_topk = None
        if self.compressor is not None:
            if metadata.compressor_operation is None:
                raise NotImplementedError(
                    f"layer {self.layer_idx} compressor requires explicit runtime metadata"
                )
            assert compressor_kv_score is not None
            metadata.compressor_operation(
                kv_score=compressor_kv_score.detach(),
                positions=metadata.positions,
                ape=self.compressor.ape.detach().float().contiguous(),
                norm_weight=self.compressor.norm.weight.detach(),
                compress_ratio=self.compress_ratio,
                head_dim=self.config.head_dim,
                metadata=metadata.compressor_metadata,
            )
        if self.indexer is not None:
            if metadata.indexer_operation is None:
                raise NotImplementedError(
                    f"layer {self.layer_idx} indexer requires explicit runtime metadata"
                )
            assert indexer_kv_score is not None and indexer_weights is not None
            metadata.compressor_operation(
                kv_score=indexer_kv_score.detach(),
                positions=metadata.positions,
                ape=self.indexer.compressor.ape.detach().float().contiguous(),
                norm_weight=self.indexer.compressor.norm.weight.detach(),
                compress_ratio=self.compress_ratio,
                head_dim=self.config.index_head_dim,
                metadata=metadata.indexer_metadata,
            )
            assert index_q is not None
            index_result = metadata.indexer_operation(
                qr=qr.detach(),
                index_q=index_q.detach(),
                index_weights=indexer_weights.detach(),
                positions=metadata.positions,
                compress_ratio=self.compress_ratio,
                topk=self.config.index_topk,
                metadata=metadata.indexer_metadata,
            )
            metadata.indices = index_result
            indexer_topk = index_result

        insert_cache = metadata.swa_cache
        if insert_cache.dtype == torch.uint8 and insert_cache.ndim == 3:
            insert_cache = insert_cache.view(insert_cache.shape[0], -1)
        scale = self.config.head_dim**-0.5
        sink = self.sinks.float().contiguous()
        if metadata.kv_workspace is None:
            raise NotImplementedError("FlashMLA prefill requires kv_workspace")

        def visible_attention(q_pre, kv_pre):
            q_visible = self.adapters.kv_insert(
                q_pre,
                kv_pre,
                insert_cache,
                metadata.slot_mapping,
                metadata.positions,
                metadata.cos_sin_cache,
                eps=self.config.rms_norm_eps,
                block_size=metadata.block_size,
                padded_heads=self.config.num_attention_heads,
            ).contiguous()
            if metadata.prepare_flash is not None:
                metadata.prepare_flash()
            flash_result = self.adapters.flash.sparse(
                q_visible,
                metadata.kv_workspace,
                indices=metadata.indices,
                sm_scale=scale,
                attn_sink=sink,
                topk_length=metadata.topk_length,
                out=metadata.output,
            )
            return (
                flash_result[0],
                flash_result[2],
                q_visible,
                metadata.kv_workspace,
                metadata.indices,
                metadata.topk_length,
            )

        result = attention_core(
            visible_attention,
            q,
            kv,
            metadata.kv_workspace,
            metadata.indices,
            metadata.topk_length,
            sink,
            (
                metadata.kv_workspace_slot_mapping
                if metadata.kv_workspace_slot_mapping is not None
                else metadata.slot_mapping
            ),
            metadata.positions,
            metadata.cos_sin_cache,
            softmax_scale=scale,
            eps=self.config.rms_norm_eps,
            rope_dim=self.config.qk_rope_head_dim,
            compressor_kv_score=compressor_kv_score,
            compressor_ape=(self.compressor.ape if self.compressor is not None else None),
            compressor_norm=(
                self.compressor.norm.weight if self.compressor is not None else None
            ),
            compressor_ratio=self.compress_ratio,
            compressor_workspace_slots=metadata.compressor_workspace_slot_mapping,
            query_start_loc=metadata.query_start_loc,
        )
        if self.indexer is not None and torch.is_grad_enabled():
            assert indexer_topk is not None
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
                indexer_topk,
                ratio=self.compress_ratio,
                rope_dim=self.config.qk_rope_head_dim,
                eps=self.config.rms_norm_eps,
                softmax_scale=scale,
                loss_coeff=self.indexer_loss_coeff,
            )
        result = result.reshape(hidden_states.shape[0], -1, self.config.head_dim)
        result = result[:, : self.config.num_attention_heads, :]

        heads_per_group = self.config.num_attention_heads // self.config.o_groups
        nope_dim = self.config.head_dim - self.config.qk_rope_head_dim
        output = o_projection(
            lambda o, wa, wb: self.adapters.o_project(
                o,
                metadata.positions,
                metadata.cos_sin_cache,
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
            positions=metadata.positions,
            cos_sin_cache=metadata.cos_sin_cache,
            n_groups=self.config.o_groups,
            heads_per_group=heads_per_group,
            nope_dim=nope_dim,
            rope_dim=self.config.qk_rope_head_dim,
            o_lora_rank=self.config.o_lora_rank,
        )
        return output


class _VLLMCSAAttention(LiteDeepseekV4CSAAttention):
    def __init__(
        self,
        config: DeepseekV4Config,
        *,
        ps: ParallelState,
        layer_idx: int,
        indexer_loss_coeff: float,
    ):
        self._indexer_loss_coeff = indexer_loss_coeff
        super().__init__(config, layer_idx=layer_idx, ps=ps)

    def _build_attention(
        self, config: DeepseekV4Config, *, layer_idx: int, ps: ParallelState
    ) -> nn.Module:
        return _AttentionState(
            config,
            ps=ps,
            layer_idx=layer_idx,
            indexer_loss_coeff=self._indexer_loss_coeff,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        metadata: AttentionKernelMetadata | dict[int, AttentionKernelMetadata] | None,
    ) -> torch.Tensor:
        return self.self_attn(hidden_states, metadata=metadata)


class DeepseekV4Layer(LiteDeepseekV4Layer):
    def __init__(
        self,
        config: DeepseekV4Config,
        ps=None,
        layer_idx: int = 0,
        *,
        use_deepep: bool = False,
        indexer_loss_coeff: float = 0.0,
    ):
        self.config = config
        self._vllm_indexer_loss_coeff = indexer_loss_coeff
        super().__init__(
            config,
            ps or ParallelState(),
            layer_idx,
            use_deepep=use_deepep,
        )

    def _build_attention(
        self, config: DeepseekV4Config, *, layer_idx: int, ps: ParallelState
    ) -> nn.Module:
        return _VLLMCSAAttention(
            config,
            ps=ps,
            layer_idx=layer_idx,
            indexer_loss_coeff=self._vllm_indexer_loss_coeff,
        )

    def _build_moe(
        self,
        config: DeepseekV4Config,
        ps: ParallelState,
        *,
        layer_idx: int,
        use_deepep: bool,
    ) -> nn.Module:
        return DeepseekV4MoE(
            config,
            ps,
            layer_idx=layer_idx,
            use_deepep=use_deepep,
        )

    def _mhc_pre(
        self,
        hidden_states: torch.Tensor,
        hc: HyperConnection,
        norm_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fn = hc.fn.float().contiguous()
        scale = hc.scale.float().contiguous()
        base = hc.base.float().contiguous()
        broadcast = hidden_states.ndim == 2
        adapter = MHCTileLangAdapter(
            MHCKernel.PRE_BROADCAST if broadcast else MHCKernel.PRE
        )

        def visible_pre(hidden, fn_, scale_, base_, norm_weight_):
            common = (
                hidden,
                fn_,
                scale_,
                base_,
                self.config.rms_norm_eps,
                self.config.hc_eps,
                self.config.hc_eps,
                2.0,
                self.config.hc_sinkhorn_iters,
            )
            kwargs = {
                "norm_weight": norm_weight_,
                "norm_eps": self.config.rms_norm_eps,
            }
            if broadcast:
                kwargs["fn_broadcast"] = (
                    fn_.view(-1, self.config.hc_mult, self.config.hidden_size)
                    .sum(dim=1)
                    .contiguous()
                )
                return adapter(*common, **kwargs)
            return (hidden, *adapter(*common, **kwargs))

        return mhc_pre_broadcast(
            visible_pre,
            hidden_states,
            fn,
            scale,
            base,
            norm_weight,
            mult=self.config.hc_mult,
            iters=self.config.hc_sinkhorn_iters,
            eps=self.config.hc_eps,
            norm_eps=self.config.rms_norm_eps,
        )

    @staticmethod
    def _mhc_post(
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        post_mix: torch.Tensor,
        res_mix: torch.Tensor,
    ) -> torch.Tensor:
        adapter = MHCTileLangAdapter(MHCKernel.POST)
        return mhc_post(
            lambda *args: adapter(*args),
            hidden_states,
            residual,
            post_mix,
            res_mix,
        )

    def _attention_block(
        self,
        x: torch.Tensor,
        *,
        position_ids: torch.Tensor,
        packed_seq_params: Any,
        metadata: AttentionKernelMetadata | None = None,
    ) -> torch.Tensor:
        del position_ids, packed_seq_params
        residual, post_mix, res_mix, hidden_states = self._mhc_pre(
            x, self.attn_hc, self.input_layernorm.weight
        )
        hidden_states = self.self_attn(hidden_states, metadata=metadata)
        return self._mhc_post(hidden_states, residual, post_mix, res_mix)

    def _mlp_block(
        self,
        x: torch.Tensor,
        *,
        input_ids: torch.Tensor | None,
        metadata: MoEKernelMetadata | None = None,
    ) -> torch.Tensor:
        residual, post_mix, res_mix, hidden_states = self._mhc_pre(
            x, self.ffn_hc, self.post_attention_layernorm.weight
        )
        hidden_states = self.mlp(
            hidden_states, input_ids=input_ids, metadata=metadata
        )
        return self._mhc_post(hidden_states, residual, post_mix, res_mix)


class DeepseekV4Model(LiteDeepseekV4Model):
    def __init__(
        self,
        config: DeepseekV4Config,
        train_config=None,
        ps=None,
        *,
        use_deepep: bool = False,
        indexer_loss_coeff: float = 0.0,
        logprob_chunk_size: int = 8192,
    ):
        ps = ps or ParallelState()
        self._vllm_indexer_loss_coeff = indexer_loss_coeff
        if logprob_chunk_size <= 0:
            raise ValueError("logprob_chunk_size must be positive")
        self._logprob_chunk_size = int(logprob_chunk_size)
        train_config = train_config or SimpleNamespace(vpp=1, fp8=False)
        super().__init__(
            config,
            train_config,
            ps,
            mtp_enable=False,
            use_deepep=use_deepep,
        )
        # The release keeps deployment GEMM masters in BF16 while the mHC
        # coefficients, sparse-attention sinks/APE, and router correction bias
        # remain FP32.  Lite supplies the parameter containers; this path only
        # applies the vLLM-visible dtype boundary before checkpoint loading.
        self.to(torch.bfloat16)
        fp32_suffixes = (
            ".attn_hc.fn",
            ".attn_hc.base",
            ".attn_hc.scale",
            ".ffn_hc.fn",
            ".ffn_hc.base",
            ".ffn_hc.scale",
            ".hc_fn",
            ".hc_base",
            ".hc_scale",
            ".sinks",
            ".ape",
            ".expert_bias",
        )
        for name, parameter in self.named_parameters():
            if name.endswith(fp32_suffixes):
                parameter.data = parameter.data.float()
        self._shared_projection_streams: list[torch.cuda.Stream] | None = None

    def _build_layer(
        self,
        config: DeepseekV4Config,
        ps: ParallelState,
        layer_idx: int,
        *,
        use_deepep: bool,
    ) -> nn.Module:
        return DeepseekV4Layer(
            config,
            ps,
            layer_idx=layer_idx,
            use_deepep=use_deepep,
            indexer_loss_coeff=self._vllm_indexer_loss_coeff,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        attention_metadata: AttentionKernelMetadata | None = None,
        moe_metadata: MoEKernelMetadata | None = None,
        labels: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
        temperature: float | torch.Tensor = 1.0,
        calculate_entropy: bool = False,
        **unused,
    ) -> dict[str, torch.Tensor]:
        del unused
        pipeline_streams = False
        if hidden_states is None:
            if not self.pre_process:
                hidden_states = self._input_tensor
                if hidden_states is not None:
                    hidden_states = unfold_mhc_hidden_from_pipeline(
                        hidden_states, hc_mult=self.config.hc_mult
                    ).reshape(-1, self.config.hc_mult, self.config.hidden_size)
                    pipeline_streams = True
            elif input_ids is not None:
                assert self.embed_tokens is not None
                hidden_states = self.embed_tokens.embedding(input_ids)
        if hidden_states is None:
            raise ValueError("input_ids or hidden_states is required.")
        if not hidden_states.is_cuda:
            raise RuntimeError("DeepSeek V4 vLLM training requires CUDA tensors")
        if hidden_states.ndim != 2 and not pipeline_streams:
            hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        if self._shared_projection_streams is None:
            self._shared_projection_streams = [torch.cuda.Stream() for _ in range(3)]
        for layer in self.layers.values():
            layer.self_attn.self_attn._projection_streams = (
                self._shared_projection_streams
            )
        for local_idx, layer_idx in enumerate(self.layer_indices):
            layer = self.layers[str(local_idx)]
            layer_attention_metadata = (
                attention_metadata.get(layer_idx)
                if isinstance(attention_metadata, dict)
                else attention_metadata
            )
            layer_moe_metadata = (
                moe_metadata.get(layer_idx)
                if isinstance(moe_metadata, dict)
                else moe_metadata
            )
            hidden_states = layer(
                hidden_states,
                position_ids=position_ids,
                attention_metadata=layer_attention_metadata,
                moe_metadata=layer_moe_metadata,
                input_ids=input_ids,
            )
        if not self.post_process:
            # Pipeline P2P is [S, B, hc_mult*H].  Packed DS4 uses B=1.
            return {
                "hidden_states": fold_mhc_hidden_for_pipeline(
                    hidden_states.unsqueeze(1)
                )
            }
        if self.norm is None or self.hc_head is None or self.lm_head is None:
            raise RuntimeError("final pipeline stage is missing the output head")
        head_adapter = MHCTileLangAdapter(MHCKernel.HEAD)
        hidden_states = mhc_head(
            lambda *args: head_adapter(
                *args, self.config.rms_norm_eps, self.config.hc_eps
            ),
            hidden_states,
            self.hc_head.hc_fn.float().contiguous(),
            self.hc_head.hc_scale.float().contiguous(),
            self.hc_head.hc_base.float().contiguous(),
            eps=self.config.hc_eps,
        )
        from vllm.model_executor.layers.batch_invariant import (
            rms_norm_batch_invariant,
        )

        hidden_states = rms_norm(
            rms_norm_batch_invariant,
            hidden_states,
            self.norm.weight,
            self.config.rms_norm_eps,
        )
        result = {"hidden_states": hidden_states}
        if labels is not None:
            temperature_value = float(
                temperature.detach().float().item()
                if isinstance(temperature, torch.Tensor)
                else temperature
            )
            if temperature_value <= 0:
                raise ValueError("temperature must be positive")
            flat_labels = labels.reshape(-1).long()
            if flat_labels.numel() != hidden_states.shape[0]:
                raise ValueError("labels must contain one target per visible token")
            selected_log_probs, entropy = _aligned_selected_log_probs(
                hidden_states,
                self.lm_head,
                flat_labels,
                temperature_value,
                self._logprob_chunk_size,
                calculate_entropy=calculate_entropy,
            )
            token_loss = -selected_log_probs
            result["log_probs"] = selected_log_probs
            if calculate_entropy:
                assert entropy is not None
                result["entropy"] = entropy
            mask = (
                torch.ones_like(token_loss)
                if loss_mask is None
                else loss_mask.reshape(-1).to(token_loss.dtype)
            )
            denominator = mask.sum()
            if not bool(denominator > 0):
                raise ValueError("loss_mask must select at least one token")
            result["loss"] = (token_loss * mask).sum() / denominator
        return result
