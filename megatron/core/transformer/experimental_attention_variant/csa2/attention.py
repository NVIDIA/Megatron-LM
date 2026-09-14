# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""DeepSeek-V4.1 CSA2 attention modules.

:class:`CSA2Attention` is the *core* attention: it receives the rotated multi-head query,
the rotated single-head window key, the normalised layer input ``x`` and the normalised
low-rank query ``qr`` from the enclosing self-attention module, and returns the attention
output before the grouped output projection. Depending on the layer role it

* runs the compressor and publishes compressed KV + indexer keys (``full``),
* runs the indexer and publishes top-k (``full`` / ``reindex``),
* reads compressed KV and / or top-k from the shared state (``reindex`` / ``reuse``),
* attends over ``[window keys | compressed keys]`` with a learned per-head sink.

:class:`DSv41SelfAttention` reuses the merged DeepSeek-V4 projections (low-rank query,
single-head KV, grouped low-rank output) from ``deepseek_v4_hybrid_attention.py`` and only
changes what V4.1 changes: every compressing layer (ratio >= 1, including ratio 1) rotates
with the compressed RoPE base and YaRN.

Milestone M0 scope: framework-free reference path, SBHD layout, context parallelism 1.
Packed (THD) sequences and CP arrive with M1 (see ``docs/dsv41/DESIGN.md``).
"""

from dataclasses import dataclass
from typing import List, Optional, Union

import torch
import torch.nn as nn

from megatron.core.models.common.embeddings import YarnRotaryEmbedding, apply_rotary_pos_emb
from megatron.core.models.common.embeddings.rope_utils import _apply_rotary_pos_emb_bshd
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant.csa2 import thd
from megatron.core.transformer.experimental_attention_variant.csa2.reference import (
    compressed_visible_counts,
    concat_window_and_compressed_indices,
    sliding_window_indices,
    sparse_attention_with_sink,
)
from megatron.core.transformer.experimental_attention_variant.csa2.roles import (
    CSA2LayerPlan,
    CSA2Plan,
    model_layer_id_from_layer_number,
    model_layer_ratios_from_pattern_ratios,
    resolve_csa2_plan,
)
from megatron.core.transformer.experimental_attention_variant.csa2.state import (
    CompressedKVRecord,
    DSv41SharedState,
    SharedStateSlot,
)
from megatron.core.transformer.experimental_attention_variant.csa_utils.fused_sparse_attention import (  # pylint: disable=line-too-long
    csa_sparse_attn,
)
from megatron.core.transformer.experimental_attention_variant.deepseek_v4_hybrid_attention import (
    DSv4HybridSelfAttention,
    DSv4HybridSelfAttentionSubmodules,
)
from megatron.core.transformer.module import MegatronModule, mark_keep_in_fp32
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import MLATransformerConfig, TransformerConfig


def get_csa2_plan(config: TransformerConfig) -> CSA2Plan:
    """Resolve the CSA2 roles of every model layer from a validated config."""
    pattern_ratios = list(config.csa_compress_ratios[: config.num_layers])
    return resolve_csa2_plan(
        model_layer_ratios_from_pattern_ratios(pattern_ratios),
        config.csa2_kv_source_layers,
        config.csa2_index_source_layers,
        config.csa2_candidate_source_layer,
        config.csa2_candidate_topk_blocks,
        config.csa2_candidate_block_size,
    )


@dataclass
class CSA2AttentionSubmodules:
    """Submodule specs for :class:`CSA2Attention`."""

    compressor: Union[ModuleSpec, type] = None
    indexer: Union[ModuleSpec, type] = None


class CSA2Attention(MegatronModule):
    """Core attention of one DeepSeek-V4.1 layer (window + shared compressed positions)."""

    def __init__(
        self,
        config: TransformerConfig,
        submodules: CSA2AttentionSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: Optional[float] = None,
        softmax_scale: Optional[float] = None,
        k_channels: Optional[int] = None,
        v_channels: Optional[int] = None,
        cp_comm_type: Optional[str] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        rotary_pos_emb: nn.Module = None,
        compress_ratio: Optional[int] = None,
        is_mtp_layer: bool = False,
        name: str | None = None,
    ) -> None:
        super().__init__(config=config)
        if is_mtp_layer:
            raise NotImplementedError("CSA2Attention does not support MTP layers")
        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
        self.pg_collection = pg_collection
        self.layer_number = layer_number
        self.model_layer_id = model_layer_id_from_layer_number(layer_number)

        plan = get_csa2_plan(config)
        self.plan: CSA2LayerPlan = plan[self.model_layer_id]
        if compress_ratio is not None and compress_ratio != self.plan.compress_ratio:
            raise ValueError(
                f"layer_number {layer_number}: compress_ratio {compress_ratio} disagrees with "
                f"the CSA2 plan ({self.plan.compress_ratio}) for model layer {self.model_layer_id}"
            )

        self.window_size = config.csa_window_size
        self.n_heads = config.num_attention_heads
        self.head_dim = config.v_head_dim
        self.rope_dim = config.qk_pos_emb_head_dim
        self.softmax_scale = softmax_scale if softmax_scale is not None else self.head_dim**-0.5
        self.rotary_pos_emb = rotary_pos_emb

        # Learned per-head sink logit, fp32 in the reference checkpoint.
        self.attn_sink = mark_keep_in_fp32(
            nn.Parameter(torch.zeros(self.n_heads, dtype=torch.float32))
        )

        self.use_fused = getattr(config, "csa2_sparse_attention_impl", "reference") == "fused"
        if self.use_fused:
            self._validate_fused_geometry(config)

        self.compressor = None
        if self.plan.runs_compressor:
            self.compressor = build_module(
                submodules.compressor,
                config=config,
                compress_ratio=self.plan.compress_ratio,
                head_dim=self.head_dim,
                name=(name + ".compressor") if name is not None else None,
            )
        self.indexer = None
        if self.plan.runs_indexer:
            self.indexer = build_module(
                submodules.indexer,
                config=config,
                plan=self.plan,
                candidate_topk_blocks=plan.candidate_topk_blocks,
                candidate_block_size=plan.candidate_block_size,
                name=(name + ".indexer") if name is not None else None,
            )

        # Set by the enclosing single-pass mHC wrapper; holds the per-microbatch state while
        # the wrapper runs its inner layer.
        self.shared_state_slot: Optional[SharedStateSlot] = None

    # ---- helpers ----------------------------------------------------------------------------

    # Head geometry the FlashMLA sparse forward / cuDNN DSA backward pair is built for
    # (DeepSeek-V3.2 / V4 / V4.1 attention: 64 or 128 heads, latent 512 + RoPE 64).
    _FUSED_HEADS = (64, 128)
    _FUSED_V_HEAD_DIM = 512
    _FUSED_ROPE_DIM = 64

    def _validate_fused_geometry(self, config: TransformerConfig) -> None:
        """Reject configurations the fused kernels cannot execute before the first forward."""
        problems = []
        if self.n_heads not in self._FUSED_HEADS:
            problems.append(f"num_attention_heads {self.n_heads} (supported: {self._FUSED_HEADS})")
        if self.head_dim != self._FUSED_V_HEAD_DIM:
            problems.append(f"v_head_dim {self.head_dim} (required: {self._FUSED_V_HEAD_DIM})")
        if self.rope_dim != self._FUSED_ROPE_DIM:
            problems.append(
                f"qk_pos_emb_head_dim {self.rope_dim} (required: {self._FUSED_ROPE_DIM})"
            )
        if not config.bf16:
            problems.append("bf16=False (the fused kernels run in bf16)")
        if problems:
            raise ValueError(
                "csa2_sparse_attention_impl='fused' does not support this attention geometry: "
                + "; ".join(problems)
                + ". Use csa2_sparse_attention_impl='reference' for other shapes."
            )

    @staticmethod
    def _empty_rows_like(source: torch.Tensor, width: int) -> torch.Tensor:
        """``[0, width]`` tensor that stays attached to ``source``'s autograd graph."""
        return source.new_zeros((0, width)) + source.reshape(-1)[:0].sum()

    def _shared_state(self) -> DSv41SharedState:
        if self.shared_state_slot is None or self.shared_state_slot.state is None:
            raise RuntimeError(
                "CSA2Attention needs the DeepSeek-V4.1 shared state; build the model through "
                "hybrid_dsv41_stack_spec so layers are wrapped by "
                "SinglePassHyperConnectionHybridLayer"
            )
        return self.shared_state_slot.state

    def _rope_freqs(self, seq_len: int) -> torch.Tensor:
        """Rotary frequencies ``[seq_len, 1, 1, rope_dim]`` for positions ``0..seq_len-1``."""
        # packed_seq=True returns the full table; CSA2 slices positions itself (also under CP).
        out = self.rotary_pos_emb(seq_len, packed_seq=True)
        if isinstance(self.rotary_pos_emb, YarnRotaryEmbedding):
            out = out[0]  # (emb, mscale); the V4 family uses pure rotation (mscale 1)
        return out

    def _make_rope_fn(self, seq_len: int):
        """Return ``rope_fn(t, stride)`` rotating the tail of ``[n, b, h, d]`` at positions
        ``0, stride, ...``; ``n * stride <= seq_len`` must hold."""
        freqs = self._rope_freqs(seq_len)

        def rope_fn(t: torch.Tensor, stride: int) -> torch.Tensor:
            n = t.size(0)
            content, rot = t[..., : -self.rope_dim], t[..., -self.rope_dim :]
            rot = apply_rotary_pos_emb(
                rot,
                freqs[: n * stride : stride],
                self.config,
                mscale=1.0,
                cp_group=self.pg_collection.cp,
                mla_rotary_interleaved=True,
                mla_output_remove_interleaving=True,
            )
            return torch.cat([content, rot], dim=-1)

        return rope_fn

    def _make_rope_rows_fn(self, positions: torch.Tensor, max_position: int):
        """Return ``rope_rows_fn(t)`` rotating the tail of ``[n, h, d]`` rows at the given
        per-row ``positions`` (packed layouts; positions are in-segment)."""
        freqs = self._rope_freqs(max_position + 1)  # [max_position + 1, 1, 1, rope_dim]
        row_freqs = freqs[positions.long()]  # [n, 1, 1, rope_dim]

        def rope_rows_fn(t: torch.Tensor) -> torch.Tensor:
            content, rot = t[..., : -self.rope_dim], t[..., -self.rope_dim :]
            rot = _apply_rotary_pos_emb_bshd(
                rot.unsqueeze(1),  # [n, 1, h, d]
                row_freqs,
                rotary_interleaved=self.config.rotary_interleaved,
                mscale=1.0,
                mla_rotary_interleaved=True,
                mla_output_remove_interleaving=True,
            ).squeeze(1)
            return torch.cat([content, rot], dim=-1)

        return rope_rows_fn

    # ---- packed (THD) path, CP >= 1 ------------------------------------------------------------

    def _gather_owned_entries(
        self, local_rows: torch.Tensor, owned_masks: List[torch.Tensor], cp_group
    ) -> torch.Tensor:
        """Assemble the global sequence-major compressed tensor from per-rank owned entries.

        ``local_rows`` are this rank's entries (``[n_owned_local, ...]``); ``owned_masks[r]`` is
        the global bool mask of the entries rank ``r`` owns (identical on every rank, so the
        capacity and the destination map are computed without communication). The equal-split
        all-gather is autograd aware (reduce-scatter in backward).
        """
        counts = [int(m.sum()) for m in owned_masks]
        capacity = max(counts) if counts else 0
        total = int(owned_masks[0].numel())
        trailing = tuple(local_rows.shape[1:])
        if capacity == 0:
            return local_rows.new_zeros((total,) + trailing)
        padded = local_rows.new_zeros((capacity,) + trailing)
        padded[: local_rows.size(0)] = local_rows
        # Consumers on every rank attend to these entries: the backward must reduce-scatter
        # (sum) their gradients back to the owner, not just keep the local slice.
        gathered = gather_from_sequence_parallel_region(
            padded, tensor_parallel_output_grad=True, group=cp_group
        )  # [cp * capacity, ...]
        dest = torch.cat([torch.nonzero(m, as_tuple=False).squeeze(1) for m in owned_masks])
        src = torch.cat(
            [torch.arange(n, device=local_rows.device) + r * capacity for r, n in enumerate(counts)]
        )
        out = gathered.new_zeros((total,) + trailing)
        out = out.index_put((dest,), gathered[src])
        return out

    def _forward_thd(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        x: torch.Tensor,
        qr: torch.Tensor,
        packed_seq_params: PackedSeqParams,
        boundary_hidden: Optional[torch.Tensor],
        boundary_kv: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Packed sequences with a contiguous CP partition (CP 1 is the degenerate case).

        Row ``i`` of this rank is global row ``global_start + i`` of the packed layout. The key
        buffer is ``[halo window keys | local window keys | global compressed keys]``. Compressed
        entries are produced by the rank holding the group's last token and all-gathered
        (autograd aware) into the sequence-major global order shared by every consumer layer.
        """
        cu, seq_lens = thd.packed_layout(packed_seq_params)
        cp_group = self.pg_collection.cp
        cp_size = cp_group.size() if cp_group is not None else 1
        cp_rank = cp_group.rank() if cp_group is not None else 0
        if cp_size > 1 and packed_seq_params.cp_partition_mode != "contiguous":
            raise ValueError("CSA2Attention with CP requires cp_partition_mode='contiguous'")

        total_q, n_heads, head_dim = query.shape
        global_start = cp_rank * total_q
        # Longest valid segment: sizes the RoPE table for token and compressed positions. One
        # small host sync per layer; the indexer already syncs cu_comp.
        max_position = int(seq_lens.max()) if seq_lens.numel() else 0
        kv_local = key.reshape(total_q, head_dim)
        x_rows = x.reshape(total_q, -1)
        qr_rows = qr.reshape(total_q, -1)

        meta = thd.row_metadata(cu, total_q, global_start, seq_lens)
        halo = 0
        key_buffer = kv_local
        halo_kv = None
        if boundary_kv is not None:
            halo_kv = boundary_kv.reshape(-1, head_dim)
            halo = halo_kv.size(0)
            key_buffer = torch.cat([halo_kv, kv_local], dim=0)
        window_idx = thd.window_indices_thd(meta, self.window_size, 0, halo)

        def attend(
            keys: torch.Tensor, indices: torch.Tensor, parts: Optional[tuple] = None
        ) -> torch.Tensor:
            if self.use_fused:
                # FlashMLA forward / cuDNN DSA backward on flat rows with global -1-padded ids.
                # ``parts`` lets the backward rebuild ``keys`` from the producer tensors instead
                # of retaining the concatenated copy per consumer layer.
                return csa_sparse_attn(
                    query.to(torch.bfloat16),
                    keys,
                    self.attn_sink,
                    indices.contiguous(),
                    self.softmax_scale,
                    is_thd=True,
                    kv_reconstruction_parts=parts,
                ).to(query.dtype)
            out = sparse_attention_with_sink(
                query.unsqueeze(1),
                keys.unsqueeze(1),
                self.attn_sink,
                indices.unsqueeze(1),
                self.softmax_scale,
            )
            return out.reshape(total_q, n_heads * head_dim)

        if not self.plan.has_compressed_path:
            if self.use_fused:
                return attend(key_buffer.to(torch.bfloat16), window_idx)
            return attend(key_buffer, window_idx)

        state = self._shared_state()
        ratio = self.plan.compress_ratio
        cu_comp = thd.compressed_cu_seqlens(cu, ratio, seq_lens)
        compressed_base = halo + total_q

        if self.plan.runs_compressor:
            seg_ids, group_ids, first_rows = thd.compressed_entry_metadata(cu, cu_comp, ratio)
            owned_masks = [
                thd.owned_compressed_entries(first_rows, ratio, r * total_q, total_q)
                for r in range(cp_size)
            ]
            owned = owned_masks[cp_rank]
            hidden_halo = 0
            hidden_rows = x_rows
            if boundary_hidden is not None:
                halo_hidden = boundary_hidden.reshape(-1, x_rows.size(-1))
                hidden_halo = halo_hidden.size(0)
                hidden_rows = torch.cat([halo_hidden, x_rows], dim=0)
            group_rows = thd.compressor_group_rows(
                first_rows[owned], ratio, row_base=global_start - hidden_halo
            )
            out_of_range = group_rows.numel() and (
                group_rows.min() < 0 or group_rows.max() >= hidden_rows.size(0)
            )
            if out_of_range:
                raise RuntimeError(
                    "CSA2 compressor needs a left halo of at least ratio - 1 hidden rows under CP"
                )
            latent = self.compressor.forward_packed(hidden_rows, group_rows)  # [n_owned, d]
            if latent.size(0) == 0:
                # Nothing to rotate or project on this rank (see forward_packed); keep the
                # empty outputs attached to the graph for the collective backward.
                index_keys_owned = self._empty_rows_like(latent, self.indexer.head_dim)
                kv_owned = latent
            else:
                positions_owned = group_ids[owned] * ratio
                rope_owned = self._make_rope_rows_fn(positions_owned, max_position)
                index_keys_owned = self.indexer.build_index_keys_packed(latent, rope_owned)
                kv_owned = rope_owned(latent.unsqueeze(1)).squeeze(1)
            if cp_size > 1:
                kv_global = self._gather_owned_entries(kv_owned, owned_masks, cp_group)
                keys_global = self._gather_owned_entries(index_keys_owned, owned_masks, cp_group)
            else:
                kv_global, keys_global = kv_owned, index_keys_owned
            record = CompressedKVRecord(
                source_layer=self.model_layer_id,
                compress_ratio=ratio,
                n_compressed=int(cu_comp[-1]),
                kv=kv_global,
                index_keys=keys_global,
            )
            state.publish_compressed(record)
        else:
            record = state.get_compressed(self.plan.kv_source)
            if record.compress_ratio != ratio:
                raise RuntimeError(
                    f"model layer {self.model_layer_id} (ratio {ratio}) reads compressed KV of "
                    f"ratio {record.compress_ratio} from layer {record.source_layer}"
                )

        if self.plan.runs_indexer:
            candidate_blocks = None
            if self.plan.uses_candidates:
                candidate_blocks = state.get_candidates(self.plan.kv_source)
            rope_rows = self._make_rope_rows_fn(meta.positions, max_position)
            topk_local, produced = self.indexer.forward_packed(
                x_rows,
                qr_rows,
                record.index_keys,
                meta.positions,
                meta.segment_ids,
                meta.valid,
                cu_comp,
                rope_rows,
                candidate_blocks,
            )
            seg_comp_start = cu_comp.to(torch.int64)[meta.segment_ids]
            topk_flat = torch.where(
                topk_local >= 0,
                topk_local.to(torch.int64) + seg_comp_start.unsqueeze(1) + compressed_base,
                torch.full_like(topk_local, -1, dtype=torch.int64),
            ).to(torch.int32)
            if produced is not None:
                state.publish_candidates(self.model_layer_id, produced)
            state.publish_topk(self.model_layer_id, topk_flat)
        else:
            topk_flat = state.get_topk(self.plan.index_source)

        indices = torch.cat([window_idx, topk_flat], dim=-1)
        if self.use_fused:
            # Producer tensors in kernel dtype; the concatenation stays the differentiable
            # input while the backward rebuilds it from these parts.
            local_bf = kv_local.to(torch.bfloat16)
            halo_bf = (
                halo_kv.to(torch.bfloat16)
                if halo_kv is not None
                else local_bf.new_zeros((0, head_dim))
            )
            comp_bf = record.kv.to(torch.bfloat16)
            keys = torch.cat([halo_bf, local_bf, comp_bf], dim=0)
            return attend(keys, indices, parts=(halo_bf, local_bf, comp_bf))
        keys = torch.cat([key_buffer, record.kv.to(key_buffer.dtype)], dim=0)
        return attend(keys, indices)

    # ---- forward -----------------------------------------------------------------------------

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor,
        x: torch.Tensor = None,
        qr: torch.Tensor = None,
        attn_mask_type: AttnMaskType = None,
        attention_bias: torch.Tensor = None,
        packed_seq_params: PackedSeqParams = None,
        boundary_hidden: Optional[torch.Tensor] = None,
        boundary_kv: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Attention over the sliding window plus (shared) compressed positions.

        Args:
            query: ``[s, b, h, d]`` rotated queries.
            key: ``[s, b, 1, d]`` rotated single-head window keys (== values).
            x: ``[s, b, hidden]`` normalised layer input (compressor / indexer weights input).
            qr: ``[s, b, q_lora_rank]`` normalised low-rank query (indexer query input).

        Returns:
            ``[s, b, h * d]``.
        """
        if attention_mask is not None:
            raise NotImplementedError(
                "CSA2Attention derives visibility from positions only (causal within a sequence "
                "or packed segment); explicit attention masks are not supported. Run with "
                "--no-create-attention-mask-in-dataloader."
            )
        if packed_seq_params is not None and packed_seq_params.qkv_format == 'thd':
            return self._forward_thd(
                query, key, x, qr, packed_seq_params, boundary_hidden, boundary_kv
            )
        if packed_seq_params is not None:
            raise NotImplementedError("CSA2Attention: only qkv_format='thd' packed inputs")
        if self.pg_collection.cp is not None and self.pg_collection.cp.size() > 1:
            raise NotImplementedError(
                "CSA2Attention: context parallelism requires packed (THD) inputs"
            )

        s, b, h, d = query.shape
        window_keys = key.squeeze(2)  # [s, b, d]
        window_idx = sliding_window_indices(s, self.window_size, device=query.device)

        if not self.plan.has_compressed_path:
            indices = concat_window_and_compressed_indices(window_idx, None, s, b)
            out = sparse_attention_with_sink(
                query, window_keys, self.attn_sink, indices, self.softmax_scale
            )
            return out.reshape(s, b, h * d)

        state = self._shared_state()
        ratio = self.plan.compress_ratio
        rope_fn = self._make_rope_fn(s)

        if self.plan.runs_compressor:
            latent = self.compressor(x)  # [n_comp, b, d], pre-RoPE
            index_keys = self.indexer.build_index_keys(latent, rope_fn)
            compressed_kv = rope_fn(latent.unsqueeze(2), ratio).squeeze(2)
            record = CompressedKVRecord(
                source_layer=self.model_layer_id,
                compress_ratio=ratio,
                n_compressed=latent.size(0),
                kv=compressed_kv,
                index_keys=index_keys,
            )
            state.publish_compressed(record)
        else:
            record = state.get_compressed(self.plan.kv_source)
            if record.compress_ratio != ratio:
                raise RuntimeError(
                    f"model layer {self.model_layer_id} (ratio {ratio}) reads compressed KV of "
                    f"ratio {record.compress_ratio} from layer {record.source_layer}"
                )

        visible = compressed_visible_counts(s, ratio, device=query.device)

        if self.plan.runs_indexer:
            candidate_blocks = None
            if self.plan.uses_candidates:
                candidate_blocks = state.get_candidates(self.plan.kv_source)
            topk, produced_candidates = self.indexer(
                x, qr, record.index_keys, visible, rope_fn, candidate_blocks
            )
            if produced_candidates is not None:
                state.publish_candidates(self.model_layer_id, produced_candidates)
            state.publish_topk(self.model_layer_id, topk)
        else:
            topk = state.get_topk(self.plan.index_source)

        indices = concat_window_and_compressed_indices(window_idx, topk, s, b)
        keys = torch.cat([window_keys, record.kv.to(window_keys.dtype)], dim=0)
        out = sparse_attention_with_sink(query, keys, self.attn_sink, indices, self.softmax_scale)
        return out.reshape(s, b, h * d)


class DSv41SelfAttention(DSv4HybridSelfAttention):
    """DeepSeek-V4.1 self-attention: V4 projections with V4.1 rotary selection.

    In V4.1 every layer that attends to compressed positions (``compress_ratio >= 1``,
    including the ratio-1 decoder layers) rotates with ``csa_compress_rotary_base`` and
    YaRN; window-only layers use the plain base without YaRN. The V4 module only treats
    ``compress_ratio > 1`` as compressed, so the ratio-1 case is corrected here.

    V4.1 also drops the weight-free per-head RMS normalisation of the query that V4 applies
    after ``wq_b`` (official ``Attention.forward``: ``q = wq_b(q_norm(wq_a(x)))`` then RoPE);
    the alignment probe traced the layer-0 attention mismatch (cosine 0.78) to it.
    """

    query_head_rms_norm = False

    def __init__(
        self,
        config: MLATransformerConfig,
        submodules: DSv4HybridSelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type=AttnMaskType.padding,
        cp_comm_type: Optional[str] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        is_mtp_layer: bool = False,
        pp_layer_offset: Optional[int] = None,
        compress_ratio: Optional[int] = None,
        name: str | None = None,
    ) -> None:
        if getattr(config, "dsv4_version", "v4") != "v4.1":
            raise ValueError("DSv41SelfAttention requires dsv4_version='v4.1'")
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            cp_comm_type=cp_comm_type,
            pg_collection=pg_collection,
            is_mtp_layer=is_mtp_layer,
            pp_layer_offset=pp_layer_offset,
            compress_ratio=compress_ratio,
            name=name,
        )
        if self._dsv4_compress_ratio == 1 and not self._dsv4_uses_yarn_rope:
            self._dsv4_uses_yarn_rope = True
            self._dsv4_rope_base = self.config.csa_compress_rotary_base
            self.rotary_pos_emb = YarnRotaryEmbedding(
                self.config.qk_pos_emb_head_dim,
                rotary_base=self._dsv4_rope_base,
                scaling_factor=self.config.rotary_scaling_factor,
                original_max_position_embeddings=self.config.original_max_position_embeddings,
                beta_fast=self.config.beta_fast,
                beta_slow=self.config.beta_slow,
                mscale=self.config.mscale,
                mscale_all_dim=self.config.mscale_all_dim,
                cp_group=self.pg_collection.cp,
            )
            self.core_attention.rotary_pos_emb = self.rotary_pos_emb
