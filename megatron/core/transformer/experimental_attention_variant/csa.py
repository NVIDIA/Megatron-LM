# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Compressed-Sparse Attention (CSA) and Heavily-Compressed Attention (HCA).

CSA and HCA are the two attention variants introduced in DeepSeek-V4 (Section 2.3 of
the technical report). Both share a single core-attention module
(``CompressedSparseAttention``); the per-layer ``compress_ratio`` controls behaviour:

* ``ratio == 4``  (CSA) — overlap=2, sliding-window + learned indexer top-k over
  compressed positions (sparse).
* ``ratio == 128`` (HCA) — overlap=1, sliding-window + dense attention over all
  compressed positions (no indexer).
* ``ratio == 0``  — sliding-window only (no compression).

This is a CP=1 / TP=1 prototype using the unfused RoPE path.
"""

import copy
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from megatron.core.models.common.embeddings import RotaryEmbedding, apply_rotary_pos_emb
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant.dsa import (
    DSAIndexerLossAutoScaler,
    DSAIndexerLossLoggingHelper,
    FusedDSAIndexerLoss,
    fused_qk_topk_naive,
    rotate_activation,
)
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import nvtx_range_pop, nvtx_range_push

# ---------------------------------------------------------------------------
# Helper functions for index computation
# ---------------------------------------------------------------------------


@lru_cache(maxsize=8)
def _get_window_topk_idxs_cached(window_size: int, seqlen: int, device_str: str) -> torch.Tensor:
    base = torch.arange(seqlen, device=device_str).unsqueeze(1)
    offsets = torch.arange(window_size, device=device_str)
    matrix = (base - window_size + 1).clamp(min=0) + offsets
    matrix = torch.where(matrix > base, -1, matrix)
    return matrix


def get_window_topk_idxs(
    window_size: int, batch_size: int, seqlen: int, device: torch.device
) -> torch.Tensor:
    """Sliding-window indices ``[batch, seqlen, window_size]``."""
    matrix = _get_window_topk_idxs_cached(window_size, seqlen, str(device))
    return matrix.unsqueeze(0).expand(batch_size, -1, -1)


@lru_cache(maxsize=8)
def _get_compress_topk_idxs_cached(
    ratio: int, seqlen: int, offset: int, device_str: str
) -> torch.Tensor:
    n_compressed = seqlen // ratio
    matrix = torch.arange(n_compressed, device=device_str).repeat(seqlen, 1)
    mask = matrix >= torch.arange(1, seqlen + 1, device=device_str).unsqueeze(1) // ratio
    matrix = torch.where(mask, -1, matrix + offset)
    return matrix


def get_compress_topk_idxs(
    ratio: int, batch_size: int, seqlen: int, offset: int, device: torch.device
) -> torch.Tensor:
    """All-compressed-position indices ``[batch, seqlen, seqlen // ratio]``."""
    matrix = _get_compress_topk_idxs_cached(ratio, seqlen, offset, str(device))
    return matrix.unsqueeze(0).expand(batch_size, -1, -1)


# ---------------------------------------------------------------------------
# Helper functions for RoPE
# ---------------------------------------------------------------------------


def _apply_partial_rope(
    x: torch.Tensor,
    nope_dim: int,
    pos_dim: int,
    rotary_pos_emb_module: RotaryEmbedding,
    config: TransformerConfig,
    rotary_seq_len: int,
    ratio: int = 1,
    cp_group: torch.distributed.ProcessGroup = None,
) -> torch.Tensor:
    """Apply RoPE to the last ``pos_dim`` dims, leave the first ``nope_dim`` unchanged.

    Accepts both 3-D ``[seq, batch, head_dim]`` and 4-D ``[seq, batch, heads, head_dim]``
    inputs (a temporary head dim is inserted for the 3-D case).
    """
    if ratio == 1:
        total_seq_len = rotary_seq_len
    else:
        total_seq_len = rotary_seq_len * ratio
    if config.rope_type == "rope":
        rotary_pos_emb = rotary_pos_emb_module(total_seq_len, packed_seq=False)
        mscale = 1.0
    else:
        rotary_pos_emb, mscale = rotary_pos_emb_module(total_seq_len, packed_seq=False)

    if ratio > 1:
        rotary_pos_emb = rotary_pos_emb[:total_seq_len:ratio][:rotary_seq_len]

    squeeze_head = x.dim() == 3
    if squeeze_head:
        x = x.unsqueeze(-2)
    x_nope, x_pe = torch.split(x, [nope_dim, pos_dim], dim=-1)
    x_pe = apply_rotary_pos_emb(
        x_pe, rotary_pos_emb, config=config, cu_seqlens=None, mscale=mscale, cp_group=cp_group
    )
    out = torch.cat([x_nope, x_pe], dim=-1)
    if squeeze_head:
        out = out.squeeze(-2)
    return out


# ---------------------------------------------------------------------------
# Sparse attention kernel (unfused, differentiable) with attention sink
# ---------------------------------------------------------------------------


def unfused_compressed_sparse_attn(
    query: torch.Tensor,
    kv_full: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_indices: torch.Tensor,
    softmax_scale: float,
    use_attn_sink: bool = True,
) -> torch.Tensor:
    """Differentiable sparse MQA with optional learnable per-head attention sink.

    Args:
        query:        ``[sq, b, np, hn]``  multi-head query.
        kv_full:      ``[n_kv, b, hn]``    single-head KV (original + compressed).
        attn_sink:    ``[np]``             per-head learnable sink logit.
        topk_indices: ``[b, sq, topk]``    indices into ``kv_full`` (-1 = invalid).
        softmax_scale: float
        use_attn_sink: when False the sink is omitted.

    Returns:
        ``[sq, b, np * hn]``
    """
    sq, b, np_, hn = query.size()

    # --- Gather KV at topk positions ---
    kv_t = kv_full.permute(1, 0, 2)  # [b, n_kv, hn]

    safe_indices = topk_indices.clamp(min=0).long()
    safe_indices_exp = safe_indices.unsqueeze(-1).expand(-1, -1, -1, hn)
    kv_gathered = torch.gather(
        kv_t.unsqueeze(1).expand(-1, sq, -1, -1), dim=2, index=safe_indices_exp
    )

    # --- Attention scores ---
    q = query.permute(1, 2, 0, 3).float()  # [b, np, sq, hn]
    kv_g = kv_gathered.float()  # [b, sq, topk, hn]

    scores = torch.einsum("bnsh,bskh->bnsk", q, kv_g) * softmax_scale

    invalid_mask = (topk_indices < 0).unsqueeze(1)  # [b, 1, sq, topk]
    scores = scores.masked_fill(invalid_mask, float("-inf"))

    if use_attn_sink:
        sink = attn_sink.view(1, np_, 1, 1).float()
        scores_max = scores.max(dim=-1, keepdim=True).values
        scores_max = torch.max(scores_max, sink)
        # If a row has no valid KV, scores_max is still finite (sink), so safe.
        exp_scores = torch.exp(scores - scores_max)
        exp_sink = torch.exp(sink - scores_max)
        sum_exp = exp_scores.sum(dim=-1, keepdim=True) + exp_sink
        attn_weights = exp_scores / sum_exp
    else:
        # Without sink, fall back to standard masked softmax.
        # If a row has no valid kv, set the attn weights to 0 (skip token).
        all_invalid = invalid_mask.all(dim=-1, keepdim=True)  # [b, 1, sq, 1]
        scores = scores.masked_fill(all_invalid, 0.0)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = attn_weights.masked_fill(all_invalid, 0.0)

    output = torch.einsum("bnsk,bskh->bnsh", attn_weights, kv_g)
    output = output.to(query.dtype)
    output = output.permute(2, 0, 1, 3).contiguous().reshape(sq, b, np_ * hn)
    return output


# ---------------------------------------------------------------------------
# Compressor
# ---------------------------------------------------------------------------


@dataclass
class CompressorSubmodules:
    """Submodule specs for ``Compressor``."""

    linear_wkv: Union[ModuleSpec, type] = None
    linear_wgate: Union[ModuleSpec, type] = None
    norm: Union[ModuleSpec, type] = None


class Compressor(MegatronModule):
    """Gated pooling compressor for CSA and HCA.

    Compresses a sequence of tokens into a shorter sequence by pooling groups of
    ``compress_ratio`` tokens with learned gated weights.

    For ``compress_ratio == 4`` (CSA) overlap=2 is used; for larger ratios (e.g. 128
    for HCA) overlap=1 is used.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: CompressorSubmodules,
        compress_ratio: int,
        head_dim: int,
        rotate: bool = False,
        rotary_pos_emb: nn.Module = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ) -> None:
        super().__init__(config=config)

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
        self.pg_collection = pg_collection

        self.compress_ratio = compress_ratio
        self.head_dim = head_dim
        self.overlap = compress_ratio == 4
        self.coff = 1 + int(self.overlap)
        self.rotate = rotate
        self.qk_pos_emb_head_dim = config.qk_pos_emb_head_dim

        self.rotary_pos_emb = rotary_pos_emb

        proj_out_dim = self.coff * head_dim

        self.linear_wkv = build_module(
            submodules.linear_wkv,
            config.hidden_size,
            proj_out_dim,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )

        self.linear_wgate = build_module(
            submodules.linear_wgate,
            config.hidden_size,
            proj_out_dim,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )

        # Learned positional bias B (kept fp32 for stability)
        _ape = torch.empty(
            compress_ratio, proj_out_dim, device=torch.cuda.current_device(), dtype=torch.float32
        )
        config.init_method(_ape)
        self.ape = nn.Parameter(_ape)

        norm_config = copy.copy(config)
        norm_config.normalization = "RMSNorm"
        self.norm = build_module(
            submodules.norm, config=norm_config, hidden_size=head_dim, eps=config.layernorm_epsilon
        )

    def _overlap_transform(self, tensor: torch.Tensor, fill_value: float = 0) -> torch.Tensor:
        """Apply overlapping window transform for 4x compression.

        ``[n_groups, ratio, b, coff * head_dim] -> [n_groups, 2 * ratio, b, head_dim]``.
        """
        n_groups, ratio, b_dim, _ = tensor.size()
        d = self.head_dim
        new_tensor = tensor.new_full((n_groups, 2 * ratio, b_dim, d), fill_value)
        new_tensor[:, ratio:] = tensor[:, :, :, d:]
        new_tensor[1:, :ratio] = tensor[:-1, :, :, :d]
        return new_tensor

    def forward(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """Compress hidden states.

        Returns ``[sq // ratio, b, head_dim]`` or ``None`` if the input is shorter
        than ``compress_ratio``.
        """
        nvtx_range_push("compressor")

        sq, b, _ = x.size()
        ratio = self.compress_ratio

        if sq < ratio:
            nvtx_range_pop("compressor")
            return None

        kv, _ = self.linear_wkv(x)
        score, _ = self.linear_wgate(x)

        cutoff = (sq // ratio) * ratio
        if cutoff < sq:
            kv = kv[:cutoff]
            score = score[:cutoff]

        n_compressed = cutoff // ratio

        kv = kv.view(n_compressed, ratio, b, -1)
        score = score.view(n_compressed, ratio, b, -1)

        score = score + self.ape.view(1, ratio, 1, -1)

        if self.overlap:
            kv = self._overlap_transform(kv, fill_value=0)
            score = self._overlap_transform(score, fill_value=float("-inf"))

        kv = (kv * torch.softmax(score, dim=1)).sum(dim=1)
        kv = self.norm(kv.to(x.dtype))

        kv = _apply_partial_rope(
            kv,
            self.head_dim - self.qk_pos_emb_head_dim,
            self.qk_pos_emb_head_dim,
            self.rotary_pos_emb,
            self.config,
            n_compressed,
            ratio=ratio,
            cp_group=self.pg_collection.cp,
        )

        if self.rotate:
            kv = rotate_activation(kv)

        nvtx_range_pop("compressor")
        return kv


# ---------------------------------------------------------------------------
# CSA Indexer (top-k retrieval over compressed positions)
# ---------------------------------------------------------------------------


@dataclass
class CSAIndexerSubmodules:
    """Submodule specs for ``CSAIndexer``."""

    linear_wq_b: Union[ModuleSpec, type] = None
    linear_weights_proj: Union[ModuleSpec, type] = None
    compressor: Union[ModuleSpec, type] = None


class CSAIndexer(MegatronModule):
    """Learned top-k retrieval over compressed KV positions for CSA.

    Reuses the index-score logic from ``DSAIndexer`` (einsum -> ReLU -> weight -> sum
    -> top-k) and ``rotate_activation`` (Hadamard transform) from ``dsa.py``.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: CSAIndexerSubmodules,
        compress_ratio: int,
        rotary_pos_emb: nn.Module = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ) -> None:
        super().__init__(config=config)

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
        self.pg_collection = pg_collection

        self.compress_ratio = compress_ratio
        self.hidden_size = config.hidden_size
        self.qk_pos_emb_head_dim = config.qk_pos_emb_head_dim
        self.q_lora_rank = (
            config.q_lora_rank if config.q_lora_rank is not None else config.hidden_size
        )

        self.index_n_heads = config.dsa_indexer_n_heads
        self.index_head_dim = config.dsa_indexer_head_dim
        self.index_topk = config.dsa_indexer_topk

        self.softmax_scale: float = self.index_head_dim**-0.5

        self.rotary_pos_emb = rotary_pos_emb

        self.linear_wq_b = build_module(
            submodules.linear_wq_b,
            self.q_lora_rank,
            self.index_n_heads * self.index_head_dim,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )

        self.linear_weights_proj = build_module(
            submodules.linear_weights_proj,
            self.hidden_size,
            self.index_n_heads,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )

        self.compressor = build_module(
            submodules.compressor,
            config=config,
            compress_ratio=compress_ratio,
            head_dim=self.index_head_dim,
            rotate=True,
            rotary_pos_emb=rotary_pos_emb,
            pg_collection=pg_collection,
        )

    def forward_before_topk(
        self, x: torch.Tensor, qr: torch.Tensor, packed_seq_params: Optional[PackedSeqParams] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute Q, compressed K, and weights before top-k selection."""
        nvtx_range_push("csa_indexer_before_topk")

        sq, bsz, _ = x.size()

        q, _ = self.linear_wq_b(qr)
        q = q.reshape(sq, bsz, self.index_n_heads, self.index_head_dim)
        q = _apply_partial_rope(
            q,
            self.index_head_dim - self.qk_pos_emb_head_dim,
            self.qk_pos_emb_head_dim,
            self.rotary_pos_emb,
            self.config,
            sq,
            ratio=1,
            cp_group=self.pg_collection.cp,
        )
        q = rotate_activation(q)

        k = self.compressor(x)  # [sq//ratio, b, index_head_dim]

        weights, _ = self.linear_weights_proj(x)
        weights = weights * (self.index_n_heads**-0.5)

        nvtx_range_pop("csa_indexer_before_topk")
        return q, k, weights

    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(index_scores, topk_indices)``."""
        nvtx_range_push("csa_indexer")
        assert packed_seq_params is None, "Packed sequence is not supported for CSAIndexer."
        q, k, weights = self.forward_before_topk(x, qr, packed_seq_params)
        nvtx_range_push("csa_indexer_qk_topk")
        effective_topk = min(self.index_topk, k.size(0))
        index_scores, topk_indices = fused_qk_topk_naive(q, k, weights, effective_topk, mask)
        nvtx_range_pop("csa_indexer_qk_topk")
        nvtx_range_pop("csa_indexer")
        return index_scores, topk_indices


# ---------------------------------------------------------------------------
# CompressedSparseAttention (core attention)
# ---------------------------------------------------------------------------


@dataclass
class CompressedSparseAttentionSubmodules:
    """Submodule specs for ``CompressedSparseAttention``."""

    compressor: Union[ModuleSpec, type] = None
    indexer: Union[ModuleSpec, type] = None


class CompressedSparseAttention(MegatronModule):
    """Core attention used by both CSA and HCA layers.

    Combines sliding-window attention with compressed KV attention. Behaviour
    depends on ``compress_ratio``:

    * ``ratio == 0``  : window-only.
    * ``ratio == 4``  : window + 4x compressed + learned indexer (CSA).
    * ``ratio >  4``  : window + ``ratio``x compressed, attend to all (HCA).
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: CompressedSparseAttentionSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: Optional[float] = None,
        softmax_scale: Optional[float] = None,
        k_channels: Optional[int] = None,
        v_channels: Optional[int] = None,
        cp_comm_type: str = "p2p",
        pg_collection: Optional[ProcessGroupCollection] = None,
        rotary_pos_emb: nn.Module = None,
        compress_ratio: int = 0,
    ):
        super().__init__(config=config)

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
        self.pg_collection = pg_collection

        self.layer_number = layer_number
        self.compress_ratio = compress_ratio
        self.window_size = config.csa_window_size
        self.v_head_dim = config.v_head_dim

        self.n_local_heads = config.num_attention_heads

        if softmax_scale is None:
            softmax_scale = config.v_head_dim**-0.5
        self.softmax_scale = softmax_scale

        self.use_attention_sink = getattr(config, 'csa_attention_sink', True)

        # Learnable per-head attention sink
        self.attn_sink = nn.Parameter(torch.zeros(self.n_local_heads, dtype=torch.float32))

        # Conditionally build Compressor (ratio > 1)
        if self.compress_ratio > 1 and submodules.compressor is not None:
            self.compressor = build_module(
                submodules.compressor,
                config=config,
                compress_ratio=self.compress_ratio,
                head_dim=config.v_head_dim,
                rotate=False,
                rotary_pos_emb=rotary_pos_emb,
                pg_collection=pg_collection,
            )
        else:
            self.compressor = None

        # Conditionally build Indexer (only ratio == 4 + non-dense mode)
        if (
            self.compress_ratio == 4
            and not config.csa_dense_mode
            and submodules.indexer is not None
        ):
            self.indexer = build_module(
                submodules.indexer,
                config=config,
                compress_ratio=self.compress_ratio,
                rotary_pos_emb=rotary_pos_emb,
                pg_collection=pg_collection,
            )
        else:
            self.indexer = None

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
    ) -> torch.Tensor:
        """Forward.

        Args:
            query: ``[sq, b, np, v_head_dim]``
            key:   ``[sq, b, 1, v_head_dim]``
            value: unused (key == value in MQA)
            attention_mask: ignored (always causal here).
            x:  ``[sq, b, hidden_size]`` original hidden states.
            qr: ``[sq, b, q_lora_rank]`` compressed query representation.
        """
        nvtx_range_push("compressed_sparse_attn")
        assert packed_seq_params is None, "Packed sequence is not supported for CSA/HCA."

        sq, b, np, hn = query.size()

        kv = key.squeeze(-2)  # [sq, b, v_head_dim]

        # --- Compression ---
        if self.compressor is not None and self.compress_ratio > 1:
            compressed_kv = self.compressor(x)
            if compressed_kv is not None:
                kv_full = torch.cat([kv, compressed_kv], dim=0)
                n_compressed = compressed_kv.size(0)
            else:
                kv_full = kv
                n_compressed = 0
        else:
            kv_full = kv
            n_compressed = 0

        offset = sq  # compressed positions are appended after original positions

        # --- Sliding window indices ---
        window_idxs = get_window_topk_idxs(self.window_size, b, sq, query.device)

        indexer_loss = None

        if self.compress_ratio > 1 and n_compressed > 0:
            nvtx_range_push("compressed_indices")
            if self.indexer is not None:
                # CSA: learned top-k indexer
                x_det = x.detach()
                qr_det = qr.detach()

                causal_mask = (
                    torch.arange(n_compressed, device=x.device).unsqueeze(0).expand(sq, -1)
                )
                positions = torch.arange(1, sq + 1, device=x.device).unsqueeze(1)
                causal_mask = (
                    torch.where(causal_mask >= positions // self.compress_ratio, float("-inf"), 0.0)
                    .unsqueeze(0)
                    .expand(b, -1, -1)
                )

                if self.training and torch.is_grad_enabled():
                    q_indexer, k_indexer, weights_indexer = self.indexer.forward_before_topk(
                        x_det, qr_det, packed_seq_params
                    )
                    indexer_loss_coeff = getattr(self.config, 'dsa_indexer_loss_coeff', 0.0) or 0.0
                    key_for_loss = compressed_kv.unsqueeze(2).expand(-1, -1, np, -1)
                    weights_for_unfused = weights_indexer * self.indexer.softmax_scale
                    topk_indices_compressed, indexer_loss = FusedDSAIndexerLoss.apply(
                        q_indexer,
                        weights_for_unfused,
                        k_indexer,
                        query.detach(),
                        key_for_loss.detach(),
                        self.softmax_scale,
                        min(self.indexer.index_topk, n_compressed),
                        indexer_loss_coeff,
                        causal_mask,
                        getattr(self.config, "dsa_indexer_use_sparse_loss", True),
                        self.indexer.pg_collection,
                    )
                    if indexer_loss_coeff > 0:
                        DSAIndexerLossLoggingHelper.save_loss_to_tracker(
                            loss=indexer_loss,
                            layer_number=self.layer_number,
                            num_layers=self.config.num_layers,
                        )
                else:
                    _, topk_indices_compressed = self.indexer(
                        x_det, qr_det, mask=causal_mask, packed_seq_params=packed_seq_params
                    )

                n_valid_per_pos = positions // self.compress_ratio  # [sq, 1]
                valid = topk_indices_compressed < n_valid_per_pos
                compress_topk_idxs = torch.where(
                    valid, topk_indices_compressed + offset, torch.tensor(-1, device=x.device)
                )
            else:
                # HCA / CSA-dense: attend to all valid compressed positions
                compress_topk_idxs = get_compress_topk_idxs(
                    self.compress_ratio, b, sq, offset, query.device
                )

            topk_idxs = torch.cat([window_idxs, compress_topk_idxs], dim=-1)
            nvtx_range_pop("compressed_indices")
        else:
            topk_idxs = window_idxs

        topk_idxs = topk_idxs.int()

        nvtx_range_push("sparse_attn_kernel")
        output = unfused_compressed_sparse_attn(
            query,
            kv_full,
            self.attn_sink.float(),
            topk_idxs,
            self.softmax_scale,
            use_attn_sink=self.use_attention_sink,
        )
        nvtx_range_pop("sparse_attn_kernel")

        if indexer_loss is not None and self.training and torch.is_grad_enabled():
            output = DSAIndexerLossAutoScaler.apply(output, indexer_loss)

        nvtx_range_pop("compressed_sparse_attn")
        return output
