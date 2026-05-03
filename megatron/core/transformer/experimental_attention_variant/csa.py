# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import copy
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from megatron.core.fusions.fused_mla_yarn_rope_apply import fused_mla_rope_inplace
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


# TODO: the lru_cache may not work well with packed sequence
@lru_cache(maxsize=8)
def _get_window_topk_idxs_cached(window_size: int, seqlen: int, device_str: str) -> torch.Tensor:
    """Compute sliding-window indices for a single sequence (cached).

    Returns:
        indices: [seqlen, window_size] int tensor, -1 for invalid positions.
    """
    base = torch.arange(seqlen, device=device_str).unsqueeze(1)
    offsets = torch.arange(window_size, device=device_str)
    matrix = (base - window_size + 1).clamp(min=0) + offsets
    matrix = torch.where(matrix > base, -1, matrix)
    return matrix


def get_window_topk_idxs(
    window_size: int, batch_size: int, seqlen: int, device: torch.device
) -> torch.Tensor:
    """Sliding-window indices [batch, seqlen, window_size]."""
    matrix = _get_window_topk_idxs_cached(window_size, seqlen, str(device))
    return matrix.unsqueeze(0).expand(batch_size, -1, -1)


# TODO: the lru_cache may not work well with packed sequence
@lru_cache(maxsize=8)
def _get_compress_topk_idxs_cached(
    ratio: int, seqlen: int, offset: int, device_str: str
) -> torch.Tensor:
    """Compute all-compressed-positions indices for a single sequence (cached).

    Returns:
        indices: [seqlen, seqlen // ratio] int tensor, -1 for future positions.
    """
    n_compressed = seqlen // ratio
    matrix = torch.arange(n_compressed, device=device_str).repeat(seqlen, 1)
    mask = matrix >= torch.arange(1, seqlen + 1, device=device_str).unsqueeze(1) // ratio
    matrix = torch.where(mask, -1, matrix + offset)
    return matrix


def get_compress_topk_idxs(
    ratio: int, batch_size: int, seqlen: int, offset: int, device: torch.device
) -> torch.Tensor:
    """All-compressed-position indices [batch, seqlen, seqlen // ratio]."""
    matrix = _get_compress_topk_idxs_cached(ratio, seqlen, offset, str(device))
    return matrix.unsqueeze(0).expand(batch_size, -1, -1)


# ---------------------------------------------------------------------------
# Helper functions for RoPE
# ---------------------------------------------------------------------------


def _apply_rope(
    x: torch.Tensor,
    nope_dim: int,
    pos_dim: int,
    rotary_pos_emb_module: RotaryEmbedding,
    config: TransformerConfig,
    rotary_seq_len: int,
    ratio: int = 1,
    cp_group: torch.distributed.ProcessGroup = None,
) -> torch.Tensor:
    """Apply RoPE to the last ``qk_pos_emb_head_dim`` dims, leaving the rest unchanged.

    Accepts both 3-D ``[seq, batch, head_dim]`` and 4-D ``[seq, batch, heads, head_dim]``
    inputs.  When the input is 3-D a temporary head dimension is inserted for
    ``apply_rotary_pos_emb`` and removed before returning.
    """
    if ratio == 1:
        total_seq_len = rotary_seq_len
    else:
        total_seq_len = rotary_seq_len * ratio
    mscale = 1.0
    rotary_pos_cos = None
    rotary_pos_sin = None
    if config.rope_type == "rope":
        rotary_pos_emb = rotary_pos_emb_module(total_seq_len, packed_seq=False)
        mscale = 1.0
    else:
        if config.apply_rope_fusion:
            rotary_pos_cos, rotary_pos_sin = rotary_pos_emb_module.get_cached_cos_sin(
                total_seq_len, dtype=x.dtype, packed_seq=False
            )
            rotary_pos_emb = None
            assert (
                fused_mla_rope_inplace is not None
            ), "Fused MLA RoPE apply is not imported successfully"
        else:
            rotary_pos_emb, mscale = rotary_pos_emb_module(total_seq_len, packed_seq=False)
    if rotary_pos_emb is not None and ratio > 1:
        rotary_pos_emb = rotary_pos_emb[:total_seq_len:ratio][:rotary_seq_len]
    if rotary_pos_cos is not None and ratio > 1:
        rotary_pos_cos = rotary_pos_cos[:total_seq_len:ratio][:rotary_seq_len]
    if rotary_pos_sin is not None and ratio > 1:
        rotary_pos_sin = rotary_pos_sin[:total_seq_len:ratio][:rotary_seq_len]

    squeeze_head = x.dim() == 3
    if squeeze_head:
        x = x.unsqueeze(-2)
    if config.apply_rope_fusion:
        out = fused_mla_rope_inplace(
            x,
            rotary_pos_cos,
            rotary_pos_sin,
            nope_dim,
            pos_dim,
            None,
            cp_group.rank(),
            cp_group.size(),
            remove_interleaving=True,
        )
    else:
        x_nope, x_pe = torch.split(x, [nope_dim, pos_dim], dim=-1)
        x_pe = apply_rotary_pos_emb(
            x_pe,
            rotary_pos_emb,
            config=config,
            cu_seqlens=None,
            mscale=mscale,
            cp_group=cp_group,
            mla_rotary_interleaved=True,
            mla_output_remove_interleaving=True,
        )
        out = torch.cat([x_nope, x_pe], dim=-1)
    if squeeze_head:
        out = out.squeeze(-2)
    return out


# ---------------------------------------------------------------------------
# Sparse attention kernel (unfused, differentiable)
# ---------------------------------------------------------------------------


def unfused_compressed_sparse_attn(
    query: torch.Tensor,
    kv_full: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_indices: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Differentiable sparse attention with MQA and attention sink.

    Args:
        query:        [sq, b, np, hn]   multi-head query.
        kv_full:      [n_kv, b, hn]     single-head KV (original + compressed).
        attn_sink:    [np]              per-head learnable bias.
        topk_indices: [b, sq, topk]     indices into kv_full (int32, -1 = invalid).
        softmax_scale: float

    Returns:
        output:       [sq, b, np * hn]
    """
    sq, b, np_, hn = query.size()

    # --- Gather KV at topk positions ---
    # kv_full: [n_kv, b, hn] -> [b, n_kv, hn]
    kv_t = kv_full.permute(1, 0, 2)

    safe_indices = topk_indices.clamp(min=0).long()  # [b, sq, topk]
    safe_indices_exp = safe_indices.unsqueeze(-1).expand(-1, -1, -1, hn)  # [b, sq, topk, hn]
    # [b, n_kv, hn] -> [b, 1, n_kv, hn] -> gather -> [b, sq, topk, hn]
    kv_gathered = torch.gather(
        kv_t.unsqueeze(1).expand(-1, sq, -1, -1), dim=2, index=safe_indices_exp
    )

    # --- Attention scores ---
    # query: [sq, b, np, hn] -> [b, np, sq, hn]
    q = query.permute(1, 2, 0, 3).float()
    kv_g = kv_gathered.float()  # [b, sq, topk, hn]

    # [b, np, sq, topk]
    scores = torch.einsum("bnsh,bskh->bnsk", q, kv_g) * softmax_scale

    # Mask invalid
    invalid_mask = (topk_indices < 0).unsqueeze(1)  # [b, 1, sq, topk]
    scores = scores.masked_fill(invalid_mask, float("-inf"))

    # --- Softmax with attention sink ---
    sink = attn_sink.view(1, np_, 1, 1).float()
    scores_max = scores.max(dim=-1, keepdim=True).values  # [b, np, sq, 1]
    scores_max = torch.max(scores_max, sink)

    exp_scores = torch.exp(scores - scores_max)  # [b, np, sq, topk]
    exp_sink = torch.exp(sink - scores_max)  # [1, np, 1, 1]

    sum_exp = exp_scores.sum(dim=-1, keepdim=True) + exp_sink
    attn_weights = exp_scores / sum_exp  # [b, np, sq, topk]

    # --- Weighted sum ---
    output = torch.einsum("bnsk,bskh->bnsh", attn_weights, kv_g)
    output = output.to(query.dtype)

    # [b, np, sq, hn] -> [sq, b, np, hn] -> [sq, b, np * hn]
    output = output.permute(2, 0, 1, 3).contiguous()
    output = output.reshape(sq, b, np_ * hn)
    return output


# ---------------------------------------------------------------------------
# Compressor
# ---------------------------------------------------------------------------


@dataclass
class CompressorSubmodules:
    """Submodule specs for CSA and HCA Compressor."""

    linear_wkv: Union[ModuleSpec, type] = None
    linear_wgate: Union[ModuleSpec, type] = None
    norm: Union[ModuleSpec, type] = None


class Compressor(MegatronModule):
    """Gated pooling compressor for CSA and HCA sparse attention.

    Compresses a sequence of tokens into a shorter sequence by pooling groups of
    ``compress_ratio`` tokens using learned gated weights.

    For ``compress_ratio == 4``, overlapping compression is used (``coff = 2``).
    For ``compress_ratio == 128``, non-overlapping compression is used (``coff = 1``).
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

        # keep to high precision
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

        Input shape:  [n_groups, ratio, b, coff * head_dim]
        Output shape: [n_groups, 2 * ratio, b, head_dim]
        """
        n_groups, ratio, b_dim, _ = tensor.size()
        d = self.head_dim
        new_tensor = tensor.new_full((n_groups, 2 * ratio, b_dim, d), fill_value)
        new_tensor[:, ratio:] = tensor[:, :, :, d:]
        new_tensor[1:, :ratio] = tensor[:-1, :, :, :d]
        return new_tensor

    def forward(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """Compress hidden states into shorter KV sequence.

        Args:
            x: [sq, b, hidden_size]

        Returns:
            compressed_kv [sq // ratio, b, head_dim] or None if too short.
        """
        nvtx_range_push("compressor")

        sq, b, _ = x.size()
        ratio = self.compress_ratio

        if sq < ratio:
            nvtx_range_pop("compressor")
            return None

        kv, _ = self.linear_wkv(x)  # [sq, b, coff * head_dim]
        score, _ = self.linear_wgate(x)  # [sq, b, coff * head_dim]

        cutoff = (sq // ratio) * ratio
        if cutoff < sq:
            kv = kv[:cutoff]
            score = score[:cutoff]

        n_compressed = cutoff // ratio

        # Reshape: [n_compressed, ratio, b, coff * head_dim]
        kv = kv.view(n_compressed, ratio, b, -1)
        score = score.view(n_compressed, ratio, b, -1)

        # APE: [ratio, coff * head_dim] -> [1, ratio, 1, coff * head_dim]
        score = score + self.ape.view(1, ratio, 1, -1)

        if self.overlap:
            kv = self._overlap_transform(kv, fill_value=0)
            score = self._overlap_transform(score, fill_value=float("-inf"))

        kv = (kv * torch.softmax(score, dim=1)).sum(dim=1)  # [n_compressed, b, head_dim]

        kv = self.norm(kv.to(x.dtype))

        kv = _apply_rope(
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
        return kv  # [n_compressed, b, head_dim]


# ---------------------------------------------------------------------------
# CSAIndexer
# ---------------------------------------------------------------------------


@dataclass
class CSAIndexerSubmodules:
    """Submodule specs for CSAIndexer."""

    linear_wq_b: Union[ModuleSpec, type] = None
    linear_weights_proj: Union[ModuleSpec, type] = None
    compressor: Union[ModuleSpec, type] = None


class CSAIndexer(MegatronModule):
    """Learned top-k retrieval over compressed positions for CSA sparse attention.

    Computes index scores to select the most relevant compressed KV positions for each
    query.  Reuses the scoring logic from ``DSAIndexer`` (einsum -> relu -> weight -> sum
    -> topk) and ``rotate_activation`` (Hadamard transform) from ``dsa.py``.
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

        # Q projection
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

        # Weights projection
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

        # Own compressor (smaller head_dim, with Hadamard rotation)
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
        nvtx_range_push("indexer_before_topk")

        sq, bsz, _ = x.size()

        # Q path
        q, _ = self.linear_wq_b(qr)  # [sq, b, n_heads * head_dim]
        q = q.reshape(sq, bsz, self.index_n_heads, self.index_head_dim)
        q = _apply_rope(
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

        # K path: own compressor
        k = self.compressor(x)  # [sq//ratio, b, index_head_dim]

        weights, _ = self.linear_weights_proj(x)  # [sq, b, n_heads]
        weights = weights * (self.index_n_heads**-0.5)

        nvtx_range_pop("indexer_before_topk")
        return q, k, weights

    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (index_scores, topk_indices)."""
        nvtx_range_push("indexer")
        assert packed_seq_params is None, "Packed sequence not supported for CSAIndexer"
        q, k, weights = self.forward_before_topk(x, qr, packed_seq_params)
        nvtx_range_push("indexer_qk_topk")
        effective_topk = min(self.index_topk, k.size(0))
        index_scores, topk_indices = fused_qk_topk_naive(q, k, weights, effective_topk, mask)
        nvtx_range_pop("indexer_qk_topk")
        nvtx_range_pop("indexer")
        return index_scores, topk_indices


# ---------------------------------------------------------------------------
# CompressedSparseAttention (core attention)
# ---------------------------------------------------------------------------


@dataclass
class CompressedSparseAttentionSubmodules:
    """Submodule specs for CompressedSparseAttention."""

    compressor: Union[ModuleSpec, type] = None
    indexer: Union[ModuleSpec, type] = None


class CompressedSparseAttention(MegatronModule):
    """Sparse core attention for CompressedSparseAttention.

    Combines sliding window attention with compressed KV attention.  The spec always
    provides compressor and indexer submodule specs; this ``__init__`` inspects
    ``config.csa_compress_ratios[layer_idx]`` and conditionally builds them:

    * ``ratio == 0``:  window-only (compressor and indexer NOT built)
    * ``ratio == 4``:  window + 4x compressed + learned Indexer (both built)
    * ``ratio == 128``: window + 128x compressed, attend to all (compressor built only)
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

        self.force_unfused_dsa = getattr(config, 'force_unfused_dsa', True)

        # Learnable attention sink per head
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

        # Conditionally build Indexer (ratio == 4)
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
        """Forward pass for CompressedSparseAttention.

        Args:
            query:  [sq, b, np, v_head_dim]
            key:    [sq, b, 1, v_head_dim]  (single-head MQA; head dim squeezed internally)
            value:  unused (key == value in MQA)
            attention_mask: attention mask (may be None for causal).
            x:      [sq, b, hidden_size]  original hidden states.
            qr:     [sq, b, q_lora_rank]  compressed query representation.

        Returns:
            output: [sq, b, np * v_head_dim]
        """
        nvtx_range_push("compressed_sparse_attn")
        assert (
            packed_seq_params is None
        ), "Packed sequence not supported for CompressedSparseAttention"

        sq, b, np, hn = query.size()

        # --- Step 1: Prepare single-head KV (squeeze singleton head dim) ---
        kv = key.squeeze(-2)  # [sq, b, 1, v_head_dim] -> [sq, b, v_head_dim]

        # --- Step 2: Compression ---
        if self.compressor is not None and self.compress_ratio > 1:
            compressed_kv = self.compressor(x)  # [n_compressed, b, v_head_dim]
            if compressed_kv is not None:
                kv_full = torch.cat([kv, compressed_kv], dim=0)
                n_compressed = compressed_kv.size(0)
            else:
                kv_full = kv
                n_compressed = 0
        else:
            kv_full = kv
            n_compressed = 0

        offset = sq  # compressed indices start after original positions

        # --- Step 3: Window indices ---
        window_idxs = get_window_topk_idxs(self.window_size, b, sq, query.device)

        # --- Step 4: Compressed indices ---
        indexer_loss = None

        if self.force_unfused_dsa:
            if self.compress_ratio > 1 and n_compressed > 0:
                nvtx_range_push("compressed_indices")
                if self.indexer is not None:
                    x_det = x.detach()
                    qr_det = qr.detach()

                    causal_mask = (
                        torch.arange(n_compressed, device=x.device).unsqueeze(0).expand(sq, -1)
                    )
                    positions = torch.arange(1, sq + 1, device=x.device).unsqueeze(1)
                    causal_mask = (
                        torch.where(
                            causal_mask >= positions // self.compress_ratio, float("-inf"), 0.0
                        )
                        .unsqueeze(0)
                        .expand(b, -1, -1)
                    )  # [b, sq, n_compressed]

                    if self.training and torch.is_grad_enabled():
                        q_indexer, k_indexer, weights_indexer = self.indexer.forward_before_topk(
                            x_det, qr_det, packed_seq_params
                        )
                        indexer_loss_coeff = getattr(self.config, 'dsa_indexer_loss_coeff', 0.0)
                        # compressed_kv is [n, b, hn]; expand to [n, b, np, hn] for loss
                        key_for_loss = compressed_kv.unsqueeze(2).expand(-1, -1, np, -1)
                        # ``FusedDSAIndexerLoss`` does not accept a separate
                        # indexer_softmax_scale; apply it here via the
                        # weights-scaling trick so the effective weights match
                        # the pre-scale-split behaviour.
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
                    compress_topk_idxs = get_compress_topk_idxs(
                        self.compress_ratio, b, sq, offset, query.device
                    )

                topk_idxs = torch.cat([window_idxs, compress_topk_idxs], dim=-1)
                nvtx_range_pop("compressed_indices")
            else:
                topk_idxs = window_idxs

            topk_idxs = topk_idxs.int()

            # --- Step 5: Sparse attention ---
            nvtx_range_push("sparse_attn_kernel")
            output = unfused_compressed_sparse_attn(
                query, kv_full, self.attn_sink.float(), topk_idxs, self.softmax_scale
            )
            nvtx_range_pop("sparse_attn_kernel")

        else:
            raise ValueError("Fused path is not supported for CompressedSparseAttention")

        # --- Step 6: Attach indexer loss ---
        if indexer_loss is not None and self.training and torch.is_grad_enabled():
            output = DSAIndexerLossAutoScaler.apply(output, indexer_loss)

        nvtx_range_pop("compressed_sparse_attn")
        return output
