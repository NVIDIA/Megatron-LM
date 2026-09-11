# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Native V4.1 CSA2 training math with explicit, forward-local cross-layer sharing.

This uses the existing DSv4 attention wrapper and sparse attention reference. Shared graph
tensors belong to one stack forward and are passed explicitly; modules retain no activation
state or inference caches.
"""

from copy import copy
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant.csa import (
    CompressedSparseAttentionSubmodules,
    CompressorSubmodules,
    _apply_rope,
    _compute_unfused_csa_non_compressed_lse,
    get_window_topk_idxs,
    get_window_topk_idxs_thd,
    unfused_compressed_sparse_attn,
)
from megatron.core.transformer.experimental_attention_variant.csa_utils.thd_utils import (
    CSA2THDCompressionLayout,
    CSA2THDLayout,
    build_csa2_thd_layout,
)
from megatron.core.transformer.experimental_attention_variant.dsa import (
    DSAIndexerLossAutoScaler,
    DSAIndexerLossLoggingHelper,
    compute_dsa_indexer_loss,
)
from megatron.core.transformer.module import MegatronModule, mark_keep_in_fp32
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.core.utils import get_pg_size


@dataclass
class CSA2State:
    """Shared attention tensors for one full-sequence stack forward.

    Source layer IDs are zero-based, matching the configuration. ``global_indices`` uses
    compressed positions (per-batch for SBHD, physical packed rows for THD, and -1 for
    unavailable positions), never offsets into a consumer's concatenated local/global KV
    tensor. Float tensors retain their autograd graph,
    so every consumer contributes to its Full owner's gradient. Create a fresh state for each
    forward, including when another microbatch's backward is still outstanding.
    """

    global_kv: torch.Tensor | None = None
    indexer_k: torch.Tensor | None = None
    global_indices: torch.Tensor | None = None
    candidates: torch.Tensor | None = None
    kv_source_layer: int | None = None
    index_source_layer: int | None = None
    candidate_source_layer: int | None = None
    sequence_length: int | None = None
    batch_size: int | None = None
    device: torch.device | None = None
    dtype: torch.dtype | None = None
    last_layer: int | None = None
    thd_layout: CSA2THDLayout | None = None
    compressed_layout: CSA2THDCompressionLayout | None = None

    def validate_forward(
        self, layer_idx: int, query: torch.Tensor, *, thd_layout: CSA2THDLayout | None = None
    ) -> None:
        """Reject reuse across forwards or incompatible sequence/batch layouts."""
        if self.last_layer is not None and layer_idx <= self.last_layer:
            raise ValueError(
                "CSA2State layers must run in increasing order; create a fresh CSA2State "
                "for each stack forward."
            )
        if thd_layout is not None:
            if query.ndim != 3 or query.shape[0] != thd_layout.total_tokens:
                raise ValueError(
                    "CSA2 THD queries must have shape [total_tokens, heads, head_dim]."
                )
            if query.device != thd_layout.valid_tokens.device:
                raise ValueError("CSA2 THD queries and layout must be on the same device.")
            sequence_length, batch_size = query.shape[0], 1
        else:
            sequence_length, batch_size = query.shape[:2]
        if self.sequence_length is None:
            self.sequence_length, self.batch_size = sequence_length, batch_size
            self.device, self.dtype = query.device, query.dtype
            self.thd_layout = thd_layout
        elif (sequence_length, batch_size, query.device, query.dtype) != (
            self.sequence_length,
            self.batch_size,
            self.device,
            self.dtype,
        ):
            raise ValueError(
                "CSA2State sequence length, batch size, device and dtype must match this "
                "forward; create a fresh CSA2State for each stack forward."
            )
        elif (self.thd_layout is None) != (thd_layout is None):
            raise ValueError("CSA2State cannot mix packed and unpacked layouts in one forward.")
        elif thd_layout is not None and self.thd_layout is not thd_layout:
            self.thd_layout.validate_layout(thd_layout)


def apply_csa2_thd_rope(
    x: torch.Tensor,
    rotary_pos_emb: nn.Module,
    config: MLATransformerConfig,
    layout: CSA2THDLayout | CSA2THDCompressionLayout,
    cp_group: torch.distributed.ProcessGroup,
) -> torch.Tensor:
    """Apply DSv4's segmented RoPE to token or compressed rows, excluding padding.

    Token positions restart at zero in every sequence; compressed row ``j`` uses
    position ``ratio * j``. Physical prefixes describe any reserved padding rows.
    Masking before rotation also gives fused RoPE private storage, so the main and
    indexer branches cannot overwrite their shared pre-RoPE latent.
    """
    if get_pg_size(cp_group) != 1:
        raise NotImplementedError("CSA2 THD RoPE currently requires CP=1.")
    compressed = isinstance(layout, CSA2THDCompressionLayout)
    valid = layout.valid_groups if compressed else layout.valid_tokens
    if x.ndim not in (3, 4) or x.shape[0] != valid.shape[0] or (x.ndim == 4 and x.shape[1] != 1):
        raise ValueError("CSA2 THD RoPE expects one row per token or compressed capacity slot.")
    if x.device != valid.device:
        raise ValueError("CSA2 THD RoPE input and layout must be on the same device.")
    mask = ~valid.reshape(-1, *([1] * (x.ndim - 1)))
    x = x.masked_fill(mask, 0)
    # All sequences can be shorter than the compression ratio. In that case
    # even a nonempty static capacity has no usable frequency-table entries.
    if x.shape[0] == 0 or layout.max_seqlen == 0 or layout.cu_seqlens_padded.numel() < 2:
        return x
    ratio = layout.ratio if compressed else 1
    pos_dim = config.qk_pos_emb_head_dim
    output = _apply_rope(
        x,
        nope_dim=x.shape[-1] - pos_dim,
        pos_dim=pos_dim,
        rotary_pos_emb_module=rotary_pos_emb,
        config=config,
        rotary_seq_len=0,
        ratio=ratio,
        cp_group=cp_group,
        cu_seqlens=layout.cu_seqlens_padded,
        max_seqlen_rope=layout.max_seqlen * ratio,
    )
    return output.masked_fill(mask, 0)


def select_candidate_blocks(
    logits: torch.Tensor, compress_lens: torch.Tensor | int, topk_blocks: int, block_size: int
) -> torch.Tensor:
    """Select block-max candidates, always including the newest visible position's block.

    ``logits`` has shape ``[..., global positions]`` and is already causally masked.
    ``compress_lens`` broadcasts against its leading dimensions with a trailing singleton
    dimension. The result is a position mask; the caller still applies the causal score mask
    because a selected block can include positions after the query.
    """
    if topk_blocks <= 0 or block_size <= 0:
        raise ValueError("CSA2 candidate block count and size must be positive.")
    width = logits.shape[-1]
    if width == 0:
        return torch.zeros_like(logits, dtype=torch.bool)
    scores = F.pad(logits, (0, -width % block_size), value=-torch.inf)
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = scores.shape[-1]
    newest_block = (compress_lens - 1) // block_size
    scores = scores.masked_fill(
        torch.arange(num_blocks, device=logits.device) == newest_block, torch.inf
    )
    # ReLU index scores often tie at zero. Break ties by the earlier local
    # block so changing packed capacity cannot change which real blocks win.
    top_indices = scores.argsort(dim=-1, descending=True, stable=True)[
        ..., : min(topk_blocks, num_blocks)
    ]
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter(
        -1, top_indices, scores.gather(-1, top_indices) > -torch.inf
    )
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


def select_candidate_blocks_thd(
    scores: torch.Tensor,
    layout: CSA2THDLayout,
    compressed_layout: CSA2THDCompressionLayout,
    topk_blocks: int,
    block_size: int,
) -> torch.Tensor:
    """Apply CSA2 block selection in each packed sequence's local group coordinates."""
    if scores.shape[-1] == 0:
        return torch.zeros_like(scores, dtype=torch.bool)
    sequence_ids = layout.sequence_ids.clamp_min(0)
    starts = compressed_layout.cu_seqlens_padded[sequence_ids].long()
    ends = compressed_layout.cu_seqlens_padded[sequence_ids + 1].long()
    local_positions = torch.arange(compressed_layout.max_seqlen, device=scores.device)
    physical_positions = starts[:, None] + local_positions
    assigned = physical_positions < ends[:, None]
    safe_positions = physical_positions.clamp(max=scores.shape[-1] - 1)
    local_scores = scores.gather(1, safe_positions).masked_fill(~assigned, -torch.inf)
    # Force the latest visible group's block, excluding the physical padding
    # after each logical sequence. Rows without any visible group stay empty.
    visible = torch.where(local_scores.isfinite(), local_positions + 1, 0).amax(
        dim=-1, keepdim=True
    )
    keep = select_candidate_blocks(local_scores, visible, topk_blocks, block_size)
    keep = keep & local_scores.isfinite()
    # Clamping unused local capacity can repeat a physical column; accumulate
    # instead of letting a padded False overwrite an earlier selected True.
    return (
        torch.zeros_like(scores, dtype=torch.int32).scatter_add_(1, safe_positions, keep.int()) > 0
    )


class CSA2Compressor(MegatronModule):
    """Produce pre-RoPE global latents from complete, non-overlapping token groups."""

    def __init__(
        self,
        config: MLATransformerConfig,
        submodules: CompressorSubmodules,
        compress_ratio: int,
        pg_collection: ProcessGroupCollection,
    ) -> None:
        super().__init__(config=config)
        if compress_ratio not in (1, 2):
            raise ValueError("CSA2 compression requires ratio 1 or 2.")
        self.compress_ratio = compress_ratio
        self.tp_group = pg_collection.tp
        projection_config = copy(config)
        if compress_ratio == 2:
            projection_config.params_dtype = torch.float32
            projection_config.bf16 = False
        linear_kwargs = dict(
            config=projection_config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )
        self.linear_wkv = build_module(
            submodules.linear_wkv, config.hidden_size, config.v_head_dim, **linear_kwargs
        )
        self.linear_wgate = None
        if compress_ratio == 2:
            self.linear_wgate = build_module(
                submodules.linear_wgate, config.hidden_size, config.v_head_dim, **linear_kwargs
            )
            mark_keep_in_fp32(self.linear_wkv.weight)
            mark_keep_in_fp32(self.linear_wgate.weight)
        self.norm = build_module(
            submodules.norm,
            config=config,
            hidden_size=config.v_head_dim,
            eps=config.attention_latent_norm_epsilon,
        )

    def forward(
        self,
        x: torch.Tensor,
        packed_seq_params: PackedSeqParams | None = None,
        *,
        thd_layout: CSA2THDLayout | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, CSA2THDCompressionLayout]:
        """Compress SBHD input or return THD latents with their physical layout.

        Packed inputs keep the usual ``[total_tokens, 1, hidden]`` shape. Their
        result uses DSv4's static compressed capacity, including zero-valued
        padding slots; ``layout.valid_groups`` identifies the complete real groups.
        A caller sharing an existing layout can pass ``thd_layout`` directly.
        """
        if packed_seq_params is not None or thd_layout is not None:
            if x.ndim != 3 or x.shape[1] != 1:
                raise ValueError("CSA2 THD compressor expects [total_tokens, 1, hidden].")
            if thd_layout is None:
                thd_layout = build_csa2_thd_layout(packed_seq_params, x.shape[0])
            elif packed_seq_params is not None:
                thd_layout.validate_compatible(packed_seq_params, x.shape[0])
            if x.shape[0] != thd_layout.total_tokens or x.device != thd_layout.valid_tokens.device:
                raise ValueError(
                    "CSA2 THD compressor input must match the layout's rows and device."
                )
            return self._forward_thd(x, thd_layout.for_compression(self.compress_ratio))
        return self._forward_sbhd(x)

    def _forward_sbhd(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``[sequence, batch, hidden]`` to ``[sequence // ratio, batch, head_dim]``."""
        if self.compress_ratio == 1:
            latent, _ = self.linear_wkv(x)
        else:
            # An incomplete final group remains available through the sliding window only.
            complete_length = x.shape[0] // self.compress_ratio * self.compress_ratio
            xf = x[:complete_length].float()
            latent, _ = self.linear_wkv(xf)
            gate, _ = self.linear_wgate(xf)
            shape = (complete_length // self.compress_ratio, self.compress_ratio, *latent.shape[1:])
            latent = (latent.reshape(shape) * gate.reshape(shape).softmax(dim=1)).sum(dim=1)
        latent = latent.to(x.dtype)
        if latent.shape[0] == 0:
            # TE RMSNorm cannot normalize zero rows. Keep its zero weight gradient in
            # the graph when the sequence has no complete compression group.
            return latent * self.norm.weight
        return self.norm(latent)

    def _forward_thd(
        self, x: torch.Tensor, layout: CSA2THDCompressionLayout
    ) -> tuple[torch.Tensor, CSA2THDCompressionLayout]:
        """Gather groups within physical segments and pool complete real tokens."""
        ratio = self.compress_ratio
        grouped = x[layout.source_indices]
        # Neutralize invalid groups before GEMM, softmax and RMSNorm. Masking only
        # their final output would still allow NaN/Inf padding to poison backward.
        grouped = grouped.masked_fill(~layout.valid_groups[:, None, None, None], 0)
        inputs = grouped.reshape(layout.capacity * ratio, 1, x.shape[-1])
        if ratio == 2:
            inputs = inputs.float()
        latent, _ = self.linear_wkv(inputs)
        if ratio == 2:
            gate, _ = self.linear_wgate(inputs)
            shape = (layout.capacity, ratio, *latent.shape[1:])
            latent = (latent.reshape(shape) * gate.reshape(shape).softmax(dim=1)).sum(dim=1)
        latent = latent.to(x.dtype)
        if layout.capacity == 0:
            latent = latent * self.norm.weight
        else:
            latent = self.norm(latent)
        return latent.masked_fill(~layout.valid_groups[:, None, None], 0), layout

    def backward_dw(self) -> None:
        """Follow the existing DSv4 linear gradient interface."""
        self.linear_wkv.backward_dw()
        if self.linear_wgate is not None:
            self.linear_wgate.backward_dw()


@dataclass
class CSA2IndexerSubmodules:
    """Projection and normalization specs for Full and Reindex CSA2 indexers."""

    linear_wq_b: ModuleSpec | type
    linear_wk: ModuleSpec | type
    k_norm: ModuleSpec | type
    linear_weights_proj: ModuleSpec | type


class CSA2Indexer(MegatronModule):
    """Select global positions using keys derived from the pre-RoPE compressor latent."""

    def __init__(
        self,
        config: MLATransformerConfig,
        submodules: CSA2IndexerSubmodules,
        compress_ratio: int,
        pg_collection: ProcessGroupCollection,
        owns_k: bool = True,
    ) -> None:
        super().__init__(config=config)
        self.compress_ratio = compress_ratio
        self.tp_group = pg_collection.tp
        self.cp_group = pg_collection.cp
        self.n_heads = config.dsa_indexer_n_heads
        self.head_dim = config.dsa_indexer_head_dim
        self.topk = config.dsa_indexer_topk
        self.owns_k = owns_k
        linear_kwargs = dict(
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )
        self.linear_wq_b = build_module(
            submodules.linear_wq_b,
            config.q_lora_rank,
            self.n_heads * self.head_dim,
            **linear_kwargs,
        )
        self.linear_weights_proj = build_module(
            submodules.linear_weights_proj, config.hidden_size, self.n_heads, **linear_kwargs
        )
        self.linear_wk = self.k_norm = None
        if owns_k:
            self.linear_wk = build_module(
                submodules.linear_wk, config.v_head_dim, self.head_dim, **linear_kwargs
            )
            self.k_norm = build_module(
                submodules.k_norm,
                config=config,
                hidden_size=self.head_dim,
                eps=config.attention_latent_norm_epsilon,
            )

    def project_keys(
        self,
        latent: torch.Tensor,
        rotary_pos_emb: nn.Module,
        *,
        thd_layout: CSA2THDCompressionLayout | None = None,
    ) -> torch.Tensor:
        """Produce graph-connected rotated index keys at a Full owner."""
        if not self.owns_k:
            raise ValueError("A CSA2 Reindex layer must consume its Full owner's indexer_k.")
        if thd_layout is not None:
            if (
                thd_layout.ratio != self.compress_ratio
                or latent.ndim != 3
                or latent.shape[:2] != (thd_layout.capacity, 1)
                or latent.device != thd_layout.valid_groups.device
            ):
                raise ValueError("CSA2 indexer latent must match its THD compression layout.")
            latent = latent.masked_fill(~thd_layout.valid_groups[:, None, None], 0)
        k, _ = self.linear_wk(latent)
        # An incomplete first compression group produces no key rows.
        k = self.k_norm(k) if k.shape[0] else k * self.k_norm.weight
        if thd_layout is not None:
            return apply_csa2_thd_rope(k, rotary_pos_emb, self.config, thd_layout, self.cp_group)
        pos_dim = self.config.qk_pos_emb_head_dim
        return _apply_rope(
            k,
            nope_dim=self.head_dim - pos_dim,
            pos_dim=pos_dim,
            rotary_pos_emb_module=rotary_pos_emb,
            config=self.config,
            rotary_seq_len=k.shape[0],
            ratio=self.compress_ratio,
            cp_group=self.cp_group,
        )

    def forward_before_topk(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        latent: torch.Tensor | None,
        rotary_pos_emb: nn.Module,
        *,
        indexer_k: torch.Tensor | None = None,
        candidates: torch.Tensor | None = None,
        thd_layout: CSA2THDLayout | None = None,
        compressed_layout: CSA2THDCompressionLayout | None = None,
    ) -> torch.Tensor:
        """Return causal scores, SBHD ``[B, S, C]`` or THD ``[T, C]``.

        The discrete top-k has no LM-loss gradient. The indexer auxiliary loss consumes
        these scores without retaining a graph in module state. Full layers
        may pass a pre-RoPE latent as before; Reindex layers pass ``latent=None`` and the Full
        owner's ``indexer_k``. Supplying shared keys preserves their graph for auxiliary losses.
        """
        if thd_layout is not None:
            if qr.ndim == 2:
                qr = qr.unsqueeze(1)
            if compressed_layout is None:
                compressed_layout = thd_layout.for_compression(self.compress_ratio)
            if compressed_layout.ratio != self.compress_ratio:
                raise ValueError("CSA2 indexer compression ratio must match its THD layout.")
            if x.shape[:2] != (thd_layout.total_tokens, 1) or qr.shape[:2] != x.shape[:2]:
                raise ValueError("CSA2 THD indexer inputs must have shape [total_tokens, 1, dim].")
            invalid = ~thd_layout.valid_tokens[:, None, None]
            x = x.masked_fill(invalid, 0)
            qr = qr.masked_fill(invalid, 0)
        elif compressed_layout is not None:
            raise ValueError("CSA2 compressed THD layout requires a token layout.")
        if indexer_k is None:
            if latent is None:
                raise ValueError("CSA2 indexer scores require latent or shared indexer_k.")
            indexer_k = self.project_keys(latent, rotary_pos_emb, thd_layout=compressed_layout)
        elif latent is not None:
            raise ValueError("Provide either latent or indexer_k, not both.")
        q, _ = self.linear_wq_b(qr)
        q = q.reshape(*q.shape[:-1], self.n_heads, self.head_dim)
        if thd_layout is not None:
            if indexer_k.shape[:2] != (compressed_layout.capacity, 1):
                raise ValueError("CSA2 shared indexer keys must match the THD compression layout.")
            indexer_k = indexer_k.masked_fill(~compressed_layout.valid_groups[:, None, None], 0)
            q = apply_csa2_thd_rope(q, rotary_pos_emb, self.config, thd_layout, self.cp_group)
            weights, _ = self.linear_weights_proj(x)
            weights = weights.float().squeeze(1) * (self.head_dim**-0.5 * self.n_heads**-0.5)
            scores = torch.einsum(
                "thd,cd->thc", q.squeeze(1).float(), indexer_k.squeeze(1).float()
            ).relu()
            scores = (scores * weights.unsqueeze(-1)).sum(dim=1)
            valid = (
                thd_layout.valid_tokens[:, None]
                & compressed_layout.valid_groups[None, :]
                & (thd_layout.sequence_ids[:, None] == compressed_layout.sequence_ids[None, :])
                & (
                    compressed_layout.position_ids[None, :] + self.compress_ratio - 1
                    <= thd_layout.position_ids[:, None]
                )
            )
            if candidates is not None:
                valid = valid & candidates
            return scores.masked_fill(~valid, -torch.inf)
        pos_dim = self.config.qk_pos_emb_head_dim
        rope_kwargs = dict(
            nope_dim=self.head_dim - pos_dim,
            pos_dim=pos_dim,
            rotary_pos_emb_module=rotary_pos_emb,
            config=self.config,
            cp_group=self.cp_group,
        )
        q = _apply_rope(q, rotary_seq_len=q.shape[0], **rope_kwargs)
        weights, _ = self.linear_weights_proj(x)
        weights = weights.float() * (self.head_dim**-0.5 * self.n_heads**-0.5)
        scores = torch.einsum("sbhd,tbd->bsht", q.float(), indexer_k.float()).relu()
        scores = (scores * weights.permute(1, 0, 2).unsqueeze(-1)).sum(dim=2)
        visible = torch.arange(1, x.shape[0] + 1, device=x.device) // self.compress_ratio
        causal = torch.arange(indexer_k.shape[0], device=x.device)[None, :] < visible[:, None]
        scores = scores.masked_fill(~causal, -torch.inf)
        if candidates is not None:
            scores = scores.masked_fill(~candidates, -torch.inf)
        return scores

    def select_indices(self, scores: torch.Tensor) -> torch.Tensor:
        """Return sorted logical positions, with -1 for masked or unavailable entries."""
        # A stable tie order preserves selection across SBHD/THD and different
        # padding capacities. Physical order within each sequence is local order.
        indices = scores.argsort(dim=-1, descending=True, stable=True)[
            ..., : min(self.topk, scores.shape[-1])
        ]
        indices = indices.sort(dim=-1).values
        return indices.masked_fill(~scores.gather(-1, indices).isfinite(), -1).int()

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        latent: torch.Tensor | None,
        rotary_pos_emb: nn.Module,
        *,
        indexer_k: torch.Tensor | None = None,
        candidates: torch.Tensor | None = None,
        thd_layout: CSA2THDLayout | None = None,
        compressed_layout: CSA2THDCompressionLayout | None = None,
    ) -> torch.Tensor:
        """Return sorted global-position indices, with ``-1`` for unavailable positions."""
        scores = self.forward_before_topk(
            x,
            qr,
            latent,
            rotary_pos_emb,
            indexer_k=indexer_k,
            candidates=candidates,
            thd_layout=thd_layout,
            compressed_layout=compressed_layout,
        )
        return self.select_indices(scores)

    def backward_dw(self) -> None:
        """Follow the existing DSv4 linear gradient interface."""
        for linear in (self.linear_wq_b, self.linear_wk, self.linear_weights_proj):
            if linear is not None:
                linear.backward_dw()


class CompressedSparseAttention2(MegatronModule):
    """Native SWA, Full, Reindex and Reuse behind the existing DSv4 attention interface."""

    def __init__(
        self,
        config: MLATransformerConfig,
        submodules: CompressedSparseAttentionSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        pg_collection: ProcessGroupCollection,
        attention_dropout: float | None = None,
        softmax_scale: float | None = None,
        k_channels: int | None = None,
        v_channels: int | None = None,
        cp_comm_type: str | None = None,
        rotary_pos_emb: nn.Module | None = None,
        compress_ratio: int | None = None,
        is_mtp_layer: bool = False,
        name: str | None = None,
    ) -> None:
        super().__init__(config=config)
        if config.dsv4_version != "v4.1" or attn_mask_type != AttnMaskType.causal:
            raise ValueError("CSA2 requires V4.1 causal attention.")
        if attention_type != "self" or is_mtp_layer:
            raise NotImplementedError("CSA2 currently supports backbone self-attention only.")
        if get_pg_size(pg_collection.tp) != 1 or get_pg_size(pg_collection.cp) != 1:
            raise NotImplementedError("Native CSA2 currently requires TP=CP=1.")
        if (attention_dropout or config.attention_dropout) != 0:
            raise NotImplementedError("Native CSA2 does not implement attention dropout.")
        layer_idx = layer_number - 1
        ratio = config.csa_compress_ratios[layer_idx]
        if compress_ratio is not None and compress_ratio != ratio:
            raise ValueError("CSA2 compress_ratio must match the configured layer ratio.")
        self.layer_idx = layer_idx
        self.is_kv_source = layer_idx in config.csa2_kv_source_layers
        self.is_index_source = layer_idx in config.csa2_index_source_layers
        self.kv_source_layer = (
            max(source for source in config.csa2_kv_source_layers if source <= layer_idx)
            if ratio
            else None
        )
        self.index_source_layer = (
            max(source for source in config.csa2_index_source_layers if source <= layer_idx)
            if ratio
            else None
        )
        self.is_candidate_source = layer_idx == config.csa2_candidate_source_layer
        self.uses_candidates = (
            config.csa2_candidate_source_layer is not None
            and layer_idx > config.csa2_candidate_source_layer
        )
        self.compress_ratio = ratio
        self.pg_collection = pg_collection
        self.tp_group = pg_collection.tp
        self.cp_group = pg_collection.cp
        self.rotary_pos_emb = rotary_pos_emb
        self.softmax_scale = config.v_head_dim**-0.5 if softmax_scale is None else softmax_scale
        device = "cpu" if config.use_cpu_initialization else torch.cuda.current_device()
        self.attn_sink = mark_keep_in_fp32(
            nn.Parameter(
                torch.zeros(config.num_attention_heads, dtype=torch.float32, device=device)
            )
        )
        self.compressor = self.indexer = None
        if self.is_kv_source:
            kwargs = dict(config=config, compress_ratio=ratio, pg_collection=pg_collection)
            self.compressor = build_module(submodules.compressor, **kwargs)
        if self.is_index_source:
            self.indexer = build_module(
                submodules.indexer,
                config=config,
                compress_ratio=ratio,
                pg_collection=pg_collection,
                owns_k=self.is_kv_source,
            )

    def _shared_global_attention(
        self, x: torch.Tensor, qr: torch.Tensor, state: CSA2State, *, use_indexer_loss: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Publish or consume graph tensors and indices in compressed KV coordinates."""
        thd_layout = state.thd_layout
        if self.is_kv_source:
            if thd_layout is None:
                latent = self.compressor(x)
                state.compressed_layout = None
            else:
                latent, state.compressed_layout = self.compressor(x, thd_layout=thd_layout)
            # The auxiliary objective trains the indexer, not the main compressor.
            # Detach BEFORE projection so Full and Reindex losses still reach the
            # owning indexer's K projection and normalization through the shared keys.
            indexer_latent = latent.detach() if use_indexer_loss else latent
            state.indexer_k = self.indexer.project_keys(
                indexer_latent, self.rotary_pos_emb, thd_layout=state.compressed_layout
            )
            if thd_layout is None:
                pos_dim = self.config.qk_pos_emb_head_dim
                state.global_kv = _apply_rope(
                    latent,
                    self.config.v_head_dim - pos_dim,
                    pos_dim,
                    self.rotary_pos_emb,
                    self.config,
                    latent.shape[0],
                    ratio=self.compress_ratio,
                    cp_group=self.cp_group,
                )
            else:
                state.global_kv = apply_csa2_thd_rope(
                    latent, self.rotary_pos_emb, self.config, state.compressed_layout, self.cp_group
                )
            state.kv_source_layer = self.layer_idx
            state.global_indices = state.candidates = None
            state.index_source_layer = state.candidate_source_layer = None
        elif (
            state.kv_source_layer != self.kv_source_layer
            or state.global_kv is None
            or state.indexer_k is None
        ):
            raise ValueError(
                f"CSA2 layer {self.layer_idx} requires KV and indexer K from Full layer "
                f"{self.kv_source_layer} in the same forward's CSA2State."
            )

        scores = None
        if thd_layout is not None and (
            state.compressed_layout is None or state.compressed_layout.ratio != self.compress_ratio
        ):
            raise ValueError("CSA2 shared compressed layout must match this layer's ratio.")
        if self.is_index_source:
            candidates = None
            if self.uses_candidates:
                if (
                    state.candidate_source_layer != self.config.csa2_candidate_source_layer
                    or state.candidates is None
                ):
                    raise ValueError(
                        f"CSA2 layer {self.layer_idx} requires candidates from layer "
                        f"{self.config.csa2_candidate_source_layer} in the same CSA2State."
                    )
                candidates = state.candidates
            with torch.set_grad_enabled(use_indexer_loss):
                scores = self.indexer.forward_before_topk(
                    x.detach(),
                    qr.detach(),
                    None,
                    self.rotary_pos_emb,
                    indexer_k=state.indexer_k,
                    candidates=candidates,
                    thd_layout=thd_layout,
                    compressed_layout=state.compressed_layout,
                )
            with torch.no_grad():
                if self.is_candidate_source:
                    if thd_layout is None:
                        visible = (
                            torch.arange(1, x.shape[0] + 1, device=x.device) // self.compress_ratio
                        ).unsqueeze(-1)
                        state.candidates = select_candidate_blocks(
                            scores,
                            visible,
                            self.config.csa2_candidate_topk_blocks,
                            self.config.csa2_candidate_block_size,
                        )
                    else:
                        state.candidates = select_candidate_blocks_thd(
                            scores,
                            thd_layout,
                            state.compressed_layout,
                            self.config.csa2_candidate_topk_blocks,
                            self.config.csa2_candidate_block_size,
                        )
                    state.candidate_source_layer = self.layer_idx
                # The candidate source itself selects over all causal positions. Only later
                # Reindex layers apply its candidate mask, matching the reference ordering.
                state.global_indices = self.indexer.select_indices(scores)
            state.index_source_layer = self.layer_idx
        elif state.index_source_layer != self.index_source_layer or state.global_indices is None:
            raise ValueError(
                f"CSA2 Reuse layer {self.layer_idx} requires indices from layer "
                f"{self.index_source_layer} in the same forward's CSA2State."
            )
        return state.global_kv, state.global_indices, scores if use_indexer_loss else None

    def _compute_indexer_loss(
        self,
        query: torch.Tensor,
        local_kv: torch.Tensor,
        global_kv: torch.Tensor,
        window_indices: torch.Tensor,
        global_indices: torch.Tensor,
        scores: torch.Tensor,
        *,
        thd_layout: CSA2THDLayout | None = None,
    ) -> torch.Tensor:
        """Apply the existing DSv4 KL objective to this Full/Reindex layer's teacher.

        Dense loss covers all causal global keys, restricted by the candidate mask
        on later Reindex layers. Sparse loss further restricts both distributions to
        this layer's top-k. The teacher includes SWA and sink mass before summing
        heads and renormalizing over global keys. Reuse layers add no objective.

        As in DSv4, this objective covers input tokens independently of the LM label
        mask. Token reduction and backward scaling use the existing DSA interfaces;
        the number of indexer layers only normalizes the reported metric.
        """
        if scores.shape[-1] == 0 or (thd_layout is not None and query.shape[0] == 0):
            # No complete compression group: preserve zero gradients for the indexer.
            return scores.sum() * 0
        non_compressed_lse = _compute_unfused_csa_non_compressed_lse(
            query, local_kv, self.attn_sink, window_indices, self.softmax_scale
        )
        mask = torch.zeros_like(scores).masked_fill(torch.isneginf(scores), -torch.inf)
        num_heads = query.shape[1] if thd_layout is not None else query.shape[2]
        query_valid_rows = None
        if thd_layout is not None:
            scores = scores.unsqueeze(0)
            global_indices = global_indices.unsqueeze(0)
            query_valid_rows = thd_layout.valid_tokens
        return compute_dsa_indexer_loss(
            # The shared helper adds masks in place; keep the scoring/selection tensor intact.
            scores.clone(),
            global_indices,
            query.detach(),
            global_kv.detach().unsqueeze(2).expand(-1, -1, num_heads, -1),
            self.softmax_scale,
            self.config.dsa_indexer_loss_coeff,
            self.config.dsa_indexer_use_sparse_loss,
            self.pg_collection,
            mask=mask,
            query_valid_rows=query_valid_rows,
            calculate_per_token_loss=self.config.calculate_per_token_loss,
            non_compressed_lse=non_compressed_lse,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        *,
        packed_seq_params: PackedSeqParams | None = None,
        x: torch.Tensor,
        qr: torch.Tensor,
        boundary_hidden: torch.Tensor | None = None,
        boundary_kv: torch.Tensor | None = None,
        csa2_state: CSA2State | None = None,
        thd_layout: CSA2THDLayout | None = None,
    ) -> torch.Tensor:
        """Attend jointly to causal window and selected global latents using one softmax."""
        if boundary_hidden is not None or boundary_kv is not None:
            raise NotImplementedError("Native CSA2 currently requires CP=1.")
        is_thd = packed_seq_params is not None
        if is_thd:
            if packed_seq_params.qkv_format != "thd":
                raise NotImplementedError("CSA2 packed sequences require qkv_format='thd'.")
            if query.ndim != 3 or key.shape != (query.shape[0], 1, query.shape[-1]):
                raise ValueError("CSA2 THD expects query [T, heads, dim] and key [T, 1, dim].")
            if thd_layout is None:
                thd_layout = build_csa2_thd_layout(packed_seq_params, query.shape[0])
            else:
                thd_layout.validate_compatible(packed_seq_params, query.shape[0])
            # The DSv4 QKV wrapper returns packed qr without its dummy batch axis.
            if qr.ndim == 2:
                qr = qr.unsqueeze(1)
            if x.ndim != 3 or x.shape[:2] != (query.shape[0], 1) or qr.shape[:2] != x.shape[:2]:
                raise ValueError(
                    "CSA2 THD hidden states and query latents must have shape [T, 1, dim]."
                )
            invalid = ~thd_layout.valid_tokens[:, None, None]
            query = query.masked_fill(invalid, 0)
            key = key.masked_fill(invalid, 0)
            x = x.masked_fill(invalid, 0)
            qr = qr.masked_fill(invalid, 0)
        elif thd_layout is not None:
            raise ValueError("CSA2 THD layout requires packed sequence metadata.")
        elif query.ndim != 4 or query.shape[0] == 0:
            raise ValueError("CSA2 expects a nonempty [sequence, batch, heads, head_dim] query.")
        if csa2_state is None:
            if self.compress_ratio and not self.is_kv_source:
                raise ValueError(
                    "CSA2 Reindex/Reuse requires an explicit csa2_state shared with its Full "
                    "owner for this forward."
                )
            csa2_state = CSA2State()
        csa2_state.validate_forward(self.layer_idx, query, thd_layout=thd_layout)
        seq_len, batch = query.shape[0], 1 if is_thd else query.shape[1]
        if attention_mask is not None:
            causal_mask = torch.ones(seq_len, seq_len, device=query.device, dtype=torch.bool).triu(
                1
            )
            valid_mask = attention_mask.dtype == torch.bool and attention_mask.shape[-2:] == (
                seq_len,
                seq_len,
            )
            if valid_mask:
                valid_mask = torch.equal(attention_mask, causal_mask.expand_as(attention_mask))
                if is_thd and not valid_mask:
                    packed_mask = causal_mask | (
                        thd_layout.sequence_ids[:, None] != thd_layout.sequence_ids[None, :]
                    )
                    valid_mask = torch.equal(attention_mask, packed_mask.expand_as(attention_mask))
            if not valid_mask:
                raise NotImplementedError(
                    "Native CSA2 only supports an ordinary causal attention mask."
                )
        if is_thd:
            indices = get_window_topk_idxs_thd(
                self.config.csa_window_size, thd_layout.cu_seqlens_padded, total_q=seq_len
            )
            # DSv4's window helper returns segment-local positions; native THD
            # sparse attention consumes physical flat rows in the packed KV.
            starts = thd_layout.cu_seqlens_padded[thd_layout.sequence_ids.clamp_min(0)]
            indices = torch.where(indices >= 0, indices + starts[:, None], -1)
            valid_window = (
                (indices >= 0)
                & thd_layout.valid_tokens[:, None]
                & thd_layout.valid_tokens[indices.clamp_min(0).long()]
            )
            indices = indices.masked_fill(~valid_window, -1)
        else:
            indices = get_window_topk_idxs(
                self.config.csa_window_size, batch, seq_len, query.device
            )
        kv = key.squeeze(-2)
        indexer_loss = None
        if self.compress_ratio:
            use_indexer_loss = (
                self.training
                and torch.is_grad_enabled()
                and (self.config.dsa_indexer_loss_coeff or 0.0) > 0
            )
            global_kv, global_indices, scores = self._shared_global_attention(
                x, qr, csa2_state, use_indexer_loss=use_indexer_loss
            )
            if scores is not None:
                indexer_loss = self._compute_indexer_loss(
                    query, kv, global_kv, indices, global_indices, scores, thd_layout=thd_layout
                )
            # Keep the shared indices logical: every consumer applies its own concatenation
            # offset out of place, without corrupting the indices seen by the next layer.
            global_indices = torch.where(global_indices >= 0, global_indices + seq_len, -1)
            kv = torch.cat((kv, global_kv.squeeze(1) if is_thd else global_kv), dim=0)
            indices = torch.cat((indices, global_indices), dim=-1)
        # Gather in FP32 so repeated KV positions also accumulate their backward contributions
        # in FP32. Casting after the gather would scatter-add those gradients in BF16.
        output = unfused_compressed_sparse_attn(
            query, kv.float(), self.attn_sink, indices, self.softmax_scale
        )
        if is_thd:
            output = output.masked_fill(~thd_layout.valid_tokens[:, None], 0)
        if indexer_loss is not None:
            # The tracker averages microbatches, so report a token mean in either
            # reduction mode. Backward still receives the raw sum in per-token mode.
            logged_loss = indexer_loss.detach()
            if self.config.calculate_per_token_loss:
                token_count = (
                    thd_layout.valid_tokens.sum().clamp_min(1) if is_thd else seq_len * batch
                )
                logged_loss = logged_loss / token_count
            DSAIndexerLossLoggingHelper.save_loss_to_tracker(
                loss=logged_loss, layer_number=self.layer_idx + 1, num_layers=self.config.num_layers
            )
            output = DSAIndexerLossAutoScaler.apply(output, indexer_loss)
        csa2_state.last_layer = self.layer_idx
        return output

    def backward_dw(self) -> None:
        """Flush the compressor and indexer linears using the existing DSv4 interface."""
        if self.compressor is not None:
            self.compressor.backward_dw()
        if self.indexer is not None:
            self.indexer.backward_dw()
