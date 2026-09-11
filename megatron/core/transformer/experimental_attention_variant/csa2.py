# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Native training math for a V4.1 SWA or Full CSA2 layer.

This uses the existing DSv4 attention wrapper and sparse attention reference. Cross-layer
KV/index reuse and hierarchical candidates require an explicit state interface and are not
implemented here. All tensors belong to the current forward; there are no inference caches.
"""

from copy import copy
from dataclasses import dataclass

import torch
from torch import nn

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant.csa import (
    CompressedSparseAttentionSubmodules,
    CompressorSubmodules,
    _apply_rope,
    get_window_topk_idxs,
    unfused_compressed_sparse_attn,
)
from megatron.core.transformer.module import MegatronModule, mark_keep_in_fp32
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.core.utils import get_pg_size


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    def backward_dw(self) -> None:
        """Follow the existing DSv4 linear gradient interface."""
        self.linear_wkv.backward_dw()
        if self.linear_wgate is not None:
            self.linear_wgate.backward_dw()


@dataclass
class CSA2IndexerSubmodules:
    """Projection and normalization specs for a Full CSA2 indexer."""

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
    ) -> None:
        super().__init__(config=config)
        self.compress_ratio = compress_ratio
        self.tp_group = pg_collection.tp
        self.cp_group = pg_collection.cp
        self.n_heads = config.dsa_indexer_n_heads
        self.head_dim = config.dsa_indexer_head_dim
        self.topk = config.dsa_indexer_topk
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
        self.linear_wk = build_module(
            submodules.linear_wk, config.v_head_dim, self.head_dim, **linear_kwargs
        )
        self.linear_weights_proj = build_module(
            submodules.linear_weights_proj, config.hidden_size, self.n_heads, **linear_kwargs
        )
        self.k_norm = build_module(
            submodules.k_norm,
            config=config,
            hidden_size=self.head_dim,
            eps=config.attention_latent_norm_epsilon,
        )

    def forward_before_topk(
        self, x: torch.Tensor, qr: torch.Tensor, latent: torch.Tensor, rotary_pos_emb: nn.Module
    ) -> torch.Tensor:
        """Return differentiable causal scores ``[batch, sequence, global positions]``.

        The discrete top-k has no LM-loss gradient. A future indexer auxiliary loss can
        consume these scores explicitly without retaining a graph in module state.
        """
        q, _ = self.linear_wq_b(qr)
        q = q.reshape(*q.shape[:-1], self.n_heads, self.head_dim)
        k, _ = self.linear_wk(latent)
        # As for the compressor norm, an incomplete first group produces no key rows.
        k = self.k_norm(k) if k.shape[0] else k * self.k_norm.weight
        pos_dim = self.config.qk_pos_emb_head_dim
        rope_kwargs = dict(
            nope_dim=self.head_dim - pos_dim,
            pos_dim=pos_dim,
            rotary_pos_emb_module=rotary_pos_emb,
            config=self.config,
            cp_group=self.cp_group,
        )
        q = _apply_rope(q, rotary_seq_len=q.shape[0], **rope_kwargs)
        k = _apply_rope(k, rotary_seq_len=k.shape[0], ratio=self.compress_ratio, **rope_kwargs)
        weights, _ = self.linear_weights_proj(x)
        weights = weights.float() * (self.head_dim**-0.5 * self.n_heads**-0.5)
        scores = torch.einsum("sbhd,tbd->bsht", q.float(), k.float()).relu()
        scores = (scores * weights.permute(1, 0, 2).unsqueeze(-1)).sum(dim=2)
        visible = torch.arange(1, x.shape[0] + 1, device=x.device) // self.compress_ratio
        causal = torch.arange(latent.shape[0], device=x.device)[None, :] < visible[:, None]
        return scores.masked_fill(~causal, -torch.inf)

    @torch.no_grad()
    def forward(
        self, x: torch.Tensor, qr: torch.Tensor, latent: torch.Tensor, rotary_pos_emb: nn.Module
    ) -> torch.Tensor:
        """Return sorted global-position indices, with ``-1`` for unavailable positions."""
        scores = self.forward_before_topk(x, qr, latent, rotary_pos_emb)
        indices = scores.topk(min(self.topk, latent.shape[0]), dim=-1, sorted=False).indices
        indices = indices.sort(dim=-1).values
        return indices.masked_fill(~scores.gather(-1, indices).isfinite(), -1).int()

    def backward_dw(self) -> None:
        """Follow the existing DSv4 linear gradient interface."""
        for linear in (self.linear_wq_b, self.linear_wk, self.linear_weights_proj):
            linear.backward_dw()


class CompressedSparseAttention2(MegatronModule):
    """Single-layer native CSA2 behind the existing DSv4 Hybrid attention interface."""

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
        if config.dsa_indexer_loss_coeff:
            raise NotImplementedError("CSA2 indexer auxiliary loss is not integrated yet.")
        layer_idx = layer_number - 1
        ratio = config.csa_compress_ratios[layer_idx]
        if compress_ratio is not None and compress_ratio != ratio:
            raise ValueError("CSA2 compress_ratio must match the configured layer ratio.")
        if ratio and layer_idx not in config.csa2_kv_source_layers:
            raise NotImplementedError(
                "CSA2 Reindex/Reuse layers require cross-layer state (step 3)."
            )
        if (
            ratio
            and config.csa2_candidate_source_layer is not None
            and layer_idx >= config.csa2_candidate_source_layer
        ):
            raise NotImplementedError(
                "CSA2 hierarchical candidates require cross-layer state (step 3)."
            )
        self.compress_ratio = ratio
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
        if ratio:
            kwargs = dict(config=config, compress_ratio=ratio, pg_collection=pg_collection)
            self.compressor = build_module(submodules.compressor, **kwargs)
            self.indexer = build_module(submodules.indexer, **kwargs)

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
    ) -> torch.Tensor:
        """Attend jointly to causal window and selected global latents using one softmax."""
        if packed_seq_params is not None or boundary_hidden is not None or boundary_kv is not None:
            raise NotImplementedError(
                "Native CSA2 currently supports unpacked sequences with CP=1."
            )
        if query.ndim != 4 or query.shape[0] == 0:
            raise ValueError("CSA2 expects a nonempty [sequence, batch, heads, head_dim] query.")
        seq_len, batch = query.shape[:2]
        if attention_mask is not None:
            causal_mask = torch.ones(seq_len, seq_len, device=query.device, dtype=torch.bool).triu(
                1
            )
            if (
                attention_mask.dtype != torch.bool
                or attention_mask.shape[-2:] != (seq_len, seq_len)
                or not torch.equal(attention_mask, causal_mask.expand_as(attention_mask))
            ):
                raise NotImplementedError(
                    "Native CSA2 only supports an ordinary causal attention mask."
                )
        indices = get_window_topk_idxs(self.config.csa_window_size, batch, seq_len, query.device)
        kv = key.squeeze(-2)
        if self.compress_ratio:
            latent = self.compressor(x)
            global_indices = self.indexer(x, qr, latent, self.rotary_pos_emb)
            global_indices = torch.where(global_indices >= 0, global_indices + seq_len, -1)
            pos_dim = self.config.qk_pos_emb_head_dim
            global_kv = _apply_rope(
                latent,
                self.config.v_head_dim - pos_dim,
                pos_dim,
                self.rotary_pos_emb,
                self.config,
                latent.shape[0],
                ratio=self.compress_ratio,
                cp_group=self.cp_group,
            )
            kv = torch.cat((kv, global_kv), dim=0)
            indices = torch.cat((indices, global_indices), dim=-1)
        # Gather in FP32 so repeated KV positions also accumulate their backward contributions
        # in FP32. Casting after the gather would scatter-add those gradients in BF16.
        return unfused_compressed_sparse_attn(
            query, kv.float(), self.attn_sink, indices, self.softmax_scale
        )

    def backward_dw(self) -> None:
        """Flush the compressor and indexer linears using the existing DSv4 interface."""
        if self.compressor is not None:
            self.compressor.backward_dw()
            self.indexer.backward_dw()
