# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""GLM-5.2 fused DSA core attention (Baseten additive module).

This lives in a separate file from the upstream ``dsa.py`` on purpose: GLM-5.2 support
stays *additive* so it never edits the actively-developed upstream DSA module, which
minimizes rebase conflicts against NVIDIA dev. It only imports the shared DSA primitives
from ``dsa`` / ``dsa_kernels``.

The IndexShare knobs (``dsa_indexer_topk_freq`` / ``dsa_indexer_skip_topk_offset``) are
declared on ``TransformerConfig`` so values set by the GLM bridge survive provider-to-config
conversion.
"""

import math
from typing import Optional

import torch

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant.absorbed_mla import (
    AbsorbedMLASelfAttentionSubmodules,
)
from megatron.core.transformer.experimental_attention_variant.dsa import DSAttentionSubmodules
from megatron.core.transformer.experimental_attention_variant.dsa_kernels import (
    build_flat_topk_idxs,
    dsa_sparse_attn,
    indexer_topk,
)
from megatron.core.transformer.experimental_attention_variant.glm_absorbed_mla import (
    GlmAbsorbedMLASelfAttention,
)
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig


def is_dsa_skip_topk_layer(layer_number: int, skip_topk_offset: int, topk_freq: int) -> bool:
    """Return whether a 1-indexed layer reuses a previous DSA top-k result.

    Layers are 1-indexed. The first ``skip_topk_offset`` layers always compute their own
    top-k (they own an indexer). After that, every ``topk_freq``-th layer computes its
    own top-k; the layers in between reuse the top-k indices from the most recent
    computing layer (GLM-5.2 "IndexShare").
    """
    if layer_number < 1:
        raise ValueError(f"layer_number must be 1-indexed and positive, got {layer_number}.")
    if topk_freq < 1:
        raise ValueError(f"topk_freq must be positive, got {topk_freq}.")
    return (max(layer_number - skip_topk_offset, 0) % topk_freq) != 0


def source_dsa_compute_layer(layer_number: int, skip_topk_offset: int, topk_freq: int) -> int:
    """Return the computing layer whose DSA top-k a skip layer reuses."""
    is_dsa_skip_topk_layer(layer_number, skip_topk_offset, topk_freq)
    if layer_number <= skip_topk_offset:
        return layer_number
    return layer_number - ((layer_number - skip_topk_offset) % topk_freq)


class DSAttentionFused(MegatronModule):
    """GLM-5.2 fused DSA core: frozen-indexer top-k + FlashMLA/cuDNN sparse attention.

    Drop-in core_attention for an absorbed-MLA outer (same signature/submodules as
    ``DSAttention``), so it receives the absorbed query ``[sq, b, np, hn]`` and the
    single-head compressed KV ``[sq, b, 1, v_head_dim]`` plus ``x``/``qr`` for the
    indexer. Uses the production ``dsa_kernels`` primitives (no compression, no
    windowing) and supports cross-layer top-k sharing (GLM-5.2 "IndexShare").

    The indexer is frozen for GLM-5.2 (``dsa_indexer_loss_coeff == 0``), so the top-k
    is always selected with the inference kernel ``indexer_topk`` (no loss, no backward);
    the training/score-recompute path is never used.
    """

    _HOLDER_ATTR = "_dsa_index_share_topk_holder"
    """layer_number -> top-k indices map for IndexShare. Packed batches store it on
    PackedSeqParams (isolated per microbatch); non-packed store it on the config for the
    duration of the forward."""

    def __init__(
        self,
        config: TransformerConfig,
        submodules: DSAttentionSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: Optional[float] = None,
        softmax_scale: Optional[float] = None,
        k_channels: Optional[int] = None,
        v_channels: Optional[int] = None,
        cp_comm_type: str = "p2p",
        pg_collection: ProcessGroupCollection = None,
        is_mtp_layer: bool = False,
    ):
        super().__init__(config=config)

        if config.tensor_model_parallel_size != 1:
            raise ValueError("Fused DSA (FlashMLA sparse) currently requires TP=1.")
        if config.context_parallel_size != 1:
            raise ValueError("Fused DSA (FlashMLA sparse) currently requires CP=1.")

        self.layer_number = layer_number
        if is_mtp_layer:
            self.layer_number = self.layer_number + self.config.num_layers

        # Cross-layer top-k sharing (IndexShare): computing layers own an indexer and
        # produce fresh top-k; skip layers reuse the most recent computing layer's top-k.
        self.index_topk = config.dsa_indexer_topk
        self.index_topk_freq = config.dsa_indexer_topk_freq
        self.index_skip_topk_offset = config.dsa_indexer_skip_topk_offset
        if self.index_topk_freq < 1:
            raise ValueError(
                f"dsa_indexer_topk_freq must be positive, got {self.index_topk_freq}."
            )
        if self.index_skip_topk_offset < 0:
            raise ValueError(
                "dsa_indexer_skip_topk_offset must be non-negative, got "
                f"{self.index_skip_topk_offset}."
            )
        self.index_share = self.index_topk_freq > 1
        self.skip_topk = self.index_share and is_dsa_skip_topk_layer(
            self.layer_number, self.index_skip_topk_offset, self.index_topk_freq
        )
        self.source_layer = (
            source_dsa_compute_layer(
                self.layer_number, self.index_skip_topk_offset, self.index_topk_freq
            )
            if self.index_share
            else self.layer_number
        )

        self.indexer = None
        if not self.skip_topk:
            self.indexer = build_module(
                submodules.indexer, config=self.config, pg_collection=pg_collection
            )

        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(
                k_channels if k_channels is not None else config.kv_channels
            )
        self.softmax_scale = softmax_scale

        # GLM-5.2 has no learnable attention sink; dsa_sparse_attn still needs an (np,) bias.
        self.register_buffer(
            "attn_sink",
            torch.zeros(config.num_attention_heads, dtype=torch.float32),
            persistent=False,
        )

    def _get_index_share_topk_holder(
        self, packed_seq_params: Optional[PackedSeqParams]
    ) -> "dict[int, torch.Tensor]":
        holder_owner = packed_seq_params if packed_seq_params is not None else self.config
        holder = getattr(holder_owner, self._HOLDER_ATTR, None)
        if holder is None:
            holder = {}
            setattr(holder_owner, self._HOLDER_ATTR, holder)
        return holder

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        x: torch.Tensor = None,
        qr: torch.Tensor = None,
        up_v_weight: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        attn_mask_type: AttnMaskType = None,
        attention_bias: torch.Tensor = None,
        packed_seq_params: PackedSeqParams = None,
    ):
        """Fused frozen-indexer sparse attention with IndexShare.

        Keyword names match ``AbsorbedMLASelfAttention``'s core_attention call, which passes
        ``value``/``attention_mask``/``x``/``qr``/``up_v_weight``/``position_ids``/
        ``packed_seq_params``/``attn_mask_type`` by keyword. ``query``: ``[sq, b, np,
        k_channels]`` absorbed query (k_channels = kv_lora_rank + rope); ``key``: ``[sq, b, 1,
        k_channels]`` single-head compressed KV (value == key under MQA). ``up_v_weight`` and
        ``position_ids`` are unused here: the outer absorbed-MLA applies the V up-projection
        after core attention, and the indexer derives RoPE positions internally.
        """
        b = query.size(1)
        kv = key.squeeze(-2) if key.dim() == 4 else key  # [sq, b, k_channels]
        seqlen_kv = kv.size(0)

        holder = (
            self._get_index_share_topk_holder(packed_seq_params) if self.index_share else None
        )

        if self.skip_topk:
            # IndexShare skip layer: reuse the source computing layer's top-k.
            if holder is None or self.source_layer not in holder:
                raise AssertionError(
                    f"DSA IndexShare skip layer (layer_number={self.layer_number}) needs top-k "
                    f"from source computing layer {self.source_layer}, but it did not run before "
                    "this layer in this pipeline stage. Cross-PP top-k sharing is not supported. "
                    f"Holder has layers {sorted(holder or {})}."
                )
            topk_local = holder[self.source_layer]
        else:
            # Computing layer: frozen-indexer inference top-k (no loss, no backward).
            assert self.indexer is not None
            q_idx, k_idx, w_idx = self.indexer.forward_before_topk(
                x.detach(), qr.detach(), packed_seq_params
            )
            topk_local, _ = indexer_topk(
                q_idx,
                k_idx,
                w_idx,
                min(self.index_topk, seqlen_kv),
                ratio=1,  # no compression
                indexer_softmax_scale=self.indexer.softmax_scale,
            )
            if holder is not None:
                holder[self.layer_number] = topk_local

        flat_idxs, flat_tlen = build_flat_topk_idxs(
            topk_local, batch_size=b, seqlen_kv=seqlen_kv, compact=True
        )
        # dsa_sparse_attn (FlashMLA convention) attends with the full absorbed query/key dim
        # (kv_lora_rank + rope) but returns only the latent value subspace
        # [sq, b, np * kv_lora_rank], which is exactly what the outer absorbed-MLA V
        # up-projection consumes.
        output = dsa_sparse_attn(
            query,
            kv,
            self.attn_sink.float(),
            flat_idxs,
            self.softmax_scale,
            topk_length=flat_tlen,
        )
        return output


def build_glm_dsa_fused_attention_spec(backend, qk_norm, indexer):
    """Build the GLM-5.2 fused-DSA self-attention ModuleSpec (absorbed MLA + fused DSA core).

    Additive Baseten entry point so the GLM fused spec lives here rather than in the upstream
    ``experimental_attention_variant_module_specs`` builder, keeping that file merge-clean
    against NVIDIA dev. ``backend``/``qk_norm``/``indexer`` are supplied by the shared upstream
    builder, so the linear submodules and HF weight mapping stay identical to the unfused path.
    """
    core_attention = ModuleSpec(
        module=DSAttentionFused,
        submodules=DSAttentionSubmodules(indexer=indexer),
    )
    return ModuleSpec(
        module=GlmAbsorbedMLASelfAttention,
        params={"attn_mask_type": AttnMaskType.causal},
        submodules=AbsorbedMLASelfAttentionSubmodules(
            linear_q_proj=backend.column_parallel_linear(),
            linear_q_down_proj=backend.linear(),
            linear_q_up_proj=backend.column_parallel_linear(),
            linear_kv_down_proj=backend.linear(),
            linear_kv_up_proj=backend.column_parallel_linear(),
            core_attention=core_attention,
            linear_proj=backend.row_parallel_linear(),
            q_layernorm=qk_norm,
            kv_layernorm=qk_norm,
        ),
        metainfo={"fuse_input_layernorm": False},
    )
