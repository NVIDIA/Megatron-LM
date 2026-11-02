# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Logical row layout of packed attention projection weights."""

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass(frozen=True)
class QKVLayout:
    """Logical row layout of a packed attention projection weight.

    Attention layers attach it to their packed weights as ``weight.qkv_layout`` so that
    optimizers splitting such weights (Muon's split-QKV Newton-Schulz) need not infer the
    layout from parameter names. Rows are ``num_groups`` repetitions of one group; both
    split-shape tuples partition one group:

    - ``projection_split_shapes``: one entry per projection (e.g. Q, gate, K, V). Muon's
      projection split orthogonalizes each projection across all groups as one matrix.
    - ``per_head_split_shapes``: one entry per head of each projection. Muon's per-head split
      orthogonalizes every head of every group on its own.

    A fused QKV weight has one group per query group. An MLA up-projection has one group per
    attention head (``is_mla=True``) and splits into its per-head components, e.g. the
    content and rotary parts of the query.

    Attributes:
        num_groups: Number of groups in the full (unsharded) weight.
        projection_split_shapes: Rows of each projection within one group.
        per_head_split_shapes: Rows of each head within one group.
        is_mla: Whether this is an MLA up-projection rather than a fused QKV weight.
    """

    num_groups: int
    projection_split_shapes: Tuple[int, ...]
    per_head_split_shapes: Tuple[int, ...]
    is_mla: bool = False

    def __post_init__(self):
        if sum(self.projection_split_shapes) != sum(self.per_head_split_shapes):
            raise ValueError(
                f"Projection split {self.projection_split_shapes} and per-head split "
                f"{self.per_head_split_shapes} must cover the same rows of a group."
            )

    @classmethod
    def from_transformer_config(cls, config) -> "QKVLayout":
        """Layout of the fused QKV weight described by a transformer config."""
        heads_per_group = config.num_attention_heads // config.num_query_groups
        projection_split_shapes = [heads_per_group * config.kv_channels]
        per_head_split_shapes = [config.kv_channels] * heads_per_group
        if getattr(config, 'attention_output_gate', False):
            projection_split_shapes.append(heads_per_group * config.kv_channels)
            per_head_split_shapes += [config.kv_channels] * heads_per_group
        projection_split_shapes += [config.kv_channels, config.kv_channels]
        per_head_split_shapes += [config.kv_channels, config.kv_channels]
        return cls(
            num_groups=config.num_query_groups,
            projection_split_shapes=tuple(projection_split_shapes),
            per_head_split_shapes=tuple(per_head_split_shapes),
        )

    @classmethod
    def for_mla_up_projection(cls, num_heads: int, head_widths: Tuple[int, ...]) -> "QKVLayout":
        """Layout of an MLA up-projection whose heads have components of ``head_widths``."""
        return cls(
            num_groups=num_heads,
            projection_split_shapes=tuple(head_widths),
            per_head_split_shapes=tuple(head_widths),
            is_mla=True,
        )


def set_qkv_layout(module, layout: QKVLayout) -> None:
    """Attach ``layout`` to ``module.weight``, keeping a layout the model already declared."""
    weight = getattr(module, 'weight', None)
    if isinstance(weight, torch.nn.Parameter) and getattr(weight, 'qkv_layout', None) is None:
        weight.qkv_layout = layout


def set_mla_up_projection_layouts(attention) -> None:
    """Attach the up-projection layouts of an MLA self-attention layer.

    Each head of the query projection is ``[qk_head_dim | qk_pos_emb_head_dim]`` and each head
    of the key-value up-projection is ``[qk_head_dim | v_head_dim]``. A layer without a fused
    key-value up-projection keeps its other up-projections whole.
    """
    config = attention.config
    query_projection = (
        attention.linear_q_proj if config.q_lora_rank is None else attention.linear_q_up_proj
    )
    set_qkv_layout(
        query_projection,
        QKVLayout.for_mla_up_projection(
            config.num_attention_heads, (config.qk_head_dim, config.qk_pos_emb_head_dim)
        ),
    )
    kv_up_projection = getattr(attention, 'linear_kv_up_proj', None)
    if kv_up_projection is not None:
        set_qkv_layout(
            kv_up_projection,
            QKVLayout.for_mla_up_projection(
                config.num_attention_heads, (config.qk_head_dim, config.v_head_dim)
            ),
        )


def set_mla_qkv_layouts(model_chunks) -> None:
    """Attach the up-projection layouts of every MLA self-attention layer in ``model_chunks``."""
    from megatron.core.transformer.multi_latent_attention import MLASelfAttention

    mla_classes = (MLASelfAttention,)
    try:
        from megatron.core.transformer.experimental_attention_variant.absorbed_mla import (
            AbsorbedMLASelfAttention,
        )

        mla_classes += (AbsorbedMLASelfAttention,)
    except ImportError:
        pass
    for model_chunk in model_chunks:
        for module in model_chunk.modules():
            if isinstance(module, mla_classes):
                set_mla_up_projection_layouts(module)
