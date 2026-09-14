# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Shared preparation of packed microbatch metadata before model execution."""

from megatron.core.context_parallel_layout import finalize_packed_seq_params
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.transformer_config import TransformerConfig


def prepare_packed_seq_params(
    packed_seq_params: PackedSeqParams | None,
    config: TransformerConfig,
    *,
    capacity: int | None = None,
) -> PackedSeqParams | None:
    """Finalize CP routes and prepare enabled attention layouts outside graph capture.

    Args:
        packed_seq_params: Metadata for this microbatch, or None for unpacked input.
        config: The model's transformer configuration.
        capacity: Global physical token capacity when it differs from the supplied
            sequence boundaries (for example, raw metadata on a middle PP stage).
            Dynamic packed graphs otherwise use the configured fixed capacity.

    Returns:
        The supplied metadata with its CP group and per-microbatch routes prepared.
    """
    packed_seq_params = finalize_packed_seq_params(packed_seq_params)
    if packed_seq_params is None or not getattr(config, "dsa_cp_balance_indexer", False):
        return packed_seq_params

    from megatron.core.transformer.cuda_graph_config import cuda_graph_captures_attention
    from megatron.core.transformer.experimental_attention_variant.cp_balanced_indexer import (
        prebuild_balanced_layouts,
    )

    dynamic_packs = getattr(config, "dsa_cp_balance_indexer_graph_dynamic_packs", False)
    if capacity is None and dynamic_packs:
        capacity = config.max_seqlen_per_dp_cp_rank * config.context_parallel_size
    prebuild_balanced_layouts(
        packed_seq_params,
        cp_group=packed_seq_params.cp_group,
        pad_alignment=config.pad_packed_seq_alignment,
        capacity=capacity,
        graphs_enabled=cuda_graph_captures_attention(config),
        graph_dynamic_packs=dynamic_packs,
    )
    return packed_seq_params
