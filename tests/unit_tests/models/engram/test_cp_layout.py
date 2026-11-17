# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The layout arithmetic the n-gram memory uses under context parallelism.

These are pure index helpers, so they are checked against an explicit statement of the
convention rather than against a second implementation. In the zigzag layout rank ``r`` of
``cp_size`` owns ``chunk[r]`` and ``chunk[2 * cp_size - 1 - r]`` of a sequence cut into
``2 * cp_size`` pieces; in the contiguous layout it owns the ``r``-th of ``cp_size`` slices.
Getting this wrong is the kind of bug an end-to-end loss comparison hides, because a
consistently permuted sequence still trains.
"""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.models.engram.cp_layout import (
    resolve_layer_cp_layout,
    restore_zigzag,
    select_contiguous,
    select_zigzag,
    stack_cp_layouts,
    stack_mixes_cp_layouts,
)


def _global(length):
    return torch.arange(length, dtype=torch.int64).view(1, length)


def test_select_matches_the_documented_chunk_pairing():
    sequence = _global(8)
    # cp_size=2 -> 4 chunks of 2: rank 0 owns chunks 0 and 3, rank 1 owns chunks 1 and 2.
    torch.testing.assert_close(
        select_zigzag(sequence, 2, 0, sequence_dim=1), torch.tensor([[0, 1, 6, 7]])
    )
    torch.testing.assert_close(
        select_zigzag(sequence, 2, 1, sequence_dim=1), torch.tensor([[2, 3, 4, 5]])
    )


def test_restore_is_the_inverse_of_select():
    for cp_size in (1, 2, 4):
        sequence = _global(8 * cp_size)
        shards = [select_zigzag(sequence, cp_size, rank, 1) for rank in range(cp_size)]
        gathered = torch.cat(shards, dim=1)  # what an all-gather produces: rank-major
        torch.testing.assert_close(restore_zigzag(gathered, cp_size, sequence_dim=1), sequence)


def test_round_trip_holds_on_the_sequence_leading_dimension():
    """Hidden states are [S, B, ...]; tokens are [B, S]. Both layouts must work."""
    hidden = torch.arange(8 * 3).view(8, 3)
    shards = [select_zigzag(hidden, 2, rank, sequence_dim=0) for rank in range(2)]
    gathered = torch.cat(shards, dim=0)
    torch.testing.assert_close(restore_zigzag(gathered, 2, sequence_dim=0), hidden)


def test_cp_size_one_is_the_identity():
    sequence = _global(5)
    torch.testing.assert_close(select_zigzag(sequence, 1, 0, 1), sequence)
    torch.testing.assert_close(restore_zigzag(sequence, 1, 1), sequence)


def test_thd_partition_matches_per_document_zigzag():
    """Packed rows zigzag inside each document, not across the row."""
    from megatron.core.models.engram.cp_layout import thd_partition_index

    # Two documents of length 8 each, cp_size=2 -> 4 chunks of 2 per document.
    cu_seqlens = torch.tensor([0, 8, 16], dtype=torch.int32)
    # Rank 0 owns chunks 0 and 3 of every document; rank 1 owns chunks 1 and 2.
    torch.testing.assert_close(
        thd_partition_index(cu_seqlens, 2, 0), torch.tensor([0, 1, 6, 7, 8, 9, 14, 15])
    )
    torch.testing.assert_close(
        thd_partition_index(cu_seqlens, 2, 1), torch.tensor([2, 3, 4, 5, 10, 11, 12, 13])
    )
    # Together the ranks cover every position exactly once.
    covered = torch.cat([thd_partition_index(cu_seqlens, 2, r) for r in range(2)]).sort().values
    torch.testing.assert_close(covered, torch.arange(16))


def test_thd_partition_rejects_indivisible_documents():
    from megatron.core.models.engram.cp_layout import thd_partition_index

    cu_seqlens = torch.tensor([0, 6], dtype=torch.int32)  # 6 is not divisible by 2*cp_size=4
    with pytest.raises(ValueError, match="divisible by 2 \\* cp_size"):
        thd_partition_index(cu_seqlens, 2, 0)


def test_contiguous_select_takes_one_slice_per_rank():
    sequence = _global(8)
    torch.testing.assert_close(
        select_contiguous(sequence, 2, 0, sequence_dim=1), torch.tensor([[0, 1, 2, 3]])
    )
    torch.testing.assert_close(
        select_contiguous(sequence, 2, 1, sequence_dim=1), torch.tensor([[4, 5, 6, 7]])
    )
    for cp_size in (1, 2, 4):
        sequence = _global(8 * cp_size)
        shards = [select_contiguous(sequence, cp_size, rank, 1) for rank in range(cp_size)]
        # What an all-gather produces, rank-major, is the sequence itself.
        torch.testing.assert_close(torch.cat(shards, dim=1), sequence)
    hidden = torch.arange(8 * 3).view(8, 3)
    torch.testing.assert_close(select_contiguous(hidden, 2, 1, sequence_dim=0), hidden[4:])
    with pytest.raises(ValueError, match="divisible by cp_size"):
        select_contiguous(_global(7), 2, 0, sequence_dim=1)


def test_layout_fields_resolve_like_the_stack():
    explicit = dict(attention_cp_layout="zigzag", linear_cp_layout="contiguous")
    # A config that is not an attention-type layer config runs in linear_cp_layout.
    assert resolve_layer_cp_layout(SimpleNamespace(**explicit)) == "contiguous"
    assert stack_cp_layouts(SimpleNamespace(**explicit)) == {"zigzag", "contiguous"}
    assert stack_mixes_cp_layouts(SimpleNamespace(**explicit))

    # Without an explicit linear_cp_layout the whole stack follows cp_partition_mode, whatever
    # the two layout fields hold.
    legacy = SimpleNamespace(
        _linear_cp_layout_explicit=False, cp_partition_mode="contiguous", **explicit
    )
    assert resolve_layer_cp_layout(legacy) == "contiguous"
    assert stack_cp_layouts(legacy) == {"contiguous"}
    assert not stack_mixes_cp_layouts(legacy)

    # A config without the fields, or with an unset linear_cp_layout, reads as zigzag.
    assert resolve_layer_cp_layout(SimpleNamespace()) == "zigzag"
    assert resolve_layer_cp_layout(SimpleNamespace(linear_cp_layout=None)) == "zigzag"
    assert stack_cp_layouts(SimpleNamespace()) == {"zigzag"}
    assert not stack_mixes_cp_layouts(SimpleNamespace())


def test_layer_layout_is_the_one_hybrid_stack_selects(monkeypatch):
    """Every layer type resolves to the layout HybridStack hands its layout manager."""
    import megatron.core.models.hybrid.hybrid_block as hybrid_block_module
    from megatron.core.models.hybrid.hybrid_block import HybridStack
    from megatron.core.models.hybrid.hybrid_layer_allocation import validate_segment_layers
    from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
    from megatron.core.transformer.transformer_config import MLATransformerConfig

    layout_manager_kwargs = {}

    class CapturingLayoutManager:
        def __init__(self, **kwargs):
            layout_manager_kwargs.update(kwargs)

    monkeypatch.setattr(hybrid_block_module, "ContextParallelLayoutManager", CapturingLayoutManager)
    monkeypatch.setattr(
        hybrid_block_module, "build_module", lambda module_spec, **kwargs: torch.nn.Identity()
    )
    config = MLATransformerConfig(
        num_layers=8,
        hidden_size=64,
        num_attention_heads=4,
        linear_cp_layout="contiguous",
        attention_cp_layout="zigzag",
    )
    layer_config_list = validate_segment_layers("MGK*-E", config) + validate_segment_layers(
        "D+", config
    )
    HybridStack(
        config=config,
        submodules=hybrid_stack_spec.submodules,
        layer_config_list=layer_config_list,
        pre_process=False,
        post_layer_norm=False,
        post_process=False,
        pg_collection=SimpleNamespace(
            pp=None, tp=None, cp=SimpleNamespace(size=lambda: 2), tp_cp=None
        ),
    )

    resolved = tuple(resolve_layer_cp_layout(layer_config) for layer_config in layer_config_list)
    assert resolved == layout_manager_kwargs["layer_layouts"]
    assert set(resolved) == {"zigzag", "contiguous"} == stack_cp_layouts(config)
