# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Indexer runtime groups and logging storage are local to each forward/window."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.experimental_attention_variant.dsa import (
    DSAIndexer,
    DSAIndexerLossLoggingHelper,
    DSAttention,
)

pytestmark = pytest.mark.launch_on_gb200


class _FakeCPGroup:
    def __init__(self, size: int, rank: int = 0):
        self._size = size
        self._rank = rank

    def size(self) -> int:
        return self._size

    def rank(self) -> int:
        return self._rank


def test_dsa_runtime_cp_groups_are_forward_local():
    """DSA and its indexer use CP1/2/4 metadata without mutating shared groups."""
    static_group = _FakeCPGroup(1)
    runtime_groups = {size: _FakeCPGroup(size) for size in (1, 2, 4)}
    pg_collection = SimpleNamespace(cp=static_group, tp=object())
    attention = SimpleNamespace(pg_collection=pg_collection)

    for cp_size in (1, 2, 4, 2, 1):
        packed_seq_params = PackedSeqParams(local_cp_size=cp_size, cp_group=runtime_groups[cp_size])
        cp_group, runtime_pg_collection = DSAttention._resolve_runtime_process_groups(
            attention, packed_seq_params
        )
        assert cp_group is runtime_groups[cp_size]
        assert runtime_pg_collection.cp is runtime_groups[cp_size]
        assert pg_collection.cp is static_group

    indexer = SimpleNamespace(
        pg_collection=SimpleNamespace(cp=static_group, tp=_FakeCPGroup(1)),
        config=SimpleNamespace(
            rope_type="rope",
            sequence_parallel=False,
            dsa_indexer_k_norm_fp32=False,
            dsa_indexer_rotate_activation=False,
        ),
        rotary_pos_emb=SimpleNamespace(get_rotary_seq_len=lambda *args: 3),
        index_n_heads=2,
        index_head_dim=4,
        softmax_scale=0.5,
        use_rope=True,
        index_kpool=1,
        index_kpool_compress_gate=None,
        linear_wq_b=lambda value: (value.new_zeros(3, 1, 8), None),
        linear_wk=lambda value: (value.new_zeros(3, 1, 4), None),
        _project_indexer_weights=lambda value: value.new_zeros(3, 1, 2),
        _projection_quantization_context=nullcontext,
        k_norm=lambda value: value,
    )
    indexer.rotary_pos_emb = lambda *args, **kwargs: torch.empty(3, 1, 1, 4)
    indexer.rotary_pos_emb.get_rotary_seq_len = lambda *args: 3
    seen_groups = []

    def capture_rope(value, rotary_pos_emb, mscale, cu_seqlens=None, cp_group=None):
        seen_groups.append(cp_group)
        return value

    indexer._apply_rope = capture_rope
    DSAIndexer.forward_before_topk(
        indexer,
        torch.randn(3, 1, 4),
        torch.randn(3, 1, 2),
        PackedSeqParams(local_cp_size=4, cp_group=runtime_groups[4]),
    )
    assert seen_groups == [runtime_groups[4], runtime_groups[4]]
    assert indexer.pg_collection.cp is static_group


def test_indexer_loss_cleanup_preserves_graph_accumulator_storage():
    """Graph replay retains accumulation storage; runtime weights need no saved groups."""
    values = torch.ones(2)
    raw_values = torch.ones(2)
    DSAIndexerLossLoggingHelper.tracker = {
        "values": values,
        "raw_values": raw_values,
        "requires_dp_cp": True,
    }
    try:
        DSAIndexerLossLoggingHelper.clean_loss_in_tracker(preserve_groups=True)
        assert DSAIndexerLossLoggingHelper.tracker["values"] is values
        assert DSAIndexerLossLoggingHelper.tracker["raw_values"] is raw_values
        assert torch.count_nonzero(values) == 0
        assert torch.count_nonzero(raw_values) == 0
        assert "reduce_group" not in DSAIndexerLossLoggingHelper.tracker
        assert "avg_group" not in DSAIndexerLossLoggingHelper.tracker
        assert DSAIndexerLossLoggingHelper.tracker["requires_dp_cp"]
        DSAIndexerLossLoggingHelper.clean_loss_in_tracker()
        assert not DSAIndexerLossLoggingHelper.tracker["requires_dp_cp"]
    finally:
        DSAIndexerLossLoggingHelper.tracker = {}
