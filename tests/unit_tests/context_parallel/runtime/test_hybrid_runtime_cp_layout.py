# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Microbatch-local hybrid layout managers and TP x CP groups."""

from types import SimpleNamespace

import pytest

from megatron.core.models.hybrid.hybrid_block import HybridStack
from megatron.core.packed_seq_params import PackedSeqParams

pytestmark = pytest.mark.launch_on_gb200


def test_runtime_cp_layout_uses_microbatch_groups_without_mutating_stack():
    """CP1/2/4 microbatches get independent layout managers and TPxCP groups."""

    class FakeGroup:
        def __init__(self, size):
            self._size = size

        def size(self):
            return self._size

    static_cp_group = FakeGroup(1)
    tp_group = FakeGroup(2)
    runtime_groups = {size: FakeGroup(size) for size in (1, 2, 4)}
    runtime_tp_cp_groups = {size: object() for size in (2, 4)}

    stack = SimpleNamespace(
        cp_group=static_cp_group,
        tp_group=tp_group,
        tp_cp_group=None,
        _cp_layout_manager=None,
        layer_cp_layouts=("contiguous", "zigzag"),
        boundary_layout="contiguous",
        config=SimpleNamespace(sequence_parallel=True),
    )

    for cp_size in (1, 2, 4, 1):
        # The batch fetch attaches the runtime TP x CP group next to the runtime CP group.
        packed_seq_params = PackedSeqParams(
            local_cp_size=cp_size,
            cp_group=runtime_groups[cp_size],
            tp_cp_group=runtime_tp_cp_groups.get(cp_size),
        )
        runtime_group, manager = HybridStack._resolve_runtime_cp_layout(stack, packed_seq_params)

        assert runtime_group is runtime_groups[cp_size]
        if cp_size == 1:
            assert manager is None
        else:
            assert manager.cp_group is runtime_groups[cp_size]
            assert manager.tp_cp_group is runtime_tp_cp_groups[cp_size]
            assert manager.layer_layouts == stack.layer_cp_layouts

        assert stack.cp_group is static_cp_group
        assert stack.tp_cp_group is None
