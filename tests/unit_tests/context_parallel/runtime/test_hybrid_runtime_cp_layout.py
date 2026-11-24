# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Microbatch-local CP groups at the computation-branch boundary."""

from types import SimpleNamespace

import pytest
import torch

import megatron.core.context_parallel.module as cp_module
from megatron.core.packed_seq_params import PackedSeqParams

pytestmark = pytest.mark.launch_on_gb200


def test_runtime_cp_layout_uses_microbatch_groups_without_mutating_module(monkeypatch):
    class Group:
        def __init__(self, size):
            self._size = size

        def size(self):
            return self._size

    static_cp = Group(1)
    tp = Group(2)
    groups = SimpleNamespace(cp=static_cp, tp=tp, tp_cp=None)
    module = SimpleNamespace(
        pg_collection=groups,
        config=SimpleNamespace(
            global_cp_layout="contiguous", sequence_parallel=True, cuda_graph_impl="none"
        ),
    )
    calls = []

    def convert(value, **kwargs):
        calls.append(kwargs)
        return value

    monkeypatch.setattr(cp_module, "convert_cp_partition_mode", convert)
    hidden = torch.zeros(4, 1, 2)
    for size in (1, 2, 4, 1):
        cp, tp_cp = Group(size), Group(2 * size)
        params = PackedSeqParams(
            local_cp_size=size, cp_group=cp, tp_cp_group=tp_cp, cp_partition_mode="contiguous"
        )
        _, _, reverse = cp_module.prepare_module_cp_input(
            module, hidden, params, target_layout="zigzag"
        )
        if size == 1:
            assert reverse is None
        else:
            assert calls[-1]["cp_group"] is cp
            assert calls[-1]["tp_cp_group"] is tp_cp
            assert reverse.cp_group is cp
        assert groups.cp is static_cp and groups.tp_cp is None
    assert len(calls) == 2
