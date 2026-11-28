# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Packed TP x runtime-CP group resolution."""

import pytest

from megatron.core.packed_seq_params import PackedSeqParams, resolve_tp_cp_group

pytestmark = pytest.mark.launch_on_gb200


class _MockCPGroup:
    def __init__(self, size):
        self._size = size

    def size(self):
        return self._size


def test_resolve_runtime_tp_cp_group():
    static_tp_cp_group, runtime_tp_cp_group = object(), object()
    packed_seq_params = PackedSeqParams(
        local_cp_size=2, cp_group=_MockCPGroup(2), tp_cp_group=runtime_tp_cp_group
    )

    kwargs = {"sequence_parallel": True, "tp_group": _MockCPGroup(2)}
    assert (
        resolve_tp_cp_group(static_tp_cp_group, packed_seq_params, **kwargs) is runtime_tp_cp_group
    )
    # Static CP, or no packed metadata, keeps the build-time group.
    assert (
        resolve_tp_cp_group(static_tp_cp_group, PackedSeqParams(), **kwargs) is static_tp_cp_group
    )
    assert resolve_tp_cp_group(static_tp_cp_group, None, **kwargs) is static_tp_cp_group
    with pytest.raises(AssertionError, match="tp_cp_group must be set"):
        resolve_tp_cp_group(
            static_tp_cp_group, PackedSeqParams(local_cp_size=2, cp_group=_MockCPGroup(2)), **kwargs
        )


@pytest.mark.parametrize("sequence_parallel,tp_size", [(False, 2), (True, 1)])
def test_runtime_tp_cp_group_is_optional_without_tp_sequence_parallelism(
    sequence_parallel, tp_size
):
    static = object()
    packed = PackedSeqParams(local_cp_size=2, cp_group=_MockCPGroup(2))
    assert (
        resolve_tp_cp_group(
            static, packed, sequence_parallel=sequence_parallel, tp_group=_MockCPGroup(tp_size)
        )
        is static
    )
