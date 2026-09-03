# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest

from megatron.core.dynamic_cp_group import LogicalCPGroup
from megatron.core.extensions import transformer_engine as mcore_te
from megatron.core.parallel_state import create_dynamic_dp_cp_groups
from megatron.training.initialize import _should_use_dynamic_cp_logical_groups


@pytest.mark.internal
@pytest.mark.parametrize(
    ("transformer_impl", "cp_comm_type", "expected"),
    (
        ("transformer_engine", "p2p", True),
        ("transformer_engine", None, True),
        ("transformer_engine", "all_gather", False),
        ("local", "p2p", False),
    ),
)
def test_dynamic_cp_logical_group_selection(transformer_impl, cp_comm_type, expected):
    args = SimpleNamespace(
        dynamic_context_parallel=True, transformer_impl=transformer_impl, cp_comm_type=cp_comm_type
    )
    assert _should_use_dynamic_cp_logical_groups(args) is expected


@pytest.mark.internal
def test_dynamic_cp_logical_group_is_topology_only(monkeypatch):
    monkeypatch.setattr(
        "megatron.core.parallel_state.create_group",
        lambda *args, **kwargs: pytest.fail("logical groups must not create a ProcessGroup"),
    )

    groups = create_dynamic_dp_cp_groups(
        rank=2, ranks=[0, 1, 2, 3], pg_options=None, min_cp_size=1, use_logical_groups=True
    )

    assert groups == {
        1: LogicalCPGroup(ranks=(2,), cp_size=1, cp_rank=0),
        2: LogicalCPGroup(ranks=(2, 3), cp_size=2, cp_rank=0),
        4: LogicalCPGroup(ranks=(0, 1, 2, 3), cp_size=4, cp_rank=2),
    }


@pytest.mark.internal
def test_dynamic_cp_p2p_transport_uses_parent_group(monkeypatch):
    logical_group = object()
    parent_group = object()
    calls = []
    monkeypatch.setattr(
        mcore_te,
        "_get_cp_p2p_transport_group_setter",
        lambda: lambda logical, parent: calls.append((logical, parent)),
    )

    mcore_te._set_dynamic_cp_p2p_transport_group(logical_group, parent_group, "p2p")
    mcore_te._set_dynamic_cp_p2p_transport_group(logical_group, parent_group, "all_gather")

    assert calls == [(logical_group, parent_group)]


@pytest.mark.internal
def test_dynamic_cp_p2p_transport_requires_parent_group():
    with pytest.raises(RuntimeError, match="requires a dp_cp process group"):
        mcore_te._set_dynamic_cp_p2p_transport_group(object(), None, "p2p")
