# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest

from megatron.core.extensions import transformer_engine as mcore_te
from megatron.training.initialize import _should_eager_initialize_dynamic_cp_communicators


@pytest.mark.internal
@pytest.mark.parametrize(
    ("transformer_impl", "cp_comm_type", "expected"),
    (
        ("transformer_engine", "p2p", False),
        ("transformer_engine", None, False),
        ("transformer_engine", "all_gather", True),
        ("local", "p2p", True),
    ),
)
def test_dynamic_cp_logical_communicator_initialization(transformer_impl, cp_comm_type, expected):
    args = SimpleNamespace(
        dynamic_context_parallel=True, transformer_impl=transformer_impl, cp_comm_type=cp_comm_type
    )
    assert _should_eager_initialize_dynamic_cp_communicators(args) is expected


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
