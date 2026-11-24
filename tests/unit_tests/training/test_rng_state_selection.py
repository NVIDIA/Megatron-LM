# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""RNG state selection across the saved per-DP-rank layouts."""

from types import SimpleNamespace

import pytest

from megatron.training import checkpointing


def _group(rank, size):
    return SimpleNamespace(rank_=rank, size_=size)


@pytest.fixture(autouse=True)
def _stub_group_queries(monkeypatch):
    monkeypatch.setattr(checkpointing, "get_pg_rank", lambda group: group.rank_)
    monkeypatch.setattr(checkpointing, "get_pg_size", lambda group: group.size_)


def _states(n):
    return [{"id": i} for i in range(n)]


def test_not_per_dp_rank_always_picks_first_entry():
    states = _states(1)
    assert checkpointing._select_rng_state(states, False, _group(3, 4), _group(1, 2)) is states[0]


def test_current_layout_is_indexed_by_dp_cp_rank():
    # DP=2, CP=2: list gathered over DP x CP has 4 entries.
    states = _states(4)
    selected = checkpointing._select_rng_state(states, True, _group(3, 4), _group(1, 2))
    assert selected == {"id": 3}


def test_old_dp_only_layout_is_indexed_by_dp_rank_when_cp_widens_group():
    # DP=2, CP=2: a list gathered over DP only has 2 entries; dp_cp rank 3 would be out of range.
    states = _states(2)
    selected = checkpointing._select_rng_state(states, True, _group(3, 4), _group(1, 2))
    assert selected == {"id": 1}


def test_layouts_agree_without_cp():
    states = _states(2)
    dp = _group(1, 2)
    assert checkpointing._select_rng_state(states, True, dp, dp) == {"id": 1}
