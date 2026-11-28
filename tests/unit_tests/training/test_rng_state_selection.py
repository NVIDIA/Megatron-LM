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


@pytest.mark.parametrize(
    ("run_config", "default", "expected"),
    [
        (None, True, True),  # no saved config: fall back to the active run's policy
        (None, False, False),
        ({}, True, False),  # saved config without the fields: single-state layout
        ({"checkpoint": {"save_rng_state_per_dp_rank": True}}, False, True),
        ({"rng": {"data_parallel_random_init": True}}, False, True),
        ({"checkpoint": None, "rng": None}, True, False),
    ],
)
def test_saved_layout_is_read_from_the_saved_run_config(run_config, default, expected):
    assert checkpointing._checkpoint_has_per_dp_rng_states(run_config, default) is expected


def _rng_object(shape, offset, replica_id=0):
    from megatron.core.dist_checkpointing.mapping import ShardedObject

    return ShardedObject("rng_state", None, shape, offset, replica_id=replica_id)


def _metadata(*objects):
    return {obj.unique_key: obj for obj in objects}


def test_metadata_with_the_exact_key_is_kept():
    current = _rng_object((2, 2, 4), (0, 0, 3))
    assert checkpointing._match_rng_state_metadata(current, _metadata(current)) is current


def test_legacy_2d_metadata_is_adopted_with_dp_rank_as_replica_id():
    current = _rng_object((2, 2, 4), (0, 0, 3))
    legacy = _rng_object((2, 2), (0, 0))
    matched = checkpointing._match_rng_state_metadata(current, _metadata(legacy))
    assert (matched.global_shape, matched.global_offset) == ((2, 2), (0, 0))
    assert matched.replica_id == 3


def test_shrinking_the_dp_cp_world_drops_the_rng_state():
    current = _rng_object((2, 2, 2), (0, 0, 1))
    stored = _metadata(*(_rng_object((2, 2, 4), (0, 0, k)) for k in range(4)))
    assert checkpointing._match_rng_state_metadata(current, stored) is None


@pytest.mark.parametrize(
    ("dp_random", "per_dp_flag", "gathers"),
    [(False, False, False), (False, True, True), (True, False, True)],
)
def test_either_flag_makes_the_save_gather_per_rank_states(
    monkeypatch, run_config, dp_random, per_dp_flag, gathers
):
    from unittest.mock import Mock

    run_config.rng.data_parallel_random_init = dp_random
    run_config.checkpoint.save_rng_state_per_dp_rank = per_dp_flag
    monkeypatch.setattr(checkpointing.torch.cuda, "get_rng_state", lambda: "cuda")
    monkeypatch.setattr(
        checkpointing.tensor_parallel,
        "get_cuda_rng_tracker",
        lambda: SimpleNamespace(get_states=lambda: {}),
    )
    monkeypatch.setattr(checkpointing.torch.distributed, "is_initialized", lambda: True)
    dp_group, dp_cp_group = _group(1, 2), _group(3, 4)  # DP=2, CP=2; this rank is DP x CP rank 3

    def fake_gather(output, state, *, group):
        assert group is dp_cp_group  # gathering over DP only would drop the CP peers
        output[:] = [{"id": i} for i in range(len(output))]

    gather = Mock(side_effect=fake_gather)
    monkeypatch.setattr(checkpointing.torch.distributed, "all_gather_object", gather)
    state = checkpointing.get_rng_state(
        "torch", object(), object(), dp_cp_group=dp_cp_group, dp_group=dp_group
    )
    assert gather.call_count == int(gathers)
    assert len(state) == (4 if gathers else 1)
    if gathers:
        selected = checkpointing._select_rng_state(state, True, dp_cp_group, dp_group)
        assert selected == {"id": 3}


def _stored_names(key, n):
    return {f"rng_state.{key}.{i}.random_rng_state": None for i in range(n)}


@pytest.mark.parametrize(
    ("saved_entries", "local_entries"),
    [
        (2, 4),
        (4, 1),
        (1, 4),
    ],  # old DP-only list; saved-true/current-false; saved-false/current-true
)
def test_fsdp_template_follows_the_stored_list_length(saved_entries, local_entries):
    local = {"(0, 0)": [{"random_rng_state": i} for i in range(local_entries)]}
    template = checkpointing._fsdp_rng_state_load_template(
        local, _stored_names("(0, 0)", saved_entries)
    )
    assert list(template) == ["(0, 0)"]
    assert len(template["(0, 0)"]) == saved_entries


def test_fsdp_template_is_unchanged_without_stored_rng_keys():
    local = {"(0, 0)": _states(2)}
    assert checkpointing._fsdp_rng_state_load_template(local, {}) is local


def test_single_stored_entry_is_used_on_a_nonzero_dp_rank():
    # saved with per-rank states off, resumed with them on: DP rank 1 must not index past the list.
    states = _states(1)
    assert checkpointing._select_rng_state(states, True, _group(1, 2), _group(1, 2)) is states[0]


def test_sidecarless_singleton_is_restored_on_dp_rank_1_regardless_of_active_flag():
    # Older DP=2 checkpoint without run_config.yaml, saved with data_parallel_random_init=False.
    per_dp = checkpointing._checkpoint_has_per_dp_rng_states(None, default=True)
    states = _states(1)
    assert checkpointing._select_rng_state(states, per_dp, _group(1, 2), _group(1, 2)) is states[0]


def test_sidecarless_legacy_per_dp_list_is_still_indexed_by_dp_rank():
    per_dp = checkpointing._checkpoint_has_per_dp_rng_states(None, default=True)
    states = _states(2)
    assert checkpointing._select_rng_state(states, per_dp, _group(1, 2), _group(1, 2)) == {"id": 1}
