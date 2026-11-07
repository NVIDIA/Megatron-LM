# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for the process groups that initialize_model_parallel() lets roles share.

These tests replay initialize_model_parallel() for every rank of a layout with
torch.distributed.new_group() replaced by a recorder, so they can check which roles share a group
and that every rank issues the same new_group() calls. The companion GPU test runs in an
isolated CI bucket with its own device-bound WORLD.
"""

import os
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import megatron.core.parallel_state as ps

# Role name -> parallel_state global holding this rank's group for that role.
ROLE_GLOBALS = {
    "gtp_remat": "_GTP_WEIGHT_REMAT_GROUP",
    "dp_cp": "_DATA_PARALLEL_GROUP_WITH_CP",
    "dp": "_DATA_PARALLEL_GROUP",
    "cp": "_CONTEXT_PARALLEL_GROUP",
    "mp": "_MODEL_PARALLEL_GROUP",
    "tp": "_TENSOR_MODEL_PARALLEL_GROUP",
    "pp": "_PIPELINE_MODEL_PARALLEL_GROUP",
    "embd": "_EMBEDDING_GROUP",
    "pos_embd": "_POSITION_EMBEDDING_GROUP",
    "tp_dp_cp": "_TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP",
    "tp_dp": "_TENSOR_AND_DATA_PARALLEL_GROUP",
    "tp_cp": "_TENSOR_AND_CONTEXT_PARALLEL_GROUP",
    "expt_gtp_remat": "_EXPERT_GTP_WEIGHT_REMAT_GROUP",
    "ep": "_EXPERT_MODEL_PARALLEL_GROUP",
    "ep_tp": "_EXPERT_TENSOR_PARALLEL_GROUP",
    "tp_ep_mp": "_EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP",
    "tp_ep_pp": "_EXPERT_TENSOR_MODEL_PIPELINE_PARALLEL_GROUP",
    "tp_ep_gtp_remat_pp": "_EXPERT_TENSOR_MODEL_PIPELINE_PARALLEL_GROUP_WITH_EGTP",
    "ep_dp": "_EXPERT_DATA_PARALLEL_GROUP",
    "grad_norm": "_INTRA_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP",
}

# Layouts on 8 ranks: name -> initialize_model_parallel() keyword arguments.
LAYOUTS = {
    "dp8": {},
    "ep8_expert_dp1": {"expert_model_parallel_size": 8},
    "tp2_cp2_ep4_expert_dp1": {
        "tensor_model_parallel_size": 2,
        "context_parallel_size": 2,
        "expert_model_parallel_size": 4,
    },
    "pp2_ep4_expert_dp1": {"pipeline_model_parallel_size": 2, "expert_model_parallel_size": 4},
    "cp8_dp1": {"context_parallel_size": 8},
    "pp2": {"pipeline_model_parallel_size": 2},
    "ep2": {"expert_model_parallel_size": 2},
    "gtp_remat2": {"tensor_model_parallel_size": 2, "gtp_remat_size": 2},
}


class _RecordedGroup:
    """Stand-in for a ProcessGroup that remembers how it was created."""

    def __init__(self, ranks, desc, backend) -> None:
        self.ranks = tuple(ranks)
        self.desc = desc
        self.backend = backend

    def __repr__(self):
        return f"_RecordedGroup({self.desc}, {self.ranks})"


class _Replay:
    """Runs initialize_model_parallel() for one rank without creating communicators."""

    def __init__(self, monkeypatch, world_size, rank) -> None:
        self.rank = rank
        self.calls: list[tuple[str | None, tuple[int, ...], str | None, str | None]] = []
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
        monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: world_size)
        monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: rank)
        monkeypatch.setattr(torch.distributed, "new_group", self._new_group)
        monkeypatch.setattr(torch.distributed, "barrier", lambda *args, **kwargs: None)
        monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
        monkeypatch.setattr(torch.cuda, "synchronize", lambda *args, **kwargs: None)
        monkeypatch.setattr(ps, "_set_global_memory_buffer", lambda: None)
        monkeypatch.setattr(ps, "_global_process_group_list", None)
        # CPU builds have no ProcessGroupNCCL.Options; stand in for configured groups.
        monkeypatch.setattr(
            ps, "get_nccl_options", lambda name, cfgs: ("options", name) if name in cfgs else None
        )

    def _new_group(self, ranks=None, backend=None, group_desc=None, **kwargs):
        self.calls.append(
            (group_desc, tuple(ranks), backend, os.environ.get("NCCL_COLLNET_ENABLE"))
        )
        if self.rank in ranks:
            return _RecordedGroup(ranks, group_desc, backend)
        return torch.distributed.GroupMember.NON_GROUP_MEMBER

    def groups(self):
        """Return role -> this rank's group (None where the rank is not a member)."""
        return {role: getattr(ps, name, None) for role, name in ROLE_GLOBALS.items()}


@pytest.fixture
def replay(monkeypatch):
    """Return run(rank, world_size=8, **kwargs) -> _Replay after initialize_model_parallel()."""

    def run(rank, world_size=8, **kwargs):
        ps.destroy_model_parallel()
        recorder = _Replay(monkeypatch, world_size, rank)
        kwargs.setdefault("create_gloo_process_groups", False)
        ps.initialize_model_parallel(**kwargs)
        return recorder

    yield run
    ps.destroy_model_parallel()


def _distinct(groups):
    return {id(group) for group in groups.values() if isinstance(group, _RecordedGroup)}


def test_create_group_creates_independent_groups_by_default(replay):
    recorder = replay(rank=0, share_process_groups=False)
    first = ps.create_group([0, 1], timeout=timedelta(minutes=1), group_desc="FIRST")
    second = ps.create_group([0, 1], timeout=timedelta(minutes=1), group_desc="SECOND")
    assert first is not second
    assert [call[0] for call in recorder.calls[-2:]] == ["FIRST", "SECOND"]


def test_dense_layout_shares_duplicate_and_single_rank_groups(replay):
    """TP=PP=CP=EP=1 on 8 ranks: 20 joined groups collapse to 3 communicators."""
    groups = replay(rank=3).groups()
    assert groups["dp"] is groups["dp_cp"]
    assert groups["ep_dp"] is groups["dp_cp"]
    assert groups["grad_norm"] is groups["dp_cp"]
    assert groups["tp_dp"] is groups["tp_dp_cp"]
    # Amax / load-balancing reductions keep their own communicator.
    assert groups["tp_dp_cp"] is not groups["dp_cp"]
    single_rank = [
        role for role, group in groups.items() if group is not None and group.ranks == (3,)
    ]
    assert len(single_rank) == 14
    assert len({id(groups[role]) for role in single_rank}) == 1
    assert len(_distinct(groups)) == 3


def test_dense_layout_without_sharing_keeps_one_group_per_role(replay):
    groups = replay(rank=3, share_process_groups=False).groups()
    assert len(_distinct(groups)) == len(ROLE_GLOBALS)


@pytest.mark.parametrize(
    "layout, separate",
    [
        # Expert DP = 1: the EP group spans the same ranks as dp-cp but keeps its own
        # communicator, so the MoE all-to-all never queues behind gradient reduce-scatters.
        ("ep8_expert_dp1", [("ep", "dp_cp"), ("ep", "tp_dp_cp"), ("tp_ep_mp", "dp_cp")]),
        ("tp2_cp2_ep4_expert_dp1", [("ep", "dp_cp"), ("tp_ep_mp", "tp_dp_cp")]),
        ("pp2_ep4_expert_dp1", [("ep", "dp_cp"), ("tp_ep_mp", "dp_cp")]),
        # DP = 1: the CP group spans the same ranks as dp-cp.
        ("cp8_dp1", [("cp", "dp_cp"), ("tp_cp", "dp_cp"), ("cp", "tp_cp")]),
    ],
)
def test_overlapping_roles_keep_their_own_groups(replay, layout, separate):
    for rank in range(8):
        groups = replay(rank=rank, **LAYOUTS[layout]).groups()
        for role, other in separate:
            assert groups[role].ranks == groups[other].ranks, (layout, rank, role, other)
            assert groups[role] is not groups[other], (layout, rank, role, other)


def test_expert_dp1_layout_still_shares_safe_roles(replay):
    groups = replay(rank=5, **LAYOUTS["ep8_expert_dp1"]).groups()
    assert groups["dp"] is groups["dp_cp"]
    assert groups["grad_norm"] is groups["dp_cp"]
    assert groups["tp_ep_gtp_remat_pp"] is groups["tp_ep_pp"]
    # Expert DP has a single rank, so it shares the rank's single-rank group.
    assert groups["ep_dp"].ranks == (5,)
    assert groups["ep_dp"] is groups["tp"]


def test_context_parallel_disables_dp_aliases(replay):
    groups = replay(rank=0, **LAYOUTS["tp2_cp2_ep4_expert_dp1"]).groups()
    assert groups["dp"] is not groups["dp_cp"]
    assert groups["tp_dp"] is not groups["tp_dp_cp"]


def test_expert_tensor_parallel_mismatch_disables_expert_dp_alias(replay):
    groups = replay(rank=0, tensor_model_parallel_size=2, expert_tensor_parallel_size=1).groups()
    assert groups["ep_dp"] is not groups["dp_cp"]


def test_gtp_remat_layout(replay):
    groups = replay(rank=0, **LAYOUTS["gtp_remat2"]).groups()
    # GTP_remat changes the expert-DP domain, so it never aliases dp-cp.
    assert groups["ep_dp"] is not groups["dp_cp"]
    assert groups["gtp_remat"].ranks != (0,)
    assert groups["tp_ep_gtp_remat_pp"] is groups["tp_ep_pp"]
    egtp = replay(rank=0, tensor_model_parallel_size=2, expert_gtp_remat_size=2).groups()
    assert egtp["tp_ep_gtp_remat_pp"] is not egtp["tp_ep_pp"]


@pytest.mark.parametrize("layout", sorted(LAYOUTS))
@pytest.mark.parametrize("share_process_groups", [True, False])
def test_every_rank_issues_the_same_new_group_calls(replay, layout, share_process_groups):
    sequences = [
        replay(rank=rank, share_process_groups=share_process_groups, **LAYOUTS[layout]).calls
        for rank in range(8)
    ]
    assert all(sequence == sequences[0] for sequence in sequences[1:])


@pytest.mark.parametrize("layout", sorted(LAYOUTS))
def test_shared_groups_have_matching_ranks(replay, layout):
    for rank in range(8):
        groups = replay(rank=rank, **LAYOUTS[layout]).groups()
        for role, group in groups.items():
            if isinstance(group, _RecordedGroup):
                assert rank in group.ranks, (layout, role)


def test_sharp_dp_cp_group_is_not_shared(replay):
    recorder = replay(rank=0, use_sharp=True)
    groups = recorder.groups()
    sharp = [call for call in recorder.calls if call[3] == "1"]
    assert {call[0] for call in sharp} == {"DATA_PARALLEL_GROUP_WITH_CP"}
    assert groups["dp"] is not groups["dp_cp"]
    assert groups["grad_norm"] is not groups["dp_cp"]
    assert groups["ep_dp"] is not groups["dp_cp"]


def test_custom_nccl_options_prevent_sharing(replay, tmp_path):
    config = tmp_path / "nccl.yaml"
    config.write_text("dp:\n  max_ctas: 4\ntp:\n  max_ctas: 4\n")
    groups = replay(rank=0, nccl_communicator_config_path=str(config)).groups()
    assert groups["dp"] is not groups["dp_cp"]
    assert groups["tp"] is not groups["cp"]
    # Roles without options keep sharing.
    assert groups["ep_dp"] is groups["dp_cp"]
    assert groups["cp"] is groups["mp"]


def test_high_priority_stream_groups_prevent_sharing(replay):
    groups = replay(rank=0, high_priority_stream_groups=["dp_cp"]).groups()
    assert groups["dp"] is not groups["dp_cp"]
    assert groups["ep_dp"] is not groups["dp_cp"]


def test_backend_device_map_keeps_sharing(replay, monkeypatch):
    """Sharing compares the backend argument, so a cpu:gloo,cuda:nccl WORLD is unaffected."""
    monkeypatch.setattr(torch.distributed, "get_backend", lambda group=None: "cpu:gloo,cuda:nccl")
    recorder = replay(rank=0)
    groups = recorder.groups()
    assert groups["dp"] is groups["dp_cp"]
    assert all(call[2] is None for call in recorder.calls)


def test_different_backend_or_timeout_is_never_shared(replay):
    replay(rank=0)
    sharing = ps._ProcessGroupSharing({"dp": ("dp_cp",)}, {})
    timeout = timedelta(minutes=1)
    world = list(range(8))
    dp_cp = ps.create_group(world, timeout=timeout, sharing=sharing, role="dp_cp")
    assert ps.create_group(world, timeout=timeout, sharing=sharing, role="dp") is dp_cp
    for kwargs in [
        {"timeout": timedelta(minutes=2)},
        {"timeout": timeout, "backend": "nccl"},
        {"timeout": timeout, "backend": "gloo"},
        {"timeout": timeout, "use_local_synchronization": True},
    ]:
        assert ps.create_group(world, sharing=sharing, role="dp", **kwargs) is not dp_cp
    # Roles without an allowed pair never share, even with identical ranks.
    assert ps.create_group(world, timeout=timeout, sharing=sharing, role="ep") is not dp_cp
    tp = ps.create_group([0], timeout=timeout, sharing=sharing, role="tp")
    assert ps.create_group([0], timeout=timeout, sharing=sharing, role="cp") is tp
    assert ps.create_group([0], timeout=timeout, sharing=sharing, role="ep") is tp
    assert sharing.describe() == "cp->tp, dp->dp_cp, ep->tp"


def test_independent_builds_never_share_groups(replay):
    first = replay(rank=0).groups()
    second = replay(rank=0).groups()
    assert _distinct(first).isdisjoint(_distinct(second))


def test_sharing_options_use_the_role_configuration(replay, monkeypatch):
    replay(rank=0)
    config = {"dp": {"max_ctas": 4}, "dp_cp": {"max_ctas": 4}}
    sharing = ps._ProcessGroupSharing({"dp": ("dp_cp",)}, config)
    options = object()
    get_options = Mock(return_value=options)
    monkeypatch.setattr(ps, "get_nccl_options", get_options)
    new_group = Mock(return_value=object())
    monkeypatch.setattr(torch.distributed, "new_group", new_group)
    first = ps.create_group([0, 1], sharing=sharing, role="dp_cp")
    assert ps.create_group([0, 1], sharing=sharing, role="dp") is first
    assert new_group.call_count == 1
    assert new_group.call_args.kwargs["pg_options"] is options
    assert get_options.call_args_list == [(("dp_cp", config),), (("dp", config),)]
    with pytest.raises(ValueError, match="sharing registry"):
        ps.create_group([0, 1], sharing=sharing, role="dp", pg_options=object())


def test_gpu_fixture_preserves_existing_world(monkeypatch):
    from tests.unit_tests.test_process_group_sharing_gpu import device_bound_world

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    initialize, destroy = Mock(), Mock()
    monkeypatch.setattr(torch.distributed, "init_process_group", initialize)
    monkeypatch.setattr(torch.distributed, "destroy_process_group", destroy)
    fixture = device_bound_world.__wrapped__()
    with pytest.raises(pytest.skip.Exception, match="pre-existing WORLD"):
        next(fixture)
    initialize.assert_not_called()
    destroy.assert_not_called()


def test_gpu_fixture_owns_and_cleans_up_device_bound_world(monkeypatch):
    from tests.unit_tests.test_process_group_sharing_gpu import device_bound_world

    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("LOCAL_RANK", "2")
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.cuda, "set_device", Mock())
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    initialize, destroy = Mock(), Mock()
    monkeypatch.setattr(torch.distributed, "init_process_group", initialize)
    monkeypatch.setattr(torch.distributed, "destroy_process_group", destroy)
    monkeypatch.setattr(
        torch.distributed.distributed_c10d,
        "_get_default_group",
        lambda: SimpleNamespace(bound_device_id=torch.device("cuda", 2)),
    )
    fixture = device_bound_world.__wrapped__()
    next(fixture)
    assert initialize.call_args.kwargs["device_id"] == torch.device("cuda", 2)
    destroy.assert_not_called()
    fixture.close()
    destroy.assert_called_once_with()


def test_shared_groups_are_registered_once_and_cleared_on_teardown(replay):
    replay(rank=0)
    registered = [group for group in ps._global_process_group_list if group is not None]
    assert len({id(group) for group in registered}) == len(registered) == 3
    ps.destroy_model_parallel()
    assert all(getattr(ps, name, None) is None for name in ROLE_GLOBALS.values())


def test_alias_map_is_logged(replay, caplog):
    with caplog.at_level("INFO", logger=ps.logger.name):
        replay(rank=0)
    message = next(r.getMessage() for r in caplog.records if "shared roles" in r.getMessage())
    assert "dp->dp_cp" in message
    assert "share_process_groups=True" in message
