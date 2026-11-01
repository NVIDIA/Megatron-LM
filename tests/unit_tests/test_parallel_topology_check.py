# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from unittest import mock

import pytest
import torch

from megatron.training import parallel_topology_check as ptc


def _simulate_world(hostnames, layouts):
    """Patch torch.distributed so the check sees a fake world.

    Args:
        hostnames: Hostname per global rank.
        layouts: Group name to a list of rank lists that partition the world.
    """

    def payload_for(rank):
        groups = {}
        for name, partition in layouts.items():
            groups[name] = next((tuple(g) for g in partition if rank in g), None)
        return hostnames[rank], groups

    def fake_all_gather_object(output, obj):
        for rank in range(len(hostnames)):
            output[rank] = payload_for(rank)

    patches = [
        mock.patch.object(torch.distributed, "get_world_size", return_value=len(hostnames)),
        mock.patch.object(torch.distributed, "get_rank", return_value=0),
        mock.patch.object(torch.distributed, "all_gather_object", fake_all_gather_object),
        mock.patch.object(ptc, "_local_group_ranks", lambda names: payload_for(0)[1]),
        mock.patch.object(ptc.socket, "gethostname", return_value=hostnames[0]),
    ]
    return patches


def _run(hostnames, layouts, **kwargs):
    patches = _simulate_world(hostnames, layouts)
    for p in patches:
        p.start()
    try:
        return ptc.check_parallel_topology(**kwargs)
    finally:
        for p in patches:
            p.stop()


# 2 nodes x 4 gpus.
HOSTS = ["node0"] * 4 + ["node1"] * 4


def test_parse_process_group_names_accepts_commas_and_dedupes():
    assert ptc.parse_process_group_names(["dp,tp", "cp", "tp", " ep "]) == ["dp", "tp", "cp", "ep"]
    assert ptc.parse_process_group_names(None) == []


def test_parse_process_group_names_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown process group 'foo'"):
        ptc.parse_process_group_names(["tp,foo"])


def test_off_mode_does_nothing():
    with mock.patch.object(torch.distributed, "all_gather_object") as gather:
        assert ptc.check_parallel_topology(mode="off", process_groups=["tp"]) is None
        gather.assert_not_called()


def test_summary_reports_ranks_hosts_and_spanning(capsys):
    layouts = {"tp": [[0, 1, 2, 3], [4, 5, 6, 7]], "dp": [[0, 4], [1, 5], [2, 6], [3, 7]]}
    with mock.patch.object(ptc, "print_rank_0", side_effect=print):
        summary = _run(
            HOSTS, layouts, mode="warn", process_groups=["tp,dp"], intra_node_process_groups=["tp"]
        )
    assert [g["ranks"] for g in summary["tp"]["groups"]] == [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert summary["tp"]["spanning"] == []
    assert len(summary["dp"]["spanning"]) == 4
    assert summary["dp"]["groups"][0]["hosts"] == ["node0", "node1"]
    out = capsys.readouterr().out
    assert "nodes=2" in out
    assert "tp: 2 group(s) of size 4, 0 span nodes" in out
    assert "dp: 4 group(s) of size 2, 4 span nodes" in out


def test_warn_mode_warns_when_intra_node_group_spans_nodes():
    # tp=8 on 4 gpu nodes spans both nodes.
    layouts = {"tp": [list(range(8))]}
    with mock.patch.object(ptc, "warn_rank_0") as warn, mock.patch.object(ptc, "print_rank_0"):
        summary = _run(
            HOSTS, layouts, mode="warn", process_groups=["tp"], intra_node_process_groups=["tp"]
        )
    assert len(summary["tp"]["spanning"]) == 1
    warn.assert_called_once()
    assert "tp" in warn.call_args[0][0]


def test_error_mode_raises_when_intra_node_group_spans_nodes():
    layouts = {"tp": [[0, 1, 4, 5], [2, 3, 6, 7]]}
    with mock.patch.object(ptc, "print_rank_0"):
        with pytest.raises(RuntimeError, match="span multiple nodes"):
            _run(HOSTS, layouts, mode="error", process_groups=[], intra_node_process_groups=["tp"])


def test_error_mode_passes_when_only_reported_groups_span():
    layouts = {"tp": [[0, 1, 2, 3], [4, 5, 6, 7]], "dp": [[0, 4], [1, 5], [2, 6], [3, 7]]}
    with mock.patch.object(ptc, "print_rank_0"):
        summary = _run(
            HOSTS, layouts, mode="error", process_groups=["dp"], intra_node_process_groups=["tp"]
        )
    assert summary["tp"]["spanning"] == []


def test_uninitialized_group_is_skipped(capsys):
    layouts = {"tp": [[0, 1, 2, 3], [4, 5, 6, 7]], "ep": []}
    with mock.patch.object(ptc, "print_rank_0", side_effect=print):
        summary = _run(
            HOSTS, layouts, mode="warn", process_groups=["tp,ep"], intra_node_process_groups=[]
        )
    assert summary["ep"]["groups"] == []
    assert "ep: not initialized, skipped" in capsys.readouterr().out


def test_local_group_ranks_handles_missing_groups():
    sentinel = object()
    getters = {
        "tp": lambda: sentinel,
        "cp": lambda: None,
        "ep": mock.Mock(side_effect=AssertionError),
    }
    with (
        mock.patch.dict(ptc.PROCESS_GROUP_GETTERS, getters),
        mock.patch.object(torch.distributed, "get_process_group_ranks", return_value=[0, 1]),
    ):
        assert ptc._local_group_ranks(["tp", "cp", "ep"]) == {"tp": (0, 1), "cp": None, "ep": None}


def test_single_process_gloo_end_to_end(tmp_path):
    if torch.distributed.is_initialized():
        pytest.skip("distributed already initialized")
    torch.distributed.init_process_group(
        "gloo", init_method=f"file://{tmp_path}/store", rank=0, world_size=1
    )
    try:
        world = torch.distributed.group.WORLD
        with mock.patch.dict(ptc.PROCESS_GROUP_GETTERS, {"tp": lambda: world, "dp": lambda: world}):
            summary = ptc.check_parallel_topology(
                mode="error", process_groups=["tp,dp"], intra_node_process_groups=["tp"]
            )
        assert summary["tp"]["groups"][0]["ranks"] == [0]
        assert summary["tp"]["spanning"] == []
    finally:
        torch.distributed.destroy_process_group()
