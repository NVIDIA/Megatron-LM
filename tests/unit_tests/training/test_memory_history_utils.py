# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace
from unittest import mock

from megatron.training.utils.utils import memory_snapshot_path, start_memory_history_recording


def test_memory_snapshot_path_disambiguates_only_when_needed(tmp_path):
    path = str(tmp_path / "memory.snapshot.pickle")

    # A single dumping rank keeps the configured path untouched.
    assert memory_snapshot_path(path, [], 7) == path
    assert memory_snapshot_path(path, [2], 2) == path

    assert memory_snapshot_path(path, [0, 7], 7) == str(tmp_path / "memory.snapshot_rank-7.pickle")
    assert memory_snapshot_path(str(tmp_path / "snapshot"), [0, 3], 3) == str(
        tmp_path / "snapshot_rank-3.pickle"
    )
    # A tagged (OOM) snapshot is always disambiguated, since any rank may produce one.
    assert memory_snapshot_path(path, [], 1, tag="_oom") == str(
        tmp_path / "memory.snapshot_oom_rank-1.pickle"
    )


def test_start_memory_history_records_only_selected_ranks(tmp_path):
    profiling = SimpleNamespace(
        record_memory_history=True,
        profile_ranks=[0, 2],
        memory_snapshot_path=str(tmp_path / "snapshot.pickle"),
    )

    with (
        mock.patch("megatron.training.utils.utils.safe_get_rank", return_value=1),
        mock.patch("torch.cuda.memory._record_memory_history") as record_history,
    ):
        start_memory_history_recording(profiling)
    record_history.assert_not_called()

    with (
        mock.patch("megatron.training.utils.utils.safe_get_rank", return_value=2),
        mock.patch("torch.cuda.memory._record_memory_history") as record_history,
        mock.patch("torch._C._cuda_attach_out_of_memory_observer"),
    ):
        start_memory_history_recording(profiling)
    record_history.assert_called_once_with(
        True, trace_alloc_max_entries=100_000, trace_alloc_record_context=True
    )
