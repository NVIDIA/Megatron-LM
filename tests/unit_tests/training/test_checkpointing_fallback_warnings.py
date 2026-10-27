# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Checkpointing helpers use the groups their caller passes, and warn for each one omitted.

A helper whose caller omits a group still reads it from the global parallel grid, with the same
accessor as before, but warns first. Given its groups, a helper reads no global state and does
not warn. On the standard grid every name and writer matches the global fallback.
"""

import contextlib
import os
import re
import warnings
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from megatron.core import process_groups_config
from megatron.core.process_groups_config import ProcessGroupCollection, ProcessGroupFallbackWarning
from megatron.training import checkpointing
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.training.test_checkpointing_pg_collection import (
    _no_global_grid,
    _StandInGroup,
)

VERSIONS = "deprecated since Megatron Core 0.21 and will be removed in 0.23"


@pytest.fixture(autouse=True)
def fresh_warning_registry(monkeypatch):
    """Each test observes the first fallback warning of every owner."""
    monkeypatch.setattr(process_groups_config, "_warned_global_process_group_fallbacks", set())


@pytest.fixture
def stand_in_ranks(monkeypatch):
    """Read ranks and sizes from stand-in groups; a missing group has rank 0 and size 1."""
    monkeypatch.setattr(
        checkpointing, "get_pg_rank", lambda group: 0 if group is None else group.rank()
    )
    monkeypatch.setattr(
        checkpointing, "get_pg_size", lambda group: 1 if group is None else group.size()
    )


@contextlib.contextmanager
def _record_warnings():
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        yield record


def _fallbacks(record):
    """Return the (owner, omitted argument) of each fallback warning in ``record``, in order."""
    fallbacks = []
    for warning in record:
        if issubclass(warning.category, ProcessGroupFallbackWarning):
            message = str(warning.message)
            assert VERSIONS in message
            fallbacks.append(re.match(r"(\S+) was called without `(\w+)`", message).groups())
    return fallbacks


def _fallback_arguments(record, owner):
    """Return, in order, the arguments whose omission ``owner`` warned about.

    Fails on a fallback warning of any other owner.
    """
    fallbacks = _fallbacks(record)
    assert {fallback_owner for fallback_owner, _ in fallbacks} <= {owner}, fallbacks
    return [argument for _, argument in fallbacks]


def test_get_checkpoint_name_warns_before_reading_global_ranks():
    """Ranks neither given nor in a collection come from the global grid, after a warning."""
    global_grid = SimpleNamespace(
        get_pipeline_model_parallel_world_size=lambda: 2,
        get_tensor_model_parallel_rank=lambda: 1,
        get_pipeline_model_parallel_rank=lambda: 1,
        get_expert_model_parallel_world_size=lambda: 4,
        get_expert_model_parallel_rank=lambda: 3,
    )
    with mock.patch.object(checkpointing, "mpu", global_grid), _record_warnings() as record:
        name = checkpointing.get_checkpoint_name("ckpt", 5, tensor_rank=0)

    assert name == os.path.join("ckpt", "iter_0000005", "mp_rank_00_001_003", "model_optim_rng.pt")
    assert _fallback_arguments(record, "get_checkpoint_name") == ["pg_collection"]


def test_get_checkpoint_name_without_global_reads_does_not_warn():
    """Explicit ranks and base directories read no global state, so they do not warn."""
    with _no_global_grid():
        explicit = checkpointing.get_checkpoint_name(
            "ckpt",
            5,
            pipeline_parallel=False,
            tensor_rank=2,
            pipeline_rank=0,
            expert_parallel=False,
            expert_rank=0,
        )
        base_dir = checkpointing.get_checkpoint_name("ckpt", 5, return_base_dir=True)

    assert explicit == os.path.join("ckpt", "iter_0000005", "mp_rank_02", "model_optim_rng.pt")
    assert base_dir == os.path.join("ckpt", "iter_0000005")


@pytest.fixture
def rng_state_without_cuda(monkeypatch, run_config, stand_in_ranks):
    """Let get_rng_state run on CPU, without gathering the states of data-parallel peers."""
    run_config.rng.data_parallel_random_init = False
    monkeypatch.setattr(checkpointing.torch.cuda, "get_rng_state", lambda: "cuda")
    monkeypatch.setattr(
        checkpointing.tensor_parallel,
        "get_cuda_rng_tracker",
        lambda: SimpleNamespace(get_states=lambda: {}),
    )


def test_get_rng_state_uses_the_given_groups(rng_state_without_cuda):
    """The RNG shard is keyed by the ranks of the given groups."""
    tp, pp, dp_cp = _StandInGroup(1, 2), _StandInGroup(0, 2), _StandInGroup(3, 4)
    with _no_global_grid():
        state = checkpointing.get_rng_state(
            "torch_dist", tp, pp, dp_cp_group=dp_cp, dp_group=_StandInGroup(0, 2)
        )

    assert state.global_shape == (2, 2, 4)
    assert state.global_offset == (0, 1, 3)


def test_get_rng_state_warns_for_each_omitted_group(rng_state_without_cuda):
    """Each omitted data-parallel group warns, then is read with the same accessor as before."""
    global_grid = mock.Mock()
    global_grid.get_data_parallel_world_size.return_value = 1
    global_grid.get_data_parallel_group.return_value = _StandInGroup(3, 4)
    single = _StandInGroup(0, 1)
    with mock.patch.object(checkpointing, "mpu", global_grid), _record_warnings() as record:
        state = checkpointing.get_rng_state("torch_dist", single, single)

    assert _fallback_arguments(record, "get_rng_state") == ["dp_group", "dp_cp_group"]
    global_grid.get_data_parallel_world_size.assert_called_once_with()
    global_grid.get_data_parallel_group.assert_called_once_with(with_context_parallel=True)
    assert state.global_offset == (0, 0, 3)


def _metadata_args():
    return SimpleNamespace(use_distributed_optimizer=False, ckpt_format="torch_dist")


def test_sharded_state_dict_metadata_group():
    """The metadata carries the caller's group, or warns and carries the global one."""
    group = _StandInGroup(0, 2)
    with _no_global_grid():
        metadata = checkpointing._build_sharded_state_dict_metadata(
            _metadata_args(), dp_cp_group=group
        )
    assert metadata["dp_cp_group"] is group

    global_grid = mock.Mock()
    global_grid.get_data_parallel_group.return_value = group
    with mock.patch.object(checkpointing, "mpu", global_grid), _record_warnings() as record:
        metadata = checkpointing._build_sharded_state_dict_metadata(_metadata_args())

    assert metadata["dp_cp_group"] is group
    global_grid.get_data_parallel_group.assert_called_once_with(with_context_parallel=True)
    assert _fallback_arguments(record, "_build_sharded_state_dict_metadata") == ["dp_cp_group"]


class _ShardedModule:
    """Records the metadata that generate_state_dict passes to sharded_state_dict."""

    def __init__(self):
        self.metadata = []

    def sharded_state_dict(self, metadata=None):
        self.metadata.append(metadata)
        return {}


def _generate_state_dict(ckpt_format, **kwargs):
    args = SimpleNamespace(
        ckpt_format=ckpt_format,
        no_save_optim=False,
        no_save_rng=True,
        use_distributed_optimizer=False,
    )
    model = _ShardedModule()
    if ckpt_format == "fsdp_dtensor":
        model.state_dict_for_save_checkpoint = dict
    optimizer = mock.Mock(is_stub_optimizer=False)
    optimizer.sharded_state_dict.return_value = {}
    checkpointing.generate_state_dict(args, [model], optimizer, None, None, **kwargs)
    return model, optimizer


@pytest.mark.parametrize("ckpt_format", ["torch_dist", "fsdp_dtensor"])
def test_generate_state_dict_metadata_uses_the_given_group(ckpt_format):
    """The default torch_dist metadata and the fsdp_dtensor metadata use the caller's group."""
    group = _StandInGroup(0, 2)
    with _no_global_grid():
        model, optimizer = _generate_state_dict(ckpt_format, dp_cp_group=group)

    assert optimizer.sharded_state_dict.call_args.kwargs["metadata"]["dp_cp_group"] is group
    if ckpt_format == "torch_dist":
        assert model.metadata == [{"dp_cp_group": group}]


@pytest.mark.parametrize("ckpt_format", ["torch_dist", "fsdp_dtensor"])
def test_generate_state_dict_warns_without_a_group(ckpt_format):
    group = _StandInGroup(0, 2)
    global_grid = mock.Mock()
    global_grid.get_data_parallel_group.return_value = group
    with mock.patch.object(checkpointing, "mpu", global_grid), _record_warnings() as record:
        model, optimizer = _generate_state_dict(ckpt_format)

    assert _fallback_arguments(record, "generate_state_dict") == ["dp_cp_group"]
    global_grid.get_data_parallel_group.assert_called_with(with_context_parallel=True)
    assert optimizer.sharded_state_dict.call_args.kwargs["metadata"]["dp_cp_group"] is group
    if ckpt_format == "torch_dist":
        assert model.metadata == [{"dp_cp_group": group}]


# The global accessors that maybe_save_dataloader_state reads for each omitted group.
_DATALOADER_GLOBAL_READS = {
    "tp_group": {"get_tensor_model_parallel_rank"},
    "pp_group": {"is_pipeline_first_stage", "get_pipeline_model_parallel_world_size"},
    "dp_group": {"get_data_parallel_rank", "get_data_parallel_group"},
    "cp_group": {"get_context_parallel_group"},
}


@pytest.mark.parametrize(
    "omitted",
    [
        (),
        ("tp_group",),
        ("pp_group",),
        ("dp_group",),
        ("cp_group",),
        tuple(_DATALOADER_GLOBAL_READS),
    ],
)
def test_maybe_save_dataloader_state_warns_for_each_omitted_group(
    tmp_path, stand_in_ranks, omitted
):
    single = _StandInGroup(0, 1)
    global_grid = mock.Mock()
    global_grid.get_context_parallel_group.return_value = None
    global_grid.is_pipeline_first_stage.return_value = True
    global_grid.get_tensor_model_parallel_rank.return_value = 0
    global_grid.get_data_parallel_rank.return_value = 0
    global_grid.get_pipeline_model_parallel_world_size.return_value = 1
    global_grid.get_data_parallel_group.return_value = single
    groups = {name: single for name in _DATALOADER_GLOBAL_READS if name not in omitted}
    iterator = SimpleNamespace(iterable=SimpleNamespace(save_state=lambda: {"position": 16}))

    with (
        mock.patch.object(checkpointing, "mpu", global_grid),
        mock.patch.object(checkpointing.torch.distributed, "barrier"),
        _record_warnings() as record,
    ):
        checkpointing.maybe_save_dataloader_state(iterator, 2, str(tmp_path), **groups)

    assert _fallback_arguments(record, "maybe_save_dataloader_state") == list(omitted)
    expected_reads = set().union(*(_DATALOADER_GLOBAL_READS[name] for name in omitted))
    assert {name for name, _, _ in global_grid.method_calls} == expected_reads
    assert (tmp_path / "iter_0000002" / "mp_rank_00" / "train_dataloader_dprank000.pt").exists()


def test_maybe_save_dataloader_state_disabled_does_not_warn():
    with _no_global_grid():
        checkpointing.maybe_save_dataloader_state(iter([1]), 2, None)


@pytest.mark.parametrize("has_sharded_state", [True, False])
@pytest.mark.parametrize("pass_group", [True, False])
def test_local_non_persistent_load_group(has_sharded_state, pass_group):
    """The local checkpoint is restored on the caller's group; only a sharded load needs one."""
    group = _StandInGroup(0, 2)
    intermediate_state_dict = mock.Mock()
    manager = mock.Mock()
    manager.load.return_value = (intermediate_state_dict, "ckpt")
    args = SimpleNamespace(
        non_persistent_ckpt_type="local", non_persistent_local_ckpt_algo="fully_parallel"
    )
    global_grid = mock.Mock()
    global_grid.get_data_parallel_group.return_value = group
    with mock.patch.object(checkpointing, "mpu", global_grid), _record_warnings() as record:
        checkpointing._load_non_persistent_base_checkpoint(
            None,
            args,
            False,
            {"model": {}} if has_sharded_state else None,
            3,
            {"local_checkpoint_manager": manager},
            dp_cp_group=group if pass_group else None,
        )

    falls_back = has_sharded_state and not pass_group
    restore_kwargs = intermediate_state_dict.to_state_dict.call_args.kwargs
    assert restore_kwargs["parallelization_group"] is (
        group if has_sharded_state or pass_group else None
    )
    assert _fallback_arguments(record, "_load_non_persistent_base_checkpoint") == (
        ["dp_cp_group"] if falls_back else []
    )
    assert global_grid.method_calls == (
        [mock.call.get_data_parallel_group(with_context_parallel=True)] if falls_back else []
    )


@pytest.mark.parametrize(
    "process_group,argument,global_read",
    [
        ("dp", "dp_cp_group", mock.call.get_data_parallel_group(with_context_parallel=True)),
        ("ep_dp", "expt_dp_group", mock.call.get_expert_data_parallel_group()),
    ],
)
@pytest.mark.parametrize("pass_group", [True, False])
def test_fully_parallel_load_group(monkeypatch, process_group, argument, global_read, pass_group):
    """The fully parallel load runs on the caller's group, or warns and reads the global one."""
    group = _StandInGroup(0, 2)
    args = SimpleNamespace(
        ckpt_assume_constant_structure=False,
        ckpt_fully_parallel_load=True,
        ckpt_fully_parallel_load_process_group=process_group,
        ckpt_fully_parallel_load_exchange_algo="broadcast",
        ckpt_load_validate_sharding_integrity=False,
        dist_ckpt_strictness="assume_ok_unexpected",
        verify_integrity=False,
    )
    wrapper = mock.Mock()
    monkeypatch.setattr(checkpointing, "TorchDistLoadShardedStrategy", mock.Mock())
    monkeypatch.setattr(checkpointing, "FullyParallelLoadStrategyWrapper", wrapper)
    monkeypatch.setattr(checkpointing, "grant_shape_mismatch_for_gtp_padding", mock.Mock())
    monkeypatch.setattr(checkpointing.dist_checkpointing, "load", mock.Mock(return_value={}))
    global_grid = mock.Mock()
    global_grid.get_data_parallel_group.return_value = group
    global_grid.get_expert_data_parallel_group.return_value = group
    with mock.patch.object(checkpointing, "mpu", global_grid), _record_warnings() as record:
        checkpointing._load_global_dist_base_checkpoint(
            "ckpt",
            args,
            False,
            {},
            3,
            False,
            gtp_pad_for_alignment=1,
            **({argument: group} if pass_group else {}),
        )

    assert wrapper.call_args.args[1] is group
    assert _fallback_arguments(record, "_load_global_dist_base_checkpoint") == (
        [] if pass_group else [argument]
    )
    assert global_grid.method_calls == ([] if pass_group else [global_read])


@pytest.mark.skipif(
    Utils.world_size < 4 or Utils.world_size % 4 != 0,
    reason="Builds a TP2 x PP2 x EP2 grid, which needs a multiple of 4 ranks",
)
def test_standard_grid_names_and_writers_match_the_global_fallback(tmp_path_dist_ckpt, run_config):
    """On the standard grid, the collection selects the names and writers of the fallback."""
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=2,
        expert_model_parallel_size=2,
        expert_tensor_parallel_size=1,
    )
    try:
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        groups = dict(
            tp_group=pg_collection.tp,
            pp_group=pg_collection.pp,
            dp_group=pg_collection.dp_gtp_remat,
            cp_group=pg_collection.cp,
        )
        rank = torch.distributed.get_rank()
        iterator = SimpleNamespace(iterable=SimpleNamespace(save_state=lambda: {"rank": rank}))

        with TempNamedDir(
            tmp_path_dist_ckpt / "test_standard_grid_names_and_writers", sync=True
        ) as save_dir:
            with _record_warnings() as record:
                fallback_name = checkpointing.get_checkpoint_name("ckpt", 3)
                fallback_rng = checkpointing.get_rng_state(
                    "torch_dist", pg_collection.tp, pg_collection.pp
                )
                checkpointing.maybe_save_dataloader_state(iterator, 2, str(save_dir / "fallback"))
            assert _fallbacks(record) == [
                ("get_checkpoint_name", "pg_collection"),
                ("get_rng_state", "dp_group"),
                ("get_rng_state", "dp_cp_group"),
                ("maybe_save_dataloader_state", "tp_group"),
                ("maybe_save_dataloader_state", "pp_group"),
                ("maybe_save_dataloader_state", "dp_group"),
                ("maybe_save_dataloader_state", "cp_group"),
            ]
            with warnings.catch_warnings():
                warnings.simplefilter("error", ProcessGroupFallbackWarning)
                name = checkpointing.get_checkpoint_name("ckpt", 3, pg_collection=pg_collection)
                rng = checkpointing.get_rng_state(
                    "torch_dist",
                    pg_collection.tp,
                    pg_collection.pp,
                    dp_cp_group=pg_collection.dp_cp_gtp_remat,
                    dp_group=pg_collection.dp,
                )
                checkpointing.maybe_save_dataloader_state(
                    iterator, 2, str(save_dir / "collection"), **groups
                )
            torch.distributed.barrier()

            assert name == fallback_name
            assert (rng.global_shape, rng.global_offset) == (
                fallback_rng.global_shape,
                fallback_rng.global_offset,
            )
            files = {}
            for kind in ("fallback", "collection"):
                root = save_dir / kind
                files[kind] = {
                    os.path.relpath(os.path.join(directory, file), root): torch.load(
                        os.path.join(directory, file)
                    )
                    for directory, _, names in os.walk(root)
                    for file in names
                }
            assert files["collection"].keys() == files["fallback"].keys()
            for path, saved in files["collection"].items():
                assert str(saved) == str(files["fallback"][path])
            torch.distributed.barrier()
    finally:
        Utils.destroy_model_parallel()
