# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import sys
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
import torch

from megatron.core import parallel_state, process_groups_config
from megatron.core.process_groups_config import ProcessGroupCollection, ProcessGroupFallbackWarning
from tests.unit_tests.test_utilities import Utils, forbid_global_process_groups

MULTIMODAL_EXAMPLE_DIR = Path(__file__).resolve().parents[3] / "examples" / "multimodal"
sys.path.insert(0, str(MULTIMODAL_EXAMPLE_DIR))

import dataloader_provider  # noqa: E402


class _Group:
    """A stand-in process group with a fixed rank and size."""

    def __init__(self, rank=0, size=1):
        self._rank, self._size = rank, size

    def rank(self):
        return self._rank

    def size(self):
        return self._size


def _pg_collection(tp_rank=0, cp_rank=0, pp_rank=0, pp_size=1, dp_rank=0, dp_size=1):
    """The groups the provider reads, on a grid with TP and CP of 2."""
    return ProcessGroupCollection(
        tp=_Group(tp_rank, 2),
        cp=_Group(cp_rank, 2),
        pp=_Group(pp_rank, pp_size),
        dp_gtp_remat=_Group(dp_rank, dp_size),
    )


@pytest.mark.parametrize(
    ("tp_rank", "cp_rank", "deduplicate", "expected"),
    [
        (0, 0, False, True),
        (0, 1, False, True),
        (0, 0, True, True),
        (0, 1, True, False),
        (1, 0, True, False),
    ],
)
def test_dataloader_rank_cp_deduplication(tp_rank, cp_rank, deduplicate, expected):
    pg_collection = _pg_collection(tp_rank=tp_rank, cp_rank=cp_rank)

    with forbid_global_process_groups():
        assert (
            dataloader_provider.is_dataloader_rank(
                encoder_pipeline_model_parallel_size=0,
                deduplicate_across_context_parallel=deduplicate,
                pg_collection=pg_collection,
            )
            is expected
        )


@pytest.mark.parametrize(
    ("encoder_pipeline_model_parallel_size", "dataloader_stages"), [(0, {0, 3}), (1, {0, 1, 3})]
)
def test_dataloader_rank_uses_the_collection_pipeline_stage(
    encoder_pipeline_model_parallel_size, dataloader_stages
):
    with forbid_global_process_groups():
        stages = {
            pp_rank
            for pp_rank in range(4)
            if dataloader_provider.is_dataloader_rank(
                encoder_pipeline_model_parallel_size,
                pg_collection=_pg_collection(pp_rank=pp_rank, pp_size=4),
            )
        }

    assert stages == dataloader_stages


@pytest.mark.parametrize(
    ("num_workers", "configured_prefetch_factor", "expected_prefetch_factor"),
    [(0, 4, None), (1, 4, 4), (1, None, 8)],
)
def test_new_dataloader_prefetch_factor(
    monkeypatch, num_workers, configured_prefetch_factor, expected_prefetch_factor
):
    values = {
        "dataloader_seed": 0,
        "encoder_pipeline_model_parallel_size": 0,
        "load": None,
        "num_workers": num_workers,
        "packing_buffer_size": 10000,
    }
    if configured_prefetch_factor is not None:
        values["dataloader_prefetch_factor"] = configured_prefetch_factor
    args = SimpleNamespace(**values)
    loader_kwargs = {}

    monkeypatch.setattr(dataloader_provider, "get_args", lambda: args)
    monkeypatch.setattr(dataloader_provider, "is_dataloader_rank", lambda _, **kwargs: True)
    monkeypatch.setattr(dataloader_provider, "WorkerConfig", lambda **kwargs: kwargs)
    monkeypatch.setattr(
        dataloader_provider,
        "datasets_provider",
        lambda task_encoder, worker_config: ("train-dataset", None, None),
    )
    monkeypatch.setattr(dataloader_provider, "use_new_dataloader_path", lambda: True)
    monkeypatch.setattr(dataloader_provider, "FileStoreCachePool", lambda **kwargs: kwargs)

    def fake_get_savable_loader(dataset, **kwargs):
        loader_kwargs.update(kwargs)
        return [dataset]

    monkeypatch.setattr(dataloader_provider, "get_savable_loader", fake_get_savable_loader)

    train_loader, valid_loader, test_loader = (
        dataloader_provider.train_valid_test_dataloaders_provider(
            train_val_test_num_samples=None, task_encoder=object(), pg_collection=_pg_collection()
        )
    )

    assert train_loader._dataloader == ["train-dataset"]
    assert valid_loader is None
    assert test_loader._dataloader is None
    if expected_prefetch_factor is None:
        assert "prefetch_factor" not in loader_kwargs
    else:
        assert loader_kwargs["prefetch_factor"] == expected_prefetch_factor


def test_new_dataloader_rejects_non_positive_prefetch_factor(monkeypatch):
    monkeypatch.setattr(
        dataloader_provider, "get_args", lambda: SimpleNamespace(dataloader_prefetch_factor=0)
    )

    with pytest.raises(ValueError, match="must be positive"):
        dataloader_provider.train_valid_test_dataloaders_provider(
            train_val_test_num_samples=None, task_encoder=object()
        )


@pytest.fixture
def strict_resume_provider(monkeypatch):
    args = SimpleNamespace(
        load="/checkpoints/model",
        dataloader_save="/checkpoints/dataloader",
        strict_dataloader_state_load=True,
        num_workers=0,
        dataloader_seed=0,
        packing_buffer_size=10000,
    )
    train_state = SimpleNamespace(iteration=100)
    loader = MagicMock()
    loader.__len__.return_value = 42
    checkpoint_name = Mock(return_value="/checkpoints/dataloader/iter_0000100/state.pt")
    load_state = Mock(return_value={"dataloader_state_dict": "saved-state"})
    monkeypatch.setattr(dataloader_provider, "get_args", lambda: args)
    monkeypatch.setattr(dataloader_provider, "get_train_state", lambda: train_state)
    monkeypatch.setattr(dataloader_provider, "is_dataloader_rank", lambda *a, **kw: True)
    monkeypatch.setattr(dataloader_provider, "WorkerConfig", lambda **kw: kw)
    monkeypatch.setattr(dataloader_provider, "datasets_provider", lambda *a: (object(), None, None))
    monkeypatch.setattr(dataloader_provider, "use_new_dataloader_path", lambda: True)
    monkeypatch.setattr(dataloader_provider, "FileStoreCachePool", lambda **kw: None)
    monkeypatch.setattr(dataloader_provider, "get_savable_loader", lambda *a, **kw: loader)
    monkeypatch.setattr(dataloader_provider, "get_checkpoint_name", checkpoint_name)
    monkeypatch.setattr(dataloader_provider.torch, "load", load_state)
    return args, loader, checkpoint_name, load_state, train_state


@pytest.mark.parametrize("iteration", [0, 100])
@pytest.mark.parametrize("state_exists", [False, True])
def test_length_probe_never_restores_state(
    monkeypatch, strict_resume_provider, iteration, state_exists
):
    args, loader, checkpoint_name, load_state, train_state = strict_resume_provider
    train_state.iteration = iteration
    monkeypatch.setattr(dataloader_provider.os.path, "exists", lambda _: state_exists)

    train, _, _ = dataloader_provider.train_valid_test_dataloaders_provider(
        None, task_encoder=object(), restore_dataloader_state=False, pg_collection=_pg_collection()
    )

    assert len(train._dataloader) == 42
    assert train_state.iteration == iteration
    checkpoint_name.assert_not_called()
    load_state.assert_not_called()
    loader.restore_state_rank.assert_not_called()


def test_default_restore_uses_resolved_iteration(monkeypatch, strict_resume_provider):
    args, loader, checkpoint_name, load_state, _ = strict_resume_provider
    monkeypatch.setattr(dataloader_provider.os.path, "exists", lambda _: True)
    dataloader_provider.train_valid_test_dataloaders_provider(
        None, task_encoder=object(), pg_collection=_pg_collection()
    )
    assert checkpoint_name.call_args.args[:2] == (args.dataloader_save, 100)
    load_state.assert_called_once()
    loader.restore_state_rank.assert_called_once_with("saved-state")


def test_probe_does_not_disable_strict_restore(monkeypatch, strict_resume_provider):
    args, loader, _, load_state, _ = strict_resume_provider
    monkeypatch.setattr(dataloader_provider.os.path, "exists", lambda _: False)
    dataloader_provider.train_valid_test_dataloaders_provider(
        None, task_encoder=object(), restore_dataloader_state=False, pg_collection=_pg_collection()
    )
    with pytest.raises(FileNotFoundError, match="refusing to resume"):
        dataloader_provider.train_valid_test_dataloaders_provider(
            None, task_encoder=object(), pg_collection=_pg_collection()
        )
    assert args.strict_dataloader_state_load
    load_state.assert_not_called()
    loader.restore_state_rank.assert_not_called()


def test_loaders_shard_over_the_collection_data_parallel_group(monkeypatch, strict_resume_provider):
    args, _, checkpoint_name, _, _ = strict_resume_provider
    args.strict_dataloader_state_load = False
    worker_configs = []
    monkeypatch.setattr(
        dataloader_provider,
        "datasets_provider",
        lambda task_encoder, worker_config: worker_configs.append(worker_config)
        or (object(), None, None),
    )
    monkeypatch.setattr(dataloader_provider.os.path, "exists", lambda _: False)
    pg_collection = _pg_collection(dp_rank=3, dp_size=4)

    with forbid_global_process_groups():
        dataloader_provider.train_valid_test_dataloaders_provider(
            None, task_encoder=object(), pg_collection=pg_collection
        )

    (worker_config,) = worker_configs
    assert worker_config["rank"] == 3
    assert worker_config["world_size"] == 4
    assert worker_config["data_parallel_group"] is pg_collection.dp_gtp_remat
    # The dataloader state is saved per data-parallel rank of the collection.
    assert checkpoint_name.call_args.kwargs["basename"] == "train_dataloader_dprank003.pt"


@pytest.mark.parametrize("pp_size", [1, 2])
def test_restore_reads_the_state_file_the_checkpoint_writes(monkeypatch, tmp_path, pp_size):
    """The provider restores from the file maybe_save_dataloader_state wrote on the same groups.

    Both sides name the file with the real get_checkpoint_name, under strict process-group
    fallbacks and with every read of the global grid forbidden.
    """
    from megatron.training import checkpointing

    args = SimpleNamespace(
        load="/checkpoints/model",
        dataloader_save=str(tmp_path),
        strict_dataloader_state_load=True,
        num_workers=0,
        dataloader_seed=0,
        packing_buffer_size=10000,
    )
    loader = MagicMock()
    monkeypatch.setattr(dataloader_provider, "get_args", lambda: args)
    monkeypatch.setattr(
        dataloader_provider, "get_train_state", lambda: SimpleNamespace(iteration=100)
    )
    monkeypatch.setattr(dataloader_provider, "is_dataloader_rank", lambda *a, **kw: True)
    monkeypatch.setattr(dataloader_provider, "WorkerConfig", lambda **kw: kw)
    monkeypatch.setattr(dataloader_provider, "datasets_provider", lambda *a: (object(), None, None))
    monkeypatch.setattr(dataloader_provider, "use_new_dataloader_path", lambda: True)
    monkeypatch.setattr(dataloader_provider, "FileStoreCachePool", lambda **kw: None)
    monkeypatch.setattr(dataloader_provider, "get_savable_loader", lambda *a, **kw: loader)
    # The writer reads ranks through these helpers, which ignore the stand-in groups when
    # torch.distributed is not initialized. Its barriers need real groups.
    monkeypatch.setattr(checkpointing, "get_pg_rank", lambda group: group.rank())
    monkeypatch.setattr(checkpointing, "get_pg_size", lambda group: group.size())
    monkeypatch.setattr(checkpointing.torch.distributed, "barrier", lambda **kwargs: None)
    monkeypatch.setattr(process_groups_config, "_warned_global_process_group_fallbacks", set())

    with warnings.catch_warnings():
        process_groups_config.enable_strict_process_group_fallbacks()
        with forbid_global_process_groups():
            # Data rank 0 creates the iteration directory. Data rank 3 writes its own file,
            # then restores from it below.
            for dp_rank in (0, 3):
                pg_collection = _pg_collection(pp_size=pp_size, dp_rank=dp_rank, dp_size=4)
                iterable = SimpleNamespace(save_state=lambda dp_rank=dp_rank: {"dp": dp_rank})
                checkpointing.maybe_save_dataloader_state(
                    SimpleNamespace(iterable=iterable),
                    100,
                    args.dataloader_save,
                    tp_group=pg_collection.tp,
                    pp_group=pg_collection.pp,
                    dp_group=pg_collection.dp_gtp_remat,
                    cp_group=pg_collection.cp,
                )
            dataloader_provider.train_valid_test_dataloaders_provider(
                None, task_encoder=object(), pg_collection=pg_collection
            )

    model_parallel_dir = "mp_rank_00_000" if pp_size > 1 else "mp_rank_00"
    saved = tmp_path / "iter_0000100" / model_parallel_dir / "train_dataloader_dprank003.pt"
    assert saved.is_file()
    loader.restore_state_rank.assert_called_once_with({"dp": 3})


def test_loaders_shard_over_the_data_parallel_group_without_gtp_fields(
    monkeypatch, strict_resume_provider
):
    """A collection without the GTP weight-remat fields shards over its replicate group."""
    args, _, _, _, _ = strict_resume_provider
    args.load = None
    worker_configs = []
    monkeypatch.setattr(
        dataloader_provider,
        "datasets_provider",
        lambda task_encoder, worker_config: worker_configs.append(worker_config)
        or (object(), None, None),
    )
    pg_collection = ProcessGroupCollection(
        tp=_Group(0, 2), cp=_Group(0, 2), pp=_Group(0, 1), dp=_Group(1, 2)
    )

    with forbid_global_process_groups():
        dataloader_provider.train_valid_test_dataloaders_provider(
            None, task_encoder=object(), pg_collection=pg_collection
        )

    (worker_config,) = worker_configs
    assert worker_config["data_parallel_group"] is pg_collection.dp
    assert (worker_config["rank"], worker_config["world_size"]) == (1, 2)


@pytest.mark.filterwarnings(
    "error::megatron.core.process_groups_config.ProcessGroupFallbackWarning"
)
def test_full_dataset_probe_builds_the_loader_on_the_run_collection(
    monkeypatch, strict_resume_provider, run_config
):
    """The --train-full-dataset probe passes the run's groups, so the provider does not warn."""
    from megatron.training import training

    _, loader, _, _, _ = strict_resume_provider
    monkeypatch.setattr(process_groups_config, "_warned_global_process_group_fallbacks", set())
    monkeypatch.setattr(
        dataloader_provider.train_valid_test_dataloaders_provider,
        "supports_train_full_dataset",
        True,
        raising=False,
    )
    # The probe passes no task encoder, and the default TaskEncoder reads the global args.
    monkeypatch.setattr(dataloader_provider, "TaskEncoder", object)
    reduce_sum = Mock(return_value=84)
    monkeypatch.setattr(training, "_reduce_sum_across_data_parallel_group", reduce_sum)
    monkeypatch.setattr(
        training, "reduce_max_stat_across_model_parallel_group", Mock(return_value=84)
    )
    pg_collection = _pg_collection(dp_rank=1, dp_size=2)
    pg_collection.mp = _Group()
    args = SimpleNamespace(deduplicate_dataloader_across_context_parallel=False)

    with forbid_global_process_groups():
        sample_count = training._get_train_full_dataset_sample_count(
            args, dataloader_provider.train_valid_test_dataloaders_provider, pg_collection
        )

    assert sample_count == 84
    assert reduce_sum.call_args.args[0] == len(loader)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA to build the global grid")
def test_provider_without_collection_warns_and_uses_the_global_grid(monkeypatch):
    """Callers that omit pg_collection keep the global groups, with a deprecation warning."""
    monkeypatch.setattr(process_groups_config, "_warned_global_process_group_fallbacks", set())
    args = SimpleNamespace(load=None, num_workers=0, dataloader_seed=0, packing_buffer_size=None)
    worker_configs = []
    monkeypatch.setattr(dataloader_provider, "get_args", lambda: args)
    monkeypatch.setattr(dataloader_provider, "WorkerConfig", lambda **kwargs: kwargs)
    monkeypatch.setattr(
        dataloader_provider,
        "datasets_provider",
        lambda task_encoder, worker_config: worker_configs.append(worker_config)
        or (object(), None, None),
    )
    monkeypatch.setattr(dataloader_provider, "get_savable_loader", lambda *a, **kw: [])
    Utils.initialize_model_parallel()
    try:
        # Only the resolver in process_groups_config.py may read the global grid.
        with forbid_global_process_groups(forbid_shim=False):
            with pytest.warns(
                ProcessGroupFallbackWarning, match="deprecated since Megatron Core 0.21"
            ):
                dataloader_provider.train_valid_test_dataloaders_provider(
                    None, task_encoder=object()
                )
        expected_group = parallel_state.get_data_parallel_group()
    finally:
        Utils.destroy_model_parallel()

    (worker_config,) = worker_configs
    assert worker_config["data_parallel_group"] is expected_group
    assert worker_config["rank"] == expected_group.rank()
    assert worker_config["world_size"] == expected_group.size()
