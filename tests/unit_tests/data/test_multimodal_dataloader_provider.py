# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

MULTIMODAL_EXAMPLE_DIR = Path(__file__).resolve().parents[3] / "examples" / "multimodal"
sys.path.insert(0, str(MULTIMODAL_EXAMPLE_DIR))

import dataloader_provider  # noqa: E402


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
def test_dataloader_rank_cp_deduplication(monkeypatch, tp_rank, cp_rank, deduplicate, expected):
    monkeypatch.setattr(dataloader_provider, "get_tensor_model_parallel_rank", lambda: tp_rank)
    monkeypatch.setattr(dataloader_provider, "get_context_parallel_rank", lambda: cp_rank)
    monkeypatch.setattr(dataloader_provider, "get_pipeline_model_parallel_world_size", lambda: 1)

    assert (
        dataloader_provider.is_dataloader_rank(
            encoder_pipeline_model_parallel_size=0, deduplicate_across_context_parallel=deduplicate
        )
        is expected
    )


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
    monkeypatch.setattr(dataloader_provider.parallel_state, "get_data_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        dataloader_provider.parallel_state, "get_data_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(
        dataloader_provider.parallel_state, "get_data_parallel_group", lambda: object()
    )
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
            train_val_test_num_samples=None, task_encoder=object()
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
        iteration=100,
        load="/checkpoints/model",
        dataloader_save="/checkpoints/dataloader",
        strict_dataloader_state_load=True,
        num_workers=0,
        dataloader_seed=0,
        packing_buffer_size=10000,
    )
    loader = MagicMock()
    loader.__len__.return_value = 42
    checkpoint_name = Mock(return_value="/checkpoints/dataloader/iter_0000100/state.pt")
    load_state = Mock(return_value={"dataloader_state_dict": "saved-state"})
    monkeypatch.setattr(dataloader_provider, "get_args", lambda: args)
    monkeypatch.setattr(dataloader_provider, "is_dataloader_rank", lambda *a, **kw: True)
    for name, value in (
        ("get_data_parallel_rank", 0),
        ("get_data_parallel_world_size", 1),
        ("get_data_parallel_group", None),
    ):
        monkeypatch.setattr(dataloader_provider.parallel_state, name, Mock(return_value=value))
    monkeypatch.setattr(dataloader_provider, "WorkerConfig", lambda **kw: kw)
    monkeypatch.setattr(dataloader_provider, "datasets_provider", lambda *a: (object(), None, None))
    monkeypatch.setattr(dataloader_provider, "use_new_dataloader_path", lambda: True)
    monkeypatch.setattr(dataloader_provider, "FileStoreCachePool", lambda **kw: None)
    monkeypatch.setattr(dataloader_provider, "get_savable_loader", lambda *a, **kw: loader)
    monkeypatch.setattr(dataloader_provider, "get_checkpoint_name", checkpoint_name)
    monkeypatch.setattr(dataloader_provider.torch, "load", load_state)
    return args, loader, checkpoint_name, load_state


@pytest.mark.parametrize("iteration", [0, 100])
@pytest.mark.parametrize("state_exists", [False, True])
def test_length_probe_never_restores_state(
    monkeypatch, strict_resume_provider, iteration, state_exists
):
    args, loader, checkpoint_name, load_state = strict_resume_provider
    args.iteration = iteration
    monkeypatch.setattr(dataloader_provider.os.path, "exists", lambda _: state_exists)

    train, _, _ = dataloader_provider.train_valid_test_dataloaders_provider(
        None, task_encoder=object(), restore_dataloader_state=False
    )

    assert len(train._dataloader) == 42
    assert args.iteration == iteration
    checkpoint_name.assert_not_called()
    load_state.assert_not_called()
    loader.restore_state_rank.assert_not_called()


def test_default_restore_uses_resolved_iteration(monkeypatch, strict_resume_provider):
    args, loader, checkpoint_name, load_state = strict_resume_provider
    monkeypatch.setattr(dataloader_provider.os.path, "exists", lambda _: True)
    dataloader_provider.train_valid_test_dataloaders_provider(None, task_encoder=object())
    assert checkpoint_name.call_args.args[:2] == (args.dataloader_save, 100)
    load_state.assert_called_once()
    loader.restore_state_rank.assert_called_once_with("saved-state")


def test_probe_does_not_disable_strict_restore(monkeypatch, strict_resume_provider):
    args, loader, _, load_state = strict_resume_provider
    monkeypatch.setattr(dataloader_provider.os.path, "exists", lambda _: False)
    dataloader_provider.train_valid_test_dataloaders_provider(
        None, task_encoder=object(), restore_dataloader_state=False
    )
    with pytest.raises(FileNotFoundError, match="refusing to resume"):
        dataloader_provider.train_valid_test_dataloaders_provider(None, task_encoder=object())
    assert args.strict_dataloader_state_load
    load_state.assert_not_called()
    loader.restore_state_rank.assert_not_called()
