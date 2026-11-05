# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The single-grid MIMO example takes its process groups from the language model it trains.

The groups here are stand-ins with fixed ranks, or a test grid built without the global one, so a
read of the global grid either raises or returns a group the assertions reject.
"""

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import megatron.training
from megatron.core import process_groups_config
from megatron.core.process_groups_config import ProcessGroupCollection, ProcessGroupFallbackWarning
from tests.unit_tests.test_utilities import Utils, forbid_global_process_groups

_MIMO_DIR = Path(__file__).resolve().parents[4] / "examples" / "mimo"


class _Group:
    """A stand-in process group with a fixed rank and size."""

    def __init__(self, rank=0, size=1):
        self._rank, self._size = rank, size

    def rank(self):
        return self._rank

    def size(self):
        return self._size


@pytest.fixture
def train(monkeypatch):
    """examples/mimo/train.py, imported with its data and model providers stubbed.

    The script imports its siblings as top-level modules; the stubs stand in for the ones these
    tests do not exercise, and utils.data_helpers is the real module.
    """
    from examples.mimo.utils import data_helpers

    def unused(*args, **kwargs):
        raise AssertionError("stubbed provider called")

    stubs = {
        "data": {},
        "data.energon_avlm_task_encoder": {"llava_avlm_dataloader_provider": unused},
        "data.energon_vlm_task_encoder": {"llava_vlm_dataloader_provider": unused},
        "data.mock": {"train_valid_test_datasets_provider": unused},
        "model_providers": {},
        "model_providers.llava_avlm": {"model_provider_llava_avlm": unused},
        "model_providers.llava_vlm": {"model_provider_llava_vlm": unused},
        "model_providers.mock": {"model_provider_mock_vlm_single_encoder": unused},
        "utils": {},
    }
    for name, attrs in stubs.items():
        module = types.ModuleType(name)
        module.__path__ = []
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setitem(sys.modules, "utils.data_helpers", data_helpers)
    spec = importlib.util.spec_from_file_location("mimo_example_train", _MIMO_DIR / "train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_forward_step_uses_the_language_model_groups(train, monkeypatch):
    pg_collection = ProcessGroupCollection(tp=_Group(), cp=_Group(), dp_gtp_remat=_Group())
    batch_groups = []

    def get_batch(data_iterator, pg_collection=None):
        batch_groups.append(pg_collection)
        return {}

    monkeypatch.setattr(train, "get_batch", get_batch)

    class _MimoModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = SimpleNamespace(pg_collection=pg_collection)

        def forward(self, **batch):
            return None, None

    with forbid_global_process_groups():
        _, loss_function = train.forward_step(iter([]), _MimoModel())

    assert batch_groups == [pg_collection]
    assert loss_function.keywords["pg_collection"] is pg_collection


def test_loss_func_reduces_over_the_language_model_groups(train, monkeypatch):
    monkeypatch.setattr(train, "get_args", lambda: SimpleNamespace(context_parallel_size=2))
    reduced_over = []
    monkeypatch.setattr(
        torch.distributed, "all_reduce", lambda tensor, group=None: reduced_over.append(group)
    )
    pg_collection = ProcessGroupCollection(cp=_Group(), dp_gtp_remat=_Group())

    with forbid_global_process_groups():
        train.loss_func(torch.ones(4), torch.ones(4), pg_collection=pg_collection)

    assert reduced_over == [pg_collection.cp, pg_collection.dp_gtp_remat]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.skipif(Utils.world_size % 2 != 0, reason="needs an even world size")
@pytest.mark.parametrize("grid_pg_collection", [{"tp": 2}], indirect=True)
def test_get_batch_broadcasts_over_the_language_model_tp_group(
    train, monkeypatch, grid_pg_collection
):
    monkeypatch.setattr(train, "get_args", lambda: SimpleNamespace(pipeline_model_parallel_size=1))
    tp_group = grid_pg_collection.tp
    source_batch = {
        "input_ids": torch.arange(8).view(1, 8) + torch.distributed.get_rank(),
        "modality_inputs": {"clip_encoder": {"images": torch.ones(1, 3, 2, 2)}},
    }
    source = torch.distributed.get_global_rank(tp_group, 0)
    data_iterator = iter([source_batch]) if tp_group.rank() == 0 else None

    # The global grid is not initialized. The guard also fails a warned fallback to it, which
    # would otherwise return None groups.
    with forbid_global_process_groups():
        batch = train.get_batch(data_iterator, pg_collection=grid_pg_collection)

    assert torch.equal(batch["input_ids"].cpu(), torch.arange(8).view(1, 8) + source)
    assert torch.equal(
        batch["modality_inputs"]["clip_encoder"]["images"].cpu(), torch.ones(1, 3, 2, 2)
    )


def _mock_data_args():
    return SimpleNamespace(
        micro_batch_size=1,
        image_size=8,
        image_seq_length=2,
        total_seq_length=8,
        pad_token_id=0,
        image_token_id=7,
    )


@pytest.mark.parametrize("tp_rank", [0, 1])
def test_mock_datasets_are_built_on_the_first_rank_of_the_collection_tp_group(monkeypatch, tp_rank):
    from examples.mimo.data import mock

    monkeypatch.setattr(megatron.training, "get_args", _mock_data_args)
    pg_collection = ProcessGroupCollection(tp=_Group(tp_rank, 2))

    with forbid_global_process_groups():
        train_ds, valid_ds, test_ds = mock.train_valid_test_datasets_provider(
            [4, 2, 0], pg_collection=pg_collection
        )

    if tp_rank == 0:
        assert (len(train_ds), len(valid_ds)) == (4, 2)
    else:
        assert (train_ds, valid_ds) == (None, None)
    assert test_ds is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA to build the global grid")
def test_mock_datasets_without_collection_warn_and_use_the_global_grid(monkeypatch):
    from examples.mimo.data import mock

    monkeypatch.setattr(process_groups_config, "_warned_global_process_group_fallbacks", set())
    monkeypatch.setattr(megatron.training, "get_args", _mock_data_args)
    Utils.initialize_model_parallel()
    try:
        # Only the resolver in process_groups_config.py may read the global grid.
        with forbid_global_process_groups(forbid_shim=False):
            with pytest.warns(ProcessGroupFallbackWarning, match="deprecated since Megatron Core"):
                train_ds, _, _ = mock.train_valid_test_datasets_provider([4, 2, 0])
    finally:
        Utils.destroy_model_parallel()

    # Every rank is the first rank of its tensor-parallel group of one.
    assert len(train_ds) == 4


def test_mock_model_provider_builds_every_module_on_its_groups(monkeypatch):
    from examples.mimo.model_providers import mock

    class _RecordingMimoModel:
        def __init__(self, mimo_config, **groups):
            self.mimo_config, self.groups = mimo_config, groups

    monkeypatch.setattr(mock, "MimoModel", _RecordingMimoModel)
    for name in (
        "get_mock_language_layer_spec",
        "get_mock_vision_layer_spec",
        "get_mock_projection_layer_spec",
    ):
        monkeypatch.setattr(mock, name, lambda: SimpleNamespace(submodules=None))
    pg_collection = ProcessGroupCollection(tp=_Group(), cp=_Group(), tp_cp=_Group())

    with forbid_global_process_groups():
        model = mock.model_provider_mock_vlm_single_encoder(pg_collection=pg_collection)

    config = model.mimo_config
    vision = config.modality_submodules_spec["images"]
    specs = {
        "language model": config.language_model_spec,
        "vision submodules": vision,
        "vision encoder": vision.submodules["encoders"]["clip_encoder"],
        "vision projection": vision.submodules["input_projections"][0],
    }
    for name, spec in specs.items():
        assert spec.params["pg_collection"] is pg_collection, name
    assert model.groups == {
        "cp_group": pg_collection.cp,
        "tp_group": pg_collection.tp,
        "tp_cp_group": pg_collection.tp_cp,
    }
    # Wrappers such as Float16Module take the model's groups from it.
    assert model.pg_collection is pg_collection
