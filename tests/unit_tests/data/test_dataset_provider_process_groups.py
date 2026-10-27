# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Training passes its process groups to the dataset provider.

A provider that accepts a ``pg_collection`` receives the run's collection, so it can build its
datasets on the ranks of that collection rather than on the global grid.

Tests that pass groups make every global parallel_state accessor fail, or give the code groups
whose layout differs from the global grid, so a read of the global grid cannot pass unnoticed.
"""

import functools
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.training import training
from megatron.training.config import TrainingConfig, ValidationConfig
from megatron.training.state import TrainState
from tests.unit_tests.test_utilities import Utils

# Tests of the explicit path fail on any fallback warning, as under strict mode.
no_fallback = pytest.mark.filterwarnings(
    "error::megatron.core.process_groups_config.ProcessGroupFallbackWarning"
)


class _FakeGroup:
    """Process-group stand-in with a rank, a size and the global ranks it spans."""

    def __init__(self, rank, ranks):
        self._rank = rank
        self.ranks = list(ranks)

    def rank(self):
        return self._rank

    def size(self):
        return len(self.ranks)


def _forbid_global_process_groups(monkeypatch):
    """Fail the test on any read of the global groups, ranks, sizes or pipeline stages."""

    def fail(*args, **kwargs):
        pytest.fail("the data path must not read the global parallel state")

    for name, value in list(vars(parallel_state).items()):
        if name.startswith(("get_", "is_pipeline_")) and callable(value):
            monkeypatch.setattr(parallel_state, name, fail)
    monkeypatch.setattr(
        ProcessGroupCollection,
        "use_mpu_process_groups",
        Mock(side_effect=AssertionError("supplied groups must not be rebuilt from globals")),
    )


def _groups(tp_rank, pp_rank, pp_size):
    """Groups whose global ranks differ from any global grid of the test's world."""
    return ProcessGroupCollection(
        tp=_FakeGroup(tp_rank, [5, 7]), pp=_FakeGroup(pp_rank, range(11, 11 + pp_size))
    )


# build_train_valid_test_datasets


def _recording_provider(calls, with_vp_stage):
    """A provider with a pg_collection parameter that records the arguments it receives."""

    def provider(train_val_test_num_samples, vp_stage=None, pg_collection=None):
        calls.append((train_val_test_num_samples, vp_stage, pg_collection))
        return "train", "valid", "test"

    if not with_vp_stage:
        return provider
    # Training binds the model chunk of a virtual pipeline stage this way.
    vp_provider = functools.partial(provider, vp_stage=1)
    vp_provider.is_distributed = True
    return vp_provider


@no_fallback
@pytest.mark.parametrize("with_vp_stage", [False, True], ids=["plain", "vp_stage_partial"])
def test_dataset_provider_with_a_pg_collection_parameter_receives_the_run_groups(
    monkeypatch, with_vp_stage
):
    _forbid_global_process_groups(monkeypatch)
    pgs = _groups(0, 0, 1)
    calls = []
    provider = _recording_provider(calls, with_vp_stage)

    datasets = training.build_train_valid_test_datasets(provider, [4, 2, 1], pg_collection=pgs)

    assert datasets == ("train", "valid", "test")
    assert calls == [([4, 2, 1], 1 if with_vp_stage else None, pgs)]


@no_fallback
@pytest.mark.parametrize("signature", ["sizes_only", "var_keyword"])
def test_dataset_provider_without_a_pg_collection_parameter_is_called_with_the_sizes(
    monkeypatch, signature
):
    _forbid_global_process_groups(monkeypatch)
    calls = []
    if signature == "sizes_only":

        def provider(train_val_test_num_samples):
            calls.append(((train_val_test_num_samples,), {}))
            return None, None, None

    else:
        # A provider that forwards **kwargs to another one does not declare pg_collection.
        def provider(*args, **kwargs):
            calls.append((args, kwargs))
            return None, None, None

    training.build_train_valid_test_datasets(provider, [4, 2, 1], pg_collection=_groups(0, 0, 1))

    assert calls == [(([4, 2, 1],), {})]


# 4-GPU tests


def _strided_tp_group(world_size):
    """Pair rank r with rank r + world_size // 2, unlike the global grid's adjacent TP pairs."""
    rank = torch.distributed.get_rank()
    half = world_size // 2
    tp_group = None
    for first in range(half):
        ranks = [first, first + half]
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            tp_group = group
    return tp_group


def _build_loaders(monkeypatch, run_config, pg_collection, provider_calls, provider_accepts_groups):
    """Run build_train_valid_test_data_loaders with a provider that records its calls."""
    run_config.train = TrainingConfig(train_samples=1, train_iters=1, global_batch_size=1)
    run_config.validation = ValidationConfig(
        eval_interval=1, eval_iters=1, eval_global_batch_size=1
    )
    args = SimpleNamespace(dataloader_type="single", phase_transition_iterations=None)

    if provider_accepts_groups:

        def provider(train_valid_test_num_samples, pg_collection=None):
            provider_calls.append(pg_collection)
            return range(1), range(1), range(1)

    else:

        def provider(train_valid_test_num_samples):
            provider_calls.append(None)
            return range(1), range(1), range(1)

    monkeypatch.setattr(training, "get_args", lambda: args)
    monkeypatch.setattr(training, "get_train_state", lambda: TrainState())
    monkeypatch.setattr(
        training,
        "build_pretraining_data_loader",
        lambda dataset, consumed_samples, pg_collection=None: dataset,
    )
    return training.build_train_valid_test_data_loaders(provider, pg_collection=pg_collection)


@no_fallback
def test_data_loaders_build_datasets_on_the_run_tensor_parallel_group(monkeypatch, run_config):
    world_size = Utils.world_size
    if world_size < 4 or world_size % 2:
        pytest.skip("needs an even world size of at least 4")
    Utils.initialize_model_parallel(tensor_model_parallel_size=2)
    try:
        tp_group = _strided_tp_group(world_size)
        pgs = ProcessGroupCollection(tp=tp_group)
        provider_calls = []

        with monkeypatch.context() as patch:
            _forbid_global_process_groups(patch)
            _build_loaders(patch, run_config, pgs, provider_calls, provider_accepts_groups=True)

        # The first rank of each strided pair builds; on the global grid, ranks 0, 2, ... would.
        assert provider_calls == ([pgs] if tp_group.rank() == 0 else [])
    finally:
        Utils.destroy_model_parallel()
