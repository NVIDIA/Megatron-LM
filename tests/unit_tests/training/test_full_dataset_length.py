# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from megatron.core import parallel_state
from megatron.training import training
from megatron.training.training import _reduce_sum_across_data_parallel_group
from megatron.training.utils import reduce_max_stat_across_model_parallel_group
from tests.unit_tests.test_utilities import Utils


def test_cp_deduplicated_dataloader_length_reaches_every_rank():
    Utils.initialize_model_parallel(tensor_model_parallel_size=2, context_parallel_size=2)

    # Match the CP-deduplicated multimodal loader: TP=0/CP=0 owns one
    # data-parallel shard length, and every other model-parallel rank has None.
    shard_length = (
        10.0
        if parallel_state.get_tensor_model_parallel_rank() == 0
        and parallel_state.get_context_parallel_rank() == 0
        else None
    )
    total_length = _reduce_sum_across_data_parallel_group(shard_length, with_context_parallel=True)
    total_length = reduce_max_stat_across_model_parallel_group(total_length)

    expected_length = 10.0 * parallel_state.get_data_parallel_world_size()
    assert total_length == expected_length

    Utils.destroy_model_parallel()


@pytest.mark.parametrize("iteration", [None, 0, 100])
@pytest.mark.parametrize("owns_data", [False, True])
def test_length_probe_preserves_iteration_and_skips_restore(monkeypatch, iteration, owns_data):
    args = SimpleNamespace(
        train_iters=None, train_samples=None, deduplicate_dataloader_across_context_parallel=True
    )
    if iteration is not None:
        args.iteration = iteration
    before = vars(args).copy()
    iterator = SimpleNamespace(_dataloader=range(42)) if owns_data else None
    provider = Mock(return_value=(iterator, None, None), supports_train_full_dataset=True)
    reduce_sum = Mock(return_value=84 if owns_data else None)
    reduce_max = Mock(return_value=84)
    monkeypatch.setattr(training, "_reduce_sum_across_data_parallel_group", reduce_sum)
    monkeypatch.setattr(training, "reduce_max_stat_across_model_parallel_group", reduce_max)

    assert training._get_train_full_dataset_sample_count(args, provider) == 84
    assert vars(args) == before
    provider.assert_called_once_with(None, restore_dataloader_state=False)
    reduce_sum.assert_called_once_with(42 if owns_data else None, with_context_parallel=True)
    reduce_max.assert_called_once_with(84 if owns_data else None)


@pytest.mark.parametrize("field", ["train_iters", "train_samples"])
def test_length_probe_rejects_explicit_horizon(field):
    args = SimpleNamespace(train_iters=None, train_samples=None)
    setattr(args, field, 10)
    provider = Mock(supports_train_full_dataset=True)
    with pytest.raises(ValueError, match="cannot be combined"):
        training._get_train_full_dataset_sample_count(args, provider)
    provider.assert_not_called()


def test_length_probe_rejects_provider_without_opt_in():
    provider = Mock(supports_train_full_dataset=False)
    with pytest.raises(ValueError, match="supports_train_full_dataset"):
        training._get_train_full_dataset_sample_count(SimpleNamespace(), provider)
    provider.assert_not_called()
