# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Disabled evaluation does not create samplers for empty mock splits."""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.datasets.utils import Split
from megatron.training.datasets import data_samplers


class EmptySplit:
    def __init__(self, split):
        self.index_split = split

    def __len__(self):
        return 0


@pytest.mark.parametrize("split", [Split.train, Split.valid, Split.test])
@pytest.mark.parametrize("eval_iters", [0, 1])
def test_empty_split_respects_evaluation_setting(monkeypatch, split, eval_iters):
    args = SimpleNamespace(
        dataloader_type="single",
        eval_iters=eval_iters,
        full_validation=False,
        micro_batch_size=1,
        eval_micro_batch_size=1,
    )
    monkeypatch.setattr(data_samplers, "get_args", lambda: args)
    monkeypatch.setattr(data_samplers.mpu, "get_data_parallel_rank", lambda: 0)
    monkeypatch.setattr(data_samplers.mpu, "get_data_parallel_world_size", lambda: 1)
    dataset = EmptySplit(split)
    # Upstream dev treats any empty split as absent before sampler construction.
    assert data_samplers.build_pretraining_data_loader(dataset, 0) is None


def test_restart_iterator_preserves_training_rng(monkeypatch):
    args = SimpleNamespace(
        dataloader_type="single",
        eval_iters=1,
        full_validation=False,
        micro_batch_size=1,
        num_workers=0,
        use_varlen_dataset=False,
        dynamic_context_parallel=False,
        sequence_packing_scheduler=None,
    )
    monkeypatch.setattr(data_samplers, "get_args", lambda: args)
    monkeypatch.setattr(data_samplers.mpu, "get_data_parallel_rank", lambda: 0)
    monkeypatch.setattr(data_samplers.mpu, "get_data_parallel_world_size", lambda: 1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(129)
        expected_rng = torch.get_rng_state().clone()
        continuous = iter(data_samplers.build_pretraining_data_loader(list(range(8)), 0))
        assert next(continuous).item() == 0
        resumed = iter(data_samplers.build_pretraining_data_loader(list(range(8)), 1))
        assert next(continuous).item() == next(resumed).item() == 1
        assert torch.equal(torch.get_rng_state(), expected_rng)
