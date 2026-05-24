# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace

import torch

from megatron.training.datasets import data_samplers


class _TinyDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, idx):
        return idx


def test_megatron_pretraining_sampler_partitions_batches_by_data_parallel_rank():
    sampler = data_samplers.MegatronPretrainingSampler(
        total_samples=10,
        consumed_samples=0,
        micro_batch_size=2,
        data_parallel_rank=1,
        data_parallel_size=2,
    )

    assert len(sampler) == 10
    assert sampler.get_start_end_idx() == (2, 4)
    assert list(sampler) == [[2, 3], [6, 7]]


def test_megatron_pretraining_sampler_can_keep_partial_batch():
    sampler = data_samplers.MegatronPretrainingSampler(
        total_samples=10,
        consumed_samples=0,
        micro_batch_size=2,
        data_parallel_rank=0,
        data_parallel_size=2,
        drop_last=False,
    )

    assert list(sampler) == [[0, 1], [4, 5], [8, 9]]


def test_hybrid_context_parallel_sampler_returns_global_batch_slices():
    sampler = data_samplers.HybridCPMegatronPretrainingSampler(
        total_samples=8,
        consumed_samples=0,
        micro_batch_size=1,
        global_batch_size=4,
        data_parallel_rank=1,
        data_parallel_size=2,
    )

    assert sampler.get_start_end_idx_global_batch() == ([1, 3], [2, 4])
    assert list(sampler) == [[1, 3], [5, 7]]


def test_random_seed_dataset_updates_seed_per_epoch():
    base = [torch.rand(1).item() for _ in range(4)]
    dataset = data_samplers.RandomSeedDataset(base, seed=123)

    first = dataset[1]
    dataset.set_epoch(10)
    second = dataset[1]

    assert len(dataset) == 4
    assert first == base[1]
    assert second == base[1]
    assert dataset.curr_seed == 133


def test_random_sampler_yields_rank_local_random_batches():
    dataset = _TinyDataset()
    sampler = data_samplers.MegatronPretrainingRandomSampler(
        dataset,
        total_samples=8,
        consumed_samples=0,
        micro_batch_size=2,
        data_parallel_rank=1,
        data_parallel_size=2,
        data_sharding=False,
    )

    batches = list(sampler)

    assert len(sampler) == 8
    assert len(batches) == 2
    assert all(len(batch) == 2 for batch in batches)
    assert sorted(idx for batch in batches for idx in batch) != [0, 1, 2, 3]


def test_random_sampler_shards_data_by_rank():
    dataset = _TinyDataset()
    sampler = data_samplers.MegatronPretrainingRandomSampler(
        dataset,
        total_samples=8,
        consumed_samples=0,
        micro_batch_size=2,
        data_parallel_rank=1,
        data_parallel_size=2,
        data_sharding=True,
    )

    batches = list(sampler)

    assert len(batches) == 2
    assert all(4 <= idx < 8 for batch in batches for idx in batch)


def test_random_sampler_updates_wrapped_dataset_epoch_from_consumed_samples():
    dataset = data_samplers.RandomSeedDataset(_TinyDataset(), seed=17)
    sampler = data_samplers.MegatronPretrainingRandomSampler(
        dataset,
        total_samples=8,
        consumed_samples=8,
        micro_batch_size=2,
        data_parallel_rank=0,
        data_parallel_size=2,
        data_sharding=False,
    )

    list(sampler)

    assert dataset.curr_seed == 18


def test_build_pretraining_data_loader_handles_external_loader(monkeypatch):
    dataset = _TinyDataset()
    monkeypatch.setattr(
        data_samplers,
        "get_args",
        lambda: SimpleNamespace(dataloader_type="external"),
    )

    assert data_samplers.build_pretraining_data_loader(dataset, consumed_samples=0) is dataset


def test_build_pretraining_data_loader_uses_single_sampler(monkeypatch):
    dataset = _TinyDataset()
    args = SimpleNamespace(
        full_validation=False,
        micro_batch_size=2,
        global_batch_size=4,
        dataloader_type="single",
        hybrid_context_parallel=False,
        data_sharding=False,
        num_workers=0,
        exit_signal_handler=False,
        exit_signal=None,
    )
    monkeypatch.setattr(data_samplers, "get_args", lambda: args)
    monkeypatch.setattr(data_samplers.mpu, "get_data_parallel_rank", lambda: 0)
    monkeypatch.setattr(data_samplers.mpu, "get_data_parallel_world_size", lambda: 2)

    loader = data_samplers.build_pretraining_data_loader(dataset, consumed_samples=0)

    assert isinstance(loader.batch_sampler, data_samplers.MegatronPretrainingSampler)
    assert list(loader.batch_sampler) == [[0, 1], [4, 5]]
