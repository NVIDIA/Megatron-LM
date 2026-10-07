# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Global batch boundaries and checkpoint positions for the cyclic data sampler."""

import pytest
import torch

from megatron.training.datasets.data_samplers import MegatronPretrainingRandomSampler


def _sampler(total, micro_batch_size, world_size, rank, consumed_samples=0, data_sharding=False):
    dataset = torch.arange(total)
    sampler = MegatronPretrainingRandomSampler(
        dataset,
        total_samples=total,
        consumed_samples=consumed_samples,
        micro_batch_size=micro_batch_size,
        data_parallel_rank=rank,
        data_parallel_size=world_size,
        data_sharding=data_sharding,
    )
    loader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler, num_workers=0)
    return sampler, loader


def _shuffle(total, micro_batch_size, epoch):
    generator = torch.Generator().manual_seed(epoch)
    return torch.randperm(
        total // micro_batch_size * micro_batch_size, generator=generator
    ).tolist()


@pytest.mark.parametrize(
    'total,micro_batch_size,world_size', [(10, 2, 3), (5, 1, 2), (14, 2, 4), (26, 3, 4)]
)
def test_random_sampler_drops_incomplete_global_batches(total, micro_batch_size, world_size):
    global_micro_batch_size = micro_batch_size * world_size
    active = total // global_micro_batch_size * global_micro_batch_size
    samplers_and_loaders = [
        _sampler(total, micro_batch_size, world_size, rank) for rank in range(world_size)
    ]

    for epoch in range(2):
        batches = [[batch.tolist() for batch in loader] for _, loader in samplers_and_loaders]
        assert [len(rank_batches) for rank_batches in batches] == [
            active // global_micro_batch_size
        ] * world_size
        assert all(
            len(batch) == micro_batch_size for rank_batches in batches for batch in rank_batches
        )
        # Reconstruct the shared shuffled stream from its interleaved rank slices.
        sampled = [
            batches[rank][step][item]
            for step in range(active // global_micro_batch_size)
            for item in range(micro_batch_size)
            for rank in range(world_size)
        ]
        assert sampled == _shuffle(total, micro_batch_size, epoch)[:active]
        assert len(set(sampled)) == active
        for sampler, _ in samplers_and_loaders:
            assert sampler.epoch == epoch
            assert sampler.consumed_samples == (epoch + 1) * active


@pytest.mark.parametrize('consumed_samples', [0, 6, 12, 18, 24, 30])
def test_random_sampler_resume_matches_epoch_suffix(consumed_samples):
    total, micro_batch_size, world_size = 23, 2, 3
    active = 18
    epoch, offset = divmod(consumed_samples, active)
    for rank in range(world_size):
        uninterrupted, loader = _sampler(total, micro_batch_size, world_size, rank, epoch * active)
        full_epoch = [batch.tolist() for batch in loader]
        resumed, resumed_loader = _sampler(
            total, micro_batch_size, world_size, rank, consumed_samples
        )
        resumed_batches = [batch.tolist() for batch in resumed_loader]
        assert resumed_batches == full_epoch[offset // (micro_batch_size * world_size) :]
        expected = _shuffle(total, micro_batch_size, epoch)[offset:active][rank::world_size]
        assert [sample for batch in resumed_batches for sample in batch] == expected
        assert resumed.consumed_samples == uninterrupted.consumed_samples == (epoch + 1) * active
        assert resumed.epoch == epoch


@pytest.mark.parametrize('total,micro_batch_size,world_size', [(24, 2, 3), (24, 3, 4), (7, 2, 1)])
@pytest.mark.parametrize('epoch', [0, 2])
def test_random_sampler_preserves_existing_complete_batch_order(
    total, micro_batch_size, world_size, epoch
):
    active = total // (micro_batch_size * world_size) * micro_batch_size * world_size
    # These layouts already yielded equally sized rank streams. Preserve their exact RNG order,
    # including the single-rank case where only a partial local microbatch is dropped.
    for rank in range(world_size):
        sampler, loader = _sampler(total, micro_batch_size, world_size, rank, epoch * active)
        actual = [sample for batch in loader for sample in batch.tolist()]
        assert actual == _shuffle(total, micro_batch_size, epoch)[rank::world_size]
        assert sampler.consumed_samples == (epoch + 1) * active


@pytest.mark.parametrize('total,micro_batch_size,world_size', [(10, 2, 3), (24, 2, 3)])
def test_random_sampler_preserves_data_sharding(total, micro_batch_size, world_size):
    active = total // (micro_batch_size * world_size) * micro_batch_size * world_size
    bucket_size = active // world_size
    samplers_and_loaders = [
        _sampler(total, micro_batch_size, world_size, rank, data_sharding=True)
        for rank in range(world_size)
    ]
    for epoch in range(2):
        all_samples = []
        for rank, (sampler, loader) in enumerate(samplers_and_loaders):
            actual = [sample for batch in loader for sample in batch.tolist()]
            generator = torch.Generator().manual_seed(epoch)
            expected = [
                rank * bucket_size + index
                for index in torch.randperm(bucket_size, generator=generator).tolist()
            ]
            assert actual == expected
            assert sampler.consumed_samples == (epoch + 1) * active
            all_samples.extend(actual)
        assert sorted(all_samples) == list(range(active))
