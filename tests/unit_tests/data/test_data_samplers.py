# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Restoring a checkpoint then building the data iterators must leave the CPU RNG where it was.

Mirrors pretrain(): _set_random_seed, then load_checkpoint restores the saved state, then
build_train_valid_test_data_iterators. Nothing between the restore and the first step draws from
the default CPU generator -- except, on the pre-fix code, iterator creation itself.
"""

from collections import Counter
from types import SimpleNamespace

import torch

import megatron.training.datasets.data_samplers as data_samplers
from tests.unit_tests.test_utilities import Utils

SEED = 1234


def _set_random_seed():
    """What _set_random_seed does at process start (tp=pp=1, so it collapses to SEED)."""
    torch.manual_seed(SEED)


def _load_checkpoint_rng(saved_rng):
    """What load_checkpoint does with the saved rng_state, before the data iterators exist."""
    torch.set_rng_state(saved_rng)


class _Dataset(torch.utils.data.Dataset):
    def __len__(self):
        return 64

    def __getitem__(self, idx):
        return torch.tensor([idx])


class _RandomValueDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 10

    def __getitem__(self, idx):
        return idx, torch.rand(1).item()


class TestDataLoaderResume:
    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_building_data_iterators_does_not_move_cpu_rng(self, monkeypatch):
        Utils.initialize_model_parallel(1, 1)
        monkeypatch.setattr(
            data_samplers,
            'get_args',
            lambda: SimpleNamespace(
                dataloader_type='single',
                micro_batch_size=1,
                global_batch_size=Utils.world_size,
                num_workers=0,
                hybrid_context_parallel=False,
                sequence_packing_scheduler=None,
                use_varlen_dataset=False,
                varlen_sbhd_validation=False,
            ),
        )
        steps_before_save = 3

        _set_random_seed()
        loader = data_samplers.build_pretraining_data_loader(_Dataset(), consumed_samples=0)
        it = iter(loader)
        for _ in range(steps_before_save):
            next(it)
        saved_rng = torch.get_rng_state()  # what save_checkpoint stores
        continuous_rest = [b.tolist() for b in it]

        # The resumed process, in pretrain()'s order.
        _set_random_seed()
        _load_checkpoint_rng(saved_rng)
        resumed = data_samplers.build_pretraining_data_loader(
            _Dataset(), consumed_samples=steps_before_save * Utils.world_size
        )
        resumed_rest = [b.tolist() for b in resumed]

        # Data position rides on consumed_samples, not RNG, so this holds either way -- it pins
        # that independence rather than guarding the fix.
        assert resumed_rest == continuous_rest, (
            f"resumed loader did not continue where the saving run stopped: "
            f"got {resumed_rest[:2]}, expected {continuous_rest[:2]}"
        )
        assert torch.equal(torch.get_rng_state(), saved_rng), (
            "building the data iterators moved the CPU RNG off the state load_checkpoint "
            "restored, so the resumed run diverges from the run that saved"
        )


def test_cyclic_sampler_no_sharding_preserves_samples_after_dp_change():
    dataset_size, micro_batch_size, global_batch_size, steps = 250, 2, 16, 32

    def draws(dp_before, dp_after, resume_step):
        drawn = []
        for dp, lo, hi in ((dp_before, 0, resume_step), (dp_after, resume_step, steps)):
            grad_accumulation = global_batch_size // (micro_batch_size * dp)
            samplers = [
                data_samplers.MegatronPretrainingRandomSampler(
                    torch.arange(dataset_size),
                    total_samples=dataset_size,
                    consumed_samples=lo * global_batch_size,
                    micro_batch_size=micro_batch_size,
                    data_parallel_rank=rank,
                    data_parallel_size=dp,
                    data_sharding=False,
                    global_batch_size=global_batch_size,
                )
                for rank in range(dp)
            ]
            iterators = [iter(sampler) for sampler in samplers]
            for _ in range(lo, hi):
                for rank in range(dp):
                    for _ in range(grad_accumulation):
                        try:
                            drawn.extend(next(iterators[rank]))
                        except StopIteration:
                            iterators[rank] = iter(samplers[rank])
                            drawn.extend(next(iterators[rank]))
        return Counter(drawn)

    for dp_before, dp_after, resume_step in ((1, 8, 15), (8, 1, 15), (1, 2, 7)):
        reference = draws(dp_before, dp_before, 0)
        resumed = draws(dp_before, dp_after, resume_step)
        assert resumed == reference


def test_cyclic_sampler_updates_random_seed_epoch_across_global_batch_boundary():
    dataset = data_samplers.RandomSeedDataset(_RandomValueDataset(), seed=1234)
    sampler = data_samplers.MegatronPretrainingRandomSampler(
        dataset,
        total_samples=len(dataset),
        consumed_samples=0,
        micro_batch_size=2,
        data_parallel_rank=0,
        data_parallel_size=1,
        data_sharding=False,
        global_batch_size=6,
    )

    observed = []
    for batch in sampler:
        observed.extend(dataset[index] for index in batch)

    expected = []
    for epoch, offset, count in ((0, 0, 10), (1, 0, 2)):
        generator = torch.Generator().manual_seed(epoch)
        permutation = torch.randperm(len(dataset), generator=generator).tolist()
        for index in permutation[offset : offset + count]:
            value_generator = torch.Generator().manual_seed(index + dataset.base_seed + epoch)
            expected.append((index, torch.rand(1, generator=value_generator).item()))

    assert [index for index, _ in observed] == [index for index, _ in expected]
    torch.testing.assert_close(
        torch.tensor([value for _, value in observed]),
        torch.tensor([value for _, value in expected]),
    )
    assert dataset.curr_seed == dataset.base_seed + 1
