# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Restoring a checkpoint then building the data iterators must leave the CPU RNG where it was.

Mirrors pretrain(): _set_random_seed, then load_checkpoint restores the saved state, then
build_train_valid_test_data_iterators. Nothing between the restore and the first step draws from
the default CPU generator -- except, on the pre-fix code, iterator creation itself.
"""

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
