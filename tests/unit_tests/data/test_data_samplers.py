# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Restoring a checkpoint then building the data iterators must leave the CPU RNG where it was.

Mirrors pretrain(): _set_random_seed, then load_checkpoint restores the saved state, then
build_train_valid_test_data_iterators. Nothing between the restore and the first step draws from
the default CPU generator -- except, on the pre-fix code, iterator creation itself.
"""

from types import SimpleNamespace

import pytest
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

    @pytest.mark.parametrize("data_seed", [None, 5678])
    def test_building_data_iterators_does_not_move_cpu_rng(
        self, monkeypatch, run_config, data_seed
    ):
        Utils.initialize_model_parallel(1, 1)
        run_config.train.micro_batch_size = 1
        run_config.train.global_batch_size = Utils.world_size
        run_config.rng.data_seed = data_seed
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


@pytest.fixture
def data_seed_loader_setup(monkeypatch, run_config):
    """Configure a small deterministic dataset without worker processes."""
    Utils.initialize_model_parallel(1, 1)
    run_config.train.micro_batch_size = 1
    run_config.train.global_batch_size = Utils.world_size
    monkeypatch.setattr(
        data_samplers,
        "get_args",
        lambda: SimpleNamespace(
            dataloader_type="single",
            num_workers=0,
            hybrid_context_parallel=False,
            sequence_packing_scheduler=None,
            use_varlen_dataset=False,
            varlen_sbhd_validation=False,
        ),
    )
    yield run_config
    Utils.destroy_model_parallel()


def test_fixed_data_seed_allows_different_model_initializations(data_seed_loader_setup):
    cfg = data_seed_loader_setup
    cfg.rng.data_seed = 5678
    weights, worker_seeds, batches = [], [], []
    for model_seed in (1234, 4321):
        cfg.rng.seed = model_seed
        torch.manual_seed(model_seed)
        weights.append(torch.nn.Linear(8, 8).weight.detach().clone())
        model_rng = torch.get_rng_state()
        loader = data_samplers.build_pretraining_data_loader(_Dataset(), consumed_samples=0)
        iterator = iter(loader)
        worker_seeds.append(iterator._base_seed)
        batches.append([batch.tolist() for batch in iterator])
        assert torch.equal(torch.get_rng_state(), model_rng)
    assert not torch.equal(weights[0], weights[1])
    assert worker_seeds[0] == worker_seeds[1]
    assert batches[0] == batches[1]


def test_data_seed_changes_worker_rng_without_changing_weights(data_seed_loader_setup):
    cfg = data_seed_loader_setup
    weights, worker_seeds = [], []
    for data_seed in (5678, 8765):
        cfg.rng.data_seed = data_seed
        torch.manual_seed(cfg.rng.seed)
        loader = data_samplers.build_pretraining_data_loader(_Dataset(), consumed_samples=0)
        worker_seeds.append(iter(loader)._base_seed)
        weights.append(torch.nn.Linear(8, 8).weight.detach().clone())
    assert torch.equal(weights[0], weights[1])
    assert worker_seeds[0] != worker_seeds[1]


def test_omitted_data_seed_preserves_legacy_worker_seed(data_seed_loader_setup):
    cfg = data_seed_loader_setup
    cfg.rng.data_seed = None
    # Preserve the runtime seed, including the offset used for a parallel rank.
    torch.manual_seed(cfg.rng.seed + 100)
    loader = data_samplers.build_pretraining_data_loader(_Dataset(), consumed_samples=0)
    assert loader.generator.initial_seed() == torch.initial_seed()
