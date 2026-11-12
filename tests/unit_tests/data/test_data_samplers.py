# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Restoring a checkpoint then building the data iterators must leave the CPU RNG where it was.

Mirrors pretrain(): _set_random_seed, then load_checkpoint restores the saved state, then
build_train_valid_test_data_iterators. Nothing between the restore and the first step draws from
the default CPU generator -- except, on the pre-fix code, iterator creation itself.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import megatron.training.datasets.data_samplers as data_samplers
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.core.datasets.utils import Split
from megatron.core.transformer import TransformerConfig
from megatron.training.config import DataConfig, DataloaderConfig
from megatron.training.config.dataset_config import is_packed_sequence
from megatron.training.datasets.sft_dataset import MockSFTDataset
from megatron.training.datasets.varlen_dataset import MockVarlenDataset, VarlenDataset
from megatron.training.models import GPTModelConfig
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

    def test_building_data_iterators_does_not_move_cpu_rng(self, monkeypatch, run_config):
        Utils.initialize_model_parallel(1, 1)
        run_config.model = GPTModelConfig(
            transformer=TransformerConfig(num_layers=2, hidden_size=32, num_attention_heads=4),
            seq_length=16,
        )

        run_config.train.micro_batch_size = 1
        run_config.train.global_batch_size = Utils.world_size
        run_config.data.dataloader.dataloader_type = 'single'
        run_config.data.dataloader.num_workers = 0
        monkeypatch.setattr(data_samplers, 'get_args', lambda: _legacy_args(), raising=False)
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

    @pytest.mark.parametrize(
        "dynamic_cp,scheduler",
        [(False, None), (False, "dp_balanced"), (True, "default_dynamic_cp")],
    )
    def test_sampler_and_collation_use_model_config(
        self, monkeypatch, run_config, dynamic_cp, scheduler
    ):
        Utils.initialize_model_parallel(1, 1)
        run_config.train.micro_batch_size = 1
        run_config.train.global_batch_size = 2 * Utils.world_size
        run_config.model = GPTModelConfig(
            transformer=TransformerConfig(num_layers=2, hidden_size=32, num_attention_heads=4),
            seq_length=16,
        )
        run_config.model.transformer.dynamic_context_parallel = dynamic_cp
        run_config.model.transformer.sequence_packing_scheduler = scheduler
        # The loader settings are the data section's (its num_workers defaults to 2).
        run_config.data.dataloader.dataloader_type = "single"
        run_config.data.dataloader.num_workers = 0
        monkeypatch.setattr(
            data_samplers,
            "get_args",
            lambda: SimpleNamespace(
                dataloader_type="single",
                micro_batch_size=1,
                global_batch_size=2 * Utils.world_size,
                num_workers=0,
                dynamic_context_parallel=not dynamic_cp,
                sequence_packing_scheduler=None if scheduler else "dp_balanced",
                use_varlen_dataset=False,
                varlen_sbhd_validation=False,
            ),
            raising=False,
        )

        loader = data_samplers.build_pretraining_data_loader(_Dataset(), consumed_samples=0)

        assert isinstance(loader.batch_sampler, data_samplers.MegatronPretrainingSampler)
        batch = next(iter(loader))
        if dynamic_cp or scheduler is not None:
            assert isinstance(batch, list)
            assert len(batch) == 1
        else:
            assert batch.shape == (1, 1)


BOOLS = dict(reset_position_ids=False, reset_attention_mask=False, eod_mask_loss=False)
BLEND = (["prefix-a"], None)


def _legacy_args(**overrides):
    """The namespace the loader no longer reads (it imports no get_args): poisoned so a read,
    should one come back, would show (each value selects another sampler or collate)."""
    values = dict(
        dataloader_type="cyclic",
        num_workers=7,
        data_sharding=False,
        use_varlen_dataset=True,
        varlen_sbhd_validation=False,
        dynamic_context_parallel=True,
        sequence_packing_scheduler="dp_balanced",
        context_parallel_size=1,
        pipeline_model_parallel_size=1,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class _DictDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, idx):
        return {"tokens": torch.tensor([idx, idx]), "length": idx}


def _section(run_config, dataset=None, dataset_type=None, **loader):
    settings = dict(dataloader_type="single", num_workers=0)
    settings.update(loader)
    run_config.data = DataConfig(
        dataset=dataset, dataset_type=dataset_type, dataloader=DataloaderConfig(**settings)
    )
    return run_config.data


def _batch(dataset):
    return next(iter(data_samplers.build_pretraining_data_loader(dataset, 0)))


class TestLoaderSettingsFromTheDataSection:
    """build_pretraining_data_loader reads cfg.data.dataloader, the dataset family and the model
    section; the legacy namespace is poisoned and must not be read."""

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_loader_settings_come_from_the_section_not_from_args(self, monkeypatch, run_config):
        Utils.initialize_model_parallel(1, 1)
        run_config.train.micro_batch_size = 1
        monkeypatch.setattr(data_samplers, "get_args", lambda: _legacy_args(), raising=False)
        _section(run_config)
        loader = data_samplers.build_pretraining_data_loader(_Dataset(), 0)
        assert isinstance(loader.batch_sampler, data_samplers.MegatronPretrainingSampler)
        assert loader.num_workers == 0 and loader.pin_memory is True
        assert loader.persistent_workers is False
        assert isinstance(next(iter(loader)), torch.Tensor)
        _section(run_config, num_workers=2)  # today's built-ins: pinned, persistent with workers
        loader = data_samplers.build_pretraining_data_loader(_Dataset(), 0)
        assert loader.num_workers == 2 and loader.pin_memory is True
        assert loader.persistent_workers is True
        _section(run_config, num_workers=2, pin_memory=False, persistent_workers=False)
        loader = data_samplers.build_pretraining_data_loader(_Dataset(), 0)
        assert loader.pin_memory is False and loader.persistent_workers is False
        _section(run_config, dataloader_type="cyclic", data_sharding=True)
        loader = data_samplers.build_pretraining_data_loader(_Dataset(), 0)
        assert isinstance(loader.batch_sampler, data_samplers.MegatronPretrainingRandomSampler)
        assert loader.batch_sampler.data_sharding is True
        _section(run_config, dataloader_type="external")
        dataset = _Dataset()
        assert data_samplers.build_pretraining_data_loader(dataset, 0) is dataset

    def test_collate_follows_the_dataset_family_and_the_topology(self, monkeypatch, run_config):
        Utils.initialize_model_parallel(1, 1)
        run_config.train.micro_batch_size = 1
        monkeypatch.setattr(data_samplers, "get_args", lambda: _legacy_args(), raising=False)
        plain = GPTDatasetConfig(sequence_length=16, random_seed=1, **BOOLS)
        # Unpacked varlen samples: the identity collate, a list of sample dicts.
        _section(run_config, dataset=plain, dataset_type=MockVarlenDataset)
        batch = _batch(_DictDataset())
        assert isinstance(batch, list) and isinstance(batch[0], dict)
        # Varlen with SBHD validation emits fixed-width samples: the default collate.
        sbhd = GPTDatasetConfig(
            sequence_length=16,
            random_seed=1,
            blend=BLEND,
            split="9,1,0",
            varlen_sbhd_validation=True,
            **BOOLS,
        )
        _section(run_config, dataset=sbhd, dataset_type=VarlenDataset)
        batch = _batch(_DictDataset())
        assert isinstance(batch, dict) and tuple(batch["tokens"].shape) == (1, 2)
        # A loader-only section: the default collate unless the model section selects dynamic CP
        # or a packing scheduler (the fixture's selects neither; the namespace says both).
        _section(run_config)
        assert isinstance(_batch(_DictDataset()), dict)
        run_config.model.transformer.dynamic_context_parallel = True
        assert isinstance(_batch(_DictDataset()), list)
        run_config.model.transformer.dynamic_context_parallel = False
        run_config.model.transformer.sequence_packing_scheduler = "dp_balanced"
        assert isinstance(_batch(_DictDataset()), list)

    def test_sft_batches_keep_the_default_collate(self, monkeypatch, run_config):
        """SFT samples are packed (they carry cu_seqlens) but fixed-width: the loader yields a
        dict with a batch dimension, as before. The packed-metadata predicate is true for SFT and
        must not drive the collate."""
        Utils.initialize_model_parallel(1, 1)
        run_config.train.micro_batch_size = 1
        monkeypatch.setattr(data_samplers, "get_args", lambda: _legacy_args(), raising=False)
        tokenizer = SimpleNamespace(
            eod=15, eos=15, pad=-1, vocab_size=16, unique_identifiers={"class": "test"}
        )
        # context_parallel_size is a container-written field (None until finalize); the SFT
        # padding divisor reads it, so give it the value the container would write.
        config = GPTDatasetConfig(
            sequence_length=16, random_seed=1, tokenizer=tokenizer, context_parallel_size=1, **BOOLS
        )
        # The sampler yields one micro-batch per data-parallel rank from world_size samples (the
        # CI runs the data tests on eight ranks), so the mock dataset carries 2 * world_size.
        n = 2 * Utils.world_size
        dataset = MockSFTDataset(
            [np.array([1, 2, 3]) if i % 2 == 0 else np.array([4, 5, 6]) for i in range(n)],
            None,
            np.arange(n),
            n,
            Split.train,
            config,
        )
        section = _section(run_config, dataset=config, dataset_type=MockSFTDataset)
        assert is_packed_sequence(MockSFTDataset, config) and not section.yields_unpacked_samples()
        batch = _batch(dataset)
        assert isinstance(batch, dict) and tuple(batch["tokens"].shape) == (1, 16)
        assert "cu_seqlens" in batch
