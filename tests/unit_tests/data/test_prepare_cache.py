# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os
import random
from argparse import ArgumentParser, Namespace

import pytest
import torch

from megatron.core.datasets.blended_dataset import BlendedDataset
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDataset
from megatron.core.datasets.indexed_dataset import DType, IndexedDatasetBuilder
from megatron.core.datasets.utils import compile_helpers
from megatron.core.tokenizers.utils.build_tokenizer import build_tokenizer
from megatron.training.arguments import add_megatron_arguments
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils
from tools.prepare_cache import (
    _normalize_prepare_cache_args,
    build_dataset_caches,
    core_gpt_dataset_config_from_args,
)


def _build_null_tokenizer(vocab_size: int = 2048):
    return build_tokenizer(
        Namespace(
            vocab_size=vocab_size,
            tokenizer_type="NullTokenizer",
            padded_vocab_size=None,
            rank=0,
            make_vocab_size_divisible_by=128,
            tensor_model_parallel_size=1,
            pad_vocab_size=True,
        )
    )


def _initialize_test_environment() -> None:
    if torch.distributed.is_available():
        Utils.initialize_distributed()
        if torch.distributed.get_rank() == 0:
            compile_helpers()
        torch.distributed.barrier()
    else:
        compile_helpers()


def _create_file_prefixes(tokenizer, dataset_dir, number_of_files: int = 4) -> list[str]:
    os.makedirs(dataset_dir, exist_ok=True)

    file_prefixes = []
    for i in range(number_of_files):
        file_prefix = os.path.join(dataset_dir, f"file_{i}")
        builder = IndexedDatasetBuilder(
            file_prefix + ".bin", dtype=DType.optimal_dtype(tokenizer.vocab_size)
        )

        for j in range(32):
            tokens = [int((i * 97 + j * 13 + k) % tokenizer.vocab_size) for k in range(64)]
            builder.add_document(tokens, [len(tokens)])

        builder.finalize(file_prefix + ".idx")
        file_prefixes.append(file_prefix)

    return file_prefixes


def _create_shared_file_prefixes(tokenizer, dataset_dir, number_of_files: int = 4) -> list[str]:
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        file_prefixes = _create_file_prefixes(tokenizer, dataset_dir, number_of_files)
    else:
        file_prefixes = [os.path.join(dataset_dir, f"file_{i}") for i in range(number_of_files)]

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    random.seed(1234)  # NOTE(asolergi-nv): re-sync random state across all ranks

    return file_prefixes


def _build_prepare_cache_args(file_prefixes, data_cache_path, **overrides):
    args = dict(
        seed=1234,
        seq_length=16,
        split="70,20,10",
        data_path=file_prefixes,
        train_data_path=None,
        valid_data_path=None,
        test_data_path=None,
        per_split_data_args_path=None,
        data_args_path=None,
        per_dataset_sequences_path=None,
        data_cache_path=str(data_cache_path),
        mmap_bin_files=True,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
        create_attention_mask_in_dataloader=False,
        object_storage_cache_path=None,
        mid_level_dataset_surplus=0.005,
        allow_ambiguous_pad_tokens=False,
        dataloader_fast_cache_load=True,
        dataloader_defer_npy_index_mmap=True,
        context_parallel_size=1,
        data_parallel_size=4,
        tensor_model_parallel_size=1,
        sequence_parallel=False,
        dynamic_context_parallel=False,
        multiple_validation_sets=False,
        full_validation=False,
        num_dataset_builder_threads=1,
        tokenizer_type="NullTokenizer",
        vocab_size=2048,
        padded_vocab_size=None,
        pad_vocab_size=True,
        make_vocab_size_divisible_by=128,
        rank=0,
        world_size=4,
        train_samples=None,
        train_iters=4,
        skip_train=False,
        eval_iters=2,
        eval_interval=2,
        start_eval_at_iter=None,
        global_batch_size=8,
        micro_batch_size=2,
        gtp_weight_remat_size=1,
        phase_transition_iterations=None,
        iteration=0,
        mock_data=False,
        sft=False,
        fim_data=False,
        step_batch_size_schedule=None,
        main_grads_dtype=torch.float32,
        main_params_dtype=torch.float32,
        exp_avg_dtype=torch.float32,
        exp_avg_sq_dtype=torch.float32,
    )
    args.update(overrides)
    parser = ArgumentParser()
    add_megatron_arguments(parser)
    parsed_args = parser.parse_args([])
    vars(parsed_args).update(args)
    return parsed_args


@pytest.mark.parametrize("data_seed", [None, 5678])
def test_prepare_cache_builds_blended_dataset_cache(tmp_path_dist_ckpt, data_seed):
    _initialize_test_environment()

    tokenizer = _build_null_tokenizer()

    with TempNamedDir(
        tmp_path_dist_ckpt / "test_prepare_cache_builds_blended_dataset_cache", sync=True
    ) as temp_dir:
        file_prefixes = _create_shared_file_prefixes(tokenizer, os.path.join(temp_dir, "dataset"))
        args = _build_prepare_cache_args(file_prefixes, temp_dir / "cache", data_seed=data_seed)

        summary = build_dataset_caches(args)

        assert args.dataloader_fast_cache_load is False
        assert args.dataloader_defer_npy_index_mmap is False
        assert summary["train_valid_test_num_samples"] == (32, 48, 16)
        assert summary["train_dataset_length"] == 32
        assert summary["valid_dataset_length"] == 48
        assert summary["test_dataset_length"] == 16
        assert list((temp_dir / "cache").glob("*document_index.npy"))
        assert list((temp_dir / "cache").glob("*dataset_index.npy"))


@pytest.mark.parametrize("seed", [1234, 4321])
@pytest.mark.parametrize("data_seed", [None, 5678])
def test_prepare_cache_seed_matches_training(seed, data_seed, tmp_path, run_config):
    from pretrain_gpt import core_gpt_dataset_config_from_args as training_dataset_config

    args = _build_prepare_cache_args(
        ["unused"],
        tmp_path,
        seed=seed,
        data_seed=data_seed,
        dataloader_fast_cache_load=False,
        dataloader_defer_npy_index_mmap=False,
    )
    run_config.rng.seed = seed
    run_config.rng.data_seed = data_seed
    cache_config = core_gpt_dataset_config_from_args(args)
    training_config = training_dataset_config(args)
    assert cache_config.random_seed == training_config.random_seed
    assert cache_config.random_seed == (seed if data_seed is None else data_seed)


def test_model_and_data_seeds_are_independent(tmp_path_dist_ckpt, run_config):
    from pretrain_gpt import core_gpt_dataset_config_from_args as training_dataset_config

    _initialize_test_environment()
    tokenizer = _build_null_tokenizer()
    with TempNamedDir(tmp_path_dist_ckpt / "independent_seeds", sync=True) as temp_dir:
        prefixes = _create_shared_file_prefixes(tokenizer, os.path.join(temp_dir, "dataset"))
        weights, samples, cache_hashes = [], [], []
        for model_seed, data_seed in ((1234, 5678), (4321, 5678), (1234, 8765)):
            run_config.rng.seed = model_seed
            run_config.rng.data_seed = data_seed
            torch.manual_seed(model_seed)
            args = _build_prepare_cache_args(
                prefixes,
                temp_dir / "cache",
                seed=model_seed,
                data_seed=data_seed,
                dataloader_fast_cache_load=False,
                dataloader_defer_npy_index_mmap=False,
            )
            config = training_dataset_config(args)
            train, _, _ = BlendedMegatronDatasetBuilder(
                GPTDataset, [32, 8, 8], lambda: True, config
            ).build()
            samples.append(torch.stack([train[i]["tokens"] for i in range(8)]))
            cache_hashes.append(train.unique_description_hash)
            weights.append(torch.nn.Linear(8, 8).weight.detach().clone())

        assert not torch.equal(weights[0], weights[1])
        assert torch.equal(samples[0], samples[1])
        assert cache_hashes[0] == cache_hashes[1]
        assert torch.equal(weights[0], weights[2])
        assert not torch.equal(samples[0], samples[2])
        assert cache_hashes[0] != cache_hashes[2]


def test_prepare_cache_world_size_override():
    args = Namespace(rank=11, world_size=1, prepare_cache_world_size=8)

    _normalize_prepare_cache_args(args)

    assert args.rank == 0
    assert args.world_size == 8


def test_prepare_cache_builds_and_hits_per_split_dataset_cache(tmp_path_dist_ckpt):
    _initialize_test_environment()

    tokenizer = _build_null_tokenizer()

    with TempNamedDir(
        tmp_path_dist_ckpt / "test_prepare_cache_builds_and_hits_per_split_dataset_cache", sync=True
    ) as temp_dir:
        file_prefixes = _create_shared_file_prefixes(tokenizer, os.path.join(temp_dir, "dataset"))
        args = _build_prepare_cache_args(
            None,
            temp_dir / "cache",
            split=None,
            data_path=None,
            train_data_path=[50, file_prefixes[0], 50, file_prefixes[1]],
            valid_data_path=[file_prefixes[2]],
            test_data_path=[file_prefixes[3]],
        )

        summary = build_dataset_caches(args)

        assert summary["train_valid_test_num_samples"] == (32, 48, 16)
        assert list((temp_dir / "cache").glob("*description.txt"))

        slow_args = _build_prepare_cache_args(
            None,
            temp_dir / "cache",
            split=None,
            data_path=None,
            train_data_path=[50, file_prefixes[0], 50, file_prefixes[1]],
            valid_data_path=[file_prefixes[2]],
            test_data_path=[file_prefixes[3]],
            dataloader_fast_cache_load=False,
            dataloader_defer_npy_index_mmap=False,
        )
        slow_config = core_gpt_dataset_config_from_args(slow_args)
        train_slow, valid_slow, test_slow = BlendedMegatronDatasetBuilder(
            GPTDataset, list(summary["train_valid_test_num_samples"]), lambda: True, slow_config
        ).build()

        fast_args = _build_prepare_cache_args(
            None,
            temp_dir / "cache",
            split=None,
            data_path=None,
            train_data_path=[50, file_prefixes[0], 50, file_prefixes[1]],
            valid_data_path=[file_prefixes[2]],
            test_data_path=[file_prefixes[3]],
            dataloader_fast_cache_load=True,
            dataloader_defer_npy_index_mmap=True,
        )
        fast_config = core_gpt_dataset_config_from_args(fast_args)
        train_fast, valid_fast, test_fast = BlendedMegatronDatasetBuilder(
            GPTDataset, list(summary["train_valid_test_num_samples"]), lambda: True, fast_config
        ).build()

        assert isinstance(train_fast, BlendedDataset)
        assert train_fast.dataset_index is None
        assert train_fast.dataset_sample_index is None
        assert isinstance(valid_fast, GPTDataset)
        assert valid_fast.document_index is None
        assert valid_fast.sample_index is None
        assert valid_fast.shuffle_index is None
        assert isinstance(test_fast, GPTDataset)
        assert test_fast.document_index is None
        assert test_fast.sample_index is None
        assert test_fast.shuffle_index is None

        assert summary["train_dataset_length"] == len(train_slow) == len(train_fast) == 32
        assert summary["valid_dataset_length"] == len(valid_slow) == len(valid_fast)
        assert summary["test_dataset_length"] == len(test_slow) == len(test_fast)
        assert summary["valid_dataset_length"] >= summary["train_valid_test_num_samples"][1]
        assert summary["test_dataset_length"] >= summary["train_valid_test_num_samples"][2]
        assert torch.all(train_slow[0]["tokens"] == train_fast[0]["tokens"])
        assert torch.all(valid_slow[0]["tokens"] == valid_fast[0]["tokens"])
        assert torch.all(test_slow[0]["tokens"] == test_fast[0]["tokens"])

        assert train_fast.dataset_index is not None
        assert train_fast.dataset_sample_index is not None
        assert valid_fast.document_index is not None
        assert valid_fast.sample_index is not None
        assert valid_fast.shuffle_index is not None
        assert test_fast.document_index is not None
        assert test_fast.sample_index is not None
        assert test_fast.shuffle_index is not None


@pytest.mark.parametrize(
    ("flag_name", "flag_value", "message"),
    [
        ("mock_data", True, "--mock-data"),
        ("sft", True, "--sft"),
        ("fim_data", True, "--fim-data"),
        ("step_batch_size_schedule", [(0, 8)], "--step-batch-size-schedule"),
    ],
)
def test_prepare_cache_rejects_unsupported_modes(tmp_path, flag_name, flag_value, message):
    args = _build_prepare_cache_args([], tmp_path / "cache", **{flag_name: flag_value})

    with pytest.raises(ValueError, match=message):
        build_dataset_caches(args)


def test_prepare_cache_builds_with_train_samples(tmp_path_dist_ckpt):
    _initialize_test_environment()

    tokenizer = _build_null_tokenizer()

    with TempNamedDir(
        tmp_path_dist_ckpt / "test_prepare_cache_builds_with_train_samples", sync=True
    ) as temp_dir:
        file_prefixes = _create_shared_file_prefixes(tokenizer, os.path.join(temp_dir, "dataset"))
        args = _build_prepare_cache_args(
            file_prefixes, temp_dir / "cache", train_iters=None, train_samples=32
        )

        summary = build_dataset_caches(args)

        assert summary["train_valid_test_num_samples"][0] == 32
        assert summary["train_valid_test_num_samples"][1:] == (48, 16)
