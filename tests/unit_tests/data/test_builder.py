# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

##
# Compile megatron.core.datasets.helpers_cpp dependencies before BlendedDataset import
##

import os
import random
import tempfile
from argparse import Namespace
from collections import defaultdict
from typing import Dict, Optional

import numpy
import pytest
import torch

from megatron.core.datasets.blended_dataset import BlendedDataset
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.blended_megatron_dataset_config import BlendedMegatronDatasetConfig
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig
from megatron.core.datasets.indexed_dataset import DType, IndexedDatasetBuilder
from megatron.core.datasets.megatron_dataset import LowLevelDataset, MegatronDataset
from megatron.core.datasets.utils import Split, compile_helpers, get_blend_from_list
from megatron.training.tokenizer import build_tokenizer
from megatron.training.utils import get_blend_and_blend_per_split
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils
from tools.build_sequences_per_dataset import build_sequences_per_dataset

_NUM_DATASETS = 10

_SEQUENCE_LENGTH = 10

_SIZES = {}
for split in Split:
    _SIZES[split] = []
    for i in range(_NUM_DATASETS):
        _SIZES[split].append({Split.train: 1000, Split.valid: 100, Split.test: 10}[split] * (i + 1))

_MARGIN = 0.005


def create_file_prefixes(tokenizer, number_of_files, maximum_number_of_documents, dataset_dir):
    # Create dataset directory
    os.makedirs(dataset_dir, exist_ok=True)

    # Create file prefixes
    file_prefixes = []
    for i in range(number_of_files):
        file_prefix_path = os.path.join(dataset_dir, f"file_{i}")
        builder = IndexedDatasetBuilder(
            file_prefix_path + ".bin", dtype=DType.optimal_dtype(tokenizer.vocab_size)
        )
        number_of_documents = random.randint(10, maximum_number_of_documents)
        for j in range(number_of_documents):
            number_of_tokens = random.randint(50, 100)
            tokenized_doc = [
                str(random.randint(0, tokenizer.vocab_size - 1)) for _ in range(number_of_tokens)
            ]
            builder.add_document(tokenized_doc, [len(tokenized_doc)])
        builder.finalize(file_prefix_path + ".idx")
        file_prefixes.append(file_prefix_path)

    return file_prefixes


def do_setup(odir):
    paths = defaultdict(list)

    for i in range(_NUM_DATASETS):
        path_to_data = os.path.join(odir, str(i))
        os.mkdir(path_to_data)

        for split in _SIZES:
            data = numpy.zeros((_SIZES[split][i], _SEQUENCE_LENGTH))
            path = os.path.join(path_to_data, f"{split.name}.npy")
            numpy.save(path, data)
            paths[split].append(path)

    return paths


def test_builder():
    if torch.distributed.is_available():
        Utils.initialize_distributed()
        if torch.distributed.get_rank() == 0:
            compile_helpers()
        torch.distributed.barrier()
    else:
        compile_helpers()

    # Define the class here to avoid pytest warnings

    class TestDataset(MegatronDataset):
        def __init__(
            self,
            dataset: LowLevelDataset,
            dataset_path: Optional[str],
            indices: numpy.ndarray,
            num_samples: Optional[int],
            index_split: Split,
            config: BlendedMegatronDatasetConfig,
        ) -> None:
            super().__init__(dataset, dataset_path, indices, num_samples, index_split, config)

            if self.num_samples is None:
                self.num_samples = len(self.indices)

            self.sample_index = numpy.random.choice(self.indices, size=self.num_samples)

        @staticmethod
        def numel_low_level_dataset(low_level_dataset: LowLevelDataset) -> int:
            return len(low_level_dataset)

        @staticmethod
        def build_low_level_dataset(
            dataset_path: str, config: BlendedMegatronDatasetConfig
        ) -> LowLevelDataset:
            return numpy.load(dataset_path)

        def __len__(self) -> int:
            return len(self.sample_index)

        def __getitem__(self, idx: int) -> Dict[str, numpy.ndarray]:
            return {"text": self.dataset[self.sample_index[idx]]}

    with tempfile.TemporaryDirectory() as temp_dir:

        paths = do_setup(temp_dir)

        blends = {
            split: get_blend_from_list(
                [
                    weight_or_path
                    for pair in zip(list(range(1, len(paths[split]) + 1, 1)), paths[split])
                    for weight_or_path in pair
                ]
            )
            for split in Split
        }

        blends_unweighted = {split: (blends[split][0], None) for split in blends}

        config = BlendedMegatronDatasetConfig(
            random_seed=1234,
            sequence_length=_SEQUENCE_LENGTH,
            blend_per_split=[blends[Split.train], None, None],
            mid_level_dataset_surplus=0.005,
        )
        try:
            datasets = BlendedMegatronDatasetBuilder(
                TestDataset, [None, None, None], lambda: True, config
            ).build()
            raise RuntimeError
        except AssertionError:
            pass

        config = BlendedMegatronDatasetConfig(
            random_seed=1234,
            sequence_length=_SEQUENCE_LENGTH,
            blend_per_split=[get_blend_from_list([paths[Split.train][0]]), None, None],
            mid_level_dataset_surplus=0.005,
        )
        datasets = BlendedMegatronDatasetBuilder(
            TestDataset, [1000, None, None], lambda: True, config
        ).build()
        assert len(datasets[0]) == 1000 and isinstance(datasets[0], TestDataset)
        assert datasets[1] is None
        assert datasets[2] is None

        config = BlendedMegatronDatasetConfig(
            random_seed=1234,
            sequence_length=_SEQUENCE_LENGTH,
            blend_per_split=[
                blends_unweighted[Split.train],
                blends_unweighted[Split.valid],
                blends_unweighted[Split.test],
            ],
            mid_level_dataset_surplus=0.005,
        )
        datasets = BlendedMegatronDatasetBuilder(
            TestDataset, [1000, 1000, 1000], lambda: True, config
        ).build()
        assert len(datasets[0]) == 1000
        assert len(datasets[1]) == 1000
        assert len(datasets[2]) == sum(_SIZES[Split.test])

        config = BlendedMegatronDatasetConfig(
            random_seed=1234,
            sequence_length=_SEQUENCE_LENGTH,
            blend_per_split=[
                blends_unweighted[Split.train],
                blends_unweighted[Split.valid],
                blends_unweighted[Split.test],
            ],
            mid_level_dataset_surplus=0.005,
        )
        datasets = BlendedMegatronDatasetBuilder(
            TestDataset, [None, None, None], lambda: True, config
        ).build()
        assert len(datasets[0]) == sum(_SIZES[Split.train])
        assert numpy.all(
            numpy.array(datasets[0].weights)
            == numpy.unique(datasets[0].dataset_index, return_counts=True)[1]
        )
        assert len(datasets[1]) == sum(_SIZES[Split.valid])
        assert numpy.all(
            numpy.array(datasets[1].weights)
            == numpy.unique(datasets[1].dataset_index, return_counts=True)[1]
        )
        assert len(datasets[2]) == sum(_SIZES[Split.test])
        assert numpy.all(
            numpy.array(datasets[2].weights)
            == numpy.unique(datasets[2].dataset_index, return_counts=True)[1]
        )

        config = BlendedMegatronDatasetConfig(
            random_seed=1234,
            sequence_length=_SEQUENCE_LENGTH,
            blend_per_split=[blends_unweighted[Split.train], None, None],
            mid_level_dataset_surplus=0.005,
        )
        datasets = BlendedMegatronDatasetBuilder(
            TestDataset, [1000, None, None], lambda: True, config
        ).build()
        assert len(datasets[0]) == 1000
        for i in range(_NUM_DATASETS):
            assert len(datasets[0].datasets[i]) == _SIZES[Split.train][i]
        assert datasets[1] is None
        assert datasets[2] is None

        # This build used to fail when building datasets without a sample buffer
        config = BlendedMegatronDatasetConfig(
            random_seed=1234,
            sequence_length=_SEQUENCE_LENGTH,
            blend_per_split=[blends[Split.train], None, None],
            mid_level_dataset_surplus=0.005,
        )
        datasets = BlendedMegatronDatasetBuilder(
            TestDataset, [1000, None, None], lambda: True, config
        ).build()

        config = BlendedMegatronDatasetConfig(
            random_seed=1234,
            sequence_length=_SEQUENCE_LENGTH,
            blend=blends_unweighted[Split.train],
            split="100,0,0",
            mid_level_dataset_surplus=0.005,
        )
        datasets = BlendedMegatronDatasetBuilder(
            TestDataset, [None, None, None], lambda: True, config
        ).build()
        assert len(datasets[0]) == sum(_SIZES[Split.train])
        assert numpy.all(
            numpy.array(datasets[0].weights)
            == numpy.unique(datasets[0].dataset_index, return_counts=True)[1]
        )
        assert datasets[1] is None
        assert datasets[2] is None

        if torch.distributed.is_initialized():
            config = BlendedMegatronDatasetConfig(
                random_seed=1234,
                sequence_length=_SEQUENCE_LENGTH,
                blend=blends_unweighted[Split.train],
                split="100,0,0",
                mid_level_dataset_surplus=0.005,
            )
            datasets = BlendedMegatronDatasetBuilder(
                TestDataset,
                [None, None, None],
                lambda: torch.distributed.get_rank() % 2 == 0,
                config,
            ).build()
            if torch.distributed.get_rank() % 2 == 0:
                assert len(datasets[0]) == sum(_SIZES[Split.train])
                assert numpy.all(
                    numpy.array(datasets[0].weights)
                    == numpy.unique(datasets[0].dataset_index, return_counts=True)[1]
                )
            else:
                assert datasets[0] is None
            assert datasets[1] is None
            assert datasets[2] is None

        config = BlendedMegatronDatasetConfig(
            random_seed=1234,
            sequence_length=_SEQUENCE_LENGTH,
            blend=blends_unweighted[Split.train],
            split="50,50,0",
            mid_level_dataset_surplus=0.005,
        )
        datasets = BlendedMegatronDatasetBuilder(
            TestDataset, [1000, 0, None], lambda: True, config
        ).build()
        assert len(datasets[0]) == 1000
        assert sum(map(len, datasets[0].datasets)) == sum(_SIZES[Split.train]) / 2
        assert sum(map(len, datasets[1].datasets)) == sum(_SIZES[Split.train]) / 2
        assert datasets[1] is not None and len(datasets[1]) == 0
        assert datasets[2] is None

        config = BlendedMegatronDatasetConfig(
            random_seed=1234,
            sequence_length=_SEQUENCE_LENGTH,
            blend=blends_unweighted[Split.train],
            split="50,50,0",
            mid_level_dataset_surplus=0.005,
        )
        datasets = BlendedMegatronDatasetBuilder(
            TestDataset,
            [int(sum(_SIZES[Split.train]) / 4), int(sum(_SIZES[Split.train])), None],
            lambda: True,
            config,
        ).build()
        assert len(datasets[0]) == sum(_SIZES[Split.train]) / 4
        assert len(datasets[1]) == sum(_SIZES[Split.train]) / 2
        assert datasets[2] is None

        # This build used to fail when building datasets without a sample buffer
        config = BlendedMegatronDatasetConfig(
            random_seed=1234,
            sequence_length=_SEQUENCE_LENGTH,
            blend=blends[Split.train],
            split="990,9,1",
            mid_level_dataset_surplus=0.005,
        )
        datasets = BlendedMegatronDatasetBuilder(
            TestDataset, [100000, 1000, 1], lambda: True, config
        ).build()


@pytest.mark.parametrize("use_split", [True, False])
@pytest.mark.parametrize("add_weights", [True, False])
@pytest.mark.parametrize("fast_cache_load", [True, False])
@pytest.mark.parametrize("sequences_per_dataset", [True, False])
@pytest.mark.parametrize("defer_npy_index_mmap", [True, False])
@pytest.mark.parametrize("vocab_size", [131072, 20000])
@pytest.mark.parametrize("mid_level_dataset_surplus", [0.005, 0.01, 0])
def test_fast_builder(
    use_split,
    add_weights,
    fast_cache_load,
    sequences_per_dataset,
    defer_npy_index_mmap,
    vocab_size,
    mid_level_dataset_surplus,
    tmp_path_dist_ckpt,
    sequence_length: int = 5,
    number_of_files: int = 10,
    number_of_documents: int = 10,
):
    if use_split and fast_cache_load:
        pytest.skip("Skipping test case when both use_split and fast_cache_load are True")

    if torch.distributed.is_available():
        Utils.initialize_distributed()
        if torch.distributed.get_rank() == 0:
            compile_helpers()
        torch.distributed.barrier()
    else:
        compile_helpers()

    tokenizer = build_tokenizer(
        Namespace(
            vocab_size=vocab_size,
            tokenizer_type="NullTokenizer",
            rank=0,
            make_vocab_size_divisible_by=128,
            tensor_model_parallel_size=1,
        )
    )

    with TempNamedDir(tmp_path_dist_ckpt / "test_fast_builder", sync=True) as temp_dir:
        # Created file_prefixes (tokenizer, Number of files, number of documents, path) --> returns file prefixes (list of strings)
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            file_prefixes = create_file_prefixes(
                tokenizer, number_of_files, number_of_documents, os.path.join(temp_dir, "dataset")
            )
        else:
            file_prefixes = []
            for i in range(number_of_files):
                file_prefix_path = os.path.join(temp_dir, "dataset", f"file_{i}")
                file_prefixes.append(file_prefix_path)

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        random.seed(1234)  # NOTE(asolergi-nv): re-sync random state across all ranks

        data_cache_path = os.path.join(temp_dir, "cache")

        args = Namespace(
            seed=1234,
            seq_length=sequence_length,
            data_cache_path=data_cache_path,
            split=None,
            data_path=None,
            train_data_path=None,
            valid_data_path=None,
            test_data_path=None,
            per_split_data_args_path=None,
            data_args_path=None,
        )

        # set up data mixture
        if use_split:
            args.data_path = file_prefixes
            args.split = "70,20,10"
        else:
            train_file_prefixes = file_prefixes[0:6]
            valid_file_prefixes = file_prefixes[6:9]
            test_file_prefixes = file_prefixes[9:10]

            if add_weights:
                # Save original lists before modifying
                train_file_prefixes_original = train_file_prefixes[:]
                valid_file_prefixes_original = valid_file_prefixes[:]
                test_file_prefixes_original = test_file_prefixes[:]

                # For train_file_prefixes, alternately append a random int (10-100) and the file prefix.
                train_file_prefixes = []
                for fp in train_file_prefixes_original:
                    train_file_prefixes.extend([random.randint(10, 100), fp])
                # For valid/test, also add random weights (10-100).
                valid_file_prefixes = []
                for fp in valid_file_prefixes_original:
                    valid_file_prefixes.extend([random.randint(10, 100), fp])
                test_file_prefixes = []
                for fp in test_file_prefixes_original:
                    test_file_prefixes.extend([random.randint(10, 100), fp])

            args.train_data_path = train_file_prefixes
            args.valid_data_path = valid_file_prefixes
            args.test_data_path = test_file_prefixes

        if sequences_per_dataset:
            args.path_to_sequences_per_dataset_json = os.path.join(
                temp_dir, "sequences_per_dataset.json"
            )
            sequences_per_dataset = build_sequences_per_dataset(args)

        blend, blend_per_split = get_blend_and_blend_per_split(args)

        data_args = {
            "random_seed": args.seed,
            "sequence_length": args.seq_length,
            "blend": blend,
            "blend_per_split": blend_per_split,
            "split": args.split,
            "path_to_cache": args.data_cache_path,
            "tokenizer": tokenizer,
            "reset_position_ids": False,
            "reset_attention_mask": False,
            "eod_mask_loss": False,
            "create_attention_mask": False,
            "mid_level_dataset_surplus": mid_level_dataset_surplus,
        }
        config = GPTDatasetConfig(**data_args)

        train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
            GPTDataset, [100, 10, 10], lambda: True, config
        ).build()

        fast_config = GPTDatasetConfig(
            **data_args,
            fast_cache_load=fast_cache_load,
            defer_npy_index_mmap=defer_npy_index_mmap,
            sequences_per_dataset=sequences_per_dataset,
        )

        train_ds_fast, valid_ds_fast, test_ds_fast = BlendedMegatronDatasetBuilder(
            GPTDataset, [100, 10, 10], lambda: True, fast_config
        ).build()

        for ds_slow, ds_fast, split_name in zip(
            [train_ds, valid_ds, test_ds],
            [train_ds_fast, valid_ds_fast, test_ds_fast],
            ["train", "valid", "test"],
        ):
            if not ds_slow:
                continue
            assert len(ds_slow) == len(
                ds_fast
            ), f"ds_slow: {len(ds_slow)}, ds_fast: {len(ds_fast)}, split_name: {split_name}"
            if isinstance(ds_slow, GPTDataset):
                assert torch.all(ds_slow[0]["tokens"] == ds_fast[0]["tokens"])
                assert torch.all(ds_slow[-1]["tokens"] == ds_fast[-1]["tokens"])
                numpy.testing.assert_array_equal(ds_slow.document_index, ds_fast.document_index)
                numpy.testing.assert_array_equal(ds_slow.sample_index, ds_fast.sample_index)
                numpy.testing.assert_array_equal(ds_slow.shuffle_index, ds_fast.shuffle_index)
                numpy.testing.assert_array_equal(
                    ds_slow.dataset.index.sequence_lengths, ds_fast.dataset.index.sequence_lengths
                )
                numpy.testing.assert_array_equal(
                    ds_slow.dataset.index.document_indices, ds_fast.dataset.index.document_indices
                )
                numpy.testing.assert_array_equal(
                    ds_slow.dataset.index.sequence_pointers, ds_fast.dataset.index.sequence_pointers
                )
            elif isinstance(ds_slow, BlendedDataset):
                assert torch.all(ds_slow[0]["tokens"] == ds_fast[0]["tokens"])
                assert torch.all(ds_slow[-1]["tokens"] == ds_fast[-1]["tokens"])
                numpy.testing.assert_array_equal(ds_slow.dataset_index, ds_fast.dataset_index)
                numpy.testing.assert_array_equal(
                    ds_slow.dataset_sample_index, ds_fast.dataset_sample_index
                )
                for ds_slow_i, ds_fast_i in zip(ds_slow.datasets, ds_fast.datasets):
                    assert torch.all(ds_slow_i[0]["tokens"] == ds_fast_i[0]["tokens"])
                    assert torch.all(ds_slow_i[-1]["tokens"] == ds_fast_i[-1]["tokens"])
                    numpy.testing.assert_array_equal(
                        ds_slow_i.document_index, ds_fast_i.document_index
                    )
                    numpy.testing.assert_array_equal(ds_slow_i.sample_index, ds_fast_i.sample_index)
                    numpy.testing.assert_array_equal(
                        ds_slow_i.shuffle_index, ds_fast_i.shuffle_index
                    )
                    numpy.testing.assert_array_equal(
                        ds_slow_i.dataset.index.sequence_lengths,
                        ds_fast_i.dataset.index.sequence_lengths,
                    )
                    numpy.testing.assert_array_equal(
                        ds_slow_i.dataset.index.document_indices,
                        ds_fast_i.dataset.index.document_indices,
                    )
                    numpy.testing.assert_array_equal(
                        ds_slow_i.dataset.index.sequence_pointers,
                        ds_fast_i.dataset.index.sequence_pointers,
                    )
        if torch.distributed.is_initialized():
            torch.distributed.barrier()


if __name__ == "__main__":
    test_builder()
