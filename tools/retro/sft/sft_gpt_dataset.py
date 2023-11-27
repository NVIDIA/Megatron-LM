# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

"""GPT style dataset."""
from types import SimpleNamespace

from megatron import print_rank_0, get_args
from megatron.core import mpu
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.blended_dataset import BlendedDataset
from megatron.data.dataset_utils import get_datasets_weights_and_num_samples
from tools.retro.sft.dataset_conv import FtDataset as SFTDataset
from tools.retro.sft.dataset_conv import get_processed_dataset


MEGATRON_CORE_DUMMY_CONFIG = SimpleNamespace(
    is_built_on_rank = lambda: mpu.get_tensor_model_parallel_rank() == 0,
    path_to_cache = getattr(get_args(), "data_cache_path")
)


def build_train_valid_test_datasets(data_prefix, seq_length):
    """Build train, valid, and test datasets."""

    assert data_prefix

    args = get_args()

    if len(data_prefix) == 1:
        processed_datasets = get_processed_dataset(data_prefix[0], args.data_folder)

        train_ds = SFTDataset(prefixes[i], processed_datasets["train"], seq_length)
        valid_ds = SFTDataset(prefixes[i], processed_datasets["valid"], seq_length)
        test_ds = SFTDataset(prefixes[i], processed_datasets["test"], seq_length)

        return train_ds, valid_ds, test_ds

    prefixes, weights, _ = get_datasets_weights_and_num_samples(data_prefix, train_valid_test_num_samples=0)
    train_datasets, valid_datasets, test_datasets = [], [], []
    train_size, valid_size, test_size = 0, 0, 0

    for i in range(len(prefixes)):
        processed_datasets = get_processed_dataset(prefixes[i], args.data_folder)

        train_ds = SFTDataset(prefixes[i], processed_datasets["train"], seq_length)
        valid_ds = SFTDataset(prefixes[i], processed_datasets["valid"], seq_length)
        test_ds = SFTDataset(prefixes[i], processed_datasets["test"], seq_length)

        if train_ds:
            train_datasets.append(train_ds)
            train_size += len(train_ds)
        if valid_ds:
            valid_datasets.append(valid_ds)
            valid_size += len(valid_ds)
        if test_ds:
            test_datasets.append(test_ds)
            test_size += len(test_ds)

    # Blend
    blending_train_dataset = None
    if train_datasets:
        blending_train_dataset = BlendedMegatronDatasetBuilder.build_generic_dataset(
            BlendedDataset,
            getattr(MEGATRON_CORE_DUMMY_CONFIG, "is_built_on_rank"),
            train_datasets,
            weights,
            train_size,
            MEGATRON_CORE_DUMMY_CONFIG,
        )
    blending_valid_dataset = None
    if valid_datasets:
        blending_valid_dataset = BlendedMegatronDatasetBuilder.build_generic_dataset(
            BlendedDataset,
            getattr(MEGATRON_CORE_DUMMY_CONFIG, "is_built_on_rank"),
            valid_datasets,
            weights,
            valid_size,
            MEGATRON_CORE_DUMMY_CONFIG,
        )
    blending_test_dataset = None
    if test_datasets:
        blending_test_dataset = BlendedMegatronDatasetBuilder.build_generic_dataset(
            BlendedDataset,
            getattr(MEGATRON_CORE_DUMMY_CONFIG, "is_built_on_rank"),
            test_datasets,
            weights,
            test_size,
            MEGATRON_CORE_DUMMY_CONFIG,
        )

    return (blending_train_dataset, blending_valid_dataset,
            blending_test_dataset)
