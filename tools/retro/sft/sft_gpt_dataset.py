# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

"""GPT style dataset."""

from megatron import print_rank_0, get_args
from megatron.data.blendable_dataset import BlendableDataset
from megatron.data.dataset_utils import get_datasets_weights_and_num_samples
from tools.retro.sft.dataset_conv import FtDataset as SFTDataset
from tools.retro.sft.dataset_conv import get_processed_dataset


def build_train_valid_test_datasets(data_prefix, splits_string,
                                    train_valid_test_num_samples,
                                    seq_length, seed, skip_warmup,
                                    train_data_prefix=None,
                                    valid_data_prefix=None,
                                    test_data_prefix=None,
                                    return_doc_ids=False):
    """Build train, valid, and test datasets."""

    if data_prefix:
        print_rank_0("Single data path provided for train, valid & test")

        # Single dataset.
        if len(data_prefix) == 1:
            return _build_train_valid_test_datasets(data_prefix[0],
                                                    splits_string,
                                                    train_valid_test_num_samples,
                                                    seq_length, seed, skip_warmup)

        # Blending dataset.
        # Parse the values.
        output = get_datasets_weights_and_num_samples(data_prefix,
                                                      train_valid_test_num_samples)
        prefixes, weights, datasets_train_valid_test_num_samples = output

        # Build individual datasets.
        train_datasets = []
        valid_datasets = []
        test_datasets = []

        train_size = 0
        valid_size = 0
        test_size = 0

        for i in range(len(prefixes)):
            train_ds, valid_ds, test_ds = _build_train_valid_test_datasets(
                prefixes[i], splits_string,
                datasets_train_valid_test_num_samples[i],
                seq_length, seed, skip_warmup,
                return_doc_ids)
            if train_ds:
                train_datasets.append(train_ds)
                train_size += len(train_ds)
            if valid_ds:
                valid_datasets.append(valid_ds)
                valid_size += len(valid_ds)
            if test_ds:
                test_datasets.append(test_ds)
                test_size += len(test_ds)

        # Blend.
        blending_train_dataset = None
        if train_datasets:
            blending_train_dataset = BlendableDataset(train_datasets, weights, train_size)
        blending_valid_dataset = None
        if valid_datasets:
            blending_valid_dataset = BlendableDataset(valid_datasets, weights, valid_size)
        blending_test_dataset = None
        if test_datasets:
            blending_test_dataset = BlendableDataset(test_datasets, weights, test_size)

        return (blending_train_dataset, blending_valid_dataset,
                blending_test_dataset)

    else:
        print_rank_0("Separate data paths provided for train, valid & test. Split string will be ignored.")

        train_dataset, valid_dataset, test_dataset = None, None, None
        # Single dataset.
        if train_data_prefix is not None:
            train_dataset = build_dataset("train", train_data_prefix,
                                          train_valid_test_num_samples[0],
                                          seq_length, seed, skip_warmup)

        if valid_data_prefix is not None:
            valid_dataset = build_dataset("valid", valid_data_prefix,
                                          train_valid_test_num_samples[1],
                                          seq_length, seed, False)

        if test_data_prefix is not None:
            test_dataset = build_dataset("test", test_data_prefix,
                                         train_valid_test_num_samples[2],
                                         seq_length, seed, False)

        return (train_dataset, valid_dataset, test_dataset)


def _build_train_valid_test_datasets(data_prefix, splits_string,
                                     train_valid_test_num_samples,
                                     seq_length, seed, skip_warmup,
                                     return_doc_ids=False):
    """Build train, valid, and xtest datasets using existing split"""

    args = get_args()
    # Indexed dataset.
    indexed_dataset = get_processed_dataset(data_prefix, args.data_folder)

    train_dataset = SFTDataset(data_prefix, indexed_dataset["train"], seq_length)
    valid_dataset = SFTDataset(data_prefix, indexed_dataset["valid"], seq_length)
    test_dataset = SFTDataset(data_prefix, indexed_dataset["test"], seq_length)
    return (train_dataset, valid_dataset, test_dataset)


def build_dataset(dataset_name, data_prefix, num_samples,
                  seq_length, seed, skip_warmup):
    dataset = None
    if len(data_prefix) == 1:
        dataset = _build_dataset(dataset_name,
                        data_prefix[0],
                        num_samples, seq_length,
                        seed, skip_warmup)
    else:
        # Blending dataset.
        # Parse the values.
        output = get_datasets_weights_and_num_samples(data_prefix, num_samples)
        prefixes, weights, dataset_num_samples = output

        # Build individual datasets.
        datasets = []
        for i in range(len(prefixes)):
            ds = _build_dataset(dataset_name, prefixes[i],
                            dataset_num_samples[i],
                            seq_length, seed, skip_warmup)
            if ds:
                datasets.append(ds)

        if datasets:
            dataset = BlendableDataset(datasets, weights)

    return dataset


def _build_dataset(dataset_name, data_prefix,
                   num_samples, seq_length, seed, skip_warmup):
    """
    Build dataset. This method is called when individual
    train, valid, test datasets are provided
    """

    args = get_args()
    # Indexed dataset.
    indexed_dataset = get_processed_dataset(data_prefix, args.data_folder)

    dataset = SFTDataset(data_prefix, indexed_dataset[dataset_name], seq_length)

    return dataset


