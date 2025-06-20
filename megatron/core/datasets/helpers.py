# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import numpy

# Implicit imports for backwards compatibility
# Explicit imports for readability
from megatron.core.datasets.helpers_cpp import *
from megatron.core.datasets.helpers_cpp import build_sample_idx_int32, build_sample_idx_int64


def build_sample_idx(
    sizes: numpy.ndarray,
    document_indices: numpy.ndarray,
    sequence_length: int,
    num_epochs: int,
    tokens_per_epoch: int,
    drop_last_partial_sequence: bool = True,
    add_extra_token_to_sequence: bool = True,
):
    """Build the 2-D sample index using the properly typed templated C++ function from helpers.cpp

    Args:
        sizes (numpy.ndarray): The 1-D array of document lengths

        document_indices (numpy.ndarray): The 1-D array of document indices

        sequence_length (int): The sequence length

        num_epochs (int): The number of epochs

        tokens_per_epoch (int): The number of tokens per epoch

        drop_last_partial_sequence (bool): Whether to omit the last partial sequence in the sample
            index should it exist. Defaults to True.

        add_extra_token_to_sequence (bool): Whether to build samples with sequence length
            `sequence_length + 1`. Defaults to True.

    Returns:
        numpy.ndarray: The 2-D sample index
    """
    sample_idx_max = max(document_indices.shape[0], sizes.max())
    if sample_idx_max <= numpy.iinfo(numpy.int32).max:
        sample_idx = build_sample_idx_int32(
            sizes,
            document_indices,
            sequence_length,
            num_epochs,
            tokens_per_epoch,
            drop_last_partial_sequence,
            1 if add_extra_token_to_sequence else 0,
        )
        assert sample_idx.min() >= 0 and sample_idx.max() <= sample_idx_max
    else:
        sample_idx = build_sample_idx_int64(
            sizes,
            document_indices,
            sequence_length,
            num_epochs,
            tokens_per_epoch,
            drop_last_partial_sequence,
            1 if add_extra_token_to_sequence else 0,
        )
    return sample_idx
